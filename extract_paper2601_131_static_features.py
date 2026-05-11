from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


AUDIO_EXTS = {".wav", ".flac", ".mp3"}
OPENSMILE_EXPECTED_DIM = 88
PARSELMOUTH_DIM = 43
TOTAL_EXPECTED_DIM = OPENSMILE_EXPECTED_DIM + PARSELMOUTH_DIM
_WORKER_SMILE = None
_WORKER_PITCH_FLOOR = 75.0
_WORKER_PITCH_CEILING = 600.0


def _optional_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "installed"))


def check_dependencies() -> dict[str, str | None]:
    return {
        "numpy": _optional_version("numpy"),
        "opensmile": _optional_version("opensmile"),
        "parselmouth": _optional_version("parselmouth"),
    }


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def infer_vowel(path: Path) -> str:
    toks = _tokens(" ".join(path.parts))
    for tok in toks:
        if tok in {"a", "i"}:
            return tok
        if tok in {"vowela", "vowel_a", "vowel-a"}:
            return "a"
        if tok in {"voweli", "vowel_i", "vowel-i"}:
            return "i"
    stem = path.stem.lower()
    if re.search(r"(^|[_\-])a($|[_\-])", stem):
        return "a"
    if re.search(r"(^|[_\-])i($|[_\-])", stem):
        return "i"
    raise ValueError(f"Cannot infer vowel from path: {path}")


def infer_dataset(path: Path) -> str:
    joined = " ".join(path.parts).lower()
    if any(key in joined for key in ("eent", "chinese", "china", "_ch", "-ch")):
        return "chinese"
    if any(key in joined for key in ("svd", "german", "_ger", "-ger")):
        return "german"
    return "unknown"


def infer_patient_id(path: Path, vowel: str) -> str:
    stem = path.stem
    toks = _tokens(stem)
    filtered = [t for t in toks if t not in {vowel, "vowel", "a", "i", "wav", "audio", "voice"}]
    digit_tokens = [t for t in filtered if any(ch.isdigit() for ch in t)]
    if digit_tokens:
        return digit_tokens[0]
    if filtered:
        return filtered[0]
    raise ValueError(f"Cannot infer patient id from path: {path}")


def _rewrite_to_data_root(path: Path, data_root: Path | None) -> Path:
    if data_root is None or path.exists():
        return path
    parts = list(path.parts)
    lowered = [p.lower() for p in parts]
    if "data" not in lowered:
        return path
    idx = lowered.index("data")
    tail = parts[idx + 1 :]
    return Path(data_root).joinpath(*tail)


def resolve_path(path_text: str, base_dir: Path | None = None, data_root: Path | None = None) -> Path:
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return _rewrite_to_data_root(path, data_root)


def _normalize_vowel(value: str) -> str:
    value = str(value).lower().strip().strip("/")
    if value in {"a", "vowel-a", "vowela", "vowel_a"}:
        return "a"
    if value in {"i", "vowel-i", "voweli", "vowel_i"}:
        return "i"
    return value


def load_manifest(path: Path, data_root: Path | None = None) -> list[dict[str, str]]:
    path = Path(path)
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_text = row.get("path") or row.get("audio_path") or row.get("wav_path") or row.get("file")
            if not file_text:
                raise ValueError("Manifest rows must include path/audio_path/wav_path/file.")
            audio_path = resolve_path(file_text, path.parent, data_root=data_root)
            vowel = _normalize_vowel(row.get("vowel") or row.get("voice_type") or "")
            if vowel not in {"a", "i"}:
                vowel = infer_vowel(audio_path)
            patient_id = (
                row.get("patient_id")
                or row.get("subject_id")
                or row.get("id")
                or row.get("ID")
                or ""
            ).strip()
            if not patient_id:
                patient_id = infer_patient_id(audio_path, vowel)
            dataset = (row.get("dataset") or row.get("corpus") or "").strip().lower()
            if dataset in {"eent", "china", "ch"}:
                dataset = "chinese"
            elif dataset in {"svd", "ger"}:
                dataset = "german"
            elif not dataset:
                dataset = infer_dataset(audio_path)
            rows.append(
                {
                    "dataset": dataset,
                    "patient_id": patient_id,
                    "vowel": vowel,
                    "path": str(audio_path),
                }
            )
    return rows


def scan_audio_root(root: Path, regex: str | None = None) -> list[dict[str, str]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    pattern = re.compile(regex) if regex else None
    rows: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS):
        if pattern is not None:
            match = pattern.search(str(path))
            if not match:
                continue
            groups = match.groupdict()
            vowel = (groups.get("vowel") or "").lower().strip("/")
            if vowel not in {"a", "i"}:
                try:
                    vowel = infer_vowel(path)
                except ValueError:
                    continue
            patient_id = groups.get("patient_id") or groups.get("id")
            if not patient_id:
                patient_id = infer_patient_id(path, vowel)
            dataset = groups.get("dataset") or infer_dataset(path)
        else:
            try:
                vowel = infer_vowel(path)
            except ValueError:
                continue
            patient_id = infer_patient_id(path, vowel)
            dataset = infer_dataset(path)
        if vowel not in {"a", "i"}:
            continue
        rows.append(
            {
                "dataset": str(dataset).lower(),
                "patient_id": str(patient_id),
                "vowel": vowel,
                "path": str(path),
            }
        )
    return rows


def finite_values(values: Iterable[float], *, drop_zero: bool = False) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if drop_zero:
        arr = arr[arr != 0.0]
    return arr


def stats5(values: Iterable[float], *, drop_zero: bool = False) -> list[float]:
    arr = finite_values(values, drop_zero=drop_zero)
    if arr.size == 0:
        return [0.0] * 5
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float(np.median(arr)),
    ]


def stats7(values: Iterable[float], *, drop_zero: bool = False) -> list[float]:
    arr = finite_values(values, drop_zero=drop_zero)
    if arr.size == 0:
        return [0.0] * 7
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
    ]


def safe_praat_call(*args) -> float:
    try:
        value = args[0].praat.call(*args[1:]) if False else None
    except Exception:
        value = None
    raise RuntimeError("safe_praat_call should not be used directly.")


def praat_scalar(praat_module, *args) -> float:
    try:
        value = praat_module.call(*args)
    except Exception:
        return 0.0
    try:
        value = float(value)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def parselmouth_feature_names() -> list[str]:
    names = [
        "pm_duration_s",
        "pm_sample_rms",
        "pm_zero_crossing_rate",
        "pm_pitch_voiced_fraction",
    ]
    names += [f"pm_pitch_hz_{s}" for s in ("mean", "std", "min", "max", "median", "q25", "q75")]
    names += [f"pm_pitch_st55_{s}" for s in ("mean", "std", "min", "max", "median")]
    names += [f"pm_intensity_db_{s}" for s in ("mean", "std", "min", "max", "median", "q25", "q75")]
    names += [f"pm_hnr_db_{s}" for s in ("mean", "std", "min", "max", "median")]
    names += [
        "pm_jitter_local",
        "pm_jitter_local_abs",
        "pm_jitter_rap",
        "pm_jitter_ppq5",
        "pm_jitter_ddp",
        "pm_shimmer_local",
        "pm_shimmer_local_db",
        "pm_shimmer_apq3",
        "pm_shimmer_apq5",
        "pm_shimmer_apq11",
        "pm_shimmer_dda",
    ]
    names += [f"pm_f{idx}_mean_hz" for idx in range(1, 5)]
    if len(names) != PARSELMOUTH_DIM:
        raise RuntimeError(f"Internal Parselmouth feature dim is {len(names)}, expected {PARSELMOUTH_DIM}.")
    return names


def extract_parselmouth_features(path: Path, pitch_floor: float, pitch_ceiling: float) -> tuple[np.ndarray, list[str]]:
    try:
        import parselmouth
        from parselmouth import praat
    except Exception as exc:
        raise RuntimeError("Missing dependency: install praat-parselmouth before extraction.") from exc

    sound = parselmouth.Sound(str(path))
    duration = float(sound.get_total_duration())
    samples = np.asarray(sound.values, dtype=np.float64).reshape(-1)
    sample_rms = float(np.sqrt(np.square(samples).mean())) if samples.size else 0.0
    if samples.size > 1:
        zcr = float(np.mean(np.diff(np.signbit(samples)) != 0))
    else:
        zcr = 0.0

    pitch = sound.to_pitch(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    pitch_values = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    voiced = finite_values(pitch_values, drop_zero=True)
    voiced_fraction = float(voiced.size / max(pitch_values.size, 1))
    pitch_stats = stats7(voiced)
    pitch_st = 12.0 * np.log2(np.maximum(voiced, 1e-6) / 55.0) if voiced.size else []
    pitch_st_stats = stats5(pitch_st)

    intensity = sound.to_intensity(time_step=0.01, minimum_pitch=pitch_floor)
    intensity_stats = stats7(np.asarray(intensity.values, dtype=np.float64).reshape(-1), drop_zero=True)

    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=pitch_floor)
    hnr_stats = stats5(np.asarray(harmonicity.values, dtype=np.float64).reshape(-1), drop_zero=False)

    shortest_period = 1.0 / float(pitch_ceiling)
    longest_period = 1.0 / float(pitch_floor)
    max_period_factor = 1.3
    max_amp_factor = 1.6
    point_process = praat.call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    jitter = [
        praat_scalar(praat, point_process, "Get jitter (local)", 0, 0, shortest_period, longest_period, max_period_factor),
        praat_scalar(
            praat,
            point_process,
            "Get jitter (local, absolute)",
            0,
            0,
            shortest_period,
            longest_period,
            max_period_factor,
        ),
        praat_scalar(praat, point_process, "Get jitter (rap)", 0, 0, shortest_period, longest_period, max_period_factor),
        praat_scalar(praat, point_process, "Get jitter (ppq5)", 0, 0, shortest_period, longest_period, max_period_factor),
        praat_scalar(praat, point_process, "Get jitter (ddp)", 0, 0, shortest_period, longest_period, max_period_factor),
    ]
    shimmer = [
        praat_scalar(
            praat,
            [sound, point_process],
            "Get shimmer (local)",
            0,
            0,
            shortest_period,
            longest_period,
            max_period_factor,
            max_amp_factor,
        ),
        praat_scalar(
            praat,
            [sound, point_process],
            "Get shimmer (local_dB)",
            0,
            0,
            shortest_period,
            longest_period,
            max_period_factor,
            max_amp_factor,
        ),
        praat_scalar(
            praat,
            [sound, point_process],
            "Get shimmer (apq3)",
            0,
            0,
            shortest_period,
            longest_period,
            max_period_factor,
            max_amp_factor,
        ),
        praat_scalar(
            praat,
            [sound, point_process],
            "Get shimmer (apq5)",
            0,
            0,
            shortest_period,
            longest_period,
            max_period_factor,
            max_amp_factor,
        ),
        praat_scalar(
            praat,
            [sound, point_process],
            "Get shimmer (apq11)",
            0,
            0,
            shortest_period,
            longest_period,
            max_period_factor,
            max_amp_factor,
        ),
        praat_scalar(
            praat,
            [sound, point_process],
            "Get shimmer (dda)",
            0,
            0,
            shortest_period,
            longest_period,
            max_period_factor,
            max_amp_factor,
        ),
    ]

    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )
    if duration > 0.04:
        times = np.linspace(0.02, max(duration - 0.02, 0.02), num=100)
    else:
        times = np.asarray([max(duration / 2.0, 0.0)])
    formant_means: list[float] = []
    for idx in range(1, 5):
        vals = []
        for t in times:
            try:
                value = float(formant.get_value_at_time(idx, float(t)))
            except Exception:
                value = 0.0
            if math.isfinite(value) and value > 0:
                vals.append(value)
        formant_means.append(float(np.mean(vals)) if vals else 0.0)

    values = [duration, sample_rms, zcr, voiced_fraction]
    values += pitch_stats
    values += pitch_st_stats
    values += intensity_stats
    values += hnr_stats
    values += jitter
    values += shimmer
    values += formant_means

    names = parselmouth_feature_names()
    if len(values) != len(names):
        raise RuntimeError(f"Parselmouth vector dim {len(values)} != name dim {len(names)}")
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), names


class OpenSmileExtractor:
    def __init__(self) -> None:
        try:
            import opensmile
        except Exception as exc:
            raise RuntimeError("Missing dependency: install opensmile before extraction.") from exc
        self.opensmile = opensmile
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, path: Path) -> tuple[np.ndarray, list[str]]:
        frame = self.smile.process_file(str(path))
        names = [f"os_{str(col)}" for col in frame.columns]
        values = frame.iloc[0].to_numpy(dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), names


def extract_feature_row(
    row: dict[str, str],
    *,
    smile: OpenSmileExtractor,
    pitch_floor: float,
    pitch_ceiling: float,
) -> tuple[dict[str, object], list[str]]:
    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    os_vec, os_names = smile.extract(path)
    pm_vec, pm_names = extract_parselmouth_features(path, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    values = np.concatenate([os_vec, pm_vec], axis=0)
    names = os_names + pm_names
    out_row: dict[str, object] = {
        "dataset": row["dataset"],
        "patient_id": row["patient_id"],
        "vowel": row["vowel"],
        "audio_path": str(path),
    }
    out_row.update({name: float(value) for name, value in zip(names, values)})
    return out_row, names


def _worker_init(pitch_floor: float, pitch_ceiling: float) -> None:
    global _WORKER_SMILE, _WORKER_PITCH_FLOOR, _WORKER_PITCH_CEILING
    _WORKER_SMILE = OpenSmileExtractor()
    _WORKER_PITCH_FLOOR = float(pitch_floor)
    _WORKER_PITCH_CEILING = float(pitch_ceiling)


def _worker_extract(index_and_row: tuple[int, dict[str, str]]):
    index, row = index_and_row
    try:
        if _WORKER_SMILE is None:
            raise RuntimeError("Worker OpenSMILE extractor was not initialized.")
        out_row, names = extract_feature_row(
            row,
            smile=_WORKER_SMILE,
            pitch_floor=_WORKER_PITCH_FLOOR,
            pitch_ceiling=_WORKER_PITCH_CEILING,
        )
        return index, True, out_row, names, ""
    except Exception as exc:
        return index, False, {"path": row.get("path", ""), "error": str(exc)}, [], str(exc)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(file_rows: list[dict[str, object]], feature_names: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in file_rows:
        key = (str(row["dataset"]), str(row["patient_id"]), str(row["vowel"]))
        grouped[key].append(row)

    out: list[dict[str, object]] = []
    for (dataset, patient_id, vowel), rows in sorted(grouped.items()):
        item: dict[str, object] = {
            "dataset": dataset,
            "patient_id": patient_id,
            "vowel": vowel,
            "n_files": len(rows),
            "audio_paths": "|".join(str(r["audio_path"]) for r in rows),
        }
        for name in feature_names:
            item[name] = float(np.mean([float(r[name]) for r in rows]))
        out.append(item)
    return out


def write_npz(path: Path, aggregate: list[dict[str, object]], feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(
        [[str(r["dataset"]), str(r["patient_id"]), str(r["vowel"])] for r in aggregate],
        dtype=object,
    )
    features = np.asarray([[float(r[name]) for name in feature_names] for r in aggregate], dtype=np.float32)
    np.savez_compressed(path, keys=keys, features=features, feature_names=np.asarray(feature_names, dtype=object))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Paper2601-style 131D static features from raw audio on Windows/Linux."
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="CSV with dataset/patient_id/vowel/path or preprocessing columns. Can be passed multiple times.",
    )
    src.add_argument(
        "--input-root",
        type=Path,
        action="append",
        default=None,
        help="Recursively scan audio files under this root. Can be passed multiple times.",
    )
    p.add_argument(
        "--filename-regex",
        type=str,
        default=None,
        help="Optional regex with named groups dataset, patient_id/id, vowel for --input-root.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data"),
        help="Rewrite old absolute manifest paths by replacing their Data/... suffix with this root.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("paper2601_static_131_features"))
    p.add_argument("--file-csv", type=str, default="paper2601_static_131_file_level.csv")
    p.add_argument("--aggregate-csv", type=str, default="paper2601_static_131_by_patient_vowel.csv")
    p.add_argument("--aggregate-npz", type=str, default="paper2601_static_131_by_patient_vowel.npz")
    p.add_argument("--metadata-json", type=str, default="paper2601_static_131_metadata.json")
    p.add_argument("--pitch-floor", type=float, default=75.0)
    p.add_argument("--pitch-ceiling", type=float, default=600.0)
    p.add_argument("--limit", type=int, default=None, help="Debug: only process first N files.")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--check-deps", action="store_true", help="Print dependency status and exit.")
    p.add_argument("--allow-dim-mismatch", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    deps = check_dependencies()
    if args.check_deps:
        print(json.dumps(deps, indent=2))
        if deps["opensmile"] is None or deps["parselmouth"] is None:
            print("Install on Windows if missing:")
            print("  py -m pip install opensmile praat-parselmouth")
        return

    if args.manifest is None and args.input_root is None:
        raise SystemExit("Provide --manifest or --input-root. Use --check-deps to inspect dependencies.")

    if args.manifest:
        rows = []
        for manifest in args.manifest:
            rows.extend(load_manifest(manifest, data_root=args.data_root))
    else:
        rows = []
        for root in args.input_root:
            rows.extend(scan_audio_root(root, args.filename_regex))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise SystemExit("No audio files found.")

    if deps["opensmile"] is None or deps["parselmouth"] is None:
        raise SystemExit(
            "Missing required dependencies for strict 131D extraction. "
            "Install with: py -m pip install opensmile praat-parselmouth"
        )

    file_rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    failures: list[dict[str, str]] = []
    indexed_results: list[tuple[int, dict[str, object]]] = []
    progress_every = max(int(args.progress_every), 1)
    workers = max(int(args.workers), 1)

    def handle_result(index: int, ok: bool, payload: dict[str, object], names: list[str], error: str = "") -> None:
        nonlocal feature_names
        row_path = rows[index - 1].get("path", "")
        if not ok:
            failures.append({"path": str(row_path), "error": str(error or payload.get("error", ""))})
            print(f"[{index}/{len(rows)}] FAIL {row_path}: {error or payload.get('error', '')}")
            return
        if len(names) != TOTAL_EXPECTED_DIM and not args.allow_dim_mismatch:
            raise RuntimeError(f"Feature dim {len(names)} != expected {TOTAL_EXPECTED_DIM}.")
        os_dim = sum(1 for name in names if name.startswith("os_"))
        if os_dim != OPENSMILE_EXPECTED_DIM and not args.allow_dim_mismatch:
            raise RuntimeError(
                f"OpenSMILE eGeMAPSv02 returned {os_dim} features, expected {OPENSMILE_EXPECTED_DIM}."
            )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("Feature names changed between files.")
        indexed_results.append((index, payload))
        if index == 1 or index == len(rows) or index % progress_every == 0:
            print(f"[{index}/{len(rows)}] ok {row_path}")

    if workers == 1:
        smile = OpenSmileExtractor()
        for idx, row in enumerate(rows, start=1):
            try:
                out_row, names = extract_feature_row(
                    row,
                    smile=smile,
                    pitch_floor=float(args.pitch_floor),
                    pitch_ceiling=float(args.pitch_ceiling),
                )
                handle_result(idx, True, out_row, names)
            except Exception as exc:
                handle_result(idx, False, {"path": row.get("path", ""), "error": str(exc)}, [], str(exc))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(float(args.pitch_floor), float(args.pitch_ceiling)),
        ) as pool:
            futures = [pool.submit(_worker_extract, (idx, row)) for idx, row in enumerate(rows, start=1)]
            for future in as_completed(futures):
                idx, ok, payload, names, error = future.result()
                handle_result(idx, ok, payload, names, error)

    indexed_results.sort(key=lambda item: item[0])
    file_rows = [row for _, row in indexed_results]

    if feature_names is None or not file_rows:
        raise SystemExit("No files were successfully extracted.")

    aggregate = aggregate_rows(file_rows, feature_names)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / args.file_csv, file_rows, ["dataset", "patient_id", "vowel", "audio_path"] + feature_names)
    write_csv(
        output_dir / args.aggregate_csv,
        aggregate,
        ["dataset", "patient_id", "vowel", "n_files", "audio_paths"] + feature_names,
    )
    write_npz(output_dir / args.aggregate_npz, aggregate, feature_names)

    metadata = {
        "feature_dim": len(feature_names),
        "opensmile_dim": sum(1 for name in feature_names if name.startswith("os_")),
        "parselmouth_dim": sum(1 for name in feature_names if name.startswith("pm_")),
        "feature_names": feature_names,
        "n_input_rows": len(rows),
        "n_file_rows": len(file_rows),
        "n_aggregate_rows": len(aggregate),
        "n_failures": len(failures),
        "failures": failures,
        "dependencies": deps,
        "pitch_floor": float(args.pitch_floor),
        "pitch_ceiling": float(args.pitch_ceiling),
    }
    (output_dir / args.metadata_json).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote file-level CSV: {output_dir / args.file_csv}")
    print(f"Wrote aggregate CSV: {output_dir / args.aggregate_csv}")
    print(f"Wrote aggregate NPZ: {output_dir / args.aggregate_npz}")
    print(f"Wrote metadata JSON: {output_dir / args.metadata_json}")


if __name__ == "__main__":
    main()
