from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from voice_disorder_torch.config import RunContext
from voice_disorder_torch.data.datasets import SsastMelDataset
from voice_disorder_torch.data.load import LoadedDataBundle, load_all_preprocessed
from voice_disorder_torch.data.partitioning import assign_partitions_by_patient, indices_for_partition


STATIC_SOURCES = {"none", "auto", "mel", "opensmile", "parselmouth", "table"}
TABLE_META_COLUMNS = {"dataset", "patient_id", "id", "vowel", "n_files", "audio_paths", "audio_path"}


@dataclass(frozen=True)
class StaticFeatureConfig:
    source: str = "none"
    audio_manifest: Optional[Path] = None
    audio_root_eent: Optional[Path] = None
    audio_root_svd: Optional[Path] = None
    feature_table: Optional[Path] = None


@dataclass(frozen=True)
class StaticFeatureInfo:
    source: str
    backend: str
    dim: int
    names: list[str]
    mean: list[float]
    std: list[float]


class SsastMelStaticDataset(Dataset):
    """SSAST mel dataset plus fixed per-segment static acoustic features."""

    def __init__(self, x_nhwc: np.ndarray, y: np.ndarray, static_features: np.ndarray, input_tdim: int):
        self.base = SsastMelDataset(x_nhwc, y, input_tdim=input_tdim)
        static = np.asarray(static_features, dtype=np.float32)
        if static.ndim != 2:
            raise ValueError(f"Expected static features (N,D), got {static.shape}")
        if static.shape[0] != len(self.base):
            raise ValueError(f"Static feature count {static.shape[0]} != dataset count {len(self.base)}")
        self.static = torch.from_numpy(static)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, self.static[idx]


def _as_path(value: Optional[Path | str]) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


def validate_static_source(source: str) -> str:
    source = str(source).lower().strip()
    if source not in STATIC_SOURCES:
        valid = ", ".join(sorted(STATIC_SOURCES))
        raise ValueError(f"Unknown static feature source {source!r}; valid choices: {valid}")
    return source


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def resolve_static_backend(config: StaticFeatureConfig) -> str:
    source = validate_static_source(config.source)
    if source in {"none", "mel", "opensmile", "parselmouth", "table"}:
        return source

    has_audio_hint = any(
        _as_path(p) is not None for p in (config.audio_manifest, config.audio_root_eent, config.audio_root_svd)
    )
    if has_audio_hint and _importable("opensmile"):
        return "opensmile"
    if has_audio_hint and _importable("parselmouth"):
        return "parselmouth"
    return "mel"


def _norm_dataset(dataset: str) -> str:
    dataset = str(dataset).lower().strip()
    if dataset in {"eent", "chinese", "china", "ch"}:
        return "chinese"
    if dataset in {"svd", "german", "ger"}:
        return "german"
    return dataset


def _norm_patient(patient_id) -> str:
    return str(patient_id).strip().lower()


@lru_cache(maxsize=8)
def _load_static_feature_table(path_text: str) -> tuple[dict[tuple[str, str, str], np.ndarray], list[str]]:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Static feature table not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Static feature table has no header: {path}")
        feature_names = [name for name in reader.fieldnames if name not in TABLE_META_COLUMNS]
        if not feature_names:
            raise ValueError(f"No feature columns found in static feature table: {path}")
        table: dict[tuple[str, str, str], np.ndarray] = {}
        for row in reader:
            dataset = _norm_dataset(row.get("dataset", ""))
            patient = _norm_patient(row.get("patient_id", row.get("id", "")))
            vowel = str(row.get("vowel", "")).lower().strip().strip("/")
            if not dataset or not patient or vowel not in {"a", "i"}:
                continue
            values = np.asarray([float(row[name]) for name in feature_names], dtype=np.float32)
            table[(dataset, vowel, patient)] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not table:
        raise ValueError(f"No usable rows found in static feature table: {path}")
    return table, feature_names


def compute_table_static_features(
    *,
    patient_ids: Iterable,
    dataset: str,
    vowel: str,
    config: StaticFeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    if config.feature_table is None:
        raise ValueError("Set --static-feature-table when using --static-feature-source table.")
    table, names = _load_static_feature_table(str(Path(config.feature_table)))
    dataset = _norm_dataset(dataset)
    vowel = str(vowel).lower().strip().strip("/")
    rows = []
    missing = []
    for pid in patient_ids:
        key = (dataset, vowel, _norm_patient(pid))
        value = table.get(key)
        if value is None:
            missing.append(str(pid))
        else:
            rows.append(value)
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            f"Static feature table is missing {len(missing)} entries for dataset={dataset!r}, "
            f"vowel={vowel!r}; first missing patient IDs: {preview}"
        )
    return np.vstack(rows).astype(np.float32), names


def mel_static_feature_names() -> list[str]:
    return [
        "mel_mean",
        "mel_std",
        "mel_min",
        "mel_max",
        "mel_p05",
        "mel_p25",
        "mel_p50",
        "mel_p75",
        "mel_p95",
        "mel_abs_mean",
        "mel_rms",
        "frame_mean_mean",
        "frame_mean_std",
        "bin_mean_mean",
        "bin_mean_std",
        "frame_energy_mean",
        "frame_energy_std",
        "bin_energy_mean",
        "bin_energy_std",
        "freq_centroid_mean",
        "freq_centroid_std",
        "time_centroid_mean",
        "time_centroid_std",
    ]


def compute_mel_static_features(x_nhwc: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(x_nhwc, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 1:
        raise ValueError(f"Expected mel input (N,F,T,1), got {x.shape}")
    x = x[..., 0]
    n, fdim, tdim = x.shape
    flat = x.reshape(n, -1)
    percentiles = np.percentile(flat, [5, 25, 50, 75, 95], axis=1).T

    frame_mean = x.mean(axis=1)
    bin_mean = x.mean(axis=2)
    frame_energy = np.square(x).mean(axis=1)
    bin_energy = np.square(x).mean(axis=2)

    weights = x - x.min(axis=(1, 2), keepdims=True)
    weights = weights + 1e-6
    freq_axis = np.linspace(0.0, 1.0, fdim, dtype=np.float32).reshape(1, fdim, 1)
    time_axis = np.linspace(0.0, 1.0, tdim, dtype=np.float32).reshape(1, 1, tdim)
    freq_centroid = (weights * freq_axis).sum(axis=1) / weights.sum(axis=1).clip(min=1e-6)
    time_centroid = (weights * time_axis).sum(axis=2) / weights.sum(axis=2).clip(min=1e-6)

    feats = np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            flat.min(axis=1),
            flat.max(axis=1),
            percentiles,
            np.abs(flat).mean(axis=1),
            np.sqrt(np.square(flat).mean(axis=1)),
            frame_mean.mean(axis=1),
            frame_mean.std(axis=1),
            bin_mean.mean(axis=1),
            bin_mean.std(axis=1),
            frame_energy.mean(axis=1),
            frame_energy.std(axis=1),
            bin_energy.mean(axis=1),
            bin_energy.std(axis=1),
            freq_centroid.mean(axis=1),
            freq_centroid.std(axis=1),
            time_centroid.mean(axis=1),
            time_centroid.std(axis=1),
        ]
    )
    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), mel_static_feature_names()


def parselmouth_feature_names() -> list[str]:
    return [
        "duration_s",
        "pitch_mean_hz",
        "pitch_std_hz",
        "pitch_min_hz",
        "pitch_max_hz",
        "intensity_mean_db",
        "intensity_std_db",
        "intensity_min_db",
        "intensity_max_db",
        "hnr_mean_db",
        "hnr_std_db",
        "samples_rms",
    ]


def _finite_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals != 0.0]
    if vals.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(vals.mean()), float(vals.std()), float(vals.min()), float(vals.max())


def _extract_parselmouth_file(path: Path) -> np.ndarray:
    try:
        import parselmouth
    except Exception as exc:
        raise RuntimeError("Install praat-parselmouth to use --static-feature-source parselmouth.") from exc

    snd = parselmouth.Sound(str(path))
    duration = float(snd.get_total_duration())
    pitch = snd.to_pitch()
    p_mean, p_std, p_min, p_max = _finite_stats(pitch.selected_array["frequency"])
    intensity = snd.to_intensity()
    i_mean, i_std, i_min, i_max = _finite_stats(intensity.values)
    harmonicity = snd.to_harmonicity_cc()
    h_mean, h_std, _, _ = _finite_stats(harmonicity.values)
    samples = np.asarray(snd.values, dtype=np.float64)
    rms = float(math.sqrt(np.square(samples).mean())) if samples.size else 0.0
    return np.asarray(
        [duration, p_mean, p_std, p_min, p_max, i_mean, i_std, i_min, i_max, h_mean, h_std, rms],
        dtype=np.float32,
    )


def _extract_opensmile_file(path: Path) -> tuple[np.ndarray, list[str]]:
    try:
        import opensmile
    except Exception as exc:
        raise RuntimeError("Install opensmile to use --static-feature-source opensmile.") from exc

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    frame = smile.process_file(str(path))
    names = [str(c) for c in frame.columns]
    values = frame.iloc[0].to_numpy(dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), names


def _load_manifest(path: Optional[Path]) -> dict[tuple[str, str, str], list[Path]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Static audio manifest not found: {path}")

    rows: list[dict[str, str]] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("items", raw.get("rows", []))
        rows = list(raw)
    else:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    out: dict[tuple[str, str, str], list[Path]] = {}
    for row in rows:
        dataset = str(row.get("dataset", row.get("corpus", ""))).lower().strip()
        if dataset in {"eent", "chinese", "china"}:
            dataset = "chinese"
        elif dataset in {"svd", "german"}:
            dataset = "german"
        else:
            dataset = ""
        vowel = str(row.get("vowel", "")).lower().strip().strip("/")
        patient = str(row.get("patient_id", row.get("id", row.get("ID", "")))).strip()
        file_path = row.get("path", row.get("file", row.get("audio_path", row.get("wav_path", ""))))
        if not patient or vowel not in {"a", "i"} or not file_path:
            continue
        resolved_path = Path(str(file_path)).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = path.parent / resolved_path
        key = (dataset, vowel, patient)
        out.setdefault(key, []).append(resolved_path)
    return out


def _list_audio_files(root: Optional[Path]) -> list[Path]:
    root = _as_path(root)
    if root is None:
        return []
    if not root.exists():
        raise FileNotFoundError(f"Static audio root does not exist: {root}")
    files: list[Path] = []
    for suffix in ("*.wav", "*.flac", "*.mp3"):
        files.extend(root.rglob(suffix))
    return files


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def _candidate_audio_files(
    *,
    patient_id: str,
    vowel: str,
    dataset: str,
    config: StaticFeatureConfig,
    manifest: dict[tuple[str, str, str], list[Path]],
    audio_files: dict[str, list[Path]],
) -> list[Path]:
    patient = str(patient_id).strip()
    direct = manifest.get((dataset, vowel, patient)) or manifest.get(("", vowel, patient))
    if direct:
        return direct

    files = audio_files.get(dataset, [])
    if not files:
        return []
    out: list[Path] = []
    for path in files:
        stem = path.stem.lower()
        toks = _tokens(stem)
        has_patient = patient.lower() in stem or patient.lower() in toks
        has_vowel = vowel in toks or f"vowel{vowel}" in stem or f"vowel-{vowel}" in stem or f"_{vowel}_" in stem
        if has_patient and has_vowel:
            out.append(path)
    return out


def compute_audio_static_features(
    *,
    patient_ids: Iterable,
    dataset: str,
    vowel: str,
    backend: str,
    config: StaticFeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    dataset = dataset.lower().strip()
    if dataset == "eent":
        dataset = "chinese"
    if dataset == "svd":
        dataset = "german"
    if backend not in {"opensmile", "parselmouth"}:
        raise ValueError(f"Audio static backend must be opensmile/parselmouth, got {backend!r}")

    manifest = _load_manifest(_as_path(config.audio_manifest))
    audio_files = {
        "chinese": _list_audio_files(_as_path(config.audio_root_eent)),
        "german": _list_audio_files(_as_path(config.audio_root_svd)),
    }

    cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    by_patient: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    missing: list[str] = []
    for patient_id in dict.fromkeys(str(pid) for pid in patient_ids):
        candidates = _candidate_audio_files(
            patient_id=patient_id,
            vowel=vowel,
            dataset=dataset,
            config=config,
            manifest=manifest,
            audio_files=audio_files,
        )
        if not candidates:
            missing.append(patient_id)
            continue

        vectors = []
        for path in candidates:
            key = str(path.resolve())
            if key not in cache:
                if backend == "opensmile":
                    cache[key] = _extract_opensmile_file(path)
                else:
                    cache[key] = (_extract_parselmouth_file(path), parselmouth_feature_names())
            vec, names = cache[key]
            vectors.append(vec)
            if feature_names is None:
                feature_names = names
        by_patient[patient_id] = np.mean(np.vstack(vectors), axis=0).astype(np.float32)

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Could not resolve {backend} audio files for {len(missing)} patient/vowel entries "
            f"({dataset=}, {vowel=}); first missing IDs: {preview}. "
            "Provide --static-audio-manifest or matching --static-audio-root-eent/--static-audio-root-svd."
        )
    if feature_names is None:
        raise ValueError("No audio files were resolved for static feature extraction.")

    features = np.vstack([by_patient[str(pid)] for pid in patient_ids]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), feature_names


def compute_static_features(
    *,
    x_nhwc: np.ndarray,
    patient_ids: Iterable,
    dataset: str,
    vowel: str,
    config: StaticFeatureConfig,
    backend: Optional[str] = None,
) -> tuple[np.ndarray, list[str], str]:
    if backend in {"opensmile_parselmouth_131", "opensmile+parselmouth", "opensmile_parselmouth"}:
        backend = "table"
    backend = resolve_static_backend(config) if backend is None else validate_static_source(backend)
    if backend == "auto":
        backend = resolve_static_backend(config)
    if backend == "none":
        return np.zeros((len(x_nhwc), 0), dtype=np.float32), [], "none"
    if backend == "table":
        feats, names = compute_table_static_features(
            patient_ids=patient_ids,
            dataset=dataset,
            vowel=vowel,
            config=config,
        )
        return feats, names, "opensmile_parselmouth_131"
    if backend == "mel":
        feats, names = compute_mel_static_features(x_nhwc)
        return feats, names, "mel"
    feats, names = compute_audio_static_features(
        patient_ids=patient_ids,
        dataset=dataset,
        vowel=vowel,
        backend=backend,
        config=config,
    )
    return feats, names, backend


def fit_static_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected features (N,D), got {features.shape}")
    if features.shape[1] == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_static_normalizer(features: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    mean_arr = np.asarray(list(mean), dtype=np.float32)
    std_arr = np.asarray(list(std), dtype=np.float32)
    if features.shape[1] != mean_arr.shape[0] or features.shape[1] != std_arr.shape[0]:
        raise ValueError(
            f"Static feature dim mismatch: features={features.shape[1]}, "
            f"mean={mean_arr.shape[0]}, std={std_arr.shape[0]}"
        )
    std_arr = np.where(std_arr < 1e-6, 1.0, std_arr).astype(np.float32)
    return ((features.astype(np.float32) - mean_arr) / std_arr).astype(np.float32)


def _select_vowel_train_val(bundle: LoadedDataBundle, vowel: str):
    if vowel == "a":
        return bundle.x_train_a, bundle.y_train_a, bundle.id_train_a, bundle.x_val_a, bundle.y_val_a, bundle.id_val_a
    return bundle.x_train_i, bundle.y_train_i, bundle.id_train_i, bundle.x_val_i, bundle.y_val_i, bundle.id_val_i


def build_static_ssast_partition_loaders(
    *,
    ctx: RunContext,
    vowel: str,
    n_partitions: int,
    partition_seed: int,
    input_tdim: int,
    batch_size: int,
    config: StaticFeatureConfig,
) -> tuple[object, StaticFeatureInfo]:
    from voice_disorder_torch.split_learning.loaders import PartitionedMelLoaders

    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    x_tr, y_tr, id_tr, x_va, y_va, id_va = _select_vowel_train_val(bundle, vowel)
    train_raw, names, backend = compute_static_features(
        x_nhwc=x_tr,
        patient_ids=id_tr,
        dataset="chinese",
        vowel=vowel,
        config=config,
    )
    val_raw, val_names, val_backend = compute_static_features(
        x_nhwc=x_va,
        patient_ids=id_va,
        dataset="chinese",
        vowel=vowel,
        config=config,
        backend=backend,
    )
    if val_backend != backend or val_names != names:
        raise RuntimeError("Static feature backend/name mismatch between train and val splits.")
    mean, std = fit_static_normalizer(train_raw)
    train_static = apply_static_normalizer(train_raw, mean, std)
    val_static = apply_static_normalizer(val_raw, mean, std)

    train_ds_full = SsastMelStaticDataset(x_tr, y_tr, train_static, input_tdim=input_tdim)
    val_ds = SsastMelStaticDataset(x_va, y_va, val_static, input_tdim=input_tdim)
    rng = np.random.default_rng(int(partition_seed))
    patient_map = assign_partitions_by_patient(id_tr, int(n_partitions), rng)
    train_loaders: list[DataLoader] = []
    for partition_id in range(int(n_partitions)):
        idxs = indices_for_partition(id_tr, patient_map, partition_id)
        if not idxs:
            raise ValueError(f"Client partition {partition_id} has no training segments.")
        train_loaders.append(
            DataLoader(
                Subset(train_ds_full, idxs),
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        )

    info = StaticFeatureInfo(
        source=validate_static_source(config.source),
        backend=backend,
        dim=int(train_static.shape[1]),
        names=names,
        mean=[float(x) for x in mean.tolist()],
        std=[float(x) for x in std.tolist()],
    )
    return (
        PartitionedMelLoaders(
            train_loaders=train_loaders,
            val_loader=DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0),
            n_partitions=int(n_partitions),
        ),
        info,
    )
