from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from split_ast_make_domain_shift_report import DEFAULT_STATIC_TABLE, build_static_shift_rows


DEFAULT_EXCLUDE_PATTERNS = [
    "duration",
    "intensity",
    "loudness",
    "rms",
    "equivalentsoundlevel",
    r"(^|[_-])min($|[_-])",
    r"(^|[_-])max($|[_-])",
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns if str(pattern).strip()]


def _is_corpus_sensitive(name: str, patterns: list[re.Pattern[str]]) -> bool:
    text = str(name).lower()
    return any(pattern.search(text) for pattern in patterns)


def _reason_for_drop(
    rows: list[dict[str, Any]],
    *,
    max_abs_mean_shift_z: float,
    max_ks: float,
    max_svd_pct_abs_z_gt3: float,
    exclude_patterns: list[re.Pattern[str]],
) -> list[str]:
    reasons: list[str] = []
    feature = str(rows[0]["feature"])
    if _is_corpus_sensitive(feature, exclude_patterns):
        reasons.append("name_matches_corpus_sensitive_pattern")
    for row in rows:
        vowel = str(row["vowel"])
        if abs(float(row["mean_shift_z"])) > max_abs_mean_shift_z:
            reasons.append(f"/{vowel}/ abs_mean_shift_z>{max_abs_mean_shift_z:g}")
        if row.get("ks_stat") is not None and float(row["ks_stat"]) > max_ks:
            reasons.append(f"/{vowel}/ ks>{max_ks:g}")
        if float(row["svd_pct_abs_z_gt3"]) > max_svd_pct_abs_z_gt3:
            reasons.append(f"/{vowel}/ svd_pct_abs_z_gt3>{max_svd_pct_abs_z_gt3:g}")
    return reasons


def select_stable_features(args: argparse.Namespace, bundle) -> tuple[list[str], dict[str, Any]]:
    shift_rows, summary_rows = build_static_shift_rows(args, bundle)
    rows_by_feature: dict[str, list[dict[str, Any]]] = {}
    for row in shift_rows:
        if str(row.get("preset")) != "all":
            continue
        rows_by_feature.setdefault(str(row["feature"]), []).append(row)

    patterns = [] if args.keep_corpus_sensitive_names else _compile_patterns(args.exclude_pattern)
    selected: list[str] = []
    dropped: list[dict[str, Any]] = []
    for feature, rows in sorted(rows_by_feature.items(), key=lambda item: item[0].lower()):
        vowels = {str(row["vowel"]) for row in rows}
        if vowels != {"a", "i"}:
            dropped.append({"feature": feature, "reasons": [f"missing_vowels:{','.join(sorted(vowels))}"], "rows": rows})
            continue
        reasons = _reason_for_drop(
            rows,
            max_abs_mean_shift_z=float(args.max_abs_mean_shift_z),
            max_ks=float(args.max_ks),
            max_svd_pct_abs_z_gt3=float(args.max_svd_pct_abs_z_gt3),
            exclude_patterns=patterns,
        )
        if reasons:
            dropped.append({"feature": feature, "reasons": reasons, "rows": rows})
        else:
            selected.append(feature)

    if not selected:
        raise SystemExit(
            "Stable static filter selected zero columns. Relax thresholds or pass --keep-corpus-sensitive-names."
        )

    report = {
        "source_table": str(args.static_feature_table),
        "out_table": str(args.out_table),
        "selection_rule": {
            "max_abs_mean_shift_z": float(args.max_abs_mean_shift_z),
            "max_ks": float(args.max_ks),
            "max_svd_pct_abs_z_gt3": float(args.max_svd_pct_abs_z_gt3),
            "exclude_pattern": [] if args.keep_corpus_sensitive_names else list(args.exclude_pattern),
            "vowel_policy": "feature must pass thresholds for both /a/ and /i/",
            "label_usage": "no labels are used; SVD is used only as an unlabeled target-domain distribution",
        },
        "static_summary": summary_rows,
        "n_input_features": len(rows_by_feature),
        "n_selected_features": len(selected),
        "n_dropped_features": len(dropped),
        "selected_features": selected,
        "dropped_features": dropped,
    }
    return selected, report


def write_filtered_table(source: Path, out_table: Path, selected_features: list[str]) -> None:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Static feature source table not found: {source}")
    out_table.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Static feature table has no header: {source}")
        meta_cols = [name for name in reader.fieldnames if name not in selected_features]
        # Preserve the original metadata columns, but do not keep dropped feature columns.
        selected_set = set(selected_features)
        meta_cols = [
            name
            for name in reader.fieldnames
            if name in {"dataset", "patient_id", "id", "vowel", "n_files", "audio_paths", "audio_path"}
        ]
        missing = [name for name in selected_features if name not in reader.fieldnames]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"Selected features missing from source table: {preview}")
        fieldnames = meta_cols + selected_features
        rows = [{name: row.get(name, "") for name in fieldnames} for row in reader]
    with out_table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    from split_ast_mae_cli import _add_data_args, _add_model_args

    p = argparse.ArgumentParser(
        description=(
            "Create a filtered SplitAST-MAE static-feature table by dropping columns with strong "
            "EENT-train to SVD distribution shift."
        )
    )
    _add_data_args(p)
    _add_model_args(p)
    p.set_defaults(static_feature_source="table", static_feature_table=DEFAULT_STATIC_TABLE, presets=["all"])
    p.add_argument(
        "--max-abs-mean-shift-z",
        type=float,
        default=3.0,
        help="Drop a feature if either vowel has |SVD mean - EENT train mean| / EENT train std above this value.",
    )
    p.add_argument("--max-ks", type=float, default=0.75, help="Drop a feature if either vowel has KS above this value.")
    p.add_argument(
        "--max-svd-pct-abs-z-gt3",
        type=float,
        default=0.30,
        help="Drop a feature if either vowel has this fraction of SVD patients outside |z| > 3.",
    )
    p.add_argument("--exclude-pattern", nargs="+", default=DEFAULT_EXCLUDE_PATTERNS)
    p.add_argument(
        "--keep-corpus-sensitive-names",
        action="store_true",
        help="Disable the name-based corpus-sensitive filter and use only numeric drift thresholds.",
    )
    p.add_argument(
        "--out-table",
        type=Path,
        default=Path("split_ast_local_artifacts/split_ast_stable_static_features/split_ast_stable_static_by_patient_vowel.csv"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("split_ast_local_artifacts/split_ast_stable_static_features/split_ast_stable_static_selection.json"),
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    from split_ast_mae_cli import _make_context
    from voice_disorder_torch.data.load import load_all_preprocessed

    args = build_parser().parse_args()
    ctx = _make_context(args)
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=not args.quiet)
    selected, report = select_stable_features(args, bundle)
    write_filtered_table(Path(args.static_feature_table), Path(args.out_table), selected)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    print(f"Selected stable static features: {len(selected)} / {report['n_input_features']}")
    print(f"Wrote filtered static table: {Path(args.out_table).resolve()}")
    print(f"Wrote selection report: {Path(args.out_json).resolve()}")
    print("Selected feature names:")
    for name in selected:
        print(f"- {name}")


if __name__ == "__main__":
    main()
