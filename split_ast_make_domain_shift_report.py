from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


DEFAULT_STATIC_TABLE = Path(
    "split_ast_local_artifacts/split_ast_static_131_features/split_ast_static_131_by_patient_vowel.csv"
)
DEFAULT_PRESETS = ["all", "pathology", "pathology_source_tilt", "pathology_voicing"]
DEFAULT_RUN_DIR_NAMES = [
    "split_ast_mae_runs_local5",
    "split_ast_mae_runs_local1",
    "split_ast_mae_runs_control_all131_gated_clip_local5",
    "split_ast_mae_runs_control_static_only_all131_local5",
    "split_ast_mae_runs_control_mae_only_local5",
    "split_ast_mae_runs_control_pathology22_gated_clip_local5",
    "split_ast_mae_runs_control_source_tilt_gated_clip_local5",
    "split_ast_mae_runs_ablation_pathology22_local5",
    "split_ast_mae_runs_ablation_pathology22_local1",
    "split_ast_mae_runs_ablation_pathology_source_tilt_local5",
    "split_ast_mae_runs_ablation_pathology_source_tilt_local1",
    "split_ast_mae_runs_ablation_pathology_voicing_local5",
    "split_ast_mae_runs_ablation_pathology_voicing_local1",
]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def _short_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _unique_preserve(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = str(value).strip()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _finite_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _describe(values: Any) -> dict[str, Optional[float]]:
    arr = _finite_1d(values)
    if arr.size == 0:
        return {key: None for key in ("mean", "std", "min", "q25", "median", "q75", "max")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def _ks_statistic(x: Any, y: Any) -> Optional[float]:
    x_arr = np.sort(_finite_1d(x))
    y_arr = np.sort(_finite_1d(y))
    if x_arr.size == 0 or y_arr.size == 0:
        return None
    vals = np.sort(np.concatenate([x_arr, y_arr]))
    cdf_x = np.searchsorted(x_arr, vals, side="right") / float(x_arr.size)
    cdf_y = np.searchsorted(y_arr, vals, side="right") / float(y_arr.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _wasserstein_approx(x: Any, y: Any) -> Optional[float]:
    x_arr = _finite_1d(x)
    y_arr = _finite_1d(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return None
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(x_arr, y_arr))
    except Exception:
        qs = np.linspace(0.0, 1.0, 201)
        return float(np.mean(np.abs(np.quantile(x_arr, qs) - np.quantile(y_arr, qs))))


def _empty_table_x(n: int) -> np.ndarray:
    return np.zeros((int(n), 1, 1, 1), dtype=np.float32)


def _ids_for_split(bundle, split_name: str, vowel: str):
    if split_name == "eent_train":
        return bundle.id_train_a if vowel == "a" else bundle.id_train_i
    if split_name == "eent_test":
        return bundle.id_test_a if vowel == "a" else bundle.id_test_i
    if split_name == "svd":
        return bundle.id_ger_a if vowel == "a" else bundle.id_ger_i
    raise ValueError(split_name)


def _x_for_split(bundle, split_name: str, vowel: str):
    if split_name == "eent_train":
        return bundle.x_train_a if vowel == "a" else bundle.x_train_i
    if split_name == "eent_test":
        return bundle.x_test_a if vowel == "a" else bundle.x_test_i
    if split_name == "svd":
        return bundle.x_ger_a if vowel == "a" else bundle.x_ger_i
    raise ValueError(split_name)


def _dataset_name_for_split(split_name: str) -> str:
    if split_name.startswith("eent"):
        return "chinese"
    if split_name == "svd":
        return "german"
    raise ValueError(split_name)


def _static_features_for_split(args: argparse.Namespace, bundle, *, split_name: str, vowel: str, preset: str):
    from split_ast_static_features import StaticFeatureConfig, compute_static_features

    source = str(args.static_feature_source).lower().strip()
    ids = _ids_for_split(bundle, split_name, vowel)
    if source == "table":
        ids = _unique_preserve(ids)
        x = _empty_table_x(len(ids))
    else:
        x = _x_for_split(bundle, split_name, vowel)
    config = StaticFeatureConfig(
        source=source,
        feature_table=args.static_feature_table,
        feature_preset=preset,
        audio_manifest=args.static_audio_manifest,
        audio_root_eent=args.static_audio_root_eent,
        audio_root_svd=args.static_audio_root_svd,
    )
    features, names, backend = compute_static_features(
        x_nhwc=x,
        patient_ids=ids,
        dataset=_dataset_name_for_split(split_name),
        vowel=vowel,
        config=config,
    )
    return np.asarray(features, dtype=np.float64), list(names), backend, len(ids)


def build_static_shift_rows(args: argparse.Namespace, bundle) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    eps = 1e-6
    for preset in args.presets:
        for vowel in ("a", "i"):
            eent, names, backend, n_eent = _static_features_for_split(
                args, bundle, split_name="eent_train", vowel=vowel, preset=preset
            )
            svd, svd_names, _, n_svd = _static_features_for_split(
                args, bundle, split_name="svd", vowel=vowel, preset=preset
            )
            if names != svd_names:
                raise SystemExit(f"Static feature names differ for preset={preset}, vowel={vowel}.")
            preset_rows: list[dict[str, Any]] = []
            for idx, name in enumerate(names):
                e_col = _finite_1d(eent[:, idx])
                s_col = _finite_1d(svd[:, idx])
                if e_col.size == 0 or s_col.size == 0:
                    continue
                e_mean = float(e_col.mean())
                e_std = float(e_col.std(ddof=0))
                safe_std = e_std if e_std >= eps else 1.0
                s_mean = float(s_col.mean())
                s_std = float(s_col.std(ddof=0))
                z = (s_col - e_mean) / safe_std
                abs_z = np.abs(z)
                row = {
                    "preset": preset,
                    "vowel": vowel,
                    "feature_idx": idx,
                    "feature": name,
                    "backend": backend,
                    "n_eent_train": n_eent,
                    "n_svd": n_svd,
                    "eent_train_mean": e_mean,
                    "eent_train_std": e_std,
                    "svd_mean": s_mean,
                    "svd_std": s_std,
                    "mean_shift_z": float((s_mean - e_mean) / safe_std),
                    "std_ratio": float(s_std / safe_std),
                    "svd_abs_z_p95": float(np.quantile(abs_z, 0.95)),
                    "svd_abs_z_max": float(abs_z.max()),
                    "svd_pct_abs_z_gt3": float(np.mean(abs_z > 3.0)),
                    "svd_pct_abs_z_gt5": float(np.mean(abs_z > 5.0)),
                    "ks_stat": _ks_statistic(e_col, s_col),
                    "wasserstein": _wasserstein_approx(e_col, s_col),
                }
                row["shift_score"] = max(
                    abs(float(row["mean_shift_z"])),
                    float(row["svd_abs_z_p95"]) / 3.0,
                    float(row["svd_pct_abs_z_gt3"]) * 5.0,
                )
                rows.append(row)
                preset_rows.append(row)
            if preset_rows:
                summaries.append(
                    {
                        "preset": preset,
                        "vowel": vowel,
                        "backend": backend,
                        "n_features": len(preset_rows),
                        "median_abs_mean_shift_z": float(
                            np.median([abs(float(r["mean_shift_z"])) for r in preset_rows])
                        ),
                        "max_abs_mean_shift_z": float(max(abs(float(r["mean_shift_z"])) for r in preset_rows)),
                        "median_svd_abs_z_p95": float(np.median([float(r["svd_abs_z_p95"]) for r in preset_rows])),
                        "max_svd_pct_abs_z_gt3": float(max(float(r["svd_pct_abs_z_gt3"]) for r in preset_rows)),
                        "median_ks_stat": float(np.median([float(r["ks_stat"]) for r in preset_rows if r["ks_stat"] is not None])),
                    }
                )
    rows.sort(key=lambda r: float(r["shift_score"]), reverse=True)
    return rows, summaries


def _discover_run_dirs(root: Path, explicit: Optional[list[Path]]) -> list[Path]:
    if explicit:
        candidates = [Path(p) for p in explicit]
    else:
        candidates = []
        for base in (root, root / "saved_models"):
            for name in DEFAULT_RUN_DIR_NAMES:
                candidates.append(base / name)
            candidates.extend(sorted(base.glob("split_ast_mae_runs*")) if base.is_dir() else [])
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        path = path.resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if (path / "ast_stage2_a_metadata.json").is_file() and (path / "ast_stage2_i_metadata.json").is_file():
            out.append(path)
    return out


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _threshold_from_run_dir(run_dir: Path, metric: str) -> tuple[float, str]:
    candidates = [
        run_dir / f"ast_stage2_ai_eval_eent_val_threshold_{metric}.json",
        run_dir / "ast_stage2_ai_eval.json",
        run_dir / "ast_stage2_ai_eval_eent_train_scaling.json",
    ]
    for path in candidates:
        payload = _load_json(path)
        if isinstance(payload, dict) and payload.get("patient_prob_threshold") is not None:
            return float(payload["patient_prob_threshold"]), path.name
    return 0.5, "default_0.5"


def _run_display_name(run_dir: Path) -> str:
    name = run_dir.name
    prefix = "split_ast_mae_runs_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _repair_moved_weight_paths(args_for_vowel: argparse.Namespace, meta: dict[str, Any], metadata_path: Path) -> None:
    for attr, meta_key in (("load_client", "client_file"), ("load_server", "server_file")):
        current = getattr(args_for_vowel, attr, None)
        if current is not None and Path(current).is_file():
            continue
        recorded = meta.get(meta_key)
        if not recorded:
            continue
        candidate = metadata_path.parent / Path(recorded).name
        if candidate.is_file():
            setattr(args_for_vowel, attr, candidate)


def _score_quantile_row(
    *,
    run_dir: Path,
    dataset: str,
    label: int,
    scores: np.ndarray,
    threshold: float,
    threshold_source: str,
) -> dict[str, Any]:
    desc = _describe(scores)
    return {
        "run": _run_display_name(run_dir),
        "run_dir": str(run_dir),
        "dataset": dataset,
        "label": int(label),
        "label_name": "patient" if int(label) == 1 else "normal",
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "n": int(len(scores)),
        **desc,
        "pct_ge_threshold": float(np.mean(np.asarray(scores) >= float(threshold))) if len(scores) else None,
    }


def _prediction_rows_for_run(args: argparse.Namespace, bundle, run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from split_ast_eval_eent_val_threshold import (
        _aggregate_patient_scores,
        _args_from_any_metadata,
        _build_any_pair,
        _predict_for_arrays,
    )
    from split_ast_mae_cli import _eval_arrays_for_dataset, _set_run_seed

    metadata_a = run_dir / "ast_stage2_a_metadata.json"
    metadata_i = run_dir / "ast_stage2_i_metadata.json"
    args_a, meta_a = _args_from_any_metadata(args, metadata_a)
    args_i, meta_i = _args_from_any_metadata(args, metadata_i)
    _repair_moved_weight_paths(args_a, meta_a, metadata_a)
    _repair_moved_weight_paths(args_i, meta_i, metadata_i)
    _set_run_seed(int(args_a.model_init_seed))
    client_a, server_a, device = _build_any_pair(args_a, meta_a)
    client_i, server_i, _ = _build_any_pair(args_i, meta_i)
    threshold, threshold_source = _threshold_from_run_dir(run_dir, args.threshold_metric)

    score_rows: list[dict[str, Any]] = []
    patient_scores: dict[str, Any] = {
        "run": _run_display_name(run_dir),
        "run_dir": str(run_dir),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "datasets": {},
    }
    for dataset_name, display_name in (("chinese", "EENT-test"), ("german", "SVD")):
        xa, ya, ida, _ = _eval_arrays_for_dataset(bundle, "a", dataset_name)
        xi, yi, idi, _ = _eval_arrays_for_dataset(bundle, "i", dataset_name)
        pa = _predict_for_arrays(
            client=client_a,
            server=server_a,
            args_for_vowel=args_a,
            meta=meta_a,
            x=xa,
            y=ya,
            ids=ida,
            dataset_name=dataset_name,
            vowel="a",
            device=device,
        )
        pi = _predict_for_arrays(
            client=client_i,
            server=server_i,
            args_for_vowel=args_i,
            meta=meta_i,
            x=xi,
            y=yi,
            ids=idi,
            dataset_name=dataset_name,
            vowel="i",
            device=device,
        )
        patient_ids, y_true, y_score = _aggregate_patient_scores(pa, pi, ya, yi, ida, idi)
        patient_scores["datasets"][display_name] = {
            "patient_ids": patient_ids,
            "y_true": y_true.tolist(),
            "y_score": y_score.tolist(),
        }
        for label in (0, 1):
            scores = y_score[y_true == label]
            score_rows.append(
                _score_quantile_row(
                    run_dir=run_dir,
                    dataset=display_name,
                    label=label,
                    scores=scores,
                    threshold=threshold,
                    threshold_source=threshold_source,
                )
            )
    return score_rows, patient_scores


def build_score_shift_rows(args: argparse.Namespace, bundle, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    run_dirs = _discover_run_dirs(root, args.run_dirs)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        print(f"Scoring patient distributions: {_short_path(run_dir, root)}")
        try:
            run_rows, run_detail = _prediction_rows_for_run(args, bundle, run_dir)
        except Exception as exc:
            skipped.append(f"{_short_path(run_dir, root)}: {exc}")
            print(f"[skip] {_short_path(run_dir, root)}: {exc}")
            continue
        rows.extend(run_rows)
        details.append(run_detail)
    return rows, details, skipped


def _static_top_table(rows: list[dict[str, Any]], limit: int) -> str:
    headers = [
        "preset",
        "vowel",
        "feature",
        "|mean shift z|",
        "SVD |z| p95",
        "SVD % |z|>3",
        "KS",
        "EENT mean/std",
        "SVD mean/std",
    ]
    body = []
    for row in rows[:limit]:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['preset']))}</td>"
            f"<td>/{html.escape(str(row['vowel']))}/</td>"
            f"<td class='feature'>{html.escape(str(row['feature']))}</td>"
            f"<td>{_fmt(abs(float(row['mean_shift_z'])), 3)}</td>"
            f"<td>{_fmt(row['svd_abs_z_p95'], 3)}</td>"
            f"<td>{_fmt(100.0 * float(row['svd_pct_abs_z_gt3']), 1)}%</td>"
            f"<td>{_fmt(row['ks_stat'], 3)}</td>"
            f"<td>{_fmt(row['eent_train_mean'], 3)} / {_fmt(row['eent_train_std'], 3)}</td>"
            f"<td>{_fmt(row['svd_mean'], 3)} / {_fmt(row['svd_std'], 3)}</td>"
            "</tr>"
        )
    return _html_table(headers, body)


def _html_table(headers: list[str], row_html: list[str]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    rows = "\n".join(row_html) if row_html else f"<tr><td colspan='{len(headers)}'>No rows.</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _summary_table(rows: list[dict[str, Any]]) -> str:
    headers = ["preset", "vowel", "dims", "median |mean shift z|", "max |mean shift z|", "median SVD |z| p95", "max % |z|>3", "median KS"]
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['preset']))}</td>"
            f"<td>/{html.escape(str(row['vowel']))}/</td>"
            f"<td>{row['n_features']}</td>"
            f"<td>{_fmt(row['median_abs_mean_shift_z'], 3)}</td>"
            f"<td>{_fmt(row['max_abs_mean_shift_z'], 3)}</td>"
            f"<td>{_fmt(row['median_svd_abs_z_p95'], 3)}</td>"
            f"<td>{_fmt(100.0 * float(row['max_svd_pct_abs_z_gt3']), 1)}%</td>"
            f"<td>{_fmt(row['median_ks_stat'], 3)}</td>"
            "</tr>"
        )
    return _html_table(headers, body)


def _box_svg(groups: list[dict[str, Any]], threshold: float) -> str:
    width = 720
    left = 112
    right = 24
    plot_w = width - left - right
    row_h = 34
    height = 26 + row_h * len(groups)

    def x_pos(v: Any) -> float:
        try:
            f = float(v)
        except Exception:
            f = 0.0
        f = min(max(f, 0.0), 1.0)
        return left + f * plot_w

    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img'>",
        f"<line x1='{left}' x2='{width-right}' y1='14' y2='14' stroke='#94a3b8'/>",
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_pos(tick)
        parts.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='10' y2='{height-6}' stroke='#e2e8f0'/>")
        parts.append(f"<text x='{x:.1f}' y='10' text-anchor='middle' font-size='10' fill='#475569'>{tick:.2f}</text>")
    tx = x_pos(threshold)
    parts.append(f"<line x1='{tx:.1f}' x2='{tx:.1f}' y1='10' y2='{height-6}' stroke='#ef4444' stroke-width='1.5'/>")
    for idx, row in enumerate(groups):
        y = 30 + idx * row_h
        label = f"{row['dataset']} {row['label_name']}"
        color = "#2563eb" if row["label"] == 0 else "#dc2626"
        parts.append(f"<text x='6' y='{y+4}' font-size='12' fill='#0f172a'>{html.escape(label)}</text>")
        if not row.get("n"):
            continue
        min_x, q25_x, med_x, q75_x, max_x = [x_pos(row[k]) for k in ("min", "q25", "median", "q75", "max")]
        mean_x = x_pos(row["mean"])
        parts.append(f"<line x1='{min_x:.1f}' x2='{max_x:.1f}' y1='{y}' y2='{y}' stroke='{color}' stroke-width='1.5'/>")
        parts.append(f"<rect x='{q25_x:.1f}' y='{y-8}' width='{max(1.0, q75_x-q25_x):.1f}' height='16' fill='{color}' opacity='0.18' stroke='{color}'/>")
        parts.append(f"<line x1='{med_x:.1f}' x2='{med_x:.1f}' y1='{y-10}' y2='{y+10}' stroke='{color}' stroke-width='2'/>")
        parts.append(f"<circle cx='{mean_x:.1f}' cy='{y}' r='3.5' fill='{color}'/>")
        parts.append(
            f"<text x='{width-right}' y='{y+4}' text-anchor='end' font-size='11' fill='#334155'>"
            f"n={row['n']} >=thr {_fmt(100.0 * float(row['pct_ge_threshold']), 1)}%</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _score_section(score_rows: list[dict[str, Any]]) -> str:
    if not score_rows:
        return "<p>No compatible SplitAST-MAE run directories with Stage 2 metadata were found.</p>"
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        by_run.setdefault(str(row["run"]), []).append(row)
    sections = []
    for run, rows in by_run.items():
        threshold = float(rows[0]["threshold"])
        source = rows[0]["threshold_source"]
        ordered = sorted(rows, key=lambda r: (str(r["dataset"]), int(r["label"])))
        table_rows = []
        for row in ordered:
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row['dataset']))}</td>"
                f"<td>{html.escape(str(row['label_name']))}</td>"
                f"<td>{row['n']}</td>"
                f"<td>{_fmt(row['mean'], 4)}</td>"
                f"<td>{_fmt(row['median'], 4)}</td>"
                f"<td>{_fmt(row['q25'], 4)}-{_fmt(row['q75'], 4)}</td>"
                f"<td>{_fmt(100.0 * float(row['pct_ge_threshold']), 1)}%</td>"
                "</tr>"
            )
        sections.append(
            "<section class='card'>"
            f"<h3>{html.escape(run)}</h3>"
            f"<p>Threshold used only for the red line and >=threshold rate: <b>{_fmt(threshold, 6)}</b> "
            f"from <code>{html.escape(source)}</code>.</p>"
            f"{_box_svg(ordered, threshold)}"
            f"{_html_table(['dataset', 'label', 'n', 'mean', 'median', 'IQR', '% >= threshold'], table_rows)}"
            "</section>"
        )
    return "\n".join(sections)


def _score_compare_table(score_rows: list[dict[str, Any]]) -> str:
    by_key: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for row in score_rows:
        key = (str(row["run"]), str(row["dataset"]))
        by_key.setdefault(key, {})[int(row["label"])] = row
    table_rows = []
    for (run, dataset), labels in sorted(by_key.items()):
        normal = labels.get(0, {})
        patient = labels.get(1, {})
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(run)}</td>"
            f"<td>{html.escape(dataset)}</td>"
            f"<td>{_fmt(normal.get('mean'), 4)}</td>"
            f"<td>{_fmt(patient.get('mean'), 4)}</td>"
            f"<td>{_fmt((patient.get('mean') or 0.0) - (normal.get('mean') or 0.0), 4)}</td>"
            f"<td>{_fmt(100.0 * float(normal.get('pct_ge_threshold') or 0.0), 1)}%</td>"
            f"<td>{_fmt(100.0 * float(patient.get('pct_ge_threshold') or 0.0), 1)}%</td>"
            "</tr>"
        )
    return _html_table(
        ["run", "dataset", "normal mean", "patient mean", "patient-normal gap", "normal %>=thr", "patient %>=thr"],
        table_rows,
    )


def build_html(
    *,
    args: argparse.Namespace,
    root: Path,
    static_rows: list[dict[str, Any]],
    static_summary: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    skipped_runs: list[str],
) -> str:
    css = """
    body { font-family: Inter, Segoe UI, Arial, sans-serif; margin: 28px; color: #0f172a; background: #f8fafc; }
    h1, h2, h3 { margin: 0.2rem 0 0.8rem; }
    p { color: #334155; line-height: 1.45; }
    code { background: #e2e8f0; padding: 1px 5px; border-radius: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 22px; background: white; }
    th, td { border-bottom: 1px solid #e2e8f0; padding: 7px 9px; text-align: right; font-size: 13px; vertical-align: top; }
    th { background: #e2e8f0; color: #0f172a; position: sticky; top: 0; }
    td:first-child, th:first-child, td.feature { text-align: left; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 18px 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
    .metric { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
    .metric b { font-size: 22px; display: block; margin-top: 4px; }
    .warn { background: #fff7ed; border-color: #fed7aa; }
    """
    n_shifted = sum(1 for row in static_rows if abs(float(row["mean_shift_z"])) >= 3.0)
    max_pct = max([float(row["svd_pct_abs_z_gt3"]) for row in static_rows], default=0.0)
    score_run_count = len({str(r["run"]) for r in score_rows})
    skipped = ""
    if skipped_runs:
        skipped_items = "".join(f"<li>{html.escape(s)}</li>" for s in skipped_runs)
        skipped = f"<section class='card warn'><h2>Skipped Runs</h2><ul>{skipped_items}</ul></section>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SplitAST-MAE Domain Shift Report</title>
<style>{css}</style>
</head>
<body>
<h1>SplitAST-MAE Domain Shift Report</h1>
<p>
Generated from <code>{html.escape(str(root))}</code>. Static report compares EENT train against SVD using
EENT-train mean/std as the reference. Score report loads saved /a/ and /i/ Stage 2 pairs and plots patient-level
positive probabilities for EENT-test and SVD.
</p>

<div class="grid">
  <div class="metric">Static rows<b>{len(static_rows)}</b></div>
  <div class="metric">Features with |mean shift z| >= 3<b>{n_shifted}</b></div>
  <div class="metric">Worst SVD % |z| &gt; 3<b>{_fmt(100.0 * max_pct, 1)}%</b></div>
  <div class="metric">Scored model runs<b>{score_run_count}</b></div>
</div>

<section class="card">
<h2>How To Read This</h2>
<p>
<b>mean shift z</b> = (SVD feature mean - EENT train feature mean) / EENT train std.
If this is large, SVD is far outside the distribution used to normalize static features during training.
<b>SVD % |z| &gt; 3</b> is the fraction of SVD patients beyond three EENT standard deviations.
<b>Patient score</b> is the final /a/+/i/ averaged positive probability per patient.
</p>
</section>

<section class="card">
<h2>Static Feature Drift Summary</h2>
{_summary_table(static_summary)}
</section>

<section class="card">
<h2>Top Shifted Static Features</h2>
{_static_top_table(static_rows, int(args.top_k))}
</section>

<section class="card">
<h2>Patient Score Mean Gap</h2>
{_score_compare_table(score_rows)}
</section>

<h2>Patient Score Distributions</h2>
{_score_section(score_rows)}
{skipped}
</body>
</html>
"""


def build_report(args: argparse.Namespace) -> None:
    from split_ast_mae_cli import _make_context
    from voice_disorder_torch.data.load import load_all_preprocessed

    root = Path.cwd()
    ctx = _make_context(args)
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=not args.quiet)

    static_rows, static_summary = build_static_shift_rows(args, bundle)
    score_rows: list[dict[str, Any]] = []
    score_details: list[dict[str, Any]] = []
    skipped_runs: list[str] = []
    if not args.static_only:
        score_rows, score_details, skipped_runs = build_score_shift_rows(args, bundle, root)

    static_fields = [
        "preset",
        "vowel",
        "feature_idx",
        "feature",
        "backend",
        "n_eent_train",
        "n_svd",
        "eent_train_mean",
        "eent_train_std",
        "svd_mean",
        "svd_std",
        "mean_shift_z",
        "std_ratio",
        "svd_abs_z_p95",
        "svd_abs_z_max",
        "svd_pct_abs_z_gt3",
        "svd_pct_abs_z_gt5",
        "ks_stat",
        "wasserstein",
        "shift_score",
    ]
    score_fields = [
        "run",
        "run_dir",
        "dataset",
        "label",
        "label_name",
        "threshold",
        "threshold_source",
        "n",
        "mean",
        "std",
        "min",
        "q25",
        "median",
        "q75",
        "max",
        "pct_ge_threshold",
    ]
    _write_tsv(args.out_static_tsv, static_rows, static_fields)
    _write_tsv(args.out_score_tsv, score_rows, score_fields)
    payload = {
        "static_feature_source": str(args.static_feature_source),
        "static_feature_table": str(args.static_feature_table) if args.static_feature_table is not None else None,
        "presets": list(args.presets),
        "static_summary": static_summary,
        "static_rows": static_rows,
        "score_summary_rows": score_rows,
        "score_details": score_details,
        "skipped_runs": skipped_runs,
    }
    args.out_json.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    args.out_html.write_text(
        build_html(
            args=args,
            root=root,
            static_rows=static_rows,
            static_summary=static_summary,
            score_rows=score_rows,
            skipped_runs=skipped_runs,
        ),
        encoding="utf-8",
    )
    print(f"Wrote HTML report: {args.out_html.resolve()}")
    print(f"Wrote JSON report: {args.out_json.resolve()}")
    print(f"Wrote static TSV: {args.out_static_tsv.resolve()}")
    print(f"Wrote score TSV: {args.out_score_tsv.resolve()}")
    print(f"Static rows: {len(static_rows)}; scored runs: {len({r['run'] for r in score_rows})}")


def build_parser() -> argparse.ArgumentParser:
    from split_ast_mae_cli import _add_data_args, _add_model_args

    p = argparse.ArgumentParser(
        description=(
            "Build two diagnostics for SplitAST-MAE cross-domain behavior: "
            "static-feature domain shift and patient-level score distributions."
        )
    )
    _add_data_args(p)
    _add_model_args(p)
    p.set_defaults(static_feature_source="table", static_feature_table=DEFAULT_STATIC_TABLE)
    p.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    p.add_argument("--run-dirs", nargs="*", type=Path, default=None, help="Stage 2 run directories to score.")
    p.add_argument("--threshold-metric", choices=["macro_f1", "f1", "accuracy", "youden"], default="macro_f1")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--static-only", action="store_true", help="Only build the static-feature drift report.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-html", type=Path, default=Path("split_ast_domain_shift_report.html"))
    p.add_argument("--out-json", type=Path, default=Path("split_ast_domain_shift_report.json"))
    p.add_argument("--out-static-tsv", type=Path, default=Path("split_ast_static_domain_shift.tsv"))
    p.add_argument("--out-score-tsv", type=Path, default=Path("split_ast_patient_score_shift.tsv"))
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
