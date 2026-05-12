from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from paper2601_splitmae_cli import (
    PAPER_FOCAL_GAMMA,
    _add_data_args,
    _add_model_args,
    _args_from_metadata,
    _build_pair,
    _eval_arrays_for_dataset,
    _eval_static_features,
    _json_ready,
    _make_context,
    _predict_stage2_positive_probs,
    _set_run_seed,
    _stage2_eval_loader,
)
from voice_disorder_torch.data.load import load_all_preprocessed
from voice_disorder_torch.io.eval_report import save_eval_json
from voice_disorder_torch.ui.eval_cli import print_eval


def _metadata_is_controlled(meta: dict[str, Any]) -> bool:
    return bool(meta.get("controlled_fusion")) or str(meta.get("stage", "")).endswith("controlled")


def _args_from_any_metadata(base_args: argparse.Namespace, metadata_path: Path):
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if _metadata_is_controlled(meta):
        from paper2601_controlled_fusion_cli import _args_from_controlled_metadata

        return _args_from_controlled_metadata(base_args, metadata_path)
    return _args_from_metadata(base_args, metadata_path)


def _build_any_pair(args: argparse.Namespace, meta: dict[str, Any]):
    if _metadata_is_controlled(meta):
        from paper2601_controlled_fusion_cli import _build_controlled_pair

        return _build_controlled_pair(args)
    return _build_pair(args)


def _eval_any_static_features(
    args_for_vowel: argparse.Namespace,
    meta: dict[str, Any],
    x,
    patient_ids,
    dataset_name: str,
    vowel: str,
):
    if _metadata_is_controlled(meta):
        from paper2601_controlled_fusion_cli import _eval_static_features_controlled

        return _eval_static_features_controlled(args_for_vowel, meta, x, patient_ids, dataset_name, vowel)
    return _eval_static_features(args_for_vowel, meta, x, patient_ids, dataset_name, vowel)


def _as_binary_labels(y) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1).astype(int)
    if arr.ndim == 1:
        return arr.astype(int)
    return np.argmax(arr, axis=1).astype(int)


def _normalize_patient_id(pid) -> str:
    if hasattr(pid, "item"):
        pid = pid.item()
    return str(pid).strip()


def _aggregate_patient_scores(
    probs_a: np.ndarray,
    probs_i: np.ndarray,
    y_a,
    y_i,
    ids_a,
    ids_i,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    probs = np.concatenate([np.asarray(probs_a).reshape(-1), np.asarray(probs_i).reshape(-1)])
    labels = np.concatenate([_as_binary_labels(y_a), _as_binary_labels(y_i)])
    ids = np.concatenate([np.asarray(ids_a).reshape(-1), np.asarray(ids_i).reshape(-1)])
    patients: dict[str, dict[str, Any]] = {}
    for prob, label, pid in zip(probs, labels, ids):
        key = _normalize_patient_id(pid)
        if key not in patients:
            patients[key] = {"label": int(label), "probs": []}
        patients[key]["probs"].append(float(prob))
    patient_ids = list(patients.keys())
    y_true = np.asarray([patients[pid]["label"] for pid in patient_ids], dtype=int)
    y_score = np.asarray([float(np.mean(patients[pid]["probs"])) for pid in patient_ids], dtype=np.float64)
    return patient_ids, y_true, y_score


def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    auc = None
    if len(set(int(x) for x in y_true.tolist())) == 2:
        auc = float(roc_auc_score(y_true, y_score))
    return {
        "auc": auc,
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "patient_prob_threshold": float(threshold),
        "metrics": {
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def _candidate_thresholds(y_score: np.ndarray) -> list[float]:
    unique = sorted({float(x) for x in np.asarray(y_score).reshape(-1)})
    if not unique:
        return [0.5]
    candidates = [unique[0] - 1e-6, *unique, unique[-1] + 1e-6]
    return [float(min(max(x, 0.0), 1.0)) for x in candidates]


def _select_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str) -> dict[str, Any]:
    metric = str(metric).lower().strip()
    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        finite = np.isfinite(thresholds)
        scores = tpr - fpr
        scores = np.where(finite, scores, -np.inf)
        idx = int(np.argmax(scores))
        threshold = float(thresholds[idx])
        selected = _metrics_at_threshold(y_true, y_score, threshold)
        selected_metric = float(scores[idx])
    else:
        best = None
        for threshold in _candidate_thresholds(y_score):
            block = _metrics_at_threshold(y_true, y_score, threshold)
            m = block["metrics"]
            if metric == "macro_f1":
                score = float(m["macro_f1"])
            elif metric == "f1":
                score = float(m["f1_score"])
            elif metric == "accuracy":
                score = float(m["accuracy"])
            else:
                raise ValueError(f"Unknown threshold metric: {metric}")
            key = (score, -abs(float(threshold) - 0.5))
            if best is None or key > best[0]:
                best = (key, threshold, block, score)
        _, threshold, selected, selected_metric = best
    return {
        "threshold": float(threshold),
        "threshold_metric": metric,
        "threshold_metric_value": float(selected_metric),
        "validation_combined": selected,
    }


def _val_arrays_for_vowel(bundle, vowel: str):
    if vowel == "a":
        return bundle.x_val_a, bundle.y_val_a, bundle.id_val_a, "Chinese-Val"
    if vowel == "i":
        return bundle.x_val_i, bundle.y_val_i, bundle.id_val_i, "Chinese-Val"
    raise ValueError(vowel)


def _predict_for_arrays(
    *,
    client,
    server,
    args_for_vowel: argparse.Namespace,
    meta: dict[str, Any],
    x,
    y,
    ids,
    dataset_name: str,
    vowel: str,
    device,
) -> np.ndarray:
    static = _eval_any_static_features(args_for_vowel, meta, x, ids, dataset_name, vowel)
    _, loader = _stage2_eval_loader(x, y, int(args_for_vowel.input_tdim), int(args_for_vowel.batch_size), static)
    return _predict_stage2_positive_probs(client, server, loader, device)


def _evaluate_dataset(
    *,
    bundle,
    dataset_name: str,
    threshold: float,
    client_a,
    server_a,
    args_a: argparse.Namespace,
    meta_a: dict[str, Any],
    client_i,
    server_i,
    args_i: argparse.Namespace,
    meta_i: dict[str, Any],
    device,
) -> dict[str, Any]:
    xa, ya, ida, display_name = _eval_arrays_for_dataset(bundle, "a", dataset_name)
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
    combined = _metrics_at_threshold(y_true, y_score, threshold)
    combined["dataset_type"] = display_name
    combined["n_patients"] = int(len(patient_ids))
    combined["patient_ids"] = patient_ids
    return {
        "combined": combined,
        "segment_a": {"n_segments": int(len(xa)), "n_patients": int(len(set(str(pid) for pid in ida)))},
        "segment_i": {"n_segments": int(len(xi)), "n_patients": int(len(set(str(pid) for pid in idi)))},
    }


def cmd_eval(args: argparse.Namespace) -> None:
    if args.run_dir is not None:
        args.metadata_a = Path(args.run_dir) / "ast_stage2_a_metadata.json"
        args.metadata_i = Path(args.run_dir) / "ast_stage2_i_metadata.json"
    if args.metadata_a is None or args.metadata_i is None:
        raise SystemExit("Set --run-dir or both --metadata-a/--metadata-i.")
    if not args.metadata_a.is_file() or not args.metadata_i.is_file():
        raise SystemExit(f"Missing metadata files: {args.metadata_a} / {args.metadata_i}")

    args_a, meta_a = _args_from_any_metadata(args, args.metadata_a)
    args_i, meta_i = _args_from_any_metadata(args, args.metadata_i)
    if args_a.vowel != "a" or args_i.vowel != "i":
        raise SystemExit("Expected metadata-a for /a/ and metadata-i for /i/.")
    if args_a.dev_test_seed != args_i.dev_test_seed:
        raise SystemExit("metadata-a and metadata-i must use the same dev_test_seed.")
    if int(args_a.num_labels) != 1 or int(args_i.num_labels) != 1:
        raise SystemExit("This script currently supports binary --num-labels 1 models only.")

    _set_run_seed(int(args_a.model_init_seed))
    ctx = _make_context(args_a)
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    client_a, server_a, device = _build_any_pair(args_a, meta_a)
    client_i, server_i, _ = _build_any_pair(args_i, meta_i)

    xva, yva, idva, _ = _val_arrays_for_vowel(bundle, "a")
    xvi, yvi, idvi, _ = _val_arrays_for_vowel(bundle, "i")
    pva = _predict_for_arrays(
        client=client_a,
        server=server_a,
        args_for_vowel=args_a,
        meta=meta_a,
        x=xva,
        y=yva,
        ids=idva,
        dataset_name="chinese",
        vowel="a",
        device=device,
    )
    pvi = _predict_for_arrays(
        client=client_i,
        server=server_i,
        args_for_vowel=args_i,
        meta=meta_i,
        x=xvi,
        y=yvi,
        ids=idvi,
        dataset_name="chinese",
        vowel="i",
        device=device,
    )
    val_patient_ids, val_true, val_score = _aggregate_patient_scores(pva, pvi, yva, yvi, idva, idvi)
    calibration = _select_threshold(val_true, val_score, args.threshold_metric)
    threshold = float(calibration["threshold"])
    calibration["validation_patient_ids"] = val_patient_ids
    calibration["n_validation_patients"] = int(len(val_patient_ids))

    selected = ["chinese", "german"] if args.eval_dataset == "both" else [args.eval_dataset]
    results = {
        "stage": "evaluate-stage2-pair-eent-val-threshold",
        "metadata_a": str(args.metadata_a),
        "metadata_i": str(args.metadata_i),
        "client_a_file": str(args_a.load_client),
        "server_a_file": str(args_a.load_server),
        "client_i_file": str(args_i.load_client),
        "server_i_file": str(args_i.load_server),
        "dev_test_seed": int(args_a.dev_test_seed),
        "train_val_seed_a": int(args_a.train_val_seed),
        "train_val_seed_i": int(args_i.train_val_seed),
        "patient_eval_strategy": "eent_validation_threshold",
        "threshold_metric": str(args.threshold_metric),
        "patient_prob_threshold": threshold,
        "focal_gamma": float(meta_a.get("stage2_focal_gamma") or PAPER_FOCAL_GAMMA),
        "loaded_metadata_a": meta_a,
        "loaded_metadata_i": meta_i,
        "calibration": calibration,
        "datasets": {},
    }
    for dataset_name in selected:
        results["datasets"][dataset_name] = _evaluate_dataset(
            bundle=bundle,
            dataset_name=dataset_name,
            threshold=threshold,
            client_a=client_a,
            server_a=server_a,
            args_a=args_a,
            meta_a=meta_a,
            client_i=client_i,
            server_i=server_i,
            args_i=args_i,
            meta_i=meta_i,
            device=device,
        )

    results = _json_ready(results)
    print(
        f"EENT validation threshold: {threshold:.6f} "
        f"({args.threshold_metric}={calibration['threshold_metric_value']:.4f})"
    )
    printable = {
        "patient_eval_strategy": "eent_validation_threshold",
        "patient_prob_threshold": threshold,
        **results["datasets"],
    }
    print_eval(printable, verbose=bool(args.verbose))
    if args.results_json is None:
        args.results_json = Path(args.metadata_a).parent / f"ast_stage2_ai_eval_eent_val_threshold_{args.threshold_metric}.json"
    save_eval_json(Path(args.results_json), results)
    print(f"Wrote validation-threshold evaluation JSON: {Path(args.results_json).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Select a patient-level probability threshold on EENT validation, then evaluate "
            "saved Paper2601 /a/+/i/ Stage 2 models on EENT test and/or SVD with that fixed threshold."
        )
    )
    _add_data_args(p)
    _add_model_args(p)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--metadata-a", type=Path, default=None)
    p.add_argument("--metadata-i", type=Path, default=None)
    p.add_argument("--eval-dataset", choices=["chinese", "german", "both"], default="both")
    p.add_argument("--threshold-metric", choices=["macro_f1", "f1", "accuracy", "youden"], default="macro_f1")
    p.add_argument("--results-json", type=Path, default=None)
    p.add_argument("--focal-gamma", type=float, default=PAPER_FOCAL_GAMMA)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_eval)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
