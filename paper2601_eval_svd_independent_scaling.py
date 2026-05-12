#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np

from paper2601_splitmae_cli import (
    PAPER_FOCAL_GAMMA,
    _add_data_args,
    _add_model_args,
    _args_from_metadata,
    _build_pair,
    _eval_arrays_for_dataset,
    _json_ready,
    _make_context,
    _predict_stage2_positive_probs,
    _printable_pair_eval,
    _set_run_seed,
    _stage2_eval_loader,
    _static_config_from_args,
)


DEFAULT_RUN_DIRS = (
    Path("paper2601_splitmae_runs_local1"),
    Path("paper2601_splitmae_runs_local5"),
)
DEFAULT_RESULTS_NAME = "ast_stage2_ai_eval_svd_independent_static_scaling.json"


def _feature_abs_z_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"max": 0.0, "p95": 0.0, "mean": 0.0}
    abs_values = np.abs(values.astype(np.float32))
    return {
        "max": float(np.max(abs_values)),
        "p95": float(np.percentile(abs_values, 95)),
        "mean": float(np.mean(abs_values)),
    }


def _eval_static_features_with_svd_policy(
    args_for_vowel: argparse.Namespace,
    meta: dict[str, Any],
    x,
    patient_ids,
    dataset_name: str,
    vowel: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    dim = int(meta.get("static_feature_dim") or getattr(args_for_vowel, "static_feature_dim", 0) or 0)
    if dim <= 0:
        return None, {"enabled": False, "dataset": dataset_name, "vowel": vowel}

    from paper2601_static_features import apply_static_normalizer, compute_static_features, fit_static_normalizer

    backend = meta.get("static_feature_backend") or getattr(args_for_vowel, "static_feature_source", "none")
    raw, names, resolved = compute_static_features(
        x_nhwc=x,
        patient_ids=patient_ids,
        dataset=dataset_name,
        vowel=vowel,
        config=_static_config_from_args(args_for_vowel),
        backend=backend,
    )
    expected_names = list(meta.get("static_feature_names") or [])
    if expected_names and names != expected_names:
        raise SystemExit(
            f"Static feature names differ for /{vowel}/ {dataset_name}: "
            f"metadata backend={backend!r}, resolved backend={resolved!r}."
        )

    train_mean = np.asarray(meta.get("static_feature_mean") or [], dtype=np.float32)
    train_std = np.asarray(meta.get("static_feature_std") or [], dtype=np.float32)
    if train_mean.shape[0] != raw.shape[1] or train_std.shape[0] != raw.shape[1]:
        raise SystemExit(
            f"Static feature normalizer dim mismatch for /{vowel}/ {dataset_name}: "
            f"raw={raw.shape[1]}, mean={train_mean.shape[0]}, std={train_std.shape[0]}."
        )

    z_against_train = apply_static_normalizer(raw, train_mean, train_std)
    if dataset_name == "german":
        mean, std = fit_static_normalizer(raw)
        policy = "fit_on_current_svd_eval_raw_features"
    else:
        mean, std = train_mean, train_std
        policy = "stage2_metadata_eent_train_normalizer"

    scaled = apply_static_normalizer(raw, mean, std)
    info = {
        "enabled": True,
        "dataset": dataset_name,
        "vowel": vowel,
        "backend": resolved,
        "dim": int(raw.shape[1]),
        "n_rows": int(raw.shape[0]),
        "policy": policy,
        "diagnostic_only": dataset_name == "german",
        "abs_z_against_stage2_train_normalizer": _feature_abs_z_summary(z_against_train),
        "abs_z_after_eval_policy": _feature_abs_z_summary(scaled),
    }
    return scaled, info


def _evaluate_run(args: argparse.Namespace, run_dir: Path) -> Path:
    from paper2601_splitmae_training import BinaryFocalWithLogitsLoss, evaluate_stage2
    from voice_disorder_torch.data.load import load_all_preprocessed
    from voice_disorder_torch.evaluation.patient_eval import combined_vowel_ai_eval, model_eval_by_id
    from voice_disorder_torch.io.eval_report import save_eval_json
    from voice_disorder_torch.ui.eval_cli import print_eval

    metadata_a = run_dir / "ast_stage2_a_metadata.json"
    metadata_i = run_dir / "ast_stage2_i_metadata.json"
    if not metadata_a.is_file() or not metadata_i.is_file():
        raise SystemExit(f"Missing stage2 metadata in {run_dir}. Expected {metadata_a.name} and {metadata_i.name}.")

    base_args = copy.copy(args)
    base_args.metadata_a = metadata_a
    base_args.metadata_i = metadata_i
    base_args.results_json = run_dir / str(args.results_name)
    base_args.save_dir = run_dir

    args_a, meta_a = _args_from_metadata(base_args, metadata_a)
    args_i, meta_i = _args_from_metadata(base_args, metadata_i)
    _set_run_seed(int(args_a.model_init_seed))
    if args_a.vowel != "a" or args_i.vowel != "i":
        raise SystemExit(f"Expected /a/ and /i/ metadata, got {args_a.vowel!r} and {args_i.vowel!r}.")
    if args_a.dev_test_seed != args_i.dev_test_seed:
        raise SystemExit("metadata-a and metadata-i must use the same dev_test_seed for paired evaluation.")
    if int(args_a.num_labels) != 1 or int(args_i.num_labels) != 1:
        raise SystemExit("This diagnostic pair evaluation supports binary --num-labels 1 models only.")

    ctx = _make_context(args_a)
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    client_a, server_a, device = _build_pair(args_a)
    client_i, server_i, _ = _build_pair(args_i)

    focal_gamma_a = meta_a.get("stage2_focal_gamma")
    focal_gamma_i = meta_i.get("stage2_focal_gamma")
    focal_gamma = float(focal_gamma_a if focal_gamma_a is not None else args.focal_gamma)
    if focal_gamma_i is not None and abs(float(focal_gamma_i) - focal_gamma) > 1e-12:
        raise SystemExit("metadata-a and metadata-i use different focal gamma values.")
    criterion = BinaryFocalWithLogitsLoss(gamma=focal_gamma)

    selected = ["chinese", "german"] if args.eval_dataset == "both" else [args.eval_dataset]
    results: dict[str, Any] = {
        "stage": "evaluate-stage2-pair-svd-independent-static-scaling",
        "diagnostic_only": True,
        "diagnostic_note": (
            "For SVD/german evaluation only, static features are scaled with mean/std fit on "
            "the current SVD eval raw static features. This uses test distribution information "
            "and should be treated as a domain-shift diagnostic, not the primary final metric."
        ),
        "run_dir": str(run_dir),
        "metadata_a": str(metadata_a),
        "metadata_i": str(metadata_i),
        "client_a_file": str(args_a.load_client),
        "server_a_file": str(args_a.load_server),
        "client_i_file": str(args_i.load_client),
        "server_i_file": str(args_i.load_server),
        "dev_test_seed": int(args_a.dev_test_seed),
        "train_val_seed_a": int(args_a.train_val_seed),
        "train_val_seed_i": int(args_i.train_val_seed),
        "patient_eval_strategy": args.patient_eval_strategy,
        "patient_prob_threshold": float(args.patient_prob_threshold),
        "focal_gamma": focal_gamma,
        "svd_static_scaling_policy": "fit_on_current_svd_eval_raw_features",
        "eent_static_scaling_policy": "stage2_metadata_eent_train_normalizer",
        "loaded_metadata_a": meta_a,
        "loaded_metadata_i": meta_i,
        "datasets": {},
    }

    for dataset_name in selected:
        xa, ya, ida, display_name = _eval_arrays_for_dataset(bundle, "a", dataset_name)
        xi, yi, idi, _ = _eval_arrays_for_dataset(bundle, "i", dataset_name)
        static_a, static_info_a = _eval_static_features_with_svd_policy(args_a, meta_a, xa, ida, dataset_name, "a")
        static_i, static_info_i = _eval_static_features_with_svd_policy(args_i, meta_i, xi, idi, dataset_name, "i")
        ds_a, loader_a = _stage2_eval_loader(xa, ya, int(args_a.input_tdim), int(args_a.batch_size), static_a)
        ds_i, loader_i = _stage2_eval_loader(xi, yi, int(args_i.input_tdim), int(args_i.batch_size), static_i)

        seg_a = evaluate_stage2(client=client_a, server=server_a, loader=loader_a, device=device, criterion=criterion)
        seg_i = evaluate_stage2(client=client_i, server=server_i, loader=loader_i, device=device, criterion=criterion)
        pa = _predict_stage2_positive_probs(client_a, server_a, loader_a, device)
        pi = _predict_stage2_positive_probs(client_i, server_i, loader_i, device)

        single_a = model_eval_by_id(
            xa,
            ya,
            list(ida),
            pa,
            vowel_type="a",
            dataset_type=display_name,
            strategy=args.patient_eval_strategy,
            patient_prob_threshold=float(args.patient_prob_threshold),
            verbose=bool(args.verbose),
        )
        single_i = model_eval_by_id(
            xi,
            yi,
            list(idi),
            pi,
            vowel_type="i",
            dataset_type=display_name,
            strategy=args.patient_eval_strategy,
            patient_prob_threshold=float(args.patient_prob_threshold),
            verbose=bool(args.verbose),
        )
        combined = combined_vowel_ai_eval(
            pa,
            pi,
            ya,
            yi,
            ida,
            idi,
            dataset_type=display_name,
            strategy=args.patient_eval_strategy,
            patient_prob_threshold=float(args.patient_prob_threshold),
            verbose=bool(args.verbose),
        )

        results["datasets"][dataset_name] = {
            "static_normalization_a": static_info_a,
            "static_normalization_i": static_info_i,
            "segment_a": {
                "segment_loss": float(seg_a.loss),
                "segment_macro_f1": float(seg_a.score),
                "n_segments": int(len(ds_a)),
                "n_patients": int(len(set(str(pid) for pid in ida))),
            },
            "segment_i": {
                "segment_loss": float(seg_i.loss),
                "segment_macro_f1": float(seg_i.score),
                "n_segments": int(len(ds_i)),
                "n_patients": int(len(set(str(pid) for pid in idi))),
            },
            "single_a": single_a,
            "single_i": single_i,
            "combined": combined,
        }

    results = _json_ready(results)
    print_eval(_printable_pair_eval(results), verbose=bool(args.verbose))
    save_eval_json(base_args.results_json, results)
    print(f"Wrote diagnostic evaluation JSON: {base_args.results_json.resolve()}")
    return base_args.results_json


def _default_existing_run_dirs() -> list[Path]:
    dirs = [path for path in DEFAULT_RUN_DIRS if (path / "ast_stage2_a_metadata.json").is_file()]
    if dirs:
        return dirs
    fallback = Path("paper2601_splitmae_runs")
    if (fallback / "ast_stage2_a_metadata.json").is_file():
        return [fallback]
    return list(DEFAULT_RUN_DIRS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Re-evaluate finished Paper2601 Stage 2 /a/+/i/ runs while fitting static-feature "
            "normalization independently on SVD eval raw features."
        )
    )
    _add_data_args(p)
    _add_model_args(p)
    p.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Finished run directory. Repeatable. Default: existing local1/local5 run dirs.",
    )
    p.add_argument("--eval-dataset", choices=["chinese", "german", "both"], default="both")
    p.add_argument(
        "--patient-eval-strategy",
        choices=["fixed", "best_threshold", "relative", "percentage", "max recall", "guding"],
        default="fixed",
    )
    p.add_argument("--patient-prob-threshold", type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=PAPER_FOCAL_GAMMA)
    p.add_argument("--results-name", type=str, default=DEFAULT_RESULTS_NAME)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = args.run_dir if args.run_dir else _default_existing_run_dirs()
    written: list[Path] = []
    for run_dir in run_dirs:
        print(f"\n=== Diagnostic evaluate: {run_dir} ===")
        written.append(_evaluate_run(args, run_dir))
    print("\nDiagnostic result files:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
