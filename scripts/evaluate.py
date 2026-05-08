#!/usr/bin/env python3
"""Evaluate a saved /a/ + /i/ model pair on Chinese test + German external set."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from _path_setup import ensure_project_package_on_path

ensure_project_package_on_path()

from voice_disorder_torch.config import DataPaths, TrainConfig, apply_default_project_paths
from voice_disorder_torch.io.eval_report import save_eval_json
from voice_disorder_torch.pipeline.eval_pair import evaluate_model_pair_test_only
from voice_disorder_torch.ui.eval_cli import print_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate two checkpoints (same architecture).")
    p.add_argument("--pickle-dir", type=Path, default=None, help="Single folder with all pkls.")
    p.add_argument(
        "--pickle-dir-eent",
        type=Path,
        default=None,
        help="Preproc EENT pickle dir: Data/EENT_processed/pickle_files.",
    )
    p.add_argument(
        "--pickle-dir-svd",
        type=Path,
        default=None,
        help="Preproc SVD pickle dir: Data/SVD_processed/pickle_files.",
    )
    p.add_argument("--eent-subjects-xlsx", type=Path, default=None)
    p.add_argument(
        "--german-subjects-xlsx",
        type=Path,
        default=None,
        help="SVD subject table (e.g. metadata/subjects/SVD.xlsx).",
    )
    p.add_argument("--save-dir", type=Path, default=Path("./saved_models"))
    p.add_argument("--model-a", type=str, required=True, help="Stem, e.g. cnn_a_d8_t100_i2718")
    p.add_argument("--model-i", type=str, required=True)
    p.add_argument("--model-type", type=str, default="cnn")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--verbose", action="store_true", help="Print single-vowel blocks too.")
    p.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Write full evaluation dict (metrics, optimal_threshold, roc_thresholds, ROC points, meta) to this JSON file.",
    )
    _strategies = ("fixed", "best_threshold", "guding", "relative", "percentage", "max recall")
    p.add_argument(
        "--patient-eval-strategy",
        type=str,
        default="fixed",
        choices=_strategies,
        help="Patient-level decision: 'fixed' = mean prob >= --patient-prob-threshold (deployment-like). "
        "'best_threshold' = Youden on this eval set (optimistic Acc/F1; not for deployment).",
    )
    p.add_argument(
        "--patient-prob-threshold",
        type=float,
        default=0.5,
        help="Used when strategy is fixed/guding: patient mean positive prob >= this -> predict 1.",
    )
    args = p.parse_args()
    apply_default_project_paths(args)

    if (args.pickle_dir_eent is None) ^ (args.pickle_dir_svd is None):
        p.error("Provide both --pickle-dir-eent and --pickle-dir-svd, or neither (then use --pickle-dir).")
    if args.pickle_dir is None and (args.pickle_dir_eent is None or args.pickle_dir_svd is None):
        p.error("Provide --pickle-dir, or both --pickle-dir-eent and --pickle-dir-svd.")

    paths = DataPaths(
        pickle_dir=args.pickle_dir,
        pickle_dir_chinese=args.pickle_dir_eent,
        pickle_dir_german=args.pickle_dir_svd,
        german_subjects_xlsx=args.german_subjects_xlsx,
        eent_subjects_xlsx=args.eent_subjects_xlsx,
    )
    train_cfg = TrainConfig()
    results = evaluate_model_pair_test_only(
        paths=paths,
        save_dir=args.save_dir,
        model_a_stem=args.model_a,
        model_i_stem=args.model_i,
        model_type=args.model_type,
        train_cfg=train_cfg,
        device=args.device,
        patient_eval_strategy=args.patient_eval_strategy,
        patient_prob_threshold=float(args.patient_prob_threshold),
        verbose=bool(args.verbose),
    )
    print_eval(results, verbose=bool(args.verbose))

    if args.results_json is not None:
        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": args.model_type,
            "device": args.device,
            "model_a": args.model_a,
            "model_i": args.model_i,
            "save_dir": str(Path(args.save_dir).resolve()),
            "pickle_dir": str(Path(args.pickle_dir).resolve()) if args.pickle_dir else None,
            "pickle_dir_eent": str(Path(args.pickle_dir_eent).resolve()) if args.pickle_dir_eent else None,
            "pickle_dir_svd": str(Path(args.pickle_dir_svd).resolve()) if args.pickle_dir_svd else None,
            "eent_subjects_xlsx": str(Path(args.eent_subjects_xlsx).resolve()) if args.eent_subjects_xlsx else None,
            "german_subjects_xlsx": str(Path(args.german_subjects_xlsx).resolve()) if args.german_subjects_xlsx else None,
            "train_config": asdict(train_cfg),
            "patient_eval_strategy": args.patient_eval_strategy,
            "patient_prob_threshold": float(args.patient_prob_threshold),
        }
        save_eval_json(args.results_json, {"meta": meta, "evaluation": results})
        print(f"Wrote evaluation JSON: {args.results_json.resolve()}")


if __name__ == "__main__":
    main()
