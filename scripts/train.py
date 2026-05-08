#!/usr/bin/env python3
"""Train mel classifiers for one vowel or an /a/+/i/ pair."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from _path_setup import ensure_project_package_on_path

ensure_project_package_on_path()

from voice_disorder_torch.config import DataPaths, RunContext, SplitSeeds, TrainConfig, apply_default_project_paths
from voice_disorder_torch.pipeline.train_vowel import train_one_vowel
from voice_disorder_torch.reproducibility import set_reproducible


def main() -> None:
    p = argparse.ArgumentParser(description="Train vowel /a/, /i/, or an /a/+/i/ pair.")
    p.add_argument(
        "--pickle-dir",
        type=Path,
        default=None,
        help="Folder with both Chinese + German mel pkls. Omit if using --pickle-dir-eent and --pickle-dir-svd.",
    )
    p.add_argument(
        "--pickle-dir-eent",
        type=Path,
        default=None,
        help="EENT pickles (Preproc 04): Data/EENT_processed/pickle_files — vowel-a_mel_ch.pkl, vowel-i_mel_ch.pkl.",
    )
    p.add_argument(
        "--pickle-dir-svd",
        type=Path,
        default=None,
        help="SVD pickles (Preproc 05): Data/SVD_processed/pickle_files — a_mel_ger.pkl, i_mel_ger.pkl.",
    )
    p.add_argument(
        "--eent-subjects-xlsx",
        type=Path,
        default=None,
        help="EENT table: first row = headers; patient ID = column 'Final Random ID' (or 2nd column).",
    )
    p.add_argument(
        "--german-subjects-xlsx",
        type=Path,
        default=None,
        help="SVD subject table (e.g. metadata/subjects/SVD.xlsx): Keep ID, Class, Gender, Age.",
    )
    p.add_argument("--save-dir", type=Path, default=Path("./saved_models"))
    p.add_argument("--vowel", choices=["a", "i", "both"], default="both")
    p.add_argument("--dev-test-seed", type=int, default=8)
    p.add_argument("--train-val-seed", type=int, default=100)
    p.add_argument("--model-init-seed", type=int, default=2718)
    p.add_argument("--model-type", type=str, default="cnn")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Default: TrainConfig.batch_size.")
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap training epochs (default: 1000 from TrainConfig). Early stopping may stop earlier.",
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
    if args.batch_size is not None:
        train_cfg = replace(train_cfg, batch_size=max(1, int(args.batch_size)))
    if args.max_epochs is not None:
        train_cfg = replace(train_cfg, max_epochs=max(1, int(args.max_epochs)))
    ctx = RunContext(
        paths=paths,
        splits=SplitSeeds(dev_test_seed=args.dev_test_seed, train_val_seed=args.train_val_seed),
        train=train_cfg,
        save_dir=args.save_dir,
        device=args.device,
    )
    for vowel in (("a", "i") if args.vowel == "both" else (args.vowel,)):
        set_reproducible(int(args.model_init_seed))
        train_one_vowel(ctx=ctx, vowel=vowel, model_type=args.model_type, cnn_init_seed=int(args.model_init_seed))


if __name__ == "__main__":
    main()
