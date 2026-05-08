#!/usr/bin/env python3
"""Split learning for one vowel or an /a/+/i/ pair."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from _path_setup import ensure_project_package_on_path, find_project_root

ensure_project_package_on_path()

from voice_disorder_torch.config import (
    DEFAULT_SSAST_MODEL_SIZE,
    DataPaths,
    RunContext,
    SplitSeeds,
    TrainConfig,
    apply_default_project_paths,
    default_ssast_checkpoint_path,
)
from voice_disorder_torch.reproducibility import set_reproducible
from voice_disorder_torch.split_learning import train_split_cnn, train_split_ssast


def main() -> None:
    p = argparse.ArgumentParser(description="Split learning (CNN or SSAST) with voice_disorder_torch data.")
    p.add_argument("--pickle-dir", type=Path, default=None)
    p.add_argument("--pickle-dir-eent", type=Path, default=None)
    p.add_argument("--pickle-dir-svd", type=Path, default=None)
    p.add_argument("--eent-subjects-xlsx", type=Path, default=None)
    p.add_argument("--german-subjects-xlsx", type=Path, default=None)
    p.add_argument("--save-dir", type=Path, default=Path("./saved_models"))
    p.add_argument("--vowel", choices=["a", "i", "both"], default="both")
    p.add_argument("--dev-test-seed", type=int, default=8)
    p.add_argument("--train-val-seed", type=int, default=100)
    p.add_argument("--model-init-seed", type=int, default=2718)
    p.add_argument("--partition-seed", type=int, default=42, help="RNG for assigning patients to client partitions.")
    p.add_argument("--model-type", type=str, default="cnn", help="cnn | ssast (requires timm==0.4.5 in env).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-partitions", type=int, default=5, help="Number of patient-level client partitions.")
    p.add_argument(
        "--n-global-epochs",
        type=int,
        default=TrainConfig().max_epochs,
        help="Maximum number of split-learning global epochs (aligned with TrainConfig.max_epochs).",
    )
    p.add_argument("--n-local-epochs", type=int, default=5, help="Local epochs per global epoch (aligned default).")
    p.add_argument("--batch-size", type=int, default=None, help="Default: TrainConfig.batch_size.")
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Global rounds without val_loss improvement before stop (default: TrainConfig=10). "
        "0=run all --n-global-epochs but still reload best-val weights before save.",
    )
    p.add_argument("--client-lr", type=float, default=TrainConfig().lr, help="Default: TrainConfig.lr.")
    p.add_argument("--server-lr", type=float, default=TrainConfig().lr, help="Default: TrainConfig.lr.")
    p.add_argument("--n-client-blocks", type=int, default=2)
    p.add_argument("--ssast-input-tdim", type=int, default=259)
    p.add_argument("--ssast-input-fdim", type=int, default=128)
    p.add_argument("--ssast-f-shape", type=int, default=16)
    p.add_argument("--ssast-t-shape", type=int, default=16)
    p.add_argument("--ssast-f-stride", type=int, default=10)
    p.add_argument("--ssast-t-stride", type=int, default=10)
    p.add_argument("--ssast-model-size", type=str, default=DEFAULT_SSAST_MODEL_SIZE)
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
        train_cfg = replace(train_cfg, batch_size=int(args.batch_size))
    if args.early_stopping_patience is not None:
        train_cfg = replace(train_cfg, early_stopping_patience=int(args.early_stopping_patience))
    ctx = RunContext(
        paths=paths,
        splits=SplitSeeds(dev_test_seed=args.dev_test_seed, train_val_seed=args.train_val_seed),
        train=train_cfg,
        save_dir=args.save_dir,
        device=args.device,
    )
    mt = args.model_type.lower().strip()
    if mt not in {"cnn", "cnn2d", "cnn2d_original", "ssast", "ast"}:
        p.error(f"Unsupported --model-type: {args.model_type}")

    pt: Path | None = None
    if mt in {"ssast", "ast"}:
        try:
            pt = default_ssast_checkpoint_path(args.ssast_model_size, find_project_root())
        except ValueError as exc:
            p.error(str(exc))
        if not pt.is_file():
            p.error(f"SSAST pretrained checkpoint not found: {pt}")

    for vowel in (("a", "i") if args.vowel == "both" else (args.vowel,)):
        set_reproducible(int(args.model_init_seed))
        if mt in {"cnn", "cnn2d", "cnn2d_original"}:
            train_split_cnn(
                ctx=ctx,
                vowel=vowel,
                model_init_seed=int(args.model_init_seed),
                n_partitions=int(args.n_partitions),
                partition_seed=int(args.partition_seed),
                n_global_epochs=int(args.n_global_epochs),
                n_local_epochs=int(args.n_local_epochs),
                client_lr=float(args.client_lr),
                server_lr=float(args.server_lr),
            )
        else:
            assert pt is not None
            train_split_ssast(
                ctx=ctx,
                vowel=vowel,
                model_init_seed=int(args.model_init_seed),
                n_partitions=int(args.n_partitions),
                partition_seed=int(args.partition_seed),
                n_global_epochs=int(args.n_global_epochs),
                n_local_epochs=int(args.n_local_epochs),
                client_lr=float(args.client_lr),
                server_lr=float(args.server_lr),
                pretrained_path=pt,
                n_client_blocks=int(args.n_client_blocks),
                input_fdim=int(args.ssast_input_fdim),
                input_tdim=int(args.ssast_input_tdim),
                f_shape=int(args.ssast_f_shape),
                t_shape=int(args.ssast_t_shape),
                f_stride=int(args.ssast_f_stride),
                t_stride=int(args.ssast_t_stride),
                model_size=str(args.ssast_model_size),
            )


if __name__ == "__main__":
    main()


