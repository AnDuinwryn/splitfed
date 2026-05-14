#!/usr/bin/env python3
"""Mainline reproducible experiment pipeline.

This script keeps the final publication-facing experiments in one entry point while
reusing the existing centralized, split-learning, and SplitAST-MAE CLIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_TABLE = Path(
    "split_ast_local_artifacts/split_ast_static_131_features/split_ast_static_131_by_patient_vowel.csv"
)
DEFAULT_MODELS = [
    "centralized_cnn",
    "split_cnn",
    "split_ssast",
    "split_ast_all131",
    "split_ast_mae_only",
    "split_ast_static_only_all131",
    "split_ast_all131_gated_clip",
    "split_ast_pathology22_gated_clip",
]
MODEL_CHOICES = [
    *DEFAULT_MODELS,
    "split_ast_audio_primary_stable_static",
]


@dataclass(frozen=True)
class EvalRecord:
    seed: int
    model: str
    protocol: str
    path: Path


def _script(path: str) -> str:
    return str(ROOT / path)


def _cmd(script: str, *args: object) -> list[str]:
    return [sys.executable, _script(script), *(str(a) for a in args)]


def _print_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print(f"\n$ {_print_command(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def _mkdir(path: Path, *, dry_run: bool = False) -> None:
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def _data_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for flag, value in (
        ("--pickle-dir", args.pickle_dir),
        ("--pickle-dir-eent", args.pickle_dir_eent),
        ("--pickle-dir-svd", args.pickle_dir_svd),
        ("--eent-subjects-xlsx", args.eent_subjects_xlsx),
        ("--german-subjects-xlsx", args.german_subjects_xlsx),
    ):
        if value is not None:
            out.extend([flag, str(value)])
    return out


def _ast_model_args(args: argparse.Namespace, seed: int) -> list[str]:
    out = [
        "--model-size",
        args.ast_model_size,
        "--input-fdim",
        args.input_fdim,
        "--input-tdim",
        args.input_tdim,
        "--n-client-blocks",
        args.n_client_blocks,
        "--mask-ratio",
        args.mask_ratio,
        "--mask-strategy",
        args.mask_strategy,
        "--pooling",
        args.pooling,
        "--model-init-seed",
        seed,
        "--device",
        args.device,
    ]
    if args.imagenet_pretrain:
        out.append("--imagenet-pretrain")
    if args.audioset_checkpoint_path is not None:
        out.extend(["--audioset-checkpoint-path", str(args.audioset_checkpoint_path)])
    return [str(x) for x in out]


def _ast_train_args(args: argparse.Namespace, *, rounds: int, local_epochs: int, save_dir: Path, run_name: str) -> list[str]:
    return [
        "--n-global-rounds",
        str(rounds),
        "--n-local-epochs",
        str(local_epochs),
        "--batch-size",
        str(args.ast_batch_size),
        "--client-lr",
        str(args.ast_client_lr),
        "--server-lr",
        str(args.ast_server_lr),
        "--weight-decay",
        str(args.ast_weight_decay),
        "--adamw-beta1",
        str(args.ast_adamw_beta1),
        "--adamw-beta2",
        str(args.ast_adamw_beta2),
        "--save-dir",
        str(save_dir),
        "--run-name",
        run_name,
    ]


def _stage1_paths(stage1_dir: Path, vowel: str) -> tuple[Path, Path]:
    stem = f"ast_stage1_{vowel}"
    return stage1_dir / f"{stem}_client.pt", stage1_dir / f"{stem}_server.pt"


def _stage2_meta(run_dir: Path, vowel: str) -> Path:
    return run_dir / f"ast_stage2_{vowel}_metadata.json"


def _centralized_stem(vowel: str, args: argparse.Namespace, seed: int) -> str:
    return f"cnn_{vowel}_d{args.dev_test_seed}_t{args.train_val_seed}_i{seed}"


def _split_cnn_stem(vowel: str, args: argparse.Namespace, seed: int) -> str:
    return f"split_cnn_{vowel}_d{args.dev_test_seed}_t{args.train_val_seed}_i{seed}"


def _split_ssast_stem(vowel: str, args: argparse.Namespace, seed: int) -> str:
    return f"split_ssast_{vowel}_d{args.dev_test_seed}_t{args.train_val_seed}_i{seed}_b{args.n_client_blocks}"


def train_eval_centralized_cnn(args: argparse.Namespace, seed: int, seed_dir: Path, dry_run: bool) -> list[EvalRecord]:
    save_dir = seed_dir / "saved_models"
    _mkdir(save_dir, dry_run=dry_run)
    _run(
        _cmd(
            "scripts/train.py",
            *_data_args(args),
            "--save-dir",
            save_dir,
            "--vowel",
            "both",
            "--model-type",
            "cnn",
            "--dev-test-seed",
            args.dev_test_seed,
            "--train-val-seed",
            args.train_val_seed,
            "--model-init-seed",
            seed,
            "--batch-size",
            args.cnn_batch_size,
            "--max-epochs",
            args.centralized_max_epochs,
            "--device",
            args.device,
        ),
        dry_run=dry_run,
    )
    if args.skip_eval:
        return []
    result = seed_dir / "eval_centralized_cnn_fixed.json"
    _run(
        _cmd(
            "scripts/evaluate.py",
            *_data_args(args),
            "--save-dir",
            save_dir,
            "--model-a",
            _centralized_stem("a", args, seed),
            "--model-i",
            _centralized_stem("i", args, seed),
            "--model-type",
            "cnn",
            "--device",
            args.device,
            "--patient-eval-strategy",
            "fixed",
            "--patient-prob-threshold",
            "0.5",
            "--results-json",
            result,
        ),
        dry_run=dry_run,
    )
    return [EvalRecord(seed, "centralized_cnn", "fixed_0.5", result)]


def train_eval_split(args: argparse.Namespace, seed: int, seed_dir: Path, model_type: str, dry_run: bool) -> list[EvalRecord]:
    save_dir = seed_dir / "saved_models"
    _mkdir(save_dir, dry_run=dry_run)
    train_cmd = _cmd(
        "scripts/split_learning/train_split.py",
        *_data_args(args),
        "--save-dir",
        save_dir,
        "--vowel",
        "both",
        "--model-type",
        model_type,
        "--dev-test-seed",
        args.dev_test_seed,
        "--train-val-seed",
        args.train_val_seed,
        "--model-init-seed",
        seed,
        "--partition-seed",
        args.partition_seed,
        "--n-partitions",
        args.n_partitions,
        "--n-global-epochs",
        args.split_global_epochs,
        "--n-local-epochs",
        args.split_local_epochs,
        "--batch-size",
        args.split_batch_size,
        "--early-stopping-patience",
        args.split_early_stopping_patience,
        "--client-lr",
        args.split_client_lr,
        "--server-lr",
        args.split_server_lr,
        "--n-client-blocks",
        args.n_client_blocks,
        "--device",
        args.device,
    )
    if model_type == "ssast":
        train_cmd.extend(
            [
                "--ssast-input-fdim",
                str(args.input_fdim),
                "--ssast-input-tdim",
                str(args.input_tdim),
                "--ssast-f-shape",
                str(args.ssast_f_shape),
                "--ssast-t-shape",
                str(args.ssast_t_shape),
                "--ssast-f-stride",
                str(args.ssast_f_stride),
                "--ssast-t-stride",
                str(args.ssast_t_stride),
                "--ssast-model-size",
                str(args.ssast_model_size),
            ]
        )
    _run(train_cmd, dry_run=dry_run)
    if args.skip_eval:
        return []
    model_key = "split_ssast" if model_type == "ssast" else "split_cnn"
    stem = _split_ssast_stem if model_type == "ssast" else _split_cnn_stem
    result = seed_dir / f"eval_{model_key}_fixed.json"
    _run(
        _cmd(
            "scripts/split_learning/evaluate_split.py",
            *_data_args(args),
            "--save-dir",
            save_dir,
            "--model-a",
            stem("a", args, seed),
            "--model-i",
            stem("i", args, seed),
            "--model-type",
            model_type,
            "--device",
            args.device,
            "--patient-eval-strategy",
            "fixed",
            "--patient-prob-threshold",
            "0.5",
            "--results-json",
            result,
        ),
        dry_run=dry_run,
    )
    return [EvalRecord(seed, model_key, "fixed_0.5", result)]


def train_stage1(args: argparse.Namespace, seed: int, stage1_dir: Path, dry_run: bool) -> None:
    _mkdir(stage1_dir, dry_run=dry_run)
    for vowel in ("a", "i"):
        _run(
            _cmd(
                "split_ast_mae_cli.py",
                "train-stage1",
                *_data_args(args),
                "--vowel",
                vowel,
                "--dev-test-seed",
                args.dev_test_seed,
                "--train-val-seed",
                args.train_val_seed,
                "--partition-seed",
                args.partition_seed,
                "--n-partitions",
                args.n_partitions,
                *_ast_model_args(args, seed),
                *_ast_train_args(
                    args,
                    rounds=args.stage1_global_rounds,
                    local_epochs=args.stage1_local_epochs,
                    save_dir=stage1_dir,
                    run_name=f"ast_stage1_{vowel}",
                ),
            ),
            dry_run=dry_run,
        )


def _standard_stage2_train(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    stage1_dir: Path,
    *,
    static_table: Optional[Path],
    static_preset: str,
    dry_run: bool,
) -> None:
    _mkdir(run_dir, dry_run=dry_run)
    for vowel in ("a", "i"):
        stage1_client, stage1_server = _stage1_paths(stage1_dir, vowel)
        static_args: list[object] = ["--static-feature-source", "none"]
        if static_table is not None:
            static_args = [
                "--static-feature-source",
                "table",
                "--static-feature-table",
                static_table,
                "--static-feature-preset",
                static_preset,
            ]
        _run(
            _cmd(
                "split_ast_mae_cli.py",
                "train-stage2",
                *_data_args(args),
                "--vowel",
                vowel,
                "--dev-test-seed",
                args.dev_test_seed,
                "--train-val-seed",
                args.train_val_seed,
                "--partition-seed",
                args.partition_seed,
                "--n-partitions",
                args.n_partitions,
                *_ast_model_args(args, seed),
                *static_args,
                *_ast_train_args(
                    args,
                    rounds=args.stage2_global_rounds,
                    local_epochs=args.stage2_local_epochs,
                    save_dir=run_dir,
                    run_name=f"ast_stage2_{vowel}",
                ),
                "--early-stopping-patience",
                args.ast_early_stopping_patience,
                "--focal-gamma",
                args.ast_focal_gamma,
                "--load-client",
                stage1_client,
                "--load-server",
                stage1_server,
            ),
            dry_run=dry_run,
        )


def _controlled_stage2_train(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    stage1_dir: Path,
    *,
    static_table: Optional[Path],
    static_preset: str,
    fusion_mode: str,
    static_projection_dim: int,
    static_dropout: float,
    static_gate_init: float,
    static_z_clip: float,
    static_max_weight: Optional[float] = None,
    static_anomaly_threshold: Optional[float] = None,
    static_anomaly_scale: Optional[float] = None,
    static_aux_hidden_dim: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    _mkdir(run_dir, dry_run=dry_run)
    extra_control_args: list[object] = []
    if static_max_weight is not None:
        extra_control_args.extend(["--static-max-weight", static_max_weight])
    if static_anomaly_threshold is not None:
        extra_control_args.extend(["--static-anomaly-threshold", static_anomaly_threshold])
    if static_anomaly_scale is not None:
        extra_control_args.extend(["--static-anomaly-scale", static_anomaly_scale])
    if static_aux_hidden_dim is not None:
        extra_control_args.extend(["--static-aux-hidden-dim", static_aux_hidden_dim])
    for vowel in ("a", "i"):
        stage1_client, stage1_server = _stage1_paths(stage1_dir, vowel)
        static_args: list[object] = ["--static-feature-source", "none"]
        if static_table is not None:
            static_args = [
                "--static-feature-source",
                "table",
                "--static-feature-table",
                static_table,
                "--static-feature-preset",
                static_preset,
            ]
        _run(
            _cmd(
                "split_ast_controlled_fusion_cli.py",
                "train-stage2-controlled",
                *_data_args(args),
                "--vowel",
                vowel,
                "--dev-test-seed",
                args.dev_test_seed,
                "--train-val-seed",
                args.train_val_seed,
                "--partition-seed",
                args.partition_seed,
                "--n-partitions",
                args.n_partitions,
                *_ast_model_args(args, seed),
                *static_args,
                *_ast_train_args(
                    args,
                    rounds=args.stage2_global_rounds,
                    local_epochs=args.stage2_local_epochs,
                    save_dir=run_dir,
                    run_name=f"ast_stage2_{vowel}",
                ),
                "--early-stopping-patience",
                args.ast_early_stopping_patience,
                "--focal-gamma",
                args.ast_focal_gamma,
                "--load-client",
                stage1_client,
                "--load-server",
                stage1_server,
                "--fusion-mode",
                fusion_mode,
                "--static-projection-dim",
                static_projection_dim,
                "--static-dropout",
                static_dropout,
                "--static-gate-init",
                static_gate_init,
                "--static-z-clip",
                static_z_clip,
                *extra_control_args,
            ),
            dry_run=dry_run,
        )


def _eval_split_ast_pair(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    model_key: str,
    seed: int,
    controlled: bool,
    dry_run: bool,
) -> list[EvalRecord]:
    cli = "split_ast_controlled_fusion_cli.py" if controlled else "split_ast_mae_cli.py"
    command = "evaluate-stage2-pair-controlled" if controlled else "evaluate-stage2-pair"
    records: list[EvalRecord] = []
    fixed = run_dir / "ast_stage2_ai_eval_fixed.json"
    _run(
        _cmd(
            cli,
            command,
            *_data_args(args),
            "--metadata-a",
            _stage2_meta(run_dir, "a"),
            "--metadata-i",
            _stage2_meta(run_dir, "i"),
            "--eval-dataset",
            "both",
            "--patient-eval-strategy",
            "fixed",
            "--patient-prob-threshold",
            "0.5",
            "--results-json",
            fixed,
        ),
        dry_run=dry_run,
    )
    records.append(EvalRecord(seed, model_key, "fixed_0.5", fixed))

    if args.include_youden:
        youden = run_dir / "ast_stage2_ai_eval_youden.json"
        _run(
            _cmd(
                cli,
                command,
                *_data_args(args),
                "--metadata-a",
                _stage2_meta(run_dir, "a"),
                "--metadata-i",
                _stage2_meta(run_dir, "i"),
                "--eval-dataset",
                "both",
                "--patient-eval-strategy",
                "best_threshold",
                "--results-json",
                youden,
            ),
            dry_run=dry_run,
        )
        records.append(EvalRecord(seed, model_key, "youden_eval_set", youden))

    if args.include_eent_val_threshold:
        val_threshold = run_dir / "ast_stage2_ai_eval_eent_val_threshold_macro_f1.json"
        _run(
            _cmd(
                "split_ast_eval_eent_val_threshold.py",
                *_data_args(args),
                "--run-dir",
                run_dir,
                "--eval-dataset",
                "both",
                "--threshold-metric",
                "macro_f1",
                "--results-json",
                val_threshold,
            ),
            dry_run=dry_run,
        )
        records.append(EvalRecord(seed, model_key, "eent_val_threshold_macro_f1", val_threshold))
    return records


def train_eval_split_ast_variant(
    args: argparse.Namespace,
    seed: int,
    seed_dir: Path,
    model_key: str,
    stage1_dir: Path,
    stable_static_table: Optional[Path],
    dry_run: bool,
) -> list[EvalRecord]:
    run_dir = seed_dir / model_key
    static_table = args.static_feature_table
    controlled = False
    if model_key == "split_ast_all131":
        _standard_stage2_train(
            args,
            seed,
            run_dir,
            stage1_dir,
            static_table=static_table,
            static_preset="all",
            dry_run=dry_run,
        )
    elif model_key == "split_ast_mae_only":
        controlled = True
        _controlled_stage2_train(
            args,
            seed,
            run_dir,
            stage1_dir,
            static_table=None,
            static_preset="all",
            fusion_mode="audio_only",
            static_projection_dim=0,
            static_dropout=0.0,
            static_gate_init=0.25,
            static_z_clip=0.0,
            dry_run=dry_run,
        )
    elif model_key == "split_ast_static_only_all131":
        controlled = True
        _controlled_stage2_train(
            args,
            seed,
            run_dir,
            stage1_dir,
            static_table=static_table,
            static_preset="all",
            fusion_mode="static_only",
            static_projection_dim=0,
            static_dropout=0.0,
            static_gate_init=0.25,
            static_z_clip=3.0,
            dry_run=dry_run,
        )
    elif model_key == "split_ast_all131_gated_clip":
        controlled = True
        _controlled_stage2_train(
            args,
            seed,
            run_dir,
            stage1_dir,
            static_table=static_table,
            static_preset="all",
            fusion_mode="gated",
            static_projection_dim=args.control_static_projection_dim,
            static_dropout=args.control_static_dropout,
            static_gate_init=args.control_static_gate_init,
            static_z_clip=args.control_static_z_clip,
            dry_run=dry_run,
        )
    elif model_key == "split_ast_pathology22_gated_clip":
        controlled = True
        _controlled_stage2_train(
            args,
            seed,
            run_dir,
            stage1_dir,
            static_table=static_table,
            static_preset="pathology",
            fusion_mode="gated",
            static_projection_dim=args.control_static_projection_dim,
            static_dropout=args.control_static_dropout,
            static_gate_init=args.control_static_gate_init,
            static_z_clip=args.control_static_z_clip,
            dry_run=dry_run,
        )
    elif model_key == "split_ast_stable_static_gated_clip":
        if stable_static_table is None:
            raise RuntimeError("stable static table was not prepared")
        controlled = True
        _controlled_stage2_train(
            args,
            seed,
            run_dir,
            stage1_dir,
            static_table=stable_static_table,
            static_preset="all",
            fusion_mode="gated",
            static_projection_dim=args.control_static_projection_dim,
            static_dropout=args.control_static_dropout,
            static_gate_init=args.control_static_gate_init,
            static_z_clip=args.control_static_z_clip,
            dry_run=dry_run,
        )
    elif model_key == "split_ast_audio_primary_stable_static":
        if stable_static_table is None:
            raise RuntimeError("stable static table was not prepared")
        controlled = True
        _controlled_stage2_train(
            args,
            seed,
            run_dir,
            stage1_dir,
            static_table=stable_static_table,
            static_preset="all",
            fusion_mode="audio_primary_aux",
            static_projection_dim=args.audio_primary_static_projection_dim,
            static_dropout=args.audio_primary_static_dropout,
            static_gate_init=args.audio_primary_static_gate_init,
            static_z_clip=0.0,
            static_max_weight=args.audio_primary_static_max_weight,
            static_anomaly_threshold=args.audio_primary_static_anomaly_threshold,
            static_anomaly_scale=args.audio_primary_static_anomaly_scale,
            static_aux_hidden_dim=args.audio_primary_static_aux_hidden_dim,
            dry_run=dry_run,
        )
    else:
        raise ValueError(f"Unsupported SplitAST-MAE model key: {model_key}")
    if args.skip_eval:
        return []
    return _eval_split_ast_pair(args, run_dir, model_key=model_key, seed=seed, controlled=controlled, dry_run=dry_run)


def prepare_stable_static_table(args: argparse.Namespace, dry_run: bool) -> Optional[Path]:
    stable_models = {"split_ast_stable_static_gated_clip", "split_ast_audio_primary_stable_static"}
    if not any(model in stable_models for model in args.models):
        return None
    if args.stable_static_table is not None:
        table = Path(args.stable_static_table)
        if not dry_run and not table.is_file():
            raise FileNotFoundError(
                f"--stable-static-table does not exist: {table}. "
                "Pass the generated stable_static_by_patient_vowel.csv, or omit this option to regenerate it."
            )
        return table
    if not dry_run and not Path(args.static_feature_table).is_file():
        raise FileNotFoundError(
            f"Static feature table not found: {args.static_feature_table}. "
            "Set --static-feature-table to the 131D table generated by "
            "extract_split_ast_131_static_features.py, or set --stable-static-table to an existing filtered table."
        )
    out_dir = args.output_dir / "static_features"
    table = out_dir / "stable_static_by_patient_vowel.csv"
    report = out_dir / "stable_static_selection.json"
    _mkdir(out_dir, dry_run=dry_run)
    _run(
        _cmd(
            "split_ast_make_stable_static_table.py",
            *_data_args(args),
            "--dev-test-seed",
            args.dev_test_seed,
            "--train-val-seed",
            args.train_val_seed,
            "--static-feature-source",
            "table",
            "--static-feature-table",
            args.static_feature_table,
            "--model-size",
            args.ast_model_size,
            "--input-fdim",
            args.input_fdim,
            "--input-tdim",
            args.input_tdim,
            "--n-client-blocks",
            args.n_client_blocks,
            "--out-table",
            table,
            "--out-json",
            report,
            "--quiet",
        ),
        dry_run=dry_run,
    )
    return table


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _fmt_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def _dataset_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("evaluation"), dict):
        return payload["evaluation"]
    if isinstance(payload.get("datasets"), dict):
        return payload["datasets"]
    return payload


def _combined_metric(combined: dict[str, Any], key: str) -> Optional[float]:
    metrics = combined.get("metrics") or {}
    value = metrics.get(key)
    if value is None and key == "f1_score":
        value = metrics.get("f1")
    if value is None:
        value = combined.get(key)
    return _float_or_none(value)


def _combined_auc(combined: dict[str, Any]) -> Optional[float]:
    return _float_or_none(combined.get("auc")) or _float_or_none(combined.get("roc_auc"))


def _combined_cm(combined: dict[str, Any]) -> Optional[list[list[int]]]:
    cm = combined.get("confusion_matrix") or (combined.get("metrics") or {}).get("confusion_matrix")
    if not cm:
        return None
    try:
        return [[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]]
    except Exception:
        return None


def _extract_result_rows(records: Iterable[EvalRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_labels = {"chinese": "EENT", "german": "SVD"}
    for record in records:
        if not record.path.exists():
            continue
        try:
            payload = json.loads(record.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip] cannot parse {record.path}: {exc}")
            continue
        if not isinstance(payload, dict):
            print(f"[skip] unsupported JSON top-level type in {record.path}: {type(payload).__name__}")
            continue
        blocks = _dataset_blocks(payload)
        for dataset_key, dataset_label in dataset_labels.items():
            block = blocks.get(dataset_key)
            if not isinstance(block, dict):
                continue
            combined = block.get("combined")
            if not isinstance(combined, dict):
                continue
            rows.append(
                {
                    "seed": record.seed,
                    "model": record.model,
                    "protocol": record.protocol,
                    "dataset": dataset_label,
                    "accuracy": _combined_metric(combined, "accuracy"),
                    "precision": _combined_metric(combined, "precision"),
                    "recall": _combined_metric(combined, "recall"),
                    "specificity": _combined_metric(combined, "specificity"),
                    "f1": _combined_metric(combined, "f1_score"),
                    "auc": _combined_auc(combined),
                    "cm": _combined_cm(combined),
                    "source_file": str(record.path),
                }
            )
    return rows


def write_manifest(records: list[EvalRecord], out_dir: Path) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": [
            {"seed": r.seed, "model": r.model, "protocol": r.protocol, "path": str(r.path)} for r in records
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {path.resolve()}")


def write_seed_rows(rows: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "seed_results.tsv"
    fields = [
        "seed",
        "model",
        "protocol",
        "dataset",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc",
        "cm",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["cm"] = json.dumps(out.get("cm"), ensure_ascii=False) if out.get("cm") is not None else ""
            writer.writerow(out)
    print(f"Wrote per-seed TSV: {path.resolve()}")


def aggregate_rows(rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["protocol"]), str(row["dataset"]))
        groups.setdefault(key, []).append(row)

    aggregate: list[dict[str, Any]] = []
    for (model, protocol, dataset), group_rows in sorted(groups.items()):
        item: dict[str, Any] = {
            "model": model,
            "protocol": protocol,
            "dataset": dataset,
            "n_seeds": len({int(r["seed"]) for r in group_rows}),
        }
        for metric in metrics:
            values = [v for v in (_float_or_none(r.get(metric)) for r in group_rows) if v is not None]
            mean, std = _mean_std(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
            item[f"{metric}_mean_std"] = _fmt_mean_std(mean, std)
        aggregate.append(item)

    tsv_path = out_dir / "summary_mean_std.tsv"
    fields = ["model", "protocol", "dataset", "n_seeds"]
    for metric in metrics:
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregate)

    json_path = out_dir / "summary_mean_std.json"
    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    html_path = out_dir / "summary_mean_std.html"
    html_path.write_text(_summary_html(aggregate), encoding="utf-8")
    print(f"Wrote aggregate TSV: {tsv_path.resolve()}")
    print(f"Wrote aggregate JSON: {json_path.resolve()}")
    print(f"Wrote aggregate HTML: {html_path.resolve()}")
    return aggregate


def _summary_html(rows: list[dict[str, Any]]) -> str:
    def esc(value: Any) -> str:
        import html

        return html.escape("" if value is None else str(value))

    metric_cols = ["accuracy", "f1", "auc", "specificity", "recall", "precision"]
    body = []
    for row in rows:
        cells = [
            esc(row["model"]),
            esc(row["protocol"]),
            esc(row["dataset"]),
            esc(row["n_seeds"]),
        ]
        cells.extend(esc(row.get(f"{m}_mean_std", "")) for m in metric_cols)
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    headers = ["model", "protocol", "dataset", "n", *metric_cols]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Final pipeline seed-averaged results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
h1 {{ font-size: 22px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d8dee9; padding: 6px 8px; text-align: left; }}
th {{ background: #f1f4f8; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #fbfcfe; }}
.note {{ color: #4b5563; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>Final Pipeline Seed-Averaged Results</h1>
<div class="note">Values are mean +/- sample std across model-init seeds. Fixed 0.5 is deployment-like; Youden is diagnostic only.</div>
<table>
<thead><tr>{''.join(f'<th>{esc(h)}</th>' for h in headers)}</tr></thead>
<tbody>
{''.join(body)}
</tbody>
</table>
</body>
</html>
"""


def write_domain_shift_report(args: argparse.Namespace, stable_table: Optional[Path], dry_run: bool) -> None:
    if not args.include_domain_shift_report:
        return
    run_dirs: list[Path] = []
    first_seed = args.seeds[0]
    stable_run = args.output_dir / f"seed_{first_seed}" / "split_ast_stable_static_gated_clip"
    if stable_run.exists() or dry_run:
        run_dirs.append(stable_run)
    cmd = _cmd(
        "split_ast_make_domain_shift_report.py",
        *_data_args(args),
        "--static-feature-source",
        "table",
        "--static-feature-table",
        args.static_feature_table,
        "--presets",
        "all",
        "pathology",
        "pathology_source_tilt",
        "pathology_voicing",
        "--out-html",
        args.output_dir / "domain_shift_report.html",
        "--out-json",
        args.output_dir / "domain_shift_report.json",
        "--out-static-tsv",
        args.output_dir / "static_domain_shift.tsv",
        "--out-score-tsv",
        args.output_dir / "patient_score_shift.tsv",
        "--quiet",
    )
    if run_dirs:
        cmd.append("--run-dirs")
        cmd.extend(str(p) for p in run_dirs)
    else:
        cmd.append("--static-only")
    _run(cmd, dry_run=dry_run)


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(x) for x in args.seeds]
    return [int(args.seed_start) + idx for idx in range(int(args.n_seeds))]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the publication-facing main pipeline from centralized CNN through split baselines "
            "and selected SplitAST-MAE variants, then average results across seeds."
        )
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/final_pipeline"))
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=MODEL_CHOICES)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=2718)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")

    p.add_argument("--pickle-dir", type=Path, default=None)
    p.add_argument("--pickle-dir-eent", type=Path, default=None)
    p.add_argument("--pickle-dir-svd", type=Path, default=None)
    p.add_argument("--eent-subjects-xlsx", type=Path, default=None)
    p.add_argument("--german-subjects-xlsx", type=Path, default=None)
    p.add_argument("--dev-test-seed", type=int, default=8)
    p.add_argument("--train-val-seed", type=int, default=100)
    p.add_argument("--partition-seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--centralized-max-epochs", type=int, default=1000)

    p.add_argument("--split-global-epochs", type=int, default=250)
    p.add_argument("--split-local-epochs", type=int, default=5)
    p.add_argument("--split-batch-size", type=int, default=256)
    p.add_argument("--split-early-stopping-patience", type=int, default=10)
    p.add_argument("--split-client-lr", type=float, default=5e-5)
    p.add_argument("--split-server-lr", type=float, default=5e-5)
    p.add_argument("--ssast-model-size", type=str, default="base")
    p.add_argument("--ssast-f-shape", type=int, default=16)
    p.add_argument("--ssast-t-shape", type=int, default=16)
    p.add_argument("--ssast-f-stride", type=int, default=10)
    p.add_argument("--ssast-t-stride", type=int, default=10)

    p.add_argument("--stage1-global-rounds", type=int, default=120)
    p.add_argument("--stage1-local-epochs", type=int, default=5)
    p.add_argument("--stage2-global-rounds", type=int, default=250)
    p.add_argument("--stage2-local-epochs", type=int, default=5)
    p.add_argument("--ast-batch-size", type=int, default=64)
    p.add_argument("--ast-client-lr", type=float, default=1.5e-4)
    p.add_argument("--ast-server-lr", type=float, default=1.5e-4)
    p.add_argument("--ast-weight-decay", type=float, default=0.05)
    p.add_argument("--ast-adamw-beta1", type=float, default=0.9)
    p.add_argument("--ast-adamw-beta2", type=float, default=0.95)
    p.add_argument("--ast-focal-gamma", type=float, default=2.0)
    p.add_argument("--ast-early-stopping-patience", type=int, default=10)
    p.add_argument("--ast-model-size", type=str, default="base384")
    p.add_argument("--input-fdim", type=int, default=128)
    p.add_argument("--input-tdim", type=int, default=259)
    p.add_argument("--n-client-blocks", type=int, default=2)
    p.add_argument("--n-partitions", type=int, default=5)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--mask-strategy", choices=["content", "random"], default="content")
    p.add_argument("--pooling", choices=["cls", "mean_patch"], default="cls")
    p.add_argument("--imagenet-pretrain", action="store_true")
    p.add_argument("--audioset-checkpoint-path", type=Path, default=None)
    p.add_argument("--static-feature-table", type=Path, default=DEFAULT_STATIC_TABLE)
    p.add_argument("--stable-static-table", type=Path, default=None)
    p.add_argument("--control-static-projection-dim", type=int, default=32)
    p.add_argument("--control-static-dropout", type=float, default=0.30)
    p.add_argument("--control-static-gate-init", type=float, default=0.25)
    p.add_argument("--control-static-z-clip", type=float, default=3.0)
    p.add_argument("--audio-primary-static-projection-dim", type=int, default=32)
    p.add_argument("--audio-primary-static-dropout", type=float, default=0.35)
    p.add_argument("--audio-primary-static-gate-init", type=float, default=0.20)
    p.add_argument("--audio-primary-static-max-weight", type=float, default=0.35)
    p.add_argument("--audio-primary-static-anomaly-threshold", type=float, default=2.5)
    p.add_argument("--audio-primary-static-anomaly-scale", type=float, default=1.0)
    p.add_argument("--audio-primary-static-aux-hidden-dim", type=int, default=64)
    p.add_argument("--include-youden", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-eent-val-threshold", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-domain-shift-report", action=argparse.BooleanOptionalAction, default=True)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.static_feature_table = Path(args.static_feature_table)
    args.seeds = parse_seeds(args)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Final pipeline")
    print(f"Output dir: {args.output_dir}")
    print(f"Seeds: {', '.join(str(s) for s in args.seeds)}")
    print(f"Models: {', '.join(args.models)}")

    stable_table = None
    if not args.skip_train:
        stable_table = prepare_stable_static_table(args, args.dry_run)
    elif "split_ast_stable_static_gated_clip" in args.models:
        stable_table = args.output_dir / "static_features" / "stable_static_by_patient_vowel.csv"

    records: list[EvalRecord] = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        seed_dir = args.output_dir / f"seed_{seed}"
        _mkdir(seed_dir, dry_run=args.dry_run)
        split_ast_models = [m for m in args.models if m.startswith("split_ast_")]
        stage1_dir = seed_dir / "split_ast_stage1"
        if split_ast_models and not args.skip_train:
            train_stage1(args, seed, stage1_dir, args.dry_run)

        if "centralized_cnn" in args.models and not args.skip_train:
            records.extend(train_eval_centralized_cnn(args, seed, seed_dir, args.dry_run))
        elif "centralized_cnn" in args.models and not args.skip_eval:
            records.append(EvalRecord(seed, "centralized_cnn", "fixed_0.5", seed_dir / "eval_centralized_cnn_fixed.json"))

        if "split_cnn" in args.models and not args.skip_train:
            records.extend(train_eval_split(args, seed, seed_dir, "cnn", args.dry_run))
        elif "split_cnn" in args.models and not args.skip_eval:
            records.append(EvalRecord(seed, "split_cnn", "fixed_0.5", seed_dir / "eval_split_cnn_fixed.json"))

        if "split_ssast" in args.models and not args.skip_train:
            records.extend(train_eval_split(args, seed, seed_dir, "ssast", args.dry_run))
        elif "split_ssast" in args.models and not args.skip_eval:
            records.append(EvalRecord(seed, "split_ssast", "fixed_0.5", seed_dir / "eval_split_ssast_fixed.json"))

        for model_key in split_ast_models:
            if args.skip_train:
                run_dir = seed_dir / model_key
                records.append(EvalRecord(seed, model_key, "fixed_0.5", run_dir / "ast_stage2_ai_eval_fixed.json"))
                if args.include_youden:
                    records.append(EvalRecord(seed, model_key, "youden_eval_set", run_dir / "ast_stage2_ai_eval_youden.json"))
                if args.include_eent_val_threshold:
                    records.append(
                        EvalRecord(
                            seed,
                            model_key,
                            "eent_val_threshold_macro_f1",
                            run_dir / "ast_stage2_ai_eval_eent_val_threshold_macro_f1.json",
                        )
                    )
                continue
            records.extend(
                train_eval_split_ast_variant(args, seed, seed_dir, model_key, stage1_dir, stable_table, args.dry_run)
            )

    if args.dry_run:
        print("\nDry run complete; no files were written.")
        return
    write_manifest(records, args.output_dir)
    rows = _extract_result_rows(records)
    write_seed_rows(rows, args.output_dir)
    aggregate_rows(rows, args.output_dir)
    write_domain_shift_report(args, stable_table, args.dry_run)


if __name__ == "__main__":
    main()
