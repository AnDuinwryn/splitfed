from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional


PAPER_ADAMW_BETAS = (0.9, 0.95)
PAPER_WEIGHT_DECAY = 0.05
PAPER_FOCAL_GAMMA = 2.0
PAPER_STAGE1_REFERENCE_EPOCHS = 120
PROJECT_STAGE1_MAX_ROUNDS = 120
PROJECT_STAGE2_MAX_ROUNDS = 250
PAPER_STAGE2_EARLY_STOPPING_PATIENCE = 10


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {value}")
    return ivalue


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pickle-dir", type=Path, default=None)
    p.add_argument("--pickle-dir-eent", type=Path, default=None)
    p.add_argument("--pickle-dir-svd", type=Path, default=None)
    p.add_argument("--eent-subjects-xlsx", type=Path, default=None)
    p.add_argument("--german-subjects-xlsx", type=Path, default=None)
    p.add_argument("--vowel", choices=["a", "i"], default="a")
    p.add_argument("--dev-test-seed", type=int, default=8)
    p.add_argument("--train-val-seed", type=int, default=100)
    p.add_argument("--partition-seed", type=int, default=42)
    p.add_argument("--n-partitions", type=_positive_int, default=5)
    p.add_argument("--batch-size", type=_positive_int, default=64)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-fdim", type=_positive_int, default=128)
    p.add_argument("--input-tdim", type=_positive_int, default=259)
    p.add_argument("--model-size", type=str, default="base384", help="tiny | small | base384 | base224")
    p.add_argument("--n-client-blocks", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--mask-strategy", choices=["content", "random"], default="content")
    p.add_argument("--num-labels", type=_positive_int, default=1)
    p.add_argument("--static-feature-dim", type=int, default=0, help="Usually inferred when static features are on.")
    p.add_argument("--static-feature-source", choices=["none", "auto", "mel", "opensmile", "parselmouth", "table"], default="none")
    p.add_argument("--static-feature-table", type=Path, default=None)
    p.add_argument(
        "--static-feature-preset",
        choices=[
            "all",
            "pathology",
            "pathology_voice_quality",
            "pathology_source_tilt",
            "pathology_plus_source_tilt",
            "pathology_voicing",
            "pathology_plus_voicing",
        ],
        default="all",
        help=(
            "Column preset for static features. pathology keeps HNR/jitter/shimmer/CPP-like columns; "
            "pathology_source_tilt adds H1-H2/H1-A3; pathology_voicing also adds voicing stability."
        ),
    )
    p.add_argument("--static-audio-manifest", type=Path, default=None)
    p.add_argument("--static-audio-root-eent", type=Path, default=None)
    p.add_argument("--static-audio-root-svd", type=Path, default=None)
    p.add_argument("--pooling", choices=["cls", "mean_patch"], default="cls")
    p.add_argument("--imagenet-pretrain", action="store_true")
    p.add_argument("--audioset-checkpoint-path", type=str, default=None)
    p.add_argument("--model-init-seed", type=int, default=2718)
    p.add_argument("--device", type=str, default=None)


def _add_train_args(
    p: argparse.ArgumentParser,
    *,
    default_rounds: int,
    default_local_epochs: int = 1,
    include_early_stopping: bool = False,
    include_focal: bool = False,
) -> None:
    p.add_argument("--n-global-rounds", type=_positive_int, default=int(default_rounds))
    p.add_argument("--n-local-epochs", type=_positive_int, default=int(default_local_epochs))
    p.add_argument("--client-lr", type=float, default=1.5e-4)
    p.add_argument("--server-lr", type=float, default=1.5e-4)
    p.add_argument("--adamw-beta1", type=float, default=PAPER_ADAMW_BETAS[0])
    p.add_argument("--adamw-beta2", type=float, default=PAPER_ADAMW_BETAS[1])
    p.add_argument("--weight-decay", type=float, default=PAPER_WEIGHT_DECAY)
    if include_focal:
        p.add_argument("--focal-gamma", type=float, default=PAPER_FOCAL_GAMMA)
        p.add_argument("--focal-alpha", type=float, default=None)
    if include_early_stopping:
        p.add_argument("--early-stopping-patience", type=int, default=PAPER_STAGE2_EARLY_STOPPING_PATIENCE)
    p.add_argument("--save-dir", type=Path, default=Path("paper2601_splitmae_runs"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--load-client", type=Path, default=None)
    p.add_argument("--load-server", type=Path, default=None)


def _add_pair_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--metadata-a", type=Path, required=True, help="Stage 2 metadata JSON for vowel /a/.")
    p.add_argument("--metadata-i", type=Path, required=True, help="Stage 2 metadata JSON for vowel /i/.")
    p.add_argument("--eval-dataset", choices=["chinese", "german", "both"], default="both")
    p.add_argument(
        "--patient-eval-strategy",
        choices=["fixed", "best_threshold", "relative", "percentage", "max recall", "guding"],
        default="fixed",
    )
    p.add_argument("--patient-prob-threshold", type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=PAPER_FOCAL_GAMMA)
    p.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Write the full evaluation JSON. Default: metadata-a directory / ast_stage2_ai_eval.json.",
    )
    p.add_argument("--verbose", action="store_true")


def _load_torch():
    import torch

    return torch


def _set_run_seed(seed: int) -> None:
    from voice_disorder_torch.reproducibility import set_reproducible

    set_reproducible(int(seed))
    torch = _load_torch()
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def _live_block(height: int):
    from voice_disorder_torch.ui.live import LiveBlock, supports_ansi

    return LiveBlock(height=height, stream=None) if supports_ansi() else None


def _json_ready(value):
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value


def _make_context(args: argparse.Namespace) -> Any:
    from voice_disorder_torch.config import (
        DataPaths,
        RunContext,
        SplitSeeds,
        TrainConfig,
        apply_default_project_paths,
    )

    apply_default_project_paths(args)
    paths = DataPaths(
        pickle_dir=args.pickle_dir,
        pickle_dir_chinese=args.pickle_dir_eent,
        pickle_dir_german=args.pickle_dir_svd,
        german_subjects_xlsx=args.german_subjects_xlsx,
        eent_subjects_xlsx=args.eent_subjects_xlsx,
    )
    train_cfg = replace(TrainConfig(), batch_size=int(args.batch_size))
    return RunContext(
        paths=paths,
        splits=SplitSeeds(dev_test_seed=int(args.dev_test_seed), train_val_seed=int(args.train_val_seed)),
        train=train_cfg,
        save_dir=getattr(args, "save_dir", Path("paper2601_splitmae_runs")),
        device=args.device,
    )


def _static_config_from_args(args: argparse.Namespace):
    from paper2601_static_features import StaticFeatureConfig

    return StaticFeatureConfig(
        source=str(getattr(args, "static_feature_source", "none")),
        feature_table=getattr(args, "static_feature_table", None),
        feature_preset=str(getattr(args, "static_feature_preset", "all")),
        audio_manifest=getattr(args, "static_audio_manifest", None),
        audio_root_eent=getattr(args, "static_audio_root_eent", None),
        audio_root_svd=getattr(args, "static_audio_root_svd", None),
    )


def _build_loaders(args: argparse.Namespace, *, include_static: bool = False):
    from voice_disorder_torch.split_learning.loaders import build_ssast_partition_loaders

    ctx = _make_context(args)
    if include_static and str(getattr(args, "static_feature_source", "none")).lower().strip() != "none":
        from paper2601_static_features import build_static_ssast_partition_loaders

        pack, info = build_static_ssast_partition_loaders(
            ctx=ctx,
            vowel=args.vowel,
            n_partitions=int(args.n_partitions),
            partition_seed=int(args.partition_seed),
            input_tdim=int(args.input_tdim),
            batch_size=int(args.batch_size),
            config=_static_config_from_args(args),
        )
        args.static_feature_dim = int(info.dim)
        args.static_feature_backend = info.backend
        args.static_feature_preset = info.preset
        args.static_feature_names = info.names
        args.static_feature_mean = info.mean
        args.static_feature_std = info.std
        return pack

    args.static_feature_dim = 0
    args.static_feature_backend = "none"
    args.static_feature_preset = "all"
    args.static_feature_names = []
    args.static_feature_mean = []
    args.static_feature_std = []
    return build_ssast_partition_loaders(
        ctx=ctx,
        vowel=args.vowel,
        n_partitions=int(args.n_partitions),
        partition_seed=int(args.partition_seed),
        input_tdim=int(args.input_tdim),
        batch_size=int(args.batch_size),
    )


def _device(args: argparse.Namespace):
    torch = _load_torch()
    return torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_matching_state_dict(module, path: Path, *, label: str) -> None:
    torch = _load_torch()
    incoming = torch.load(path, map_location="cpu")
    current = module.state_dict()
    classifier_mismatch = any(
        key.startswith("classifier.")
        and key in current
        and tuple(current[key].shape) != tuple(value.shape)
        for key, value in incoming.items()
    )
    matched = {}
    skipped = []
    for key, value in incoming.items():
        if classifier_mismatch and key.startswith("classifier."):
            skipped.append(key)
            continue
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    missing, unexpected = module.load_state_dict(matched, strict=False)
    if skipped:
        print(f"Loaded {label}: {len(matched)} tensors; skipped {len(skipped)} shape/key mismatches.")
    elif missing or unexpected:
        print(f"Loaded {label}: {len(matched)} tensors; missing={len(missing)} unexpected={len(unexpected)}.")


def _build_pair(args: argparse.Namespace):
    torch = _load_torch()
    from paper2601_splitmae_client import Paper2601SplitMAEClient, SplitMAEClientConfig
    from paper2601_splitmae_server import Paper2601SplitMAEServer, SplitMAEServerConfig

    device = _device(args)
    torch.manual_seed(int(args.model_init_seed))
    client = Paper2601SplitMAEClient(
        SplitMAEClientConfig(
            input_fdim=int(args.input_fdim),
            input_tdim=int(args.input_tdim),
            model_size=str(args.model_size),
            n_client_blocks=int(args.n_client_blocks),
            mask_ratio=float(args.mask_ratio),
            mask_strategy=str(args.mask_strategy),
            imagenet_pretrain=bool(args.imagenet_pretrain),
            audioset_checkpoint_path=args.audioset_checkpoint_path,
        )
    )
    torch.manual_seed(int(args.model_init_seed))
    server = Paper2601SplitMAEServer(
        SplitMAEServerConfig(
            input_fdim=int(args.input_fdim),
            input_tdim=int(args.input_tdim),
            model_size=str(args.model_size),
            n_client_blocks=int(args.n_client_blocks),
            num_labels=int(args.num_labels),
            static_feature_dim=int(args.static_feature_dim),
            pooling=str(args.pooling),
            imagenet_pretrain=bool(args.imagenet_pretrain),
            audioset_checkpoint_path=args.audioset_checkpoint_path,
        )
    )
    load_client = getattr(args, "load_client", None)
    load_server = getattr(args, "load_server", None)
    if load_client is not None:
        _load_matching_state_dict(client, load_client, label="client")
    if load_server is not None:
        _load_matching_state_dict(server, load_server, label="server")
    return client.to(device), server.to(device), device


def _save_pair(args: argparse.Namespace, client, server, stage: str) -> dict[str, str]:
    torch = _load_torch()
    run_name = args.run_name or f"{stage}_{args.vowel}_seed{args.model_init_seed}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    client_path = args.save_dir / f"{run_name}_client.pt"
    server_path = args.save_dir / f"{run_name}_server.pt"
    meta_path = args.save_dir / f"{run_name}_metadata.json"
    torch.save(client.state_dict(), client_path)
    torch.save(server.state_dict(), server_path)
    metadata = {
        "stage": stage,
        "vowel": args.vowel,
        "dev_test_seed": int(args.dev_test_seed),
        "train_val_seed": int(args.train_val_seed),
        "partition_seed": int(args.partition_seed),
        "model_init_seed": int(args.model_init_seed),
        "model_size": args.model_size,
        "input_fdim": int(args.input_fdim),
        "input_tdim": int(args.input_tdim),
        "n_client_blocks": int(args.n_client_blocks),
        "batch_size": int(args.batch_size),
        "num_labels": int(args.num_labels),
        "static_feature_dim": int(args.static_feature_dim),
        "static_feature_source": str(getattr(args, "static_feature_source", "none")),
        "static_feature_backend": str(getattr(args, "static_feature_backend", "none")),
        "static_feature_preset": str(getattr(args, "static_feature_preset", "all")),
        "static_feature_table": (
            str(args.static_feature_table) if getattr(args, "static_feature_table", None) is not None else None
        ),
        "static_feature_names": list(getattr(args, "static_feature_names", [])),
        "static_feature_mean": list(getattr(args, "static_feature_mean", [])),
        "static_feature_std": list(getattr(args, "static_feature_std", [])),
        "static_audio_manifest": (
            str(args.static_audio_manifest) if getattr(args, "static_audio_manifest", None) is not None else None
        ),
        "static_audio_root_eent": (
            str(args.static_audio_root_eent) if getattr(args, "static_audio_root_eent", None) is not None else None
        ),
        "static_audio_root_svd": (
            str(args.static_audio_root_svd) if getattr(args, "static_audio_root_svd", None) is not None else None
        ),
        "mask_ratio": float(args.mask_ratio),
        "mask_strategy": str(args.mask_strategy),
        "optimizer": "AdamW",
        "adamw_betas": [float(args.adamw_beta1), float(args.adamw_beta2)],
        "weight_decay": float(args.weight_decay),
        "client_lr": float(args.client_lr),
        "server_lr": float(args.server_lr),
        "n_global_rounds": int(args.n_global_rounds),
        "n_local_epochs": int(args.n_local_epochs),
        "stage1_reference_epochs": PAPER_STAGE1_REFERENCE_EPOCHS if stage == "stage1" else None,
        "stage2_loss": "BinaryFocalWithLogitsLoss" if stage == "stage2" else None,
        "stage2_focal_gamma": getattr(args, "focal_gamma", None),
        "stage2_focal_alpha": getattr(args, "focal_alpha", None),
        "stage2_primary_metric": "macro_f1" if stage == "stage2" else None,
        "stage2_early_stopping_patience": getattr(args, "early_stopping_patience", None),
        "client_file": str(client_path),
        "server_file": str(server_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"client": str(client_path), "server": str(server_path), "metadata": str(meta_path)}


def _print_saved_pair_summary(saved: dict[str, str], *, best_round: int | None = None, best_score: float | None = None) -> None:
    print(f"[done] saved client: {saved['client']}")
    print(f"[done] saved server: {saved['server']}")
    print(f"[done] saved metadata: {saved['metadata']}")
    if best_round is not None and best_score is not None:
        print(f"[done] best_round: {best_round}  best_val_macro_f1: {best_score:.4f}")


def _apply_metadata_to_args(args: argparse.Namespace) -> dict:
    if args.metadata is None:
        return {}
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    mapping = {
        "vowel": "vowel",
        "dev_test_seed": "dev_test_seed",
        "train_val_seed": "train_val_seed",
        "partition_seed": "partition_seed",
        "model_init_seed": "model_init_seed",
        "model_size": "model_size",
        "input_fdim": "input_fdim",
        "input_tdim": "input_tdim",
        "n_client_blocks": "n_client_blocks",
        "batch_size": "batch_size",
        "num_labels": "num_labels",
        "static_feature_dim": "static_feature_dim",
        "static_feature_source": "static_feature_source",
        "static_feature_preset": "static_feature_preset",
        "mask_ratio": "mask_ratio",
        "mask_strategy": "mask_strategy",
        "stage2_focal_gamma": "focal_gamma",
    }
    for meta_key, arg_name in mapping.items():
        if meta.get(meta_key) is not None and hasattr(args, arg_name):
            setattr(args, arg_name, meta[meta_key])
    backend = meta.get("static_feature_backend")
    if backend in {"none", "mel", "opensmile", "parselmouth", "table"} and hasattr(args, "static_feature_source"):
        args.static_feature_source = backend
    if backend in {"opensmile_parselmouth_131", "opensmile+parselmouth", "opensmile_parselmouth"} and hasattr(
        args, "static_feature_source"
    ):
        args.static_feature_source = "table"
    for meta_key, arg_name in (
        ("static_feature_table", "static_feature_table"),
        ("static_audio_manifest", "static_audio_manifest"),
        ("static_audio_root_eent", "static_audio_root_eent"),
        ("static_audio_root_svd", "static_audio_root_svd"),
    ):
        if meta.get(meta_key) is not None and hasattr(args, arg_name) and getattr(args, arg_name) is None:
            setattr(args, arg_name, Path(meta[meta_key]))
    if args.load_client is None and meta.get("client_file"):
        args.load_client = Path(meta["client_file"])
    if args.load_server is None and meta.get("server_file"):
        args.load_server = Path(meta["server_file"])
    return meta


def _args_from_metadata(base_args: argparse.Namespace, metadata_path: Path) -> tuple[argparse.Namespace, dict]:
    args = copy.copy(base_args)
    args.metadata = metadata_path
    args.load_client = None
    args.load_server = None
    meta = _apply_metadata_to_args(args)
    return args, meta


def _eval_arrays_for_dataset(bundle, vowel: str, dataset_name: str):
    if dataset_name == "chinese":
        if vowel == "a":
            return bundle.x_test_a, bundle.y_test_a, bundle.id_test_a, "Chinese-Test"
        return bundle.x_test_i, bundle.y_test_i, bundle.id_test_i, "Chinese-Test"
    if dataset_name == "german":
        if vowel == "a":
            return bundle.x_ger_a, bundle.y_ger_a, bundle.id_ger_a, "German-Test"
        return bundle.x_ger_i, bundle.y_ger_i, bundle.id_ger_i, "German-Test"
    raise ValueError(f"Unknown eval dataset: {dataset_name}")

def cmd_inspect(args: argparse.Namespace) -> None:
    _set_run_seed(int(args.model_init_seed))
    torch = _load_torch()
    client, server, device = _build_pair(args)
    x = torch.randn(2, int(args.input_tdim), int(args.input_fdim), device=device)
    static = None
    if int(args.static_feature_dim) > 0:
        static = torch.randn(2, int(args.static_feature_dim), device=device)
    with torch.no_grad():
        pre = client(x, mode="pretrain")
        pre_out = server(pre.to(device))
        ft = client(x, mode="finetune", static_features=static)
        ft_out = server(ft.to(device), return_attention=static is not None)
    print(
        json.dumps(
            {
                "device": str(device),
                "pretrain_tokens": list(pre.tokens.shape),
                "pretrain_mask": list(pre.mask.shape),
                "pretrain_pred_patches": list(pre_out["pred_patches"].shape),
                "finetune_tokens": list(ft.tokens.shape),
                "finetune_logits": list(ft_out["logits"].shape),
            },
            indent=2,
        )
    )


def cmd_smoke(args: argparse.Namespace) -> None:
    _set_run_seed(7)
    import paper2601_splitmae_smoke

    paper2601_splitmae_smoke.main()


def cmd_train_stage1(args: argparse.Namespace) -> None:
    from paper2601_splitmae_training import Paper2601SplitServerPool, run_stage1_splitfed_round

    _set_run_seed(int(args.model_init_seed))
    pack = _build_loaders(args, include_static=False)
    client, server, device = _build_pair(args)
    optimizer_betas = (float(args.adamw_beta1), float(args.adamw_beta2))
    server_pool = Paper2601SplitServerPool(
        server_template=server,
        n_partitions=int(args.n_partitions),
        server_lr=float(args.server_lr),
        device=device,
        optimizer_betas=optimizer_betas,
        weight_decay=float(args.weight_decay),
    )
    block = _live_block(3 + int(args.n_partitions))
    last_summary = "ma_error: _"
    for round_idx in range(int(args.n_global_rounds)):
        header = f"global_epoch: {round_idx}/{int(args.n_global_rounds)}"
        summary = last_summary
        part_lines = [f"p{idx}: ma_error: _" for idx in range(int(args.n_partitions))]
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)

        def on_partition(partition_id, part_stats):
            part_lines[int(partition_id)] = f"p{partition_id}: ma_error: {part_stats.loss:.4f}"
            if block is not None:
                block.redraw([header, summary, "participants:"] + part_lines)

        stats = run_stage1_splitfed_round(
            client_base=client,
            server_pool=server_pool,
            train_loaders=pack.train_loaders,
            n_local_epochs=int(args.n_local_epochs),
            client_lr=float(args.client_lr),
            device=device,
            optimizer_betas=optimizer_betas,
            weight_decay=float(args.weight_decay),
            progress_fn=on_partition,
        )
        mean_ma_error = sum(s.loss for s in stats) / max(len(stats), 1)
        summary = f"ma_error: {mean_ma_error:.4f}"
        last_summary = summary
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)
        else:
            print(f"{header}  {summary}")
    saved = _save_pair(args, client, server_pool.global_model(), "stage1")
    _print_saved_pair_summary(saved)


def cmd_train_stage2(args: argparse.Namespace) -> None:
    from paper2601_splitmae_training import (
        BinaryFocalWithLogitsLoss,
        Paper2601SplitServerPool,
        evaluate_stage2,
        run_stage2_splitfed_round,
    )

    _set_run_seed(int(args.model_init_seed))
    pack = _build_loaders(args, include_static=True)
    client, server, device = _build_pair(args)
    optimizer_betas = (float(args.adamw_beta1), float(args.adamw_beta2))
    server_pool = Paper2601SplitServerPool(
        server_template=server,
        n_partitions=int(args.n_partitions),
        server_lr=float(args.server_lr),
        device=device,
        finetune_criterion=BinaryFocalWithLogitsLoss(gamma=float(args.focal_gamma), alpha=args.focal_alpha),
        optimizer_betas=optimizer_betas,
        weight_decay=float(args.weight_decay),
    )
    val_criterion = BinaryFocalWithLogitsLoss(gamma=float(args.focal_gamma), alpha=args.focal_alpha)
    patience = int(args.early_stopping_patience)
    patience_left = patience
    best_score = float("-inf")
    best_client_sd = copy.deepcopy(client.state_dict())
    best_server_sd = copy.deepcopy(server_pool.global_model().state_dict())
    best_round = -1
    block = _live_block(3 + int(args.n_partitions))
    last_summary = "val_macro_f1: _  val_loss: _  patience: _"
    for round_idx in range(int(args.n_global_rounds)):
        header = f"global_epoch: {round_idx}/{int(args.n_global_rounds)}"
        summary = last_summary
        part_lines = [f"p{idx}: train_macro_f1: _  train_loss: _" for idx in range(int(args.n_partitions))]
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)

        def on_partition(partition_id, part_stats):
            part_lines[int(partition_id)] = (
                f"p{partition_id}: train_macro_f1: {part_stats.score:.3f}  train_loss: {part_stats.loss:.4f}"
            )
            if block is not None:
                block.redraw([header, summary, "participants:"] + part_lines)

        stats = run_stage2_splitfed_round(
            client_base=client,
            server_pool=server_pool,
            train_loaders=pack.train_loaders,
            n_local_epochs=int(args.n_local_epochs),
            client_lr=float(args.client_lr),
            device=device,
            optimizer_betas=optimizer_betas,
            weight_decay=float(args.weight_decay),
            progress_fn=on_partition,
        )
        val = evaluate_stage2(
            client=client,
            server=server_pool.global_model(),
            loader=pack.val_loader,
            device=device,
            criterion=val_criterion,
        )
        pstr = f"{patience_left}" if patience > 0 else "off"
        summary = f"val_macro_f1: {val.score:.4f}  val_loss: {val.loss:.4f}  patience: {pstr}"
        last_summary = summary
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)
        else:
            print(f"{header}  {summary}")
        if val.score > best_score + 1e-12:
            best_score = float(val.score)
            best_round = int(round_idx)
            best_client_sd = copy.deepcopy(client.state_dict())
            best_server_sd = copy.deepcopy(server_pool.global_model().state_dict())
            if patience > 0:
                patience_left = patience
        elif patience > 0:
            patience_left -= 1
            if patience_left <= 0:
                print(f"! early_stop  best_round: {best_round}  best_val_macro_f1: {best_score:.4f}")
                break
    client.load_state_dict(best_client_sd)
    best_server = server_pool.global_model()
    best_server.load_state_dict(best_server_sd)
    saved = _save_pair(args, client, best_server, "stage2")
    _print_saved_pair_summary(saved, best_round=best_round, best_score=best_score)


def _predict_stage2_positive_probs(client, server, loader, device):
    torch = _load_torch()
    probs = []
    client.eval()
    server.eval()
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            static = batch[2].to(device) if isinstance(batch, (tuple, list)) and len(batch) == 3 else None
            smashed = client(xb, mode="finetune", static_features=static).to(device)
            logits = server.forward_finetune(smashed)["logits"]
            sigmoid = torch.sigmoid(logits)
            if sigmoid.shape[-1] != 1:
                raise ValueError("Patient-level final evaluation currently expects --num-labels 1.")
            probs.append(sigmoid[:, 0].detach().cpu())
    return torch.cat(probs, dim=0).numpy()


def _stage2_eval_loader(x, y, input_tdim: int, batch_size: int, static_features=None):
    from torch.utils.data import DataLoader

    from paper2601_static_features import SsastMelStaticDataset
    from voice_disorder_torch.data.datasets import SsastMelDataset

    if static_features is None:
        ds = SsastMelDataset(x, y, input_tdim=int(input_tdim))
    else:
        ds = SsastMelStaticDataset(x, y, static_features, input_tdim=int(input_tdim))
    return ds, DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)


def _default_pair_results_json(args: argparse.Namespace) -> Path:
    return Path(args.metadata_a).parent / "ast_stage2_ai_eval.json"


def _printable_pair_eval(results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "patient_eval_strategy": results.get("patient_eval_strategy"),
        "patient_prob_threshold": results.get("patient_prob_threshold"),
    }
    for key, value in results.get("datasets", {}).items():
        out[key] = value
    return out


def _eval_static_features(args_for_vowel: argparse.Namespace, meta: dict, x, patient_ids, dataset_name: str, vowel: str):
    dim = int(meta.get("static_feature_dim") or getattr(args_for_vowel, "static_feature_dim", 0) or 0)
    if dim <= 0:
        return None
    from paper2601_static_features import apply_static_normalizer, compute_static_features

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
    mean = meta.get("static_feature_mean") or []
    std = meta.get("static_feature_std") or []
    return apply_static_normalizer(raw, mean, std)


def cmd_evaluate_stage2_pair(args: argparse.Namespace) -> None:
    from paper2601_splitmae_training import BinaryFocalWithLogitsLoss, evaluate_stage2
    from voice_disorder_torch.data.load import load_all_preprocessed
    from voice_disorder_torch.evaluation.patient_eval import combined_vowel_ai_eval, model_eval_by_id
    from voice_disorder_torch.io.eval_report import save_eval_json
    from voice_disorder_torch.ui.eval_cli import print_eval

    args_a, meta_a = _args_from_metadata(args, args.metadata_a)
    args_i, meta_i = _args_from_metadata(args, args.metadata_i)
    _set_run_seed(int(args_a.model_init_seed))
    if args_a.vowel != "a" or args_i.vowel != "i":
        raise SystemExit(
            f"evaluate-stage2-pair expects metadata-a for /a/ and metadata-i for /i/, "
            f"got {args_a.vowel!r} and {args_i.vowel!r}."
        )
    if args_a.dev_test_seed != args_i.dev_test_seed:
        raise SystemExit("metadata-a and metadata-i must use the same dev_test_seed for paired evaluation.")
    if int(args_a.num_labels) != 1 or int(args_i.num_labels) != 1:
        raise SystemExit("evaluate-stage2-pair currently supports binary --num-labels 1 models only.")

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
    results = {
        "stage": "evaluate-stage2-pair",
        "metadata_a": str(args.metadata_a),
        "metadata_i": str(args.metadata_i),
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
        "loaded_metadata_a": meta_a,
        "loaded_metadata_i": meta_i,
        "datasets": {},
    }

    for dataset_name in selected:
        xa, ya, ida, display_name = _eval_arrays_for_dataset(bundle, "a", dataset_name)
        xi, yi, idi, _ = _eval_arrays_for_dataset(bundle, "i", dataset_name)
        static_a = _eval_static_features(args_a, meta_a, xa, ida, dataset_name, "a")
        static_i = _eval_static_features(args_i, meta_i, xi, idi, dataset_name, "i")
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
    results_json = Path(args.results_json) if args.results_json is not None else _default_pair_results_json(args)
    save_eval_json(results_json, results)
    print(f"Wrote evaluation JSON: {results_json.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isolated CLI for the Paper 2601 Split-MAE experiment.")
    sub = p.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Print split payload shapes using synthetic tensors.")
    _add_model_args(inspect_p)
    inspect_p.set_defaults(func=cmd_inspect)

    smoke_p = sub.add_parser("smoke", help="Run the tiny synthetic smoke script.")
    smoke_p.set_defaults(func=cmd_smoke)

    stage1_p = sub.add_parser("train-stage1", help="Run Stage 1 domain-adaptive MAE split training.")
    _add_data_args(stage1_p)
    _add_model_args(stage1_p)
    _add_train_args(stage1_p, default_rounds=PROJECT_STAGE1_MAX_ROUNDS, default_local_epochs=5)
    stage1_p.set_defaults(func=cmd_train_stage1)

    stage2_p = sub.add_parser("train-stage2", help="Run Stage 2 Attention-FFNN split fine-tuning.")
    _add_data_args(stage2_p)
    _add_model_args(stage2_p)
    _add_train_args(
        stage2_p,
        default_rounds=PROJECT_STAGE2_MAX_ROUNDS,
        default_local_epochs=5,
        include_early_stopping=True,
        include_focal=True,
    )
    stage2_p.set_defaults(func=cmd_train_stage2)

    eval_pair_p = sub.add_parser(
        "evaluate-stage2-pair",
        help="Evaluate saved Stage 2 /a/ and /i/ models with combined patient-level scoring.",
    )
    _add_data_args(eval_pair_p)
    _add_model_args(eval_pair_p)
    _add_pair_eval_args(eval_pair_p)
    eval_pair_p.set_defaults(func=cmd_evaluate_stage2_pair)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
