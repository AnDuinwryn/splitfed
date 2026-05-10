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
PAPER_STAGE1_EPOCHS = 120
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
    p.add_argument("--batch-size", type=_positive_int, default=256)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-fdim", type=_positive_int, default=128)
    p.add_argument("--input-tdim", type=_positive_int, default=259)
    p.add_argument("--model-size", type=str, default="base384", help="tiny | small | base384 | base224")
    p.add_argument("--n-client-blocks", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--mask-strategy", choices=["content", "random"], default="content")
    p.add_argument("--num-labels", type=_positive_int, default=1)
    p.add_argument("--static-feature-dim", type=int, default=0)
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


def _load_torch():
    import torch

    return torch


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
        save_dir=args.save_dir,
        device=args.device,
    )


def _build_loaders(args: argparse.Namespace):
    from voice_disorder_torch.split_learning.loaders import build_ssast_partition_loaders

    ctx = _make_context(args)
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
        client.load_state_dict(torch.load(load_client, map_location="cpu"), strict=False)
    if load_server is not None:
        server.load_state_dict(torch.load(load_server, map_location="cpu"), strict=False)
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
        "model_size": args.model_size,
        "input_fdim": int(args.input_fdim),
        "input_tdim": int(args.input_tdim),
        "n_client_blocks": int(args.n_client_blocks),
        "num_labels": int(args.num_labels),
        "static_feature_dim": int(args.static_feature_dim),
        "mask_ratio": float(args.mask_ratio),
        "mask_strategy": str(args.mask_strategy),
        "optimizer": "AdamW",
        "adamw_betas": [float(args.adamw_beta1), float(args.adamw_beta2)],
        "weight_decay": float(args.weight_decay),
        "client_lr": float(args.client_lr),
        "server_lr": float(args.server_lr),
        "n_global_rounds": int(args.n_global_rounds),
        "n_local_epochs": int(args.n_local_epochs),
        "stage1_reference_epochs": PAPER_STAGE1_EPOCHS if stage == "stage1" else None,
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


def cmd_inspect(args: argparse.Namespace) -> None:
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
    import paper2601_splitmae_smoke

    paper2601_splitmae_smoke.main()


def cmd_train_stage1(args: argparse.Namespace) -> None:
    from paper2601_splitmae_training import Paper2601SplitServerPool, run_stage1_splitfed_round

    pack = _build_loaders(args)
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
    history = []
    for round_idx in range(int(args.n_global_rounds)):
        stats = run_stage1_splitfed_round(
            client_base=client,
            server_pool=server_pool,
            train_loaders=pack.train_loaders,
            n_local_epochs=int(args.n_local_epochs),
            client_lr=float(args.client_lr),
            device=device,
            optimizer_betas=optimizer_betas,
            weight_decay=float(args.weight_decay),
        )
        round_stats = [{"ma_error": s.loss} for s in stats]
        history.append(round_stats)
        print(json.dumps({"round": round_idx, "partition_stats": round_stats}))
    saved = _save_pair(args, client, server_pool.global_model(), "stage1")
    print(json.dumps({"saved": saved, "history": history}, indent=2))


def cmd_train_stage2(args: argparse.Namespace) -> None:
    from paper2601_splitmae_training import (
        BinaryFocalWithLogitsLoss,
        Paper2601SplitServerPool,
        evaluate_stage2,
        run_stage2_splitfed_round,
    )

    pack = _build_loaders(args)
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
    history = []
    patience = int(args.early_stopping_patience)
    patience_left = patience
    best_score = float("-inf")
    best_client_sd = copy.deepcopy(client.state_dict())
    best_server_sd = copy.deepcopy(server_pool.global_model().state_dict())
    best_round = -1
    for round_idx in range(int(args.n_global_rounds)):
        stats = run_stage2_splitfed_round(
            client_base=client,
            server_pool=server_pool,
            train_loaders=pack.train_loaders,
            n_local_epochs=int(args.n_local_epochs),
            client_lr=float(args.client_lr),
            device=device,
            optimizer_betas=optimizer_betas,
            weight_decay=float(args.weight_decay),
        )
        val = evaluate_stage2(
            client=client,
            server=server_pool.global_model(),
            loader=pack.val_loader,
            device=device,
            criterion=val_criterion,
        )
        round_stats = [{"loss": s.loss, "macro_f1": s.score} for s in stats]
        history.append({"train": round_stats, "val": {"loss": val.loss, "macro_f1": val.score}})
        print(
            json.dumps(
                {
                    "round": round_idx,
                    "partition_stats": round_stats,
                    "val_loss": val.loss,
                    "val_macro_f1": val.score,
                    "patience_left": patience_left if patience > 0 else "off",
                }
            )
        )
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
                print(json.dumps({"early_stop": True, "best_round": best_round, "best_val_macro_f1": best_score}))
                break
    client.load_state_dict(best_client_sd)
    best_server = server_pool.global_model()
    best_server.load_state_dict(best_server_sd)
    saved = _save_pair(args, client, best_server, "stage2")
    print(json.dumps({"saved": saved, "history": history, "best_round": best_round, "best_val_macro_f1": best_score}, indent=2))


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
    _add_train_args(stage1_p, default_rounds=PAPER_STAGE1_EPOCHS, default_local_epochs=1)
    stage1_p.set_defaults(func=cmd_train_stage1)

    stage2_p = sub.add_parser("train-stage2", help="Run Stage 2 Attention-FFNN split fine-tuning.")
    _add_data_args(stage2_p)
    _add_model_args(stage2_p)
    _add_train_args(
        stage2_p,
        default_rounds=PROJECT_STAGE2_MAX_ROUNDS,
        default_local_epochs=1,
        include_early_stopping=True,
        include_focal=True,
    )
    stage2_p.set_defaults(func=cmd_train_stage2)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
