from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from paper2601_splitmae_cli import (
    PAPER_FOCAL_GAMMA,
    PAPER_STAGE2_EARLY_STOPPING_PATIENCE,
    PROJECT_STAGE2_MAX_ROUNDS,
    _add_data_args,
    _add_model_args,
    _add_pair_eval_args,
    _add_train_args,
    _args_from_metadata,
    _build_loaders,
    _default_pair_results_json,
    _device,
    _eval_arrays_for_dataset,
    _json_ready,
    _load_matching_state_dict,
    _make_context,
    _predict_stage2_positive_probs,
    _print_saved_pair_summary,
    _printable_pair_eval,
    _set_run_seed,
    _stage2_eval_loader,
    _static_config_from_args,
)
from paper2601_splitmae_training import (
    BinaryFocalWithLogitsLoss,
    Paper2601SplitServerPool,
    evaluate_stage2,
    run_stage2_splitfed_round,
)
from paper2601_static_features import (
    StaticFeatureInfo,
    apply_static_normalizer,
    compute_static_features,
)


CONTROLLED_FUSION_VERSION = 1


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return float(math.log(p / (1.0 - p)))


class ControlledFeatureAttentionFFNN(nn.Module):
    """Attention-FFNN head with controlled static-feature fusion.

    The server still receives normal smashed AST tokens. For static-only
    ablations the head ignores audio features but keeps a zero-valued dependency
    on the audio branch so SplitFed gradient plumbing remains valid.
    """

    def __init__(
        self,
        *,
        audio_feature_dim: int,
        static_feature_dim: int,
        num_labels: int,
        hidden: tuple[int, ...],
        dropout: float,
        fusion_mode: str = "audio_static",
        static_projection_dim: int = 0,
        static_dropout: float = 0.0,
        static_gate_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        fusion_mode = str(fusion_mode).lower().strip()
        if fusion_mode not in {"audio_only", "static_only", "audio_static", "gated"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        self.audio_feature_dim = int(audio_feature_dim)
        self.static_feature_dim = int(static_feature_dim)
        self.num_labels = int(num_labels)
        self.fusion_mode = fusion_mode
        self.use_audio = fusion_mode in {"audio_only", "audio_static", "gated"}
        self.use_static = fusion_mode in {"static_only", "audio_static", "gated"} and self.static_feature_dim > 0
        if fusion_mode in {"static_only", "audio_static", "gated"} and self.static_feature_dim <= 0:
            raise ValueError(f"fusion_mode={fusion_mode!r} requires static_feature_dim > 0")

        self.static_dropout = nn.Dropout(float(static_dropout)) if float(static_dropout) > 0 else nn.Identity()
        self.static_projection_dim = int(static_projection_dim)
        if self.use_static and self.static_projection_dim > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(self.static_feature_dim, self.static_projection_dim),
                nn.GELU(),
                nn.LayerNorm(self.static_projection_dim),
            )
            static_out_dim = self.static_projection_dim
        else:
            self.static_projection = nn.Identity()
            static_out_dim = self.static_feature_dim if self.use_static else 0

        if self.use_static and fusion_mode == "gated":
            init = 0.25 if static_gate_init is None else float(static_gate_init)
            self.static_gate_logit = nn.Parameter(torch.tensor(_logit(init), dtype=torch.float32))
        else:
            self.register_parameter("static_gate_logit", None)

        self.input_dim = (self.audio_feature_dim if self.use_audio else 0) + static_out_dim
        if self.input_dim <= 0:
            raise ValueError("Controlled classifier needs at least one input branch.")
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.input_dim, self.input_dim),
            nn.Sigmoid(),
        )

        layers: list[nn.Module] = []
        prev = self.input_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, int(width)), nn.GELU(), nn.Dropout(float(dropout))])
            prev = int(width)
        layers.append(nn.Linear(prev, int(num_labels)))
        self.ffnn = nn.Sequential(*layers)

    def _static_branch(self, audio_features: torch.Tensor, static_features: Optional[torch.Tensor]) -> torch.Tensor:
        if static_features is None:
            static_features = audio_features.new_zeros(audio_features.shape[0], self.static_feature_dim)
        static_features = static_features.float().to(audio_features.device)
        if static_features.shape[-1] != self.static_feature_dim:
            raise ValueError(f"Expected static dim {self.static_feature_dim}, got {static_features.shape[-1]}")
        static_features = self.static_dropout(static_features)
        static_rep = self.static_projection(static_features)
        if self.static_gate_logit is not None:
            static_rep = static_rep * torch.sigmoid(self.static_gate_logit)
        return static_rep

    def forward(
        self,
        audio_features: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ):
        parts: list[torch.Tensor] = []
        if self.use_audio:
            parts.append(audio_features)
        if self.use_static:
            parts.append(self._static_branch(audio_features, static_features))
        features = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        features = self.input_norm(features)
        attn = self.attention(features)
        logits = self.ffnn(features * attn)
        if not self.use_audio:
            logits = logits + 0.0 * audio_features[:, :1].sum(dim=-1, keepdim=True)
        if return_attention:
            return logits, attn
        return logits


@dataclass(frozen=True)
class ControlledStaticInfo:
    info: StaticFeatureInfo
    z_clip: Optional[float]


def _add_controlled_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fusion-mode",
        choices=["audio_only", "static_only", "audio_static", "gated"],
        default="gated",
        help="Controlled Stage 2 feature-fusion mode.",
    )
    p.add_argument(
        "--static-projection-dim",
        type=int,
        default=32,
        help="Project static features before fusion. Use 0 to concatenate raw normalized features.",
    )
    p.add_argument("--static-dropout", type=float, default=0.30)
    p.add_argument("--static-gate-init", type=float, default=0.25)
    p.add_argument(
        "--static-z-clip",
        type=float,
        default=3.0,
        help="Clip normalized static z-scores to [-clip, clip]. Use <=0 to disable.",
    )
    p.add_argument(
        "--freeze-audio-for-static-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze client and server encoder parameters in static_only ablations.",
    )


def _controlled_dict_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": CONTROLLED_FUSION_VERSION,
        "fusion_mode": str(args.fusion_mode),
        "static_projection_dim": int(args.static_projection_dim),
        "static_dropout": float(args.static_dropout),
        "static_gate_init": None if args.static_gate_init is None else float(args.static_gate_init),
        "static_z_clip": None if float(args.static_z_clip) <= 0 else float(args.static_z_clip),
        "freeze_audio_for_static_only": bool(args.freeze_audio_for_static_only),
    }


def _apply_controlled_metadata(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    cfg = meta.get("controlled_fusion") or {}
    if not cfg:
        return
    for key, default in (
        ("fusion_mode", "audio_static"),
        ("static_projection_dim", 0),
        ("static_dropout", 0.0),
        ("static_gate_init", 0.25),
        ("static_z_clip", 0.0),
        ("freeze_audio_for_static_only", True),
    ):
        setattr(args, key, cfg.get(key, default))


def _clip_array(features: np.ndarray, z_clip: Optional[float]) -> np.ndarray:
    if z_clip is None or float(z_clip) <= 0:
        return features.astype(np.float32, copy=False)
    return np.clip(features, -float(z_clip), float(z_clip)).astype(np.float32)


def _clip_static_in_loader_dataset(dataset, z_clip: Optional[float]) -> None:
    if z_clip is None or float(z_clip) <= 0:
        return
    target = dataset.dataset if hasattr(dataset, "dataset") else dataset
    if hasattr(target, "static"):
        target.static = torch.clamp(target.static.float(), -float(z_clip), float(z_clip))


def _build_controlled_loaders(args: argparse.Namespace) -> tuple[Any, Optional[ControlledStaticInfo]]:
    if str(getattr(args, "static_feature_source", "none")).lower().strip() == "none":
        pack = _build_loaders(args, include_static=False)
        return pack, None
    pack = _build_loaders(args, include_static=True)
    z_clip = None if float(args.static_z_clip) <= 0 else float(args.static_z_clip)
    for loader in pack.train_loaders:
        _clip_static_in_loader_dataset(loader.dataset, z_clip)
    _clip_static_in_loader_dataset(pack.val_loader.dataset, z_clip)
    info = StaticFeatureInfo(
        source=str(args.static_feature_source),
        backend=str(args.static_feature_backend),
        preset=str(args.static_feature_preset),
        dim=int(args.static_feature_dim),
        names=list(args.static_feature_names),
        mean=list(args.static_feature_mean),
        std=list(args.static_feature_std),
    )
    return pack, ControlledStaticInfo(info=info, z_clip=z_clip)


def _replace_classifier(server, args: argparse.Namespace) -> None:
    hidden = tuple(getattr(server.config, "classifier_hidden", (512, 128)))
    server.classifier = ControlledFeatureAttentionFFNN(
        audio_feature_dim=int(server.embed_dim),
        static_feature_dim=int(args.static_feature_dim),
        num_labels=int(args.num_labels),
        hidden=hidden,
        dropout=float(getattr(server.config, "dropout", 0.1)),
        fusion_mode=str(args.fusion_mode),
        static_projection_dim=int(args.static_projection_dim),
        static_dropout=float(args.static_dropout),
        static_gate_init=getattr(args, "static_gate_init", 0.25),
    )


def _freeze_audio_stack(client, server) -> None:
    # Keep the client graph trainable so the SplitFed token-backward call stays
    # valid. The static-only head adds a zero-valued audio dependency, so client
    # gradients are zero while the server encoder is protected from weight decay.
    for module in (server.server_blocks, server.encoder_norm):
        for param in module.parameters():
            param.requires_grad_(False)


def _build_controlled_pair(args: argparse.Namespace):
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
    _replace_classifier(server, args)
    if str(args.fusion_mode) == "static_only" and bool(args.freeze_audio_for_static_only):
        _freeze_audio_stack(client, server)
    if getattr(args, "load_client", None) is not None:
        _load_matching_state_dict(client, Path(args.load_client), label="client")
    if getattr(args, "load_server", None) is not None:
        _load_matching_state_dict(server, Path(args.load_server), label="server")
    return client.to(device), server.to(device), device


def _save_controlled_pair(
    args: argparse.Namespace,
    client,
    server,
    *,
    best_round: int,
    best_score: float,
) -> dict[str, str]:
    run_name = args.run_name or f"controlled_stage2_{args.vowel}_seed{args.model_init_seed}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    client_path = args.save_dir / f"{run_name}_client.pt"
    server_path = args.save_dir / f"{run_name}_server.pt"
    meta_path = args.save_dir / f"{run_name}_metadata.json"
    torch.save(client.state_dict(), client_path)
    torch.save(server.state_dict(), server_path)
    metadata = {
        "stage": "stage2-controlled",
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
        "static_feature_table": str(args.static_feature_table) if args.static_feature_table is not None else None,
        "static_feature_names": list(getattr(args, "static_feature_names", [])),
        "static_feature_mean": list(getattr(args, "static_feature_mean", [])),
        "static_feature_std": list(getattr(args, "static_feature_std", [])),
        "static_audio_manifest": str(args.static_audio_manifest) if args.static_audio_manifest is not None else None,
        "static_audio_root_eent": str(args.static_audio_root_eent) if args.static_audio_root_eent is not None else None,
        "static_audio_root_svd": str(args.static_audio_root_svd) if args.static_audio_root_svd is not None else None,
        "mask_ratio": float(args.mask_ratio),
        "mask_strategy": str(args.mask_strategy),
        "pooling": str(args.pooling),
        "optimizer": "AdamW",
        "adamw_betas": [float(args.adamw_beta1), float(args.adamw_beta2)],
        "weight_decay": float(args.weight_decay),
        "client_lr": float(args.client_lr),
        "server_lr": float(args.server_lr),
        "n_global_rounds": int(args.n_global_rounds),
        "n_local_epochs": int(args.n_local_epochs),
        "stage2_loss": "BinaryFocalWithLogitsLoss",
        "stage2_focal_gamma": float(args.focal_gamma),
        "stage2_focal_alpha": args.focal_alpha,
        "stage2_primary_metric": "macro_f1",
        "stage2_early_stopping_patience": int(args.early_stopping_patience),
        "best_round": int(best_round),
        "best_val_macro_f1": float(best_score),
        "controlled_fusion": _controlled_dict_from_args(args),
        "client_file": str(client_path),
        "server_file": str(server_path),
    }
    meta_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    return {"client": str(client_path), "server": str(server_path), "metadata": str(meta_path)}


def cmd_train_stage2_controlled(args: argparse.Namespace) -> None:
    _set_run_seed(int(args.model_init_seed))
    pack, _ = _build_controlled_loaders(args)
    client, server, device = _build_controlled_pair(args)
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
    last_summary = "val_macro_f1: _  val_loss: _  patience: _"

    for round_idx in range(int(args.n_global_rounds)):
        header = f"global_epoch: {round_idx}/{int(args.n_global_rounds)}"
        part_lines = [f"p{idx}: train_macro_f1: _  train_loss: _" for idx in range(int(args.n_partitions))]

        def on_partition(partition_id, part_stats):
            part_lines[int(partition_id)] = (
                f"p{partition_id}: train_macro_f1: {part_stats.score:.3f}  train_loss: {part_stats.loss:.4f}"
            )

        run_stage2_splitfed_round(
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
        last_summary = f"val_macro_f1: {val.score:.4f}  val_loss: {val.loss:.4f}  patience: {pstr}"
        print(header)
        print(last_summary)
        print("participants:")
        for line in part_lines:
            print(line)
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
    saved = _save_controlled_pair(args, client, best_server, best_round=best_round, best_score=best_score)
    _print_saved_pair_summary(saved, best_round=best_round, best_score=best_score)


def _eval_static_features_controlled(
    args_for_vowel: argparse.Namespace,
    meta: dict[str, Any],
    x,
    patient_ids,
    dataset_name: str,
    vowel: str,
):
    dim = int(meta.get("static_feature_dim") or getattr(args_for_vowel, "static_feature_dim", 0) or 0)
    if dim <= 0:
        return None
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
    scaled = apply_static_normalizer(raw, meta.get("static_feature_mean") or [], meta.get("static_feature_std") or [])
    cfg = meta.get("controlled_fusion") or {}
    return _clip_array(scaled, cfg.get("static_z_clip"))


def _args_from_controlled_metadata(base_args: argparse.Namespace, metadata_path: Path):
    args, meta = _args_from_metadata(base_args, metadata_path)
    _apply_controlled_metadata(args, meta)
    return args, meta


def cmd_evaluate_stage2_pair_controlled(args: argparse.Namespace) -> None:
    from voice_disorder_torch.data.load import load_all_preprocessed
    from voice_disorder_torch.evaluation.patient_eval import combined_vowel_ai_eval, model_eval_by_id
    from voice_disorder_torch.io.eval_report import save_eval_json
    from voice_disorder_torch.ui.eval_cli import print_eval

    args_a, meta_a = _args_from_controlled_metadata(args, args.metadata_a)
    args_i, meta_i = _args_from_controlled_metadata(args, args.metadata_i)
    _set_run_seed(int(args_a.model_init_seed))
    if args_a.vowel != "a" or args_i.vowel != "i":
        raise SystemExit("evaluate-stage2-pair-controlled expects metadata-a for /a/ and metadata-i for /i/.")
    if args_a.dev_test_seed != args_i.dev_test_seed:
        raise SystemExit("metadata-a and metadata-i must use the same dev_test_seed for paired evaluation.")
    if int(args_a.num_labels) != 1 or int(args_i.num_labels) != 1:
        raise SystemExit("Pair evaluation currently supports binary --num-labels 1 models only.")

    ctx = _make_context(args_a)
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    client_a, server_a, device = _build_controlled_pair(args_a)
    client_i, server_i, _ = _build_controlled_pair(args_i)
    focal_gamma_a = meta_a.get("stage2_focal_gamma")
    focal_gamma_i = meta_i.get("stage2_focal_gamma")
    focal_gamma = float(focal_gamma_a if focal_gamma_a is not None else args.focal_gamma)
    if focal_gamma_i is not None and abs(float(focal_gamma_i) - focal_gamma) > 1e-12:
        raise SystemExit("metadata-a and metadata-i use different focal gamma values.")
    criterion = BinaryFocalWithLogitsLoss(gamma=focal_gamma)

    selected = ["chinese", "german"] if args.eval_dataset == "both" else [args.eval_dataset]
    results = {
        "stage": "evaluate-stage2-pair-controlled",
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
        static_a = _eval_static_features_controlled(args_a, meta_a, xa, ida, dataset_name, "a")
        static_i = _eval_static_features_controlled(args_i, meta_i, xi, idi, dataset_name, "i")
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
    p = argparse.ArgumentParser(description="Controlled fusion CLI for Paper2601 Stage 2 ablations.")
    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train-stage2-controlled", help="Run controlled Stage 2 split fine-tuning.")
    _add_data_args(train_p)
    _add_model_args(train_p)
    _add_train_args(
        train_p,
        default_rounds=PROJECT_STAGE2_MAX_ROUNDS,
        default_local_epochs=5,
        include_early_stopping=True,
        include_focal=True,
    )
    _add_controlled_args(train_p)
    train_p.set_defaults(func=cmd_train_stage2_controlled)

    eval_p = sub.add_parser(
        "evaluate-stage2-pair-controlled",
        help="Evaluate controlled Stage 2 /a/ and /i/ models with combined patient-level scoring.",
    )
    _add_data_args(eval_p)
    _add_model_args(eval_p)
    _add_pair_eval_args(eval_p)
    _add_controlled_args(eval_p)
    eval_p.set_defaults(func=cmd_evaluate_stage2_pair_controlled)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
