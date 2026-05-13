from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from paper2601_splitmae_cli import (
    PROJECT_STAGE1_MAX_ROUNDS,
    PROJECT_STAGE2_MAX_ROUNDS,
    _add_data_args,
    _add_model_args,
    _add_train_args,
    _default_pair_results_json,
    _device,
    _eval_arrays_for_dataset,
    _json_ready,
    _live_block,
    _load_matching_state_dict,
    _make_context,
    _print_saved_pair_summary,
    _printable_pair_eval,
    _set_run_seed,
    _stage2_eval_loader,
)
from paper2601_splitmae_client import Paper2601SplitMAEClient, SplitMAEClientConfig
from paper2601_splitmae_server import Paper2601SplitMAEServer, SplitMAEServerConfig
from paper2601_splitmae_utils import SmashedData, average_state_dicts


SSAST_LIKE_VERSION = 1


class MeanPatchLinearHead(nn.Module):
    """SSAST-style fine-tuning head: LayerNorm then Linear over pooled patch tokens."""

    def __init__(self, embed_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(embed_dim))
        self.linear = nn.Linear(int(embed_dim), int(num_classes))

    def forward(
        self,
        audio_features: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ):
        _ = static_features
        logits = self.linear(self.norm(audio_features))
        if return_attention:
            return logits, audio_features.new_ones(audio_features.shape)
        return logits


class CEStats:
    def __init__(self, loss: float, acc: float) -> None:
        self.loss = float(loss)
        self.acc = float(acc)


def _class_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 2 and labels.shape[1] > 1:
        return torch.argmax(labels, dim=1).long()
    return labels.long().view(-1)


def _class_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    target = _class_labels(labels.to(logits.device))
    pred = torch.argmax(logits.detach(), dim=1)
    return float((pred == target).float().mean().item())


def _build_ssast_like_pair(args: argparse.Namespace):
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
            num_labels=2,
            static_feature_dim=0,
            pooling="mean_patch",
            imagenet_pretrain=bool(args.imagenet_pretrain),
            audioset_checkpoint_path=args.audioset_checkpoint_path,
        )
    )
    server.classifier = MeanPatchLinearHead(server.embed_dim, num_classes=2)
    if getattr(args, "load_client", None) is not None:
        _load_matching_state_dict(client, Path(args.load_client), label="client")
    if getattr(args, "load_server", None) is not None:
        _load_matching_state_dict(server, Path(args.load_server), label="server")
    return client.to(device), server.to(device), device


class CrossEntropyServerPool:
    def __init__(
        self,
        *,
        server_template: Paper2601SplitMAEServer,
        n_partitions: int,
        server_lr: float,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.server_models = [copy.deepcopy(server_template).to(self.device) for _ in range(int(n_partitions))]
        self.server_lr = float(server_lr)
        self.server_opts = [torch.optim.Adam(model.parameters(), lr=self.server_lr) for model in self.server_models]
        self.criterion = nn.CrossEntropyLoss()

    def step(self, partition_id: int, smashed: SmashedData, labels: torch.Tensor) -> tuple[torch.Tensor, CEStats]:
        model = self.server_models[int(partition_id)]
        opt = self.server_opts[int(partition_id)]
        model.train()
        opt.zero_grad(set_to_none=True)
        payload = smashed.to(self.device).detach_for_server(requires_grad=True)
        logits = model.forward_finetune(payload)["logits"]
        target = _class_labels(labels.to(self.device))
        loss = self.criterion(logits, target)
        acc = _class_accuracy(logits, target)
        loss.backward()
        grad = payload.tokens.grad.detach().clone().to(smashed.tokens.device)
        opt.step()
        return grad, CEStats(float(loss.detach().item()), acc)

    def average_replicas(self) -> None:
        averaged = average_state_dicts([model.state_dict() for model in self.server_models])
        for model in self.server_models:
            model.load_state_dict(averaged)
        self.server_opts = [torch.optim.Adam(model.parameters(), lr=self.server_lr) for model in self.server_models]

    def global_model(self) -> Paper2601SplitMAEServer:
        return copy.deepcopy(self.server_models[0])


def _mean_stats(stats: list[CEStats]) -> CEStats:
    if not stats:
        return CEStats(0.0, 0.0)
    return CEStats(
        sum(s.loss for s in stats) / len(stats),
        sum(s.acc for s in stats) / len(stats),
    )


def _train_client_partition(
    *,
    client_model: Paper2601SplitMAEClient,
    train_loader: DataLoader,
    partition_id: int,
    n_local_epochs: int,
    client_lr: float,
    server_pool: CrossEntropyServerPool,
    device: torch.device | str,
) -> tuple[dict, list[CEStats]]:
    device = torch.device(device)
    client_model.to(device).train()
    opt = torch.optim.Adam(client_model.parameters(), lr=float(client_lr))
    stats: list[CEStats] = []
    for _ in range(int(n_local_epochs)):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            smashed = client_model(xb, mode="finetune")
            grad, step_stats = server_pool.step(partition_id, smashed, yb)
            smashed.tokens.backward(grad)
            opt.step()
            stats.append(step_stats)
    return copy.deepcopy(client_model.state_dict()), stats


def _run_splitfed_round(
    *,
    client_base: Paper2601SplitMAEClient,
    server_pool: CrossEntropyServerPool,
    train_loaders: list[DataLoader],
    n_local_epochs: int,
    client_lr: float,
    device: torch.device | str,
    progress_fn=None,
) -> list[CEStats]:
    client_sds: list[dict] = []
    out_stats: list[CEStats] = []
    for partition_id, loader in enumerate(train_loaders):
        client = copy.deepcopy(client_base)
        client_sd, stats = _train_client_partition(
            client_model=client,
            train_loader=loader,
            partition_id=int(partition_id),
            n_local_epochs=int(n_local_epochs),
            client_lr=float(client_lr),
            server_pool=server_pool,
            device=device,
        )
        client_sds.append(client_sd)
        part_stats = _mean_stats(stats)
        out_stats.append(part_stats)
        if progress_fn is not None:
            progress_fn(int(partition_id), part_stats)
    client_base.load_state_dict(average_state_dicts(client_sds))
    server_pool.average_replicas()
    return out_stats


@torch.no_grad()
def _evaluate_ce(
    *,
    client: Paper2601SplitMAEClient,
    server: Paper2601SplitMAEServer,
    loader: DataLoader,
    device: torch.device | str,
) -> CEStats:
    device = torch.device(device)
    client.to(device).eval()
    server.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        smashed = client(xb, mode="finetune").to(device)
        logits = server.forward_finetune(smashed)["logits"]
        target = _class_labels(yb.to(device))
        n = int(xb.shape[0])
        total_loss += float(criterion(logits, target).item()) * n
        total_acc += _class_accuracy(logits, target) * n
        total_n += n
    denom = max(total_n, 1)
    return CEStats(total_loss / denom, total_acc / denom)


@torch.no_grad()
def _predict_positive_probs(
    client: Paper2601SplitMAEClient,
    server: Paper2601SplitMAEServer,
    loader: DataLoader,
    device: torch.device | str,
) -> np.ndarray:
    device = torch.device(device)
    client.to(device).eval()
    server.to(device).eval()
    probs = []
    for batch in loader:
        xb = batch[0].to(device)
        smashed = client(xb, mode="finetune").to(device)
        logits = server.forward_finetune(smashed)["logits"]
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())
    return torch.cat(probs, dim=0).numpy()


def _save_pair(
    args: argparse.Namespace,
    client: Paper2601SplitMAEClient,
    server: Paper2601SplitMAEServer,
    *,
    best_round: int,
    best_val_loss: float,
    best_val_acc: float,
) -> dict[str, str]:
    run_name = args.run_name or f"ssast_like_stage2_{args.vowel}_seed{args.model_init_seed}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    client_path = args.save_dir / f"{run_name}_client.pt"
    server_path = args.save_dir / f"{run_name}_server.pt"
    meta_path = args.save_dir / f"{run_name}_metadata.json"
    torch.save(client.state_dict(), client_path)
    torch.save(server.state_dict(), server_path)
    metadata = {
        "stage": "stage2-ssast-like",
        "paper2601_ssast_like_version": SSAST_LIKE_VERSION,
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
        "num_labels": 2,
        "mask_ratio": float(args.mask_ratio),
        "mask_strategy": str(args.mask_strategy),
        "pooling": "mean_patch",
        "classifier": "LayerNorm+Linear(2)",
        "static_feature_dim": 0,
        "optimizer": "Adam",
        "client_lr": float(args.client_lr),
        "server_lr": float(args.server_lr),
        "n_global_rounds": int(args.n_global_rounds),
        "n_local_epochs": int(args.n_local_epochs),
        "stage2_loss": "CrossEntropyLoss",
        "stage2_primary_metric": "val_loss",
        "stage2_early_stopping_patience": int(args.early_stopping_patience),
        "best_round": int(best_round),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "client_file": str(client_path),
        "server_file": str(server_path),
        "load_client": str(args.load_client) if args.load_client is not None else None,
        "load_server": str(args.load_server) if args.load_server is not None else None,
    }
    meta_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    return {"client": str(client_path), "server": str(server_path), "metadata": str(meta_path)}


def _apply_metadata_to_args(base_args: argparse.Namespace, metadata_path: Path) -> tuple[argparse.Namespace, dict[str, Any]]:
    args = copy.copy(base_args)
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    for key in (
        "vowel",
        "model_size",
        "input_fdim",
        "input_tdim",
        "n_client_blocks",
        "batch_size",
        "mask_ratio",
        "mask_strategy",
        "model_init_seed",
        "dev_test_seed",
        "train_val_seed",
        "partition_seed",
    ):
        if key in meta and hasattr(args, key):
            setattr(args, key, meta[key])
    for attr, meta_key in (("load_client", "client_file"), ("load_server", "server_file")):
        value = meta.get(meta_key)
        if value is None:
            setattr(args, attr, None)
            continue
        path = Path(value)
        if not path.is_file():
            candidate = Path(metadata_path).parent / path.name
            if candidate.is_file():
                path = candidate
        setattr(args, attr, path)
    args.num_labels = 2
    return args, meta


def cmd_train_stage2(args: argparse.Namespace) -> None:
    _set_run_seed(int(args.model_init_seed))
    args.num_labels = 2
    ctx = _make_context(args)
    from voice_disorder_torch.split_learning.loaders import build_ssast_partition_loaders

    pack = build_ssast_partition_loaders(
        ctx=ctx,
        vowel=args.vowel,
        n_partitions=int(args.n_partitions),
        partition_seed=int(args.partition_seed),
        input_tdim=int(args.input_tdim),
        batch_size=int(args.batch_size),
    )
    client, server, device = _build_ssast_like_pair(args)
    server_pool = CrossEntropyServerPool(
        server_template=server,
        n_partitions=int(args.n_partitions),
        server_lr=float(args.server_lr),
        device=device,
    )

    patience = int(args.early_stopping_patience)
    patience_left = patience
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_round = -1
    best_client_sd = copy.deepcopy(client.state_dict())
    best_server_sd = copy.deepcopy(server_pool.global_model().state_dict())
    block = _live_block(3 + int(args.n_partitions))
    last_summary = "val_acc: _  val_loss: _  patience: _"

    for round_idx in range(int(args.n_global_rounds)):
        header = f"global_epoch: {round_idx}/{int(args.n_global_rounds)}"
        part_lines = [f"p{idx}: train_acc: _  train_loss: _" for idx in range(int(args.n_partitions))]
        if block is not None:
            block.redraw([header, last_summary, "participants:"] + part_lines)

        def on_partition(partition_id: int, stats: CEStats) -> None:
            part_lines[int(partition_id)] = f"p{partition_id}: train_acc: {stats.acc:.3f}  train_loss: {stats.loss:.4f}"
            if block is not None:
                block.redraw([header, last_summary, "participants:"] + part_lines)

        _run_splitfed_round(
            client_base=client,
            server_pool=server_pool,
            train_loaders=pack.train_loaders,
            n_local_epochs=int(args.n_local_epochs),
            client_lr=float(args.client_lr),
            device=device,
            progress_fn=on_partition,
        )
        val = _evaluate_ce(client=client, server=server_pool.global_model(), loader=pack.val_loader, device=device)
        pstr = f"{patience_left}" if patience > 0 else "off"
        last_summary = f"val_acc: {val.acc:.4f}  val_loss: {val.loss:.4f}  patience: {pstr}"
        if block is not None:
            block.redraw([header, last_summary, "participants:"] + part_lines)
        else:
            print(f"{header}  {last_summary}")

        if val.loss + 1e-12 < best_val_loss:
            best_val_loss = float(val.loss)
            best_val_acc = float(val.acc)
            best_round = int(round_idx)
            best_client_sd = copy.deepcopy(client.state_dict())
            best_server_sd = copy.deepcopy(server_pool.global_model().state_dict())
            if patience > 0:
                patience_left = patience
        elif patience > 0:
            patience_left -= 1
            if patience_left <= 0:
                print(f"! early_stop  best_round: {best_round}  best_val_loss: {best_val_loss:.4f}")
                break

    client.load_state_dict(best_client_sd)
    best_server = server_pool.global_model()
    best_server.load_state_dict(best_server_sd)
    saved = _save_pair(
        args,
        client,
        best_server,
        best_round=best_round,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
    )
    _print_saved_pair_summary(saved)
    print(f"[done] best_round: {best_round}  best_val_loss: {best_val_loss:.4f}  best_val_acc: {best_val_acc:.4f}")


def _evaluate_dataset(
    *,
    bundle,
    dataset_name: str,
    client_a,
    server_a,
    args_a: argparse.Namespace,
    client_i,
    server_i,
    args_i: argparse.Namespace,
    device,
    patient_eval_strategy: str,
    patient_prob_threshold: float,
    verbose: bool,
) -> dict[str, Any]:
    from paper2601_splitmae_training import macro_f1_score
    from voice_disorder_torch.evaluation.patient_eval import combined_vowel_ai_eval, model_eval_by_id

    xa, ya, ida, display_name = _eval_arrays_for_dataset(bundle, "a", dataset_name)
    xi, yi, idi, _ = _eval_arrays_for_dataset(bundle, "i", dataset_name)
    ds_a, loader_a = _stage2_eval_loader(xa, ya, int(args_a.input_tdim), int(args_a.batch_size), None)
    ds_i, loader_i = _stage2_eval_loader(xi, yi, int(args_i.input_tdim), int(args_i.batch_size), None)
    seg_a = _evaluate_ce(client=client_a, server=server_a, loader=loader_a, device=device)
    seg_i = _evaluate_ce(client=client_i, server=server_i, loader=loader_i, device=device)
    pa = _predict_positive_probs(client_a, server_a, loader_a, device)
    pi = _predict_positive_probs(client_i, server_i, loader_i, device)
    single_a = model_eval_by_id(
        xa,
        ya,
        list(ida),
        pa,
        vowel_type="a",
        dataset_type=display_name,
        strategy=patient_eval_strategy,
        patient_prob_threshold=float(patient_prob_threshold),
        verbose=bool(verbose),
    )
    single_i = model_eval_by_id(
        xi,
        yi,
        list(idi),
        pi,
        vowel_type="i",
        dataset_type=display_name,
        strategy=patient_eval_strategy,
        patient_prob_threshold=float(patient_prob_threshold),
        verbose=bool(verbose),
    )
    combined = combined_vowel_ai_eval(
        pa,
        pi,
        ya,
        yi,
        ida,
        idi,
        dataset_type=display_name,
        strategy=patient_eval_strategy,
        patient_prob_threshold=float(patient_prob_threshold),
        verbose=bool(verbose),
    )
    return {
        "segment_a": {
            "segment_loss": float(seg_a.loss),
            "segment_acc": float(seg_a.acc),
            "n_segments": int(len(ds_a)),
            "n_patients": int(len(set(str(pid) for pid in ida))),
        },
        "segment_i": {
            "segment_loss": float(seg_i.loss),
            "segment_acc": float(seg_i.acc),
            "n_segments": int(len(ds_i)),
            "n_patients": int(len(set(str(pid) for pid in idi))),
        },
        "single_a": single_a,
        "single_i": single_i,
        "combined": combined,
    }


def _load_pair_for_eval(args: argparse.Namespace):
    from voice_disorder_torch.data.load import load_all_preprocessed

    args_a, meta_a = _apply_metadata_to_args(args, args.metadata_a)
    args_i, meta_i = _apply_metadata_to_args(args, args.metadata_i)
    if args_a.vowel != "a" or args_i.vowel != "i":
        raise SystemExit("Expected metadata-a for /a/ and metadata-i for /i/.")
    _set_run_seed(int(args_a.model_init_seed))
    ctx = _make_context(args_a)
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    client_a, server_a, device = _build_ssast_like_pair(args_a)
    client_i, server_i, _ = _build_ssast_like_pair(args_i)
    return bundle, client_a, server_a, args_a, meta_a, client_i, server_i, args_i, meta_i, device


def cmd_evaluate_stage2_pair(args: argparse.Namespace) -> None:
    from voice_disorder_torch.io.eval_report import save_eval_json
    from voice_disorder_torch.ui.eval_cli import print_eval

    bundle, client_a, server_a, args_a, meta_a, client_i, server_i, args_i, meta_i, device = _load_pair_for_eval(args)
    selected = ["chinese", "german"] if args.eval_dataset == "both" else [args.eval_dataset]
    results = {
        "stage": "evaluate-stage2-pair-ssast-like",
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
        "loaded_metadata_a": meta_a,
        "loaded_metadata_i": meta_i,
        "datasets": {},
    }
    for dataset_name in selected:
        results["datasets"][dataset_name] = _evaluate_dataset(
            bundle=bundle,
            dataset_name=dataset_name,
            client_a=client_a,
            server_a=server_a,
            args_a=args_a,
            client_i=client_i,
            server_i=server_i,
            args_i=args_i,
            device=device,
            patient_eval_strategy=args.patient_eval_strategy,
            patient_prob_threshold=float(args.patient_prob_threshold),
            verbose=bool(args.verbose),
        )
    results = _json_ready(results)
    print_eval(_printable_pair_eval(results), verbose=bool(args.verbose))
    results_json = Path(args.results_json) if args.results_json is not None else _default_pair_results_json(args)
    save_eval_json(results_json, results)
    print(f"Wrote evaluation JSON: {results_json.resolve()}")


def _val_arrays_for_vowel(bundle, vowel: str):
    if vowel == "a":
        return bundle.x_val_a, bundle.y_val_a, bundle.id_val_a
    return bundle.x_val_i, bundle.y_val_i, bundle.id_val_i


def _predict_for_arrays(client, server, args_for_vowel, x, y, device) -> np.ndarray:
    _, loader = _stage2_eval_loader(x, y, int(args_for_vowel.input_tdim), int(args_for_vowel.batch_size), None)
    return _predict_positive_probs(client, server, loader, device)


def cmd_evaluate_eent_val_threshold(args: argparse.Namespace) -> None:
    from paper2601_eval_eent_val_threshold import _aggregate_patient_scores, _metrics_at_threshold, _select_threshold
    from voice_disorder_torch.io.eval_report import save_eval_json
    from voice_disorder_torch.ui.eval_cli import print_eval

    bundle, client_a, server_a, args_a, meta_a, client_i, server_i, args_i, meta_i, device = _load_pair_for_eval(args)
    xva, yva, idva = _val_arrays_for_vowel(bundle, "a")
    xvi, yvi, idvi = _val_arrays_for_vowel(bundle, "i")
    pva = _predict_for_arrays(client_a, server_a, args_a, xva, yva, device)
    pvi = _predict_for_arrays(client_i, server_i, args_i, xvi, yvi, device)
    val_patient_ids, val_true, val_score = _aggregate_patient_scores(pva, pvi, yva, yvi, idva, idvi)
    calibration = _select_threshold(val_true, val_score, args.threshold_metric)
    threshold = float(calibration["threshold"])
    calibration["validation_patient_ids"] = val_patient_ids
    calibration["n_validation_patients"] = int(len(val_patient_ids))

    selected = ["chinese", "german"] if args.eval_dataset == "both" else [args.eval_dataset]
    results = {
        "stage": "evaluate-stage2-pair-ssast-like-eent-val-threshold",
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
        "loaded_metadata_a": meta_a,
        "loaded_metadata_i": meta_i,
        "calibration": calibration,
        "datasets": {},
    }
    for dataset_name in selected:
        xa, ya, ida, display_name = _eval_arrays_for_dataset(bundle, "a", dataset_name)
        xi, yi, idi, _ = _eval_arrays_for_dataset(bundle, "i", dataset_name)
        pa = _predict_for_arrays(client_a, server_a, args_a, xa, ya, device)
        pi = _predict_for_arrays(client_i, server_i, args_i, xi, yi, device)
        patient_ids, y_true, y_score = _aggregate_patient_scores(pa, pi, ya, yi, ida, idi)
        combined = _metrics_at_threshold(y_true, y_score, threshold)
        combined["dataset_type"] = display_name
        combined["n_patients"] = int(len(patient_ids))
        combined["patient_ids"] = patient_ids
        results["datasets"][dataset_name] = {
            "combined": combined,
            "segment_a": {"n_segments": int(len(xa)), "n_patients": int(len(set(str(pid) for pid in ida)))},
            "segment_i": {"n_segments": int(len(xi)), "n_patients": int(len(set(str(pid) for pid in idi)))},
        }
    results = _json_ready(results)
    print(
        f"EENT validation threshold: {threshold:.6f} "
        f"({args.threshold_metric}={calibration['threshold_metric_value']:.4f})"
    )
    print_eval(
        {
            "patient_eval_strategy": "eent_validation_threshold",
            "patient_prob_threshold": threshold,
            **results["datasets"],
        },
        verbose=bool(args.verbose),
    )
    results_json = Path(args.results_json)
    save_eval_json(results_json, results)
    print(f"Wrote validation-threshold evaluation JSON: {results_json.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SSAST-like finetuning head for the Paper2601 Stage 1 MAE encoder.")
    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train-stage2-ssast-like")
    _add_data_args(train_p)
    _add_model_args(train_p)
    _add_train_args(train_p, default_rounds=PROJECT_STAGE2_MAX_ROUNDS, default_local_epochs=5, include_early_stopping=True)
    train_p.set_defaults(client_lr=5e-5, server_lr=5e-5, num_labels=2, func=cmd_train_stage2)

    eval_p = sub.add_parser("evaluate-stage2-pair")
    _add_data_args(eval_p)
    _add_model_args(eval_p)
    eval_p.add_argument("--metadata-a", type=Path, required=True)
    eval_p.add_argument("--metadata-i", type=Path, required=True)
    eval_p.add_argument("--eval-dataset", choices=["chinese", "german", "both"], default="both")
    eval_p.add_argument(
        "--patient-eval-strategy",
        choices=["fixed", "best_threshold", "relative", "percentage", "max recall", "guding"],
        default="fixed",
    )
    eval_p.add_argument("--patient-prob-threshold", type=float, default=0.5)
    eval_p.add_argument("--results-json", type=Path, default=None)
    eval_p.add_argument("--verbose", action="store_true")
    eval_p.set_defaults(num_labels=2, func=cmd_evaluate_stage2_pair)

    val_p = sub.add_parser("evaluate-stage2-pair-eent-val-threshold")
    _add_data_args(val_p)
    _add_model_args(val_p)
    val_p.add_argument("--metadata-a", type=Path, required=True)
    val_p.add_argument("--metadata-i", type=Path, required=True)
    val_p.add_argument("--eval-dataset", choices=["chinese", "german", "both"], default="both")
    val_p.add_argument("--threshold-metric", choices=["macro_f1", "f1", "accuracy", "youden"], default="macro_f1")
    val_p.add_argument("--results-json", type=Path, required=True)
    val_p.add_argument("--verbose", action="store_true")
    val_p.set_defaults(num_labels=2, func=cmd_evaluate_eent_val_threshold)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
