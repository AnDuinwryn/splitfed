"""Training entry points for split-learning experiments."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from voice_disorder_torch.config import RunContext
from voice_disorder_torch.data.datasets import MelSegmentDataset, SsastMelDataset
from voice_disorder_torch.data.load import load_all_preprocessed
from voice_disorder_torch.naming import generate_split_cnn_stem, generate_split_ssast_stem
from voice_disorder_torch.ui.live import LiveBlock, supports_ansi

from .aggregation import uniform_average_state_dicts
from .engine import SplitServerPool, train_client_partition
from .loaders import build_cnn_partition_loaders, build_ssast_partition_loaders
from .metadata import save_split_run
from .models.cnn import build_split_cnn_parts
from .models.ssast import SsastClientEncoder, SsastServerHead


def _snapshot_split_weights(client_base: nn.Module, server_models: list[nn.Module]) -> tuple[dict, dict]:
    return copy.deepcopy(client_base.state_dict()), copy.deepcopy(server_models[0].state_dict())


def _restore_split_weights(
    client_base: nn.Module,
    server_models: list[nn.Module],
    client_sd: dict,
    server_sd: dict,
) -> None:
    client_base.load_state_dict(client_sd)
    for model in server_models:
        model.load_state_dict(server_sd)


def _cnn_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    pred = (torch.sigmoid(outputs) >= 0.5).long().view(-1)
    return (pred == labels.view(-1).long()).float().mean().item()


def _class_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    return (torch.argmax(outputs, dim=1) == labels).float().mean().item()


def _cnn_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.float()


def _class_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.long().view(-1)


@torch.no_grad()
def _cnn_val_accuracy(client: nn.Module, server: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    client.eval()
    server.eval()
    correct = 0
    total = 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = server(client(xb))
        pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
        y = yb.view(-1).long()
        correct += int((pred == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def _cnn_val_metrics(
    client: nn.Module, server: nn.Module, val_loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    client.eval()
    server.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    correct = 0
    total = 0
    running = 0.0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = server(client(xb))
        y = yb.view(-1).float()
        running += float(loss_fn(logits.view(-1), y).item()) * int(y.numel())
        pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
        correct += int((pred == yb.view(-1).long()).sum().item())
        total += int(y.numel())
    acc = correct / max(total, 1)
    loss = running / max(total, 1)
    return float(acc), float(loss)


@torch.no_grad()
def _ssast_val_accuracy(client: nn.Module, server: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    client.eval()
    server.eval()
    correct = 0
    total = 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = torch.argmax(server(client(xb)), dim=1)
        correct += int((pred == yb).sum().item())
        total += yb.numel()
    return correct / max(total, 1)


@torch.no_grad()
def _ssast_val_metrics(
    client: nn.Module, server: nn.Module, val_loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    client.eval()
    server.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running = 0.0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = server(client(xb))
        running += float(loss_fn(logits, yb).item()) * int(yb.numel())
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    acc = correct / max(total, 1)
    loss = running / max(total, 1)
    return float(acc), float(loss)


@torch.no_grad()
def _cnn_test_accuracy(
    client: nn.Module,
    server: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = DataLoader(MelSegmentDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)
    client.eval()
    server.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = server(client(xb))
        pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
        y = yb.view(-1).long()
        correct += int((pred == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def _ssast_test_accuracy(
    client: nn.Module,
    server: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    batch_size: int,
    input_tdim: int,
) -> float:
    loader = DataLoader(
        SsastMelDataset(x_test, y_test, input_tdim=input_tdim),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    client.eval()
    server.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = torch.argmax(server(client(xb)), dim=1)
        correct += int((pred == yb).sum().item())
        total += yb.numel()
    return correct / max(total, 1)


def train_split_cnn(
    *,
    ctx: RunContext,
    vowel: Literal["a", "i"],
    model_init_seed: int,
    n_partitions: int,
    partition_seed: int,
    n_global_epochs: int,
    n_local_epochs: int,
    client_lr: float,
    server_lr: float,
) -> dict:
    device = torch.device(ctx.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pack = build_cnn_partition_loaders(
        ctx=ctx,
        vowel=vowel,
        n_partitions=n_partitions,
        partition_seed=partition_seed,
    )
    if pack.shape_probe_nchw is None:
        raise RuntimeError("CNN split training requires a NCHW shape probe tensor.")
    client_template, server_template = build_split_cnn_parts(
        pack.shape_probe_nchw, ctx.train, init_seed=model_init_seed
    )
    client_base = copy.deepcopy(client_template).to(device)
    server_pool = SplitServerPool(
        server_template=copy.deepcopy(server_template).to(device),
        n_partitions=n_partitions,
        server_lr=server_lr,
        criterion=nn.BCEWithLogitsLoss(),
        prepare_labels=_cnn_labels,
        accuracy_fn=_cnn_accuracy,
        device=device,
    )

    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=False)
    x_te, y_te = (bundle.x_test_a, bundle.y_test_a) if vowel == "a" else (bundle.x_test_i, bundle.y_test_i)

    val_log: list[float] = []
    val_loss_log: list[float] = []
    patience = int(ctx.train.early_stopping_patience)
    best_vl = float("inf")
    best_client_sd: dict | None = None
    best_server_sd: dict | None = None
    best_ge = -1
    patience_left = patience
    stopped_early = False

    live = supports_ansi()
    block = LiveBlock(height=3 + int(n_partitions), stream=None) if live else None
    last_summary = "val_acc: _  val_loss: _  patience: _"

    for ge in range(n_global_epochs):
        if block is not None:
            block.redraw([""] * (3 + int(n_partitions)))
        header = f"global_epoch: {ge}/{int(n_global_epochs)}"
        client_sds: list[dict] = []
        summary = last_summary
        part_lines: list[str] = [f"p{idx}: train_acc: _  train_loss: _" for idx in range(int(n_partitions))]
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)
        for idx in range(n_partitions):
            model = copy.deepcopy(client_base).to(device)
            client_sds.append(
                copy.deepcopy(
                    train_client_partition(
                        client_model=model,
                        train_loader=pack.train_loaders[idx],
                        partition_id=idx,
                        n_local_epochs=n_local_epochs,
                        client_lr=client_lr,
                        server_pool=server_pool,
                        device=device,
                    )
                )
            )
            if block is not None:
                acc, loss = server_pool.last_partition_stats.get(int(idx), (0.0, 0.0))
                part_lines[int(idx)] = f"p{idx}: train_acc: {acc:.3f}  train_loss: {loss:.4f}"
                block.redraw([header, summary, "participants:"] + part_lines)
        client_base.load_state_dict(uniform_average_state_dicts(client_sds))

        c_eval = copy.deepcopy(client_base).to(device).eval()
        s_eval = server_pool.global_model().to(device).eval()
        va, vl = _cnn_val_metrics(c_eval, s_eval, pack.val_loader, device)
        val_log.append(va)
        val_loss_log.append(vl)
        pstr = f"{patience_left}" if patience > 0 else "off"
        summary = f"val_acc: {va:.4f}  val_loss: {vl:.4f}  patience: {pstr}"
        last_summary = summary
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)
        else:
            print(f"{header}  {summary}")

        if vl + 1e-12 < best_vl:
            best_vl = vl
            best_client_sd, best_server_sd = _snapshot_split_weights(client_base, server_pool.server_models)
            best_ge = ge
            if patience > 0:
                patience_left = patience
        elif patience > 0:
            patience_left -= 1
            if patience_left <= 0:
                stopped_early = True
                if block is None:
                    print("! early_stop")
                break

    if best_client_sd is not None and best_server_sd is not None:
        _restore_split_weights(client_base, server_pool.server_models, best_client_sd, best_server_sd)

    stem = generate_split_cnn_stem(
        vowel, ctx.splits.dev_test_seed, ctx.splits.train_val_seed, model_init_seed
    )
    n, h, w, c = (bundle.x_train_a if vowel == "a" else bundle.x_train_i).shape
    payload = {
        "framework": "pytorch_split_learning",
        "metadata_version": 2,
        "model_type": "cnn",
        "vowel": vowel,
        "model_name": stem,
        "client_file": f"{stem}_split_client.pt",
        "server_file": f"{stem}_split_server.pt",
        "seeds": {
            "dev_test_seed": ctx.splits.dev_test_seed,
            "train_val_seed": ctx.splits.train_val_seed,
            "model_init_seed": model_init_seed,
            "partition_seed": partition_seed,
        },
        "partitioning": {
            "n_partitions": n_partitions,
            "n_global_epochs": n_global_epochs,
            "n_local_epochs": n_local_epochs,
            "client_lr": client_lr,
            "server_lr": server_lr,
        },
        "architecture": asdict(ctx.train),
        "input_shape_nhwc": [int(n), int(h), int(w), int(c)],
        "loss": "BCEWithLogitsLoss",
        "val_acc_per_global": val_log,
        "val_loss_per_global": val_loss_log,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": patience,
            "stopped_early": stopped_early,
            "best_val_loss": float(best_vl),
            "best_global_epoch": int(best_ge),
            "global_epochs_run": len(val_log),
            "n_global_epochs_max": n_global_epochs,
        },
    }
    save_split_run(
        save_dir=ctx.save_dir,
        stem=stem,
        client_sd=client_base.state_dict(),
        server_sd=server_pool.global_model().state_dict(),
        payload=payload,
    )
    return {"model_name": stem, "payload": payload}


def train_split_ssast(
    *,
    ctx: RunContext,
    vowel: Literal["a", "i"],
    model_init_seed: int,
    n_partitions: int,
    partition_seed: int,
    n_global_epochs: int,
    n_local_epochs: int,
    client_lr: float,
    server_lr: float,
    pretrained_path: Path,
    n_client_blocks: int,
    input_fdim: int,
    input_tdim: int,
    f_shape: int,
    t_shape: int,
    f_stride: int,
    t_stride: int,
    model_size: str,
) -> dict:
    device = torch.device(ctx.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(model_init_seed))
    np.random.seed(int(model_init_seed) % (2**32))

    pack = build_ssast_partition_loaders(
        ctx=ctx,
        vowel=vowel,
        n_partitions=n_partitions,
        partition_seed=partition_seed,
        input_tdim=input_tdim,
        batch_size=ctx.train.batch_size,
    )
    pt = str(pretrained_path.resolve()) if pretrained_path.is_absolute() else str(pretrained_path)
    client_base = SsastClientEncoder(
        label_dim=2,
        f_shape=f_shape,
        t_shape=t_shape,
        f_stride=f_stride,
        t_stride=t_stride,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        model_size=model_size,
        load_pretrained_mdl_path=pt,
        n_client_blocks=n_client_blocks,
    ).to(device)
    server_template = SsastServerHead(
        label_dim=2,
        f_shape=f_shape,
        t_shape=t_shape,
        f_stride=f_stride,
        t_stride=t_stride,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        model_size=model_size,
        load_pretrained_mdl_path=pt,
        n_client_blocks=n_client_blocks,
    ).to(device)
    server_pool = SplitServerPool(
        server_template=server_template,
        n_partitions=n_partitions,
        server_lr=server_lr,
        criterion=nn.CrossEntropyLoss(),
        prepare_labels=_class_labels,
        accuracy_fn=_class_accuracy,
        device=device,
    )

    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=False)
    x_te, y_te = (bundle.x_test_a, bundle.y_test_a) if vowel == "a" else (bundle.x_test_i, bundle.y_test_i)

    val_log: list[float] = []
    val_loss_log: list[float] = []
    patience = int(ctx.train.early_stopping_patience)
    best_vl = float("inf")
    best_client_sd: dict | None = None
    best_server_sd: dict | None = None
    best_ge = -1
    patience_left = patience
    stopped_early = False

    live = supports_ansi()
    block = LiveBlock(height=3 + int(n_partitions), stream=None) if live else None
    last_summary = "val_acc: _  val_loss: _  patience: _"

    for ge in range(n_global_epochs):
        if block is not None:
            block.redraw([""] * (3 + int(n_partitions)))
        header = f"global_epoch: {ge}/{int(n_global_epochs)}"
        client_sds: list[dict] = []
        summary = last_summary
        part_lines: list[str] = [f"p{idx}: train_acc: _  train_loss: _" for idx in range(int(n_partitions))]
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)
        for idx in range(n_partitions):
            model = copy.deepcopy(client_base).to(device)
            client_sds.append(
                copy.deepcopy(
                    train_client_partition(
                        client_model=model,
                        train_loader=pack.train_loaders[idx],
                        partition_id=idx,
                        n_local_epochs=n_local_epochs,
                        client_lr=client_lr,
                        server_pool=server_pool,
                        device=device,
                    )
                )
            )
            if block is not None:
                acc, loss = server_pool.last_partition_stats.get(int(idx), (0.0, 0.0))
                part_lines[int(idx)] = f"p{idx}: train_acc: {acc:.3f}  train_loss: {loss:.4f}"
                block.redraw([header, summary, "participants:"] + part_lines)
        client_base.load_state_dict(uniform_average_state_dicts(client_sds))

        c_eval = copy.deepcopy(client_base).to(device).eval()
        s_eval = server_pool.global_model().to(device).eval()
        va, vl = _ssast_val_metrics(c_eval, s_eval, pack.val_loader, device)
        val_log.append(va)
        val_loss_log.append(vl)
        pstr = f"{patience_left}" if patience > 0 else "off"
        summary = f"val_acc: {va:.4f}  val_loss: {vl:.4f}  patience: {pstr}"
        last_summary = summary
        if block is not None:
            block.redraw([header, summary, "participants:"] + part_lines)
        else:
            print(f"{header}  {summary}")

        if vl + 1e-12 < best_vl:
            best_vl = vl
            best_client_sd, best_server_sd = _snapshot_split_weights(client_base, server_pool.server_models)
            best_ge = ge
            if patience > 0:
                patience_left = patience
        elif patience > 0:
            patience_left -= 1
            if patience_left <= 0:
                stopped_early = True
                if block is None:
                    print("! early_stop")
                break

    if best_client_sd is not None and best_server_sd is not None:
        _restore_split_weights(client_base, server_pool.server_models, best_client_sd, best_server_sd)

    stem = generate_split_ssast_stem(
        vowel, ctx.splits.dev_test_seed, ctx.splits.train_val_seed, model_init_seed, n_client_blocks
    )
    n, h, w, c = (bundle.x_train_a if vowel == "a" else bundle.x_train_i).shape
    payload = {
        "framework": "pytorch_split_learning",
        "metadata_version": 2,
        "model_type": "ssast",
        "vowel": vowel,
        "model_name": stem,
        "client_file": f"{stem}_split_client.pt",
        "server_file": f"{stem}_split_server.pt",
        "pretrained_path": pt,
        "ssast": {
            "n_client_blocks": n_client_blocks,
            "input_fdim": input_fdim,
            "input_tdim": input_tdim,
            "f_shape": f_shape,
            "t_shape": t_shape,
            "f_stride": f_stride,
            "t_stride": t_stride,
            "model_size": model_size,
        },
        "seeds": {
            "dev_test_seed": ctx.splits.dev_test_seed,
            "train_val_seed": ctx.splits.train_val_seed,
            "model_init_seed": model_init_seed,
            "partition_seed": partition_seed,
        },
        "partitioning": {
            "n_partitions": n_partitions,
            "n_global_epochs": n_global_epochs,
            "n_local_epochs": n_local_epochs,
            "client_lr": client_lr,
            "server_lr": server_lr,
        },
        "input_shape_nhwc": [int(n), int(h), int(w), int(c)],
        "loss": "CrossEntropyLoss",
        "val_acc_per_global": val_log,
        "val_loss_per_global": val_loss_log,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": patience,
            "stopped_early": stopped_early,
            "best_val_loss": float(best_vl),
            "best_global_epoch": int(best_ge),
            "global_epochs_run": len(val_log),
            "n_global_epochs_max": n_global_epochs,
        },
    }
    save_split_run(
        save_dir=ctx.save_dir,
        stem=stem,
        client_sd=client_base.state_dict(),
        server_sd=server_pool.global_model().state_dict(),
        payload=payload,
    )
    return {"model_name": stem, "payload": payload}
