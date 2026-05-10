from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper2601_splitmae_client import Paper2601SplitMAEClient
from paper2601_splitmae_server import Paper2601SplitMAEServer
from paper2601_splitmae_utils import SmashedData, average_state_dicts, labels_for_bce


PAPER_ADAMW_BETAS = (0.9, 0.95)
PAPER_WEIGHT_DECAY = 0.05
PAPER_FOCAL_GAMMA = 2.0


@dataclass
class SplitStepStats:
    loss: float
    score: float = 0.0


def unpack_batch(batch) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Accept `(x,y)` from current loaders or `(x,y,static_features)`."""

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1], None
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise ValueError("Expected DataLoader batch shaped as (x,y) or (x,y,static_features).")


def _binary_f1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    denom = (2.0 * tp + fp + fn).clamp_min(1e-12)
    return (2.0 * tp) / denom


def macro_f1_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels_for_bce(labels.to(logits.device), logits.shape[-1])
    pred = (torch.sigmoid(logits) >= 0.5).float()
    if pred.shape[-1] == 1:
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        positive_f1 = _binary_f1(pred_flat, labels_flat)
        negative_f1 = _binary_f1(1.0 - pred_flat, 1.0 - labels_flat)
        return float(torch.stack([negative_f1, positive_f1]).mean().item())
    scores = [_binary_f1(pred[:, idx], labels[:, idx]) for idx in range(pred.shape[-1])]
    return float(torch.stack(scores).mean().item())


class BinaryFocalWithLogitsLoss(nn.Module):
    """Multi-label focal loss over logits, matching the paper's Stage 2 loss family."""

    def __init__(self, gamma: float = PAPER_FOCAL_GAMMA, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt).clamp_min(1e-8) ** self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


class Paper2601SplitServerPool:
    """One server replica per client partition, matching the existing SplitFed style."""

    def __init__(
        self,
        *,
        server_template: Paper2601SplitMAEServer,
        n_partitions: int,
        server_lr: float,
        device: torch.device | str,
        finetune_criterion: Optional[nn.Module] = None,
        optimizer_betas: tuple[float, float] = PAPER_ADAMW_BETAS,
        weight_decay: float = PAPER_WEIGHT_DECAY,
    ) -> None:
        self.device = torch.device(device)
        self.server_models = [copy.deepcopy(server_template).to(self.device) for _ in range(int(n_partitions))]
        self.server_lr = float(server_lr)
        self.optimizer_betas = (float(optimizer_betas[0]), float(optimizer_betas[1]))
        self.weight_decay = float(weight_decay)
        self.server_opts = [
            torch.optim.AdamW(
                model.parameters(),
                lr=self.server_lr,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay,
            )
            for model in self.server_models
        ]
        self.finetune_criterion = finetune_criterion or BinaryFocalWithLogitsLoss()

    def _payload_for_server(self, smashed: SmashedData) -> SmashedData:
        return smashed.to(self.device).detach_for_server(requires_grad=True)

    def step_pretrain(self, partition_id: int, smashed: SmashedData) -> tuple[torch.Tensor, SplitStepStats]:
        net = self.server_models[int(partition_id)]
        opt = self.server_opts[int(partition_id)]
        net.train()
        opt.zero_grad(set_to_none=True)
        payload = self._payload_for_server(smashed)
        out = net.forward_pretrain(payload)
        loss = out["loss"]
        loss.backward()
        grad = payload.tokens.grad.detach().clone().to(smashed.tokens.device)
        opt.step()
        return grad, SplitStepStats(loss=float(loss.detach().item()))

    def step_finetune(
        self,
        partition_id: int,
        smashed: SmashedData,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, SplitStepStats]:
        net = self.server_models[int(partition_id)]
        opt = self.server_opts[int(partition_id)]
        net.train()
        opt.zero_grad(set_to_none=True)
        payload = self._payload_for_server(smashed)
        out = net.forward_finetune(payload)
        logits = out["logits"]
        target = labels_for_bce(labels.to(self.device), logits.shape[-1])
        loss = self.finetune_criterion(logits, target)
        score = macro_f1_score(logits.detach(), target.detach())
        loss.backward()
        grad = payload.tokens.grad.detach().clone().to(smashed.tokens.device)
        opt.step()
        return grad, SplitStepStats(loss=float(loss.detach().item()), score=score)

    def average_replicas(self) -> None:
        averaged = average_state_dicts([model.state_dict() for model in self.server_models])
        for model in self.server_models:
            model.load_state_dict(averaged)
        for idx, model in enumerate(self.server_models):
            lr = self.server_opts[idx].param_groups[0]["lr"]
            self.server_opts[idx] = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay,
            )

    def global_model(self) -> Paper2601SplitMAEServer:
        return copy.deepcopy(self.server_models[0])


def train_stage1_client_partition(
    *,
    client_model: Paper2601SplitMAEClient,
    train_loader: DataLoader,
    partition_id: int,
    n_local_epochs: int,
    client_lr: float,
    server_pool: Paper2601SplitServerPool,
    device: torch.device | str,
    optimizer_betas: tuple[float, float] = PAPER_ADAMW_BETAS,
    weight_decay: float = PAPER_WEIGHT_DECAY,
) -> tuple[dict, list[SplitStepStats]]:
    device = torch.device(device)
    client_model.to(device).train()
    opt = torch.optim.AdamW(
        client_model.parameters(),
        lr=float(client_lr),
        betas=(float(optimizer_betas[0]), float(optimizer_betas[1])),
        weight_decay=float(weight_decay),
    )
    stats: list[SplitStepStats] = []
    for _ in range(int(n_local_epochs)):
        for batch in train_loader:
            xb, _, _ = unpack_batch(batch)
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            smashed = client_model(xb, mode="pretrain")
            grad, step_stats = server_pool.step_pretrain(partition_id, smashed)
            smashed.tokens.backward(grad)
            opt.step()
            stats.append(step_stats)
    return copy.deepcopy(client_model.state_dict()), stats


def train_stage2_client_partition(
    *,
    client_model: Paper2601SplitMAEClient,
    train_loader: DataLoader,
    partition_id: int,
    n_local_epochs: int,
    client_lr: float,
    server_pool: Paper2601SplitServerPool,
    device: torch.device | str,
    optimizer_betas: tuple[float, float] = PAPER_ADAMW_BETAS,
    weight_decay: float = PAPER_WEIGHT_DECAY,
) -> tuple[dict, list[SplitStepStats]]:
    device = torch.device(device)
    client_model.to(device).train()
    opt = torch.optim.AdamW(
        client_model.parameters(),
        lr=float(client_lr),
        betas=(float(optimizer_betas[0]), float(optimizer_betas[1])),
        weight_decay=float(weight_decay),
    )
    stats: list[SplitStepStats] = []
    for _ in range(int(n_local_epochs)):
        for batch in train_loader:
            xb, yb, static = unpack_batch(batch)
            xb = xb.to(device)
            yb = yb.to(device)
            static = None if static is None else static.to(device)
            opt.zero_grad(set_to_none=True)
            smashed = client_model(xb, mode="finetune", static_features=static)
            grad, step_stats = server_pool.step_finetune(partition_id, smashed, yb)
            smashed.tokens.backward(grad)
            opt.step()
            stats.append(step_stats)
    return copy.deepcopy(client_model.state_dict()), stats


def _mean_stats(stats: Iterable[SplitStepStats]) -> SplitStepStats:
    stats = list(stats)
    if not stats:
        return SplitStepStats(loss=0.0, score=0.0)
    return SplitStepStats(
        loss=float(sum(s.loss for s in stats) / len(stats)),
        score=float(sum(s.score for s in stats) / len(stats)),
    )


def run_stage1_splitfed_round(
    *,
    client_base: Paper2601SplitMAEClient,
    server_pool: Paper2601SplitServerPool,
    train_loaders: list[DataLoader],
    n_local_epochs: int,
    client_lr: float,
    device: torch.device | str,
    average_servers: bool = True,
    optimizer_betas: tuple[float, float] = PAPER_ADAMW_BETAS,
    weight_decay: float = PAPER_WEIGHT_DECAY,
    progress_fn: Optional[Callable[[int, SplitStepStats], None]] = None,
) -> list[SplitStepStats]:
    client_sds: list[dict] = []
    partition_stats: list[SplitStepStats] = []
    for partition_id, loader in enumerate(train_loaders):
        client = copy.deepcopy(client_base)
        client_sd, stats = train_stage1_client_partition(
            client_model=client,
            train_loader=loader,
            partition_id=partition_id,
            n_local_epochs=n_local_epochs,
            client_lr=client_lr,
            server_pool=server_pool,
            device=device,
            optimizer_betas=optimizer_betas,
            weight_decay=weight_decay,
        )
        client_sds.append(client_sd)
        part_stats = _mean_stats(stats)
        partition_stats.append(part_stats)
        if progress_fn is not None:
            progress_fn(int(partition_id), part_stats)
    client_base.load_state_dict(average_state_dicts(client_sds))
    if average_servers:
        server_pool.average_replicas()
    return partition_stats


def run_stage2_splitfed_round(
    *,
    client_base: Paper2601SplitMAEClient,
    server_pool: Paper2601SplitServerPool,
    train_loaders: list[DataLoader],
    n_local_epochs: int,
    client_lr: float,
    device: torch.device | str,
    average_servers: bool = True,
    optimizer_betas: tuple[float, float] = PAPER_ADAMW_BETAS,
    weight_decay: float = PAPER_WEIGHT_DECAY,
    progress_fn: Optional[Callable[[int, SplitStepStats], None]] = None,
) -> list[SplitStepStats]:
    client_sds: list[dict] = []
    partition_stats: list[SplitStepStats] = []
    for partition_id, loader in enumerate(train_loaders):
        client = copy.deepcopy(client_base)
        client_sd, stats = train_stage2_client_partition(
            client_model=client,
            train_loader=loader,
            partition_id=partition_id,
            n_local_epochs=n_local_epochs,
            client_lr=client_lr,
            server_pool=server_pool,
            device=device,
            optimizer_betas=optimizer_betas,
            weight_decay=weight_decay,
        )
        client_sds.append(client_sd)
        part_stats = _mean_stats(stats)
        partition_stats.append(part_stats)
        if progress_fn is not None:
            progress_fn(int(partition_id), part_stats)
    client_base.load_state_dict(average_state_dicts(client_sds))
    if average_servers:
        server_pool.average_replicas()
    return partition_stats


@torch.no_grad()
def evaluate_stage2(
    *,
    client: Paper2601SplitMAEClient,
    server: Paper2601SplitMAEServer,
    loader: DataLoader,
    device: torch.device | str,
    criterion: Optional[nn.Module] = None,
) -> SplitStepStats:
    device = torch.device(device)
    client.to(device).eval()
    server.to(device).eval()
    criterion = criterion or BinaryFocalWithLogitsLoss()
    total_loss = 0.0
    total_score = 0.0
    total_items = 0
    for batch in loader:
        xb, yb, static = unpack_batch(batch)
        xb = xb.to(device)
        yb = yb.to(device)
        static = None if static is None else static.to(device)
        smashed = client(xb, mode="finetune", static_features=static).to(device)
        logits = server.forward_finetune(smashed)["logits"]
        target = labels_for_bce(yb, logits.shape[-1])
        batch_n = int(xb.shape[0])
        total_loss += float(criterion(logits, target).item()) * batch_n
        total_score += macro_f1_score(logits, target) * batch_n
        total_items += batch_n
    denom = max(total_items, 1)
    return SplitStepStats(loss=total_loss / denom, score=total_score / denom)
