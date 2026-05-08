from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from paper2601_splitmae_client import Paper2601SplitMAEClient
from paper2601_splitmae_server import Paper2601SplitMAEServer
from paper2601_splitmae_utils import SmashedData, average_state_dicts, labels_for_bce


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


def multilabel_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels_for_bce(labels.to(logits.device), logits.shape[-1])
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return float((pred == labels).float().mean().item())


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
    ) -> None:
        self.device = torch.device(device)
        self.server_models = [copy.deepcopy(server_template).to(self.device) for _ in range(int(n_partitions))]
        self.server_opts = [torch.optim.Adam(model.parameters(), lr=float(server_lr)) for model in self.server_models]
        self.finetune_criterion = finetune_criterion or nn.BCEWithLogitsLoss()

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
        score = multilabel_accuracy(logits.detach(), target.detach())
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
            self.server_opts[idx] = torch.optim.Adam(model.parameters(), lr=lr)

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
) -> tuple[dict, list[SplitStepStats]]:
    device = torch.device(device)
    client_model.to(device).train()
    opt = torch.optim.Adam(client_model.parameters(), lr=float(client_lr))
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
) -> tuple[dict, list[SplitStepStats]]:
    device = torch.device(device)
    client_model.to(device).train()
    opt = torch.optim.Adam(client_model.parameters(), lr=float(client_lr))
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
        )
        client_sds.append(client_sd)
        partition_stats.append(_mean_stats(stats))
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
        )
        client_sds.append(client_sd)
        partition_stats.append(_mean_stats(stats))
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
) -> SplitStepStats:
    device = torch.device(device)
    client.to(device).eval()
    server.to(device).eval()
    criterion = nn.BCEWithLogitsLoss()
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
        total_score += multilabel_accuracy(logits, target) * batch_n
        total_items += batch_n
    denom = max(total_items, 1)
    return SplitStepStats(loss=total_loss / denom, score=total_score / denom)
