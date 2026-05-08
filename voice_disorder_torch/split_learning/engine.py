"""Reusable split-learning training engine."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .aggregation import uniform_average_state_dicts

LabelPrep = Callable[[torch.Tensor], torch.Tensor]
AccuracyFn = Callable[[torch.Tensor, torch.Tensor], float]


class SplitServerPool:
    """One server-side model replica per client partition."""

    def __init__(
        self,
        *,
        server_template: nn.Module,
        n_partitions: int,
        server_lr: float,
        criterion: nn.Module,
        prepare_labels: LabelPrep,
        accuracy_fn: AccuracyFn,
        device: torch.device,
    ) -> None:
        self.server_models = [copy.deepcopy(server_template) for _ in range(n_partitions)]
        self.server_opts: list[torch.optim.Optimizer] = [
            torch.optim.Adam(model.parameters(), lr=server_lr) for model in self.server_models
        ]
        self.n_partitions = int(n_partitions)
        self.server_lr = float(server_lr)
        self.criterion = criterion
        self.prepare_labels = prepare_labels
        self.accuracy_fn = accuracy_fn
        self.device = device
        self.finished_partitions: list[int] = []
        self.acc_stack: list[float] = []
        self.loss_stack: list[float] = []
        self.last_partition_stats: dict[int, tuple[float, float]] = {}

    def step(
        self,
        partition_id: int,
        smashed: torch.Tensor,
        labels: torch.Tensor,
        is_last_epoch: bool,
        is_last_batch: bool,
    ) -> torch.Tensor:
        net = self.server_models[partition_id].to(self.device)
        smashed = smashed.to(self.device)
        labels = self.prepare_labels(labels.to(self.device))
        net.train()
        optimizer = self.server_opts[partition_id]
        optimizer.zero_grad(set_to_none=True)
        outputs = net(smashed)
        loss = self.criterion(outputs, labels)
        self.acc_stack.append(self.accuracy_fn(outputs, labels))
        self.loss_stack.append(loss.item())
        loss.backward()
        optimizer.step()
        smashed_grad = smashed.grad.clone().detach()
        if is_last_batch:
            self.last_partition_stats[int(partition_id)] = (
                float(np.mean(self.acc_stack)) if self.acc_stack else 0.0,
                float(np.mean(self.loss_stack)) if self.loss_stack else 0.0,
            )
            self.acc_stack, self.loss_stack = [], []
            if is_last_epoch:
                self.finished_partitions.append(partition_id)
                if len(self.finished_partitions) == self.n_partitions:
                    averaged = uniform_average_state_dicts([m.state_dict() for m in self.server_models])
                    for model in self.server_models:
                        model.load_state_dict(averaged)
                    self.server_opts = [
                        torch.optim.Adam(model.parameters(), lr=self.server_lr)
                        for model in self.server_models
                    ]
                    self.finished_partitions = []
        return smashed_grad

    def global_model(self) -> nn.Module:
        assert len(self.server_models) == self.n_partitions
        return copy.deepcopy(self.server_models[0])


def train_client_partition(
    *,
    client_model: nn.Module,
    train_loader: DataLoader,
    partition_id: int,
    n_local_epochs: int,
    client_lr: float,
    server_pool: SplitServerPool,
    device: torch.device,
) -> dict:
    client_model.train()
    optimizer = torch.optim.Adam(client_model.parameters(), lr=client_lr)
    n_batches = len(train_loader)
    for epoch_idx in range(n_local_epochs):
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            smashed = client_model(xb)
            detached = smashed.clone().detach().requires_grad_(True)
            smashed_grad = server_pool.step(
                partition_id,
                detached,
                yb,
                epoch_idx == n_local_epochs - 1,
                batch_idx == n_batches - 1,
            )
            smashed.backward(smashed_grad)
            optimizer.step()
    return client_model.state_dict()
