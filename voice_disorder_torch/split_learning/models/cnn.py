"""CNN client/server parts for split learning."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from voice_disorder_torch.config import TrainConfig
from voice_disorder_torch.models.cnn2d import CNN2DOriginal, build_cnn2d_original, init_weights_deterministic


class CNNClientPart(nn.Module):
    """Early conv stack: through pool2 + dropout (first half of CNN2DOriginal)."""

    def __init__(self, template: CNN2DOriginal):
        super().__init__()
        self.conv1 = copy.deepcopy(template.conv1)
        self.pool1 = copy.deepcopy(template.pool1)
        self.conv2 = copy.deepcopy(template.conv2)
        self.pool2 = copy.deepcopy(template.pool2)
        self.drop1 = copy.deepcopy(template.drop1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop1(x)
        return x


class CNNServerPart(nn.Module):
    """Late conv + dense head (second half of CNN2DOriginal); outputs logits (N, 1)."""

    def __init__(self, template: CNN2DOriginal):
        super().__init__()
        self.conv3 = copy.deepcopy(template.conv3)
        self.pool3 = copy.deepcopy(template.pool3)
        self.conv4 = copy.deepcopy(template.conv4)
        self.pool4 = copy.deepcopy(template.pool4)
        self.drop2 = copy.deepcopy(template.drop2)
        self.fc1 = copy.deepcopy(template.fc1)
        self.fc2 = copy.deepcopy(template.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


def build_split_cnn_parts(
    sample_x_nchw: torch.Tensor, cfg: TrainConfig, init_seed: int
) -> tuple[CNNClientPart, CNNServerPart]:
    full = build_cnn2d_original(sample_x_nchw, cfg)
    full.eval()
    with torch.no_grad():
        _ = full(sample_x_nchw)
    init_weights_deterministic(full, seed=int(init_seed))
    return CNNClientPart(full), CNNServerPart(full)
