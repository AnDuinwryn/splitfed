from __future__ import annotations

import torch
import torch.nn as nn

from ..config import TrainConfig


class CNN2DOriginal(nn.Module):
    """
    PyTorch port of `build_model_cnn2d_original` (Keras Sequential).
    Uses BCEWithLogitsLoss at training time — forward returns logits (1 channel).
    """

    def __init__(
        self,
        in_channels: int = 1,
        k_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        pool_size: tuple[int, int] = (2, 3),
        dropout: float = 0.5,
        dense_hidden: int = 128,
    ):
        super().__init__()
        p = (k_size[0] // 2, k_size[1] // 2)
        self.conv1 = nn.Conv2d(in_channels, 32, k_size, stride=stride, padding=p)
        self.pool1 = nn.MaxPool2d(pool_size, stride=pool_size, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 32, k_size, stride=stride, padding=p)
        self.pool2 = nn.MaxPool2d(pool_size, stride=pool_size, ceil_mode=True)
        self.drop1 = nn.Dropout(dropout / 2)
        self.conv3 = nn.Conv2d(32, 64, k_size, stride=stride, padding=p)
        self.pool3 = nn.MaxPool2d(pool_size, stride=pool_size, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 64, k_size, stride=stride, padding=p)
        self.pool4 = nn.MaxPool2d(pool_size, stride=pool_size, ceil_mode=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.LazyLinear(dense_hidden)
        self.fc2 = nn.Linear(dense_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop1(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


def build_cnn2d_original(sample_x_nchw: torch.Tensor, cfg: TrainConfig) -> CNN2DOriginal:
    _ = sample_x_nchw  # shape may vary; in_channels from C
    c = int(sample_x_nchw.shape[1])
    return CNN2DOriginal(
        in_channels=c,
        k_size=cfg.kernel_size,
        stride=cfg.conv_stride,
        pool_size=cfg.pool_size,
        dropout=cfg.dropout,
        dense_hidden=128,
    )


def init_weights_deterministic(module: nn.Module, seed: int) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def _fill(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            n = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            std = (2.0 / n) ** 0.5
            bound = (3.0**0.5) * std
            with torch.no_grad():
                m.weight.uniform_(-bound, bound, generator=gen)
                if m.bias is not None:
                    m.bias.zero_()
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            if fan_in == 0:
                return
            std = (2.0 / fan_in) ** 0.5
            bound = (3.0**0.5) * std
            with torch.no_grad():
                m.weight.uniform_(-bound, bound, generator=gen)
                if m.bias is not None:
                    m.bias.zero_()

    module.apply(_fill)
