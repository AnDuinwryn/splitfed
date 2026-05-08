from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class BinaryClassifier(Protocol):
    """Contract for mel-spectrogram binary classifiers (logits or prob last layer)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
