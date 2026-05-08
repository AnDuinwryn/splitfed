from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.datasets import MelSegmentDataset


@torch.no_grad()
def predict_positive_proba(
    model: nn.Module,
    x_nhwc: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return shape (N,) positive-class probabilities (sigmoid logits)."""
    model.eval()
    ds = MelSegmentDataset(x_nhwc, np.zeros((len(x_nhwc), 1), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)
        chunks.append(p)
    return np.concatenate(chunks, axis=0)
