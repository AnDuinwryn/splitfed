from __future__ import annotations

import torch
import torch.nn as nn

from ..config import TrainConfig
from .cnn2d import build_cnn2d_original, init_weights_deterministic


def build_trainable_backbone(
    model_type: str, sample_x_nchw: torch.Tensor, cfg: TrainConfig, init_seed: int
) -> nn.Module:
    """Entry point to swap CNN for another architecture later."""
    mt = model_type.lower().strip()
    if mt in {"cnn", "cnn2d", "cnn2d_original"}:
        model = build_cnn2d_original(sample_x_nchw, cfg)
        # LazyLinear has in_features=0 until first forward; custom init needs materialized shapes.
        model.eval()
        with torch.no_grad():
            _ = model(sample_x_nchw)
        init_weights_deterministic(model, seed=int(init_seed))
        return model
    raise ValueError(f"Unknown model_type: {model_type}")
