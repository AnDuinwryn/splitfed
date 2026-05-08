from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


def channels_last_to_nchw(x: np.ndarray) -> torch.Tensor:
    """(N, H, W, C) float numpy -> (N, C, H, W) float tensor."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D (N,H,W,C), got {x.shape}")
    t = torch.from_numpy(x.astype(np.float32, copy=False))
    return t.permute(0, 3, 1, 2).contiguous()


def numpy_to_tensor_xy(
    x: np.ndarray,
    y: np.ndarray,
) -> TensorDataset:
    x_t = channels_last_to_nchw(x)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
    return TensorDataset(x_t, y_t)


class MelSegmentDataset(Dataset):
    """Thin wrapper if future augmentations are needed per segment."""

    def __init__(self, x_nhwc: np.ndarray, y: np.ndarray):
        self.x = channels_last_to_nchw(x_nhwc)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class SsastMelDataset(Dataset):
    """Mel segments (N,H,W,C) after preprocess -> (T, F) tensor for SSAST."""

    def __init__(self, x_nhwc: np.ndarray, y: np.ndarray, input_tdim: int):
        self.x = x_nhwc
        self.y = np.asarray(y).reshape(-1)
        self.input_tdim = int(input_tdim)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        # (H, W, C) = (freq, time, 1)
        feat = np.asarray(self.x[idx], dtype=np.float32)
        feat = feat[:, :, 0]
        t = feat.shape[1]
        if t > self.input_tdim:
            feat = feat[:, : self.input_tdim]
        elif t < self.input_tdim:
            feat = np.pad(feat, ((0, 0), (0, self.input_tdim - t)), mode="constant")
        feat_t = torch.from_numpy(feat.T.copy())
        return feat_t, torch.tensor(int(self.y[idx]), dtype=torch.long)
