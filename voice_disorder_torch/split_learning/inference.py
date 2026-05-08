"""Inference helpers for split-learning checkpoint pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from voice_disorder_torch.config import TrainConfig
from voice_disorder_torch.data.datasets import MelSegmentDataset, SsastMelDataset, channels_last_to_nchw
from voice_disorder_torch.io.artifacts import load_state_dict

from .metadata import load_split_metadata
from .models.cnn import build_split_cnn_parts
from .models.ssast import SsastClientEncoder, SsastServerHead


def build_loaded_cnn_split(
    save_dir: Path,
    stem: str,
    x_sample_nhwc: np.ndarray,
    train_cfg: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    sample = channels_last_to_nchw(x_sample_nhwc[:1])
    client, server = build_split_cnn_parts(sample, train_cfg, init_seed=0)
    client.load_state_dict(load_state_dict(save_dir / f"{stem}_split_client.pt", map_location="cpu"))
    server.load_state_dict(load_state_dict(save_dir / f"{stem}_split_server.pt", map_location="cpu"))
    return client.to(device).eval(), server.to(device).eval()


def build_loaded_ssast_split(save_dir: Path, stem: str, device: torch.device) -> tuple[nn.Module, nn.Module]:
    meta = load_split_metadata(save_dir, stem)
    ss = meta["ssast"]
    pretrained_path = meta["pretrained_path"]
    client = SsastClientEncoder(
        label_dim=2,
        f_shape=ss["f_shape"],
        t_shape=ss["t_shape"],
        f_stride=ss["f_stride"],
        t_stride=ss["t_stride"],
        input_fdim=ss["input_fdim"],
        input_tdim=ss["input_tdim"],
        model_size=ss["model_size"],
        load_pretrained_mdl_path=pretrained_path,
        n_client_blocks=ss["n_client_blocks"],
    )
    server = SsastServerHead(
        label_dim=2,
        f_shape=ss["f_shape"],
        t_shape=ss["t_shape"],
        f_stride=ss["f_stride"],
        t_stride=ss["t_stride"],
        input_fdim=ss["input_fdim"],
        input_tdim=ss["input_tdim"],
        model_size=ss["model_size"],
        load_pretrained_mdl_path=pretrained_path,
        n_client_blocks=ss["n_client_blocks"],
    )
    client.load_state_dict(load_state_dict(save_dir / f"{stem}_split_client.pt", map_location="cpu"))
    server.load_state_dict(load_state_dict(save_dir / f"{stem}_split_server.pt", map_location="cpu"))
    return client.to(device).eval(), server.to(device).eval()


@torch.no_grad()
def predict_cnn_split_proba(
    client: nn.Module,
    server: nn.Module,
    x_nhwc: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dataset = MelSegmentDataset(x_nhwc, np.zeros((len(x_nhwc), 1), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = server(client(xb))
        chunks.append(torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def predict_ssast_split_proba(
    client: nn.Module,
    server: nn.Module,
    x_nhwc: np.ndarray,
    device: torch.device,
    batch_size: int,
    input_tdim: int,
) -> np.ndarray:
    dataset = SsastMelDataset(x_nhwc, np.zeros((len(x_nhwc), 1), dtype=np.float32), input_tdim=input_tdim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = server(client(xb))
        chunks.append(torch.softmax(logits, dim=1)[:, 1].detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(chunks, axis=0)
