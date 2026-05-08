from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_training_run(
    model: nn.Module,
    payload: dict[str, Any],
    *,
    save_dir: Path,
    model_stem: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = model_stem
    torch.save(model.state_dict(), save_dir / f"{stem}.pt")
    meta_path = save_dir / f"{stem}_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint: {save_dir / f'{stem}.pt'}; metadata: {meta_path}")
    return save_dir / f"{stem}.pt"


def load_state_dict(path: Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_training_run(
    model: nn.Module,
    stem: str,
    *,
    save_dir: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    sd = load_state_dict(save_dir / f"{stem}.pt", map_location=map_location)
    model.load_state_dict(sd, strict=True)
    meta = json.loads((save_dir / f"{stem}_metadata.json").read_text(encoding="utf-8"))
    print(f"Loaded checkpoint: {save_dir / f'{stem}.pt'}")
    return meta
