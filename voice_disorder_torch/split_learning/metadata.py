"""Checkpoint metadata helpers for split-learning artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_split_run(
    *,
    save_dir: Path,
    stem: str,
    client_sd: dict,
    server_sd: dict,
    payload: dict,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    client_path = save_dir / f"{stem}_split_client.pt"
    server_path = save_dir / f"{stem}_split_server.pt"
    torch.save(client_sd, client_path)
    torch.save(server_sd, server_path)
    metadata_path = save_dir / f"{stem}_split_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved split client: {client_path}")
    print(f"Saved split server: {server_path}")
    print(f"Saved metadata: {metadata_path}")


def load_split_metadata(save_dir: Path, stem: str) -> dict:
    metadata_path = save_dir / f"{stem}_split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))
