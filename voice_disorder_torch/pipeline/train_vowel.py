from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch

from ..config import DataPaths, RunContext, SplitSeeds, TrainConfig
from ..data.datasets import MelSegmentDataset
from ..data.load import load_all_preprocessed
from ..io.artifacts import save_training_run
from ..models.factory import build_trainable_backbone
from ..naming import generate_model_name
from ..reproducibility import set_reproducible
from ..training.trainer import fit_binary_classifier, make_loaders


def train_one_vowel(
    *,
    ctx: RunContext,
    vowel: Literal["a", "i"],
    model_type: str = "cnn",
    cnn_init_seed: int,
) -> dict:
    """Train a single-vowel binary classifier; saves `{name}.pt` and `{name}_metadata.json`."""
    set_reproducible(cnn_init_seed)
    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)

    if vowel == "a":
        x_tr, y_tr, x_va, y_va = bundle.x_train_a, bundle.y_train_a, bundle.x_val_a, bundle.y_val_a
    else:
        x_tr, y_tr, x_va, y_va = bundle.x_train_i, bundle.y_train_i, bundle.x_val_i, bundle.y_val_i

    device = torch.device(ctx.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = MelSegmentDataset(x_tr, y_tr)
    val_ds = MelSegmentDataset(x_va, y_va)
    sample_x = train_ds[0][0].unsqueeze(0)
    model = build_trainable_backbone(model_type, sample_x, ctx.train, init_seed=cnn_init_seed)
    train_loader, val_loader = make_loaders(train_ds, val_ds, ctx.train, dataloader_seed=cnn_init_seed)
    set_reproducible(cnn_init_seed)

    history = fit_binary_classifier(
        model,
        train_loader,
        val_loader,
        ctx.train,
        device,
        dataloader_seed=cnn_init_seed,
    )

    n, h, w, c = x_tr.shape
    name_kind = model_type if model_type else "cnn"
    model_name = generate_model_name(
        name_kind,
        vowel,
        ctx.splits.dev_test_seed,
        ctx.splits.train_val_seed,
        cnn_init_seed,
    )
    payload = {
        "framework": "pytorch",
        "model_type": model_type,
        "vowel": vowel,
        "model_name": model_name,
        "seeds": {
            "dev_test_seed": ctx.splits.dev_test_seed,
            "train_val_seed": ctx.splits.train_val_seed,
            "model_init_seed": cnn_init_seed,
        },
        "architecture": asdict(ctx.train),
        "input_shape_nhwc": [int(n), int(h), int(w), int(c)],
        "training_epochs": len(history.train_loss),
        "final_train_loss": float(history.train_loss[-1]) if history.train_loss else None,
        "final_val_loss": float(history.val_loss[-1]) if history.val_loss else None,
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
    }
    save_training_run(model.cpu(), payload, save_dir=ctx.save_dir, model_stem=model_name)
    return {"model_name": model_name, "history": history, "payload": payload}
