#!/usr/bin/env python3
"""Centralized AST (SSAST) finetuning on one vowel — same data CLI as scripts/train.py (no SpecAugment)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _path_setup import ensure_project_package_on_path, find_project_root

ensure_project_package_on_path()

from voice_disorder_torch.config import (
    DEFAULT_SSAST_MODEL_SIZE,
    DataPaths,
    RunContext,
    SplitSeeds,
    TrainConfig,
    apply_default_project_paths,
    default_ssast_checkpoint_path,
)
from voice_disorder_torch.data.datasets import SsastMelDataset
from voice_disorder_torch.data.load import load_all_preprocessed
from voice_disorder_torch.models.ssast_ast import ASTModel
from voice_disorder_torch.naming import generate_model_name
from voice_disorder_torch.reproducibility import set_reproducible


def main() -> None:
    _defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Centralized AST finetuning (single vowel).")
    p.add_argument("--pickle-dir", type=Path, default=None)
    p.add_argument("--pickle-dir-eent", type=Path, default=None)
    p.add_argument("--pickle-dir-svd", type=Path, default=None)
    p.add_argument("--eent-subjects-xlsx", type=Path, default=None)
    p.add_argument("--german-subjects-xlsx", type=Path, default=None)
    p.add_argument("--save-dir", type=Path, default=Path("./saved_models"))
    p.add_argument("--vowel", choices=["a", "i"], required=True)
    p.add_argument("--dev-test-seed", type=int, default=8)
    p.add_argument("--train-val-seed", type=int, default=100)
    p.add_argument("--model-init-seed", type=int, default=2718)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=int(_defaults.max_epochs))
    p.add_argument("--batch-size", type=int, default=None, help="Default: TrainConfig.batch_size.")
    p.add_argument("--lr", type=float, default=float(_defaults.lr))
    p.add_argument("--input-tdim", type=int, default=259)
    p.add_argument("--input-fdim", type=int, default=128)
    args = p.parse_args()
    apply_default_project_paths(args)

    if (args.pickle_dir_eent is None) ^ (args.pickle_dir_svd is None):
        p.error("Provide both --pickle-dir-eent and --pickle-dir-svd, or neither (then use --pickle-dir).")
    if args.pickle_dir is None and (args.pickle_dir_eent is None or args.pickle_dir_svd is None):
        p.error("Provide --pickle-dir, or both --pickle-dir-eent and --pickle-dir-svd.")

    paths = DataPaths(
        pickle_dir=args.pickle_dir,
        pickle_dir_chinese=args.pickle_dir_eent,
        pickle_dir_german=args.pickle_dir_svd,
        german_subjects_xlsx=args.german_subjects_xlsx,
        eent_subjects_xlsx=args.eent_subjects_xlsx,
    )
    train_cfg = TrainConfig()
    if args.batch_size is not None:
        train_cfg = replace(train_cfg, batch_size=int(args.batch_size))
    ctx = RunContext(
        paths=paths,
        splits=SplitSeeds(dev_test_seed=args.dev_test_seed, train_val_seed=args.train_val_seed),
        train=train_cfg,
        save_dir=args.save_dir,
        device=args.device,
    )
    set_reproducible(int(args.model_init_seed))
    device = torch.device(ctx.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    bundle = load_all_preprocessed(ctx.paths, ctx.splits, verbose=True)
    if args.vowel == "a":
        x_tr, y_tr, x_va, y_va = bundle.x_train_a, bundle.y_train_a, bundle.x_val_a, bundle.y_val_a
    else:
        x_tr, y_tr, x_va, y_va = bundle.x_train_i, bundle.y_train_i, bundle.x_val_i, bundle.y_val_i

    train_ds = SsastMelDataset(x_tr, y_tr, input_tdim=args.input_tdim)
    val_ds = SsastMelDataset(x_va, y_va, input_tdim=args.input_tdim)
    train_ldr = DataLoader(train_ds, batch_size=ctx.train.batch_size, shuffle=True, num_workers=0)
    val_ldr = DataLoader(val_ds, batch_size=ctx.train.batch_size, shuffle=False, num_workers=0)

    pt = default_ssast_checkpoint_path(DEFAULT_SSAST_MODEL_SIZE, find_project_root())
    if not pt.is_file():
        p.error(f"Pretrained checkpoint not found: {pt}")

    model = ASTModel(
        label_dim=2,
        fshape=16,
        tshape=16,
        fstride=10,
        tstride=10,
        input_fdim=args.input_fdim,
        input_tdim=args.input_tdim,
        model_size=DEFAULT_SSAST_MODEL_SIZE,
        pretrain_stage=False,
        load_pretrained_mdl_path=str(pt.resolve()),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    stem = generate_model_name(
        "ast",
        args.vowel,
        ctx.splits.dev_test_seed,
        ctx.splits.train_val_seed,
        int(args.model_init_seed),
    )
    ctx.save_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_acc = -1.0

    for epoch in range(int(args.epochs)):
        model.train()
        tr_loss = 0.0
        n = 0
        for xb, yb in train_ldr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb, task="ft_avgtok")
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * yb.size(0)
            n += yb.size(0)
        tr_loss /= max(n, 1)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_ldr:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb, task="ft_avgtok")
                pred = out.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += yb.size(0)
        va = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} train_loss={tr_loss:.4f} val_acc={va:.4f}")
        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    out_pt = ctx.save_dir / f"{stem}.pt"
    torch.save(model.state_dict(), out_pt)
    n, h, w, c = x_tr.shape
    meta = {
        "framework": "pytorch",
        "model_type": "ast",
        "vowel": args.vowel,
        "model_name": stem,
        "seeds": {
            "dev_test_seed": ctx.splits.dev_test_seed,
            "train_val_seed": ctx.splits.train_val_seed,
            "model_init_seed": int(args.model_init_seed),
        },
        "pretrained_path": str(pt.resolve()),
        "input_tdim": args.input_tdim,
        "input_fdim": args.input_fdim,
        "input_shape_nhwc": [int(n), int(h), int(w), int(c)],
        "best_val_acc": float(best_acc),
    }
    (ctx.save_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_pt}")


if __name__ == "__main__":
    main()
