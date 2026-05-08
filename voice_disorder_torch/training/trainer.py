from __future__ import annotations

from dataclasses import dataclass, field

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..ui.status import _supports_ansi
@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def fit_binary_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    dataloader_seed: int,
) -> TrainHistory:
    """Adam + BCEWithLogits; early stopping on val_loss (like Keras)."""
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    history = TrainHistory()
    best_val = float("inf")
    best_state: dict | None = None
    patience_left = cfg.early_stopping_patience

    # Materialize LazyLinear (and any lazy modules) before constructing the optimizer.
    model.train()
    torch.manual_seed(int(dataloader_seed))
    probe = next(iter(train_loader))[0][:1].to(device)
    _ = model(probe)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    live = _supports_ansi(sys.stderr)  # best-effort: live refresh only on real terminal
    line_width = 0
    epoch_w = max(1, len(str(int(cfg.max_epochs))))
    finalized_line = False

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        n_seen = 0
        n_correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
            preds = (logits.detach() >= 0).to(yb.dtype)
            n_correct += int((preds == yb).sum().item())
        tr_loss = running / max(n_seen, 1)
        tr_acc = float(n_correct) / float(max(n_seen, 1))
        history.train_loss.append(tr_loss)

        model.eval()
        v_running = 0.0
        v_n = 0
        v_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                v_running += float(loss.item()) * xb.size(0)
                v_n += xb.size(0)
                preds = (logits.detach() >= 0).to(yb.dtype)
                v_correct += int((preds == yb).sum().item())
        va_loss = v_running / max(v_n, 1)
        va_acc = float(v_correct) / float(max(v_n, 1))
        history.val_loss.append(va_loss)

        if va_loss + 1e-12 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stopping_patience
        else:
            patience_left -= 1
        msg = (
            f"epoch: {epoch + 1:{epoch_w}d}/{int(cfg.max_epochs)}  "
            f"train_loss: {tr_loss:.5f}  "
            f"train_acc: {tr_acc:.4f}  "
            f"val_loss: {va_loss:.5f}  "
            f"val_acc: {va_acc:.4f}  "
            f"patience: {int(patience_left)}"
        )
        if live:
            pad = max(0, line_width - len(msg))
            sys.stderr.write("\r" + msg + (" " * pad))
            sys.stderr.flush()
            line_width = max(line_width, len(msg))
        else:
            print(msg)

        if patience_left <= 0:
            if live:
                sys.stderr.write("\n")
                sys.stderr.flush()
                finalized_line = True
                # yellow stop marker (not a check)
                sys.stderr.write("\x1b[33m!\x1b[0m early_stop\n")
                sys.stderr.flush()
            else:
                print("! early_stop")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if live and not finalized_line:
        sys.stderr.write("\n")
        sys.stderr.flush()
    return history


def make_loaders(
    train_ds,
    val_ds,
    cfg: TrainConfig,
    *,
    dataloader_seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    from ..reproducibility import dataloader_generator

    g = dataloader_generator(dataloader_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
