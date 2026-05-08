from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class PatchGrid:
    """Spectrogram patch layout used by the split MAE client and server."""

    freq: int
    time: int
    patch_freq: int = 16
    patch_time: int = 16

    @property
    def num_patches(self) -> int:
        return int(self.freq * self.time)

    @property
    def patch_dim(self) -> int:
        return int(self.patch_freq * self.patch_time)

    @property
    def padded_fdim(self) -> int:
        return int(self.freq * self.patch_freq)

    @property
    def padded_tdim(self) -> int:
        return int(self.time * self.patch_time)


@dataclass
class SmashedData:
    """Payload passed from split-learning client to server.

    `tokens` is the only tensor expected to carry gradients back to the client.
    The remaining tensors are detached metadata needed by the server for Stage 1
    MAE reconstruction or Stage 2 classification.
    """

    tokens: Tensor
    mode: str
    cls_token_count: int
    patch_grid: PatchGrid
    ids_keep: Optional[Tensor] = None
    ids_restore: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    target_patches: Optional[Tensor] = None
    static_features: Optional[Tensor] = None

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "SmashedData":
        kwargs = {}
        for name in ("ids_keep", "ids_restore", "mask", "target_patches", "static_features"):
            value = getattr(self, name)
            kwargs[name] = None if value is None else value.to(device, non_blocking=non_blocking)
        return replace(
            self,
            tokens=self.tokens.to(device, non_blocking=non_blocking),
            **kwargs,
        )

    def detach_for_server(self, requires_grad: bool = True) -> "SmashedData":
        kwargs = {}
        for name in ("ids_keep", "ids_restore", "mask", "target_patches", "static_features"):
            value = getattr(self, name)
            kwargs[name] = None if value is None else value.detach()
        tokens = self.tokens.detach().requires_grad_(requires_grad)
        return replace(self, tokens=tokens, **kwargs)


def padded_dim(size: int, patch: int) -> int:
    if size <= 0 or patch <= 0:
        raise ValueError(f"Expected positive size and patch, got {size=} {patch=}")
    return int(((size + patch - 1) // patch) * patch)


def make_patch_grid(input_fdim: int, input_tdim: int, patch_size: tuple[int, int]) -> PatchGrid:
    pf, pt = int(patch_size[0]), int(patch_size[1])
    padded_f = padded_dim(int(input_fdim), pf)
    padded_t = padded_dim(int(input_tdim), pt)
    return PatchGrid(freq=padded_f // pf, time=padded_t // pt, patch_freq=pf, patch_time=pt)


def ensure_b1ft(x: Tensor, input_fdim: Optional[int] = None) -> Tensor:
    """Convert common repo/audio tensor layouts to (B, 1, F, T)."""

    if x.ndim == 4:
        if x.shape[1] == 1:
            return x.contiguous()
        if x.shape[-1] == 1:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Expected 4D spectrogram with singleton channel, got {tuple(x.shape)}")

    if x.ndim != 3:
        raise ValueError(f"Expected 3D or 4D spectrogram tensor, got {tuple(x.shape)}")

    if input_fdim is not None and int(x.shape[1]) == int(input_fdim):
        return x.unsqueeze(1).contiguous()

    # The existing SsastMelDataset returns (B, T, F), matching ASTModel.forward.
    return x.unsqueeze(1).transpose(2, 3).contiguous()


def crop_or_pad_b1ft(x: Tensor, input_fdim: int, input_tdim: int) -> Tensor:
    """Crop/pad to the configured input size before patch-grid padding."""

    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected (B,1,F,T), got {tuple(x.shape)}")

    b, c, f, t = x.shape
    target_f, target_t = int(input_fdim), int(input_tdim)
    x = x[:, :, :target_f, :target_t]
    pad_f = max(target_f - int(x.shape[2]), 0)
    pad_t = max(target_t - int(x.shape[3]), 0)
    if pad_f or pad_t:
        x = torch.nn.functional.pad(x, (0, pad_t, 0, pad_f))
    return x.contiguous()


def pad_to_patch_grid(x: Tensor, grid: PatchGrid) -> Tensor:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected (B,1,F,T), got {tuple(x.shape)}")
    x = x[:, :, : grid.padded_fdim, : grid.padded_tdim]
    pad_f = max(grid.padded_fdim - int(x.shape[2]), 0)
    pad_t = max(grid.padded_tdim - int(x.shape[3]), 0)
    if pad_f or pad_t:
        x = torch.nn.functional.pad(x, (0, pad_t, 0, pad_f))
    return x.contiguous()


def patchify_b1ft(x: Tensor, grid: PatchGrid) -> Tensor:
    """Patchify (B,1,F,T) into (B,N,patch_freq*patch_time)."""

    b, c, f, t = x.shape
    if c != 1:
        raise ValueError(f"Expected one channel, got {c}")
    if f != grid.padded_fdim or t != grid.padded_tdim:
        raise ValueError(
            f"Input shape {(f, t)} does not match patch grid "
            f"{(grid.padded_fdim, grid.padded_tdim)}"
        )
    pf, pt = grid.patch_freq, grid.patch_time
    x = x.reshape(b, c, grid.freq, pf, grid.time, pt)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.reshape(b, grid.num_patches, pf * pt)


def unpatchify_b1ft(patches: Tensor, grid: PatchGrid) -> Tensor:
    """Reverse `patchify_b1ft`, returning (B,1,F,T)."""

    b, n, patch_dim = patches.shape
    if n != grid.num_patches or patch_dim != grid.patch_dim:
        raise ValueError(
            f"Patch shape {(n, patch_dim)} does not match grid "
            f"{(grid.num_patches, grid.patch_dim)}"
        )
    pf, pt = grid.patch_freq, grid.patch_time
    x = patches.reshape(b, grid.freq, grid.time, pf, pt, 1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.reshape(b, 1, grid.padded_fdim, grid.padded_tdim)


def patchwise_normalize(patches: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize each patch independently over its flattened bins."""

    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + float(eps))
    return (patches - mean) / std, mean, std


def gather_tokens(tokens: Tensor, ids: Tensor) -> Tensor:
    if tokens.ndim != 3 or ids.ndim != 2:
        raise ValueError(f"Expected tokens (B,N,D) and ids (B,K), got {tokens.shape} {ids.shape}")
    return torch.gather(tokens, dim=1, index=ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def _mask_from_mask_ids(mask_ids: Tensor, num_patches: int) -> Tensor:
    mask = torch.zeros(mask_ids.shape[0], int(num_patches), device=mask_ids.device, dtype=torch.float32)
    return mask.scatter(1, mask_ids.long(), 1.0)


def _ids_from_mask(mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    ids_keep: list[Tensor] = []
    ids_mask: list[Tensor] = []
    for row in mask:
        keep = torch.nonzero(row == 0, as_tuple=False).flatten()
        masked = torch.nonzero(row == 1, as_tuple=False).flatten()
        ids_keep.append(keep)
        ids_mask.append(masked)
    keep_t = torch.stack(ids_keep, dim=0).long()
    mask_t = torch.stack(ids_mask, dim=0).long()
    ids_shuffle = torch.cat([keep_t, mask_t], dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    return keep_t, mask_t, ids_restore


def content_aware_mask(
    patches: Tensor,
    mask_ratio: float = 0.75,
) -> tuple[Tensor, Tensor, Tensor]:
    """Paper-style content-aware masking based on per-patch variance.

    The method ranks patches by variance, builds a high-information candidate
    pool of size `max(70% * N_mask, 50% * N_total)`, samples 70% of masked
    patches from that pool, and samples the remaining 30% from the rest.
    """

    if patches.ndim != 3:
        raise ValueError(f"Expected patches (B,N,P), got {tuple(patches.shape)}")
    b, n, _ = patches.shape
    n_mask = int(round(float(mask_ratio) * int(n)))
    n_mask = min(max(n_mask, 1), int(n) - 1)
    variance = patches.var(dim=-1, unbiased=False).clamp_min(0.0)
    high_quota = min(max(int(round(0.7 * n_mask)), 1), n_mask)
    high_pool_size = min(max(int(round(0.7 * n_mask)), int(round(0.5 * n))), n)
    mask_ids = []
    for i in range(b):
        ranked = torch.argsort(variance[i], descending=True)
        high_pool = ranked[:high_pool_size]
        low_pool = ranked[high_pool_size:]
        high_pick = high_pool[torch.randperm(high_pool.numel(), device=patches.device)[:high_quota]]
        low_quota = n_mask - int(high_pick.numel())
        if low_quota > 0 and low_pool.numel() > 0:
            low_pick = low_pool[torch.randperm(low_pool.numel(), device=patches.device)[:low_quota]]
        else:
            low_pick = high_pool.new_empty(0)
        if high_pick.numel() + low_pick.numel() < n_mask:
            chosen = torch.cat([high_pick, low_pick], dim=0)
            remaining_mask = torch.ones(n, device=patches.device, dtype=torch.bool)
            remaining_mask[chosen] = False
            remaining = torch.nonzero(remaining_mask, as_tuple=False).flatten()
            extra = remaining[torch.randperm(remaining.numel(), device=patches.device)[: n_mask - chosen.numel()]]
            chosen = torch.cat([chosen, extra], dim=0)
        else:
            chosen = torch.cat([high_pick, low_pick], dim=0)
        mask_ids.append(chosen[:n_mask])
    mask_id_t = torch.stack(mask_ids, dim=0)
    mask = _mask_from_mask_ids(mask_id_t, n)
    ids_keep, _, ids_restore = _ids_from_mask(mask)
    return ids_keep, ids_restore, mask


def random_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Tensor]:
    n = int(num_patches)
    n_keep = n - min(max(int(round(float(mask_ratio) * n)), 1), n - 1)
    noise = torch.rand(int(batch_size), n, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    mask = torch.ones(int(batch_size), n, device=device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    ids_keep, _, ids_restore_sorted = _ids_from_mask(mask)
    return ids_keep.long(), ids_restore_sorted.long(), mask


def make_mask(
    patches: Tensor,
    mask_ratio: float,
    strategy: str = "content",
) -> tuple[Tensor, Tensor, Tensor]:
    strategy = strategy.lower().strip()
    if strategy in {"content", "content-aware", "content_aware"}:
        ids_keep, ids_restore, mask = content_aware_mask(patches, mask_ratio=mask_ratio)
        return ids_keep.long(), ids_restore.long(), mask
    if strategy in {"random", "mae"}:
        return random_mask(patches.shape[0], patches.shape[1], mask_ratio, patches.device)
    raise ValueError(f"Unknown mask strategy: {strategy}")


def ma_error_loss(pred_patches: Tensor, target_patches: Tensor, mask: Tensor) -> Tensor:
    """Masked absolute reconstruction error over normalized spectrogram patches."""

    if pred_patches.shape != target_patches.shape:
        raise ValueError(f"Prediction and target shape differ: {pred_patches.shape} vs {target_patches.shape}")
    if mask.shape != pred_patches.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} incompatible with predictions {pred_patches.shape}")
    patch_l1 = torch.abs(pred_patches - target_patches).mean(dim=-1)
    denom = mask.sum().clamp_min(1.0)
    return (patch_l1 * mask).sum() / denom


def average_state_dicts(state_dicts: Iterable[dict[str, Tensor]]) -> dict[str, Tensor]:
    state_dicts = list(state_dicts)
    if not state_dicts:
        raise ValueError("Cannot average an empty state_dict list.")
    avg: dict[str, Tensor] = {}
    for key in state_dicts[0].keys():
        values = [sd[key] for sd in state_dicts]
        if not torch.is_floating_point(values[0]):
            avg[key] = values[0].clone()
            continue
        avg[key] = torch.stack([v.detach().float() for v in values], dim=0).mean(dim=0).to(values[0].dtype)
    return avg


def labels_for_bce(labels: Tensor, num_labels: int) -> Tensor:
    if labels.ndim == 2 and labels.shape[1] == int(num_labels):
        return labels.float()
    if int(num_labels) == 1:
        return labels.float().view(-1, 1)
    return torch.nn.functional.one_hot(labels.long().view(-1), num_classes=int(num_labels)).float()
