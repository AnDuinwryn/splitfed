from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from paper2601_standard_ast import StandardASTConfig, StandardASTBackbone
from paper2601_splitmae_utils import (
    PatchGrid,
    SmashedData,
    crop_or_pad_b1ft,
    ensure_b1ft,
    gather_tokens,
    make_mask,
    make_patch_grid,
    pad_to_patch_grid,
    patchify_b1ft,
    patchwise_normalize,
    unpatchify_b1ft,
)


@dataclass(frozen=True)
class SplitMAEClientConfig:
    input_fdim: int = 128
    input_tdim: int = 259
    patch_size: tuple[int, int] = (16, 16)
    model_size: str = "base384"
    n_client_blocks: int = 2
    mask_ratio: float = 0.75
    mask_strategy: str = "content"
    normalize_input: bool = True
    imagenet_pretrain: bool = False
    audioset_checkpoint_path: Optional[str] = None


class Paper2601SplitMAEClient(nn.Module):
    """Client-side split AST for Stage 1 MAE and Stage 2 classification.

    Responsibilities:
    - accepts the repository's SSAST tensor shape `(B,T,F)`;
    - pads/crops to the configured AST input grid;
    - applies patch-wise normalization;
    - performs content-aware masking for Stage 1;
    - runs patch embedding and the first `n_client_blocks` AST encoder blocks;
    - returns a `SmashedData` payload for the server.
    """

    def __init__(self, config: SplitMAEClientConfig = SplitMAEClientConfig()) -> None:
        super().__init__()
        self.config = config
        self.patch_grid = make_patch_grid(config.input_fdim, config.input_tdim, config.patch_size)

        ast = StandardASTBackbone(
            StandardASTConfig(
                input_fdim=self.patch_grid.padded_fdim,
                input_tdim=self.patch_grid.padded_tdim,
                patch_size=config.patch_size,
                model_size=config.model_size,
                imagenet_pretrain=config.imagenet_pretrain,
                audioset_checkpoint_path=config.audioset_checkpoint_path,
            )
        )

        if config.n_client_blocks < 0 or config.n_client_blocks > len(ast.v.blocks):
            raise ValueError(f"n_client_blocks must be in [0, {len(ast.v.blocks)}], got {config.n_client_blocks}")

        self.v = ast.v
        self.cls_token_count = int(ast.cls_token_count)
        self.embed_dim = int(ast.embed_dim)
        self.client_blocks = nn.ModuleList(self.v.blocks[: config.n_client_blocks])

        # Keep only client-side ownership in this isolated module.
        del self.v.blocks
        if hasattr(self.v, "norm"):
            del self.v.norm
        for attr in ("head", "head_dist"):
            if hasattr(self.v, attr):
                delattr(self.v, attr)

    def _canonical_input(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_b1ft(x, input_fdim=self.config.input_fdim)
        x = crop_or_pad_b1ft(x, self.config.input_fdim, self.config.input_tdim)
        return pad_to_patch_grid(x, self.patch_grid)

    def _patchwise_model_input(self, x_b1ft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_patches = patchify_b1ft(x_b1ft, self.patch_grid)
        norm_patches, _, _ = patchwise_normalize(raw_patches)
        if self.config.normalize_input:
            x_model = unpatchify_b1ft(norm_patches, self.patch_grid)
        else:
            x_model = x_b1ft
        return x_model, norm_patches

    def _prepend_special_tokens(self, patch_tokens: torch.Tensor, patch_pos: torch.Tensor) -> torch.Tensor:
        b = patch_tokens.shape[0]
        cls_pos = self.v.pos_embed[:, : self.cls_token_count, :].expand(b, -1, -1)
        if self.cls_token_count == 2:
            cls_tokens = self.v.cls_token.expand(b, -1, -1)
            dist_token = self.v.dist_token.expand(b, -1, -1)
            special = torch.cat([cls_tokens, dist_token], dim=1)
        else:
            special = self.v.cls_token.expand(b, -1, -1)
        return torch.cat([special + cls_pos, patch_tokens + patch_pos], dim=1)

    def _run_client_blocks(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.v.pos_drop(tokens)
        for block in self.client_blocks:
            tokens = block(tokens)
        return tokens

    def forward(
        self,
        x: torch.Tensor,
        *,
        mode: str = "pretrain",
        static_features: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
        mask_strategy: Optional[str] = None,
    ) -> SmashedData:
        mode = mode.lower().strip()
        if mode not in {"pretrain", "finetune", "stage1", "stage2"}:
            raise ValueError(f"mode must be pretrain/stage1 or finetune/stage2, got {mode!r}")

        stage = "pretrain" if mode in {"pretrain", "stage1"} else "finetune"
        x_b1ft = self._canonical_input(x)
        x_model, norm_patches = self._patchwise_model_input(x_b1ft)
        patch_tokens = self.v.patch_embed(x_model)
        b, n, _ = patch_tokens.shape
        if n != self.patch_grid.num_patches:
            raise RuntimeError(f"Patch embedding returned {n} patches, expected {self.patch_grid.num_patches}")

        patch_pos = self.v.pos_embed[:, self.cls_token_count :, :].expand(b, -1, -1)

        if stage == "pretrain":
            ids_keep, ids_restore, mask = make_mask(
                norm_patches.detach(),
                mask_ratio=self.config.mask_ratio if mask_ratio is None else float(mask_ratio),
                strategy=self.config.mask_strategy if mask_strategy is None else str(mask_strategy),
            )
            visible_tokens = gather_tokens(patch_tokens, ids_keep)
            visible_pos = gather_tokens(patch_pos, ids_keep)
            tokens = self._prepend_special_tokens(visible_tokens, visible_pos)
            tokens = self._run_client_blocks(tokens)
            return SmashedData(
                tokens=tokens,
                mode="pretrain",
                cls_token_count=self.cls_token_count,
                patch_grid=self.patch_grid,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
                mask=mask,
                target_patches=norm_patches.detach(),
                static_features=None,
            )

        tokens = self._prepend_special_tokens(patch_tokens, patch_pos)
        tokens = self._run_client_blocks(tokens)
        return SmashedData(
            tokens=tokens,
            mode="finetune",
            cls_token_count=self.cls_token_count,
            patch_grid=self.patch_grid,
            ids_keep=None,
            ids_restore=None,
            mask=None,
            target_patches=None,
            static_features=None if static_features is None else static_features.detach(),
        )


def build_paper2601_client(
    *,
    input_fdim: int = 128,
    input_tdim: int = 259,
    n_client_blocks: int = 2,
    model_size: str = "base",
    mask_ratio: float = 0.75,
    mask_strategy: str = "content",
    normalize_input: bool = True,
    imagenet_pretrain: bool = False,
    audioset_checkpoint_path: Optional[str] = None,
) -> Paper2601SplitMAEClient:
    return Paper2601SplitMAEClient(
        SplitMAEClientConfig(
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            n_client_blocks=n_client_blocks,
            model_size=model_size,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            normalize_input=normalize_input,
            imagenet_pretrain=imagenet_pretrain,
            audioset_checkpoint_path=audioset_checkpoint_path,
        )
    )
