from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import to_2tuple, trunc_normal_


class RelaxedPatchEmbed(nn.Module):
    """Patch embedding compatible with non-224/384 spectrogram inputs."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


@dataclass(frozen=True)
class StandardASTConfig:
    input_fdim: int = 128
    input_tdim: int = 256
    patch_size: tuple[int, int] = (16, 16)
    model_size: str = "base384"
    imagenet_pretrain: bool = False
    audioset_checkpoint_path: Optional[str] = None


def resolve_ast_model_name(model_size: str) -> str:
    aliases = {
        "tiny": "tiny224",
        "small": "small224",
        "base": "base384",
        "base_nokd": "base384",
    }
    name = aliases.get(model_size.lower().strip(), model_size.lower().strip())
    valid = {"tiny224", "small224", "base224", "base384"}
    if name not in valid:
        raise ValueError(f"model_size must be one of {sorted(valid)} or aliases {sorted(aliases)}, got {model_size}")
    return name


def _timm_model_name(model_size: str) -> str:
    name = resolve_ast_model_name(model_size)
    if name == "tiny224":
        return "vit_deit_tiny_distilled_patch16_224"
    if name == "small224":
        return "vit_deit_small_distilled_patch16_224"
    if name == "base224":
        return "vit_deit_base_distilled_patch16_224"
    return "vit_deit_base_distilled_patch16_384"


def _conv_hw(input_fdim: int, input_tdim: int, patch_size: tuple[int, int]) -> tuple[int, int]:
    pf, pt = int(patch_size[0]), int(patch_size[1])
    if int(input_fdim) % pf or int(input_tdim) % pt:
        raise ValueError(
            "Paper 2601 Stage 1 uses non-overlapping 16x16 patches; "
            f"input_fdim/input_tdim must be divisible by patch_size, got "
            f"{input_fdim=} {input_tdim=} {patch_size=}"
        )
    return int(input_fdim) // pf, int(input_tdim) // pt


def _resize_pretrained_pos_embed(
    pos_embed: torch.Tensor,
    *,
    original_num_patches: int,
    target_hw: tuple[int, int],
    embed_dim: int,
    cls_token_count: int,
) -> torch.Tensor:
    original_hw = int(original_num_patches**0.5)
    patch_pos = pos_embed[:, cls_token_count:, :]
    patch_pos = patch_pos.reshape(1, original_num_patches, embed_dim).transpose(1, 2)
    patch_pos = patch_pos.reshape(1, embed_dim, original_hw, original_hw)
    target_f, target_t = int(target_hw[0]), int(target_hw[1])

    if target_t <= original_hw:
        start = int(original_hw / 2) - int(target_t / 2)
        patch_pos = patch_pos[:, :, :, start : start + target_t]
    else:
        patch_pos = F.interpolate(patch_pos, size=(original_hw, target_t), mode="bilinear")

    if target_f <= original_hw:
        start = int(original_hw / 2) - int(target_f / 2)
        patch_pos = patch_pos[:, :, start : start + target_f, :]
    else:
        patch_pos = F.interpolate(patch_pos, size=(target_f, target_t), mode="bilinear")

    patch_pos = patch_pos.reshape(1, embed_dim, target_f * target_t).transpose(1, 2)
    return torch.cat([pos_embed[:, :cls_token_count, :].detach(), patch_pos], dim=1)


def _load_official_ast_checkpoint(vit: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"AST checkpoint not found: {path}")
    sd = torch.load(str(path), map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    cleaned = {}
    for key, value in sd.items():
        k = key
        for prefix in ("module.v.", "v.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        if k.startswith(("head.", "head_dist.", "mlp_head.")):
            continue
        if k in vit.state_dict() and vit.state_dict()[k].shape == value.shape:
            cleaned[k] = value
    vit.load_state_dict(cleaned, strict=False)


class StandardASTBackbone(nn.Module):
    """Standard AST encoder backbone, matching Gong et al. AST rather than SSAST."""

    cls_token_count: int = 2

    def __init__(self, config: StandardASTConfig = StandardASTConfig()) -> None:
        super().__init__()
        if timm.__version__ != "0.4.5":
            raise RuntimeError(f"Standard AST reference expects timm==0.4.5, got {timm.__version__}")

        self.config = config
        timm.models.vision_transformer.PatchEmbed = RelaxedPatchEmbed
        self.v = timm.create_model(
            _timm_model_name(config.model_size),
            pretrained=bool(config.imagenet_pretrain),
        )
        self.original_num_patches = int(self.v.patch_embed.num_patches)
        self.embed_dim = int(self.v.pos_embed.shape[-1])
        self.patch_hw = _conv_hw(config.input_fdim, config.input_tdim, config.patch_size)
        self.num_patches = int(self.patch_hw[0] * self.patch_hw[1])
        self._adapt_patch_embedding()
        self._adapt_positional_embedding()
        if config.audioset_checkpoint_path:
            _load_official_ast_checkpoint(self.v, config.audioset_checkpoint_path)

    def _adapt_patch_embedding(self) -> None:
        pf, pt = self.config.patch_size
        new_proj = nn.Conv2d(1, self.embed_dim, kernel_size=(pf, pt), stride=(pf, pt))
        if self.config.imagenet_pretrain:
            old_proj = self.v.patch_embed.proj
            new_proj.weight = nn.Parameter(old_proj.weight.sum(dim=1, keepdim=True).detach().clone())
            new_proj.bias = old_proj.bias
        self.v.patch_embed.proj = new_proj
        self.v.patch_embed.num_patches = self.num_patches

    def _adapt_positional_embedding(self) -> None:
        if self.config.imagenet_pretrain:
            pos = _resize_pretrained_pos_embed(
                self.v.pos_embed,
                original_num_patches=self.original_num_patches,
                target_hw=self.patch_hw,
                embed_dim=self.embed_dim,
                cls_token_count=self.cls_token_count,
            )
            self.v.pos_embed = nn.Parameter(pos)
            return
        pos = nn.Parameter(torch.zeros(1, self.num_patches + self.cls_token_count, self.embed_dim))
        trunc_normal_(pos, std=0.02)
        self.v.pos_embed = pos


def build_standard_ast_backbone(
    *,
    input_fdim: int = 128,
    input_tdim: int = 256,
    patch_size: tuple[int, int] = (16, 16),
    model_size: str = "base384",
    imagenet_pretrain: bool = False,
    audioset_checkpoint_path: Optional[str] = None,
) -> StandardASTBackbone:
    return StandardASTBackbone(
        StandardASTConfig(
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            patch_size=patch_size,
            model_size=model_size,
            imagenet_pretrain=imagenet_pretrain,
            audioset_checkpoint_path=audioset_checkpoint_path,
        )
    )
