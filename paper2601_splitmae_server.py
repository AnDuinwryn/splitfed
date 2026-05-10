from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from paper2601_standard_ast import StandardASTConfig, StandardASTBackbone
from paper2601_splitmae_utils import (
    SmashedData,
    gather_tokens,
    ma_error_loss,
    make_patch_grid,
)


@dataclass(frozen=True)
class SplitMAEServerConfig:
    input_fdim: int = 128
    input_tdim: int = 259
    patch_size: tuple[int, int] = (16, 16)
    model_size: str = "base384"
    n_client_blocks: int = 2
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_labels: int = 1
    static_feature_dim: int = 0
    classifier_hidden: tuple[int, ...] = (512, 128)
    pooling: str = "cls"
    imagenet_pretrain: bool = False
    audioset_checkpoint_path: Optional[str] = None


class LightweightMAEDecoder(nn.Module):
    """MAE decoder operating entirely on the server side."""

    def __init__(
        self,
        *,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        decoder_mlp_ratio: float,
        patch_dim: int,
        num_patches: int,
        cls_token_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.cls_token_count = int(cls_token_count)
        self.decoder_embed = nn.Linear(int(encoder_embed_dim), int(decoder_embed_dim), bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(decoder_embed_dim)))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.cls_token_count, int(decoder_embed_dim))
        )
        ff_dim = int(float(decoder_mlp_ratio) * int(decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=int(decoder_embed_dim),
                    nhead=int(decoder_num_heads),
                    dim_feedforward=ff_dim,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(int(decoder_depth))
            ]
        )
        self.decoder_norm = nn.LayerNorm(int(decoder_embed_dim))
        self.decoder_pred = nn.Linear(int(decoder_embed_dim), int(patch_dim), bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.zeros_(self.decoder_embed.bias)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)

    def forward(self, encoded_visible: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        if ids_restore is None:
            raise ValueError("ids_restore is required for MAE decoding.")
        b = encoded_visible.shape[0]
        x = self.decoder_embed(encoded_visible)
        cls = x[:, : self.cls_token_count, :]
        visible = x[:, self.cls_token_count :, :]
        n_visible = visible.shape[1]
        n_mask = self.num_patches - n_visible
        if n_mask <= 0:
            raise ValueError(f"Expected at least one masked patch, got n_visible={n_visible}")
        mask_tokens = self.mask_token.expand(b, n_mask, -1)
        restored = gather_tokens(torch.cat([visible, mask_tokens], dim=1), ids_restore)
        x = torch.cat([cls, restored], dim=1)
        x = x + self.decoder_pos_embed[:, : x.shape[1], :]
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x[:, self.cls_token_count :, :])


class FeatureAttentionFFNN(nn.Module):
    """Feature-level attention followed by an FFNN multi-label head."""

    def __init__(
        self,
        *,
        audio_feature_dim: int,
        static_feature_dim: int,
        num_labels: int,
        hidden: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.audio_feature_dim = int(audio_feature_dim)
        self.static_feature_dim = int(static_feature_dim)
        self.input_dim = self.audio_feature_dim + self.static_feature_dim
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.input_dim, self.input_dim),
            nn.Sigmoid(),
        )

        layers: list[nn.Module] = []
        prev = self.input_dim
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev, int(width)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            prev = int(width)
        layers.append(nn.Linear(prev, int(num_labels)))
        self.ffnn = nn.Sequential(*layers)

    def forward(
        self,
        audio_features: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ):
        if self.static_feature_dim:
            if static_features is None:
                static_features = audio_features.new_zeros(audio_features.shape[0], self.static_feature_dim)
            static_features = static_features.float()
            if static_features.shape[-1] != self.static_feature_dim:
                raise ValueError(
                    f"Expected static features dim {self.static_feature_dim}, "
                    f"got {static_features.shape[-1]}"
                )
            features = torch.cat([audio_features, static_features.to(audio_features.device)], dim=-1)
        else:
            features = audio_features

        features = self.input_norm(features)
        attn = self.attention(features)
        logits = self.ffnn(features * attn)
        if return_attention:
            return logits, attn
        return logits


class Paper2601SplitMAEServer(nn.Module):
    """Server-side model for the paper-inspired split MAE/classifier pipeline."""

    def __init__(self, config: SplitMAEServerConfig = SplitMAEServerConfig()) -> None:
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

        self.cls_token_count = int(ast.cls_token_count)
        self.embed_dim = int(ast.embed_dim)
        self.server_blocks = nn.ModuleList(ast.v.blocks[config.n_client_blocks :])
        self.encoder_norm = ast.v.norm

        self.decoder = LightweightMAEDecoder(
            encoder_embed_dim=self.embed_dim,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            decoder_mlp_ratio=config.decoder_mlp_ratio,
            patch_dim=self.patch_grid.patch_dim,
            num_patches=self.patch_grid.num_patches,
            cls_token_count=self.cls_token_count,
            dropout=config.dropout,
        )
        self.classifier = FeatureAttentionFFNN(
            audio_feature_dim=self.embed_dim,
            static_feature_dim=config.static_feature_dim,
            num_labels=config.num_labels,
            hidden=config.classifier_hidden,
            dropout=config.dropout,
        )

    def encode(self, smashed: SmashedData | torch.Tensor) -> torch.Tensor:
        tokens = smashed.tokens if isinstance(smashed, SmashedData) else smashed
        for block in self.server_blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens)

    def pooled_audio_features(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        pooling = self.config.pooling.lower().strip()
        if pooling == "mean_patch":
            return encoded_tokens[:, self.cls_token_count :, :].mean(dim=1)
        if pooling == "cls":
            if self.cls_token_count == 2:
                return 0.5 * (encoded_tokens[:, 0, :] + encoded_tokens[:, 1, :])
            return encoded_tokens[:, 0, :]
        raise ValueError(f"Unknown pooling mode: {self.config.pooling}")

    def forward_pretrain(self, smashed: SmashedData) -> dict[str, torch.Tensor]:
        if smashed.ids_restore is None or smashed.mask is None or smashed.target_patches is None:
            raise ValueError("Stage 1 requires ids_restore, mask, and target_patches in SmashedData.")
        encoded = self.encode(smashed)
        pred = self.decoder(encoded, smashed.ids_restore)
        loss = ma_error_loss(pred, smashed.target_patches.to(pred.device), smashed.mask.to(pred.device))
        return {
            "loss": loss,
            "pred_patches": pred,
            "encoded_tokens": encoded,
        }

    def forward_finetune(
        self,
        smashed: SmashedData,
        *,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(smashed)
        audio_features = self.pooled_audio_features(encoded)
        out = self.classifier(
            audio_features,
            smashed.static_features,
            return_attention=return_attention,
        )
        if return_attention:
            logits, attention = out
            return {
                "logits": logits,
                "attention": attention,
                "audio_features": audio_features,
                "encoded_tokens": encoded,
            }
        return {
            "logits": out,
            "audio_features": audio_features,
            "encoded_tokens": encoded,
        }

    def forward(
        self,
        smashed: SmashedData,
        *,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        if smashed.mode == "pretrain":
            return self.forward_pretrain(smashed)
        if smashed.mode == "finetune":
            return self.forward_finetune(smashed, return_attention=return_attention)
        raise ValueError(f"Unknown SmashedData mode: {smashed.mode}")


def build_paper2601_server(
    *,
    input_fdim: int = 128,
    input_tdim: int = 259,
    n_client_blocks: int = 2,
    model_size: str = "base",
    num_labels: int = 1,
    static_feature_dim: int = 0,
    pooling: str = "cls",
    imagenet_pretrain: bool = False,
    audioset_checkpoint_path: Optional[str] = None,
) -> Paper2601SplitMAEServer:
    return Paper2601SplitMAEServer(
        SplitMAEServerConfig(
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            n_client_blocks=n_client_blocks,
            model_size=model_size,
            num_labels=num_labels,
            static_feature_dim=static_feature_dim,
            pooling=pooling,
            imagenet_pretrain=imagenet_pretrain,
            audioset_checkpoint_path=audioset_checkpoint_path,
        )
    )
