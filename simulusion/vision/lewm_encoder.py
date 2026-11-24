"""LeWorldModel's ViT encoder: a frame in, its [CLS] embedding out."""

from __future__ import annotations

import torch.nn as nn

from simulusion.export import ExportableAgentMixin
import torch
from functools import partial
from simulusion.models.vit import Block
import torch.nn.functional as F


class LeWMEncoder(ExportableAgentMixin, nn.Module):
    """ViT encoder whose [CLS] token is the frame's representation.

    The architecture of Hugging Face's ``ViTModel`` (pre-norm blocks with QKV
    biases, a learned [CLS] token and learned position embeddings, a final
    LayerNorm) at its "tiny" size by default: patch 14, 12 layers, 3 heads,
    width 192. Trained from scratch. Position embeddings are interpolated
    bicubically for frames of another size.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        init_std: float = 0.02,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        grid = self.img_size // self.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, embed_dim))
        norm = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm,  # type: ignore[arg-type]
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm(embed_dim)
        self.init_std = float(init_std)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=self.init_std)
        nn.init.trunc_normal_(self.pos_embed, std=self.init_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _positions(self, height: int, width: int) -> torch.Tensor:
        grid = self.img_size // self.patch_size
        h, w = height // self.patch_size, width // self.patch_size
        if (h, w) == (grid, grid):
            return self.pos_embed
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h, w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` normalized frames -> ``(B, D)`` [CLS] embeddings."""
        x = self.patch_embed(frames).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._positions(frames.shape[-2], frames.shape[-1])
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


__all__ = ["LeWMEncoder"]
