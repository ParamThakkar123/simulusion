"""Causal multi-head self-attention over a sequence of tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from simulusion.export import ExportableAgentMixin


class CausalAttention(ExportableAgentMixin, nn.Module):
    """Pre-norm multi-head self-attention where step ``t`` sees steps ``<= t`` only.

    LayerNorm, a bias-free QKV projection to ``heads * dim_head``, causal
    scaled dot-product attention, then an output projection back to ``dim``
    (skipped when a single head already has width ``dim``). LeWorldModel's
    predictor uses it inside :class:`~simulusion.models.lewm.AdaLNBlock`.

    Args:
        dim: Token width.
        heads: Attention heads.
        dim_head: Width per head.
        dropout: Dropout on the attention weights and the output, in training.
    """

    def __init__(
        self, dim: int, heads: int, dim_head: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out: nn.Module = (
            nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
            if not (heads == 1 and dim_head == dim)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, dim)`` -> ``(B, T, dim)``."""
        B, T, _ = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.to_out(out.transpose(1, 2).reshape(B, T, -1))


__all__ = ["CausalAttention"]
