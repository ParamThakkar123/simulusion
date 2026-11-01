import torch
import torch.nn as nn
import torch.nn.functional as F
from simulusion.export import ExportableAgentMixin


class MultiHeadSelfAttention(ExportableAgentMixin, nn.Module):
    """Multi-head scaled dot-product self-attention over sequence tokens.

    This module projects the input sequence into query/key/value heads, performs
    attention independently per head, and merges the heads back into the original
    feature dimension. It is used as a lightweight transformer attention block.
    """

    def __init__(self, d: int, n_heads: int = 2) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        context = F.scaled_dot_product_attention(Q, K, V)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(context)

        return out
