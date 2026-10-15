import torch
import torch.nn as nn
from simulusion.export import ExportableAgentMixin


class RMSNorm(ExportableAgentMixin, nn.Module):
    """Root Mean Square Layer Normalization with a learned gain parameter.

    RMSNorm rescales activations using their RMS magnitude without centering,
    providing a lightweight normalization alternative to LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms = norm_x * (x.size(-1) ** -0.5)
        x_normed = x / (rms + self.eps)
        return x_normed * self.g
