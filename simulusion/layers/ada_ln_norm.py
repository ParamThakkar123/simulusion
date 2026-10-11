import torch
import torch.nn as nn
from simulusion.layers.rms_norm import RMSNorm
from simulusion.export import ExportableAgentMixin


class AdaLNNormalization(ExportableAgentMixin, nn.Module):
    """Adaptive layer normalization conditioned on an external embedding.

    The module applies RMS normalization and predicts per-channel scale/shift
    from a conditioning vector (for example diffusion timestep embeddings).
    """

    def __init__(self, d_model: int, t_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.to_scale_shift = nn.Linear(t_dim, d_model * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale_shift = self.to_scale_shift(t_emb).unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        while scale.dim() < h.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return h * (1 + scale) + shift
