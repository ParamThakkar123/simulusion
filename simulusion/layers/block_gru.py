"""Block-diagonal GRU used by the DreamerV3 sequence model.

A standard GRU's recurrent weights grow quadratically with the number of hidden
units, which makes wide recurrent states expensive. DreamerV3 instead splits the
recurrent state into ``blocks`` equally sized groups and applies a separate
recurrent weight matrix within each group. Parameters and FLOPs then grow
linearly in the number of units for a fixed block size.

Mixing between blocks is not lost: the GRU input at each step is a dense linear
embedding of the sampled latent, the action, *and* the previous recurrent state,
so information can still cross block boundaries once per timestep.

Reference:
    Van Keirsbilck et al., 2019 - https://arxiv.org/abs/1905.12340
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from simulusion.export import ExportableAgentMixin
from simulusion.layers.rms_norm import RMSNorm

__all__ = ["BlockLinear", "BlockGRUCell"]


class BlockLinear(ExportableAgentMixin, nn.Module):
    """Linear layer applied independently within each of ``blocks`` groups.

    Equivalent to a dense layer whose weight matrix is constrained to be
    block-diagonal, but stored and applied as a batched matmul.

    Args:
        in_features: Total input size; must be divisible by ``blocks``.
        out_features: Total output size; must be divisible by ``blocks``.
        blocks: Number of independent groups.
        bias: Whether to learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        blocks: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features % blocks != 0:
            raise ValueError(
                f"in_features={in_features} is not divisible by blocks={blocks}"
            )
        if out_features % blocks != 0:
            raise ValueError(
                f"out_features={out_features} is not divisible by blocks={blocks}"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.blocks = int(blocks)
        self.in_per_block = self.in_features // self.blocks
        self.out_per_block = self.out_features // self.blocks

        self.weight = nn.Parameter(
            torch.empty(self.blocks, self.in_per_block, self.out_per_block)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_per_block)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the block-diagonal transform.

        Args:
            inputs: Tensor of shape ``(*batch, in_features)``.

        Returns:
            Tensor of shape ``(*batch, out_features)``.
        """
        batch_shape = inputs.shape[:-1]
        blocked = inputs.reshape(*batch_shape, self.blocks, self.in_per_block)
        out = torch.einsum("...bi,bio->...bo", blocked, self.weight)
        out = out.reshape(*batch_shape, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"blocks={self.blocks}, bias={self.bias is not None}"
        )


class BlockGRUCell(ExportableAgentMixin, nn.Module):
    """GRU cell with block-diagonal recurrent weights and RMSNorm gating.

    The cell expects the caller to supply a dense input embedding of size
    ``hidden_size`` (the DreamerV3 RSSM builds it from the latent, the action,
    and the previous recurrent state, which is what allows blocks to mix).

    Args:
        hidden_size: Size of the recurrent state; must be divisible by ``blocks``.
        blocks: Number of independent recurrent groups.
        norm: Whether to normalize the gate pre-activations with RMSNorm.
    """

    def __init__(self, hidden_size: int, blocks: int = 8, norm: bool = True) -> None:
        super().__init__()
        if hidden_size % blocks != 0:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by blocks={blocks}"
            )
        self.hidden_size = int(hidden_size)
        self.blocks = int(blocks)
        self.per_block = self.hidden_size // self.blocks

        # Each block sees its own slice of the input embedding and of the state.
        self.block_linear = BlockLinear(
            2 * self.hidden_size, 3 * self.hidden_size, self.blocks, bias=not norm
        )
        self.norm = RMSNorm(3 * self.per_block) if norm else None

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Advance the recurrent state by one step.

        Args:
            inputs: Input embedding of shape ``(*batch, hidden_size)``.
            state: Previous recurrent state of shape ``(*batch, hidden_size)``.

        Returns:
            Updated recurrent state of shape ``(*batch, hidden_size)``.
        """
        batch_shape = inputs.shape[:-1]
        # Interleave input and state per block so block b sees only slice b of
        # each. A plain concatenation would hand block 0 only input features.
        blocked_inputs = inputs.reshape(*batch_shape, self.blocks, self.per_block)
        blocked_state = state.reshape(*batch_shape, self.blocks, self.per_block)
        combined = torch.cat([blocked_inputs, blocked_state], dim=-1)
        combined = combined.reshape(*batch_shape, 2 * self.hidden_size)

        parts = self.block_linear(combined)
        parts = parts.reshape(*batch_shape, self.blocks, 3 * self.per_block)
        if self.norm is not None:
            parts = self.norm(parts)

        reset, cand, update = torch.chunk(parts, 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        # The -1 offset biases the gate towards keeping the previous state,
        # which stabilizes long-horizon credit assignment early in training.
        update = torch.sigmoid(update - 1.0)

        cand = cand.reshape(*batch_shape, self.hidden_size)
        update = update.reshape(*batch_shape, self.hidden_size)
        return update * cand + (1.0 - update) * state

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, blocks={self.blocks}"
