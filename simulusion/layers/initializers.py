"""Weight initializers shared by the world models.

:func:`trunc_normal_in_` is DreamerV3's ``trunc_normal_in`` initializer
(``embodied/jax/nets.py`` in the reference code): a normal truncated to
``[-2, 2]``, rescaled to unit variance (x 1.1368) and to ``1 / fan_in``, times
an output scale; biases start at zero. :func:`dreamer_v3_init_` applies it to
every layer of a module, with the fan-in the reference computes for each layer
type.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

#: A standard normal truncated to [-2, 2] has std 0.8796; this restores 1.
TRUNC_NORMAL_UNIT_SCALE = 1.1368


@torch.no_grad()
def trunc_normal_in_(weight: torch.Tensor, fan_in: int, scale: float = 1.0) -> torch.Tensor:
    """Fill ``weight`` with ``trunc_normal(-2, 2) * 1.1368 * sqrt(1 / fan_in) * scale``."""
    nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
    weight.mul_(TRUNC_NORMAL_UNIT_SCALE * math.sqrt(1.0 / max(1, int(fan_in))) * scale)
    return weight


def _fan_in(layer: nn.Module) -> int | None:
    """The fan-in DreamerV3's initializer uses for ``layer``, or ``None`` to skip it."""
    from simulusion.layers.block_gru import BlockLinear

    if isinstance(layer, nn.Linear):
        return int(layer.in_features)
    if isinstance(layer, BlockLinear):
        # The reference kernel is (blocks, in / blocks, out / blocks); its fan-in
        # counts every block: the full input width.
        return int(layer.in_features)
    if isinstance(layer, nn.ConvTranspose2d):
        # Weight (in, out, kh, kw); the reference kernel is HWIO over the input.
        kh, kw = layer.kernel_size
        return int(layer.in_channels) * int(kh) * int(kw) // int(layer.groups)
    if isinstance(layer, nn.Conv2d):
        kh, kw = layer.kernel_size
        return int(layer.in_channels) * int(kh) * int(kw) // int(layer.groups)
    return None


@torch.no_grad()
def dreamer_v3_init_(module: nn.Module, outscale: float = 1.0) -> nn.Module:
    """Initialize every linear, convolutional and block-linear layer of ``module``.

    Weights get :func:`trunc_normal_in_` at scale ``outscale``; biases zero.
    Normalization layers keep their unit scale and zero shift.
    """
    for layer in module.modules():
        fan = _fan_in(layer)
        if fan is None:
            continue
        trunc_normal_in_(cast(torch.Tensor, layer.weight), fan, outscale)
        bias = getattr(layer, "bias", None)
        if bias is not None:
            nn.init.zeros_(bias)
    return module


@torch.no_grad()
def scale_output_(layer: nn.Module, outscale: float) -> nn.Module:
    """Re-initialize one output layer at ``outscale`` (0 zeroes it)."""
    fan = _fan_in(layer)
    if fan is None:
        raise TypeError(f"{type(layer).__name__} is not a linear or conv layer.")
    weight = cast(torch.Tensor, layer.weight)
    if outscale == 0.0:
        nn.init.zeros_(weight)
    else:
        trunc_normal_in_(weight, fan, outscale)
    bias = getattr(layer, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)
    return layer


__all__ = [
    "TRUNC_NORMAL_UNIT_SCALE",
    "dreamer_v3_init_",
    "scale_output_",
    "trunc_normal_in_",
]
