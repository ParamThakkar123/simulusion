"""Weight-only int8 quantisation for world-model inference.

Weight-only quantisation stores ``nn.Linear`` weights as int8 with one fp scale
per output channel and dequantises on the fly. It cuts weight memory ~4x (vs
fp32) and, for the memory-bound small-batch steps world models run, usually
speeds them up - especially under ``torch.compile``, which fuses the dequantise
into the matmul. Activations stay in floating point, so no calibration data is
needed.

What is *not* quantised by default, and why:

* VQ codebooks / quantizers (IRIS, Genie tokenizers): token identity is decided
  by nearest-neighbour distances, so small weight error flips tokens.
* Stochastic-state heads (the RSSM prior/posterior projections): they output
  distribution parameters that are sampled from and fed back every step.
* Small layers (``min_in_features``): they save little memory and are where
  quantisation error is proportionally largest.

Embeddings, norms and convolutions are never touched. Always check the result
with :func:`simulusion.inference.rollout_drift`.
"""

from __future__ import annotations

from importlib import import_module, util
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SKIP = (
    "quantizer",
    "codebook",
    "vq",
    "fc_state_prior",
    "fc_state_posterior",
)


class Int8WeightOnlyLinear(nn.Module):
    """``nn.Linear`` with int8 weights and per-output-channel scales."""

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = weight_int8.shape[1]
        self.out_features = weight_int8.shape[0]
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8WeightOnlyLinear":
        weight = linear.weight.detach().to(torch.float32)
        # Symmetric per-channel: the largest |w| in each row maps to 127.
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        weight_int8 = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        module = cls(
            weight_int8,
            scale.to(linear.weight.dtype),
            linear.bias,
        )
        return module.to(linear.weight.device)

    weight_int8: torch.Tensor
    scale: torch.Tensor

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.weight_int8.to(dtype) * self.scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantized_weight(x.dtype), bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _skipped(name: str, skip: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in skip)


def quantize_weights(
    module: nn.Module,
    *,
    skip: Iterable[str] = DEFAULT_SKIP,
    min_in_features: int = 64,
    backend: str = "native",
) -> list[str]:
    """Quantise ``nn.Linear`` weights of ``module`` to int8, in place.

    Args:
        module: Module to quantise. Put it in eval mode first; the result is
            for inference only.
        skip: Substrings of qualified module names to leave in floating point.
            A layer is skipped if any of its ancestors' names matches too.
        min_in_features: Leave smaller ``nn.Linear`` layers alone.
        backend: ``"native"`` (default) uses :class:`Int8WeightOnlyLinear`,
            which needs no extra dependency and exports with ``torch.export``.
            ``"torchao"`` delegates to torchao's int8 weight-only kernels,
            which are faster on recent GPUs; it applies the same filter.

    Returns:
        Qualified names of the layers that were quantised.
    """
    skip = tuple(skip)
    selected = [
        name
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
        and name
        and not _skipped(name, skip)
        and child.in_features >= min_in_features
    ]
    if backend == "torchao":
        _quantize_torchao(module, set(selected))
        return selected
    if backend != "native":
        raise ValueError("backend must be 'native' or 'torchao'.")

    for name in selected:
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, Int8WeightOnlyLinear.from_linear(getattr(parent, attr)))
    return selected


def _quantize_torchao(module: nn.Module, selected: set[str]) -> None:
    if util.find_spec("torchao") is None:
        raise RuntimeError(
            "backend='torchao' needs the optional torchao package "
            "(pip install torchao), or use backend='native'."
        )
    quantization = import_module("torchao.quantization")
    config_cls = getattr(quantization, "Int8WeightOnlyConfig", None)
    config = (
        config_cls()
        if config_cls is not None
        else getattr(quantization, "int8_weight_only")()
    )
    names = {id(m): n for n, m in module.named_modules()}

    def keep(child: nn.Module, fqn: str = "") -> bool:
        return names.get(id(child), fqn) in selected

    quantization.quantize_(module, config, filter_fn=keep)


def weight_memory_bytes(module: nn.Module) -> int:
    """Bytes held by parameters and buffers, e.g. to compare before/after."""
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += tensor.element_size() * tensor.nelement()
    return total


__all__ = [
    "DEFAULT_SKIP",
    "Int8WeightOnlyLinear",
    "quantize_weights",
    "weight_memory_bytes",
]
