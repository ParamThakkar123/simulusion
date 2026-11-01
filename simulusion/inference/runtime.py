"""In-process inference optimisation for any Simulusion module.

World-model inference is a loop of small steps (an RSSM transition, one token,
one denoising pass). Each step does little arithmetic, so on a GPU its cost is
dominated by Python and kernel-launch overhead rather than FLOPs. The two levers
that matter most are therefore:

* ``torch.compile(mode="reduce-overhead")``, which fuses kernels and replays the
  step as a CUDA graph - one launch instead of hundreds;
* reduced precision (bf16/fp16), which halves memory traffic.

:func:`optimize_for_inference` applies both behind one wrapper, and every
setting is opt-in: with the defaults the wrapped module is numerically identical
to calling it under ``torch.inference_mode``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from simulusion.inference.precision import (
    inference_context,
    precision_dtype,
    resolve_precision,
)
from simulusion.utils.memory_utils import maybe_compile, to_channels_last
from simulusion.export import ExportableAgentMixin


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _cast_floating(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype)
    return value


def _clone_tensor(value: Any) -> Any:
    return value.clone() if isinstance(value, torch.Tensor) else value


class InferenceModel(ExportableAgentMixin, nn.Module):
    """A module wrapped for fast, gradient-free inference.

    Args:
        module: Module to wrap. It is put in eval mode and its parameters stop
            requiring grad; its weights are only modified if ``cast_weights``.
        precision: ``"fp32"`` (default), ``"bf16"``, ``"fp16"`` or ``"auto"``.
            Reduced precision runs under autocast, so numerically sensitive ops
            (softmax, norms, reductions) stay in fp32.
        compile: Wrap the forward in ``torch.compile``. Falls back to eager,
            with a warning, if the backend is unavailable (e.g. no Triton).
        compile_mode: ``torch.compile`` mode. ``"reduce-overhead"`` (default)
            uses CUDA graphs, the right choice for small, fixed-shape steps.
            Use ``"max-autotune"`` for large batched steps.
        channels_last: Convert the module (and 4D inputs) to NHWC, which lets
            cuDNN use tensor-core convolution kernels.
        cast_weights: Store weights in the reduced dtype as well, halving their
            memory. Inputs are cast to match. Ignored at fp32.
        output_dtype: Cast floating outputs to this dtype (default fp32) so the
            wrapper is a drop-in replacement. ``None`` returns them as computed.
        clone_outputs: Clone outputs after each call. CUDA-graph replays reuse
            their output buffers, so an output kept across calls would otherwise
            be overwritten by the next step. Defaults to on exactly when
            compiling with a CUDA-graph mode.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        precision: str | None = "fp32",
        compile: bool = False,
        compile_mode: str = "reduce-overhead",
        channels_last: bool = False,
        cast_weights: bool = False,
        output_dtype: torch.dtype | None = torch.float32,
        clone_outputs: bool | None = None,
    ) -> None:
        super().__init__()
        module.eval()
        module.requires_grad_(False)
        device = _module_device(module)
        self.precision = resolve_precision(precision, device)
        self.channels_last = channels_last
        self.output_dtype = output_dtype
        self.input_dtype: torch.dtype | None = None
        if cast_weights and self.precision != "fp32":
            self.input_dtype = precision_dtype(self.precision, device)
            module.to(self.input_dtype)
        if channels_last:
            to_channels_last(module)
        self.module = module

        uses_cuda_graphs = compile and "overhead" in compile_mode
        self._cuda_graphs = uses_cuda_graphs and device.type == "cuda"
        self.clone_outputs = (
            self._cuda_graphs if clone_outputs is None else bool(clone_outputs)
        )
        self._call = maybe_compile(self._eager_call, enabled=compile, mode=compile_mode)

    @property
    def device(self) -> torch.device:
        return _module_device(self.module)

    def _eager_call(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def _prepare(self, value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        if self.input_dtype is not None:
            value = _cast_floating(value, self.input_dtype)
        if self.channels_last and value.dim() == 4:
            value = value.contiguous(memory_format=torch.channels_last)
        return value

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        args = tree_map(self._prepare, args)
        kwargs = tree_map(self._prepare, kwargs)
        if self._cuda_graphs:
            # Tells the CUDA-graph tree that a new inference step begins, so the
            # previous step's output buffers may be reused.
            torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
        with inference_context(self.device, self.precision):
            out = self._call(*args, **kwargs)
        if self.clone_outputs:
            out = tree_map(_clone_tensor, out)
        output_dtype = self.output_dtype
        if output_dtype is not None:
            out = tree_map(lambda v: _cast_floating(v, output_dtype), out)
        return out


def optimize_for_inference(module: nn.Module, **kwargs: Any) -> InferenceModel:
    """Wrap ``module`` for inference. See :class:`InferenceModel` for options.

    Example::

        policy = optimize_for_inference(agent.dreamer.actor, precision="auto",
                                        compile=True)
        action = policy(features)
    """
    return InferenceModel(module, **kwargs)


__all__ = ["InferenceModel", "optimize_for_inference"]
