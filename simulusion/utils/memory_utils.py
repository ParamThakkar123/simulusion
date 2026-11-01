import warnings

import torch
import torch.nn as nn
from typing import Any, Callable, Optional


def apply_gradient_checkpointing(
    model: nn.Module, checkpoint_ratio: float = 0.5
) -> None:
    """Apply gradient checkpointing to reduce memory usage during training."""
    # Some models expose gradient_checkpointing_enable as a callable while
    # others may have an attribute with the same name that isn't callable.
    # Use getattr and only call when it's actually callable to avoid mypy
    # complaining about "Tensor" not callable at type-check time.
    gc_enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(gc_enable):
        gc_enable()
    else:
        # For custom modules, apply selective checkpointing
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                # Wrap the original forward in a callable that uses
                # torch.utils.checkpoint.checkpoint. We capture the original
                # method to avoid recursive lookup and assign a plain
                # function to the instance attribute (allowed at runtime).
                orig_forward: Callable[..., Any] = module.forward  # capture

                def _checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
                    # Torch's checkpoint API is present at runtime but some stubs
                    # do not expose it. Ignore attribute errors from type-checker
                    # here while preserving runtime behavior.
                    return torch.utils.checkpoint.checkpoint(
                        orig_forward, *args, **kwargs, use_reentrant=False
                    )

                setattr(module, "forward", _checkpointed_forward)
            elif hasattr(module, "checkpoint_forward"):
                # Create a wrapper that calls checkpoint at runtime. Do not
                # call checkpoint here (that would execute the function and
                # assign a Tensor to `forward`). Some torch stubs do not
                # expose `utils.checkpoint`, so silence attribute checks.
                def _checkpointed_forward2(*args: Any, **kwargs: Any) -> Any:
                    # Use a targeted ignore for the missing `checkpoint` attr in
                    # some torch stubs while preserving runtime behaviour.
                    return torch.utils.checkpoint.checkpoint(
                        module.checkpoint_forward, *args, **kwargs, use_reentrant=False
                    )

                setattr(module, "forward", _checkpointed_forward2)


def enable_mixed_precision(
    model: nn.Module, scaler: Optional[torch.amp.GradScaler] = None
) -> torch.amp.GradScaler:
    """Enable mixed precision training."""
    if scaler is None:
        scaler = torch.amp.GradScaler()
    return scaler


def enable_performance_defaults(
    *, tf32: bool = True, cudnn_benchmark: bool = True
) -> None:
    """Turn on the CUDA throughput settings Simulusion trainers expect.

    Call this once at the start of a training run. It is deliberately explicit
    and not applied on import, because both settings trade something away:

    ``cudnn_benchmark`` autotunes convolution algorithms on first sight of each
    input shape. That is a large win for fixed-shape training and a loss for
    workloads whose shapes keep changing, and the autotuner's choice is not
    guaranteed stable run to run.

    ``tf32`` lets matmuls and convolutions use TensorFloat-32 on Ampere and
    later: same exponent range as float32, but a 10-bit mantissa. For
    model-based RL and vision training this is the standard setting and costs no
    measurable quality, but it is *not* bit-reproducible against a float32
    baseline - leave it off for numerics regression work.

    Both are no-ops without CUDA, so this is safe to call unconditionally.
    """
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = cudnn_benchmark
    if tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def optimize_memory_efficient_ops() -> None:
    """Deprecated alias for :func:`enable_performance_defaults`.

    The old name also set ``cudnn.deterministic = False`` globally, which
    silently overrode any determinism a caller had asked for.
    """
    warnings.warn(
        "optimize_memory_efficient_ops() is deprecated; call "
        "enable_performance_defaults() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    enable_performance_defaults()


def maybe_compile(
    module_or_fn: Any,
    *,
    enabled: bool = False,
    mode: str = "reduce-overhead",
) -> Any:
    """Optionally wrap a callable in ``torch.compile``.

    RSSM rollouts and CEM planning are sequences of very small kernels driven by
    a Python loop, so they are launch-bound rather than FLOP-bound; compiling
    the step collapses that overhead. It is off by default because compilation
    costs seconds to minutes on first call, recompiles whenever an input shape
    changes, and is unavailable on some builds - none of which a short run or a
    test wants to pay for.

    Falls back to the eager callable if compilation is unsupported, so callers
    never need to guard. ``torch.compile`` is lazy: a missing backend (for
    example no Triton on a Windows CUDA build) surfaces on the *first call*, not
    when wrapping, so the fallback also covers that first call. Once a compiled
    call has succeeded, later errors propagate unchanged - by then they are
    genuine errors, not a missing backend.

    An ``nn.Module`` is compiled by swapping its instance ``forward``, so the
    module object, its attributes and its ``state_dict`` keys are unchanged.
    (``torch.compile(module)`` would return an ``OptimizedModule`` whose keys
    gain an ``_orig_mod.`` prefix, and checkpoints written from it would not load
    into an uncompiled model.)
    """
    if not enabled:
        return module_or_fn
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:  # pragma: no cover - very old torch
        return module_or_fn

    if isinstance(module_or_fn, nn.Module):
        eager_forward = module_or_fn.forward
        compiled_forward = _compile_with_fallback(compile_fn, eager_forward, mode)
        if compiled_forward is not eager_forward:
            module_or_fn.forward = compiled_forward
        return module_or_fn
    return _compile_with_fallback(compile_fn, module_or_fn, mode)


class _CompiledWithFallback:
    """Call a compiled callable, dropping to eager if its first call fails."""

    def __init__(self, compiled: Callable[..., Any], eager: Callable[..., Any]):
        self._compiled = compiled
        self._eager = eager
        self._succeeded = False
        self._failed = False
        self.__wrapped__ = eager

    @property
    def is_compiled(self) -> bool:
        """False once compilation has failed and calls run eagerly."""
        return not self._failed

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._failed:
            return self._eager(*args, **kwargs)
        try:
            result = self._compiled(*args, **kwargs)
        except Exception as exc:
            if self._succeeded:
                raise
            self._failed = True
            warnings.warn(
                f"torch.compile failed on first call ({type(exc).__name__}: "
                f"{exc}); continuing without it.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._eager(*args, **kwargs)
        self._succeeded = True
        return result


def _compile_with_fallback(
    compile_fn: Callable[..., Any], fn: Callable[..., Any], mode: str
) -> Callable[..., Any]:
    try:
        compiled = compile_fn(fn, mode=mode)
    except Exception:  # pragma: no cover - backend unavailable on this platform
        warnings.warn(
            "torch.compile is unavailable here; continuing without it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return fn
    return _CompiledWithFallback(compiled, fn)


def to_channels_last(model: nn.Module) -> nn.Module:
    """Convert a conv-heavy model to NHWC memory format.

    This is a layout change only - values are untouched - but it lets cuDNN pick
    NHWC tensor-core kernels for convolutions instead of transposing on every
    call. Inputs must be converted to match; mixing layouts silently costs more
    than it saves.
    """
    # `Module.to` has no typed overload accepting only `memory_format`, though
    # it is a documented and supported call.
    return model.to(memory_format=torch.channels_last)  # type: ignore[call-overload, unused-ignore]
