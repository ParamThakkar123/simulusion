"""One precision policy for inference, instead of a flag per model.

Models in Simulusion grew their own switches (``use_amp``, ``use_bfloat16``, ...).
At inference time the question is always the same - which dtype should matmuls
and convolutions run in on *this* device - so it is answered once here.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Literal

import torch

Precision = Literal["fp32", "bf16", "fp16", "auto"]

_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    "32": "fp32",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "16": "fp16",
    "auto": "auto",
}

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _device_type(device: torch.device | str) -> str:
    return torch.device(device).type


def resolve_precision(precision: str | None, device: torch.device | str) -> str:
    """Normalise a precision name and resolve ``"auto"`` for ``device``.

    ``auto`` picks bf16 on CUDA devices that support it (Ampere and later), fp16
    on older CUDA devices, and fp32 elsewhere. On CPU, reduced precision is
    usually *slower* unless the CPU has AMX/AVX512-BF16, so auto stays at fp32
    and bf16 must be requested explicitly.
    """
    if precision is None:
        return "fp32"
    try:
        name = _ALIASES[str(precision).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown precision {precision!r}; expected one of fp32, bf16, fp16, auto."
        ) from exc
    if name != "auto":
        if name == "fp16" and _device_type(device) == "cpu":
            raise ValueError(
                "fp16 is not supported for CPU inference; use bf16 or fp32."
            )
        return name
    if _device_type(device) == "cuda" and torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def precision_dtype(precision: str | None, device: torch.device | str) -> torch.dtype:
    """The ``torch.dtype`` that ``precision`` resolves to on ``device``."""
    return _DTYPES[resolve_precision(precision, device)]


@contextlib.contextmanager
def inference_context(
    device: torch.device | str, precision: str | None = "fp32"
) -> Iterator[None]:
    """``torch.inference_mode`` plus autocast at ``precision``.

    fp32 enters no autocast region at all, so it is bit-identical to calling the
    model directly under ``inference_mode``.
    """
    resolved = resolve_precision(precision, device)
    with torch.inference_mode():
        if resolved == "fp32":
            yield
        else:
            with torch.autocast(
                device_type=_device_type(device), dtype=_DTYPES[resolved]
            ):
                yield


__all__ = [
    "Precision",
    "inference_context",
    "precision_dtype",
    "resolve_precision",
]
