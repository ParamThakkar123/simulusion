"""Device selection shared by models, trainers and the CLI."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _why_no_cuda() -> str:
    """Explain why CUDA is unavailable, so a silent CPU fallback is diagnosable."""
    built_with = getattr(torch.version, "cuda", None)
    if not built_with:
        return (
            f"this torch ({torch.__version__}) is a CPU-only build, so no GPU can "
            "be used. Install a CUDA build from https://pytorch.org/get-started/locally/"
        )
    return (
        f"torch {torch.__version__} is built for CUDA {built_with}, but no GPU is "
        "visible to this process (driver missing, or CUDA_VISIBLE_DEVICES excludes it)"
    )


def mps_is_available() -> bool:
    """Whether Apple's Metal (MPS) backend can be used."""
    backend = getattr(torch.backends, "mps", None)
    try:
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def get_default_device(allow_gpu: bool = True) -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if allow_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
    return torch.device("cpu")


def default_device_name(allow_gpu: bool = True) -> str:
    """String form of :func:`get_default_device`, for configs and CLI defaults."""
    return get_default_device(allow_gpu).type


def resolve_device(requested: str | torch.device | None) -> torch.device:
    """Honour a requested device, falling back when it is not available.

    ``None`` selects :func:`get_default_device`. A CUDA request on a machine
    without CUDA, or an MPS request without MPS, falls back to the best device
    that does exist instead of failing later inside ``.to(device)``.

    A fallback is logged as a warning rather than applied silently: a run that
    asked for ``cuda`` and quietly trained on CPU looks identical to a working
    one apart from being far slower.
    """
    if requested is None:
        return get_default_device()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        fallback = get_default_device()
        logger.warning(
            "device %r was requested but CUDA is not available: %s. Falling back "
            "to %s.",
            str(requested),
            _why_no_cuda(),
            fallback.type,
        )
        return fallback
    if device.type == "mps" and not mps_is_available():
        logger.warning(
            "device %r was requested but Apple MPS is not available. Falling "
            "back to cpu.",
            str(requested),
        )
        return torch.device("cpu")
    return device


__all__ = [
    "default_device_name",
    "get_default_device",
    "mps_is_available",
    "resolve_device",
]
