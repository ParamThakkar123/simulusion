"""Helper utilities for training and evaluating Simulusion models."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["init_model", "init_opt", "load_checkpoint"]


_HELPER_EXPORTS = {name: "simulusion.helpers.jepa_helper" for name in __all__}


def __getattr__(name: str) -> Any:
    if name not in _HELPER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_HELPER_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
