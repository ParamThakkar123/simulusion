"""Backward-compatible wrapper around :mod:`simulusion.cli`."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_impl = import_module("simulusion.cli")

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


if __name__ == "__main__":
    _impl.run()
