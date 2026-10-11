"""Tooling helpers for Simulusion.

All heavyweight or optional helpers are exposed lazily so ``python -m simulusion.cli``
and the installed ``simulusion`` command do not import documentation/browser tooling
or pre-import the CLI module during package initialization.
"""

from __future__ import annotations

import importlib
from typing import Any

_CLI_EXPORTS = {
    "cli_app": "app",
    "cli_main": "main",
    "run": "run",
    "version": "version",
    "envs_list": "envs_list",
    "datasets_list": "datasets_list",
    "datasets_convert": "datasets_convert",
    "collect": "collect",
    "benchmark": "benchmark",
    "train": "train",
}
_DOCS_EXPORTS = {"check_docs_main", "check_page", "file_url", "OUT_DIR"}

__all__ = [
    "cli",
    "cli_app",
    "cli_main",
    "run",
    "version",
    "envs_list",
    "datasets_list",
    "datasets_convert",
    "collect",
    "benchmark",
    "train",
    "check_docs_main",
    "check_page",
    "file_url",
    "OUT_DIR",
]


def __getattr__(name: str) -> Any:
    """Load optional tool modules only when their exports are requested."""
    if name == "cli":
        value = importlib.import_module("tools.cli")
    elif name in _CLI_EXPORTS:
        module = importlib.import_module("tools.cli")
        value = getattr(module, _CLI_EXPORTS[name])
    elif name in _DOCS_EXPORTS:
        module = importlib.import_module("tools.check_docs_render")
        value = module.main if name == "check_docs_main" else getattr(module, name)
    else:
        raise AttributeError(f"module 'tools' has no attribute {name!r}")

    globals()[name] = value
    return value
