"""Lazy config exports.

Configuration modules can have optional training dependencies, so the package
initializer avoids importing every config eagerly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DreamerConfig": "simulusion.configs.dreamer_config",
    "JEPAConfig": "simulusion.configs.jepa_config",
    "DiTConfig": "simulusion.configs.dit_config",
    "get_dit_config": "simulusion.configs.dit_config",
    "dit_preset_config": "simulusion.configs.dit_config",
    "list_dit_presets": "simulusion.configs.dit_config",
    "DIT_PRESETS": "simulusion.configs.dit_config",
    "DiamondConfig": "simulusion.configs.diamond_config",
    "IRISConfig": "simulusion.configs.iris_config",
    "ATARI_100K_GAMES": "simulusion.configs.diamond_config",
    "HUMAN_SCORES": "simulusion.configs.diamond_config",
    "RANDOM_SCORES": "simulusion.configs.diamond_config",
    "GenieConfig": "simulusion.configs.genie_config",
    "GenieSmallConfig": "simulusion.configs.genie_config",
    "STTransformerConfig": "simulusion.configs.genie_config",
    "VideoTokenizerConfig": "simulusion.configs.genie_config",
    "LatentActionModelConfig": "simulusion.configs.genie_config",
    "DynamicsModelConfig": "simulusion.configs.genie_config",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = list(_EXPORTS)
