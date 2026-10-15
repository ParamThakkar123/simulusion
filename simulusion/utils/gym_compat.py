from __future__ import annotations

from typing import Any


def import_gymnasium() -> Any | None:
    try:
        import gymnasium as gymnasium
    except ModuleNotFoundError:
        return None
    return gymnasium


def import_gym() -> Any:
    gymnasium = import_gymnasium()
    if gymnasium is not None:
        return gymnasium

    try:
        import gym as legacy_gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Neither 'gymnasium' nor legacy 'gym' is installed. "
            "Install `simulusion[gym]` to enable gym-compatible environments."
        ) from exc
    return legacy_gym


class _LazyGym:
    """Proxy that resolves gymnasium/gym on first attribute access.

    Importing modules that merely *reference* gym (type annotations, space
    construction inside methods) must not require the optional dependency to be
    installed. The real import -- and its helpful ``simulusion[gym]`` error -- is
    therefore deferred until an attribute such as ``gym.spaces`` or ``gym.make``
    is actually accessed at runtime.
    """

    _module: Any = None

    @classmethod
    def _resolve(cls) -> Any:
        if cls._module is None:
            cls._module = import_gym()
        return cls._module

    def __getattr__(self, name: str) -> Any:
        # Never resolve the backend just to answer dunder probes (copy, pickle,
        # ``hasattr`` on special names); those must behave as "not present".
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)


gym: Any = _LazyGym()


def __getattr__(name: str) -> Any:
    if name == "spaces":
        return gym.spaces
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ``spaces`` is still importable (``from ... import spaces``) via the lazy
# module-level ``__getattr__`` above; it is deliberately kept out of ``__all__``
# because static analysis cannot see dynamically provided names.
__all__ = ["gym", "import_gym", "import_gymnasium"]
