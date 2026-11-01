from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any

import gymnasium as gym

from simulusion.envs.gym_env import GymImageEnv


_GYMNASIUM_ROBOTICS_PACKAGE = "gymnasium_robotics"
_MOVED_MUJOCO_MESSAGE = "gymnasium-robotics"


def _registry_ids() -> set[str]:
    return {str(env_id) for env_id in gym.registry}


def _robotics_ids_from_registry() -> set[str]:
    env_ids: set[str] = set()
    for env_id, spec in gym.registry.items():
        entry_point = getattr(spec, "entry_point", "")
        if isinstance(entry_point, str):
            is_robotics_entry = entry_point.startswith(
                f"{_GYMNASIUM_ROBOTICS_PACKAGE}."
            )
        else:
            is_robotics_entry = getattr(entry_point, "__module__", "").startswith(
                f"{_GYMNASIUM_ROBOTICS_PACKAGE}."
            )
        if is_robotics_entry:
            env_ids.add(str(env_id))
    return env_ids


def is_moved_mujoco_error(exc: BaseException) -> bool:
    """Return whether Gymnasium reported the v2/v3 MuJoCo move."""
    return _MOVED_MUJOCO_MESSAGE in str(exc).lower()


def register_gymnasium_robotics_envs() -> Any:
    """Import Gymnasium Robotics so its environments are registered with Gymnasium.

    Gymnasium moved legacy MuJoCo v2/v3 task registrations into the external
    ``gymnasium-robotics`` package. Current Gymnasium Robotics versions register
    environments during import, while older plugin-style installations may rely
    on ``gymnasium.register_envs``; this helper supports both paths.
    """
    try:
        package_spec = importlib.util.find_spec(_GYMNASIUM_ROBOTICS_PACKAGE)
    except ValueError:
        package_spec = None
    if package_spec is None and _GYMNASIUM_ROBOTICS_PACKAGE not in sys.modules:
        raise ImportError(
            "Gymnasium Robotics is required for env_backend='robotics' and "
            "Gymnasium MuJoCo v2/v3 task ids. Install it with "
            "`pip install gymnasium-robotics` or `pip install simulusion[robotics]`."
        )

    before_ids = _registry_ids()
    if _GYMNASIUM_ROBOTICS_PACKAGE in sys.modules:
        module = sys.modules[_GYMNASIUM_ROBOTICS_PACKAGE]
    else:
        module = importlib.import_module(_GYMNASIUM_ROBOTICS_PACKAGE)

    # Some Gymnasium third-party environment packages expose registration via
    # gym.register_envs(package). Call it only when import did not reveal any
    # Gymnasium Robotics specs to avoid duplicate-registration errors.
    if not _robotics_ids_from_registry() and before_ids == _registry_ids():
        register_envs = getattr(gym, "register_envs", None)
        if callable(register_envs):
            register_envs(module)
    return module


def list_gymnasium_robotics_envs() -> list[str]:
    """List all Gymnasium Robotics ids registered by the installed package.

    Returns an empty list when the optional dependency is not installed. When it
    is installed, the list is derived from Gymnasium's registry rather than a
    hand-maintained subset, so newly added Robotics environments are exposed
    automatically.
    """
    try:
        register_gymnasium_robotics_envs()
    except ImportError:
        return []
    return sorted(_robotics_ids_from_registry())


def make_gymnasium_env_with_robotics_fallback(
    env: str,
    *,
    render_mode: str = "rgb_array",
    gym_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Gymnasium env and retry after Robotics registration if needed."""
    env_kwargs = dict(gym_kwargs or {})
    env_kwargs.update(kwargs)
    try:
        return gym.make(str(env), render_mode=render_mode, **env_kwargs)
    except ImportError as exc:
        if not is_moved_mujoco_error(exc):
            raise
        register_gymnasium_robotics_envs()
        try:
            return gym.make(str(env), render_mode=render_mode, **env_kwargs)
        except TypeError:
            return gym.make(str(env), **env_kwargs)
    except TypeError:
        return gym.make(str(env), **env_kwargs)


def make_robotics_env(
    env: str,
    *,
    seed: int = 0,
    size: tuple[int, int] = (64, 64),
    render_mode: str = "rgb_array",
    gym_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> GymImageEnv:
    """Create a Simulusion image wrapper for a Gymnasium Robotics environment.

    Args:
        env: Any environment id registered by ``gymnasium-robotics``.
        seed: Seed forwarded to ``GymImageEnv``.
        size: Target ``(height, width)`` image size.
        render_mode: Render mode forwarded to ``gymnasium.make``.
        gym_kwargs: Optional keyword arguments forwarded to ``gymnasium.make``.
        **kwargs: Additional keyword arguments forwarded to ``gymnasium.make``.

    Returns:
        A ``GymImageEnv`` that emits ``{"image": uint8[C, H, W]}`` observations.
    """
    register_gymnasium_robotics_envs()
    base_env = make_gymnasium_env_with_robotics_fallback(
        env,
        render_mode=render_mode,
        gym_kwargs=gym_kwargs,
        **kwargs,
    )
    return GymImageEnv(base_env, seed=seed, size=size, render_mode=render_mode)
