"""Small, import-safe catalog of available environments and backends.

This module replaces the previous `simulusion.ui.catalog` and is safe to
import from lightweight CLI tools and tests without pulling in any UI
dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DREAMER_ENVS = [
    "cartpole-balance",
    "cartpole-swingup",
    "cheetah-run",
    "finger-spin",
    "reacher-easy",
    "walker-walk",
    "walker-run",
    "quadruped-walk",
]

DEFAULT_GYM_ENV_PREFERENCES = (
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
)


def _iter_gymnasium_registry() -> list[tuple[str, Any]]:
    """Return Gymnasium registry entries without hardcoding API versions."""
    try:
        import gymnasium as gym
    except Exception:
        logger.debug("Gymnasium registry unavailable", exc_info=True)
        return []

    registry = getattr(getattr(gym, "envs", None), "registry", {})
    items = registry.items() if hasattr(registry, "items") else []
    entries: list[tuple[str, Any]] = []
    for env_id, spec in items:
        resolved_id = getattr(spec, "id", env_id) or env_id
        entries.append((resolved_id, spec))
    return entries


def _entry_point_text(spec: Any) -> str:
    return str(getattr(spec, "entry_point", "") or "").lower()


def _is_atari_env(env_id: str, spec: Any) -> bool:
    lowered_id = env_id.lower()
    entry_point = _entry_point_text(spec)
    namespace = str(getattr(spec, "namespace", "") or "").lower()
    return (
        lowered_id.startswith("ale/")
        or namespace == "ale"
        or "ale_py" in entry_point
        or "atari" in entry_point
    )


def _is_robotics_env(env_id: str, spec: Any) -> bool:
    entry_point = _entry_point_text(spec)
    namespace = str(getattr(spec, "namespace", "") or "").lower()
    return "gymnasium_robotics" in entry_point or namespace in {
        "fetch",
        "hand",
        "maze",
    }


def _list_available_gymnasium_envs() -> list[str]:
    """Return non-Atari, non-Robotics Gymnasium ids from the installed registry."""
    return sorted(
        env_id
        for env_id, spec in _iter_gymnasium_registry()
        if not _is_atari_env(env_id, spec) and not _is_robotics_env(env_id, spec)
    )


def _dedupe_envs(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    combined: list[str] = []
    for group in groups:
        for env_id in group:
            if env_id not in seen:
                combined.append(env_id)
                seen.add(env_id)
    return combined


def _preferred_available_envs(
    env_ids: list[str], preferred: tuple[str, ...]
) -> list[str]:
    """Put stable starter envs first when they exist, then include the registry."""
    available = set(env_ids)
    preferred_available = [env_id for env_id in preferred if env_id in available]
    return _dedupe_envs(preferred_available, env_ids)


GYM_ENVS = _preferred_available_envs(
    _list_available_gymnasium_envs(), DEFAULT_GYM_ENV_PREFERENCES
)
PLANET_BASE_ENVS = GYM_ENVS


def _list_available_robotics_envs() -> list[str]:
    """Return all Gymnasium Robotics ids when the optional package exists."""
    try:
        from simulusion.envs import list_gymnasium_robotics_envs

        return list_gymnasium_robotics_envs()
    except Exception:
        logger.debug("Gymnasium Robotics environments unavailable", exc_info=True)
        return []


ROBOTICS_ENVS: list[str] = _list_available_robotics_envs()

UNITY_ENVS: list[str] = []

PROCGEN_ENVS = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def _list_available_bsuite_ids() -> list[str]:
    """Return BSuite sweep ids or examples without making catalog imports heavy."""
    try:
        from simulusion.envs import list_available_bsuite_ids

        return list_available_bsuite_ids()
    except Exception:
        logger.debug(
            "BSuite environment ids unavailable; using examples", exc_info=True
        )
        return [
            "bandit/0",
            "cartpole/0",
            "catch/0",
            "deep_sea/0",
            "discounting_chain/0",
            "memory_len/0",
            "mnist/0",
            "mountain_car/0",
            "umbrella_chain/0",
        ]


BSUITE_ENVS: list[str] = _list_available_bsuite_ids()


def _list_available_atari_envs() -> list[str]:
    """Return registered Atari ids without making catalog imports heavy."""
    try:
        from simulusion.envs import list_available_atari_envs

        return list_available_atari_envs()
    except Exception:
        logger.debug("Atari environment ids unavailable", exc_info=True)
        return []


ATARI_ENVS: list[str] = _list_available_atari_envs()


def _list_available_dmlab_levels() -> list[str]:
    try:
        from simulusion.envs.dmlab import DMLAB_LEVELS

        return list(DMLAB_LEVELS)
    except Exception:
        logger.debug("DeepMind Lab levels unavailable", exc_info=True)
        return []


ENV_BACKENDS: dict[str, dict[str, Any]] = {
    "dm_control": {
        "label": "DM Control",
        "description": "DeepMind Control Suite",
        "environments": DREAMER_ENVS,
    },
    "mujoco": {
        "label": "MuJoCo",
        "description": "MuJoCo physics environments",
        "environments": GYM_ENVS,
    },
    "gym": {
        "label": "Gym",
        "description": "OpenAI Gym environments",
        "environments": PLANET_BASE_ENVS,
    },
    "robotics": {
        "label": "Gymnasium Robotics",
        "description": "Gymnasium Robotics and legacy MuJoCo v2/v3 environments",
        "environments": ROBOTICS_ENVS,
    },
    "unity": {
        "label": "Unity ML Agents",
        "description": "Unity ML Agents environments",
        "environments": UNITY_ENVS,
    },
    "atari": {
        "label": "Atari",
        "description": "Atari 2600 environments via ALE",
        "environments": ATARI_ENVS,
    },
    "procgen": {
        "label": "Procgen",
        "description": "Procedurally generated benchmark games",
        "environments": PROCGEN_ENVS,
    },
    "bsuite": {
        "label": "BSuite",
        "description": "DeepMind Behaviour Suite diagnostic RL tasks",
        "environments": BSUITE_ENVS,
    },
    "dmlab": {
        "label": "DeepMind Lab",
        "description": "DeepMind Lab 3D navigation and puzzle tasks",
        "environments": _list_available_dmlab_levels(),
    },
}


def _build_env_catalog() -> dict[str, list[str]]:
    atari_envs = _list_available_atari_envs()
    robotics_envs = _list_available_robotics_envs()
    gym_envs = _preferred_available_envs(
        _list_available_gymnasium_envs(), DEFAULT_GYM_ENV_PREFERENCES
    )
    general_control_envs = _dedupe_envs(gym_envs, robotics_envs)
    atari_and_robotics_envs = _dedupe_envs(atari_envs[:80], robotics_envs)
    bsuite_envs = _list_available_bsuite_ids()
    dmlab_envs = _list_available_dmlab_levels()
    dreamer_envs = _dedupe_envs(
        DREAMER_ENVS, general_control_envs, PROCGEN_ENVS, bsuite_envs, dmlab_envs
    )
    planet_envs = _dedupe_envs(gym_envs, atari_envs[:80], robotics_envs, PROCGEN_ENVS)
    return {
        "dreamer": dreamer_envs,
        "dreamerv1": dreamer_envs,
        "dreamerv2": dreamer_envs,
        "planet": planet_envs,
        "rssm": planet_envs,
        "iris": atari_and_robotics_envs,
        "diamond": atari_and_robotics_envs,
        "genie": atari_and_robotics_envs,
        "dit": atari_and_robotics_envs,
        # I-JEPA/JEPA is intentionally omitted because it trains on image datasets
        # rather than online Gymnasium environments.
    }


ENVIRONMENTS_BY_MODEL = _build_env_catalog()
