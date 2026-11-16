"""Craftax adapter for Simulusion agents.

Craftax (Matthews et al., 2024) reimplements Crafter in JAX and extends it with
NetHack-style dungeon floors, combat and magic. Running entirely on the
accelerator, it is orders of magnitude faster than Crafter.

Four variants are supported:

==============================  ==============================  =======  ==========
Name                            Observation                     Actions  Max steps
==============================  ==============================  =======  ==========
``Craftax-Classic-Pixels-v1``   63x63 RGB, resized to ``size``  17       10000
``Craftax-Classic-Symbolic-v1`` 1345-d vector                   17       10000
``Craftax-Pixels-v1``           130x110 RGB, resized            43       100000
``Craftax-Symbolic-v1``         8268-d vector                   43       100000
==============================  ==============================  =======  ==========

Craftax-Classic has Crafter's 22 achievements, so
:func:`simulusion.envs.crafter_env.crafter_score` scores it from the
``info["achievements"]`` dicts this adapter returns.

This adapter steps one environment at a time with jitted ``reset`` and
``step`` functions, so it plugs into the same training loops as every other
backend. Throughput is still bounded by Python round trips; batched training
on the accelerator should call the Craftax API directly.

Pixel variants produce ``{"image": (3, H, W) uint8}``; symbolic variants
produce ``{"state": float32 vector}``. Actions are one-hot-like. Death (and, in
full Craftax, beating the final boss) is a termination; reaching the step
limit is a truncation.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from simulusion.envs._actions import encode_discrete_action
from simulusion.envs._contract import finalize_step_info
from simulusion.envs._images import to_chw_uint8, to_hwc_uint8

CRAFTAX_ENVS = (
    "Craftax-Classic-Pixels-v1",
    "Craftax-Classic-Symbolic-v1",
    "Craftax-Pixels-v1",
    "Craftax-Symbolic-v1",
)

_ALIASES = {
    "classic": "Craftax-Classic-Pixels-v1",
    "classic-pixels": "Craftax-Classic-Pixels-v1",
    "classic-symbolic": "Craftax-Classic-Symbolic-v1",
    "craftax": "Craftax-Pixels-v1",
    "full": "Craftax-Pixels-v1",
    "pixels": "Craftax-Pixels-v1",
    "symbolic": "Craftax-Symbolic-v1",
}


def normalize_craftax_env(name: str) -> str:
    """Resolve a Craftax environment name or alias (``"classic"``, ``"full"``, ...)."""
    key = str(name).strip()
    for env_name in CRAFTAX_ENVS:
        if key.lower() == env_name.lower():
            return env_name
    alias = key.lower().replace("_", "-")
    if alias.startswith("craftax-") and alias != "craftax":
        alias = alias[len("craftax-") :]
    if alias in _ALIASES:
        return _ALIASES[alias]
    valid = ", ".join(CRAFTAX_ENVS)
    raise ValueError(f"Unknown Craftax environment {name!r}. Valid: {valid}.")


def _require_craftax() -> tuple[Any, Any]:
    try:
        jax = importlib.import_module("jax")
        craftax_env = importlib.import_module("craftax.craftax_env")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Craftax backend requires the 'craftax' package (and JAX). "
            "Install it with `pip install simulusion[craftax]`."
        ) from exc
    return jax, craftax_env


def _achievement_names(env_name: str) -> list[str]:
    module = (
        "craftax.craftax_classic.constants"
        if "Classic" in env_name
        else "craftax.craftax.constants"
    )
    achievement = importlib.import_module(module).Achievement
    ordered = sorted(achievement, key=lambda item: item.value)
    return [item.name.lower() for item in ordered]


class CraftaxEnv:
    """One Craftax environment with Simulusion's observation and action contract.

    Args:
        env: Variant name, see :func:`normalize_craftax_env`.
        seed: Seed for the JAX PRNG key that drives world generation and
            environment randomness.
        size: Output image size ``(height, width)`` for pixel variants.
        jit: Compile ``reset`` and ``step`` with ``jax.jit`` (default).
    """

    def __init__(
        self,
        env: str = "Craftax-Classic-Pixels-v1",
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        jit: bool = True,
    ) -> None:
        from simulusion.envs._spaces import OneHotActionSpace, image_space
        from simulusion.utils.gym_compat import gym

        jax, craftax_env = _require_craftax()
        self._jax = jax
        self.env_name = normalize_craftax_env(env)
        self.pixels = "Pixels" in self.env_name
        self._size = (int(size[0]), int(size[1]))
        self._env = craftax_env.make_craftax_env_from_name(
            self.env_name, auto_reset=False
        )
        self._params = self._env.default_params
        self._max_steps = int(self._params.max_timesteps)
        self._achievements = _achievement_names(self.env_name)

        params = self._params
        reset_fn = lambda key: self._env.reset(key, params)  # noqa: E731
        step_fn = lambda key, state, action: self._env.step(  # noqa: E731
            key, state, action, params
        )
        self._reset_fn = jax.jit(reset_fn) if jit else reset_fn
        self._step_fn = jax.jit(step_fn) if jit else step_fn

        self._num_actions = int(self._env.action_space(params).n)
        self._seed = int(seed)
        self._key = jax.random.PRNGKey(self._seed)
        self._state: Any = None
        self._last_frame: np.ndarray | None = None
        self._action_space = OneHotActionSpace(self._num_actions, seed=self._seed)
        if self.pixels:
            spaces: dict[str, Any] = {"image": image_space(self._size)}
        else:
            obs_shape = tuple(self._env.observation_space(params).shape)
            spaces = {"state": gym.spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)}
        self._observation_space = gym.spaces.Dict(spaces)

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    @property
    def achievement_names(self) -> list[str]:
        return list(self._achievements)

    def _next_key(self) -> Any:
        self._key, key = self._jax.random.split(self._key)
        return key

    def _observe(self, obs: Any) -> dict[str, np.ndarray]:
        array = np.asarray(obs)
        if self.pixels:
            self._last_frame = to_hwc_uint8(array)
            return {"image": to_chw_uint8(self._last_frame, self._size)}
        return {"state": array.astype(np.float32, copy=False).reshape(-1)}

    def _achievement_info(self) -> dict[str, int]:
        unlocked = np.asarray(self._state.achievements).reshape(-1)
        return {
            name: int(bool(unlocked[i])) for i, name in enumerate(self._achievements)
        }

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._seed = int(seed)
            self._key = self._jax.random.PRNGKey(self._seed)
            self._action_space.seed(self._seed)
        obs, self._state = self._reset_fn(self._next_key())
        return self._observe(obs)

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        index, model_action = encode_discrete_action(action, self._num_actions)
        obs, self._state, reward, done, raw_info = self._step_fn(
            self._next_key(), self._state, index
        )
        done = bool(done)
        timestep = int(self._state.timestep)
        alive = float(self._state.player_health) > 0
        # The step limit is the only non-terminal way an episode ends.
        truncated = done and alive and timestep >= self._max_steps
        terminated = done and not truncated
        info = finalize_step_info(
            {
                "achievements": self._achievement_info(),
                "timestep": timestep,
                "action": model_action,
                "executed_action": index,
            },
            done=done,
            terminated=terminated,
            truncated=truncated,
        )
        if done and "score" in raw_info:
            info["craftax_score"] = float(raw_info["score"])
        observation = self._observe(obs)
        if not self.pixels:
            info["vector_observation"] = observation["state"].copy()
        return observation, float(reward), done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if not self.pixels:
            raise RuntimeError(
                f"{self.env_name} is a symbolic variant and has no image to render."
            )
        if self._last_frame is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._last_frame.copy()

    def close(self) -> None:
        self._state = None


def make_craftax_env(
    env_id: str = "Craftax-Classic-Pixels-v1", **kwargs: Any
) -> CraftaxEnv:
    """Create a Craftax environment, e.g. ``make_craftax_env("Craftax-Classic-Pixels-v1")``."""
    return CraftaxEnv(env_id, **kwargs)


__all__ = ["CRAFTAX_ENVS", "CraftaxEnv", "make_craftax_env", "normalize_craftax_env"]
