from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from simulusion.envs._actions import clip_box_action
from simulusion.envs._contract import finalize_step_info
from simulusion.envs._observations import add_optional_state_space
import gymnasium as gym
import numpy as np
from PIL import Image


def make_brax_env(env: str | Any, **kwargs: Any) -> BraxImageEnv:
    """Create a Simulusion image wrapper for Brax environments.

    Args:
        env: Brax environment name (for example, ``"ant"``) or a pre-built Brax
            environment object exposing ``reset(rng)`` and ``step(state, action)``.
        **kwargs: Additional keyword arguments passed to :class:`BraxImageEnv`.

    Returns:
        BraxImageEnv: A Gym-like wrapper that returns ``{"image": (C, H, W)}``
        observations and exposes continuous actions in the Brax ``[-1, 1]`` range.
    """
    return BraxImageEnv(env=env, **kwargs)


class BraxImageEnv:
    """Gym-like adapter for training Simulusion world models on Brax tasks.

    Brax environments are functional JAX environments: ``reset`` consumes a PRNG
    key and returns a state, while ``step`` consumes the previous state plus an
    action and returns the next state. This adapter stores the Brax state between
    calls and converts state observations into image observations compatible with
    pixel-based Simulusion agents such as Dreamer.

    If a Brax renderer is not available, vector observations are rendered as
    deterministic feature-band images so training code can still consume a pixel
    stream. The original vector observation is also exposed through
    ``info["vector_observation"]`` after ``step`` for diagnostics. When
    ``include_state=True``, observations also expose a flattened ``"state"`` key.
    """

    def __init__(
        self,
        env: str | Any,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        backend: str | None = None,
        episode_length: int | None = None,
        auto_reset: bool = False,
        jit: bool = True,
        suppress_warp_warnings: bool = True,
        include_state: bool = False,
        **env_kwargs: Any,
    ) -> None:
        self._size = (int(size[0]), int(size[1]))
        self._seed = int(seed)
        self._jit = bool(jit)
        self._include_state = bool(include_state)
        self._state = None

        install_hint = "Install Brax support with `pip install simulusion[brax]`."
        self._jax = _require_module("jax", install_hint)
        self._jnp = _require_module("jax.numpy", install_hint)
        self._suppress_warp_warnings = bool(suppress_warp_warnings)
        self._brax_envs = _require_module(
            "brax.envs",
            install_hint,
            suppress_warp_warnings=self._suppress_warp_warnings,
        )

        self._env = self._make_env(
            env,
            backend=backend,
            episode_length=episode_length,
            auto_reset=auto_reset,
            env_kwargs=env_kwargs,
        )
        self._rng = self._jax.random.PRNGKey(self._seed)
        self._reset_fn = (
            self._jax.jit(self._env.reset) if self._jit else self._env.reset
        )
        self._step_fn = self._jax.jit(self._env.step) if self._jit else self._env.step

        action_size = getattr(self._env, "action_size", None)
        if action_size is None:
            action_size = getattr(self._env, "action_size", None)
        if action_size is None:
            raise ValueError("Brax environment must define an action_size attribute.")
        self._action_size = int(action_size)
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._action_size,),
            dtype=np.float32,
        )
        state_space = None
        if self._include_state:
            obs_size = getattr(self._env, "observation_size", None)
            if obs_size is not None:
                state_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(int(obs_size),),
                    dtype=np.float32,
                )
        self._observation_space = add_optional_state_space(
            gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, self._size[0], self._size[1]),
                dtype=np.uint8,
            ),
            state_space=state_space,
        )
        self._seed_spaces(self._seed)

    def _make_env(
        self,
        env: str | Any,
        backend: str | None,
        episode_length: int | None,
        auto_reset: bool,
        env_kwargs: dict[str, Any],
    ) -> Any:
        if not isinstance(env, str):
            return env

        kwargs = dict(env_kwargs)
        if backend is not None:
            kwargs.setdefault("backend", backend)

        if episode_length is None:
            return self._brax_envs.get_environment(env, **kwargs)

        return self._brax_envs.create(
            env_name=env,
            episode_length=int(episode_length),
            action_repeat=1,
            auto_reset=bool(auto_reset),
            **kwargs,
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        for name in ("episode_length", "eps_length", "_episode_length"):
            value = getattr(self._env, name, None)
            if value is not None:
                return int(value)
        return 1000

    def _split_key(self) -> Any:
        self._rng, key = self._jax.random.split(self._rng)
        return key

    def _seed_spaces(self, seed: int | None) -> None:
        if seed is None:
            return
        for space in (self._action_space, self._observation_space):
            if hasattr(space, "seed"):
                try:
                    space.seed(seed)
                except Exception:
                    pass

    def _to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(self._jax.device_get(value))

    def _vector_to_image(self, vector: Any) -> np.ndarray:
        vec: np.ndarray = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        finite = np.isfinite(vec)
        if not finite.all():
            vec = np.where(finite, vec, 0.0)
        vmin = float(vec.min())
        vmax = float(vec.max())
        if vmax > vmin:
            vec_norm = (vec - vmin) / (vmax - vmin)
            vec = vec_norm
        else:
            vec = np.zeros_like(vec)

        image = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        bands = min(16, vec.size)
        band_w = max(1, self._size[1] // max(1, bands))
        for i in range(bands):
            start = i * band_w
            end = (
                self._size[1] if i == bands - 1 else min(self._size[1], start + band_w)
            )
            image[:, start:end, :] = int(255.0 * float(vec[i]))
        return image

    def _obs_to_hwc_image(self, obs: Any) -> np.ndarray:
        arr = self._to_numpy(obs)
        if arr.ndim == 1:
            image = self._vector_to_image(arr)
        elif arr.ndim == 2:
            image = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3:
            image = arr
            if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
                image = image.transpose(1, 2, 0)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
        else:
            image = self._vector_to_image(arr)

        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0 and image.min() >= 0.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
        return image

    def _to_chw_uint8_image(self, obs: Any) -> np.ndarray:
        image = self._obs_to_hwc_image(obs)
        if image.shape[0] != self._size[0] or image.shape[1] != self._size[1]:
            image = np.array(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.Resampling.BILINEAR
                )
            )
        return image.transpose(2, 0, 1).copy()

    def _state_to_obs(self, state: Any) -> dict[str, Any]:
        image = self._to_chw_uint8_image(state.obs)
        observation = {"image": image}
        if self._include_state:
            observation["state"] = (
                self._to_numpy(state.obs).astype(np.float32).reshape(-1).copy()
            )
        return observation

    def _metrics_to_info(self, state: Any) -> dict[str, Any]:
        info = {}
        for source_name in ("metrics", "info"):
            source = getattr(state, source_name, None)
            if isinstance(source, dict):
                for key, value in source.items():
                    try:
                        info[key] = self._to_numpy(value)
                    except TypeError:
                        info[key] = value
        return info

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._seed = int(seed)
            self._rng = self._jax.random.PRNGKey(self._seed)
            self._seed_spaces(self._seed)
        self._state = None
        self._state = self._reset_fn(self._split_key())
        return self._state_to_obs(self._state)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Must call reset() before step().")

        clipped = clip_box_action(
            action,
            -np.ones((self._action_size,), dtype=np.float32),
            np.ones((self._action_size,), dtype=np.float32),
        )
        brax_action = self._jnp.asarray(clipped)
        self._state = self._step_fn(self._state, brax_action)

        reward = float(np.asarray(self._to_numpy(self._state.reward)).reshape(()))
        done = bool(np.asarray(self._to_numpy(self._state.done)).reshape(()))
        info = finalize_step_info(
            self._metrics_to_info(self._state),
            done=done,
            terminated=done,
            truncated=False,
        )
        info["action"] = clipped.copy()
        info["executed_action"] = clipped.copy()
        info["vector_observation"] = self._to_numpy(self._state.obs).copy()
        return self._state_to_obs(self._state), reward, done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._to_chw_uint8_image(self._state.obs).transpose(1, 2, 0).copy()

    def close(self) -> None:
        self._state = None


def _require_module(
    module_name: str, install_hint: str, *, suppress_warp_warnings: bool = False
) -> Any:
    parent_name = module_name.split(".", 1)[0]
    if importlib.util.find_spec(parent_name) is None:
        raise ImportError(
            f"Missing optional dependency `{parent_name}`. {install_hint}"
        )
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Missing optional dependency `{module_name}`. {install_hint}"
        )
    # Some optional backends (notably MuJoCo/MJX's Warp shim) print noisy
    # import-time messages like:
    #   Failed to import warp: No module named 'warp'
    #   Failed to import mujoco_warp: No module named 'mujoco_warp'
    # These messages are harmless when the optional backend is not present
    # but pollute console output during tests and normal runs. When
    # `suppress_warp_warnings=True` filter those two lines while replaying
    # any other import output.
    if suppress_warp_warnings and module_name.startswith("brax"):
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        buf = io.StringIO()
        # Capture stdout/stderr produced during import.
        with redirect_stdout(buf), redirect_stderr(buf):
            module = importlib.import_module(module_name)

        # Replay any captured lines except the known Warp messages.
        captured = buf.getvalue().splitlines()
        original_stdout = sys.stdout
        for line in captured:
            if line.startswith("Failed to import warp:"):
                continue
            if line.startswith("Failed to import mujoco_warp:"):
                continue
            print(line, file=original_stdout)
        return module

    return importlib.import_module(module_name)
