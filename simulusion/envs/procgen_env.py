"""Procgen environment adapter for Simulusion image-based agents."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any

from simulusion.envs._actions import encode_discrete_action
from simulusion.envs._contract import finalize_step_info
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from PIL import Image

_PROCGEN_PACKAGE = "procgen"

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


def _require_procgen_env_class() -> type[Any]:
    """Return the Procgen vector environment class with a helpful install error."""
    try:
        package_spec = importlib.util.find_spec(_PROCGEN_PACKAGE)
    except ValueError:
        package_spec = None
    if package_spec is None and _PROCGEN_PACKAGE not in sys.modules:
        install_hint = "Install it with `pip install simulusion[procgen]` or `pip install procgen`."
        if sys.version_info >= (3, 11):
            install_hint += (
                " Upstream Procgen wheels currently support Python 3.10 and below."
            )
        raise ImportError(
            "Procgen support requires the optional 'procgen' package. " + install_hint
        )

    if _PROCGEN_PACKAGE in sys.modules:
        module = sys.modules[_PROCGEN_PACKAGE]
    else:
        module = importlib.import_module(_PROCGEN_PACKAGE)

    for attr in ("ProcgenEnv", "ProcgenGym3Env"):
        env_cls = getattr(module, attr, None)
        if env_cls is not None:
            return env_cls

    raise ImportError(
        "The installed 'procgen' package did not expose ProcgenEnv or ProcgenGym3Env. "
        "Check that a supported Procgen wheel is installed for this Python version."
    )


def _unbatch_procgen_info(info: Any) -> dict[str, Any]:
    """Normalize Procgen vector info to a single-environment info dict."""
    if isinstance(info, (list, tuple)):
        return dict(info[0]) if info else {}
    if not isinstance(info, dict):
        return {}

    unbatched: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            unbatched[key] = value[0]
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            unbatched[key] = value[0]
        else:
            unbatched[key] = value
    return unbatched


def list_procgen_envs() -> list[str]:
    """Return the Procgen game names understood by :class:`ProcgenImageEnv`."""
    return list(PROCGEN_ENVS)


def normalize_procgen_env_name(env: str) -> str:
    """Normalize Procgen Gym ids and shorthand names to Procgen game names.

    Accepted forms include ``"coinrun"``, ``"procgen-coinrun-v0"``, and
    ``"procgen:procgen-coinrun-v0"``.
    """
    name = str(env).strip()
    if ":" in name:
        name = name.split(":", 1)[1]
    if name.startswith("procgen-"):
        name = name[len("procgen-") :]
    if name.endswith("-v0"):
        name = name[: -len("-v0")]
    if name not in PROCGEN_ENVS:
        valid = ", ".join(PROCGEN_ENVS)
        raise ValueError(f"Unknown Procgen environment '{env}'. Valid names: {valid}.")
    return name


class _ProcgenActionSpace(gym.spaces.Box):
    """One-hot-like continuous action space for discrete Procgen actions."""

    def __init__(self, n: int, seed: int = 0):
        self.n = int(n)
        self._rng = np.random.default_rng(int(seed))
        super().__init__(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is None:
            seed = int(self._rng.integers(0, 2**31 - 1))
        self._rng = np.random.default_rng(int(seed))
        return [int(seed)]

    def sample(self, mask: Any = None, probability: Any = None) -> NDArray[np.float32]:
        del mask, probability
        idx = int(self._rng.integers(0, self.n))
        action: NDArray[np.float32] = -np.ones((self.n,), dtype=np.float32)
        action[idx] = 1.0
        return action


def make_procgen_env(env: str, **kwargs: Any) -> "ProcgenImageEnv":
    """Create a single-environment Procgen adapter.

    Args:
        env: Procgen game name or Gym-style id.
        **kwargs: Options forwarded to :class:`ProcgenImageEnv`.

    Returns:
        ProcgenImageEnv: Simulusion-compatible image wrapper exposing
        ``{"image": (3, H, W) uint8}`` observations and one-hot-like actions.
    """
    return ProcgenImageEnv(env=env, **kwargs)


class ProcgenImageEnv:
    """Adapt Procgen's vector API to Simulusion's single-env image interface.

    The upstream ``procgen.ProcgenEnv`` API is vectorized, so this wrapper builds
    a one-environment vector and unwraps the leading batch dimension. Actions are
    exposed as a continuous one-hot-like ``Box[-1, 1]`` with one element per
    discrete Procgen action, matching Simulusion's other discrete image adapters.
    """

    def __init__(
        self,
        env: str,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        distribution_mode: str = "easy",
        num_levels: int = 0,
        start_level: int | None = None,
        action_n: int = 15,
        **procgen_kwargs: Any,
    ):
        self._procgen_env_class = _require_procgen_env_class()

        self.env_name = normalize_procgen_env_name(env)
        self._size = (int(size[0]), int(size[1]))
        self._seed = int(seed)
        self._distribution_mode = distribution_mode
        self._num_levels = int(num_levels)
        self._procgen_kwargs = dict(procgen_kwargs)
        self._start_level = self._seed if start_level is None else int(start_level)
        self._last_image: np.ndarray | None = None
        self._last_obs: Any = None

        max_episode_steps = int(self._procgen_kwargs.pop("max_episode_steps", 1000))

        self._env = self._make_env(start_level=self._start_level)

        base_action_space = getattr(self._env, "action_space", None)
        if base_action_space is not None and hasattr(base_action_space, "n"):
            action_n = int(base_action_space.n)
        self._discrete_n = int(action_n)
        self._action_space = _ProcgenActionSpace(self._discrete_n, seed=self._seed)
        self._observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self._size[0], self._size[1]),
                    dtype=np.uint8,
                )
            }
        )
        self._max_episode_steps = max_episode_steps
        self._seed_spaces(self._seed)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> _ProcgenActionSpace:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def _make_env(self, *, start_level: int) -> Any:
        return self._procgen_env_class(
            num_envs=1,
            env_name=self.env_name,
            distribution_mode=self._distribution_mode,
            num_levels=self._num_levels,
            start_level=int(start_level),
            **self._procgen_kwargs,
        )

    def _seed_spaces(self, seed: int | None) -> None:
        if seed is None:
            return
        for space in (self._action_space, self._observation_space):
            if hasattr(space, "seed"):
                try:
                    space.seed(seed)
                except Exception:
                    pass

    def _to_native_action(self, action: Any) -> tuple[int, NDArray[np.float32]]:
        return encode_discrete_action(action, self._discrete_n)

    def _extract_rgb(self, obs: Any) -> np.ndarray:
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            for key in ("rgb", "image", "pixels", "observation"):
                if key in obs:
                    return self._extract_rgb(obs[key])
            for value in obs.values():
                candidate = np.asarray(value)
                if candidate.ndim in (3, 4):
                    return self._extract_rgb(candidate)
            raise RuntimeError("Procgen observation did not contain an RGB frame.")

        image = np.asarray(obs)
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise RuntimeError(
                "Expected Procgen RGB observation with 3 dimensions, "
                f"got {image.shape}."
            )
        if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
            image = image.transpose(1, 2, 0)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255.0
            image = image.clip(0, 255).astype(np.uint8)
        return image

    def _to_chw_uint8_image(self, obs: Any) -> NDArray[np.uint8]:
        image = self._extract_rgb(obs)
        if image.shape[0] != self._size[0] or image.shape[1] != self._size[1]:
            image = np.array(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.Resampling.BILINEAR
                )
            )
        return image.transpose(2, 0, 1).copy()

    def reset(self, seed: int | None = None) -> dict[str, NDArray[np.uint8]]:
        if seed is not None:
            self._seed = int(seed)
            self._start_level = self._seed
            if hasattr(self._env, "close"):
                self._env.close()
            self._env = self._make_env(start_level=self._start_level)
            self._seed_spaces(self._seed)
        self._last_obs = None
        self._last_image = None
        obs = self._env.reset()
        self._last_obs = obs
        image = self._to_chw_uint8_image(obs)
        self._last_image = image
        return {"image": image}

    def step(
        self, action: Any
    ) -> tuple[dict[str, NDArray[np.uint8]], float, bool, dict[str, Any]]:
        native_action, model_action = self._to_native_action(action)
        action_batch = np.asarray([native_action], dtype=np.int32)
        obs, reward, done, info = self._env.step(action_batch)
        done_value = bool(np.asarray(done).reshape(-1)[0])
        reward_value = float(np.asarray(reward).reshape(-1)[0])
        info_value = finalize_step_info(
            _unbatch_procgen_info(info),
            done=done_value,
        )
        info_value["action"] = np.asarray(model_action, dtype=np.float32).copy()
        info_value["executed_action"] = int(native_action)

        self._last_obs = obs
        image = self._to_chw_uint8_image(obs)
        self._last_image = image
        return {"image": image}, reward_value, done_value, info_value

    def render(self, *args: Any, **kwargs: Any) -> NDArray[np.uint8]:
        if self._last_image is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._last_image.transpose(1, 2, 0).copy()

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()
