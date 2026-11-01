from __future__ import annotations

from typing import Any
import warnings

from simulusion.envs._actions import clip_box_action, encode_discrete_action
from simulusion.envs._contract import finalize_step_info
from simulusion.envs._observations import (
    add_optional_state_space,
    flatten_vector_observation,
    infer_state_space_from_observation_space,
)
from simulusion.utils.gym_compat import gym, import_gymnasium
import numpy as np


from PIL import Image


def make_gym_env(env: Any, **kwargs: Any) -> Any:
    """Create a GymImageEnv wrapper for generic Gym/Gymnasium environments.

    Args:
        env: Either a string environment ID (e.g., "Pendulum-v1") or a pre-built
            gym environment instance.
        **kwargs: Additional keyword arguments passed to GymImageEnv, including:
            - seed (int): Random seed for environment (default: 0)
            - size (tuple): Target image size as (height, width) (default: (64, 64))
            - render_mode (str): Render mode for environment (default: "rgb_array")

    Returns:
        GymImageEnv: A wrapper that always returns image observations in the
            format {"image": (C, H, W)} suitable for pixel-based world models.
    """
    return GymImageEnv(env=env, **kwargs)


class GymImageEnv:
    """Gym-like environment wrapper that always returns image observations.

    This wrapper normalizes diverse environment interfaces to return consistent
    image-based observations suitable for pixel-based world models like Dreamer.

    Features:
        - Supports environment IDs (string) and pre-built environment objects.
        - For vector observations, uses the environment's own ``render()`` frame,
          and only synthesizes band images from the vector when rendering fails.
        - Exposes continuous action spaces mapped to [-1, 1] range.
        - Converts discrete actions to one-hot vectors.
        - Returns observations as dicts with required key ``"image"`` and optional
          key ``"state"`` when ``include_state=True`` and a vector observation is available.

    Args:
        env: Either a string environment ID (e.g., "Pendulum-v1") or a pre-built
            gym environment instance.
        seed (int): Random seed for environment reset (default: 0).
        size (tuple): Target image size as (height, width) (default: (64, 64)).
        render_mode (str): Render mode for environment (default: "rgb_array").
        include_state (bool): Include a flattened low-dimensional ``"state"`` key
            in observations when one can be derived from the underlying env.

    Attributes:
        observation_space: Dict space with required ``"image"`` key and optional
            ``"state"`` key when enabled.
        action_space: Box space with actions in [-1, 1] range.
        max_episode_steps: Maximum steps per episode (default: 1000).
    """

    def __init__(
        self,
        env: Any,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        render_mode: str = "rgb_array",
        include_state: bool = False,
    ) -> None:
        self._size = (int(size[0]), int(size[1]))
        self._seed = seed
        self._render_mode = render_mode
        self._include_state = bool(include_state)
        self._seed_applied = False
        self._rng = np.random.default_rng(seed)

        if isinstance(env, str):
            self._env = self._make_env_from_id(env, render_mode)
        else:
            self._env = env

        self._last_obs: Any = None
        self._last_image: Any = None
        self._warned_band_fallback = False

        action_space = getattr(self._env, "action_space", None)
        if action_space is None:
            raise ValueError("Wrapped environment must define action_space.")

        self._discrete_n = int(action_space.n) if hasattr(action_space, "n") else None

        if self._discrete_n is None:
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            self._action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self._action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self._discrete_n,), dtype=np.float32
            )
            self._action_space.sample = self._sample_discrete_action

        self._seed_spaces(seed)
        self._state_space = (
            infer_state_space_from_observation_space(
                getattr(self._env, "observation_space", None)
            )
            if self._include_state
            else None
        )

        self._observation_space = add_optional_state_space(
            gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, self._size[0], self._size[1]),
                dtype=np.uint8,
            ),
            state_space=self._state_space,
        )

    def _make_env_from_id(self, env_id: str, render_mode: str) -> Any:
        # Prefer gymnasium if available, then fallback to gym.
        gymnasium = import_gymnasium()
        if gymnasium is not None:
            try:
                return gymnasium.make(env_id, render_mode=render_mode)
            except ImportError as exc:
                from simulusion.envs.robotics_env import (
                    is_moved_mujoco_error,
                    register_gymnasium_robotics_envs,
                )

                if not is_moved_mujoco_error(exc):
                    raise
                register_gymnasium_robotics_envs()
                try:
                    return gymnasium.make(env_id, render_mode=render_mode)
                except TypeError:
                    return gymnasium.make(env_id)
            except TypeError:
                return gymnasium.make(env_id)

        try:
            return gym.make(env_id, render_mode=render_mode)
        except TypeError:
            return gym.make(env_id)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        if (
            hasattr(self._env, "_max_episode_steps")
            and self._env._max_episode_steps is not None
        ):
            return int(self._env._max_episode_steps)
        if (
            getattr(self._env, "spec", None) is not None
            and getattr(self._env.spec, "max_episode_steps", None) is not None
        ):
            return int(self._env.spec.max_episode_steps)
        return 1000

    def _seed_spaces(self, seed: int | None) -> None:
        if seed is None:
            return
        for space in (
            self._action_space,
            getattr(self._env, "action_space", None),
            getattr(self._env, "observation_space", None),
        ):
            if space is not None and hasattr(space, "seed"):
                try:
                    space.seed(seed)
                except Exception:
                    pass

    def _sample_discrete_action(self) -> np.ndarray:
        assert self._discrete_n is not None
        idx = int(self._rng.integers(0, self._discrete_n))
        action = -np.ones((self._discrete_n,), dtype=np.float32)
        action[idx] = 1.0
        return action

    def _vector_to_image(self, vector: Any) -> np.ndarray:
        vec: np.ndarray = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        vmin = float(vec.min())
        vmax = float(vec.max())
        if vmax > vmin:
            vec_norm = (vec - vmin) / (vmax - vmin)
            vec = vec_norm
        else:
            vec = np.zeros_like(vec)

        image = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        bands = min(8, vec.size)
        band_w = max(1, self._size[1] // max(1, bands))
        for i in range(bands):
            start = i * band_w
            end = min(self._size[1], start + band_w)
            image[:, start:end, :] = int(255.0 * float(vec[i]))
        return image

    def _obs_to_hwc_image(self, obs: Any) -> np.ndarray | None:
        if isinstance(obs, tuple):
            obs = obs[0]

        if isinstance(obs, dict):
            for key in ("image", "pixels", "rgb", "observation"):
                if key in obs:
                    candidate = np.asarray(obs[key])
                    if candidate.ndim in (1, 2, 3):
                        return self._obs_to_hwc_image(candidate)
            for value in obs.values():
                candidate = np.asarray(value)
                if candidate.ndim in (1, 2, 3):
                    return self._obs_to_hwc_image(candidate)
            return None

        arr = np.asarray(obs)
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
            return None

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
        return image

    def _render_hwc_image(self, last_obs: Any = None) -> np.ndarray | None:
        frame = None
        try:
            frame = self._env.render()
        except Exception:
            frame = None
        if isinstance(frame, tuple):
            frame = frame[0]
        if isinstance(frame, np.ndarray):
            return frame

        try:
            frame = self._env.render(mode=self._render_mode)
        except Exception:
            frame = None
        if isinstance(frame, tuple):
            frame = frame[0]
        if isinstance(frame, np.ndarray):
            return frame

        if last_obs is not None:
            return self._obs_to_hwc_image(last_obs)
        return None

    @staticmethod
    def _is_vector_observation(obs: Any) -> bool:
        """True when ``obs`` carries no pixels, only low-dimensional numbers."""
        if isinstance(obs, tuple):
            obs = obs[0]
        values = obs.values() if isinstance(obs, dict) else (obs,)
        for value in values:
            try:
                if np.asarray(value).ndim >= 2:
                    return False
            except Exception:
                continue
        return True

    def _rendered_frame(self) -> np.ndarray | None:
        frame = self._render_hwc_image()
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            return None
        return self._obs_to_hwc_image(frame)

    def _to_chw_uint8_image(self, obs: Any) -> np.ndarray:
        image = None
        if self._is_vector_observation(obs):
            # A state vector has no pixels of its own. The env's renderer draws
            # the actual scene (Pendulum's pole, CartPole's cart); painting the
            # numbers as grey bands is a last resort, and a world model trained
            # on bands learns nothing a person can look at.
            image = self._rendered_frame()
            if image is None and not self._warned_band_fallback:
                warnings.warn(
                    "GymImageEnv: the environment returns vector observations and "
                    "render() produced no RGB frame, so observations are synthesized "
                    "as grey bands (one per state value). Create the env with "
                    "render_mode='rgb_array' to train on real pixels.",
                    stacklevel=3,
                )
                self._warned_band_fallback = True
        if image is None:
            image = self._obs_to_hwc_image(obs)
        if image is None:
            image = self._render_hwc_image(last_obs=obs)
        if image is None:
            raise RuntimeError(
                "Failed to obtain an RGB frame from environment observation or render()."
            )

        if image.shape[0] != self._size[0] or image.shape[1] != self._size[1]:
            image = np.array(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.Resampling.BILINEAR
                )
            )
        return image.transpose(2, 0, 1).copy()

    def _to_native_action(
        self, action: Any
    ) -> tuple[np.ndarray, np.ndarray] | tuple[int, np.ndarray]:
        if self._discrete_n is None:
            low = np.asarray(self._env.action_space.low, dtype=np.float32)
            high = np.asarray(self._env.action_space.high, dtype=np.float32)
            clipped = clip_box_action(action, low, high)
            return clipped, clipped.copy()

        idx, encoded = encode_discrete_action(action, self._discrete_n)
        return idx, encoded

    def _build_observation(self, raw_obs: Any, image: np.ndarray) -> dict[str, Any]:
        observation = {"image": image}
        vector_observation = flatten_vector_observation(raw_obs)
        if self._state_space is not None:
            if vector_observation is None:
                raise RuntimeError(
                    "include_state=True but no low-dimensional state could be extracted."
                )
            observation["state"] = vector_observation.copy()
        return observation

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
            self._seed_spaces(self._seed)
            self._seed_applied = False
        self._last_obs = None
        self._last_image = None
        if not self._seed_applied:
            try:
                result = self._env.reset(seed=self._seed)
            except TypeError:
                result = self._env.reset()
            self._seed_applied = True
        else:
            result = self._env.reset()

        obs = result[0] if isinstance(result, tuple) else result
        self._last_obs = obs
        image = self._to_chw_uint8_image(obs)
        self._last_image = image
        return self._build_observation(obs, image)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        native_action, model_action = self._to_native_action(action)
        result = self._env.step(native_action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
            done = bool(done)
            truncated = bool(info.get("TimeLimit.truncated", False)) if info else False
            terminated = bool(done and not truncated)

        info = finalize_step_info(
            info,
            done=done,
            terminated=terminated,
            truncated=truncated,
        )
        info["action"] = np.asarray(model_action, dtype=np.float32).copy()
        info["executed_action"] = (
            int(np.asarray(native_action).item())
            if np.isscalar(native_action)
            else np.asarray(native_action, dtype=np.float32).copy()
        )
        vector_observation = flatten_vector_observation(obs)
        if vector_observation is not None:
            info["vector_observation"] = vector_observation.copy()

        self._last_obs = obs
        image = self._to_chw_uint8_image(obs)
        self._last_image = image
        return self._build_observation(obs, image), float(reward), done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        frame = self._render_hwc_image(last_obs=self._last_obs)
        if frame is None:
            if self._last_image is None:
                raise RuntimeError("No frame available. Call reset() before render().")
            return self._last_image.transpose(1, 2, 0).copy()
        return frame

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()
