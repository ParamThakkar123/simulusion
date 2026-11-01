from __future__ import annotations

import datetime
from collections import deque
import uuid
from typing import Any

from simulusion.envs._actions import clip_box_action
from simulusion.envs._contract import finalize_step_info
from simulusion.utils.gym_compat import gym
import numpy as np
from PIL import Image


class TimeLimit:
    """Terminate episodes after a fixed number of wrapper steps.

    If the wrapped environment does not provide a discount flag at timeout,
    the wrapper injects a default discount of `1.0` for downstream learners.
    """

    def __init__(self, env: Any, duration: int) -> None:
        self._env = env
        self._duration = duration
        self._step: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> tuple[Any, Any, bool, dict[str, Any]]:
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        timeout = self._step >= self._duration
        upstream_terminated = bool(info.get("terminated", False)) if info else False
        if timeout:
            done = True
            self._step = None
        info = finalize_step_info(
            info,
            done=done,
            terminated=upstream_terminated,
            truncated=timeout and not upstream_terminated,
            discount=np.array(
                1.0
                if timeout and not upstream_terminated
                else ((info or {}).get("discount", 0.0 if done else 1.0)),
                dtype=np.float32,
            ),
        )
        return obs, reward, done, info

    def reset(self, *, seed: int | None = None) -> Any:
        self._step = 0
        if seed is None:
            return self._env.reset()
        return self._env.reset(seed=seed)


class FrameStack:
    """Stack the most recent image observations along the channel axis.

    The wrapped environment must emit dict observations with an ``"image"`` key
    in ``(C, H, W)`` layout. Non-image keys such as ``"state"`` are passed
    through unchanged from the latest observation.
    """

    def __init__(self, env: Any, num_frames: int) -> None:
        if int(num_frames) < 1:
            raise ValueError("num_frames must be >= 1")
        self._env = env
        self._num_frames = int(num_frames)
        self._frames: deque[np.ndarray] = deque(maxlen=self._num_frames)
        image_space = env.observation_space["image"]
        if len(image_space.shape) != 3:
            raise ValueError(
                "FrameStack expects image observations with shape (C, H, W)."
            )
        channels, height, width = image_space.shape
        spaces = dict(env.observation_space.spaces)
        image_space_cls = image_space.__class__
        spaces["image"] = image_space_cls(
            low=0,
            high=255,
            shape=(channels * self._num_frames, height, width),
            dtype=image_space.dtype,
        )
        dict_space_cls = env.observation_space.__class__
        self._observation_space = dict_space_cls(spaces)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    def _stack_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(obs["image"], dtype=np.uint8)
        if image.ndim != 3:
            raise ValueError(
                "FrameStack expects image observations with shape (C, H, W)."
            )
        stacked = np.concatenate(list(self._frames), axis=0)
        out = dict(obs)
        out["image"] = stacked.copy()
        return out

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        obs = self._env.reset() if seed is None else self._env.reset(seed=seed)
        image = np.asarray(obs["image"], dtype=np.uint8)
        self._frames.clear()
        for _ in range(self._num_frames):
            self._frames.append(image.copy())
        return self._stack_observation(obs)

    def step(self, action: Any) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        image = np.asarray(obs["image"], dtype=np.uint8)
        self._frames.append(image.copy())
        return self._stack_observation(obs), reward, done, info


class ActionRepeat:
    """Repeat each action for a fixed number of environment steps.

    Rewards are accumulated and the loop stops early if the environment
    terminates, mirroring common action-repeat behavior in world model papers.
    """

    def __init__(self, env: Any, amount: int) -> None:
        self._env = env
        self._amount = amount

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        done = False
        total_reward = 0
        current_step = 0
        info: dict[str, Any] = {}
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        info = dict(info or {})
        info["action_repeat"] = current_step
        return obs, total_reward, done, info


class NormalizeActions:
    """Expose a normalized `[-1, 1]` action space for bounded continuous controls.

    Incoming normalized actions are mapped back to the wrapped environment
    action bounds before stepping the environment.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def action_space(self) -> gym.spaces.Box:
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[Any, Any, bool, dict[str, Any]]:
        normalized = clip_box_action(
            action,
            -np.ones_like(self._low, dtype=np.float32),
            np.ones_like(self._high, dtype=np.float32),
        )
        original = (normalized + 1.0) / 2.0 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, normalized).astype(
            np.float32, copy=False
        )
        obs, reward, done, info = self._env.step(original)
        info = dict(info or {})
        if "action" in info and "executed_action" not in info:
            existing = info["action"]
            info["executed_action"] = (
                int(np.asarray(existing).item())
                if np.isscalar(existing)
                else np.asarray(existing).copy()
            )
        info["action"] = normalized.copy()
        return obs, reward, done, info


class ObsDict:
    """Convert scalar/array observations into a dictionary observation format.

    This harmonizes outputs for code paths that expect keyed observations
    (for example `{"image": ...}` style world model inputs).
    """

    def __init__(self, env: Any, key: str = "obs") -> None:
        self._env = env
        self._key = key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    def step(self, action: Any) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        result = self._env.reset() if seed is None else self._env.reset(seed=seed)
        obs = result[0] if isinstance(result, tuple) else result
        obs = {self._key: np.array(obs)}
        return obs


class OneHotAction:
    """Wrap discrete-action environments to accept one-hot action vectors.

    The wrapper validates one-hot inputs and converts them to integer action
    indices before forwarding to the underlying environment.
    """

    def __init__(self, env: Any) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env
        self._random = np.random.default_rng()
        shape = (self._env.action_space.n,)
        self._action_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )
        self._action_space.sample = self._sample_action

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def step(self, action: np.ndarray) -> tuple[Any, Any, bool, dict[str, Any]]:
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step(index)

    def reset(self, *, seed: int | None = None) -> Any:
        if seed is not None:
            self._random = np.random.default_rng(seed)
            if hasattr(self._action_space, "seed"):
                try:
                    self._action_space.seed(seed)
                except Exception:
                    pass
            return self._env.reset(seed=seed)
        return self._env.reset()

    def _sample_action(self) -> np.ndarray:
        actions = self._env.action_space.n
        index = int(self._random.integers(0, actions))
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:
    """Augment observations with the latest scalar reward under `obs["reward"]`.

    Useful for agents that consume reward as part of the observation stream
    during model learning or recurrent policy inference.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        spaces = dict(self._env.observation_space.spaces)
        assert "reward" not in spaces
        spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action: Any) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        obs["reward"] = reward
        return obs, reward, done, info

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        result = self._env.reset() if seed is None else self._env.reset(seed=seed)
        obs = result[0] if isinstance(result, tuple) else result
        obs["reward"] = 0.0
        return obs


class ResizeImage:
    """Resize image-like observation entries to a target spatial size.

    The wrapper discovers image keys from `env.obs_space`, applies nearest
    neighbor resizing, and updates the advertised observation space shapes.
    """

    def __init__(self, env: Any, size: tuple[int, int] = (64, 64)) -> None:
        self._env = env
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f"Resizing keys {','.join(self._keys)} to {self._size}.")
        if self._keys:
            self._Image = Image

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name)

    @property
    def obs_space(self) -> dict[str, Any]:
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action: Any) -> Any:
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self, *, seed: int | None = None) -> Any:
        obs = self._env.reset() if seed is None else self._env.reset(seed=seed)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image: np.ndarray) -> np.ndarray:
        img = self._Image.fromarray(image)
        img = img.resize(self._size, self._Image.Resampling.NEAREST)
        return np.array(img)


class RenderImage:
    """Inject RGB renders from `env.render("rgb_array")` into observations.

    This is useful when the base environment returns non-image observations
    but a rendered camera view is needed for world-model training.
    """

    def __init__(self, env: Any, key: str = "image") -> None:
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name)

    @property
    def obs_space(self) -> dict[str, Any]:
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action: Any) -> Any:
        obs = self._env.step(action)
        obs[self._key] = self._env.render("rgb_array")
        return obs

    def reset(self, *, seed: int | None = None) -> Any:
        obs = self._env.reset() if seed is None else self._env.reset(seed=seed)
        obs[self._key] = self._env.render("rgb_array")
        return obs


class UUID(gym.Wrapper):
    """Gym wrapper that tracks a unique run identifier per environment reset.

    The ID combines timestamp and UUID and can be used to tag episodes or
    artifacts generated during data collection.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self, **kwargs: Any) -> Any:
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset(**kwargs)


class SelectAction(gym.Wrapper):
    """Gym wrapper for dictionary actions that forwards a selected key only.

    This enables integration with policies that emit action dicts while the
    environment expects a single tensor/array action payload.
    """

    def __init__(self, env: Any, key: str) -> None:
        super().__init__(env)
        self._key = key

    def step(self, action: dict[str, Any]) -> Any:
        return self.env.step(action[self._key])
