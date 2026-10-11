from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from simulusion.envs._actions import encode_discrete_action
from simulusion.envs._contract import finalize_step_info
import gymnasium as gym
import numpy as np
from PIL import Image

RGB_OBSERVATION = "RGB_INTERLEAVED"

# Compact discrete action set commonly used by DMLab agents. Each row maps to
# DeepMind Lab's native 7-element action vector:
# [look_lr, look_ud, strafe_lr, move_fb, fire, jump, crouch].
DEFAULT_ACTION_SET = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],  # noop
        [20, 0, 0, 0, 0, 0, 0],  # look right
        [-20, 0, 0, 0, 0, 0, 0],  # look left
        [0, 0, 0, 1, 0, 0, 0],  # move forward
        [0, 0, 0, -1, 0, 0, 0],  # move backward
        [0, 0, -1, 0, 0, 0, 0],  # strafe left
        [0, 0, 1, 0, 0, 0, 0],  # strafe right
        [20, 0, 0, 1, 0, 0, 0],  # forward + look right
        [-20, 0, 0, 1, 0, 0, 0],  # forward + look left
    ],
    dtype=np.intc,
)


DMLAB_LEVELS = [
    "rooms_collect_good_objects_train",
    "rooms_collect_good_objects_test",
    "rooms_exploit_deferred_effects_train",
    "rooms_exploit_deferred_effects_test",
    "rooms_select_nonmatching_object",
    "rooms_watermaze",
    "rooms_keys_doors_puzzle",
    "language_select_described_object",
    "language_select_located_object",
    "language_execute_random_task",
    "nav_maze_static_01",
    "nav_maze_static_02",
    "nav_maze_random_goal_01",
    "nav_maze_random_goal_02",
    "lt_chasm",
]


def make_dmlab_env(level: str, **kwargs: Any) -> "DMLabEnv":
    """Create a DeepMind Lab environment adapter for Simulusion.

    Args:
        level: DeepMind Lab level name, for example
            ``"rooms_collect_good_objects_train"``.
        **kwargs: Additional keyword arguments passed to :class:`DMLabEnv`.

    Returns:
        DMLabEnv: A Gym-like wrapper returning ``{"image": (C, H, W)}`` uint8
        observations and normalized one-hot discrete actions.
    """
    return DMLabEnv(level=level, **kwargs)


class _OneHotActionSpace(gym.spaces.Box):
    """Box action space that samples normalized one-hot vectors."""

    def __init__(self, actions: int, seed: int = 0):
        self._actions = int(actions)
        self._rng = np.random.default_rng(int(seed))
        super().__init__(
            low=-1.0,
            high=1.0,
            shape=(self._actions,),
            dtype=np.float32,
        )

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is None:
            seed = int(self._rng.integers(0, 2**31 - 1))
        self._rng = np.random.default_rng(int(seed))
        return [int(seed)]

    def sample(self, mask: None = None, probability: None = None) -> np.ndarray:
        if mask is not None or probability is not None:
            raise ValueError("DMLab action sampling does not support masks.")
        action: np.ndarray = -np.ones((self._actions,), dtype=np.float32)
        action[int(self._rng.integers(0, self._actions))] = 1.0
        return action


class DMLabEnv:
    """Gym-style adapter for DeepMind Lab 3D environments.

    The native ``deepmind_lab`` API exposes RGB observations as HWC arrays and
    expects a seven-element integer action vector. This adapter presents a
    Simulusion-friendly image observation dict and a Box action space containing a
    one-hot vector in ``[-1, 1]`` so it composes with Dreamer's normalization
    wrappers.
    """

    def __init__(
        self,
        level: str,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        action_repeat: int = 4,
        action_set: Sequence[Sequence[int]] | np.ndarray | None = None,
        observations: Sequence[str] | None = None,
        config: dict[str, Any] | None = None,
        renderer: str = "hardware",
        **lab_kwargs: Any,
    ):
        self._level = str(level)
        self._seed = int(seed)
        self._episode = 0
        self._size = (int(size[0]), int(size[1]))
        self._action_repeat = int(action_repeat)
        if self._action_repeat < 1:
            raise ValueError("action_repeat must be >= 1.")

        self._action_set = np.asarray(
            DEFAULT_ACTION_SET if action_set is None else action_set, dtype=np.intc
        )
        if self._action_set.ndim != 2:
            raise ValueError("action_set must be a 2D array of DMLab action vectors.")

        self._observations = list(observations or [RGB_OBSERVATION])
        if RGB_OBSERVATION not in self._observations:
            self._observations.insert(0, RGB_OBSERVATION)

        lab_config = {"width": str(self._size[1]), "height": str(self._size[0])}
        if config:
            lab_config.update({str(key): str(value) for key, value in config.items()})
        self._config = lab_config

        deepmind_lab = _require_deepmind_lab()
        self._env = deepmind_lab.Lab(
            self._level,
            self._observations,
            config=self._config,
            renderer=renderer,
            **lab_kwargs,
        )
        self._last_obs: dict[str, np.ndarray] | None = None

        self._action_space = _OneHotActionSpace(
            self._action_set.shape[0], seed=self._seed
        )
        self._observation_space = self._build_observation_space()
        self._seed_spaces(self._seed)

    def _build_observation_space(self) -> gym.spaces.Dict:
        spaces: dict[str, gym.spaces.Space[Any]] = {
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, self._size[0], self._size[1]),
                dtype=np.uint8,
            )
        }
        spec_fn = getattr(self._env, "observation_spec", None)
        if spec_fn is not None:
            try:
                specs = spec_fn()
            except Exception:
                specs = []
            if isinstance(specs, dict):
                specs = [
                    {"name": key, **value} if isinstance(value, dict) else value
                    for key, value in specs.items()
                ]
            for spec in specs or []:
                name = (
                    spec.get("name")
                    if isinstance(spec, dict)
                    else getattr(spec, "name", None)
                )
                if not name or name == RGB_OBSERVATION:
                    continue
                shape = (
                    spec.get("shape")
                    if isinstance(spec, dict)
                    else getattr(spec, "shape", ())
                )
                dtype = (
                    spec.get("dtype")
                    if isinstance(spec, dict)
                    else getattr(spec, "dtype", np.float32)
                )
                np_dtype = np.dtype(dtype)
                if np.issubdtype(np_dtype, np.integer):
                    info = np.iinfo(np_dtype)  # type: ignore[type-var]
                    low: float = int(info.min)
                    high: float = int(info.max)
                else:
                    low, high = -np.inf, np.inf
                shape_t = tuple(shape) if shape is not None else ()
                spaces[name] = gym.spaces.Box(low, high, shape_t, dtype=np_dtype.type)
        return gym.spaces.Dict(spaces)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> _OneHotActionSpace:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        fps = int(self._config.get("fps", 60))
        episode_length = int(self._config.get("episode_length_seconds", 60))
        return max(1, (fps * episode_length) // self._action_repeat)

    def _seed_spaces(self, seed: int | None) -> None:
        if seed is None:
            return
        for space in (self._action_space, self._observation_space):
            if hasattr(space, "seed"):
                try:
                    space.seed(seed)
                except Exception:
                    pass

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._seed = int(seed)
            self._episode = 0
            self._seed_spaces(self._seed)
        self._last_obs = None
        episode_seed = self._seed + self._episode
        self._episode += 1
        self._env.reset(seed=episode_seed)
        self._last_obs = self._read_obs()
        return self._last_obs

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        native_action = self._to_native_action(action)
        reward = float(self._env.step(native_action, num_steps=self._action_repeat))
        done = not bool(self._env.is_running())
        if done:
            obs = self._last_obs if self._last_obs is not None else self._empty_obs()
        else:
            obs = self._read_obs()
            self._last_obs = obs
        info = finalize_step_info(
            {
                "discount": np.array(0.0 if done else 1.0, dtype=np.float32),
                "action": self._action_to_one_hot(native_action),
                "executed_action": native_action.copy(),
                "dmlab_action": native_action.copy(),
            },
            done=done,
            terminated=done,
            truncated=False,
        )
        return obs, reward, done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._last_obs is None:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        return self._last_obs["image"].transpose(1, 2, 0).copy()

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()

    def _to_native_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action)
        if arr.shape == (self._action_set.shape[1],) and np.issubdtype(
            arr.dtype, np.integer
        ):
            native = arr.astype(np.intc, copy=True)
            matches = np.all(self._action_set == native, axis=1)
            if not matches.any():
                raise ValueError(f"Invalid native DMLab action: {native!r}")
            return native
        index, _ = encode_discrete_action(arr, self._action_set.shape[0])
        return self._action_set[index].astype(np.intc, copy=True)

    def _action_to_one_hot(self, native_action: np.ndarray) -> np.ndarray:
        matches = np.all(self._action_set == native_action, axis=1)
        index = int(np.argmax(matches)) if matches.any() else 0
        action: np.ndarray = -np.ones((self._action_set.shape[0],), dtype=np.float32)
        action[index] = 1.0
        return action

    def _read_obs(self) -> dict[str, np.ndarray]:
        raw = self._env.observations()
        obs = {"image": self._to_chw_uint8(raw[RGB_OBSERVATION])}
        for key, value in raw.items():
            if key != RGB_OBSERVATION:
                obs[key] = np.asarray(value)
        return obs

    def _empty_obs(self) -> dict[str, np.ndarray]:
        return {"image": np.zeros((3, self._size[0], self._size[1]), dtype=np.uint8)}

    def _to_chw_uint8(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(
                f"Expected DMLab RGB image with 3 dims, got {image.shape}."
            )
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = image.transpose(1, 2, 0)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = image.astype(np.float32).clip(0, 255).astype(np.uint8)
        if image.shape[:2] != self._size:
            image = np.array(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.Resampling.BILINEAR
                )
            )
        return image.transpose(2, 0, 1).copy()


def _require_deepmind_lab() -> Any:
    try:
        return importlib.import_module("deepmind_lab")
    except ImportError as exc:
        raise ImportError(
            "DeepMind Lab support requires the `deepmind_lab` Python module. "
            "Install DeepMind Lab manually or build it with `pip install dmlab-gym` "
            "and `dmlab-gym build`."
        ) from exc
