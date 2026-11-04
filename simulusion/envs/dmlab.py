from __future__ import annotations

import functools
import importlib
import re
import zlib
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

# The small action set of IMPALA (Espeholt et al. 2018).
IMPALA_ACTION_SET = np.array(
    [
        [0, 0, 0, 1, 0, 0, 0],  # forward
        [0, 0, 0, -1, 0, 0, 0],  # backward
        [0, 0, -1, 0, 0, 0, 0],  # strafe left
        [0, 0, 1, 0, 0, 0, 0],  # strafe right
        [-20, 0, 0, 0, 0, 0, 0],  # look left
        [20, 0, 0, 0, 0, 0, 0],  # look right
        [-20, 0, 0, 1, 0, 0, 0],  # look left + forward
        [20, 0, 0, 1, 0, 0, 0],  # look right + forward
        [0, 0, 0, 0, 1, 0, 0],  # fire
    ],
    dtype=np.intc,
)

# The larger fixed action set of PopArt (Hessel et al. 2019) and R2D2
# (Kapturowski et al. 2019), which DreamerV3 uses on DMLab-30.
POPART_ACTION_SET = np.array(
    [
        [0, 0, 0, 1, 0, 0, 0],  # forward
        [0, 0, 0, -1, 0, 0, 0],  # backward
        [0, 0, -1, 0, 0, 0, 0],  # strafe left
        [0, 0, 1, 0, 0, 0, 0],  # strafe right
        [-10, 0, 0, 0, 0, 0, 0],  # small look left
        [10, 0, 0, 0, 0, 0, 0],  # small look right
        [-60, 0, 0, 0, 0, 0, 0],  # large look left
        [60, 0, 0, 0, 0, 0, 0],  # large look right
        [0, 10, 0, 0, 0, 0, 0],  # look down
        [0, -10, 0, 0, 0, 0, 0],  # look up
        [-10, 0, 0, 1, 0, 0, 0],  # forward + small look left
        [10, 0, 0, 1, 0, 0, 0],  # forward + small look right
        [-60, 0, 0, 1, 0, 0, 0],  # forward + large look left
        [60, 0, 0, 1, 0, 0, 0],  # forward + large look right
        [0, 0, 0, 0, 1, 0, 0],  # fire
    ],
    dtype=np.intc,
)

ACTION_SETS = {
    "default": DEFAULT_ACTION_SET,
    "impala": IMPALA_ACTION_SET,
    "popart": POPART_ACTION_SET,
}

INSTR_OBSERVATION = "INSTR"
# Language levels give a text instruction. It is embedded as in the DreamerV3
# reference: lowercase word/punctuation tokens, each hashed (CRC32) into one of
# 64K buckets with a fixed random 32-d embedding, padded to 32 tokens and
# concatenated into a 1024-d vector under the "instr" key.
INSTR_LENGTH = 32
INSTR_EMBED_SIZE = 32
INSTR_VOCAB_BUCKETS = 64 * 1024
_INSTR_TOKENIZER = re.compile(r"([A-Za-z_]+|[^A-Za-z_ ]+)")


@functools.lru_cache(maxsize=1)
def _instr_embeddings() -> np.ndarray:
    return (
        np.random.default_rng(seed=0)
        .normal(0.0, 1.0, (INSTR_VOCAB_BUCKETS, INSTR_EMBED_SIZE))
        .astype(np.float32)
    )


def embed_instruction(text: str | bytes) -> np.ndarray:
    """Embed a DMLab instruction string as a ``(1024,)`` float32 vector."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    tokens = _INSTR_TOKENIZER.findall(str(text).lower())[:INSTR_LENGTH]
    indices = [
        zlib.crc32(token.encode("utf-8")) % INSTR_VOCAB_BUCKETS for token in tokens
    ]
    indices += [0] * (INSTR_LENGTH - len(indices))
    return _instr_embeddings()[indices].reshape(-1).copy()


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


# The 30 DMLab-30 tasks (two have separate train and test levels), found under
# contributed/dmlab30/. DMLabEnv accepts these bare names, as the DreamerV3
# reference does.
DMLAB30_LEVELS = (
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
    "language_answer_quantitative_question",
    "lasertag_one_opponent_small",
    "lasertag_three_opponents_small",
    "lasertag_one_opponent_large",
    "lasertag_three_opponents_large",
    "natlab_fixed_large_map",
    "natlab_varying_map_regrowth",
    "natlab_varying_map_randomized",
    "skymaze_irreversible_path_hard",
    "skymaze_irreversible_path_varied",
    "psychlab_arbitrary_visuomotor_mapping",
    "psychlab_continuous_recognition",
    "psychlab_sequential_comparison",
    "psychlab_visual_search",
    "explore_object_locations_small",
    "explore_object_locations_large",
    "explore_obstructed_goals_small",
    "explore_obstructed_goals_large",
    "explore_goal_locations_small",
    "explore_goal_locations_large",
    "explore_object_rewards_few",
    "explore_object_rewards_many",
)
DMLAB30_PREFIX = "contributed/dmlab30/"


def resolve_dmlab_level(level: str) -> str:
    """The Lab level path for ``level``: DMLab-30 names gain their folder.

    ``"explore_goal_locations_small"`` becomes
    ``"contributed/dmlab30/explore_goal_locations_small"``; other names and
    full paths are returned unchanged.
    """
    level = str(level)
    return DMLAB30_PREFIX + level if level in DMLAB30_LEVELS else level


def make_dmlab_env(level: str, **kwargs: Any) -> "DMLabEnv":
    """Create a DeepMind Lab environment adapter for Simulusion.

    Args:
        level: DeepMind Lab level name, for example
            ``"rooms_collect_good_objects_train"``. DMLab-30 names are looked
            up under ``contributed/dmlab30/`` (:func:`resolve_dmlab_level`).
        **kwargs: Additional keyword arguments passed to :class:`DMLabEnv`.

    Returns:
        DMLabEnv: A Gym-like wrapper returning ``{"image": (C, H, W)}`` uint8
        observations and normalized one-hot discrete actions.
    """
    return DMLabEnv(level=level, **kwargs)


class _OneHotActionSpace(gym.spaces.Box):
    """Box action space that samples normalized one-hot vectors."""

    # Marks the Box as a one-hot discrete choice (categorical, not Gaussian).
    one_hot = True

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

    Args:
        action_set: An array of native action vectors, or the name of a fixed
            set: ``"default"`` (9 actions with a no-op), ``"impala"`` or
            ``"popart"`` (the 15-action set DreamerV3 uses on DMLab-30).
        text: Add the embedded language instruction under ``"instr"`` (see
            :func:`embed_instruction`). ``None`` turns it on for ``language_*``
            levels only.
        mode: ``"train"`` maps ``*_test`` levels to their ``*_train`` version;
            ``"eval"`` enables the held-out level set (``allowHoldOutLevels``)
            with a fixed mixer seed, as for DMLab-30 test scores.
    """

    def __init__(
        self,
        level: str,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        action_repeat: int = 4,
        action_set: Sequence[Sequence[int]] | np.ndarray | str | None = None,
        observations: Sequence[str] | None = None,
        config: dict[str, Any] | None = None,
        renderer: str = "hardware",
        text: bool | None = None,
        mode: str | None = None,
        **lab_kwargs: Any,
    ):
        self._level = resolve_dmlab_level(level)
        if mode is not None:
            mode = str(mode).lower()
            if mode not in ("train", "eval"):
                raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}.")
            if mode == "train" and self._level.endswith("_test"):
                self._level = self._level[: -len("_test")] + "_train"
        self._seed = int(seed)
        self._episode = 0
        self._size = (int(size[0]), int(size[1]))
        self._action_repeat = int(action_repeat)
        if self._action_repeat < 1:
            raise ValueError("action_repeat must be >= 1.")

        if isinstance(action_set, str):
            try:
                action_set = ACTION_SETS[action_set.lower()]
            except KeyError:
                raise ValueError(
                    f"Unknown DMLab action set {action_set!r}; options: "
                    f"{', '.join(ACTION_SETS)}."
                ) from None
        self._action_set = np.asarray(
            DEFAULT_ACTION_SET if action_set is None else action_set, dtype=np.intc
        )
        if self._action_set.ndim != 2:
            raise ValueError("action_set must be a 2D array of DMLab action vectors.")

        if text is None:
            text = self._level.rsplit("/", 1)[-1].startswith("language")
        self._text = bool(text)
        self._observations = list(observations or [RGB_OBSERVATION])
        if RGB_OBSERVATION not in self._observations:
            self._observations.insert(0, RGB_OBSERVATION)
        if self._text and INSTR_OBSERVATION not in self._observations:
            self._observations.append(INSTR_OBSERVATION)

        lab_config = {
            "width": str(self._size[1]),
            "height": str(self._size[0]),
            "logLevel": "WARN",
        }
        if mode == "eval":
            lab_config.update(
                {"allowHoldOutLevels": "true", "mixerSeed": str(0x600D5EED)}
            )
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
                if not name or name in (RGB_OBSERVATION, INSTR_OBSERVATION):
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
                try:
                    np_dtype = np.dtype(dtype)
                except TypeError:
                    continue
                if not np.issubdtype(np_dtype, np.number):
                    continue  # text observations have no Box space
                if np.issubdtype(np_dtype, np.integer):
                    info = np.iinfo(np_dtype)  # type: ignore[type-var]
                    low: float = int(info.min)
                    high: float = int(info.max)
                else:
                    low, high = -np.inf, np.inf
                shape_t = tuple(shape) if shape is not None else ()
                spaces[name] = gym.spaces.Box(low, high, shape_t, dtype=np_dtype.type)
        if self._text:
            spaces["instr"] = gym.spaces.Box(
                -np.inf, np.inf, (INSTR_LENGTH * INSTR_EMBED_SIZE,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> _OneHotActionSpace:
        return self._action_space

    @property
    def native_action_repeat(self) -> int:
        """Lab frames per step; training counts frames, as the reference."""
        return self._action_repeat

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
            if key == INSTR_OBSERVATION:
                if self._text:
                    obs["instr"] = embed_instruction(value)
                continue
            if key != RGB_OBSERVATION:
                array = np.asarray(value)
                if np.issubdtype(array.dtype, np.number):
                    obs[key] = array
        return obs

    def _empty_obs(self) -> dict[str, np.ndarray]:
        obs = {"image": np.zeros((3, self._size[0], self._size[1]), dtype=np.uint8)}
        if self._text:
            obs["instr"] = embed_instruction("")
        return obs

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
            "DeepMind Lab has no wheels; it builds from source on Linux with "
            "Python <= 3.11. Use the docker/envs image, or run "
            "docker/envs/install_dmlab.sh as root in a Linux environment."
        ) from exc
