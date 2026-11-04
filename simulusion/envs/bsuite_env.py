from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from simulusion.envs._contract import finalize_step_info
from simulusion.envs._observations import add_optional_state_space
import gymnasium as gym
import numpy as np
from PIL import Image

BSUITE_EXAMPLE_IDS = [
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


def list_available_bsuite_ids() -> list[str]:
    """Return the installed BSuite sweep ids, or examples if BSuite is absent."""
    if (
        importlib.util.find_spec("bsuite") is None
        or importlib.util.find_spec("bsuite.sweep") is None
    ):
        return list(BSUITE_EXAMPLE_IDS)

    sweep = importlib.import_module("bsuite.sweep")
    ids = getattr(sweep, "SWEEP", None)
    if ids is None:
        return list(BSUITE_EXAMPLE_IDS)
    return list(ids)


BSUITE_OBS_TYPES = ("pixels", "state", "both")


def normalize_bsuite_id(bsuite_id: str) -> str:
    """``"catch"`` -> ``"catch/0"``; full ids such as ``"catch/7"`` pass through.

    A bare environment name means its first setting, as in the DreamerV3
    reference implementation.
    """
    bsuite_id = str(bsuite_id).strip()
    return bsuite_id if "/" in bsuite_id else f"{bsuite_id}/0"


def list_bsuite_settings(environment: str) -> list[str]:
    """Every sweep id of one BSuite environment, e.g. ``"catch/0"`` .. ``"catch/19"``.

    The benchmark score of an environment aggregates all of its settings, so a
    full BSuite run trains once per id returned here (468 ids over the 23
    environments). Requires the ``bsuite`` package.
    """
    sweep = importlib.import_module("bsuite.sweep")
    name = str(environment).split("/", 1)[0]
    ids = [bsuite_id for bsuite_id in sweep.SWEEP if bsuite_id.split("/")[0] == name]
    if not ids:
        raise ValueError(f"Unknown BSuite environment {environment!r}.")
    return ids


def make_bsuite_env(bsuite_id: str, **kwargs: Any) -> "BSuiteImageEnv":
    """Create a Dreamer-compatible image wrapper around a BSuite task."""
    return BSuiteImageEnv(bsuite_id=bsuite_id, **kwargs)


class _BSuiteDiscreteActionSpace(gym.spaces.Box):
    """Box action space that samples normalized one-hot actions for BSuite."""

    # Marks the Box as a one-hot discrete choice, so agents that pick their
    # policy from the action space use a categorical one, not a Gaussian.
    one_hot = True

    def __init__(self, n: int, seed: int | None = None):
        self.n = int(n)
        self._rng = np.random.default_rng(seed)
        super().__init__(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def seed(self, seed: int | None = None) -> list[int]:
        self._rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> np.ndarray:
        del mask, probability
        action: np.ndarray = np.full((self.n,), -1.0, dtype=np.float32)
        action[self._rng.integers(self.n)] = 1.0
        return action


class BSuiteImageEnv:
    """Gym-like wrapper for DeepMind BSuite ``dm_env`` environments.

    BSuite tasks expose compact ``dm_env`` observations and mostly discrete
    actions. This adapter presents a Gym/Gymnasium-style API with image
    observations under a dict with required ``"image"`` key so Simulusion's
    pixel-based world models can train and evaluate on BSuite diagnostic tasks
    without requiring the base environment to implement rendering. When
    ``include_state=True``, observations also include a flattened ``"state"`` key.

    Args:
        bsuite_id: A sweep id such as ``"catch/0"``, or a bare environment name
            (``"catch"``), which means its first setting.
        obs_type: ``"pixels"`` (the default) renders the observation as an
            image; ``"state"`` gives only the flattened observation vector under
            ``"state"``, which is what DreamerV3 learns from on BSuite; ``"both"``
            gives both keys.
        results_dir: When set, BSuite records the run to CSV files there
            (``bsuite.load_and_record_to_csv``), which ``bsuite.logging`` and
            ``bsuite.experiments.summary_analysis`` turn into benchmark scores.
        overwrite: Replace existing result CSVs in ``results_dir``.

    BSuite fixes a budget in episodes, not steps: ``num_episodes`` reports it
    and ``budget_exhausted`` turns true once that many episodes have finished.
    """

    def __init__(
        self,
        bsuite_id: str,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        env: Any | None = None,
        include_state: bool = False,
        obs_type: str | None = None,
        results_dir: str | None = None,
        overwrite: bool = False,
    ):
        self.bsuite_id = normalize_bsuite_id(bsuite_id)
        self._seed = int(seed)
        self._size = (int(size[0]), int(size[1]))
        if obs_type is None:
            obs_type = "both" if include_state else "pixels"
        obs_type = str(obs_type).lower()
        if obs_type not in BSUITE_OBS_TYPES:
            raise ValueError(
                f"obs_type must be one of {BSUITE_OBS_TYPES}, got {obs_type!r}."
            )
        self._obs_type = obs_type
        self._include_image = obs_type in ("pixels", "both")
        self._include_state = obs_type in ("state", "both")
        self._env = (
            env
            if env is not None
            else self._load_bsuite_env(self.bsuite_id, results_dir, overwrite)
        )
        self._discrete_n: int | None = None
        self._last_time_step: Any | None = None
        self.episodes_completed = 0
        bootstrap_time_step = self._env.reset()
        self._action_space = self._make_action_space(self._env.action_spec())
        state_space = None
        if self._include_state:
            flattened = self._flatten_observation(bootstrap_time_step.observation)
            state_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=flattened.shape,
                dtype=np.float32,
            )
        if self._include_image:
            self._observation_space = add_optional_state_space(
                gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self._size[0], self._size[1]),
                    dtype=np.uint8,
                ),
                state_space=state_space,
            )
        else:
            assert state_space is not None
            self._observation_space = gym.spaces.Dict({"state": state_space})
        self._pending_reset_time_step = bootstrap_time_step

    @staticmethod
    def _load_bsuite_env(
        bsuite_id: str, results_dir: str | None = None, overwrite: bool = False
    ) -> Any:
        if importlib.util.find_spec("bsuite") is None:
            raise ImportError(
                "BSuite support requires the optional 'bsuite' package. "
                "Install it with `pip install bsuite` or `pip install simulusion[bsuite]`."
            )
        bsuite = importlib.import_module("bsuite")
        if results_dir:
            return bsuite.load_and_record_to_csv(
                bsuite_id, results_dir=str(results_dir), overwrite=bool(overwrite)
            )
        return bsuite.load_from_id(bsuite_id)

    @property
    def num_episodes(self) -> int | None:
        """The number of episodes BSuite's protocol runs this setting for."""
        value = getattr(self._env, "bsuite_num_episodes", None)
        if value is None:
            # Recording wrappers keep the raw environment as `_env`.
            value = getattr(
                getattr(self._env, "_env", None), "bsuite_num_episodes", None
            )
        return None if value is None else int(value)

    @property
    def budget_exhausted(self) -> bool:
        """True once ``num_episodes`` episodes have finished."""
        budget = self.num_episodes
        return budget is not None and self.episodes_completed >= budget

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def discrete_actions(self) -> bool:
        """Whether the Box action space stands for a discrete one (one-hot)."""
        return self._discrete_n is not None

    @property
    def max_episode_steps(self) -> int:
        # BSuite tasks end their own episodes (the longest, mountain_car, after
        # 1000 steps); this bounds the step count of one episode, not the run.
        return 1000

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        # BSuite seeds are encoded in the bsuite_id/settings. Accepting seed keeps
        # this wrapper compatible with Gym-style callers.
        if seed is not None:
            self._seed = int(seed)
            if isinstance(self._action_space, _BSuiteDiscreteActionSpace):
                self._action_space.seed(seed)
        if self._pending_reset_time_step is not None:
            time_step = self._pending_reset_time_step
            self._pending_reset_time_step = None
        else:
            time_step = self._env.reset()
        self._last_time_step = time_step
        return self._time_step_to_obs(time_step)

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        native_action = self._to_native_action(action)
        time_step = self._env.step(native_action)
        self._last_time_step = time_step
        obs = self._time_step_to_obs(time_step)
        reward = float(time_step.reward or 0.0)
        done = bool(time_step.last())
        if done:
            self.episodes_completed += 1
        info = finalize_step_info(
            {
                "discount": np.asarray(
                    0.0 if time_step.discount is None else time_step.discount,
                    dtype=np.float32,
                ),
                "bsuite_id": self.bsuite_id,
                "action": self._one_hot_action(native_action),
                "vector_observation": self._flatten_observation(
                    time_step.observation
                ).astype(np.float32, copy=False),
            },
            done=done,
            terminated=done,
            truncated=False,
        )
        return obs, reward, done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._last_time_step is None:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        return self._obs_to_hwc_image(self._last_time_step.observation)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def _make_action_space(self, action_spec: Any) -> gym.Space:
        num_values = getattr(action_spec, "num_values", None)
        if num_values is not None:
            n = int(num_values)
            self._discrete_n = n
            return _BSuiteDiscreteActionSpace(n, seed=self._seed)

        minimum = np.asarray(getattr(action_spec, "minimum", -1.0), dtype=np.float32)
        maximum = np.asarray(getattr(action_spec, "maximum", 1.0), dtype=np.float32)
        shape = tuple(getattr(action_spec, "shape", minimum.shape or maximum.shape))
        if shape == ():
            shape = (1,)
        self._discrete_n = None
        return gym.spaces.Box(low=minimum, high=maximum, shape=shape, dtype=np.float32)

    def _to_native_action(self, action: Any) -> int | np.ndarray:
        if self._discrete_n is None:
            return np.asarray(action, dtype=np.float32)
        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return 0
        return int(np.argmax(vec[: self._discrete_n]))

    def _one_hot_action(self, action: int | np.ndarray) -> np.ndarray:
        if self._discrete_n is None:
            return np.asarray(action, dtype=np.float32)
        out: np.ndarray = np.full((self._discrete_n,), -1.0, dtype=np.float32)
        out[int(action)] = 1.0
        return out

    def _time_step_to_obs(self, time_step: Any) -> dict[str, np.ndarray]:
        observation: dict[str, np.ndarray] = {}
        if self._include_image:
            observation["image"] = (
                self._obs_to_hwc_image(time_step.observation).transpose(2, 0, 1).copy()
            )
        if self._include_state:
            observation["state"] = self._flatten_observation(
                time_step.observation
            ).astype(np.float32, copy=False)
        return observation

    def _obs_to_hwc_image(self, obs: Any) -> np.ndarray:
        arr = self._flatten_observation(obs)
        if arr.size == 0:
            arr = np.zeros((1,), dtype=np.float32)
        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)
        if finite.any():
            lo = float(arr[finite].min())
            hi = float(arr[finite].max())
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            else:
                arr = np.zeros_like(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

        side = int(np.ceil(np.sqrt(arr.size)))
        canvas: np.ndarray = np.zeros((side * side,), dtype=np.float32)
        canvas[: arr.size] = arr
        image = (canvas.reshape(side, side) * 255.0).astype(np.uint8)
        image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[:2] != self._size:
            resampling = getattr(Image, "Resampling", Image)
            image = np.asarray(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), resampling.BILINEAR
                )
            )
        return image

    def _flatten_observation(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            parts = [self._flatten_observation(obs[key]) for key in sorted(obs)]
            return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
        return np.asarray(obs, dtype=np.float32).reshape(-1)
