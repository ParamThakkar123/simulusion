from __future__ import annotations

from typing import Any

from simulusion.envs._actions import clip_box_action
from simulusion.envs._contract import finalize_step_info
from simulusion.utils.gym_compat import gym
import numpy as np


#: Observation types the adapter can produce.
OBS_TYPES = ("pixels", "state")


class DeepMindControlEnv:
    """Gym-style adapter for DeepMind Control Suite tasks.

    The wrapper exposes DMC observations and actions through Gym spaces. Every
    observation dict carries the task's own state entries (``position``,
    ``velocity``, ...) and ``"state"``: those entries flattened into one float32
    vector in the order of the task's observation spec. That vector is the
    proprioceptive input of the "Proprio Control" benchmark (Hafner et al.,
    2023).

    ``obs_type`` selects what else is produced:

    * ``"pixels"`` (default) also renders an RGB ``"image"`` each step, for
      image-based world models (the "Visual Control" benchmark).
    * ``"state"`` does not render at all, which is much faster; the observation
      space then has no ``"image"`` key. :meth:`render` still works, for videos.

    Episode ends follow the task's discount. DMC tasks end at a time limit with
    discount 1, which is reported as a truncation; only a discount of 0 is
    reported as a termination.

    Features:
        - Parses domain-task names (e.g., "cheetah-run" -> domain="cheetah", task="run")
        - Automatically handles special cases like "cup" -> "ball_in_cup"
        - Renders RGB images at configurable resolution

    Args:
        name (str): Environment name in format "domain-task" (e.g., "cheetah-run").
        seed (int): Random seed for environment initialization.
        size (tuple): Target image size as (height, width) (default: (64, 64)).
        camera (int, optional): Camera ID for rendering. Defaults to 0 for most
            domains, 2 for quadruped.
        obs_type (str): ``"pixels"`` or ``"state"``.

    Attributes:
        observation_space (gym.spaces.Dict): Dict space with the state keys,
            ``"state"`` and, for pixels, ``"image"``.
        action_space (gym.spaces.Box): Continuous action space from DMC spec.

    Example:
        >>> env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
        >>> sorted(env.reset())
        ['image', 'position', 'state', 'velocity']
        >>> env = DeepMindControlEnv("cheetah-run", seed=0, obs_type="state")
        >>> env.reset()["state"].shape
        (17,)
    """

    def __init__(
        self,
        name: str,
        seed: int,
        size: tuple[int, int] = (64, 64),
        camera: int | None = None,
        obs_type: str = "pixels",
    ) -> None:
        obs_type = str(obs_type).lower()
        if obs_type not in OBS_TYPES:
            raise ValueError(
                f"obs_type must be one of {', '.join(OBS_TYPES)}; got {obs_type!r}."
            )
        self._obs_type = obs_type
        if "-" not in name:
            raise ValueError(
                f"DeepMind Control task names use 'domain-task' form, e.g. "
                f"'cartpole-balance' or 'walker-walk'; got {name!r}."
            )
        domain, task = name.split("-", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        self._name = name
        self._domain = domain
        self._task = task
        self._seed = int(seed)
        self._size = (int(size[0]), int(size[1]))
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self._env = self._make_env(self._seed)

        spaces: dict[str, gym.spaces.Space[Any]] = {}
        self._state_keys: list[str] = []
        state_size = 0
        for key, value in self._env.observation_spec().items():
            if key in ("state", "image"):
                raise ValueError(
                    f"DMC task {name!r} has an observation named {key!r}, which "
                    "collides with a key the adapter adds."
                )
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
            self._state_keys.append(key)
            state_size += int(np.prod(value.shape, dtype=np.int64))
        spaces["state"] = gym.spaces.Box(
            -np.inf, np.inf, (state_size,), dtype=np.float32
        )
        if self._obs_type == "pixels":
            spaces["image"] = gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8)
        self._observation_space = gym.spaces.Dict(spaces)
        spec = self._env.action_spec()
        self._action_space = gym.spaces.Box(
            spec.minimum, spec.maximum, dtype=np.float32
        )
        self._seed_spaces(self._seed)

    def _make_env(self, seed: int) -> Any:
        try:
            from dm_control import suite
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The DeepMind Control backend requires the 'dm_control' package, "
                "which is not installed. Install it with:\n\n"
                "    pip install simulusion[dmc]          # Python <= 3.12\n"
                "    python -m simulusion.install_dmc     # any version, incl. 3.13\n\n"
                "On CPython 3.13 the extra alone is not enough: dm-control pins "
                "`labmaze`, which has no 3.13 wheel and builds with Bazel. The "
                "installer above uses the pure-Python `labmaze-new` instead.\n\n"
                "or pick a backend that is already available (for example a "
                "Gymnasium task via env_backend='gym', such as 'Pendulum-v1')."
            ) from exc

        return suite.load(self._domain, self._task, task_kwargs={"random": int(seed)})

    def _seed_spaces(self, seed: int | None) -> None:
        if seed is None:
            return
        for space in (self._action_space, self._observation_space):
            if hasattr(space, "seed"):
                try:
                    space.seed(seed)
                except Exception:
                    pass

    @property
    def obs_type(self) -> str:
        return self._obs_type

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    def _observation(self, time_step: Any) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        for key in self._state_keys:
            obs[key] = np.asarray(time_step.observation[key], dtype=np.float32)
        parts = [obs[key].reshape(-1) for key in self._state_keys]
        obs["state"] = (
            np.concatenate(parts, axis=0) if parts else np.zeros(0, np.float32)
        )
        if self._obs_type == "pixels":
            obs["image"] = self.render().transpose(2, 0, 1).copy()
        return obs

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        clipped = clip_box_action(action, self.action_space.low, self.action_space.high)
        time_step = self._env.step(clipped)
        obs = self._observation(time_step)
        reward = time_step.reward or 0
        done = bool(time_step.last())
        discount = 1.0 if time_step.discount is None else float(time_step.discount)
        # dm_control ends every episode at the task's time limit with discount
        # 1: that state is not terminal and its value should be bootstrapped.
        # Only a discount of 0 means the episode genuinely terminated.
        terminated = done and discount == 0.0
        info = finalize_step_info(
            {
                "discount": np.array(discount, np.float32),
                "action": clipped.copy(),
                "executed_action": clipped.copy(),
                "vector_observation": obs["state"].copy(),
            },
            done=done,
            terminated=terminated,
            truncated=done and not terminated,
        )
        return obs, float(reward), done, info

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._seed = int(seed)
            self._env = self._make_env(self._seed)
            self._seed_spaces(self._seed)
        return self._observation(self._env.reset())

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


def make_dmc_env(
    env_id: str,
    seed: int = 0,
    size: tuple[int, int] = (64, 64),
    camera: int | None = None,
    obs_type: str = "pixels",
) -> DeepMindControlEnv:
    """Create a DeepMind Control Suite task, e.g. ``make_dmc_env("walker-walk")``.

    Pass ``obs_type="state"`` for proprioceptive observations without rendering.
    """
    return DeepMindControlEnv(
        env_id, seed=seed, size=size, camera=camera, obs_type=obs_type
    )
