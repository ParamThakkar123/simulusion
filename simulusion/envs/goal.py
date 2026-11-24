"""Goal-conditioned environments: reach a goal given as an image.

Reward-free world models (LeWM, DINO-WM, PLDM) are evaluated by planning: the
agent sees its current frame and a goal frame and must reach the goal within
a step budget. That needs more from an environment than stepping. It must
start from a given state, render any state (the goal image is the frame of
the goal state), and say when the goal is reached. :class:`GoalEnv` is that
contract, shared by :class:`~simulusion.envs.two_room.TwoRoomEnv`,
:class:`~simulusion.envs.pusht.PushTEnv`,
:class:`~simulusion.envs.ogbench_cube.CubeEnv` and
:class:`~simulusion.envs.reacher_goal.ReacherGoalEnv`, and used by the
goal-reaching benchmark (:mod:`simulusion.benchmarks.goal_reaching`).

The contract on top of Simulusion's image environments:

* ``reset(seed=None, options=None)`` returns ``{"image": (3, H, W) uint8}``.
  ``options["state"]`` starts the episode in that state and
  ``options["goal_state"]`` sets the goal; without them the environment draws
  its own start and goal.
* :meth:`GoalEnv.get_state` / :meth:`GoalEnv.set_state`: the full physical
  state as a flat float vector, the same vector the data collectors store as
  ``observations["state"]``.
* :meth:`GoalEnv.goal_observation`: the goal frame, ``{"image": ...}``.
* :meth:`GoalEnv.goal_distance` / :meth:`GoalEnv.is_success`: compare two
  states; every step's info has ``success``, ``goal_distance`` and ``state``.

Rewards are ``1.0`` on a successful step and ``0.0`` otherwise, so the
environments also work with reward-based agents; the reward-free models
ignore them. Episodes end at ``max_episode_steps`` (a truncation), and on
success only with ``terminate_on_success=True``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import numpy as np

from simulusion.envs._contract import finalize_step_info
from simulusion.envs._images import to_chw_uint8


class GoalEnv(ABC):
    """Base class of the goal-conditioned environments.

    Subclasses implement the physics (:meth:`_reset_random`,
    :meth:`_apply_action`, :meth:`_get_state`, :meth:`_set_state`), the
    frame (:meth:`_render`), the default goal (:meth:`_sample_goal`) and the
    comparison of two states (:meth:`goal_distance`, :meth:`is_success`).

    Args:
        size: Square frame size in pixels.
        max_episode_steps: Steps before the episode is truncated; ``None``
            for no limit.
        terminate_on_success: End the episode on the first successful step.
        seed: Seed of the environment's random start and goal.
    """

    #: Length of :meth:`get_state` vectors.
    state_dim: int = 0

    def __init__(
        self,
        *,
        size: int = 64,
        max_episode_steps: Optional[int] = 200,
        terminate_on_success: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.size = int(size)
        self.max_episode_steps = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self.terminate_on_success = bool(terminate_on_success)
        self.np_random = np.random.default_rng(seed)
        self._goal_state: Optional[np.ndarray] = None
        self._goal_image: Optional[np.ndarray] = None
        self._frame: Optional[np.ndarray] = None
        self._steps = 0

    # ------------------------------------------------------------ subclass

    @abstractmethod
    def _reset_random(self) -> None:
        """Put the environment in a random start state (``self.np_random``)."""

    @abstractmethod
    def _sample_goal(self) -> np.ndarray:
        """A goal state for an episode that was not given one."""

    @abstractmethod
    def _apply_action(self, action: np.ndarray) -> None:
        """Advance the physics by one step."""

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        """The current state, ``(state_dim,)`` float64."""

    @abstractmethod
    def _set_state(self, state: np.ndarray) -> None:
        """Put the environment in ``state``."""

    @abstractmethod
    def _render(self) -> np.ndarray:
        """The current frame, ``(size, size, 3)`` uint8."""

    @abstractmethod
    def goal_distance(self, state: Any, goal: Any) -> float:
        """Distance from ``state`` to ``goal`` in the environment's own units."""

    @abstractmethod
    def is_success(self, state: Any, goal: Any) -> bool:
        """Whether ``state`` reaches ``goal``."""

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """The continuous action space."""

    # ------------------------------------------------------------- contract

    @property
    def observation_space(self) -> Any:
        from simulusion.envs._spaces import image_space
        from simulusion.utils.gym_compat import gym

        return gym.spaces.Dict({"image": image_space((self.size, self.size))})

    def get_state(self) -> np.ndarray:
        """The full physical state as a flat float64 vector."""
        return np.asarray(self._get_state(), np.float64).copy()

    def set_state(self, state: Any) -> None:
        """Put the environment in ``state`` (from :meth:`get_state` or a dataset)."""
        self._set_state(self._check_state(state))
        self._frame = None

    def render_state(self, state: Any) -> np.ndarray:
        """The frame ``(size, size, 3)`` of ``state``, leaving the environment as it was."""
        saved = self.get_state()
        try:
            self._set_state(self._check_state(state))
            return self._render()
        finally:
            self._set_state(saved)

    def set_goal(self, goal_state: Any) -> None:
        """Make ``goal_state`` the goal and render its frame."""
        goal = self._check_state(goal_state)
        self._goal_state = goal
        self._goal_image = self.render_state(goal)

    @property
    def goal_state(self) -> np.ndarray:
        if self._goal_state is None:
            raise RuntimeError("No goal yet. Call reset() first.")
        return self._goal_state.copy()

    def goal_observation(self) -> dict[str, np.ndarray]:
        """The goal frame, ``{"image": (3, H, W) uint8}``."""
        if self._goal_image is None:
            raise RuntimeError("No goal yet. Call reset() first.")
        return {"image": to_chw_uint8(self._goal_image, (self.size, self.size))}

    def seed(self, seed: Optional[int] = None) -> None:
        self.np_random = np.random.default_rng(seed)
        seed_space = getattr(self.action_space, "seed", None)
        if callable(seed_space) and seed is not None:
            seed_space(int(seed))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> dict[str, np.ndarray]:
        if seed is not None:
            self.seed(seed)
        options = dict(options or {})
        if options.get("state") is not None:
            self._set_state(self._check_state(options["state"]))
        else:
            self._reset_random()
        goal = options.get("goal_state")
        self.set_goal(self._sample_goal() if goal is None else goal)
        self._steps = 0
        return self._observe()

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self._goal_state is None:
            raise RuntimeError("Call reset() before step().")
        act = self._clip_action(action)
        self._apply_action(act)
        self._steps += 1
        state = self.get_state()
        success = bool(self.is_success(state, self._goal_state))
        terminated = success and self.terminate_on_success
        truncated = (
            not terminated
            and self.max_episode_steps is not None
            and self._steps >= self.max_episode_steps
        )
        done = terminated or truncated
        info = finalize_step_info(
            {
                "success": success,
                "goal_distance": float(self.goal_distance(state, self._goal_state)),
                "state": state,
                "action": act,
            },
            done=done,
            terminated=terminated,
            truncated=truncated,
            discount=np.array(0.0 if terminated else 1.0, np.float32),
        )
        return self._observe(), float(success), done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """The current frame, ``(size, size, 3)`` uint8."""
        if self._frame is None:
            self._frame = self._render()
        return self._frame.copy()

    def close(self) -> None:
        pass

    # -------------------------------------------------------------- helpers

    def _observe(self) -> dict[str, np.ndarray]:
        self._frame = self._render()
        return {"image": to_chw_uint8(self._frame, (self.size, self.size))}

    def _clip_action(self, action: Any) -> np.ndarray:
        space = self.action_space
        act = np.asarray(action, np.float64).reshape(space.shape)
        if not np.all(np.isfinite(act)):
            raise ValueError(f"Actions must be finite, got {act}.")
        return np.clip(act, space.low, space.high)

    def _check_state(self, state: Any) -> np.ndarray:
        arr = np.asarray(state, np.float64).reshape(-1)
        if arr.shape != (self.state_dim,):
            raise ValueError(
                f"{type(self).__name__} states have {self.state_dim} values, "
                f"got {arr.shape[0]}."
            )
        return arr


#: Goal environments by name: ``(module, class)``.
GOAL_ENVS = {
    "two_room": ("simulusion.envs.two_room", "TwoRoomEnv"),
    "pusht": ("simulusion.envs.pusht", "PushTEnv"),
    "cube": ("simulusion.envs.ogbench_cube", "OGBenchCubeEnv"),
    "reacher_goal": ("simulusion.envs.reacher_goal", "ReacherGoalEnv"),
}

_ALIASES = {
    "tworoom": "two_room",
    "two-room": "two_room",
    "push_t": "pusht",
    "push-t": "pusht",
    "ogbench_cube": "cube",
    "ogbench-cube": "cube",
    "cube_single": "cube",
    "reacher": "reacher_goal",
    "reacher-goal": "reacher_goal",
}


def goal_env_name(name: str) -> str:
    """The canonical name of a goal environment (``"two-room"`` -> ``"two_room"``)."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in GOAL_ENVS:
        raise ValueError(
            f"Unknown goal environment {name!r}. Choose from {sorted(GOAL_ENVS)}."
        )
    return key


def make_goal_env(name: str, **kwargs: Any) -> GoalEnv:
    """Create a goal environment by name, e.g. ``make_goal_env("pusht", size=224)``."""
    import importlib

    module, cls = GOAL_ENVS[goal_env_name(name)]
    return getattr(importlib.import_module(module), cls)(**kwargs)


__all__ = ["GOAL_ENVS", "GoalEnv", "goal_env_name", "make_goal_env"]
