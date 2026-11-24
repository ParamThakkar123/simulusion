"""Reacher goal: bring a two-joint arm into a goal pose.

DeepMind Control's Reacher (Tassa et al., 2018) made harder as in DINO-WM
(Zhou et al., 2025, App. A.1) and LeWorldModel (Maes et al., 2026): the whole
arm, not just its tip, must match a goal pose. A goal is a pair of joint
angles; its frame shows the arm in that pose.

* Physics and actions are DMC's: two joint torques in ``[-1, 1]``, 0.02 s per
  step.
* The suite's red target sphere is hidden, so the goal frame is the only sign
  of the goal.
* Success: both joint angles within ``angle_tol`` radians of the goal's. The
  papers do not give a tolerance; the default is 0.15 rad (about 9 degrees).
* Without a ``goal_state`` option the goal is a random pose.

State: shoulder and wrist angles, then their velocities.

Needs DeepMind Control (``pip install simulusion[dmc]``).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from simulusion.envs.brax_env import _require_module
from simulusion.envs.goal import GoalEnv


def _wrap(angle: Any) -> Any:
    return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi


class ReacherGoalEnv(GoalEnv):
    """Move DMC Reacher's arm into the pose of a goal state.

    Args:
        task: ``"easy"`` or ``"hard"``; they differ only in the hidden target's
            size, so the choice does not change this task.
        size: Frame size in pixels (DINO-WM and LeWM use 224).
        camera: DMC camera id.
        angle_tol: Per-joint tolerance for success, in radians.
        max_episode_steps: Steps per episode.
        terminate_on_success: End the episode once the goal is reached.
        seed: Seed of the random start and goal.
    """

    state_dim = 4

    def __init__(
        self,
        *,
        task: str = "easy",
        size: int = 64,
        camera: int = 0,
        angle_tol: float = 0.15,
        max_episode_steps: Optional[int] = 200,
        terminate_on_success: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            size=size,
            max_episode_steps=max_episode_steps,
            terminate_on_success=terminate_on_success,
            seed=seed,
        )
        _require_module("dm_control", "Install it with `pip install simulusion[dmc]`.")
        from dm_control import suite

        self.angle_tol = float(angle_tol)
        self.camera = int(camera)
        self._env = suite.load(
            "reacher",
            task,
            task_kwargs={
                "random": 0 if seed is None else int(seed),
                "time_limit": float("inf"),
            },
        )
        self._env.reset()
        physics = self._env.physics
        # Hide the target sphere: the goal is shown by the goal frame instead.
        physics.named.model.geom_rgba["target", 3] = 0.0
        spec = self._env.action_spec()
        from simulusion.utils.gym_compat import gym

        self._action_space = gym.spaces.Box(
            spec.minimum.astype(np.float32),
            spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def physics(self) -> Any:
        return self._env.physics

    # ------------------------------------------------------------ GoalEnv API

    def _random_pose(self) -> np.ndarray:
        return self.np_random.uniform(-math.pi, math.pi, 2)

    def _reset_random(self) -> None:
        self._set_state(np.concatenate([self._random_pose(), np.zeros(2)]))

    def _sample_goal(self) -> np.ndarray:
        return np.concatenate([self._random_pose(), np.zeros(2)])

    def _apply_action(self, action: np.ndarray) -> None:
        self._env.step(action)

    def _get_state(self) -> np.ndarray:
        p = self._env.physics
        return np.concatenate([p.data.qpos[:2], p.data.qvel[:2]]).astype(np.float64)

    def _set_state(self, state: np.ndarray) -> None:
        p = self._env.physics
        with p.reset_context():
            p.data.qpos[:2] = state[:2]
            p.data.qvel[:2] = state[2:4]

    def _render(self) -> np.ndarray:
        return np.asarray(
            self._env.physics.render(self.size, self.size, camera_id=self.camera),
            np.uint8,
        )

    def goal_distance(self, state: Any, goal: Any) -> float:
        """The larger of the two joint-angle errors, in radians."""
        s, g = np.asarray(state, np.float64), np.asarray(goal, np.float64)
        return float(np.abs(_wrap(s[:2] - g[:2])).max())

    def is_success(self, state: Any, goal: Any) -> bool:
        return self.goal_distance(state, goal) <= self.angle_tol


def make_reacher_goal_env(env_id: str = "easy", **kwargs: Any) -> ReacherGoalEnv:
    """Create Reacher goal, e.g. ``make_reacher_goal_env("easy", size=224)``."""
    return ReacherGoalEnv(task=env_id, **kwargs)


__all__ = ["ReacherGoalEnv", "make_reacher_goal_env"]
