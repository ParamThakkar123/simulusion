"""OGBench Cube: a robot arm moves cubes to goal positions.

The manipulation task of OGBench (Park et al., 2025), used from pixels by
LeWorldModel (Maes et al., 2026) in its single-cube form: a 6-DoF arm with a
gripper, above a table with one to four cubes, seen by a fixed camera. The
simulation is OGBench's own (MuJoCo); this adapter gives it the goal contract
of :class:`~simulusion.envs.goal.GoalEnv`.

* Actions are OGBench's 5 values in ``[-1, 1]``: the gripper's displacement
  (x, y, z), its yaw change and its opening change, turned into joint targets
  by inverse kinematics.
* A goal is a full simulator state; its frame is rendered from that state.
  OGBench's translucent target cubes are hidden, so the goal frame is the
  only sign of the goal.
* Success is OGBench's: every cube within ``success_radius`` (4 cm) of its
  goal position.
* Without a ``goal_state`` option, the goal moves each cube to a random spot
  on the table (OGBench's target region), arm unchanged.

State: MuJoCo ``qpos`` then ``qvel`` (41 values for one cube). The arm is
drawn translucent, as in OGBench's pixel tasks, so it does not hide the cubes.

Needs ``pip install simulusion[ogbench]``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from simulusion.envs.brax_env import _require_module
from simulusion.envs.goal import GoalEnv

_CUBE_TYPES = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}


def _require_ogbench_cube() -> Any:
    _require_module(
        "ogbench",
        "Install it with `pip install simulusion[ogbench]` or `pip install ogbench`.",
    )
    from ogbench.manipspace.envs import cube_env

    return cube_env


class OGBenchCubeEnv(GoalEnv):
    """Move OGBench's cubes to the positions of a goal state.

    Args:
        cubes: Number of cubes, 1 to 4 (LeWM uses 1).
        size: Frame size in pixels (OGBench's pixel tasks use 64, LeWM 224).
        success_radius: Distance in metres within which a cube is at its goal.
        max_episode_steps: Steps per episode (200, OGBench's episode length).
        terminate_on_success: End the episode once the goal is reached.
        seed: Seed of the random start and goal.
    """

    def __init__(
        self,
        *,
        cubes: int = 1,
        size: int = 64,
        success_radius: float = 0.04,
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
        if cubes not in _CUBE_TYPES:
            raise ValueError("cubes must be 1, 2, 3 or 4.")
        cube_env = _require_ogbench_cube()
        self.cubes = int(cubes)
        self.success_radius = float(success_radius)
        self._env = cube_env.CubeEnv(
            env_type=_CUBE_TYPES[self.cubes],
            permute_blocks=False,
            ob_type="pixels",
            width=self.size,
            height=self.size,
            mode="data_collection",
            visualize_info=False,
            terminate_at_goal=False,
        )
        self._env.reset(seed=seed)
        model = self._env.model
        self._nq, self._nv = int(model.nq), int(model.nv)
        self.state_dim = self._nq + self._nv
        #: ``qpos`` index of each cube's position (x, y, z, then a quaternion).
        self.cube_qpos = [
            int(model.jnt_qposadr[model.joint(f"object_joint_{i}").id])
            for i in range(self.cubes)
        ]
        self._target_bounds = np.asarray(self._env._target_sampling_bounds)  # noqa: SLF001

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def unwrapped(self) -> Any:
        """OGBench's environment, for its oracles and data tools."""
        return self._env

    def cube_positions(self, state: Any) -> np.ndarray:
        """Positions ``(cubes, 3)`` of the cubes in ``state``."""
        s = np.asarray(state, np.float64)
        return np.stack([s[i : i + 3] for i in self.cube_qpos])

    # ------------------------------------------------------------ GoalEnv API

    def seed(self, seed: Optional[int] = None) -> None:
        super().seed(seed)
        if seed is not None:
            self._env.reset(seed=int(seed))

    def _reset_random(self) -> None:
        # OGBench's data-collection reset draws the arm and the cubes.
        self._env.reset(seed=int(self.np_random.integers(2**31)))

    def _sample_goal(self) -> np.ndarray:
        goal = self._get_state()
        for addr in self.cube_qpos:
            xy = self.np_random.uniform(*self._target_bounds)
            goal[addr : addr + 3] = (xy[0], xy[1], 0.02)
            goal[addr + 3 : addr + 7] = (1.0, 0.0, 0.0, 0.0)
        goal[self._nq :] = 0.0
        return goal

    def _apply_action(self, action: np.ndarray) -> None:
        self._env.step(action.astype(np.float32))

    def _get_state(self) -> np.ndarray:
        data = self._env.data
        return np.concatenate([data.qpos, data.qvel]).astype(np.float64)

    def _set_state(self, state: np.ndarray) -> None:
        self._env.set_state(state[: self._nq].copy(), state[self._nq :].copy())

    def _render(self) -> np.ndarray:
        return np.asarray(self._env.render(), np.uint8)

    def goal_distance(self, state: Any, goal: Any) -> float:
        """The largest distance, in metres, of a cube from its goal position."""
        gaps = np.linalg.norm(
            self.cube_positions(state) - self.cube_positions(goal), axis=-1
        )
        return float(gaps.max())

    def is_success(self, state: Any, goal: Any) -> bool:
        return self.goal_distance(state, goal) <= self.success_radius

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


def make_ogbench_cube_env(env_id: str = "cube", **kwargs: Any) -> OGBenchCubeEnv:
    """Create OGBench Cube, e.g. ``make_ogbench_cube_env(cubes=1, size=224)``."""
    return OGBenchCubeEnv(**kwargs)


__all__ = ["OGBenchCubeEnv", "make_ogbench_cube_env"]
