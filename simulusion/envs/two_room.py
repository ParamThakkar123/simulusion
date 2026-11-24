"""Two-Room: point-mass navigation between two rooms joined by a door.

The environment of Sobal et al. (2025), "Learning from Reward-Free Offline
Data: A Case for Planning with Latent Dynamics Models" (PLDM, Sec. 4.1 and
App. C.1), also used by LeWorldModel (Maes et al., 2026). A square arena,
``arena`` units on a side, is split by a vertical wall with one door. The
agent is a point with no momentum: each action is the displacement of its
position, with norm at most ``max_step`` (2.45). A move that would cross a
wall stops where it meets the wall.

The paper fixes the arena (64) and the step norm; it does not give the wall
thickness, the door's size or a success radius, so they are arguments here,
with the defaults below. The frame is drawn at any ``size``: the walls in
white on black and the agent as a red dot, so a 64-pixel frame shows the
arena one unit per pixel and a 224-pixel frame (LeWM's input) the same scene
sharper.

State: ``[x, y]``. Actions: ``(dx, dy)`` in ``[-max_step, max_step]``, with
the norm clipped to ``max_step``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from simulusion.envs.goal import GoalEnv

#: Arena side, in units (PLDM: observations are 64 x 64).
ARENA = 64.0
#: Largest displacement per step (PLDM, Sec. 4.1).
MAX_STEP = 2.45
# Moves stop this far before a wall, so the agent never sits on one.
_SKIN = 1e-6


class TwoRoomEnv(GoalEnv):
    """Navigate from one room to a goal in the other, through the door.

    Args:
        size: Frame size in pixels (64 in PLDM, 224 for LeWM).
        arena: Side of the square arena, in units.
        wall_thickness: Width of the dividing wall and of the border walls.
        door_center: Height of the door's centre; ``None`` puts it in the
            middle.
        door_width: Height of the door opening.
        max_step: Largest step norm.
        success_radius: The goal counts as reached within this distance.
        agent_radius: Radius of the drawn agent, in units (drawing only).
        max_episode_steps: Steps per episode (200, PLDM's evaluation budget).
        terminate_on_success: End the episode once the goal is reached.
        seed: Seed of the random start and goal.

    Without a ``state`` option, ``reset`` starts the agent anywhere in the
    free space; without a ``goal_state`` option, the goal is a random point
    in the other room.
    """

    state_dim = 2

    def __init__(
        self,
        *,
        size: int = 64,
        arena: float = ARENA,
        wall_thickness: float = 2.0,
        door_center: Optional[float] = None,
        door_width: float = 10.0,
        max_step: float = MAX_STEP,
        success_radius: float = 4.0,
        agent_radius: float = 1.5,
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
        self.arena = float(arena)
        self.wall_thickness = float(wall_thickness)
        self.door_center = self.arena / 2 if door_center is None else float(door_center)
        self.door_width = float(door_width)
        self.max_step = float(max_step)
        self.success_radius = float(success_radius)
        self.agent_radius = float(agent_radius)
        if not 0 < self.door_width < self.arena - 2 * self.wall_thickness:
            raise ValueError("door_width must fit inside the wall.")
        half = self.wall_thickness / 2
        mid = self.arena / 2
        t = self.wall_thickness
        a = self.arena
        door_lo = self.door_center - self.door_width / 2
        door_hi = self.door_center + self.door_width / 2
        #: Axis-aligned wall rectangles ``(x0, y0, x1, y1)``.
        self.walls = np.array(
            [
                (0.0, 0.0, a, t),  # bottom border
                (0.0, a - t, a, a),  # top border
                (0.0, 0.0, t, a),  # left border
                (a - t, 0.0, a, a),  # right border
                (mid - half, 0.0, mid + half, door_lo),  # wall below the door
                (mid - half, door_hi, mid + half, a),  # wall above the door
            ],
            np.float64,
        )
        self._pos: np.ndarray = np.array([mid / 2, mid], np.float64)
        self._wall_mask = self._draw_walls()
        from simulusion.utils.gym_compat import gym

        self._action_space = gym.spaces.Box(
            -self.max_step, self.max_step, (2,), dtype=np.float32
        )

    @property
    def action_space(self) -> Any:
        return self._action_space

    # -------------------------------------------------------------- geometry

    def room_of(self, point: Any) -> int:
        """``0`` for the left room, ``1`` for the right (by the wall's centre line)."""
        return int(np.asarray(point, np.float64)[0] >= self.arena / 2)

    def is_free(self, point: Any) -> bool:
        """Whether ``point`` lies outside every wall."""
        x, y = np.asarray(point, np.float64)
        return not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self.walls)

    def sample_free(self, room: Optional[int] = None) -> np.ndarray:
        """A uniformly random free point, in ``room`` if given."""
        t = self.wall_thickness
        half = self.wall_thickness / 2
        mid = self.arena / 2
        for _ in range(10000):
            if room is None:
                x = self.np_random.uniform(t, self.arena - t)
            elif room == 0:
                x = self.np_random.uniform(t, mid - half)
            else:
                x = self.np_random.uniform(mid + half, self.arena - t)
            y = self.np_random.uniform(t, self.arena - t)
            point = np.array([x, y])
            if self.is_free(point):
                return point
        raise RuntimeError("Could not sample a free point.")

    def move(self, start: Any, action: Any) -> np.ndarray:
        """Where a step of ``action`` from ``start`` ends, stopping at walls."""
        p = np.asarray(start, np.float64)
        d: np.ndarray = np.asarray(action, np.float64).reshape(2)
        norm = float(np.linalg.norm(d))
        if norm > self.max_step:
            d = d * (self.max_step / norm)
        t_hit = 1.0
        for rect in self.walls:
            t = _segment_enters_box(p, d, rect)
            if t is not None and t < t_hit:
                t_hit = t
        if t_hit < 1.0:
            length = float(np.linalg.norm(d))
            back = _SKIN / length if length > 0 else 0.0
            t_hit = max(0.0, t_hit - back)
        return p + t_hit * d

    # ----------------------------------------------------------- GoalEnv API

    def _reset_random(self) -> None:
        self._pos = self.sample_free()

    def _sample_goal(self) -> np.ndarray:
        return self.sample_free(room=1 - self.room_of(self._pos))

    def _apply_action(self, action: np.ndarray) -> None:
        self._pos = self.move(self._pos, action)

    def _get_state(self) -> np.ndarray:
        return self._pos.copy()

    def _set_state(self, state: np.ndarray) -> None:
        self._pos = np.asarray(state, np.float64).copy()

    def goal_distance(self, state: Any, goal: Any) -> float:
        return float(
            np.linalg.norm(np.asarray(state, np.float64) - np.asarray(goal, np.float64))
        )

    def is_success(self, state: Any, goal: Any) -> bool:
        return self.goal_distance(state, goal) <= self.success_radius

    # -------------------------------------------------------------- drawing

    def _draw_walls(self) -> np.ndarray:
        scale = self.size / self.arena
        centers = (np.arange(self.size) + 0.5) / scale
        mask = np.zeros((self.size, self.size), bool)
        for x0, y0, x1, y1 in self.walls:
            cols = (centers >= x0) & (centers <= x1)
            rows = (centers >= y0) & (centers <= y1)
            mask |= rows[:, None] & cols[None, :]
        # Image rows run top to bottom; the arena's y axis points up.
        return mask[::-1]

    def _render(self) -> np.ndarray:
        frame = np.zeros((self.size, self.size, 3), np.uint8)
        frame[self._wall_mask] = 255
        scale = self.size / self.arena
        cx = self._pos[0] * scale
        cy = self.size - self._pos[1] * scale
        r = max(self.agent_radius * scale, 0.75)
        coords = np.arange(self.size) + 0.5
        dist2 = (coords[None, :] - cx) ** 2 + (coords[:, None] - cy) ** 2
        # Anti-aliased disc: full inside, a one-pixel ramp at the edge.
        alpha = np.clip(r + 0.5 - np.sqrt(dist2), 0.0, 1.0)[..., None]
        red = np.array([230, 40, 40], np.float64)
        frame = (frame * (1 - alpha) + red * alpha).round().astype(np.uint8)
        return frame


def _segment_enters_box(
    p: np.ndarray, d: np.ndarray, rect: np.ndarray
) -> Optional[float]:
    """First ``t`` in ``[0, 1]`` where ``p + t d`` enters the box, if it does."""
    x0, y0, x1, y1 = rect
    t_enter, t_exit = 0.0, 1.0
    for axis, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
        if abs(d[axis]) < 1e-12:
            if p[axis] < lo or p[axis] > hi:
                return None
            continue
        a = (lo - p[axis]) / d[axis]
        b = (hi - p[axis]) / d[axis]
        if a > b:
            a, b = b, a
        t_enter = max(t_enter, a)
        t_exit = min(t_exit, b)
        if t_enter > t_exit:
            return None
    return t_enter


def make_two_room_env(env_id: str = "two_room", **kwargs: Any) -> TwoRoomEnv:
    """Create Two-Room, e.g. ``make_two_room_env(size=224, seed=0)``."""
    return TwoRoomEnv(**kwargs)


__all__ = ["ARENA", "MAX_STEP", "TwoRoomEnv", "make_two_room_env"]
