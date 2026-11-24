"""Push-T: push a T-shaped block into a goal pose with a round pusher.

The task of Chi et al. (2023, Diffusion Policy), as used for goal-reaching by
DINO-WM (Zhou et al., 2025) and LeWorldModel (Maes et al., 2026): a disc
(the agent) pushes a T-shaped block on a 512 x 512 table, seen from above.
This is Simulusion's own implementation on the 2D physics engine pymunk.

* The agent is a kinematic disc of radius 15. An action is a target point;
  a PD controller (``kp``, ``kv``) drives the disc toward it. Physics runs at
  ``sim_hz`` (100) and one action lasts ``1 / control_hz`` (10 Hz), as in the
  original task. ``action_mode="relative"`` makes actions offsets from the
  agent's position instead.
* The block is a T (a 120 x 30 bar over a 30 x 90 stem) with mass 1.
  Friction with the table is modelled by damping its linear and angular
  velocity (``block_damping`` per second), so it stops soon after it is
  pushed, as a block on a table does.
* For goal-reaching (DINO-WM), success needs both the block and the agent at
  the goal: block position within ``block_pos_tol`` pixels, block angle
  within ``block_angle_tol`` radians and agent within ``agent_pos_tol``.
  The papers do not give these thresholds; the defaults are stated below.
  Without a ``goal_state`` option the goal is the classic task's: the block
  on the fixed light-green T, agent anywhere.

The light-green T is always drawn at the centre: in goal-reaching it is a
fixed visual reference, not the goal (DINO-WM, App. A.1).

State (10 values): agent ``x, y``, block ``x, y, angle``, agent velocity
``vx, vy``, block velocity ``vx, vy`` and angular velocity. The block's
position is its reference point, the middle of the bar's top edge.

Needs ``pip install simulusion[pusht]`` (pymunk).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from simulusion.envs.brax_env import _require_module
from simulusion.envs.goal import GoalEnv

#: Side of the table, in pixels of the native 512 x 512 frame.
TABLE = 512.0
#: Pose ``(x, y, angle)`` of the fixed T drawn on the table.
ANCHOR_POSE = (256.0, 256.0, math.pi / 4)

_AGENT_RADIUS = 15.0
# The T in its own frame: a bar over a stem, origin at the top edge's middle.
_BAR = ((-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0))
_STEM = ((-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0))
# Turns the block's angle error into a length for goal_distance: the T's
# half-extent, so a rotation counts like the distance its tip moves.
_ANGLE_SCALE = 60.0


def _require_pymunk() -> Any:
    return _require_module(
        "pymunk",
        "Install it with `pip install simulusion[pusht]` or `pip install pymunk`.",
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PushTEnv(GoalEnv):
    """Push a T-shaped block to a goal pose.

    Args:
        size: Frame size in pixels (DINO-WM and LeWM use 224).
        action_mode: ``"absolute"``: an action is the target point in table
            pixels, ``[0, 512]``; ``"relative"``: an offset from the agent of
            at most ``max_offset`` per axis. The controller is critically
            damped, so in one step the agent covers only part of the way to
            its target (about a quarter of a fresh offset).
        max_offset: Largest relative action per axis.
        control_hz: Actions per second (10).
        sim_hz: Physics steps per second (100).
        kp: PD gain on the position error.
        kv: PD gain on the velocity.
        block_damping: Per-second decay rate of the block's velocity (friction
            with the table).
        block_pos_tol: Block position tolerance for success, in pixels (20).
        block_angle_tol: Block angle tolerance for success, in radians (pi/9).
        agent_pos_tol: Agent position tolerance for success, in pixels (20).
        max_episode_steps: Steps per episode.
        terminate_on_success: End the episode once the goal is reached.
        seed: Seed of the random start.
    """

    state_dim = 10

    def __init__(
        self,
        *,
        size: int = 224,
        action_mode: str = "absolute",
        max_offset: float = 50.0,
        control_hz: float = 10.0,
        sim_hz: float = 100.0,
        kp: float = 100.0,
        kv: float = 20.0,
        block_damping: float = 10.0,
        block_pos_tol: float = 20.0,
        block_angle_tol: float = math.pi / 9,
        agent_pos_tol: float = 20.0,
        max_episode_steps: Optional[int] = 300,
        terminate_on_success: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            size=size,
            max_episode_steps=max_episode_steps,
            terminate_on_success=terminate_on_success,
            seed=seed,
        )
        if action_mode not in ("absolute", "relative"):
            raise ValueError("action_mode must be 'absolute' or 'relative'.")
        self.action_mode = action_mode
        self.max_offset = float(max_offset)
        self.substeps = max(1, int(round(sim_hz / control_hz)))
        self.dt = 1.0 / float(sim_hz)
        self.kp, self.kv = float(kp), float(kv)
        self.block_damping = float(block_damping)
        self.agent_mass = 10.0
        self.block_pos_tol = float(block_pos_tol)
        self.block_angle_tol = float(block_angle_tol)
        self.agent_pos_tol = float(agent_pos_tol)
        #: Whether success also needs the agent at the goal (dataset goals).
        self.check_agent = True
        self._build_space()
        from simulusion.utils.gym_compat import gym

        if action_mode == "absolute":
            self._action_space = gym.spaces.Box(0.0, TABLE, (2,), dtype=np.float32)
        else:
            self._action_space = gym.spaces.Box(
                -self.max_offset, self.max_offset, (2,), dtype=np.float32
            )

    @property
    def action_space(self) -> Any:
        return self._action_space

    # ---------------------------------------------------------------- physics

    def _build_space(self) -> None:
        pymunk = _require_pymunk()
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        # More solver passes keep the block out of the walls under a hard push.
        space.iterations = 30
        # Walls just inside the table's edge.
        for a, b in (
            ((5, 506), (5, 5)),
            ((5, 5), (506, 5)),
            ((506, 5), (506, 506)),
            ((5, 506), (506, 506)),
        ):
            wall = pymunk.Segment(space.static_body, a, b, 2.0)
            wall.friction = 1.0
            space.add(wall)
        # A heavy dynamic body whose velocity the controller sets: it pushes
        # the block, but walls still stop it (a kinematic body would force
        # the block through them).
        agent = pymunk.Body(self.agent_mass, float("inf"))
        disc = pymunk.Circle(agent, _AGENT_RADIUS)
        disc.friction = 1.0
        space.add(agent, disc)
        mass = 1.0
        moment = pymunk.moment_for_poly(mass / 2, _BAR) + pymunk.moment_for_poly(
            mass / 2, _STEM
        )
        block = pymunk.Body(mass, moment)
        shapes = [pymunk.Poly(block, _BAR), pymunk.Poly(block, _STEM)]
        for shape in shapes:
            shape.mass = mass / 2
            shape.friction = 1.0
        space.add(block, *shapes)
        self._space, self._agent, self._block = space, agent, block
        self._target = np.array([256.0, 400.0])

    def _substep(self) -> None:
        agent = self._agent
        pos = np.array(agent.position)
        vel = np.array(agent.velocity)
        acc = self.kp * (self._target - pos) - self.kv * vel
        agent.velocity = tuple(vel + acc * self.dt)
        decay = math.exp(-self.block_damping * self.dt)
        block = self._block
        block.velocity = tuple(np.array(block.velocity) * decay)
        block.angular_velocity = block.angular_velocity * decay
        self._space.step(self.dt)

    def _apply_action(self, action: np.ndarray) -> None:
        pos = np.array(self._agent.position)
        target = action if self.action_mode == "absolute" else pos + action
        self._target = np.clip(np.asarray(target, np.float64), 0.0, TABLE)
        for _ in range(self.substeps):
            self._substep()

    # ------------------------------------------------------------- GoalEnv API

    def reset(
        self, seed: Optional[int] = None, options: Optional[Any] = None
    ) -> dict[str, np.ndarray]:
        self.check_agent = options is not None and options.get("goal_state") is not None
        return super().reset(seed=seed, options=options)

    def _reset_random(self) -> None:
        rng = self.np_random
        agent: np.ndarray = np.zeros(2)
        block: np.ndarray = np.zeros(2)
        for _ in range(1000):
            agent = rng.uniform(50.0, 450.0, 2)
            block = rng.uniform(100.0, 400.0, 2)
            if np.linalg.norm(agent - block) > 120.0:
                break
        angle = rng.uniform(-math.pi, math.pi)
        self._set_state(np.array([*agent, *block, angle, 0, 0, 0, 0, 0], np.float64))

    def _sample_goal(self) -> np.ndarray:
        state = self._get_state()
        state[2:5] = ANCHOR_POSE
        state[5:] = 0.0
        return state

    def _get_state(self) -> np.ndarray:
        a, b = self._agent, self._block
        return np.array(
            [
                a.position.x,
                a.position.y,
                b.position.x,
                b.position.y,
                _wrap(b.angle),
                a.velocity.x,
                a.velocity.y,
                b.velocity.x,
                b.velocity.y,
                b.angular_velocity,
            ],
            np.float64,
        )

    def _set_state(self, state: np.ndarray) -> None:
        s = np.asarray(state, np.float64)
        self._agent.position = (s[0], s[1])
        self._agent.velocity = (s[5], s[6])
        # pymunk turns a body about its centre of mass, which moves its
        # origin: set the angle first, then the position.
        self._block.angle = float(s[4])
        self._block.position = (s[2], s[3])
        self._block.velocity = (s[7], s[8])
        self._block.angular_velocity = float(s[9])
        self._target = s[:2].copy()
        self._space.reindex_shapes_for_body(self._agent)
        self._space.reindex_shapes_for_body(self._block)

    def errors(self, state: Any, goal: Any) -> tuple[float, float, float]:
        """Agent position, block position and block angle errors."""
        s, g = np.asarray(state, np.float64), np.asarray(goal, np.float64)
        return (
            float(np.linalg.norm(s[:2] - g[:2])),
            float(np.linalg.norm(s[2:4] - g[2:4])),
            abs(_wrap(float(s[4] - g[4]))),
        )

    def goal_distance(self, state: Any, goal: Any) -> float:
        """Block position error plus its angle error as an arc, plus the agent's error."""
        agent, block, angle = self.errors(state, goal)
        return block + _ANGLE_SCALE * angle + (agent if self.check_agent else 0.0)

    def is_success(self, state: Any, goal: Any) -> bool:
        agent, block, angle = self.errors(state, goal)
        ok = block <= self.block_pos_tol and angle <= self.block_angle_tol
        return ok and (not self.check_agent or agent <= self.agent_pos_tol)

    # ---------------------------------------------------------------- drawing

    def _render(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        ss = 2  # supersampling, for smooth edges
        px = self.size * ss
        scale = px / TABLE
        img = Image.new("RGB", (px, px), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        def poly(
            pose: tuple[float, float, float], pts: Any
        ) -> list[tuple[float, float]]:
            x, y, a = pose
            c, s = math.cos(a), math.sin(a)
            return [
                ((x + c * u - s * v) * scale, (y + s * u + c * v) * scale)
                for u, v in pts
            ]

        for part in (_BAR, _STEM):
            draw.polygon(poly(ANCHOR_POSE, part), fill=(144, 238, 144))
        b = self._block
        pose = (b.position.x, b.position.y, b.angle)
        for part in (_BAR, _STEM):
            draw.polygon(poly(pose, part), fill=(119, 136, 153))
        ax, ay = self._agent.position
        r = _AGENT_RADIUS * scale
        draw.ellipse(
            (ax * scale - r, ay * scale - r, ax * scale + r, ay * scale + r),
            fill=(65, 105, 225),
        )
        img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
        return np.asarray(img, np.uint8)


def make_pusht_env(env_id: str = "pusht", **kwargs: Any) -> PushTEnv:
    """Create Push-T, e.g. ``make_pusht_env(size=224, seed=0)``."""
    return PushTEnv(**kwargs)


__all__ = ["ANCHOR_POSE", "TABLE", "PushTEnv", "make_pusht_env"]
