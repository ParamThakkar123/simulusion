"""Scripted policies that collect offline data in the goal environments.

Reward-free world models learn from logged episodes, and what the episodes
cover decides what the model can plan. These policies produce the kinds of
data the papers describe; :func:`simulusion.data.collect.collect_goal_episodes`
runs one and writes an episode folder.

=====================  ================================================================
Policy                 Data
=====================  ================================================================
:class:`VonMisesWalk`  Two-Room: a random heading per episode, each step's direction
                       drawn around it (PLDM, Sec. 4.1: concentration 5, step length
                       uniform up to the maximum).
:class:`DoorSeeker`    Two-Room: a noisy straight line to the door, then to a point in
                       the other room (LeWM's dataset, App. E).
:class:`BlockPusher`   Push-T: the pusher moves to points near the block more often
                       than anywhere else, so it pushes the block in varied ways.
:class:`CubeOracle`    OGBench Cube: OGBench's own scripted pick-and-place, with action
                       noise, picking a new target each time one is placed.
:class:`PoseSeeker`    Reacher: a PD controller towards random poses, re-drawn every
                       few steps, with torque noise.
:class:`RandomPolicy`  Any environment: uniform actions.
=====================  ================================================================

A policy is called as ``policy(env, observation)`` and may read the
environment's true state (:meth:`~simulusion.envs.goal.GoalEnv.get_state`);
:meth:`reset` is called after each ``env.reset()``. Policies draw from
their own generator, seeded at construction.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


class GoalPolicy:
    """Base class: ``reset(env)`` then ``policy(env, observation)`` per step.

    A policy whose :attr:`done` turns true ends its episode early, so
    episodes can vary in length (as LeWM's Two-Room data does).
    """

    #: True once the policy has finished its episode.
    done: bool = False

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, env: Any) -> None:
        """Start a new episode in ``env``."""

    def __call__(self, env: Any, observation: Any) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(GoalPolicy):
    """Uniform random actions within the action space."""

    def __call__(self, env: Any, observation: Any) -> np.ndarray:
        space = env.action_space
        return self.rng.uniform(space.low, space.high).astype(np.float32)


# ----------------------------------------------------------------- Two-Room


class VonMisesWalk(GoalPolicy):
    """Two-Room exploration of PLDM (Sobal et al., 2025, Sec. 4.1).

    Each episode draws a heading uniformly; every step's direction is drawn
    from a von Mises distribution around it with concentration ``kappa`` (5),
    and its length uniformly from ``[0, max_step]``.
    """

    def __init__(self, kappa: float = 5.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.kappa = float(kappa)
        self._heading = 0.0

    def reset(self, env: Any) -> None:
        self._heading = float(self.rng.uniform(0.0, 2 * math.pi))

    def __call__(self, env: Any, observation: Any) -> np.ndarray:
        angle = self.rng.vonmises(self._heading, self.kappa)
        length = self.rng.uniform(0.0, env.max_step)
        return np.array([math.cos(angle), math.sin(angle)], np.float32) * length


class DoorSeeker(GoalPolicy):
    """Two-Room: head for the door, then for a random point in the other room.

    Each step moves ``speed`` times ``max_step`` along the current direction,
    plus Gaussian noise of ``noise`` times ``max_step`` per axis. LeWM does
    not give its heuristic's speed or noise; the defaults (0.3, 0.3) make
    trips of about 80 steps, and with LeWM's protocol (goals 100 steps ahead)
    put most goals in the other room, where a random agent rarely reaches
    them (about 12%, against 0% in the paper). On reaching its point
    (within ``arrive``) it ends the episode, so episodes are one trip of
    varying length, like LeWM's Two-Room data (App. E: 92 steps on average).
    With ``single_trip=False`` it picks a new point in the opposite room
    instead and keeps crossing until the episode's length runs out.
    """

    def __init__(
        self,
        speed: float = 0.3,
        noise: float = 0.3,
        arrive: float = 2.0,
        single_trip: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed)
        self.speed = float(speed)
        self.noise = float(noise)
        self.arrive = float(arrive)
        self.single_trip = bool(single_trip)
        self._target = np.zeros(2)
        self._through = False

    def reset(self, env: Any) -> None:
        self.done = False
        self._new_target(env)

    def _new_target(self, env: Any) -> None:
        self._room = env.room_of(env.get_state())
        saved = env.np_random
        env.np_random = self.rng
        try:
            self._target = env.sample_free(room=1 - self._room)
        finally:
            env.np_random = saved
        self._through = False

    def __call__(self, env: Any, observation: Any) -> np.ndarray:
        pos = env.get_state()
        if not self._through and env.room_of(pos) != self._room:
            self._through = True
        if self._through and np.linalg.norm(self._target - pos) <= self.arrive:
            if self.single_trip:
                self.done = True
            else:
                self._new_target(env)
        door = np.array([env.arena / 2, env.door_center])
        aim = self._target if self._through else door
        gap = aim - pos
        dist = float(np.linalg.norm(gap))
        stride = self.speed * env.max_step
        step = gap / dist * min(stride, dist) if dist > 1e-9 else np.zeros(2)
        step = step + self.rng.normal(0.0, self.noise * env.max_step, 2)
        return step.astype(np.float32)


# -------------------------------------------------------------------- Push-T


class BlockPusher(GoalPolicy):
    """Push-T: move towards random points, mostly ones near the block.

    Every ``hold`` steps (drawn from ``[hold_min, hold_max]``) a new point is
    picked: with probability ``near_block`` it is the block's position plus
    Gaussian noise of ``spread`` pixels, otherwise uniform on the table. The
    action moves the pusher towards it by at most ``max_move`` pixels per step.
    """

    def __init__(
        self,
        near_block: float = 0.8,
        spread: float = 60.0,
        hold_min: int = 3,
        hold_max: int = 10,
        max_move: float = 60.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed)
        self.near_block = float(near_block)
        self.spread = float(spread)
        self.hold_min, self.hold_max = int(hold_min), int(hold_max)
        self.max_move = float(max_move)
        self._point = np.zeros(2)
        self._left = 0

    def reset(self, env: Any) -> None:
        self._left = 0

    def __call__(self, env: Any, observation: Any) -> np.ndarray:
        from simulusion.envs.pusht import TABLE

        state = env.get_state()
        if self._left <= 0:
            if self.rng.random() < self.near_block:
                point = state[2:4] + self.rng.normal(0.0, self.spread, 2)
            else:
                point = self.rng.uniform(0.0, TABLE, 2)
            self._point = np.clip(point, 20.0, TABLE - 20.0)
            self._left = int(self.rng.integers(self.hold_min, self.hold_max + 1))
        self._left -= 1
        gap = self._point - state[:2]
        dist = float(np.linalg.norm(gap))
        move = gap if dist <= self.max_move else gap / dist * self.max_move
        if env.action_mode == "absolute":
            return (state[:2] + move).astype(np.float32)
        return np.clip(move, -env.max_offset, env.max_offset).astype(np.float32)


# ---------------------------------------------------------------------- Cube


class CubeOracle(GoalPolicy):
    """OGBench Cube: OGBench's scripted pick-and-place, with action noise.

    Uses OGBench's Markov oracle on the simulator's privileged state (cube
    and target poses); each time a cube is placed, OGBench draws a new target.
    Gaussian noise of ``noise`` is added to every action, which is clipped to
    ``[-1, 1]``.
    """

    def __init__(self, noise: float = 0.1, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.noise = float(noise)
        self._oracle: Any = None

    def reset(self, env: Any) -> None:
        from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

        inner = env.unwrapped
        self._oracle = CubeMarkovOracle(env=inner)
        self._oracle.reset(None, inner.compute_ob_info())

    def __call__(self, env: Any, observation: Any) -> np.ndarray:
        inner = env.unwrapped
        info = inner.compute_ob_info()
        if self._oracle.done:
            inner.set_new_target(return_info=False)
            info = inner.compute_ob_info()
            self._oracle.reset(None, info)
        action = np.asarray(self._oracle.select_action(None, info), np.float64)
        action = action + self.rng.normal(0.0, self.noise, action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------- Reacher


class PoseSeeker(GoalPolicy):
    """Reacher: a PD controller towards random poses, plus torque noise.

    A new pose is drawn every ``hold_min`` to ``hold_max`` steps; the torque is
    ``kp * angle_error - kd * velocity`` plus Gaussian ``noise``, clipped to
    the action range.
    """

    def __init__(
        self,
        kp: float = 2.0,
        kd: float = 0.2,
        noise: float = 0.3,
        hold_min: int = 10,
        hold_max: int = 40,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed)
        self.kp, self.kd, self.noise = float(kp), float(kd), float(noise)
        self.hold_min, self.hold_max = int(hold_min), int(hold_max)
        self._pose: np.ndarray = np.zeros(2)
        self._left = 0

    def reset(self, env: Any) -> None:
        self._left = 0

    def __call__(self, env: Any, observation: Any) -> np.ndarray:
        state = env.get_state()
        if self._left <= 0:
            self._pose = self.rng.uniform(-math.pi, math.pi, 2)
            self._left = int(self.rng.integers(self.hold_min, self.hold_max + 1))
        self._left -= 1
        error = (self._pose - state[:2] + math.pi) % (2 * math.pi) - math.pi
        torque = self.kp * error - self.kd * state[2:4]
        torque = torque + self.rng.normal(0.0, self.noise, 2)
        space = env.action_space
        return np.clip(torque, space.low, space.high).astype(np.float32)


#: The default collection policy of each environment.
DEFAULT_POLICIES = {
    "two_room": DoorSeeker,
    "pusht": BlockPusher,
    "cube": CubeOracle,
    "reacher_goal": PoseSeeker,
}

POLICIES = {
    "random": RandomPolicy,
    "von_mises": VonMisesWalk,
    "door": DoorSeeker,
    "block_pusher": BlockPusher,
    "cube_oracle": CubeOracle,
    "pose_seeker": PoseSeeker,
}


__all__ = [
    "BlockPusher",
    "CubeOracle",
    "DEFAULT_POLICIES",
    "DoorSeeker",
    "GoalPolicy",
    "POLICIES",
    "PoseSeeker",
    "RandomPolicy",
    "VonMisesWalk",
]
