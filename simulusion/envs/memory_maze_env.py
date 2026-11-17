"""Memory Maze adapter for Simulusion image-based agents.

Memory Maze (Pasukonis et al., 2022) is a 3D first-person navigation task built
on dm_control. The agent must reach a sequence of coloured targets in a
randomly generated maze, so scoring well requires remembering the layout over
episodes of 1000 to 4000 steps. It is the standard long-horizon memory benchmark
for world models (TransDreamer, Recall to Imagine, and the Memory Maze paper's
own Dreamer baselines).

Four sizes are provided. Episode lengths are fixed by the task (250 to 1000
seconds at 4 Hz):

============  =====================  =======  =============
Task          Gym id                 Targets  Episode steps
============  =====================  =======  =============
``9x9``       MemoryMaze-9x9-v0      3        1000
``11x11``     MemoryMaze-11x11-v0    4        2000
``13x13``     MemoryMaze-13x13-v0    5        3000
``15x15``     MemoryMaze-15x15-v0    6        4000
============  =====================  =======  =============

Observations are ``{"image": (3, H, W) uint8, "target_color": (3,) float32}``:
the first-person view plus the RGB colour of the current target, which the
agent also sees as a border in the image (``target_color_in_image``). The six
discrete actions are exposed one-hot-like. Episodes end only at the time limit,
which is reported as a truncation with discount 1.

Generating mazes needs labmaze's compiled ``RandomMaze``, which on CPython 3.13
comes from ``labmaze-new>=0.0.3``. Rendering uses MuJoCo; this module selects
the GLFW backend on Windows, where Memory Maze's default (EGL) is unavailable.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import numpy as np

from simulusion.envs._actions import encode_discrete_action
from simulusion.envs._contract import finalize_step_info
from simulusion.envs._images import to_chw_uint8, to_hwc_uint8

#: Maze sizes and their episode lengths in agent steps at the default 4 Hz.
MEMORY_MAZE_TASKS: dict[str, int] = {
    "9x9": 1000,
    "11x11": 2000,
    "13x13": 3000,
    "15x15": 4000,
}


def normalize_memory_maze_task(task: str) -> str:
    """Map ``"9x9"``, ``"memory_maze_9x9"`` or ``"MemoryMaze-9x9-v0"`` to ``"9x9"``."""
    key = str(task).strip()
    lowered = key.lower()
    for prefix in ("memorymaze-", "memory_maze_", "memory-maze-", "memmaze_"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    if lowered.endswith("-v0"):
        lowered = lowered[: -len("-v0")]
    if lowered.endswith(("-extobs", "-hd", "-top")):
        raise ValueError(
            f"{task!r} is a Memory Maze variant with different observations. "
            "MemoryMazeEnv builds the standard 64x64 first-person task; pass "
            "task options such as camera_resolution or global_observables as "
            "keyword arguments instead."
        )
    if lowered not in MEMORY_MAZE_TASKS:
        valid = ", ".join(MEMORY_MAZE_TASKS)
        raise ValueError(f"Unknown Memory Maze task {task!r}. Valid sizes: {valid}.")
    return lowered


def _select_mujoco_gl() -> None:
    # memory_maze sets MUJOCO_GL=egl on import when it is unset. EGL does not
    # exist on Windows, so pick GLFW there before that import runs.
    if "MUJOCO_GL" not in os.environ and sys.platform.startswith("win"):
        os.environ["MUJOCO_GL"] = "glfw"


def _require_memory_maze_tasks() -> Any:
    _select_mujoco_gl()
    try:
        tasks = importlib.import_module("memory_maze.tasks")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Memory Maze backend requires the 'memory-maze' package. Install "
            "it with `pip install simulusion[memory-maze]`."
        ) from exc
    except TypeError as exc:
        # labmaze without its compiled extension: memory_maze subclasses
        # labmaze.RandomMaze, which is then None.
        raise ImportError(
            "Memory Maze needs labmaze's compiled RandomMaze, which this "
            "labmaze install lacks. On CPython 3.13 install "
            "`labmaze-new>=0.0.3` with a C++17 compiler available "
            "(`pip install --force-reinstall --no-cache-dir labmaze-new`), or use Python 3.12 "
            "or older with `labmaze`."
        ) from exc
    return tasks


class MemoryMazeEnv:
    """Memory Maze with Simulusion's image observation and one-hot action contract.

    Args:
        task: Maze size, see :func:`normalize_memory_maze_task`.
        seed: Seed for maze generation and target placement.
        size: Output image size ``(height, width)``. The task renders at
            ``camera_resolution``; other sizes are resized.
        camera_resolution: Native render resolution (64 in the benchmark).
        **task_kwargs: Forwarded to ``memory_maze.tasks.memory_maze_<size>``,
            for example ``good_visibility=True`` or ``randomize_colors=True``.
    """

    def __init__(
        self,
        task: str = "9x9",
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        camera_resolution: int = 64,
        **task_kwargs: Any,
    ) -> None:
        from simulusion.envs._spaces import OneHotActionSpace, image_space
        from simulusion.utils.gym_compat import gym

        self.task = normalize_memory_maze_task(task)
        self._seed = int(seed)
        self._size = (int(size[0]), int(size[1]))
        self._camera_resolution = int(camera_resolution)
        for reserved in ("discrete_actions", "image_only_obs", "seed"):
            if reserved in task_kwargs:
                raise ValueError(f"{reserved!r} is fixed by MemoryMazeEnv.")
        self._task_kwargs = dict(task_kwargs)
        self._env = self._make_env(self._seed)

        spec = self._env.action_spec()
        self._num_actions = int(spec.num_values)
        self._action_space = OneHotActionSpace(self._num_actions, seed=self._seed)
        self._observation_space = gym.spaces.Dict(
            {
                "image": image_space(self._size),
                "target_color": gym.spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
            }
        )
        self._last_frame: np.ndarray | None = None

    def _make_env(self, seed: int) -> Any:
        tasks = _require_memory_maze_tasks()
        factory = getattr(tasks, "memory_maze_" + self.task)
        return factory(
            seed=int(seed),
            discrete_actions=True,
            image_only_obs=False,
            camera_resolution=self._camera_resolution,
            **self._task_kwargs,
        )

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        """Episode length at the default 4 Hz control frequency."""
        control_freq = float(self._task_kwargs.get("control_freq", 4.0))
        return int(round(MEMORY_MAZE_TASKS[self.task] * control_freq / 4.0))

    def _observe(self, time_step: Any) -> dict[str, np.ndarray]:
        observation = time_step.observation
        self._last_frame = to_hwc_uint8(observation["image"])
        target = np.asarray(
            observation.get("target_color", np.zeros(3)), dtype=np.float32
        ).reshape(3)
        return {
            "image": to_chw_uint8(self._last_frame, self._size),
            "target_color": target,
        }

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._seed = int(seed)
            self._env.close()
            self._env = self._make_env(self._seed)
            self._action_space.seed(self._seed)
        return self._observe(self._env.reset())

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        index, model_action = encode_discrete_action(action, self._num_actions)
        time_step = self._env.step(index)
        obs = self._observe(time_step)
        done = bool(time_step.last())
        discount = 1.0 if time_step.discount is None else float(time_step.discount)
        # The time limit is the only way a Memory Maze episode ends, with
        # discount 1. A discount of 0 would mean a genuine terminal state.
        terminated = done and discount == 0.0
        info = finalize_step_info(
            {
                "action": model_action,
                "executed_action": index,
                "vector_observation": obs["target_color"].copy(),
            },
            done=done,
            terminated=terminated,
            truncated=done and not terminated,
            discount=np.array(discount, np.float32),
        )
        return obs, float(time_step.reward or 0.0), done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if self._last_frame is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._last_frame.copy()

    def close(self) -> None:
        self._env.close()


def make_memory_maze_env(env_id: str = "9x9", **kwargs: Any) -> MemoryMazeEnv:
    """Create a Memory Maze task, e.g. ``make_memory_maze_env("MemoryMaze-9x9-v0")``."""
    return MemoryMazeEnv(env_id, **kwargs)


__all__ = [
    "MEMORY_MAZE_TASKS",
    "MemoryMazeEnv",
    "make_memory_maze_env",
    "normalize_memory_maze_task",
]
