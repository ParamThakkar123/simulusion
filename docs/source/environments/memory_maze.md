# Memory Maze

[Memory Maze](https://github.com/jurgisp/memory-maze) (Pasukonis et al., 2022)
is a 3D first-person navigation task on dm_control. The agent must reach a
sequence of coloured targets in a random maze, so doing well requires
remembering the layout over episodes of 1000 to 4000 steps. It is the standard
long-horizon memory benchmark for world models.

## Install

```bash
pip install "simulusion[memory-maze]"              # Python 3.12 and below
python -m simulusion.install_dmc --memory-maze     # any version, including 3.13
```

Memory Maze generates mazes with labmaze's compiled `RandomMaze`. On Python
3.12 and below that comes from the `labmaze` wheel that dm-control pulls in. On
3.13 it comes from [`labmaze-new`](https://pypi.org/project/labmaze-new/) 0.0.3
or later, which builds the extension with setuptools: that needs a C++17
compiler (MSVC on Windows) unless a prebuilt wheel exists for your platform.
`memory-maze` depends only on dm-control, whose `labmaze` pin cannot resolve on
3.13 (see [DeepMind Control](dmc.md)), so the installer adds it with
`--no-deps`.

`python -m simulusion.install_dmc --check --memory-maze` checks an existing
environment.

Rendering uses MuJoCo. On Windows the adapter selects `MUJOCO_GL=glfw`, because
Memory Maze's default, EGL, does not exist there. Set `MUJOCO_GL` yourself
(`egl` for headless GPU, `osmesa` for headless CPU) to override it.

## Main API

```python
from simulusion import MemoryMazeEnv

env = MemoryMazeEnv("9x9", seed=0)           # or make_memory_maze_env("MemoryMaze-9x9-v0")
obs = env.reset()
print(obs["image"].shape, obs["target_color"])    # (3, 64, 64) [r g b]
print(env.action_space.n, env.max_episode_steps)  # 6 1000
```

| Task | Gym id | Targets | Episode steps |
| --- | --- | --- | --- |
| `9x9` | `MemoryMaze-9x9-v0` | 3 | 1000 |
| `11x11` | `MemoryMaze-11x11-v0` | 4 | 2000 |
| `13x13` | `MemoryMaze-13x13-v0` | 5 | 3000 |
| `15x15` | `MemoryMaze-15x15-v0` | 6 | 4000 |

Extra keyword arguments go to `memory_maze.tasks.memory_maze_<size>`, for
example `good_visibility=True`, `randomize_colors=True` or
`camera_resolution=128`. The `ExtObs`, `HD` and `Top` Gym variants change the
observations and are not accepted as task names; use these options instead.

## Observations, actions and episodes

- `image`: the first-person view as `(3, H, W)` uint8. The current target's
  colour is drawn as a border, as in the benchmark.
- `target_color`: the target's RGB colour as a `float32` vector, also available
  as `info["vector_observation"]`.
- Actions are one-hot vectors over Memory Maze's six discrete actions (noop,
  forward, turn left, turn right, forward-left, forward-right).
- Reward is +1 for each target reached.
- Episodes end only at the time limit, reported as `truncated` with discount 1.

## Training with Dreamer

```python
from simulusion import DreamerConfig

cfg = DreamerConfig(env_backend="memory_maze", env="9x9", action_repeat=1)
```

Memory Maze fixes its own control rate (4 Hz) and episode length, so
`action_repeat` must be 1 and no `time_limit` is applied.
