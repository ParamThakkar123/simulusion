# Environment Backends

Simulusion ships environment adapters for pixel-based world-model training, model-based reinforcement learning, and benchmark collection. Each backend page explains the installation requirements, factory functions, observation and action conventions, configuration fields, and common troubleshooting steps for that backend.

```{toctree}
:maxdepth: 1

DeepMind Control Suite <dmc>
DeepMind Lab <dmlab>
Gym and Gymnasium <gym>
Brax <brax>
Atari <atari>
Atari 100k <atari100k>
BSuite <bsuite>
Crafter <crafter>
Craftax <craftax>
Memory Maze <memory_maze>
Minecraft Diamond <minecraft>
Procgen <procgen>
MuJoCo <mujoco>
Gymnasium Robotics <robotics>
Unity ML-Agents <unity>
Info Contract <info_contract>
Vectorized Environments <vectorized>
World Model Env <world_model>
Wrappers <wrappers>
```

## Choosing a backend

| Backend | Best for | Primary APIs | Typical observations | Typical actions |
| --- | --- | --- | --- | --- |
| [DeepMind Control Suite](dmc.md) | Dreamer-style continuous-control tasks with state and rendered image observations | `DeepMindControlEnv`, `env_backend="dmc"` | Dict with DMC state keys plus `image` | Continuous `Box` from the DMC action spec |
| [DeepMind Lab](dmlab.md) | 3D navigation and puzzle tasks from DeepMind Lab | `DMLabEnv`, `make_dmlab_env`, `env_backend="dmlab"` | Dict with `image` plus requested Lab observations | Normalized one-hot `Box[-1, 1]` over native Lab actions |
| [Gym and Gymnasium](gym.md) | Classic control, Box2D, custom Gym environments, and generic rendered tasks | `GymImageEnv`, `make_gym_env`, `env_backend="gym"` | Dict with `image` only | Original continuous `Box` or one-hot vector for discrete actions |
| [DeepMind BSuite](bsuite.md) | Small diagnostic RL benchmark tasks such as `catch/0` and `deep_sea/0` | `BSuiteImageEnv`, `make_bsuite_env`, `env_backend="bsuite"` | Dict with synthetic `image`, the flat `state` vector, or both | One-hot vector for discrete actions |
| [Minecraft Diamond](minecraft.md) | DreamerV3's Minecraft benchmark: obtain a diamond from sparse milestone rewards | `MinecraftDiamondEnv`, `env_backend="minecraft"` | Dict with `image` plus inventory and status vectors | One-hot vector over 25 actions |
| [Brax](brax.md) | JAX/Brax continuous-control tasks through a Gym-like image adapter | `BraxImageEnv`, `make_brax_env`, `env_backend="brax"` | Dict with synthesized `image`; raw vector in `info["vector_observation"]` | Continuous `Box[-1, 1]` matching `env.action_size` |
| [Atari](atari.md) | Atari 2600 environments through Gymnasium/ALE | `make_atari_env`, `make_atari_vector_env` | ALE RGB/RAM observations | Discrete Atari actions |
| [Atari 100k](atari100k.md) | The 26 Atari 100k games under the exact IRIS, DIAMOND, STORM or DreamerV3 preprocessing | `Atari100kEnv`, `make_atari100k_env`, `env_backend="atari100k"` | Dict with 64x64 `image` | One-hot vector over the minimal action set |
| [Crafter](crafter.md) | DreamerV3's open-world survival benchmark, scored by achievement success rates | `CrafterEnv`, `crafter_score`, `env_backend="crafter"` | Dict with `image` | One-hot vector over 17 actions |
| [Craftax](craftax.md) | JAX Crafter and its extended version | `CraftaxEnv`, `make_craftax_env`, `env_backend="craftax"` | Dict with `image`, or `state` for symbolic variants | One-hot vector over 17 or 43 actions |
| [Memory Maze](memory_maze.md) | Long-horizon memory: 3D navigation to remembered targets | `MemoryMazeEnv`, `make_memory_maze_env`, `env_backend="memory_maze"` | Dict with `image` and `target_color` | One-hot vector over 6 actions |
| [Procgen](procgen.md) | Procedurally generated benchmark games | `ProcgenImageEnv`, `make_procgen_env`, `env_backend="procgen"` | Dict with `image` | One-hot vector for discrete Procgen actions |
| [MuJoCo](mujoco.md) | Gymnasium MuJoCo task ids and native MJCF/MJB models | `make_mujoco_env` | Image dict via `GymImageEnv`/`MuJoCoImageEnv` | Continuous `Box` |
| [Gymnasium Robotics](robotics.md) | All ids registered by the installed Gymnasium Robotics package, including moved legacy MuJoCo v2/v3 ids | `make_robotics_env`, `list_gymnasium_robotics_envs` | Image dict via `GymImageEnv` | Continuous `Box` |
| [Unity ML-Agents](unity.md) | External Unity executable simulations with continuous-control behaviors | `UnityMLAgentsEnv`, `env_backend="unity_mlagents"` | Dict with `image` | Continuous `Box[-1, 1]` |
| [Vectorized environments](vectorized.md) | Multiprocess/vector rollout collection and native ALE vectorization | `TorchVectorizedEnv`, `make_atari_vector_env` | Batched observations | Batched actions |
| [World Model Env](world_model.md) | Model-based RL, policy optimization, and evaluation inside learned dynamics | `WorldModelEnv`, `make_world_model_env`, `env_backend="world-model"` | Adapter-defined Gymnasium space | Adapter-defined Gymnasium space |
| [Wrappers](wrappers.md) | Shared preprocessing, action conversion, time limits, reward observations, and image transforms | `simulusion.envs.wrappers` | Backend-dependent | Backend-dependent |

## Shared conventions

Most Simulusion training code expects image observations as a dictionary entry named `image` with channel-first shape `(3, H, W)` and `uint8` values. Backend adapters that wrap vector-only environments synthesize an image representation so pixel-based agents can still run.

DIAMOND-style Atari support is documented on the Atari page as a preprocessing helper for Atari rollouts. It is not a separate environment backend.

Dreamer environment construction applies a standard wrapper stack after creating DMC, DMLab, Gym/Gymnasium, MuJoCo, Gymnasium Robotics, Procgen, BSuite, Minecraft, Crafter, Memory Maze, Atari 100k, Craftax, Brax, or Unity environments:

1. `ActionRepeat` repeats each selected action for `cfg.action_repeat` environment steps.
2. `NormalizeActions` exposes finite continuous action bounds as normalized `[-1, 1]` policy outputs.
3. `TimeLimit` truncates episodes after `cfg.time_limit // cfg.action_repeat` wrapper steps.

DMLab, BSuite, Minecraft, Crafter, Memory Maze, Atari 100k and Craftax are
benchmarks that fix their own action repeat and episode length. For them
`cfg.action_repeat` must be 1 and no `TimeLimit` is applied.

Use `simulusion envs list` to print the lightweight backend catalog used by the CLI.
