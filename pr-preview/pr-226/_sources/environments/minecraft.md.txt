# Minecraft Diamond

`MinecraftDiamondEnv` is the Minecraft Diamond environment of DreamerV3
(Hafner et al., 2023), the benchmark on which DreamerV3 became the first agent to
collect diamonds from scratch, without human data or curricula. It reproduces the
reference implementation (`embodied/envs/minecraft_flat.py`), built on MineRL 0.4.4
and Minecraft 1.11.2.

For MineRL's own competition tasks (Treechop, Navigate, ObtainDiamond) with a
generic discrete action set, use `make_minecraft_env` from
`simulusion.envs.minecraft_env` instead.

## Install

MineRL 0.4.4 runs a real Minecraft client. It needs Linux, Python 3.11, a JDK 8
and a display (or `xvfb`), and the patched wheel that adds the block break speed
setting. The `docker/envs` image installs all of it, with DeepMind Lab and BSuite:

```bash
docker build -f docker/envs/Dockerfile -t simulusion-envs .
docker run --gpus all --rm -v "$PWD/runs:/data/simulusion" simulusion-envs \
  simulusion train --config dreamer model.env_backend=minecraft model.env=diamond \
    model.action_repeat=1
```

Outside Docker, on Linux with Python 3.11:

```bash
sudo apt-get install openjdk-8-jdk xvfb
pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
xvfb-run -a python your_script.py
```

## The task

| | |
| --- | --- |
| Goal | Obtain a diamond, through 12 milestone items |
| Rewards | +1 the first time per episode each of `log`, `planks`, `stick`, `crafting_table`, `wooden_pickaxe`, `cobblestone`, `stone_pickaxe`, `iron_ore`, `furnace`, `iron_ingot`, `iron_pickaxe`, `diamond` is held; plus 0.01 x the change in health (health scaled to [0, 1]) |
| Episode end | Death, or 36000 steps (30 minutes at 20 Hz). Breaking diamond ore does not end the episode. If Minecraft's Malmo backend fails, the episode is cut short there (truncated, without reward) and the next reset restarts the game, as in the reference. |
| Block breaking | 100x faster (`break_speed=100`), so the attack key need not be held |
| Actions | 25: `noop`, `attack`, `turn_up/down/left/right` (15 degrees), `forward`, `back`, `left`, `right`, `jump` (with forward, held for 10 steps), `place_dirt`, and 13 abstract crafting, placing, equipping and smelting actions |
| Camera | Pitch kept within +-60 degrees |

`task="wood"` is the reference's quick sanity task: the 12 basic actions and +1 per
log collected.

The paper describes the health reward as +-0.01 per heart. The reference code, which
produced the reported results, scales the health change by health in [0, 1], so a
heart is worth 0.001; this environment follows the code.

## Observations and actions

```python
from simulusion import make_minecraft_diamond_env

env = make_minecraft_diamond_env("diamond", size=(64, 64))
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
print(env.action_names[:4], info["achievements"]["log"])
```

| Key | Shape | Content |
| --- | --- | --- |
| `image` | `(3, 64, 64)` uint8 | First-person view |
| `inventory` | `(num_items,)` | Count of every item |
| `inventory_max` | `(num_items,)` | Largest count this episode, which tells the agent which milestones it reached |
| `equipped` | `(num_types,)` | One-hot of the held item |
| `health`, `hunger`, `breath` | `(1,)` | Scaled to [0, 1] |

Actions are one-hot vectors over `env.action_names` (an integer index works too).
The action space is marked `one_hot`, so agents that choose their policy from the
action space can use a categorical one. `info["achievements"]` reports which
milestone items the episode has reached.

## Dreamer config

```python
from simulusion.configs.dreamer_config import DreamerConfig

cfg = DreamerConfig(
    env_backend="minecraft",
    env="diamond",
    action_repeat=1,
)
```

The agent acts on every Minecraft step, as in the reference, so `action_repeat`
must be 1; `make_env` raises otherwise. Episodes last `minecraft_length` steps and
`time_limit` does not apply.

| Option | Default | Description |
| --- | --- | --- |
| `cfg.env` | `"diamond"` | `"diamond"` or `"wood"` |
| `cfg.minecraft_break_speed` | `100.0` | Block breaking speed multiplier |
| `cfg.minecraft_length` | `36000` | Episode length in steps |
