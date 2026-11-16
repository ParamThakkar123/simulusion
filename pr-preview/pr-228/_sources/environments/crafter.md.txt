# Crafter

[Crafter](https://github.com/danijar/crafter) (Hafner, 2021) is an open-world
survival game on a 64x64 grid. The agent collects resources, crafts tools,
fights monsters and keeps itself fed, watered and rested. It is a headline
benchmark of DreamerV3, trained for 1M environment steps.

Install: `pip install "simulusion[crafter]"` (or `pip install crafter`). Crafter
is pure Python and runs on the CPU on Linux, macOS and Windows.

## Main API

```python
from simulusion import CrafterEnv, crafter_score

env = CrafterEnv("reward", seed=0)          # or make_crafter_env("reward")
obs = env.reset()
print(obs["image"].shape)                   # (3, 64, 64)
print(env.action_space.n)                   # 17

finished = []
obs, reward, done, info = env.step(env.action_space.sample())
if done:
    finished.append(info["achievements"])   # one dict per episode
```

Two tasks are available. `"reward"` gives Crafter's reward signal (+1 for each
achievement unlocked for the first time in an episode, plus a small health
term). `"noreward"` returns zero reward, for unsupervised and exploration
studies. DreamerV3's `crafter_reward`, the Gym id `CrafterReward-v1`, and the
`noreward` spellings are accepted too.

| Argument | Default | Meaning |
| --- | --- | --- |
| `seed` | `0` | World-generation seed. Each reset draws a new world. |
| `size` | `(64, 64)` | Rendered image size `(H, W)`. |
| `area` | `(64, 64)` | World size in cells. |
| `view` | `(9, 9)` | Agent view in cells, including the inventory rows. |
| `length` | `10000` | Episode step limit; `None` disables it. |

## Scoring

The Crafter benchmark is not scored by return. Each of the 22 achievements gets
a success rate: the percentage of episodes in which it was unlocked at least
once. The score is their geometric mean:

```
score = exp(mean_i ln(1 + s_i)) - 1,   s_i in [0, 100]
```

`crafter_score(episodes)` computes it from the `info["achievements"]` dicts of
finished episodes. `crafter_success_rates` returns the per-achievement rates,
and `crafter_score_from_rates` scores precomputed rates.

## Episodes and the info contract

Episodes end when the player dies or after `length` steps. Death has discount 0
and is reported as `terminated`. The time limit keeps discount 1 and is reported
as `truncated`.

Crafter's own info keys pass through next to the [contract keys](info_contract.md):
`achievements` (unlock counts so far in the episode), `inventory`, `player_pos`
and `semantic` (the semantic map).

## Training with Dreamer

```python
from simulusion import DreamerConfig

cfg = DreamerConfig(env_backend="crafter", env="reward", action_repeat=1)
```

Crafter fixes its own episode length, so Dreamer applies no `time_limit` on
top, and `action_repeat` must be 1, as in the DreamerV3 reference.
