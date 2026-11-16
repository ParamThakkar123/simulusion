# Atari 100k

Atari 100k (Kaiser et al., 2020) trains an agent on 26 Atari games for 100k
agent steps, 400k frames at a frame skip of 4. Papers that report on it share
the game list and the frame skip, but not the rest of the preprocessing, and
those differences move scores. `Atari100kEnv` drives ALE directly and offers
each paper's settings as a named protocol, taken from the reference code.

Install: `pip install "simulusion[atari100k]"` (ale-py, which bundles the ROMs,
and OpenCV for the `area` resize).

## Main API

```python
from simulusion import Atari100kEnv, ATARI_100K_PROTOCOLS

env = Atari100kEnv("MsPacman", protocol="iris", seed=0)
obs = env.reset()
print(obs["image"].shape)          # (3, 64, 64)
print(env.action_meanings[:3])     # ['NOOP', 'UP', 'RIGHT']
```

Game names are accepted in any common spelling: `"MsPacman"`,
`"ALE/MsPacman-v5"`, `"MsPacmanNoFrameskip-v4"`, `"ms_pacman"` or DreamerV3's
`"atari100k_ms_pacman"`. `list_atari100k_games()` returns the 26 games.

## Protocols

| Setting | `iris` | `diamond` | `storm` | `dreamerv3` (= `atari100k`) |
| --- | --- | --- | --- | --- |
| Sticky actions | 0 | 0 | 0.25 | 0 |
| No-op starts | 1–30 | 1–30 | none | 0–30 |
| Life loss | ends the episode | ends the episode | ends the episode | ignored |
| Frame limit | 20000 | 108000 | 108000 | 108000 |
| Resize | Pillow bilinear, before pooling | OpenCV area | OpenCV area | Pillow bilinear |

All four use ALE's minimal action set, repeat each action for 4 frames,
max-pool the last two frames, keep RGB at 64x64, and leave rewards unclipped in
the environment.

Sources:

- **IRIS**: `eloialonso/iris`, `config/env/default.yaml` and
  `src/envs/wrappers.py`. Training uses a 20000-frame limit; evaluation
  (`iris-eval`) uses 108000 frames, one no-op and no life-loss episode ends.
- **DIAMOND**: `eloialonso/diamond`, `src/envs/env.py` and
  `atari_preprocessing.py`. `diamond-eval` turns off life-loss episode ends.
- **STORM**: `weipu-zhang/STORM`, `train.py`. STORM builds `ALE/<game>-v5`
  with `frameskip=1` and never overrides v5's default sticky-action
  probability of 0.25, so its numbers include sticky actions.
- **DreamerV3**: `danijar/dreamerv3`, `configs.yaml` (`env.atari100k`) and
  `embodied/envs/atari.py`.

Override single settings with keyword arguments, for example
`Atari100kEnv("Pong", protocol="iris", max_frames=108000)`, or pass your own
`AtariProtocol`.

## Life loss

With `life_loss="terminate"`, losing a life returns `done=True`,
`terminated=True` and `info["life_lost"] = True`, but the next `reset()`
continues the same game; only a game over or the frame limit restarts the
emulator. This is the `EpisodicLifeEnv` behaviour these codebases use. IRIS
also takes one no-op step before continuing (`continue_noop`). The info keys
`game_over`, `lives` and `episode_frame_number` (frames since the last real
reset) tell the cases apart.

The frame limit is reported as `truncated`. DreamerV3's reference marks it
terminal; Simulusion reports it as a truncation, like every other backend.

## Training with Dreamer

```python
from simulusion import DreamerConfig

cfg = DreamerConfig(
    env_backend="atari100k", env="Pong", action_repeat=1, atari100k_protocol="iris"
)
```

The frame skip happens inside the environment, so `action_repeat` must be 1.
The evaluation environment uses the protocol's `-eval` variant when one exists.

The general-purpose `make_atari_env` (Gymnasium's ALE ids) and the DIAMOND
helper are documented on the [Atari](atari.md) page.
