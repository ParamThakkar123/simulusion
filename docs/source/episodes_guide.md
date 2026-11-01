# Your Own Data: Episodes, LeRobot and Games

Every dataset-driven task reads one format: a folder of **episodes**. Video,
reinforcement-learning trajectories, robot demonstrations and video games
all fit it, so
one dataset can train Genie, Dreamer's or IRIS's world model, or World
Models' VAE and MDN-RNN.

```{contents} Contents
:depth: 2
```

## The episode

An {py:class}`~simulusion.data.Episode` with `T` frames has `T - 1`
transitions, the same convention as the
{py:class}`~simulusion.world_model.WorldModel` protocol: `actions[t]` is the
action taken at `frames[t]`, which led to `frames[t + 1]` and earned
`rewards[t]`; `terminated[t]` says `frames[t + 1]` ended the episode and
`truncated[t]` that it was cut off (a time limit). Only `frames` is required.

| `.npz` key | Contents | Required |
|---|---|---|
| `frames` | `(T, H, W, C)` uint8, the main camera | yes |
| `actions` | `(T-1,)` int (discrete) or `(T-1, A)` float | no |
| `rewards` | `(T-1,)` float | no |
| `terminated`, `truncated` | `(T-1,)` bool | no |
| `frames.<camera>` | `(T, H, W, C)` uint8, further cameras | no |
| `obs.<name>` | `(T, ...)` low-dimensional observations (joint state) | no |
| `language` | a task instruction | no |

```python
import numpy as np
from simulusion.data import Episode, EpisodeWriter

with EpisodeWriter("data/mine") as writer:  # numbers files, writes meta.json
    writer.write(
        Episode(
            frames=np.zeros((50, 64, 64, 3), np.uint8),
            actions=np.zeros((49, 7), np.float32),
            observations={"state": np.zeros((50, 9), np.float32)},
            language="pick up the cup",
        )
    )
```

A folder may also hold loose videos (`.mp4`, `.webm`, ...) and `.npy` frame
arrays, read as frame-only episodes, so a directory of clips is already a
dataset for Genie. `meta.json` records the action space, cameras, frame rate
and source; tasks read the action count from it.

## Training windows

{py:class}`~simulusion.data.SequenceDataset` cuts every `sequence_length`
window that fits inside an episode (never across two) and returns a dict of
tensors: `frames` `(L, C, H, W)` uint8, and when present `actions`,
`rewards`, `terminated`, `truncated` (`L - 1` each), extra cameras and
observations. Tasks configure it through their `data` section:

```yaml
data:
  path: data/pusht
  camera: null          # which view to train on, for multi-camera data
  sequence_length: null # the model's own context when null
  image_size: null      # the model's own input size when null
  batch_size: 16
  val_fraction: 0.05    # held-out episodes for evaluation
```

## LeRobot

Local [LeRobot](https://github.com/huggingface/lerobot) dataset folders (v2.x
and v3.0) are read directly, without the `lerobot` package
(`pip install "simulusion[lerobot]"` adds PyArrow and PyAV). Convert once:

```bash
simulusion datasets import lerobot datasets/pusht data/pusht
simulusion datasets import lerobot ./my_dataset data/mine --camera top --all-cameras
simulusion datasets info data/pusht
simulusion train --config dreamer-wm data.path=data/pusht
```

or train on it in place with `data.source=lerobot data.path=datasets/pusht`,
or read it from Python:

```python
from simulusion.data import LeRobotEpisodes

episodes = LeRobotEpisodes("datasets/pusht", camera="observation.image")
episodes[0].actions.shape  # (T - 1, 2)
```

The chosen camera becomes `frames` and the others `cameras`; `action`
becomes `actions` without its last row (it leads past the final frame);
`next.reward` and `next.done` become `rewards` and `terminated`; numeric
`observation.*` features become `observations` (`observation.state` is
`observations["state"]`); the episode's task is `language`. Cameras stored as
videos or as images in the Parquet tables both work.

## RLDS episodes

RLDS, the format of Open X-Embodiment and many other robot datasets, stores
an episode as a list of steps. {py:func}`~simulusion.data.episode_from_rlds`
converts one, given as plain Python and NumPy values (reading the dataset
files is left to whatever produced the steps), and
{py:class}`~simulusion.data.RLDSKeys` says where the main camera, extra
cameras, state, actions and instruction live in a step:

```python
from simulusion.data import EpisodeWriter, RLDSKeys, episode_from_rlds

keys = RLDSKeys(image="image", state=("state",), language="language_instruction")
with EpisodeWriter("data/robot") as writer:
    for episode in my_rlds_episodes:  # each {"steps": [...]}
        writer.write(episode_from_rlds(episode, keys))
```

Dictionary actions (the RT-1 format) are flattened in `RLDSKeys.action`
order. RLDS stores an action in every step, so the last one (past the final
observation) is dropped; `is_terminal` on the final step marks the last
transition terminated, and an episode that ends without it is truncated.

## Games

Video games are the classic world-model playground. Which route leads to an
episode folder depends on whether the game can run inside Python.

### Retro console games (stable-retro)

[stable-retro](https://github.com/Farama-Foundation/stable-retro) emulates
over a thousand NES, SNES, Genesis and other console games
(`pip install "simulusion[retro]"`; it has no Windows wheel, so use WSL
there). Only Airstriker ships with its ROM; import ROMs you own with
`python -m retro.import /path/to/roms`.

Record a game played by a policy (random buttons by default):

```bash
simulusion datasets games                                   # the presets
simulusion datasets record airstriker data/airstriker --episodes 5
simulusion datasets record street-fighter-2 data/sf2 --episodes 50 --max-steps 3000
simulusion train genie data.path=data/sf2
```

or replay `.bk2` movies, the button-by-button recordings of human play that
stable-retro and BizHawk write. Replays are exact, frame for frame, and human
play reaches far more of a game than random buttons:

```bash
simulusion datasets import retro-movie movies/ data/sf2_human --game street-fighter-2
```

```python
from simulusion.data import import_retro_movies, record_retro

record_retro("super-mario-bros", "data/smb", episodes=20, max_steps=2000)
import_retro_movies("movies/", "data/smb_human", game="super-mario-bros")
```

Actions are the console's buttons, one `0 / 1` column each, in the order
`meta.json` lists under `info.buttons` (for a two-player movie, `P1_*` then
`P2_*`). `record_retro(..., actions="discrete")` (`--discrete`) records an
index into the integration's valid button combinations instead. Rewards and
episode ends come from the game's integration, and `record_retro` marks an
episode cut off by `max_steps` as truncated. The recorded buttons are the ones
the console received, after the integration drops invalid presses (UP with
DOWN).

### Games without an emulator (Terraria)

PC games such as Terraria cannot run in a training loop, so their data is
captured footage (OBS, ShadowPlay, ...) plus, for action-conditioned models,
a log of the inputs held on each frame:

```text
captures/
    session_01.mp4
    session_01.inputs.csv     # optional: a header, then one row per video frame
    session_02.mp4
    session_02.inputs.csv
```

```text
frame,left,right,jump,mouse_x,mouse_y
0,0,1,0,812,440
1,0,1,1,812,441
```

A `frame` column, if present, is only checked to be increasing; the other
columns become `actions` (numbers: `0 / 1` for keys, positions for the
mouse). `.inputs.npy`, a `(frames, inputs)` array, works too. Videos are
decoded one frame at a time, so long 1080p captures fit in memory, and cut
into clips:

```bash
simulusion datasets import video captures/ data/terraria --game terraria --clip-length 500
simulusion datasets import video captures/ data/terraria --no-inputs   # frames only
```

Either every video has an input log or none does.

### Preprocessing and presets

A {py:class}`~simulusion.data.FramePipeline` describes how raw frames become
training frames: `crop` (pixels cut from the `(top, bottom, left, right)`
edges, for a score bar or letterboxing), `size` `(height, width)`,
`grayscale`, and `frame_skip`. Frame skip merges the transitions between kept
frames: rewards add up, an episode end in any of them ends the merged one, and
actions combine by `action_reduce`: `"first"` (exact for action-repeat
recordings), `"any"` (keeps a short button press; used for human replays and
captures) or `"mean"`. The pipeline is saved in `meta.json` under
`info.pipeline`.

| Preset | Game | Source | Default pipeline |
|---|---|---|---|
| `street-fighter-2` | Street Fighter II' Special Champion Edition (Genesis) | stable-retro | 64 x 64, every 4th frame |
| `super-mario-bros` | Super Mario Bros. (NES) | stable-retro | 64 x 64, every 4th frame |
| `super-mario-world` | Super Mario World (SNES) | stable-retro | 64 x 64, every 4th frame |
| `sonic` | Sonic The Hedgehog (Genesis) | stable-retro | 64 x 64, every 4th frame |
| `airstriker` | Airstriker (Genesis, ROM included) | stable-retro | 64 x 64, every 4th frame |
| `terraria` | Terraria | gameplay video | 72 x 128, every 4th frame, `"any"` |

Command-line options override a preset's pipeline (`--size 84x84 --crop
16,0,0,0 --grayscale --frame-skip 2`), and from Python every importer takes a
`pipeline=`. Any stable-retro integration works without a preset
(`simulusion datasets record SonicTheHedgehog2-Genesis data/sonic2`), and
{py:func}`~simulusion.data.register_game` adds presets:

```python
from simulusion.data import FramePipeline, GamePreset, register_game

register_game(
    GamePreset(
        name="streets-of-rage-2",
        title="Streets of Rage 2 (Genesis)",
        source="retro",
        retro_game="StreetsOfRage2-Genesis",
        pipeline=FramePipeline(crop=(16, 0, 0, 0), size=(64, 64), frame_skip=4),
    )
)
```
