# Your Own Data: Episodes, LeRobot and Open X-Embodiment

Every dataset-driven task reads one format: a folder of **episodes**. Video,
reinforcement-learning trajectories and robot demonstrations all fit it, so
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

[LeRobot](https://github.com/huggingface/lerobot) datasets (v2.x and v3.0)
are read directly, from a local folder or the Hugging Face Hub, without the
`lerobot` package (`pip install "simulusion[lerobot]"` adds PyArrow and
PyAV). Convert once:

```bash
simulusion datasets import lerobot lerobot/pusht data/pusht
simulusion datasets import lerobot ./my_dataset data/mine --camera top --all-cameras
simulusion datasets info data/pusht
simulusion train --config dreamer-wm data.path=data/pusht
```

or train on it in place with `data.source=lerobot data.path=lerobot/pusht`,
or read it from Python:

```python
from simulusion.data import LeRobotEpisodes

episodes = LeRobotEpisodes("lerobot/pusht", camera="observation.image")
episodes[0].actions.shape  # (T - 1, 2)
```

The chosen camera becomes `frames` and the others `cameras`; `action`
becomes `actions` without its last row (it leads past the final frame);
`next.reward` and `next.done` become `rewards` and `terminated`; numeric
`observation.*` features become `observations` (`observation.state` is
`observations["state"]`); the episode's task is `language`. Only the files
of the selected episodes are downloaded.

## Open X-Embodiment and RLDS

[Open X-Embodiment](https://robotics-transformer-x.github.io/) publishes its
datasets in RLDS. Reading them needs TensorFlow Datasets, which Simulusion
does not depend on (`pip install tensorflow tensorflow-datasets`):

```bash
simulusion datasets import oxe fractal20220817_data data/fractal --split 'train[:100]'
```

```python
from simulusion.data import load_oxe

for episode in load_oxe("bridge", split="train[:10]"):
    ...
```

Common datasets have presets ({py:data}`simulusion.data.rlds.OXE_PRESETS`)
that say where the main camera, state, actions and instruction live; for
others pass an {py:class}`~simulusion.data.RLDSKeys`. Dictionary actions (the
RT-1 format) are flattened in the preset's key order. RLDS stores an action in
every step, so the last one (past the final observation) is dropped;
`is_terminal` on the final step marks the last transition terminated, and an
episode that ends without it is truncated.
{py:func}`~simulusion.data.episode_from_rlds` converts any RLDS episode
given as plain Python and NumPy values, whatever produced it.
