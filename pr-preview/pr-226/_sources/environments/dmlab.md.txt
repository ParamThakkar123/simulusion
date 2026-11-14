# DeepMind Lab

Simulusion supports DeepMind Lab through `DMLabEnv`, a lightweight adapter around the
native `deepmind_lab` Python module. It converts Lab RGB observations into the
channel-first image dictionary used by Dreamer and other pixel-based world-model
code.

Install: DeepMind Lab has no wheels and builds from source with Bazel on Linux,
for Python 3.11 or older. The `docker/envs` image builds it (together with MineRL
and BSuite):

```bash
docker build -f docker/envs/Dockerfile -t simulusion-envs .
```

Without a GPU (a CPU-only container, for example) set `renderer="software"`
(`cfg.dmlab_renderer = "software"`); the default hardware renderer needs a GPU
OpenGL context.

On a Linux machine or in WSL, `sudo PYTHON=$(which python) bash docker/envs/install_dmlab.sh`
builds and installs it into that interpreter, including the object images the
`psychlab_*` levels need.

Dreamer uses `cfg.env_backend = "dmlab"` to select this backend. DMLab-specific fields (`dmlab_action_repeat`, `dmlab_observations`, `dmlab_action_set`, `dmlab_config`, `dmlab_renderer`, `dmlab_text`, `dmlab_mode`) are forwarded from `DreamerConfig` to `DMLabEnv`. See {doc}`../dreamer` for the full Dreamer config reference.

## Direct usage

```python
from simulusion import make_dmlab_env

env = make_dmlab_env("rooms_collect_good_objects_train", seed=0, size=(64, 64))
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

The default action space is a normalized one-hot `Box[-1, 1]` over a compact set
of navigation actions. `action_set=` (or `cfg.dmlab_action_set`) takes a custom
list of native seven-element Lab actions, or the name of a fixed set:

| Name | Actions | Used by |
| --- | --- | --- |
| `"default"` | 9, including a no-op | Simulusion's default |
| `"impala"` | 9 | IMPALA |
| `"popart"` | 15, with small and large turns and looking up/down | PopArt, R2D2, DreamerV3 on DMLab-30 |

The action space is marked `one_hot`, so agents that choose their policy from the
action space can use a categorical one.

## Language levels

The `language_*` levels of DMLab-30 give a text instruction. `DMLabEnv` requests
the `INSTR` observation for them and embeds it as in the DreamerV3 reference:
lowercase word and punctuation tokens are hashed into 64K buckets with fixed random
32-dimensional embeddings, padded to 32 tokens, and concatenated into a
`(1024,)` float32 vector under `obs["instr"]`. `text=True/False` (`cfg.dmlab_text`) overrides
the automatic choice.

## Train and test levels

`mode="train"` (`cfg.dmlab_mode`, the default in a Dreamer config) plays the
`*_train` version of `*_test` levels; the DreamerV3 reference trains and
evaluates in this mode. `mode="eval"` keeps the level and enables DMLab's held-out
level set (`allowHoldOutLevels`) with a fixed mixer seed, for DMLab-30 test scores.

## DMLab-30 in a Dreamer config

```python
from simulusion.configs.dreamer_config import DreamerConfig

cfg = DreamerConfig(
    env_backend="dmlab",
    env="explore_goal_locations_small",
    action_repeat=1,
)
```

This is the DreamerV3 protocol: the 15 PopArt actions, train mode, and each action
repeated for 4 Lab frames inside the environment (`dmlab_action_repeat`).
Dreamer's own `action_repeat` must be 1 (`make_env` raises otherwise), `time_limit`
does not apply since levels end their own episodes, and training budgets and logs
count Lab frames.

The 32 DMLab-30 level names (`DMLAB30_LEVELS`; two tasks have separate train and
test levels) are looked up under `contributed/dmlab30/`, so the bare name works;
other levels take their full path (`resolve_dmlab_level`).

## Backend options

| Option | Default | Description |
| --- | --- | --- |
| `cfg.dmlab_action_repeat` | `4` | Native Lab frame repeat passed to `env.step(..., num_steps=...)`. |
| `cfg.dmlab_action_set` | `"popart"` | `"popart"`, `"impala"`, `"default"`, or a custom 2D array of native Lab actions. |
| `cfg.dmlab_observations` | `None` | Additional native observation names. `RGB_INTERLEAVED` is always included. |
| `cfg.dmlab_config` | `None` | Extra Lab config values. Width and height are derived from `cfg.image_size`. |
| `cfg.dmlab_renderer` | `"hardware"` | Renderer argument forwarded to `deepmind_lab.Lab`. |
| `cfg.dmlab_text` | `None` | Embed the language instruction as `obs["instr"]`; `None` turns it on for `language_*` levels. |
| `cfg.dmlab_mode` | `"train"` | `"train"` maps `*_test` levels to `*_train`; `"eval"` enables held-out levels; `None` neither. |
