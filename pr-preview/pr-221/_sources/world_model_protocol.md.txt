# One interface for every world model

Each world model family in Simulusion predicts the future its own way: Dreamer
and PlaNet roll a recurrent latent forward, IRIS and Genie generate tokens with
a Transformer, DIAMOND samples pixels with a diffusion model, and World Models
pairs a VAE with an MDN-RNN. The `WorldModel` protocol puts one interface over
all of them, so code that evaluates, plans with, or plays a world model works
with any family.

```python
import torch
from simulusion.world_model import as_world_model, load_world_model

model = as_world_model(agent)  # a model you trained in this process
# or: model = load_world_model("runs/dreamer/ckpts/61000_ckpt.pt", "dreamer")

state = model.encode(frames, actions)  # real context -> model state
out = model.step(state, action)  # one imagined step
out.reward, out.terminated  # from the model's own heads
image = model.decode(out.state)  # (B, C, H, W) in [0, 1]

torch.manual_seed(0)
traj = model.rollout(state, planned_actions)  # many steps
traj.frames, traj.rewards, traj.terminated  # (B, H, ...)
```

## The methods

| Method | Does |
|---|---|
| `encode(frames, actions=None)` | Turns real context frames, and the actions taken between them, into a model state. |
| `step(state, action)` | Imagines one step. Returns a `Transition`: the next state, plus `reward` and `terminated` where the model has those heads. |
| `decode(state)` | The current frame of a state, float `(B, C, H, W)` in `[0, 1]`. |
| `rollout(state, actions, decode=True)` | `H` steps at once; returns a `Rollout` of frames, rewards and terminations, each `(B, H, ...)`. |
| `save(path)` | Writes the model to a directory that `load_world_model` reads back. |
| `spec` | A `WorldModelSpec`: what the model reads and predicts (below). |

The built-in models also have `encode_steps(frames, actions=None, context=None)`,
which returns the state after every frame: `states[t]` equals
`encode(frames[:, :t + 1], actions[:, :t])`. Dreamer, PlaNet and World Models
filter the sequence in one pass. The others encode one window per frame,
batched. Pass `context` to limit how many frames of history each state sees;
`context=1` gives what one frame alone holds. The first `context_frames - 1`
entries are `None`, because the model can't encode that little context. It is
not part of the `WorldModel` protocol, so your own models don't need it.
{doc}`probing` uses it to read a latent for every frame.

## Conventions

The same for every family, so you never need to know a model's internals:

- **Frames** are RGB, channel-first and batch-first: `(B, T, C, H, W)`, either
  float in `[0, 1]` or `uint8` in `[0, 255]`. A 4-D input is one sequence and
  a 3-D input one frame. Channel-last input and other sizes are converted to
  the model's own (`spec.obs_shape`).
- **Actions between context frames:** `actions[:, t]` is the action taken at
  frame `t`, which led to frame `t + 1`. `T` frames come with `T - 1`
  actions. Discrete actions are integers `(B, ...)`, continuous ones float
  `(B, ..., A)`. A single action passed to `step` is used for the whole batch.
- **State** is a `dict[str, Tensor]` with the batch on dim 0. It is plain
  data: keep a copy to branch a dream from the same point.
- **Randomness** comes from PyTorch's global RNG. `torch.manual_seed` before
  `encode` and `rollout` makes a dream repeatable.
- **Inference only:** methods run without autograd and put the model in eval
  mode. Training stays with each family's trainer.

## What each family provides

`model.spec` says what a model can do, so generic code can check instead of
special-casing families:

| Family | Actions | Context | Reward | Termination | Notes |
|---|---|---|---|---|---|
| `dreamer` (V1, V2) | continuous in `[-1, 1]` | 1+ frames, recurrent | yes | if trained with the discount head | |
| `planet` | continuous, unbounded | 1+ frames, recurrent | yes | no | frames are quantised to the trained bit depth |
| `iris` | discrete | up to `transformer_timesteps` frames | yes, on the trained (sign) scale | yes | `sample=False` makes steps deterministic |
| `diamond` | discrete | exactly `num_conditioning_frames` | yes, `{-1, 0, 1}` | yes | `sampling_steps=` trades quality for speed |
| `genie` | discrete, **latent** (`spec.action.latent`) | up to `num_frames` | no | no | without `actions`, Genie labels the context itself |
| `world-models` | continuous, unbounded | 1+ frames, recurrent | yes | yes | `temperature=` as in the paper |

JEPA and DiT have no adapter: I-JEPA learns image representations without
actions and DiT generates images from a class label, so neither predicts what
an action does. `as_world_model` says so if you pass one.

## Loading checkpoints

`load_world_model(path)` reads a directory written by `save`, which records its
family in `world_model.json`. It also reads each family's own training
checkpoints when you name the family:

```python
load_world_model("runs/dreamer/ckpts/61000_ckpt.pt", "dreamer")  # V1 or V2
load_world_model("runs/planet/rssm.pth", "planet")  # bare RSSM weights
load_world_model("runs/iris/iris.pt", "iris")
load_world_model("runs/diamond/checkpoint_10.pt", "diamond")  # no emulator needed
load_world_model("runs/genie/genie_sonic_final.pt", "genie")
```

World Models trains its two halves separately, so it loads from both files:

```python
from simulusion.world_model import MDRNNWorldModel

model = MDRNNWorldModel.from_checkpoints("vae/best.tar", "mdrnn/best.tar")
```

Constructor options such as a sampling temperature are saved with the model and
can be overridden on load: `load_world_model(path, temperature=0.5)`.

## Models and agents

The protocol covers the model: what happens to the world under an action. The
policy that chooses actions stays with each family's agent, and
{doc}`steppers <inference_guide>` remain the interface for acting in a real
environment. A DIAMOND world model, for example, loads without the Atari
emulator or the replay buffer its agent needs for training.

## Your own model

Implement the methods above (and `spec` and `device`) and any code written
against `WorldModel` accepts it; `isinstance(obj, WorldModel)` checks the
shape at runtime. Subclassing `simulusion.world_model.WorldModelBase` gives
you input conversion, `rollout` and the save format for free: implement
`_encode`, `_step`, `_decode`, `_save_weights` and `_load_weights`.
