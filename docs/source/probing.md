# Probing latents for the true state

A world model trained on pixels has to build some internal description of the
scene. Whether that description recovers the variables that actually generated
the scene, the positions, angles and velocities a physics engine integrates,
is an open question
([unsupervised discovery of true generative factors](https://www.emergentmind.com/open-problems/unsupervised-discovery-of-true-generative-factors-in-environments)).
DeepMind Control answers half of it for free: the simulator's state is known
exactly, so a latent can be checked against it.

`simulusion.evals.probing` builds that check. This page covers the labelled
datasets, reading a model's latent for every frame, and scoring how well each
latent holds the true state.

## Probe datasets

```python
from simulusion.evals import collect_probe_episodes, ProbeDataset

data = collect_probe_episodes(
    "cartpole-swingup",
    episodes=50,
    seed=0,
    size=(64, 64),       # match the world model's input
    action_repeat=2,     # match its training setup (Dreamer uses 2 on DMC)
)
train, val = data.split(0.2)
data.save("data/probe/cartpole-swingup")
data = ProbeDataset.load("data/probe/cartpole-swingup")
```

Every recorded frame keeps MuJoCo's `qpos`, `qvel` and `act` (see
{doc}`environments/dmc`) and the task's own observations as `task.<name>`.
On disk a probe dataset is an ordinary episode folder ({doc}`episodes_guide`)
whose `meta.json` also records the task, camera, action repeat and the
`PhysicsStateSpec`.

`split` divides by whole episodes. Neighbouring frames are near duplicates,
so splitting by frame would put almost the same image on both sides and
overstate every probe.

### Policies

`policy="random"` samples uniform actions. In some tasks that covers little of
the state space: a random swing-up rarely brings the pole above the cart. Pass
a callable from the observation dict to an action, such as a trained agent, to
record the states that matter, and probe on both.

## Regression targets

```python
targets = train.targets()                       # qpos and qvel
targets.values      # (N, K) float32, one row per frame, episodes in order
targets.names       # ("qpos/slider", "qpos/hinge_1/sin", "qpos/hinge_1/cos", ...)
targets.episode     # (N,) episode index of each row
targets.step        # (N,) frame index of each row within its episode
```

The raw state is not a good regression target as it stands, so `targets()`
transforms it:

| Raw dimension | Target |
| --- | --- |
| Hinge angle (`qpos`, kind `hinge`) | `sin` and `cos` columns. The raw angle is unbounded, and a pendulum that spins sits at many values for one pose. Turn this off with `wrap_angles=False`. |
| Quaternion (`free_quat`, `ball_quat`) | Sign fixed so `qw >= 0`, because `q` and `-q` are the same rotation. |
| Root position under a tracking camera | Left out (see below). |
| Everything else | Unchanged. |

`include` picks the groups and their order: `"qpos"`, `"qvel"`, `"act"`, and
`"task"` for the task's own observations, which are a useful comparison
because they are already engineered features.

### What the camera cannot show

The default cameras of `walker`, `cheetah`, `hopper`, `swimmer`, `humanoid`,
`quadruped` and `fish` follow the body's centre of mass. The distance travelled
then shows only as the phase of a repeating floor pattern, so no encoder can
be expected to recover it. `unobservable="auto"` (the default) leaves these
dimensions out when the dataset was rendered from a tracking camera:

```python
from simulusion.evals import unobservable_dims

unobservable_dims("walker", camera=0)     # ("rootx",)
unobservable_dims("quadruped", camera=3)  # () - the egocentric camera is fixed
```

Pass a list of `qpos` names to choose for yourself, or `None` to keep every
dimension. The velocities stay either way: across frames, motion against the
floor shows them.

## Latents for every frame

```python
from simulusion.evals import extract_latents, pixel_features, reinitialized
from simulusion.world_model import load_world_model

wm = load_world_model("runs/dreamer-cartpole")
latents = extract_latents(wm, val)
latents.features["deter"]          # (N, D) float32, one row per frame
latents.features["features"]       # deter and stoch together, as Dreamer's heads read them
targets = latents.match(val.targets())   # the target rows for these frames, same order
```

`extract_latents` runs the model's `encode_steps` over each episode (see
{doc}`world_model_protocol`), so the latent at frame `t` has seen frames
`0..t` and the actions between them. Every part of the state becomes its own
feature:

| Family | Features |
| --- | --- |
| Dreamer | `deter`, `stoch`, `mean`, `std`, and `features` (`deter` + `stoch`) |
| PlaNet | `deter`, `stoch`, `features` |
| World Models | `z`, `h`, `c`, `features` (`z` + `h`, what the controller reads) |
| IRIS, Genie | `tokens`: the latest frame's tokens looked up in the codebook, every position concatenated; `tokens_pooled`: their mean |
| DIAMOND | `pixels`: the latest frame, since its state is the frame window itself; `reward_h`, `reward_c`: the reward model's LSTM |

Inputs echoed in a state (`actions`, `prev_actions`) are dropped. Pass `keys`
to keep only some features, or `features=` your own function of
`(model, state)` for a model of your own.

**History.** `context=None` lets recurrent models filter the whole episode and
windowed models use their full window. `context=1` asks what one frame holds.
A single frame has no velocity in it, so the gap between the two on `qvel`
targets shows how much a model's memory carries.

**Actions.** The recorded DMC actions are continuous. IRIS, Genie and DIAMOND
take discrete actions. Give `action_fn` to map an episode's `(T - 1, A)`
actions to the model's, or they encode without actions (with a warning). Pass
`actions=False` to leave them out for any model.

**Controls.** Read every probe against two baselines that go through the same
pipeline:

- `pixel_features(data, size=16)`: the frames themselves, downsampled. What a
  probe can decode with no model at all.
- `reinitialized(model)`: the same architecture with fresh weights. Random
  encoders keep much of what is in the image, so training only gets credit
  for what it adds over this.

## Probes and scores

Split by episode three ways. Probes fit on `train`, pick their penalty and stop
early on `val`, and are scored on `test`, which nothing was tuned on:

```python
from simulusion.evals import score_latents

train, rest = data.split(0.4)
val, test = rest.split(0.5)

def pair(split, key="deter"):
    latents = extract_latents(wm, split, keys=[key])
    return latents.features[key], latents.match(split.targets()).values

scores = score_latents(pair(train), pair(val), pair(test), train.targets().names)
scores.linear_r2          # (K,) held-out R² per target column
scores.mlp_r2             # the same with a small MLP
scores.dci.disentanglement, scores.dci.completeness
scores.mig.mig
```

| Score | Question it answers |
| --- | --- |
| Linear R² (`fit_linear_probe`) | Is the variable in the latent, linearly? Ridge regression, solved for every penalty from one SVD, penalty chosen per target on `val`. |
| MLP R² (`fit_mlp_probe`) | Is it there at all? A two-hidden-layer MLP. A large gap over linear means the variable is present but entangled. |
| DCI disentanglement (`dci`) | Does each latent dimension serve one variable? |
| DCI completeness | Does each variable live in one latent dimension? |
| DCI informativeness | Mean held-out linear R². |
| MIG (`mig`) | For each variable, how much more the best latent dimension knows about it than the second best, as a share of its entropy. |

Being decodable is not being discovered. A latent can be a perfect rotation of
the true state, with R² of 1 and low disentanglement. DCI and MIG measure
whether the model found the variables themselves rather than some mixture of
them.

DCI here uses the absolute weights of the linear probe on standardised data as
importances. The original paper uses gradient-boosted trees; the linear
version is a common variant and needs no extra dependency. Columns that
describe one variable count as one factor: the `sin` and `cos` of an angle,
and a quaternion's four parts (`factor_groups`). Pass `group=False` to score
every column separately.

MIG builds a histogram for every latent dimension and target pair, so
`score_latents` skips it for latents wider than `max_mig_dims` (512). That
covers IRIS and Genie's concatenated tokens and DIAMOND's pixels.
