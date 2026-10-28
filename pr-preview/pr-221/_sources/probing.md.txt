# Probing latents for the true state

A world model trained on pixels has to build some internal description of the
scene. Whether that description recovers the variables that actually generated
the scene, the positions, angles and velocities a physics engine integrates,
is an open question
([unsupervised discovery of true generative factors](https://www.emergentmind.com/open-problems/unsupervised-discovery-of-true-generative-factors-in-environments)).
DeepMind Control answers half of it for free: the simulator's state is known
exactly, so a latent can be checked against it.

`simulusion.evals.probing` builds that check. This page covers the labelled
datasets. Extracting latents and fitting probes come next.

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
