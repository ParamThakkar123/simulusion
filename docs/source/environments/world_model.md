# World Model Env

## Train agents inside a Simulusion world model: `DreamEnv`

`DreamEnv` turns any world model that implements the
[`WorldModel` protocol](../world_model_protocol.md) (Dreamer, PlaNet, IRIS,
DIAMOND, Genie, the VAE + MDN-RNN World Models) into a Gymnasium environment.
`DreamVectorEnv` runs many of these environments together and steps them all
in one batched model call. Any library that trains on Gymnasium environments
then trains inside the model:

```python
from simulusion.envs import DreamEnv, DreamVectorEnv

env = DreamEnv(
    "runs/iris-wm_.../world_model",  # a saved world model, or a WorldModel
    starts="data/pong",              # real episodes that dreams start from
    max_episode_steps=200,
)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# 64 dreams per model call, for on-policy libraries such as CleanRL's PPO.
envs = DreamVectorEnv("runs/iris-wm_.../world_model", "data/pong", num_envs=64,
                      max_episode_steps=200)
```

- **Starts.** Every dream starts from real frames. Each reset picks a random
  clip from `starts` and encodes it into a model state. `starts` can be a
  folder in Simulusion's [episode format](../episodes_guide.md), an
  `EpisodeFolder`, or `Episode` objects. By default the clip fills the model's
  context window (`spec.max_context`, or `spec.context_frames` for recurrent
  models), and the actions taken between its frames come with it.
- **Observations** are the model's decoded frames: `uint8` arrays of shape
  `(H, W, C)` at the model's resolution, starting with the first frame. The
  agent only ever sees what the model imagines.
- **Actions.** Discrete models give a `Discrete(n)` action space and continuous
  ones a `Box` with the model's bounds. Genie's actions are latent: what each
  one does is learned by the model.
- **Rewards and episode ends** come from the model's heads. A model without a
  reward head, such as Genie, needs `reward_fn(frames, actions) -> rewards`,
  which takes the frames after the step as an array of shape `(B, H, W, C)`.
  `reward_fn` also replaces the head of a model that has one, so the same dream
  can be used to train a different task. Episodes end when the model predicts
  it, or are truncated after `max_episode_steps`. Set it for models without a
  termination head, which otherwise never end an episode.
- **Seeding.** `reset(seed=...)` makes a stochastic dream repeatable. Each
  environment keeps its own PyTorch random state and uses it only around its
  own model calls, so the training code's global random state is left alone.
- **Branching.** `DreamEnv.snapshot()` and `restore()` save a dream, including
  its random state, so it can be replayed exactly from that point.
- **Resets in a batch.** `DreamVectorEnv` resets dreams automatically in
  Gymnasium's default `NEXT_STEP` mode. On the step after a dream ends, it
  returns that dream's new first frame with reward 0 and ignores the action
  given for it. IRIS and Genie keep a window of recent frames whose length
  grows until it fills the context window. Because a dream starts from a full
  window, every dream's state has the same shape, which lets one dream reset
  while the others carry on. A shorter `context` breaks this, and so do starts
  without actions.

Imagined rollouts drift from the real environment as they get longer. Keep
episodes short (Dreamer and IRIS train on horizons of 10–20 steps), and
evaluate the trained agent in the real environment.

## Adapting other models: `WorldModelEnv`

`WorldModelEnv` wraps a trained or adapter-backed world model as a Gymnasium-compliant environment. It lets you run simulated rollouts with the same API shape expected by RL libraries such as Stable-Baselines3, TorchRL, and CleanRL:

- `reset(seed=..., options=...) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `observation_space`, `action_space`, `render()`, and `close()`

The wrapper is model-agnostic. You can either expose common model methods (`env_step`, `step`, `predict_step`, `predict`, `imagine_step`, `transition`, or `__call__`) or provide explicit adapter callables. For end-to-end examples with Stable-Baselines3, TorchRL, and CleanRL, see the [RL library integration tutorial](../tutorials/world_model_env_rl_libraries.md).

### Basic usage

```python
import gymnasium as gym
import numpy as np

from simulusion import WorldModelEnv


def transition(model, state, action):
    next_state = model.imagine_step(state, action)
    obs = model.decode(next_state)
    reward = model.reward(next_state)
    done = model.termination(next_state)
    return {
        "state": next_state,
        "observation": obs,
        "reward": reward,
        "terminated": done,
    }


env = WorldModelEnv(
    trained_model,
    observation_space=gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
    action_space=gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
    initial_state=initial_latent,
    transition_fn=transition,
    max_episode_steps=50,
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```


### Action adapters

Use `action_transform_fn` when the RL library should optimize a different action representation than the world model consumes. For example, expose a `Discrete` action space to an RL library while converting each action into a learned one-hot or latent action vector before the model step:

```python
def action_transform(model, action):
    one_hot = np.zeros(model.num_actions, dtype=np.float32)
    one_hot[int(action)] = 1.0
    return one_hot


env = WorldModelEnv(
    trained_model,
    observation_space=obs_space,
    action_space=gym.spaces.Discrete(trained_model.num_actions),
    transition_fn=transition,
    action_transform_fn=action_transform,
)
```

### Factory and public API

The direct factory is `make_world_model_env`:

```python
from simulusion import make_world_model_env

env = make_world_model_env(
    trained_model,
    observation_space=obs_space,
    action_space=act_space,
    transition_fn=transition,
)
```

The top-level factory also includes a `world-model` backend:

```python
import simulusion

env = simulusion.make_env(
    trained_model,
    backend="world-model",
    observation_space=obs_space,
    action_space=act_space,
    transition_fn=transition,
)
```

Aliases include `world_model`, `model`, and `wm`.

### Accepted adapter return forms

`reset_fn` may return:

| Return form | Description |
|---|---|
| `obs` | Observation only |
| `(obs, info)` | Observation with info dict |
| `(state, obs)` | State and observation |
| `(state, obs, info)` | State, observation, and info |
| Mapping with `state`, `observation`/`obs`/`image`, optional `info` | Explicit dict return |

`transition_fn` or model transition methods may return:

| Return form | Description |
|---|---|
| `(obs, reward, terminated, truncated, info)` | Full Gymnasium tuple |
| `(obs, reward, done, info)` | Gym-style tuple |
| `(state, obs, reward)` | Compact state-observation-reward |
| `(state, obs, reward, terminated, truncated, info)` | Full tuple with state |
| Mapping with `state`/`next_state`, `observation`/`obs`/`image`, `reward`, `terminated`/`done`, `truncated`, optional `info` | Explicit dict return |

If a transition omits a reward or termination flag, pass `reward_fn` and `terminal_fn`. Missing rewards default to `0.0`; missing termination defaults to `False`. `max_episode_steps` sets `truncated=True` when the simulated rollout reaches the time limit.
