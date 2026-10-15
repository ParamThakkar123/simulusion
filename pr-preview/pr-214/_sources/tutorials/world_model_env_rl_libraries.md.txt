# Tutorial: Plug Simulusion world models into RL libraries

`WorldModelEnv` is a Gymnasium-compatible facade around a trained world model. The key idea is that RL libraries do **not** need to know whether a transition came from MuJoCo, Atari, Brax, or a learned dynamics model. They only need the Gymnasium contract:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

Because `WorldModelEnv` implements that contract, you can hand it to Stable-Baselines3, wrap it with TorchRL's `GymWrapper`, or use it inside a CleanRL-style training script. The world model itself does not need to implement a single Simulusion-specific interface: provide small adapter callables whenever the model uses custom method names, latent-state objects, reward heads, or action encodings.

A notebook version of this tutorial is also available (requires pandoc to build).

## 1. Build one adapter once

Start by writing the smallest glue layer between your trained model and the Gymnasium API. The example below uses placeholder model methods; replace them with your checkpoint loading, encoder/decoder, dynamics, reward, and terminal heads.

```python
import gymnasium as gym
import numpy as np

from simulusion import WorldModelEnv


trained_model = (
    ...
)  # Load a Simulusion model, an exported module, or your own adapter object.
initial_latent = ...  # Optional: pass None if reset_fn creates the initial state.

obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(64,), dtype=np.float32)
action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)


def reset_world_model(model, seed=None, options=None):
    latent = model.sample_initial_state(seed=seed)
    obs = model.decode_observation(latent)
    return {"state": latent, "observation": obs, "info": {"source": "world_model"}}


def transition_world_model(model, state, action):
    next_state = model.predict_next_state(state, action)
    obs = model.decode_observation(next_state)
    reward = model.predict_reward(next_state, action)
    done = model.predict_terminal(next_state)
    return {
        "state": next_state,
        "observation": obs,
        "reward": reward,
        "terminated": done,
    }


env = WorldModelEnv(
    trained_model,
    observation_space=obs_space,
    action_space=action_space,
    initial_state=initial_latent,
    reset_fn=reset_world_model,
    transition_fn=transition_world_model,
    max_episode_steps=50,
)
```

### Adapter return forms

You can return dictionaries, Gymnasium tuples, or compact `(state, obs, reward)` tuples. Dictionary returns are recommended for production because they make state, observation, reward, termination, truncation, and diagnostic info explicit.

If your library action format differs from your model action format, add `action_transform_fn`:

```python
def action_transform(model, action):
    # Example: convert a discrete library action into a learned one-hot action vector.
    one_hot = np.zeros(model.num_actions, dtype=np.float32)
    one_hot[int(action)] = 1.0
    return one_hot


env = WorldModelEnv(
    trained_model,
    observation_space=obs_space,
    action_space=gym.spaces.Discrete(trained_model.num_actions),
    reset_fn=reset_world_model,
    transition_fn=transition_world_model,
    action_transform_fn=action_transform,
)
```

## 2. Stable-Baselines3

Stable-Baselines3 custom environments must follow the Gymnasium interface. Its docs recommend running `check_env` for custom environments, and image observations should be `np.uint8` within a `[0, 255]` `Box` when using CNN policies.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

check_env(env, warn=True)

sb3_env = Monitor(env)
model = PPO("MlpPolicy", sb3_env, verbose=1)
model.learn(total_timesteps=10_000)
```

For dictionary observations, use `MultiInputPolicy`:

```python
model = PPO("MultiInputPolicy", sb3_env, verbose=1)
```

For image observations, prefer channel-first `uint8` images and choose `CnnPolicy` or `MultiInputPolicy` depending on whether the observation is a plain image or a dictionary containing an image.

## 3. TorchRL

TorchRL can wrap an already-created Gymnasium environment with `GymWrapper`. This is useful for `WorldModelEnv` because your world model is usually a Python object or checkpoint rather than a registered Gymnasium id.

```python
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter

base_env = GymWrapper(env)
torchrl_env = TransformedEnv(
    base_env,
    Compose(
        DoubleToFloat(),
        StepCounter(max_steps=50),
    ),
)

td = torchrl_env.reset()
td = torchrl_env.rand_step(td)
```

From here you can attach TorchRL collectors, modules, and losses exactly as you would for a physical simulator. The wrapper boundary is still Gymnasium, so the model adapter remains reusable outside TorchRL.

## 4. CleanRL

CleanRL examples are intentionally single-file. The common pattern is to define a `make_env` closure that returns a Gymnasium environment, then pass a list of closures to `gym.vector.SyncVectorEnv` or `AsyncVectorEnv`. `WorldModelEnv` fits that pattern directly.

```python
import gymnasium as gym


def make_env(seed: int):
    def thunk():
        env = WorldModelEnv(
            trained_model,
            observation_space=obs_space,
            action_space=action_space,
            reset_fn=reset_world_model,
            transition_fn=transition_world_model,
            max_episode_steps=50,
            seed=seed,
        )
        return env

    return thunk


envs = gym.vector.SyncVectorEnv([make_env(seed=i) for i in range(4)])
obs, infos = envs.reset(seed=0)
```

If every vector worker should own an independent model copy, load or clone the model inside `thunk()` instead of closing over a shared `trained_model` object.

## 5. Practical checklist

- **Observation spaces:** match the decoded observation exactly. Use `Box(..., dtype=np.float32)` for vectors and `Box(0, 255, dtype=np.uint8)` for images.
- **Action spaces:** expose the action format the RL library should optimize. Use `action_transform_fn` to convert library actions into model-specific actions.
- **Episode length:** set `max_episode_steps` to the imagination horizon you trust. This marks time-limit endings as `truncated=True`.
- **State diagnostics:** `info["model_state"]` contains the latest latent/model state; keep it for debugging but do not feed it to policies unless you intentionally include it in observations.
- **Vectorization:** create one environment instance per worker. Avoid sharing mutable model state across parallel workers unless the model is explicitly thread/process safe.
- **Reality checks:** validate imagined policies in the real environment periodically. Learned rollouts can exploit model errors.

## External references

- Stable-Baselines3 custom environment guide: <https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html>
- Stable-Baselines3 PPO example: <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>
- TorchRL `GymWrapper` reference: <https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.GymWrapper.html>
- TorchRL environment tutorial: <https://docs.pytorch.org/rl/main/tutorials/torchrl_envs.html>
- CleanRL PPO docs: <https://docs.cleanrl.dev/rl-algorithms/ppo/>
- Gymnasium basic usage: <https://gymnasium.farama.org/content/basic_usage/>
