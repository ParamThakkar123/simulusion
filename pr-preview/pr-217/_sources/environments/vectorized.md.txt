# Vectorized Environments

Simulusion provides two vectorization paths: ALE's native Atari vector environment and Simulusion's multiprocessing vector wrapper for arbitrary Gym-like environment factories.

## Native ALE vectorization

Use `make_atari_vector_env()` for high-throughput Atari simulation backed by `ale_py.vector_env.AtariVectorEnv`:

```python
from simulusion import make_atari_vector_env

vec_env = make_atari_vector_env(
    game="pong",
    num_envs=16,
    obs_type="rgb",
    frameskip=4,
    repeat_action_probability=0.25,
    full_action_space=False,
    seed=0,
)
```

This path is specific to Atari and returns ALE's vector environment object directly.

## TorchVectorizedEnv

`TorchVectorizedEnv` runs multiple environment instances across worker processes. It is useful for rollout collection in RL harnesses and algorithms that expect batched observations, rewards, done flags, and info dictionaries.

```python
from simulusion import make_env

# For arbitrary Gym-like factories, create each environment through simulusion.
# Pass this factory to your vectorization utility of choice.


def env_factory():
    return make_env("CartPole-v1", backend="gym", size=(64, 64))


# Example: hand `env_factory` to your multiprocessing/vector rollout code.
env = env_factory()
obs = env.reset()
```

The total number of environments is `num_workers * envs_per_worker`.

## Batched stepping

Actions passed to Simulusion vector environments should have a leading batch dimension equal to `total_envs`. The wrapper distributes each action to the corresponding worker environment and returns batched results.

```python
actions = vec_env.action_space.sample()  # example only; shape depends on environment
obs, rewards, dones, infos = vec_env.step(actions)
```

## Choosing a vectorization strategy

| Strategy | Choose when |
| --- | --- |
| `make_atari_vector_env()` | You are running Atari and want ALE's native vectorized implementation |
| `TorchVectorizedEnv` | You have a Gym-like factory and want multiprocessing across arbitrary environments |
| Single environment + wrappers | You are debugging or collecting small rollouts |

## Troubleshooting

- **Factory cannot be pickled**: define `env_factory` at module scope instead of inside another function when using multiprocessing-heavy workflows.
- **Batch shape errors**: ensure the first action dimension matches `vec_env.total_envs`.
- **Worker startup failures**: test a single environment from the same factory before vectorizing.
- **Native Atari naming**: ALE vectorization expects game names such as `pong`; Gymnasium factories often expect IDs such as `ALE/Pong-v5`.
