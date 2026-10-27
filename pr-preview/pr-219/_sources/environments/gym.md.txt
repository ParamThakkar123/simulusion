# Gym and Gymnasium

The Gym/Gymnasium backend adapts standard Gym-like environments to Simulusion's image-first training interface. It accepts either an environment ID string or a pre-built environment instance and returns observations as `{"image": ...}`.

Install: `pip install simulusion[gym]` for Gymnasium extras.

## Main APIs

```python
from simulusion import GymImageEnv, make_gym_env

env = make_gym_env("Pendulum-v1", seed=0, size=(64, 64), render_mode="rgb_array")
obs = env.reset()
```

You can also wrap an already-created environment:

```python
import gymnasium as gym
from simulusion import GymImageEnv

base_env = gym.make("CartPole-v1", render_mode="rgb_array")
env = GymImageEnv(base_env, seed=123, size=(64, 64))
```

Dreamer uses `cfg.env_backend = "gym"` (or `"gymnasium"`, `"generic"`) to select this backend. Set `cfg.gym_render_mode = "rgb_array"` for image observations. See {doc}`../dreamer` for the full Dreamer config reference.

## Observation conversion

`GymImageEnv` always exposes:

```python
{"image": uint8 array with shape (3, H, W)}
```

The wrapper handles several observation styles:

- Tuple reset/step outputs from Gymnasium by taking the first item as the observation.
- Dict observations by preferring image-like keys such as `image`, `pixels`, `rgb`, `observation`, or `state`.
- Vector observations by rendering simple vertical intensity bands into an RGB image.
- HWC, CHW, grayscale, and RGBA images by converting to RGB, resizing, and transposing to CHW.

When the wrapped environment supports `render()`, Simulusion attempts to use rendered frames for visual observations. If rendering fails or only vector observations are available, it falls back to vector-to-image synthesis.

## Seed determinism

``GymImageEnv`` accepts a ``seed`` parameter at construction. It calls ``reset(seed=seed)`` on the wrapped environment (seeding the simulation RNG) and also seeds its own action/observation spaces and internal ``_rng`` for discrete action sampling. Repeated construction with the same ``seed`` produces identical rollouts when the underlying Gym environment is deterministic:

```python
env_a = GymImageEnv("CartPole-v1", seed=0, size=(64, 64))
obs_a = env_a.reset()["image"]

env_b = GymImageEnv("CartPole-v1", seed=0, size=(64, 64))
obs_b = env_b.reset()["image"]
assert (obs_a == obs_b).all()
```

Calling ``reset(seed=...)`` on an existing instance reseeds both the simulation and the wrapper RNGs, so subsequent rollouts are also reproducible.

## Action conversion

For continuous action spaces, `GymImageEnv.action_space` mirrors the wrapped environment's `Box` bounds.

For discrete action spaces, `GymImageEnv.action_space` is a continuous `Box` of shape `(n,)` in `[-1, 1]`. The wrapper expects a one-hot-like action vector and converts it to the discrete index with `argmax` before stepping the base environment. Its `sample()` method returns one-hot vectors with `1.0` at the selected action and `-1.0` elsewhere.

## Example environments

The lightweight catalog now queries the installed Gymnasium registry at runtime instead of maintaining a hardcoded list of versioned environment IDs. Use Gymnasium's environment docs and `simulusion envs list` to inspect the IDs available in your local installation and optional extras.

## CLI collection

The CLI can collect random-policy rollouts from Gym-like environments:

```bash
simulusion collect --env CartPole-v1 --steps 1000 --out cartpole.npz
```

The command first tries `simulusion.make_env()` and falls back to `gym.make()`.

## Troubleshooting

- **Black frames or missing render output**: create the environment with `render_mode="rgb_array"` and pass the same render mode to `GymImageEnv`.
- **Box2D import errors**: install the Box2D Gymnasium extra.
- **Discrete policies produce invalid actions**: emit vectors of length `env.action_space.shape[0]`; the wrapper chooses `argmax`.
- **Custom environment reset signatures**: Gymnasium-style `(obs, info)` and Gym-style `obs` resets are both supported by the wrapper.
