# Brax Environments

This page explains how to configure Simulusion world-model training for
[JAX/Brax](https://github.com/google/brax) continuous-control environments.

## Overview

Brax environments use a functional API: `reset(rng)` returns an environment
state, and `step(state, action)` returns the next state. Simulusion wraps this API
with `BraxImageEnv`, a Gym-like adapter that stores the latest Brax state between
calls and returns image observations in the format expected by pixel-based agents
such as Dreamer.

The adapter supports:

- Brax environment IDs such as `"ant"`, `"humanoid"`, `"hopper"`,
  `"halfcheetah"`, and `"walker2d"`.
- Pre-built Brax environment instances with `reset(rng)`, `step(state, action)`,
  and `action_size` attributes.
- Optional JIT compilation of Brax `reset` and `step` functions.
- Continuous action spaces exposed as `[-1, 1]` vectors.
- Deterministic RGB feature-band images for vector observations.
- Raw vector observations in `info["vector_observation"]` for diagnostics.

Install: `pip install simulusion[brax]`

Dreamer uses `cfg.env_backend = "brax"` to select this backend. See {doc}`../dreamer` for the full Dreamer config reference. Brax-specific fields (`brax_backend`, `brax_jit`, `brax_suppress_warp_warnings`) are forwarded from `DreamerConfig` to `BraxImageEnv`.

## Direct adapter usage

You can also construct the adapter directly:

```python
from simulusion import make_brax_env

env = make_brax_env(
    "ant",
    seed=0,
    size=(64, 64),
    backend="generalized",
    episode_length=1000,
    jit=True,
)

obs = env.reset()
action = env.action_space.sample()
next_obs, reward, done, info = env.step(action)

print(obs["image"].shape)  # (3, 64, 64)
print(info["vector_observation"].shape)  # Raw Brax observation vector.
```

## Observation format

Many Brax tasks return vector observations rather than rendered camera frames.
Simulusion converts those vectors into deterministic RGB feature-band images so the
same pixel-based model code can consume the environment stream. This conversion
is intended as a compatibility layer for world-model pipelines that expect
images; use `info["vector_observation"]` when you need access to the raw Brax
state observation for debugging or custom losses.

The adapter always advertises this observation space:

```python
{"image": Box(low=0, high=255, shape=(3, height, width), dtype=uint8)}
```

## Action format

Brax actions are continuous vectors. Simulusion exposes a continuous Gymnasium
`Box` action space with shape `(env.action_size,)` and bounds `[-1, 1]`. Incoming
actions are clipped to this range before being forwarded to Brax.

## Troubleshooting

### Missing Brax or JAX

If you see an import error for `brax`, `jax`, or `jax.numpy`, install the Brax
extra:

```bash
pip install simulusion[brax]
```

### Backend selection

The default Brax backend is `"generalized"`. Some Brax versions and tasks also
support `"mjx"`; set `cfg.brax_backend = "mjx"` only when your installed Brax
version supports it for the selected environment.

### JIT compilation

JIT is enabled by default through `cfg.brax_jit = True`. Disable it while
debugging shape or dtype issues:

```python
cfg.brax_jit = False
```
