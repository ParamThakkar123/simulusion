# Public API Quick Reference

Simulusion exposes `simulusion` as the friendly public namespace for both
application code and direct component imports. Use it for factory helpers,
model classes, config classes, and environment constructors.

## Common Workflow

```python
import simulusion

# Discover supported factories.
models = simulusion.list_models()
backends = simulusion.list_env_backends()

# Configure and create a model in one step.
agent = simulusion.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=1_000_000,
)

# Create standalone environments through a consistent backend selector.
env = simulusion.make_env("CartPole-v1", backend="gym")
```

## Factory Helpers

| Helper | Purpose |
|--------|---------|
| `create_config(model, **overrides)` | Build the default config for a model family and apply validated overrides. |
| `create_model(model, config=None, **overrides)` | Instantiate a model or high-level agent by name. Config overrides are applied before construction; unknown fields are passed as constructor arguments when appropriate. |
| `make_env(env_id, backend="auto", **kwargs)` | Create an environment through a named backend such as `gym`, `atari`, `mujoco`, `robotics`, `brax`, or `unity`. |
| `list_models()` | Return canonical names accepted by `create_model`. |
| `list_env_backends()` | Return backend names accepted by `make_env`. |
| `list_envs(model=None)` | Return known environment IDs, optionally filtered by model family. |

## Direct Imports Still Work

The factory API is a convenience layer. Advanced and research workflows can keep
using direct imports from `simulusion`:

```python
from simulusion import DreamerAgent, DreamerConfig, RSSM

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

Use direct imports when you are composing custom modules, subclassing internals,
or need access to implementation-specific constructors.

## 1.0 scope

These names stay in the 1.x public surface, with the following documented
limits:

- There is no DreamerV3 implementation in 1.0, and no `dreamer-v3` registry
  name: the Dreamer family is `dreamer-v1` and `dreamer-v2`.
- `agent.train()` with a step budget is the Dreamer-family path. Other
  registered models train through their dedicated trainers or CLI commands.
- Genie `VideoDataset` loads `.npy` / `.pt` clips, or video files when the
  `viz` extra (OpenCV) is installed. TinyWorlds training still goes through
  `scripts/train_genie_tinyworlds.py`.
