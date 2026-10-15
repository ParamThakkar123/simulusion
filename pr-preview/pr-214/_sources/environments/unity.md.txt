# Unity ML-Agents

The Unity ML-Agents backend connects Simulusion to an external Unity executable. It supports continuous-action ML-Agents behaviors, extracts visual observations when available, and converts vector observations into image-like inputs when necessary.

Install: `pip install simulusion[ml-agents]` (requires a Unity executable at runtime)

## Main API

```python
from simulusion import UnityMLAgentsEnv

env = UnityMLAgentsEnv(
    file_name="/path/to/UnityEnvironment.x86_64",
    behavior_name=None,
    seed=0,
    size=(64, 64),
    worker_id=0,
    base_port=5005,
    no_graphics=True,
    time_scale=20.0,
    quality_level=1,
    max_episode_steps=1000,
)
```

If `behavior_name` is omitted, Simulusion uses the first behavior advertised by the Unity executable. The wrapper currently supports continuous action spaces only.

Dreamer uses `cfg.env_backend = "unity_mlagents"` (or `"unity"`, `"mlagents"`) to select this backend. Requires `cfg.unity_file_name`. See {doc}`../dreamer` for the full Dreamer config reference.

## Observation contract

`UnityMLAgentsEnv` exposes:

```python
{"image": uint8 array with shape (3, H, W)}
```

For visual Unity observations, the wrapper normalizes and resizes the image, handles grayscale/RGBA/channel-order variants, and returns channel-first RGB. For vector-only observations, it synthesizes a simple RGB image using value bands.

## Action contract

The action space is a continuous Gymnasium `Box` with shape `(continuous_size,)` and bounds `[-1.0, 1.0]`. Discrete ML-Agents action spaces are not supported by the current wrapper.

## Engine settings

Simulusion configures Unity through `EngineConfigurationChannel`:

| Setting | Source / behavior |
|---|---|
| `width` / `height` | Derived from the requested Simulusion image size |
| `quality_level` | Controls graphics quality |
| `time_scale` | Speeds up or slows down simulation |
| `no_graphics=True` | Improves throughput for headless training when the executable supports it |

Use unique `worker_id` values when launching multiple Unity environments on the same machine so ML-Agents ports do not collide.

## Troubleshooting

- **No behaviors found**: verify the executable starts and has an ML-Agents behavior configured.
- **Behavior name not found**: inspect available behavior names or leave `behavior_name=None` to select the first one.
- **Discrete action error**: configure the Unity behavior for continuous actions or extend the wrapper before using discrete branches.
- **Port conflicts**: change `worker_id` or `base_port` for parallel runs.
- **Slow simulation**: increase `time_scale`, lower `quality_level`, and use `no_graphics=True` where supported.
