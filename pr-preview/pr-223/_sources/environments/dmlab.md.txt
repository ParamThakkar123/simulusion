# DeepMind Lab

Simulusion supports DeepMind Lab through `DMLabEnv`, a lightweight adapter around the
native `deepmind_lab` Python module. It converts Lab RGB observations into the
channel-first image dictionary used by Dreamer and other pixel-based world-model
code.

Install: `pip install dmlab-gym && dmlab-gym build` (or install `deepmind_lab` directly)

Dreamer uses `cfg.env_backend = "dmlab"` to select this backend. DMLab-specific fields (`dmlab_action_repeat`, `dmlab_observations`, `dmlab_action_set`, `dmlab_config`, `dmlab_renderer`) are forwarded from `DreamerConfig` to `DMLabEnv`. See {doc}`../dreamer` for the full Dreamer config reference.

## Direct usage

```python
from simulusion import make_dmlab_env

env = make_dmlab_env("rooms_collect_good_objects_train", seed=0, size=(64, 64))
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

The default action space is a normalized one-hot `Box[-1, 1]` over a compact set
of navigation actions. You can pass `action_set=` to `DMLabEnv` or set
`cfg.dmlab_action_set` to use a custom list of native seven-element Lab actions.

## Backend options

| Option | Default | Description |
| --- | --- | --- |
| `cfg.dmlab_action_repeat` | `4` | Native Lab frame repeat passed to `env.step(..., num_steps=...)`. |
| `cfg.dmlab_action_set` | `None` | Optional custom 2D array of native Lab actions. |
| `cfg.dmlab_observations` | `None` | Additional native observation names. `RGB_INTERLEAVED` is always included. |
| `cfg.dmlab_config` | `None` | Extra Lab config values. Width and height are derived from `cfg.image_size`. |
| `cfg.dmlab_renderer` | `"hardware"` | Renderer argument forwarded to `deepmind_lab.Lab`. |

Simulusion's shared Dreamer wrapper stack still applies `cfg.action_repeat` outside
the DMLab adapter. If you only want native Lab frame repeat, leave
`cfg.action_repeat = 1`.
