# MuJoCo

Simulusion includes convenience factories for Gymnasium MuJoCo environments, currently focused on configurable Humanoid and HalfCheetah creation. These factories return standard Gymnasium environments; wrap them with `GymImageEnv` if a Simulusion model needs image observations.

Install: `pip install "gymnasium[mujoco]"`

## Humanoid factory

```python
from simulusion import make_mujoco_env

env = make_mujoco_env(
    "Humanoid-v4",
    forward_reward_weight=1.25,
    ctrl_cost_weight=0.1,
    contact_cost_weight=5e-7,
    healthy_reward=5.0,
    terminate_when_unhealthy=True,
    healthy_z_range=(1.0, 2.0),
)
```

The factory builds `Humanoid-{version}` and forwards common reward, reset, health, and observation-component parameters to Gymnasium.

## HalfCheetah factory

```python
from simulusion import make_mujoco_env

env = make_mujoco_env(
    "HalfCheetah-v4",
    forward_reward_weight=0.1,
    reset_noise_scale=0.1,
    exclude_current_positions_from_observation=True,
    render_mode="rgb_array",
)
```

The factory builds `HalfCheetah-{version}` and forwards the selected reward, reset, observation, and render parameters.

## Using MuJoCo with Dreamer-style image observations

Because the MuJoCo factories return raw Gymnasium environments, wrap the environment with `GymImageEnv` or configure Dreamer through the Gym backend:

```python
from simulusion import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "HalfCheetah-v4"
cfg.gym_render_mode = "rgb_array"
```

For custom factory output:

```python
from simulusion import GymImageEnv
from simulusion import make_mujoco_env

base_env = make_mujoco_env("HalfCheetah-v4", render_mode="rgb_array")
env = GymImageEnv(base_env, seed=0, size=(64, 64))
```

## Observation and action contract

Raw MuJoCo factories expose Gymnasium's default MuJoCo observations and continuous `Box` actions. `GymImageEnv` converts observations/renders into `{"image": (3, H, W)}` and keeps continuous action bounds from the base environment.

## Troubleshooting

- **Import or renderer errors**: install `gymnasium[mujoco]` and validate local MuJoCo rendering.
- **Missing images**: pass `render_mode="rgb_array"` for environments that support rendering.
- **Action normalization**: raw factories do not normalize actions; Dreamer `make_env()` adds `NormalizeActions` only when using the configured backend path.
