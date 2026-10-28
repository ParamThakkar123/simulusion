# Environments Guide

For the full environment backend reference — including installation, factory
functions, observation/action contracts, and troubleshooting — see the
{doc}`environments/index` page.

```{toctree}
:maxdepth: 1
:titlesonly:

Environment Backends <environments/index>
```

## Quick start

```python
from simulusion import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.image_size = 64
cfg.action_repeat = 2
cfg.time_limit = 1000
```

For direct environment construction, use the top-level factories:

```python
from simulusion import (
    DeepMindControlEnv,
    make_atari_env,
    make_brax_env,
    make_bsuite_env,
    make_dmlab_env,
    make_gym_env,
    make_procgen_env,
    make_robotics_env,
)

dmc_env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
gym_env = make_gym_env("Pendulum-v1", seed=0, size=(64, 64))
atari_env = make_atari_env("ALE/Pong-v5", obs_type="rgb", frameskip=4)
```

See {doc}`environments/index` for the full list of backends, their
installation requirements, and conventions.
