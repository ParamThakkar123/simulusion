# Procgen

The Procgen backend adapts the `procgen.ProcgenEnv` vector API to Simulusion's single-environment image interface for procedurally generated benchmark games such as CoinRun, Maze, Heist, and StarPilot.

Install: `pip install procgen` manually. Upstream Procgen wheels only support Python <= 3.10, while Simulusion requires Python >= 3.11, so there is no `simulusion[procgen]` extra.

## Main APIs

```python
from simulusion.envs.procgen_env import make_procgen_env, list_procgen_envs

env = make_procgen_env("coinrun", seed=0, size=(64, 64))
obs = env.reset()
print(obs["image"].shape)  # (3, 64, 64)
```

The factory accepts Procgen shorthand names and Gym-style ids:

```python
make_procgen_env("coinrun")
make_procgen_env("procgen-coinrun-v0")
make_procgen_env("procgen:procgen-coinrun-v0")
```

Use `list_procgen_envs()` to inspect the supported game names.

Dreamer uses `cfg.env_backend = "procgen"` (or `"coinrun"`) to select this backend. See {doc}`../dreamer` for the full Dreamer config reference.

## Observations and actions

`ProcgenImageEnv` unwraps the leading vector dimension from `ProcgenEnv(num_envs=1)` and returns:

```python
{"image": uint8 array with shape (3, H, W)}
```

Procgen actions are discrete. For consistency with Simulusion's other discrete image adapters, the wrapper exposes a continuous one-hot-like action space with shape `(n,)` and values in `[-1, 1]`. The selected action is the index of the largest value in the model action vector.

## Example script

Simulusion includes a compact Dreamer example for Procgen environments:

```bash
python examples/run_dreamer_procgen.py --env coinrun --total-steps 2000
python examples/run_dreamer_procgen.py --list-envs
```

## Available games

Simulusion recognizes the standard Procgen games: `bigfish`, `bossfight`, `caveflyer`, `chaser`, `climber`, `coinrun`, `dodgeball`, `fruitbot`, `heist`, `jumper`, `leaper`, `maze`, `miner`, `ninja`, `plunder`, and `starpilot`.
