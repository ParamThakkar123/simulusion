# Craftax

[Craftax](https://github.com/MichaelTMatthews/Craftax) (Matthews et al., 2024)
reimplements Crafter in JAX and extends it with dungeon floors, combat and
magic. Because it runs entirely on the accelerator, it is orders of magnitude
faster than Crafter.

Install: `pip install "simulusion[craftax]"`. It installs JAX; for GPU support
install the matching `jax[cuda12]` build as well.

## Main API

```python
from simulusion import CraftaxEnv

env = CraftaxEnv("Craftax-Classic-Pixels-v1", seed=0)   # or make_craftax_env("classic")
obs = env.reset()
print(obs["image"].shape, env.action_space.n)            # (3, 64, 64) 17
```

| Name | Alias | Observation | Actions | Max steps |
| --- | --- | --- | --- | --- |
| `Craftax-Classic-Pixels-v1` | `classic` | 63x63 RGB, resized to `size` | 17 | 10000 |
| `Craftax-Classic-Symbolic-v1` | `classic-symbolic` | 1345-d vector | 17 | 10000 |
| `Craftax-Pixels-v1` | `full`, `pixels` | 130x110 RGB, resized | 43 | 100000 |
| `Craftax-Symbolic-v1` | `symbolic` | 8268-d vector | 43 | 100000 |

Pixel variants return `{"image": (3, H, W) uint8}`, scaled from Craftax's
`[0, 1]` floats. Symbolic variants return `{"state": float32 vector}`, also
available as `info["vector_observation"]`.

## Episodes and scoring

Death, standing in lava (Classic), or beating the final boss (full Craftax)
ends an episode as `terminated`. Reaching the step limit is `truncated`.

`info["achievements"]` maps every achievement name to 1 if it has been unlocked
in the episode so far. Craftax-Classic has Crafter's 22 achievements, so
[`crafter_score`](crafter.md) scores it directly. On the last step of an
episode, `info["craftax_score"]` carries Craftax's own score for it.

## Throughput

`CraftaxEnv` steps a single environment through jitted `reset` and `step`
calls, so it fits the same training loops as every other backend. It runs at
roughly 500 steps per second on a CPU after JIT compilation. Batched training
on the accelerator, where Craftax is fastest, should call the Craftax API
directly with `jax.vmap`.

## Training with Dreamer

```python
from simulusion import DreamerConfig

cfg = DreamerConfig(env_backend="craftax", env="classic", action_repeat=1)
```
