# Atari

Simulusion exposes Atari 2600 environments through Gymnasium and the Arcade Learning Environment (ALE). Atari is the environment family; DIAMOND-style Atari support in Simulusion is an optional preprocessing wrapper/factory for Atari, not a separate environment backend.

Install: `pip install ale-py simulusion[gym]` (ROMs required separately via AutoROM)

## Raw Atari APIs

```python
from simulusion import make_atari_env, list_available_atari_envs

env = make_atari_env("ALE/Pong-v5", obs_type="rgb", frameskip=4)
print(list_available_atari_envs()[:10])
```

`make_atari_env()` registers `ale_py` environments with Gymnasium and delegates to `gymnasium.make()`.

## Raw Atari factory options

| Option | Default | Description |
| --- | --- | --- |
| `env_id` | Required | Gymnasium ID such as `ALE/Pong-v5` or `ALE/Breakout-v5` |
| `obs_type` | `"rgb"` | Use `"rgb"` for image observations or `"ram"` for RAM observations |
| `frameskip` | `4` | Number of frames skipped by ALE between actions |
| `repeat_action_probability` | `0.25` | Sticky-action probability |
| `full_action_space` | `False` | Whether to expose the full Atari action set |
| `max_episode_steps` | `None` | Optional episode cap passed to Gymnasium |

Additional keyword arguments are forwarded to `gymnasium.make()`.

## Atari preprocessing for DIAMOND-style training

This is not a separate environment in Simulusion. It is an Atari preprocessing path implemented by `DiamondAtariWrapper` and `make_diamond_atari_env()` for users who want DIAMOND-compatible Atari rollouts.

```python
from simulusion import make_diamond_atari_env

env = make_diamond_atari_env(
    game="ALE/Breakout-v5",
    frameskip=4,
    max_noop=30,
    terminate_on_life_loss=True,
    reward_clip=True,
    resize=(64, 64),
    seed=0,
)
```

The factory creates the underlying Gymnasium Atari environment with `obs_type="rgb"`, `frameskip=1`, `repeat_action_probability=0.0`, and `full_action_space=False`, then applies Atari-specific preprocessing.

| Preprocessing option | Behavior |
| --- | --- |
| `frameskip` | Repeat each selected discrete Atari action and sum rewards |
| `max_noop` | Run a random no-op reset phase before the episode begins |
| `terminate_on_life_loss` | Mark the step done when an ALE life is lost and lives remain |
| `reward_clip` | Clip summed rewards to `[-1, 1]` |
| `resize` | Resize RGB Atari frames to `(height, width)` |
| `seed` | Seed the internal RNG used for no-op count sampling at reset |

`DiamondAtariWrapper.step()` returns a legacy four-tuple, `obs, reward, done, info`, while `reset()` returns Gymnasium's `(obs, info)` tuple. The preprocessed observation is an HWC `uint8` RGB frame, so transpose it if your model expects channel-first images.

## Native vectorized Atari

For parallel Atari simulation, use `make_atari_vector_env()`:

```python
from simulusion import make_atari_vector_env

vec_env = make_atari_vector_env(
    game="pong",
    num_envs=8,
    obs_type="rgb",
    frameskip=4,
    seed=0,
)
```

This returns ALE's native `AtariVectorEnv`, not Simulusion's multiprocessing `TorchVectorizedEnv`.

## Model usage

- IRIS and Atari-focused workflows can use ALE IDs discovered by `list_available_atari_envs()`.
- Planet catalog entries include a subset of ALE environments.
- Dreamer `make_env()` does not have a dedicated `env_backend="atari"` branch; wrap Atari as a Gym environment with `env_backend="gym"` if you want Dreamer-style image conversion. Use DIAMOND-style preprocessing only when your Atari training recipe expects that specific preprocessing stack.

## Seed determinism

``make_diamond_atari_env`` passes its ``seed`` parameter to the underlying ``gymnasium.make()`` (seeding the ALE RNG) and to ``DiamondAtariWrapper`` (seeding the no-op count RNG). Two consecutive calls with the same seed produce the same initial ALE state and the same number of reset no-ops:

```python
env_a = make_diamond_atari_env("ALE/Breakout-v5", seed=0)
obs_a, _ = env_a.reset()  # deterministic no-op count

env_b = make_diamond_atari_env("ALE/Breakout-v5", seed=0)
obs_b, _ = env_b.reset()  # identical to env_a
```

The raw Atari factory (``make_atari_env``) delegates seeding to ``gymnasium.make()`` via the ``seed`` parameter passed through ``env.reset(seed=...)``.

## Observation and action contract

The raw ALE factory returns Gymnasium's Atari environment directly. Observation shape and dtype depend on `obs_type`; action space is discrete. DIAMOND-style preprocessing still uses Atari as the environment and returns resized HWC RGB frames. If your model expects `{"image": (3, H, W)}`, wrap raw Atari with `GymImageEnv` or transpose/preprocess DIAMOND-style frames before model input.

## Troubleshooting

- **No environments listed**: ensure `ale-py` is installed and importable.
- **ROM errors**: verify ROM installation for the target games.
- **Different game naming conventions**: Gymnasium IDs usually include `ALE/` and `-v5`; native vectorization takes a game name such as `pong`.
- **Preprocessing mismatch**: use the Atari preprocessing section on this page for frame skip, no-op reset, reward clipping, life-loss termination, and resizing when your training recipe expects DIAMOND-style rollouts.
