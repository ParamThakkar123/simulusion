# DeepMind Control Suite

The DeepMind Control Suite (DMC) backend is the default Dreamer environment path in Simulusion. It wraps `dm_control.suite` tasks with a Gym-like interface, keeps all native DMC state observations, and adds a rendered RGB image so image-based world models can train on a consistent observation contract.

## Install

```bash
pip install "simulusion[dmc]"          # pip, Python 3.12 and below
python -m simulusion.install_dmc       # pip or uv, any version, including 3.13
uv sync --extra dmc-uv              # uv, any version
```

On CPython 3.13 the extra alone is not enough. dm-control requires the `labmaze`
distribution, whose newest release (1.0.6) publishes wheels only through CPython
3.12, so resolving it on 3.13 builds labmaze from source with Bazel and the
install fails. Simulusion therefore leaves dm-control out of the `dmc` extra on 3.13
— so the extra always installs cleanly — and `python -m simulusion.install_dmc`
finishes the job:

1. installs [`labmaze-new`](https://pypi.org/project/labmaze-new/), the same
   package with the Bazel extension made opt-in via
   `LABMAZE_BUILD_BAZEL_EXTENSIONS=1`, so it installs as pure Python and provides
   the `labmaze` module `dm_control` imports;
2. installs `dm-control` with `--no-deps`, so pip never resolves that pin;
3. installs dm-control's remaining dependencies explicitly.

`python -m simulusion.install_dmc --check` verifies an existing environment, and
`--dry-run` prints the commands; it uses `uv pip` automatically inside a
uv-managed virtualenv.

With uv, prefer the `dmc-uv` extra: it lists dm-control unconditionally and uv
drops the `labmaze` pin through the `override-dependencies` entry in
`pyproject.toml`, which pip has no equivalent for. `uv sync --extra dmc` installs
everything except dm-control on 3.13, the same as pip.

The installer keeps whatever mujoco is already installed and picks the newest
dm-control that works with it, because dm-control raises its mujoco floor with
almost every release and mujoco 3.13 removed `mjtEnableBit.mjENBL_MULTICCD`,
which mujoco-mjx (and therefore the `brax` extra) still uses. Upgrading mujoco
for dm-control's sake breaks brax in the same environment. Pass
`--upgrade-mujoco` to install the newest mujoco and dm-control together instead.
For the same reason `[tool.uv] constraint-dependencies` caps `mujoco<3.13`.

pip will warn that `dm-control ... requires labmaze, which is not installed`.
That is cosmetic: `labmaze-new` provides the module under a different
distribution name, which pip cannot match to the requirement.

Only `dm_control.locomotion` maze generation needs labmaze's compiled extension.
`dm_control.suite`, which is all this backend uses, does not. If you need the
locomotion mazes, install with `LABMAZE_BUILD_BAZEL_EXTENSIONS=1` and Bazel
available, or use a Python 3.12 environment.

## Main API

```python
from simulusion import DeepMindControlEnv

env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
obs = env.reset()
```

The environment name uses a `domain-task` string. Simulusion splits the string at the first hyphen. For example, `cheetah-run` maps to `domain="cheetah"` and `task="run"`. The special shorthand `cup-*` maps to DMC's `ball_in_cup` domain.

Dreamer uses `cfg.env_backend = "dmc"` to select this backend. See {doc}`../dreamer` for the full Dreamer config reference.

## Common task IDs

| Category | Examples |
| --- | --- |
| Balance | `cartpole-balance`, `cartpole-swingup` |
| Locomotion | `cheetah-run`, `walker-walk`, `walker-run`, `quadruped-walk` |
| Manipulation | `finger-spin`, `finger-turn_easy`, `finger-turn_hard`, `reacher-easy` |
| Catching | `cup-catch` |

The environment catalog includes the canonical Dreamer examples: `cartpole-balance`, `cartpole-swingup`, `cheetah-run`, `finger-spin`, `reacher-easy`, `walker-walk`, `walker-run`, and `quadruped-walk`.

## Observation contract

`DeepMindControlEnv.reset()` returns a dictionary containing:

- Every key from `dm_control`'s `observation_spec()` as a `float32` Gymnasium `Box`.
- An additional `image` key with shape `(3, H, W)` and dtype `uint8`.

The image is rendered from DMC physics with `physics.render(height, width, camera_id=...)`, transposed from HWC to CHW, and copied so downstream code can store it safely.

## Action contract

The action space is a Gymnasium `Box` built from DMC's action spec minimum and maximum arrays. Dreamer creation wraps the backend in `NormalizeActions`, so policy code can emit normalized actions while the wrapper maps finite bounds back to the native DMC range.

## Seed determinism

``DeepMindControlEnv`` passes its ``seed`` parameter directly to ``dm_control.suite.load(..., task_kwargs={"random": seed})``, which seeds the underlying MuJoCo simulation RNG. The wrapper also seeds its internal action-space RNG. Two environments constructed with the same seed produce identical initial states:

```python
env_a = DeepMindControlEnv("cartpole-swingup", seed=0)
env_b = DeepMindControlEnv("cartpole-swingup", seed=0)
obs_a = env_a.reset()["image"]
obs_b = env_b.reset()["image"]
assert (obs_a == obs_b).all()
```

## Cameras and rendering

Pass `camera=<id>` when constructing `DeepMindControlEnv` directly. If no camera is provided, Simulusion uses camera `2` for `quadruped` and camera `0` for other domains. Only `rgb_array` rendering is supported.

## Troubleshooting

- **`ModuleNotFoundError: dm_control`**: install `dm-control` in the active environment.
- **Task name parsing errors**: use `domain-task` format, such as `walker-walk`; use `cup-catch` for `ball_in_cup/catch`.
- **Unexpected image size**: set `cfg.image_size` or pass `size=(height, width)` directly.
- **Action range issues**: if you bypass Dreamer `make_env()`, add `NormalizeActions` yourself when the policy emits normalized actions.
