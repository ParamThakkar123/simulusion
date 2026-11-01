# Environment Wrappers

`simulusion` exposes reusable environment wrappers used by Dreamer and other environment pipelines. The wrapper module contains reusable preprocessing wrappers used by Dreamer and other environment pipelines. These wrappers let you compose time limits, action repeats, action normalization, observation dictionaries, one-hot actions, reward observations, and image transforms.

## Standard Dreamer wrapper stack

Dreamer environment construction applies wrappers in this order:

```python
env = ActionRepeat(env, cfg.action_repeat)
env = NormalizeActions(env)
env = TimeLimit(env, cfg.time_limit // cfg.action_repeat)
```

This creates a stable interface for policies that emit normalized actions and train on fixed-length episodes.

## Common wrappers

| Wrapper | Purpose |
| --- | --- |
| `TimeLimit` | End an episode after a fixed number of wrapper steps |
| `ActionRepeat` | Repeat each action and accumulate reward |
| `NormalizeActions` | Map normalized `[-1, 1]` actions back to finite environment bounds |
| `ObsDict` | Convert plain observations into a dictionary under a named key |
| `OneHotAction` | Convert one-hot vectors into discrete action indices |
| `RewardObs` | Add the latest reward to the observation dictionary under `reward` |
| `ResizeImage` | Resize image entries in an observation dictionary |
| `RenderImage` | Add `env.render("rgb_array")` output to observations |
| `SelectAction` | Select a named action entry before passing it to the environment |

## Example composition

```python
from simulusion import ActionRepeat, NormalizeActions, TimeLimit, make_gym_env

env = make_gym_env("Pendulum-v1", size=(64, 64), render_mode="rgb_array")
env = ActionRepeat(env, amount=2)
env = NormalizeActions(env)
env = TimeLimit(env, duration=500)
```

## Action wrappers

Use `NormalizeActions` for continuous-control environments when the policy emits normalized values but the simulator expects task-specific bounds. Use `OneHotAction` for raw discrete environments when your policy outputs one-hot action vectors.

`GymImageEnv` already provides a one-hot-style action space for discrete base environments, so avoid double-applying discrete conversion unless you intentionally bypass `GymImageEnv`.

## Observation wrappers

Use `ObsDict` when a base environment returns a plain array but downstream model code expects a dictionary. Use `RewardObs` when the model should observe the previous reward. Use `RenderImage` and `ResizeImage` to add or resize images for pixel-based training.

## Troubleshooting

- **Episode lengths are shorter than expected**: account for `ActionRepeat`; Dreamer divides `time_limit` by `action_repeat` before applying `TimeLimit`.
- **Actions outside bounds**: add `NormalizeActions` or check whether your policy already emits native environment actions.
- **Missing observation keys**: inspect the environment after each wrapper in your stack to verify the expected dictionary keys.
- **Image shape mismatch**: confirm whether the wrapper returns HWC or CHW images and transpose before model input if needed.
