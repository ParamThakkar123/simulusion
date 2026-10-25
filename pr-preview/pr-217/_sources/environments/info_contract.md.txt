# Info dict contract

Every Simulusion environment adapter and wrapper guarantees a minimum set of keys in the `info` dict returned by `step()`. Downstream code (value models, replay buffers, logging) may rely on these keys being present.

## Universally guaranteed keys

The function `finalize_step_info()` in `simulusion.envs._contract` normalises every `step()` return. These three keys are **always** present:

| Key | Type | Description |
|---|---|---|
| `terminated` | `bool` | The episode ended because the task reached a terminal state (goal reached, failure, etc.). |
| `truncated` | `bool` | The episode ended because of an external limit (time horizon, max steps, etc.). |
| `discount` | `np.ndarray` (`float32`) | Discount factor for the transition. `0.0` when `terminated` is true (absorbing state) or `discount` was explicitly set to `0.0` by the backend; `1.0` otherwise, unless overridden by a model or wrapper. |

**Important:** ``GymImageEnv`` without a wrapping ``TimeLimit`` sets ``discount=0.0`` on *any* ``done=True`` return, including truncation, because it treats all terminal transitions the same way. Wrap with ``TimeLimit`` to get ``discount=1.0`` on time-limit truncations (the Dreamer wrapper stack does this automatically). The cross-backend contract tests in ``tests/envs/test_env_contract.py`` (``test_env_contract_shared_assertions``) verify this behavior for each backend adapter.

```python
obs, reward, terminated, truncated, info = env.step(action)
# info always contains:
assert "terminated" in info
assert "truncated" in info
assert "discount" in info
```

## Commonly guaranteed keys

All backends except `WorldModelEnv` and `DiamondAtari` (passthrough) set these keys:

| Key | Type | Description |
|---|---|---|
| `action` | `np.ndarray` (`float32`) | The action as seen by the policy after normalisation / one-hot encoding. |
| `executed_action` | `int` or `np.ndarray` (`float32`) | The action actually sent to the underlying environment **before** normalisation. For discrete actions this is the integer index; for continuous actions it is the raw float array. This value can be stored and replayed identically even when the policy maps normalised actions. |

## Per-backend keys

### DeepMind Control Suite (`dmc.py`)

| Key | Always? | Description |
|---|---|---|
| `vector_observation` | Yes | Concatenated float array of all non-visual sensor values from the DMC `TimeStep`. |

### Gym / Gymnasium (`gym_env.py`)

| Key | Always? | Description |
|---|---|---|
| `vector_observation` | No | Present only when the wrapped environment exposes non-visual observations that can be meaningfully flattened (detected by `flatten_vector_observation`). |

### Brax (`brax_env.py`)

| Key | Always? | Description |
|---|---|---|
| `vector_observation` | Yes | Raw `obs` vector from the Brax `State`. |
| *Dynamic keys* | No | Every key present in `state.metrics` and `state.info` is copied through (e.g. `x_position`, `x_velocity`, `reward_ctrl`, etc.). |

### MuJoCo (`mujoco_env.py`)

| Key | Always? | Description |
|---|---|---|
| `vector_observation` | Yes | Concatenated non-visual sensor values. |
| `time` | Yes | Current simulation time from `mjData.time`. |
| `qpos` | `np.ndarray` (`float64`) | Joint positions. |
| `qvel` | `np.ndarray` (`float64`) | Joint velocities. |

### Unity ML-Agents (`unity_env.py`)

| Key | Always? | Description |
|---|---|---|
| `vector_observation` | No | Present only when the Unity environment provides non-visual observations. |
| `interrupted` | No | `bool` set only when `done` is `True`; indicates whether the episode was interrupted (e.g. by the Unity trainer). |

### DeepMind Lab (`dmlab.py`)

| Key | Always? | Description |
|---|---|---|
| `dmlab_action` | Yes | The raw native action array from DeepMind Lab (before one-hot encoding). |

### BSuite (`bsuite_env.py`)

| Key | Always? | Description |
|---|---|---|
| `vector_observation` | Yes | Flattened observation array from the BSuite environment. |
| `bsuite_id` | Yes | The string identifier of the BSuite task (e.g. `"catch/0"`). |

### Atari / Diamond (`diamond_atari.py`)

| Key | Always? | Description |
|---|---|---|
| `life_lost` | No | `bool` set only when `terminate_on_life_loss` is enabled and the agent loses a life while at least one life remains. All upstream ALE / Gymnasium info keys are passed through unchanged. |

### Procgen (`procgen_env.py`)

| Key | Always? | Description |
|---|---|---|
| *Upstream keys* | No | All keys from the underlying Procgen vector environment (for env index `0`) are passed through. |

### World Model (`world_model_env.py`)

| Key | Always? | Description |
|---|---|---|
| `model_state` | No | The full next state dict from the world model transition; set via `setdefault` so only added when the upstream info does not already provide it. |
| `elapsed_steps` | No | Number of steps taken inside the `WorldModelEnv`; set via `setdefault`. All upstream keys from the model's transition result are passed through. |

## Wrapper-added keys

| Wrapper | Key | Always? | Description |
|---|---|---|---|
| `TimeLimit` | (normalises `terminated`, `truncated`, `discount`) | Yes | Overrides the terminated/truncated/discount from the wrapped environment according to the time-limit logic. See {doc}`wrappers` for details. |
| `ActionRepeat` | `action_repeat` | Yes | The number of times the action was actually repeated before the episode ended. |
| `NormalizeActions` | `action` (overwritten) | Yes | Stores the normalised action under `"action"`. If the wrapped env provides `"action"` but not `"executed_action"`, the unnormalised action is backed up to `"executed_action"`. |

## Using info dict keys

### Logging and debugging

```python
obs, reward, terminated, truncated, info = env.step(action)
logger.store(reward=reward, terminated=terminated, truncated=truncated)
# MuJoCo-specific
if "qpos" in info:
    logger.store(qpos=info["qpos"])
```

### Replay buffers

Always store `info["action"]` and `info["executed_action"]` together so the policy update can reconstruct the exact action that was sent to the environment:

```python
transition = dict(
    obs=obs,
    action=info.get("action"),
    executed_action=info.get("executed_action"),
    reward=reward,
    terminated=terminated,
    truncated=truncated,
    discount=info["discount"],
)
```

### Value-function bootstrapping

The `discount` key controls whether the value of the next state is bootstrapped:

```python
if terminated:
    target = reward
elif truncated:
    target = reward + info["discount"] * next_value  # bootstrap normally
else:
    target = reward + info["discount"] * next_value
```

Note that `discount` is `0.0` for absorbing terminal states (so the bootstrap term is zero), but `1.0` for non-terminal transitions and typically also `1.0` for truncated transitions (the discount is applied **after** multiplying with `discount`).
