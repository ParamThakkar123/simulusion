# DeepMind BSuite

The BSuite backend adapts DeepMind's Behaviour Suite for Reinforcement Learning (BSuite) diagnostic tasks to Simulusion's image-first interface. BSuite tasks provide compact ``dm_env`` observations and mostly discrete actions with known optimal reward profiles, making them useful for detecting fundamental issues in world-model training (e.g. credit assignment, memory, exploration).

Install: ``pip install "simulusion[bsuite]"`` (or ``pip install bsuite``). All 23
environments and their 468 settings run on Linux, macOS and Windows.

A task id names one setting of an environment, ``"<environment>/<setting>"``. A
bare name such as ``"catch"`` means its first setting, ``"catch/0"``.

## Main API

```python
from simulusion import make_bsuite_env, list_available_bsuite_ids

env = make_bsuite_env("catch/0", seed=42, size=(64, 64))
obs = env.reset()
print(obs["image"].shape)  # (3, 64, 64)

# List all installed BSuite task IDs
print(list_available_bsuite_ids()[:5])
```

When BSuite is not installed, ``list_available_bsuite_ids()`` returns example IDs: ``bandit/0``, ``cartpole/0``, ``catch/0``, ``deep_sea/0``, ``discounting_chain/0``, ``memory_len/0``, ``mnist/0``, ``mountain_car/0``, ``umbrella_chain/0``.

## Seed determinism

``BSuiteImageEnv`` accepts a ``seed`` parameter at construction. The seed controls the internal RNG for action-space sampling in ``_BSuiteDiscreteActionSpace``. BSuite itself does not expose a native ``seed`` API (its randomness is encoded in the ``bsuite_id``), so ``reset(seed=...)`` reseeds only the wrapper's action-space RNG:

```python
env = make_bsuite_env("catch/0", seed=10)
action_a = env.action_space.sample()  # deterministic for seed=10

env.reset(seed=20)
action_b = env.action_space.sample()  # deterministic for seed=20
```

## Observations

``BSuiteImageEnv`` converts BSuite's compact ``dm_env`` observations into synthetic RGB images:

- Native BSuite observations are flattened and normalised to ``[0, 1]``.
- The flattened array is arranged onto a square canvas and repeated to 3 channels.
- The canvas is resized to ``(H, W)`` using bilinear interpolation.
- The result is returned as ``{"image": uint8 array with shape (3, H, W)}``.

``obs_type`` picks what the observation holds:

| ``obs_type`` | Keys |
| --- | --- |
| ``"pixels"`` (default) | ``image`` |
| ``"state"`` | ``state``: the flattened observation vector, ``float32``. DreamerV3 learns from this. |
| ``"both"`` | ``image`` and ``state`` (the same as ``include_state=True``) |

```python
env = make_bsuite_env("catch/0", obs_type="state")
obs = env.reset()
assert set(obs) == {"state"} and obs["state"].shape == (50,)
```

## Episode budget and scoring

BSuite runs each setting for a fixed number of episodes
(``env.num_episodes``, 1000 or 10000) rather than a step budget.
``env.episodes_completed`` counts finished episodes and ``env.budget_exhausted``
turns true at the budget; Dreamer training stops there on its own.

BSuite scores come from its own logs. ``results_dir=`` (``cfg.bsuite_results_dir``)
records the run with ``bsuite.load_and_record_to_csv``; load the CSVs with
``bsuite.logging.csv_load`` and score them with
``bsuite.experiments.summary_analysis``. A benchmark score covers every setting of
every environment; ``list_bsuite_settings("catch")`` lists one environment's.

## BSuite in a Dreamer config

```python
from simulusion.configs.dreamer_config import DreamerConfig

cfg = DreamerConfig(
    env_backend="bsuite",
    env="catch/3",
    bsuite_obs_type="state",
    bsuite_results_dir="runs/bsuite",
    action_repeat=1,
)
```

Training stops once the setting's episode budget is spent.

## Actions

BSuite tasks expose discrete actions. ``BSuiteImageEnv`` maps them to a continuous one-hot ``Box`` of shape ``(n,)`` with values in ``[-1, 1]``. The selected action is the index of the largest value. ``info["action"]`` stores the one-hot vector; ``info["executed_action"]`` stores the integer index.

## Info contract

| Key | Always? | Description |
| --- | ------- | ----------- |
| ``discount`` | Yes | ``float32`` scalar from the BSuite ``TimeStep`` |
| ``bsuite_id`` | Yes | Task identifier string (e.g. ``"catch/0"``) |
| ``action`` | Yes | One-hot action vector |
| ``executed_action`` | Yes | Integer index sent to the BSuite environment |
| ``vector_observation`` | Yes | Flattened raw observation array |
| ``terminated`` | Yes | ``bool`` — True when BSuite ``TimeStep.last()`` |
| ``truncated`` | Yes | Always ``False`` (BSuite does not expose time limits) |

## Troubleshooting

- **``ImportError: No module named bsuite``**: install the optional dependency with ``pip install bsuite``.
- **``"catch/0" not recognised``**: verify the task ID exists in your BSuite installation with ``list_available_bsuite_ids()``.
- **Synthetic images are uninformative**: BSuite observations are small numeric vectors tiled onto a canvas; the synthetic image preserves all information but may look unfamiliar. Use ``obs_type="state"`` to learn from the raw vector.
