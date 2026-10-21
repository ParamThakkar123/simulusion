# Training Guide

This guide covers each family's own training entrypoints. Every family also
trains on the shared {py:class}`~simulusion.training.Trainer` from one YAML run
config, with resuming, data parallelism and reproducible run directories:
see {doc}`trainer_guide`, and {doc}`episodes_guide` for training world models
from your own or LeRobot data.

```{contents} Contents
```

## Overview

Simulusion supports training multiple world model algorithms with a unified interface.

## Basic Training Flow

1. Select an algorithm
2. Override environment, dataset, or optimization parameters
3. Initialize the agent
4. Call `train()` and monitor logs/checkpoints

The simplest path is the top-level `simulusion` API:

```python
import simulusion

agent = simulusion.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=1_000_000,
)
agent.train()
```

For research code, the lower-level config and agent classes remain available.

## Dreamer Training

Preferred application API:

```python
import simulusion

agent = simulusion.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=1_000_000,
)
agent.train()
```

Equivalent direct API:

```python
from simulusion import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.total_steps = 1_000_000

agent = DreamerAgent(cfg)
agent.train()
```

## JEPA Training

```python
import simulusion

agent = simulusion.create_model(
    "jepa",
    dataset="imagenet",
    batch_size=64,
    epochs=100,
)
agent.train()
```

## IRIS Training

`IRISAgent` needs constructor arguments such as `action_size` and `device` in
addition to its config, so pass those as constructor overrides:

```python
import torch
import simulusion

agent = simulusion.create_model(
    "iris",
    env="ALE/Pong-v5",
    total_epochs=100,
    action_size=4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

## Custom Training Loop

For advanced users, implement custom training:

```python
from simulusion import DreamerAgent, ReplayBuffer

agent = DreamerAgent(cfg)
memory = ReplayBuffer(
    size=100_000,
    obs_shape=(3, 64, 64),
    action_size=6,
    seq_len=50,
    batch_size=50,
)

for step in range(cfg.total_steps):
    # Collect experience
    experience = agent.collect_episode()
    memory.add_episode(experience)

    # Train
    if step % cfg.update_steps == 0:
        batch = memory.sample_batch()
        metrics = agent.update(batch)

    # Log
    if step % cfg.scalar_freq == 0:
        print(f"Step {step}: {metrics}")
```

## Configuration

All training is controlled via config objects:

### Common Parameters

| Parameter | Description |
|---|---|
| `seed` | Random seed |
| `device` | Training device |
| `total_steps` / `epochs` | Training duration |
| `batch_size` | Batch size |
| `learning_rate` | Learning rate |
| `grad_clip_norm` | Gradient clipping |

### Logging

| Parameter | Description |
|---|---|
| `enable_wandb` | Weights & Biases logging |
| `log_dir` | Base log directory for supported agents |
| `scalar_freq` / `log_interval` | Metric logging cadence, depending on the agent |
| `checkpoint_interval` | Save frequency |

### Starter YAML configs

DIAMOND, IRIS, and JEPA include starter experiment YAML files in
`simulusion/configs/experiments/`. Use them with the unified CLI and optional
OmegaConf/Hydra-style dot-list overrides:

```bash
simulusion train diamond --config simulusion/configs/experiments/diamond.yaml preset=small seed=1
simulusion train iris --config simulusion/configs/experiments/iris.yaml total_epochs=100 env=ALE/Breakout-v5
simulusion train jepa --config simulusion/configs/experiments/jepa.yaml optimization.epochs=50 data.batch_size=128
```

Add `--print-config` to inspect the composed configuration without launching a run.

## Environment Setup

### DMC
```python
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
```

### DeepMind Lab
```python
cfg.env_backend = "dmlab"
cfg.env = "rooms_collect_good_objects_train"
cfg.dmlab_action_repeat = 4
```

### Gym
```python
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
```

### Brax
```python
cfg.env_backend = "brax"
cfg.env = "ant"
cfg.brax_backend = "generalized"
```

### Unity ML-Agents
```python
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = "path/to/env.exe"
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs
```

### Weights & Biases
```python
cfg.enable_wandb = True
cfg.wandb_project = "simulusion"
cfg.wandb_entity = "your-entity"
```

## Checkpointing

Models are automatically saved:

```python
# Resume training
cfg.restore = True
cfg.checkpoint_path = "path/to/checkpoint"
```

## Best Practices

1. **Start small**: Use short episodes and few steps for debugging
2. **Monitor metrics**: Watch loss curves and environment rewards
3. **Tune hyperparameters**: Adjust learning rates and batch sizes
4. **Use checkpoints**: Save frequently and resume from failures
5. **Log experiments**: Use WandB or TensorBoard for tracking

## See Also

- {doc}`evaluation_guide` — how to evaluate trained agents
- {doc}`memory_guide` — replay buffer setup for each agent
- {doc}`dreamer` — Dreamer-specific training loop details
- {doc}`diamond` — DIAMOND diffusion-based training loop