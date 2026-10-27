# Controllers and Policies

Simulusion provides several controller and policy classes that convert latent
representations into environment actions. The right choice depends on whether
you plan online (CEM at every step), use a trained actor-critic, or evolve
a controller with black-box optimization.

```{contents} Contents
```

## Overview

| Class | Approach | Used by |
|---|---|---|
| `RSSMPolicy` | Cross-entropy method (CEM) planning in latent space | PlaNet, RSSM training |
| `RolloutGenerator` | Collects environment rollouts with optional policy | PlaNet, RSSM training |
| `Controller` | Learned linear mapping `[z, h] → action` via CMA-ES | World Models (Ha & Schmidhuber) |
| `IRISActor` | CNN + LSTM + action head (trained with REINFORCE) | IRIS |
| `IRISCritic` | CNN + LSTM + value head (λ-return baseline) | IRIS |
| `IRISPolicy` | Convenience wrapper around `IRISActor` | IRIS |
| `CNNFeatureExtractor` | Shared 4-layer CNN backbone | IRISActor, IRISCritic |

## RSSMPolicy — CEM planning

`RSSMPolicy` runs a cross-entropy method planner inside an RSSM latent-dynamics
model. At each environment step it samples candidate action sequences, rolls
them out through the RSSM prior, scores them by predicted reward, and refits a
Gaussian to the best candidates.

```python
from simulusion import RSSMPolicy

policy = RSSMPolicy(
    model=rssm,  # RecurrentStateSpaceModel instance
    planning_horizon=20,  # H — imagined steps per candidate
    num_candidates=1000,  # N — candidates per iteration
    num_iterations=10,  # I — CEM refitting iterations
    top_candidates=100,  # K — elite candidates kept
    device=torch.device("cuda"),
)

# At each environment step:
action = policy.poll(observation)
```

The CEM loop:

1. Initialise Gaussian `N(μ, σ)` over action sequences of length `H`.
2. Sample `N` candidate sequences.
3. Roll out each candidate through `deterministic_state_fwd` + `state_prior`,
   scoring by cumulative `pred_reward`.
4. Keep the top `K` elite sequences and refit `(μ, σ)`.
5. Repeat for `I` iterations, then execute the first action of the best mean.

Call `policy.reset()` to zero the hidden and latent state at the start of a new
episode.

## RolloutGenerator — collecting rollouts

`RolloutGenerator` wraps an environment and an optional policy for collecting
episode data. It produces `Episode` objects compatible with PlaNet's
episode-based `Memory`.

```python
from simulusion import RSSMPolicy, RolloutGenerator, Episode

generator = RolloutGenerator(
    env,  # Gymnasium-compatible environment
    device=torch.device("cuda"),
    policy=policy,  # RSSMPolicy (or None for random actions)
    episode_gen=lambda: Episode(postprocess_fn),
    max_episode_steps=1000,
    enable_streaming_video=False,  # set True to stream rollouts to disk
)
```

### Collecting data

```python
# Single random episode (warmup):
episode = generator.rollout_once(random_policy=True)

# Multiple episodes:
episodes = generator.rollout_n(n=5, random_policy=True)

# Single episode with learned policy:
episode = generator.rollout_once(explore=True)
```

### Evaluation

```python
episode, frames, metrics = generator.rollout_eval()

# metrics contains:
#   "eval/episode_reward"       — total undiscounted return
#   "eval/reconstruction_loss"  — MSE between reconstructed and true frames
#   "eval/reward_pred_loss"     — MSE between predicted and true reward
```

The evaluation rollout collects reconstructed frames through the RSSM decoder
and logs prediction quality metrics.

### Streaming video

```python
generator = RolloutGenerator(
    env,
    device,
    policy=policy,
    enable_streaming_video=True,
    streaming_video_path="rollouts/",
    streaming_video_fps=20,
)
```

## IRIS Actor-Critic

IRIS trains a separate actor and critic inside imagined rollouts (no online
planning). Both share the same CNN + LSTM architecture but have different
output heads.

### CNNFeatureExtractor

The shared 4-layer CNN backbone:

```python
from simulusion import CNNFeatureExtractor

cnn = CNNFeatureExtractor(
    frame_shape=(3, 64, 64),
    output_size=512,
)

frames = torch.randn(4, 3, 64, 64)
features = cnn(frames)  # (4, 512)
```

Architecture: `Conv2D(3→32) → Conv2D(32→64) → Conv2D(64→128) → Conv2D(128→256) → Linear(4096 → output_size)`, each conv with kernel 3, stride 2, padding 1 and ReLU.

### IRISActor

```python
from simulusion import IRISActor

actor = IRISActor(
    action_size=6,
    hidden_size=512,
    num_layers=4,
    frame_shape=(3, 64, 64),
)

# Forward with time dimension:
frames = torch.randn(4, 20, 3, 64, 64)  # (B, T, C, H, W)
logits, hidden = actor(frames)  # logits: (B, T, 6)

# Single-step action selection:
action = actor.get_action(frame, temperature=1.0, deterministic=False)
```

The actor processes frames through `CNNFeatureExtractor`, then a 4-layer LSTM,
then a linear action head. For burn-in (initialising the LSTM hidden state from
context frames), pass `burn_in_frames` to `forward()`.

### IRISCritic

```python
from simulusion import IRISCritic

critic = IRISCritic(
    hidden_size=512,
    num_layers=4,
    frame_shape=(3, 64, 64),
)

frames = torch.randn(4, 20, 3, 64, 64)
values, hidden = critic(frames)  # values: (B, T)
```

The critic matches the actor's CNN + LSTM architecture but outputs a scalar
value instead of action logits. It maintains a separate `CNNFeatureExtractor`
instance from the actor.

### IRISPolicy

```python
from simulusion import IRISPolicy

policy = IRISPolicy(action_size=6)

frames = torch.randn(4, 3, 64, 64)
logits = policy(frames)  # (4, 6)
action = policy.act(frames[0])  # scalar action index
hidden = policy.init_hidden(4, "cuda")
```

`IRISPolicy` is a convenience wrapper around `IRISActor`. It does **not**
automatically create a critic — instantiate `IRISCritic` separately if needed.

### Putting it together

```python
import torch
import simulusion

actor = simulusion.IRISActor(action_size=6)
critic = simulusion.IRISCritic()

# Imagined rollout loop:
frames = torch.randn(1, 20, 3, 64, 64)
action_logits, _ = actor(frames)  # (1, 20, 6)
values, _ = critic(frames)  # (1, 20)

# REINFORCE with λ-return baseline:
advantages = ...  # computed from values and predicted rewards
actor_loss = -(action_logits * advantages).mean()
critic_loss = F.mse_loss(values, target_values)
```

## Controller — CMA-ES linear policy

`Controller` is a simple linear layer that maps the concatenated latent and
deterministic states to an action vector:

```python
from simulusion.models.controller import Controller

ctrl = Controller(latent_size=32, hidden_size=256, action_size=3)
action = ctrl(torch.cat([z, h], dim=-1))  # (B, action_size)
```

Weights are trained with CMA-ES (black-box evolution) rather than gradient
descent. See `train_controller.py` in the World Models (Ha & Schmidhuber)
pipeline for a complete example.

## Export

RSSM policies and IRIS actor-critic networks can be exported to ONNX /
TorchScript for deployment:

```python
# Export RSSMPolicy's underlying RSSM model:
rssm.export("rssm.onnx", format="onnx", example_inputs=...)

# Export IRIS actor:
actor.export("actor.onnx", format="onnx", example_inputs=...)
```

## See Also

- {doc}`planet` — uses RSSMPolicy + RolloutGenerator for online CEM planning
- {doc}`iris` — trains IRIS actor-critic inside imagined rollouts
- {doc}`dreamer` — Dreamer-style actor-critic training (separate from these classes)
- {doc}`world_models_guide` — conceptual overview of World Models pipeline (Controller + CMA-ES)
- {doc}`export_guide` — deploying policies to ONNX / TorchScript
