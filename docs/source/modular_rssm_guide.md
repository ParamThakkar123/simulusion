# Modular RSSM

The Modular RSSM lets you mix and match encoder, backbone, and decoder
components for world-model research. The standard Dreamer RSSM is a fixed
architecture; the modular variant exposes every piece as a pluggable component.

## See it in action

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/modular_rssm_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>The same encoder and decoder with a GRU and an LSTM backbone, both predicting open loop next to the real frames.</figcaption>
</figure>
```

Trained from scratch in one to two hours on a 4 GB laptop GPU with
`python demos/modular_rssm_demo.py train`, then recorded with `python demos/modular_rssm_demo.py record`.
More clips are in the {ref}`gallery <gallery-modular-rssm>`.

```{contents} Contents
:depth: 3
```

## Overview

The architecture has three slots:

```
Observation → Encoder → Backbone → Decoder → Reconstruction
                           │
                    Reward Decoder → Reward prediction
```

| Slot | Built-in options | Custom |
|---|---|---|
| **Encoder** | `ConvEncoder`, `MLPEncoder`, `ViTEncoder` | Subclass `EncoderBase` |
| **Backbone** | `GRUBackbone`, `LSTMBackbone`, `TransformerBackbone` | Subclass `BackboneBase` |
| **Decoder** | `ConvDecoder`, `MLPDecoder` | Subclass `DecoderBase` |

A `reward_decoder` (always an `MLPDecoder`) is attached automatically when using
the factory, or can be provided manually.

## Quick start

Use the factory for the most common combinations:

```python
from simulusion.models.modular_rssm import create_modular_rssm

rssm = create_modular_rssm(
    encoder_type="conv",  # "conv" | "mlp" | "vit"
    decoder_type="conv",  # "conv" | "mlp"
    backbone_type="gru",  # "gru" | "lstm" | "transformer"
    obs_shape=(3, 64, 64),  # (C, H, W) for images, (D,) for state vectors
    action_size=6,
    stoch_size=32,
    deter_size=200,
    embed_size=1024,
)
```

The factory creates the three components plus a reward decoder and wraps them
in a `ModularRSSM` container. All components are also importable individually
for direct construction.

## Components

### Encoders

```python
from simulusion.models.modular_rssm import ConvEncoder, MLPEncoder, ViTEncoder

# Convolutional encoder — Dreamer-style, for image observations
enc = ConvEncoder(input_shape=(3, 64, 64), embed_size=1024, depth=32)

# MLP encoder — for low-dimensional state observations
enc = MLPEncoder(input_dim=10, embed_size=256, hidden_sizes=[256, 256])

# Vision Transformer encoder — for image observations with global context
enc = ViTEncoder(input_shape=(3, 64, 64), embed_size=512, patch_size=8, depth=6)
```

| Encoder | Input | Embedding |
|---|---|---|
| `ConvEncoder` | `(B, C, H, W)` uint8/float | `(B, embed_size)` |
| `MLPEncoder` | `(B, D)` float | `(B, embed_size)` |
| `ViTEncoder` | `(B, C, H, W)` float | `(B, embed_size)` |

### Backbones

```python
from simulusion.models.modular_rssm import (
    GRUBackbone,
    LSTMBackbone,
    TransformerBackbone,
)

# GRU — standard RSSM recurrent dynamics
bb = GRUBackbone(
    action_size=6, stoch_size=32, deter_size=200, hidden_size=200, embed_size=1024
)

# LSTM — longer memory than GRU, at higher compute cost
bb = LSTMBackbone(
    action_size=6, stoch_size=32, deter_size=200, hidden_size=200, embed_size=1024
)

# Transformer — global dependencies, no recurrent state
bb = TransformerBackbone(
    action_size=6,
    stoch_size=32,
    deter_size=200,
    embed_size=256,
    num_heads=4,
    num_layers=2,
)
```

| Backbone | State keys | Use when |
|---|---|---|
| `GRUBackbone` | `mean`, `std`, `stoch`, `deter` | Standard Dreamer-style dynamics |
| `LSTMBackbone` | `mean`, `std`, `stoch`, `deter`, `cell` | Longer horizons, slower drift |
| `TransformerBackbone` | `mean`, `std`, `stoch`, `deter` | Non-recurrent, parallel training |

### Decoders

```python
from simulusion.models.modular_rssm import ConvDecoder, MLPDecoder

# Convolutional decoder — reconstructs images from latent features
dec = ConvDecoder(stoch_size=32, deter_size=200, output_shape=(3, 64, 64), depth=32)

# MLP decoder — reconstructs low-dimensional observations
dec = MLPDecoder(stoch_size=32, deter_size=200, output_dim=10, hidden_sizes=[256, 256])
```

Both decoders return a `torch.distributions` object (the convolutional decoder
returns a pixel-wise `Independent(Normal(mean, 1))`).

## Direct construction

When you need fine-grained control over component configuration, construct
each piece and pass them to `ModularRSSM` directly:

```python
from simulusion.models.modular_rssm import (
    ModularRSSM,
    ConvEncoder,
    ConvDecoder,
    GRUBackbone,
    MLPDecoder,
)

encoder = ConvEncoder((3, 64, 64), embed_size=1024)
decoder = ConvDecoder(32, 200, (3, 64, 64))
backbone = GRUBackbone(6, 32, 200, 200, 1024)
reward_decoder = MLPDecoder(32, 200, 1)

rssm = ModularRSSM(encoder, decoder, backbone, reward_decoder)
```

When `reward_decoder` is omitted, calling `rssm.decode_reward(...)` raises
a clear error.

## Forward pass

The `ModularRSSM` operates in two modes — observation (encode + posterior) and
imagination (prior only):

### Single-step operations

```python
state = rssm.init_state(batch_size=4, device="cpu")
#   state = {"mean": (4, 32), "std": (4, 32), "stoch": (4, 32), "deter": (4, 200)}

action = torch.randn(4, 6)

# Observe step: encode observation, compute posterior
prior, posterior = rssm.observe_step(state, action, observation)

# Imagine step: prior only (no observation available)
next_state = rssm.imagine_step(state, action)
```

### Rollout operations

```python
# Observe a full trajectory — returns stacked states
priors, posteriors = rssm.observe_rollout(
    obs=observations,  # (T, B, C, H, W)
    actions=actions,  # (T, B, action_size)
    nonterms=nonterms,  # (T, B)  or (T, B, 1)
    prev_state=state,
    horizon=T,
)
#   priors["stoch"]:     (T, B, 32)
#   posteriors["stoch"]: (T, B, 32)

# Imagine a rollout using an actor policy
imagined = rssm.imagine_rollout(
    actor=policy_network,  # callable: features → action
    prev_state=state,
    horizon=15,
)
#   imagined["stoch"]: (15, B, 32)
```

### Decoding

```python
# Decode observations from latent features
features = torch.cat([posterior["stoch"], posterior["deter"]], dim=-1)
obs_dist = rssm.decode_observation(features)
reconstruction = obs_dist.mean  # or obs_dist.sample()

# Decode rewards
reward_dist = rssm.decode_reward(features)
```

## Ablation studies

Swapping one component at a time isolates its contribution:

```python
# Same encoder and decoder, different backbone
bb_gru = GRUBackbone(6, 32, 200, 200, 1024)
rssm_gru = ModularRSSM(enc, dec, bb_gru)

bb_lstm = LSTMBackbone(6, 32, 200, 200, 1024)
rssm_lstm = ModularRSSM(enc, dec, bb_lstm)

bb_transformer = TransformerBackbone(6, 32, 200, 256)
rssm_transformer = ModularRSSM(enc, dec, bb_transformer)
```

Or swap the encoder while keeping the backbone fixed:

```python
enc_conv = ConvEncoder((3, 64, 64), 1024)
enc_vit = ViTEncoder((3, 64, 64), 512, patch_size=8, depth=6)

rssm_conv = ModularRSSM(enc_conv, dec, bb)
rssm_vit = ModularRSSM(enc_vit, dec, bb)
```

## Custom components

Subclass `EncoderBase`, `DecoderBase`, or `BackboneBase` and implement the
required interface:

```python
from simulusion.models.modular_rssm import EncoderBase


class MyEncoder(EncoderBase):
    def __init__(self, input_dim: int, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


rssm = ModularRSSM(MyEncoder(64, 256), decoder, backbone)
```

The base classes enforce the contract via `ABC`:

| Base class | Contract |
|---|---|
| `EncoderBase` | `forward(obs) → embedding`, attribute `embed_size` |
| `DecoderBase` | `forward(features) → distribution or tensor` |
| `BackboneBase` | `forward(state, action, obs_embed, nonterm) → (prior, posterior)`, `init_state(batch, device) → state_dict` |

## Integration with training

The `ModularRSSM` works with any Dreamer-style training loop. The standard
pattern is:

```python
state = rssm.init_state(batch_size, device)

for t in range(horizon):
    prior, posterior = rssm.observe_step(state, actions[t], observations[t])
    state = posterior

# Train on the posterior states
features = torch.cat([posterior["stoch"], posterior["deter"]], dim=-1)
recon_loss = -rssm.decode_observation(features).log_prob(observations).mean()
reward_loss = -rssm.decode_reward(features).log_prob(rewards).mean()

# Imagine for actor-critic
imagined = rssm.imagine_rollout(actor, state, horizon=15)
```

The `seq_to_batch` and `detach_state` helpers are available for managing
sequence dimensions and gradient flow in recurrent training.

## Available via the public API

```python
import simulusion

rssm = simulusion.create_modular_rssm(
    encoder_type="vit",
    backbone_type="transformer",
    obs_shape=(3, 64, 64),
    action_size=6,
)

# Or import directly
from simulusion import ModularRSSM, create_modular_rssm
```

## See Also

- {doc}`vision_guide` — available encoders (ConvEncoder, ViTEncoder, MLPEncoder) and decoders (ConvDecoder, MLPDecoder)
- {doc}`dreamer` — using the ModularRSSM inside a full Dreamer training loop
