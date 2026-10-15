# Vision Components

Simulusion provides a family of reusable vision modules — encoders, decoders,
tokenizers, quantization layers, and distribution transforms — that serve as
building blocks for world models and representation learning.

```{contents} Contents
:depth: 3
```

## Overview

All components live under `simulusion.vision` and are accessible from the
top-level package:

```python
import simulusion

# Imports
from simulusion import ConvEncoder, ConvDecoder, VideoTokenizer
from simulusion import VectorQuantizer, VectorQuantizerEMA
```

| Category | Component | Used by |
|---|---|---|
| **Encoders** | `ConvEncoder` | Dreamer (image → embedding) |
| | `CNNEncoder` | PlaNet (image → embedding) |
| | `IRISEncoder` | IRIS (image → discrete tokens) |
| **Decoders** | `ConvDecoder` | Dreamer (latent → image distribution) |
| | `CNNDecoder` | PlaNet (latent → image) |
| | `DenseDecoder` | Dreamer (latent → reward/value/discount) |
| | `ActionDecoder` | Dreamer (latent → action distribution) |
| | `IRISDecoder` | IRIS (tokens → image) |
| **Video tokenization** | `VideoTokenizer` | Genie (video → discrete tokens) |
| **Quantization** | `VectorQuantizer` | IRIS, Genie (embedding → codebook index) |
| | `VectorQuantizerEMA` | IRIS, Genie (EMA codebook updates) |
| **Distributions** | `TanhBijector` | Dreamer (action squashing) |
| | `SampleDist` | Dreamer (MC-sampled distribution stats) |
| | `_TwoHotDistribution` | DreamerV2 (symlog two-hot encoding) |
| **VAE** | `ConvVAE` | Standalone convolutional VAE |

## Encoders

### `ConvEncoder` — Dreamer convolutional encoder

```python
from simulusion import ConvEncoder

encoder = ConvEncoder(
    input_shape=(3, 64, 64),  # (C, H, W)
    embed_size=256,  # output dimension
    activation="elu",  # see below
    depth=32,  # base channel count
)

obs = torch.randn(4, 3, 64, 64)
embedding = encoder(obs)  # (4, 256)
```

Architecture: 4 convolutional layers (kernel 4, stride 2) with channel
doubling `32 → 64 → 128 → 256`, then a linear projection to `embed_size`.
Input values in `[-0.5, 0.5]`.

### `CNNEncoder` — PlaNet encoder

```python
from simulusion import CNNEncoder

encoder = CNNEncoder(embedding_size=256, activation_function="relu")
```

Same depth-doubling pattern as `ConvEncoder` but hardcoded to 3 input
channels and no configurable depth parameter.

### `IRISEncoder` — IRIS discrete encoder

```python
from simulusion import IRISEncoder

encoder = IRISEncoder(
    vocab_size=512,  # codebook size
    tokens_per_frame=16,  # 4×4 grid of tokens
    embedding_dim=512,
    in_channels=3,
    frame_shape=(3, 64, 64),
)
```

Architecture: 4 conv layers (stride 2, 64×64 → 4×4), self-attention at 16×16
and 8×8 resolutions, residual blocks, then a `VectorQuantizerEMA` produces
discrete token indices. Input should be 64×64 images.

## Decoders

### `ConvDecoder` — Dreamer convolutional decoder

```python
from simulusion import ConvDecoder

decoder = ConvDecoder(
    stoch_size=30,  # stochastic latent dimension
    deter_size=200,  # deterministic latent dimension
    output_shape=(3, 64, 64),  # (C, H, W)
    activation="elu",
    depth=32,
)

features = torch.randn(4, 230)  # stoch + deter concatenated
dist = decoder(features)  # Independent(Normal(mean, 1), 3)
reconstruction = dist.mean  # (4, 3, 64, 64)
loss = -dist.log_prob(target)  # reconstruction loss
```

Architecture: linear projection from `(stoch+deter)` to `32×depth`, then
4 transposed convolutions (stride 2, kernels `[5, 5, 6, 6]`). Returns a
`torch.distributions.Independent(Normal(mean, 1))` distribution so you can
compute `log_prob` directly.

### `DenseDecoder` — reward/value/discount head

```python
from simulusion import DenseDecoder

# Regression (reward, value)
decoder = DenseDecoder(
    stoch_size=30,
    deter_size=200,
    output_shape=(1,),
    n_layers=2,
    units=400,
    activation="elu",
    dist="normal",
)

# Binary classification (discount)
decoder = DenseDecoder(
    stoch_size=30,
    deter_size=200,
    output_shape=(1,),
    n_layers=2,
    units=400,
    activation="elu",
    dist="binary",
)

# Symlog two-hot (DreamerV2)
decoder = DenseDecoder(
    stoch_size=30,
    deter_size=200,
    output_shape=(1,),
    n_layers=2,
    units=400,
    activation="elu",
    dist="symlog_twohot",
    num_buckets=255,
    symlog_range=10.0,
)
```

| `dist` | Return type | Use case |
|---|---|---|
| `"normal"` | `Independent(Normal)` | Reward prediction, value function |
| `"binary"` | `Independent(Bernoulli)` | Discount / termination prediction |
| `"symlog_twohot"` | `_TwoHotDistribution` | DreamerV2 reward/value |
| `"none"` | Raw tensor | Custom downstream processing |

### `ActionDecoder` — Dreamer policy head

```python
from simulusion import ActionDecoder

actor = ActionDecoder(
    action_size=6,
    stoch_size=30,
    deter_size=200,
    n_layers=2,
    units=400,
    activation="elu",
    min_std=1e-4,
    init_std=5,
    mean_scale=5,
)

features = torch.randn(4, 230)
action = actor(features)  # stochastic sample
action = actor(features, deter=True)  # deterministic mode
```

Outputs a Gaussian distribution squashed through `TanhBijector` to `[-1, 1]`.
The deterministic mode (`deter=True`) returns the distribution mode for
deployment; the stochastic mode is used during training.

## Quantization

### `VectorQuantizer` and `VectorQuantizerEMA`

```python
from simulusion import VectorQuantizer, VectorQuantizerEMA

# Standard VQ (gradient-based codebook updates)
vq = VectorQuantizer(vocab_size=512, embedding_dim=64, commitment_weight=0.25)

# EMA VQ (more stable codebook learning)
vq = VectorQuantizerEMA(
    vocab_size=512,
    embedding_dim=64,
    commitment_weight=0.25,
    ema_decay=0.99,
)

z = torch.randn(4, 64, 8, 8)  # (B, C, H, W)
z_q, indices, loss_dict = vq(z)
#   z_q:      (4, 64, 8, 8)  quantized embeddings
#   indices:  (4, 8, 8)      codebook indices per spatial location
#   loss_dict: {"vq_loss": tensor, "perplexity": tensor}
```

Both layers implement the same interface. The EMA variant updates codebook
vectors via exponential moving average of encoder outputs rather than gradient
descent, which typically produces higher codebook utilization.

## Video tokenization

### `VideoTokenizer` — Genie-style VQ-VAE

```python
from simulusion import VideoTokenizer

tokenizer = VideoTokenizer(
    num_frames=16,
    image_size=64,
    in_channels=3,
    encoder_dim=512,
    decoder_dim=1024,
    encoder_depth=12,
    decoder_depth=20,
    num_heads=16,
    patch_size=4,
    vocab_size=1024,
    embedding_dim=32,
    use_ema=True,
    ema_decay=0.99,
)

video = torch.randn(2, 3, 16, 64, 64)  # (B, C, T, H, W)
reconstructed, indices, loss = tokenizer(video)
#   reconstructed:  (2, 3, 16, 64, 64)
#   indices:        (2, 16, 16, 16)  (T × H' × W')
#   loss: {"recon_loss": ..., "vq_loss": ..., "perplexity": ...}
```

Architecture:
1. Patch embedding `(B, C, T, H, W) → (B, T×P, encoder_dim)`
2. Encoder ST-Transformer
3. Per-frame vector quantization
4. Decoder ST-Transformer
5. Patch unembedding `→ (B, C, T, H, W)`

Key features: causal processing (each frame only uses previous frames),
spatiotemporal transformer instead of full 3D ViT, per-frame VQ with shared
codebook.

```python
# Encode to discrete tokens
z_q, indices, vq_loss = tokenizer.encode(video)

# Decode from indices (for training downstream models)
embeddings = tokenizer.decode_indices(indices)  # (B, T, 16, 16, 32)
recon = tokenizer.decode(z_q)

# Factory shortcut
from simulusion.vision.video_tokenizer import create_video_tokenizer

tokenizer = create_video_tokenizer(num_frames=16, image_size=64)
```

## Distribution utilities

### `TanhBijector`

Bijective tanh transform for squashing Gaussian actions to `[-1, 1]`. Used
internally by `ActionDecoder`.

```python
from torch.distributions import TransformedDistribution, Normal
from simulusion import TanhBijector

dist = TransformedDistribution(Normal(mean, std), TanhBijector())
action = dist.sample()  # bounded to [-1, 1]
```

### `SampleDist`

Wraps a distribution and approximates `mean`, `mode`, `entropy` via Monte
Carlo sampling (100 samples by default). Used internally when the analytic
form is unavailable (e.g., after tanh squashing).

## `_TwoHotDistribution`

DreamerV2's symlog two-hot encoding for scalar prediction. Internally
encodes targets into a categorical distribution over `num_buckets` evenly
spaced bins within `[-symlog_range, symlog_range]`, then decodes via symexp.

```python
from simulusion.vision.dreamer_decoder import _TwoHotDistribution

dist = _TwoHotDistribution(logits, num_buckets=255, symlog_range=10.0)
dist.log_prob(target)  # categorical cross-entropy in symlog space
dist.mean()  # expectation decoded via symexp
```

## ConvVAE

```python
from simulusion.vision.VAE.ConvVAE import ConvVAE

vae = ConvVAE(
    latent_dim=32,
    input_shape=(3, 64, 64),
)
recon, mu, logvar = vae(images)
loss = vae.loss_function(recon, images, mu, logvar)
```

A standalone convolutional VAE for representation learning and generative
modeling. Can be used as a building block or baseline.

## Which component to use

| Task | Encoder | Decoder |
|---|---|---|
| Dreamer world model | `ConvEncoder` | `ConvDecoder` + `DenseDecoder` |
| Dreamer policy | — | `ActionDecoder` |
| PlaNet world model | `CNNEncoder` | `CNNDecoder` |
| IRIS discrete AE | `IRISEncoder` | `IRISDecoder` |
| Genie video tokenizer | `VideoTokenizer` | (built-in) |
| Custom VQ-VAE | custom + `VectorQuantizer` | custom |
| Reward/value head | — | `DenseDecoder(dist="normal")` |
| Discount head | — | `DenseDecoder(dist="binary")` |

## See Also

- {doc}`modular_rssm_guide` — uses ConvEncoder, ConvDecoder, MLPEncoder, MLPDecoder as pluggable components
- {doc}`genie` — uses VideoTokenizer for video tokenization
- {doc}`iris` — uses IRISEncoder, IRISDecoder, VectorQuantizer, VectorQuantizerEMA
