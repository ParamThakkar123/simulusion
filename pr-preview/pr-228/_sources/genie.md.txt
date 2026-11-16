# Genie: Generative Interactive Environment

Genie is a generative model trained from video-only data that can be used as an
interactive environment for reinforcement learning and decision-making tasks,
without requiring any action labels.

Based on paper: [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) (Bruce et al., 2024)

## See it in action

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/genie_replay.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: a real Sonic clip. Right: Genie generating the clip from its first frame alone, driven by latent actions it inferred without any action labels.</figcaption>
</figure>
```

Trained from scratch in one to two hours on a 4 GB laptop GPU with
`python demos/genie_demo.py train`, then recorded with `python demos/genie_demo.py record`.
More clips are in the {ref}`gallery <gallery-genie>`.

```{contents} Contents
:depth: 3
```

## Overview

Genie learns to understand world dynamics from unlabeled videos by learning:

1. **Video Tokenization**: Converts raw video frames into discrete tokens
2. **Latent Actions**: Infers the underlying actions that caused transitions between frames
3. **Dynamics Prediction**: Predicts future frames given past frames and latent actions

This enables agents to imagine and plan in a learned latent action space without
needing explicit action labels.

```{mermaid}
graph TD
    J["Video frames"] --> K["Video tokenizer"]
    K --> L["Video tokens"]
    M["Frame pairs (xₜ, xₜ₊₁)"] --> N["Latent action model"]
    N --> O["Latent action âₜ"]
    L --> P["Dynamics model"]
    O --> P
    P --> Q["Next video tokens"]
    Q --> R["Interactive generation"]
```

## Architecture

Genie consists of three main components trained end-to-end on video-only data:

### 1. Video Tokenizer

Converts raw video frames into discrete tokens using a VQ-VAE approach with
spatio-temporal downsampling:

```text
Input:  (3, 16, 64, 64) video clip
  └─ 3D convolutions (spatio-temporal downsampling)
  └─ VQ layer (codebook size: 1024)
Output: (16, 16, 16) discrete token grid
```

Total tokens per frame: `(64/4) × (64/4) = 16 × 16 = 256` tokens.

### 2. Latent Action Model (LAM)

Learns to infer discrete latent actions from frame-to-frame transitions
without any supervision:

```text
Input:  frame_t, frame_t+1
  └─ Encoder: process both frames
  └─ VQ layer: quantize to action token
Output: latent action index (e.g., {0, ..., 7})
```

**Training loss:**

```{math}
\mathcal{L}_{\text{LAM}} =
\underbrace{\|x_{t+1} - \hat{x}_{t+1}(x_t, \hat{a}_t)\|^2}_{\text{reconstruction}}
+ \underbrace{\|\text{sg}[z_e] - e_k\|^2}_{\text{codebook}}
+ \beta \cdot \underbrace{\|z_e - \text{sg}[e_k]\|^2}_{\text{commitment}}
```

The key insight: the action that best explains the frame transition is the one
that minimizes the reconstruction error of the next frame.

### 3. Dynamics Model

Transformer-based model that predicts future video tokens conditioned on past
tokens and latent actions:

```text
Input:  past video tokens + latent action
  └─ Transformer (causal masking)
  └─ Token prediction head
Output: next video tokens (as logits)
```

**Training loss**: Cross-entropy on predicted vs. actual tokens.

During generation, the dynamics model uses **MaskGIT** sampling — an iterative
refinement strategy that is faster than autoregressive decoding:

```python
# MaskGIT sampling (25 steps)
mask = all_masked
for step in range(maskgit_steps):
    logits = dynamics_model(tokens, mask, latent_action)
    tokens = sample_top_k(logits, mask)
    mask = update_mask(step)  # gradually unmask
```

## Training

### Training Losses

Genie is trained with multiple loss components:

1. **Tokenizer Loss**: VQ-VAE reconstruction loss for video tokenization
2. **Latent Action Loss**: VQ commitment loss + prediction loss for action learning
3. **Dynamics Loss**: Cross-entropy for token prediction with masking

```
Total Loss = L_tokenizer + λ₁·L_action + λ₂·L_dynamics
```

### Data Format

```
Dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
```

Each video should contain at least `num_frames` frames.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 8 | Number of frames per video |
| `image_size` | 32 | Input image size |
| `tokenizer_vocab_size` | 1024 | Video token vocabulary size |
| `action_vocab_size` | 8 | Latent action vocabulary size |
| `dynamics_dim` | 512 | Transformer hidden dimension |
| `dynamics_depth` | 8 | Number of transformer layers |
| `dynamics_num_heads` | 8 | Number of attention heads |
| `batch_size` | 4 | Training batch size |
| `learning_rate` | 3e-5 | Learning rate |
| `maskgit_steps` | 25 | Number of MaskGIT sampling steps |
| `warmup_steps` | 5000 | Learning rate warmup steps |
| `max_steps` | 125000 | Total training steps |

## Usage in Simulusion

### Quick start

```python
from simulusion import GenieConfig, create_genie_small

cfg = GenieConfig()
cfg.num_frames = 16
cfg.image_size = 64
cfg.epochs = 100

model = create_genie_small(num_frames=16, image_size=64)
```

### Generation

Generate new video frames from a prompt frame:

```python
prompt_frame = torch.randn(1, 3, 64, 64)
generated = model.generate(prompt_frame, num_frames=16)
```

### Interactive Play

Step through the environment using inferred or specified actions:

```python
current_frame = torch.randn(1, 3, 64, 64)
action = torch.tensor([3])  # Latent action index
next_frame = model.play(current_frame, action)
```

### Action Inference

Infer latent actions from real video frames:

```python
frames = torch.randn(1, 3, 16, 64, 64)
actions = model.infer_actions(frames)
```

### CLI

```bash
simulusion train genie --config path/to/genie_config.yaml
```

See {doc}`configs_reference` for the full GenieConfig field reference with defaults.

## Model Variants

| Variant | Params | Use Case |
|---------|--------|----------|
| `create_genie_small` | ~50M | Development, debugging |
| `create_genie_large` | ~11B | Production, research |

## Comparison: IRIS vs Genie

| Aspect | IRIS | Genie |
|--------|------|-------|
| Actions | Provided by environment (known) | Inferred from video (latent) |
| Tokenizer | Per-frame VQ-VAE | Spatio-temporal VQ-VAE |
| Tokens per frame | 16 | 256 (typically) |
| Dynamics | Autoregressive (GPT) | Autoregressive + MaskGIT |
| Policy | Actor-critic (REINFORCE) | N/A (interactive gen.) |
| Data requirement | ~100k env steps | ~50k+ videos |
| Use case | Model-based RL | Video world modeling |

## Related Components

Genie is built from several core components in this library:

- `VideoTokenizer`: Encodes video into discrete tokens
- `LatentActionModel`: Learns latent actions from frame pairs
- `DynamicsModel`: Transformer for future token prediction

## Advantages

1. **Video-only training**: No action labels required
2. **Interactive**: Can be used as a simulated environment
3. **Generalizable**: Learns from diverse video data
4. **Latent action space**: Enables efficient planning

## Common Pitfalls

### Codebook collapse

Most codebook entries go unused.

**Fixes:**
- Use EMA codebook updates (default in Genie)
- Lower commitment loss weight
- Increase codebook dimension

### Transformer memory

Sequence: 256 × 16 = 4096 tokens.

**Fixes:**
- Use gradient checkpointing
- Use sparse attention patterns

### Latent action disentanglement

The LAM might learn trivial actions.

**Fixes:**
- Increase action codebook size
- Add entropy regularization on action distribution

## See Also

- {doc}`vision_guide` — VideoTokenizer, VectorQuantizer, and ViT components
- {doc}`datasets_guide` — TinyWorlds datasets for Genie training
- {doc}`iris` — predecessor with known actions instead of latent actions

## References

- Bruce, J., et al. (2024). Genie: Generative Interactive Environments. *arXiv:2402.15391.*
- Van Den Oord, A., & Vinyals, O. (2017). Neural Discrete Representation Learning. *NeurIPS 2017.*
- Chang, H., et al. (2022). MaskGIT: Masked Generative Image Transformer. *CVPR 2022.*
