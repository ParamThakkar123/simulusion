# IRIS: Transformers for Sample-Efficient World Models

IRIS (Imagination with auto-Regression over an Inner Speech) is an implementation of the paper
"Transformers are Sample-Efficient World Models" (Micheli et al., 2023).

## See it in action

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/iris_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>The IRIS policy acting inside the Transformer's imagination of Pong, starting from a single real frame.</figcaption>
</figure>
```

Trained from scratch in one to two hours on a 4 GB laptop GPU with
`python demos/iris_demo.py train`, then recorded with `python demos/iris_demo.py record`.
More clips are in the {ref}`gallery <gallery-iris>`.

```{contents} Contents
:depth: 3
```

## Overview

IRIS achieves **human-level performance on Atari with only ~2 hours of gameplay** (100k environment steps)
by learning entirely in the imagination of a world model:

1. **Train world model** from real interactions
2. **Generate imagined trajectories** in the latent space
3. **Train policy** purely on imagined data

## Architecture

### High-level diagram

<div class="architecture-diagram" aria-label="IRIS architecture diagram">
  <section class="diagram-section">
    <h3>Discrete Autoencoder</h3>
    <div class="diagram-row">
      <span class="diagram-node info">Encoder CNN 64x64</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">VQ-VAE 512 vocab 16 tokens</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Decoder transposed CNN</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Autoregressive Transformer</h3>
    <div class="diagram-row">
      <span class="diagram-node">Latent tokens</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Action token</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Next latent tokens</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Reward and termination heads</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Actor-Critic in Imagination</h3>
    <div class="diagram-row">
      <span class="diagram-node success">Actor CNN and LSTM</span>
      <span class="diagram-node success">Critic CNN and LSTM</span>
    </div>
  </section>
</div>

### VQ-VAE: Discrete Autoencoder

Both IRIS and Genie use Vector Quantized Variational Autoencoders (VQ-VAE) to convert
continuous visual observations into discrete token sequences.

```{mermaid}
graph LR
    A["Image x"] --> B["CNN Encoder"]
    B --> C["Continuous z_e(x)"]
    C --> D["Vector Quantization"]
    E["Codebook {e_k}"] --> D
    D --> F["Discrete indices + z_q(x)"]
    F --> G["CNN Decoder"]
    G --> H["Reconstructed x̂"]
```

**Quantization:**

The encoder output `z_e(x)` is mapped to the nearest codebook vector:

```{math}
z_q(x) = e_k, \quad \text{where } k = \arg\min_j \|z_e(x) - e_j\|_2
```

**VQ-VAE Loss:**

```{math}
\mathcal{L}_{\text{VQ}} =
\underbrace{\|\hat{x} - x\|^2}_{\text{reconstruction}}
+ \underbrace{\|\text{sg}[z_e(x)] - e_k\|^2}_{\text{codebook loss}}
+ \beta \cdot \underbrace{\|z_e(x) - \text{sg}[e_k]\|^2}_{\text{commitment loss}}
```

IRIS uses EMA (Exponential Moving Average) for codebook updates instead
of the codebook loss, producing more stable training.

### Discrete Autoencoder Architecture

The encoder maps a 64×64 RGB frame to **16 tokens** from a **512-entry** codebook:

```
Input:  (3, 64, 64)
  └─ Conv2D(3, 64, 3, stride 2, pad 1)   → (64, 32, 32)
  └─ Conv2D(64, 128, 3, stride 2, pad 1) → (128, 16, 16)  + self-attention
  └─ Conv2D(128, 256, 3, stride 2, pad 1) → (256, 8, 8)   + self-attention
  └─ Conv2D(256, 512, 3, stride 2, pad 1) → (512, 4, 4)
  └─ ResBlocks + 1x1 projection to embedding dim
  └─ VQ layer over the 4×4 grid → 16 discrete indices
Output: 16 token indices (4 × 4, each ∈ {0, ..., 511})
```

### Transformer World Model

The transformer is a GPT-style autoregressive model:

```
Params:
  - vocab_size: 512 visual tokens (separate embedding table for the actions)
  - embed_dim: 256
  - num_layers: 10
  - num_heads: 4
  - seq_length: 20 timesteps × (16 tokens + 1 action) = 340 positions

Architecture:
  Token/Action Embedding → Positional Embedding → Causal Transformer Blocks
    → token head (next tokens) + reward head + termination head
```

Reward and termination are predicted by dedicated linear heads (read from the
action position), **not** encoded as extra tokens in the sequence.

**Input sequence format** — frame tokens and actions are interleaved with a
causal mask, and the next frame's tokens are generated autoregressively from the
action position onward:

```
[zₜ_0, zₜ_1, ..., zₜ_15, aₜ] → predict zₜ₊₁_0, then zₜ₊₁_1, ..., zₜ₊₁_15
                              (each conditioned on previously generated tokens)
plus, from the aₜ position:   predict reward rₜ and termination dₜ
```

### Actor-Critic

| Component | Purpose |
|---|---|
| **CNN + LSTM** | Processes reconstructed frames |
| **λ-returns** | Balances bias and variance in value estimation |
| **REINFORCE** | Policy gradient with baseline |
| **Entropy bonus** | Maintains exploration |

### Imagination Rollout

```python
# Imagine H steps: sample tokens autoregressively, decode to frames, feed to actor-critic
for h in range(imagination_horizon):
    tokens = transformer.generate(prev_tokens, action)
    frame = autoencoder.decode(tokens)  # decode to pixels
    action = actor(frame, hidden_state)  # policy
    reward = transformer.reward_head(tokens)  # predicted reward
    hidden_state = lstm(hidden_state, action, tokens)
```

## Training

### Staged training schedule

| Component | Start Epoch | Description |
|-----------|-------------|-------------|
| Autoencoder | 5 | Learn frame compression first |
| Transformer | 25 | Learn dynamics once tokens are good |
| Actor-Critic | 50 | Learn policy in imagination |

### Key Hyperparameters

| Parameter | Value |
|---|---|
| **Frame size** | 64×64 |
| **Tokens per frame** | 16 (from 512 vocabulary) |
| **Transformer sequence length** | 20 timesteps |
| **Imagination horizon** | 20 steps |
| **Discount (γ)** | 0.995 |
| **λ for λ-return** | 0.95 |

## Usage in Simulusion

### Quick start

```python
import torch
import simulusion

agent = simulusion.create_model(
    "iris",
    action_size=4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

### Using config directly

```python
from simulusion import IRISConfig

config = IRISConfig()

# Autoencoder
config.vocab_size = 512
config.tokens_per_frame = 16

# Transformer
config.transformer_layers = 10
config.transformer_embed_dim = 256

# Training
config.total_epochs = 600
config.env_steps_per_epoch = 200
config.env = "ALE/Pong-v5"
```

### CLI

```bash
simulusion train iris --env ALE/Pong-v5 --device cuda
```

For custom research code:

```bash
python -m simulusion.training.train_iris --game "ALE/Pong-v5"
```

See {doc}`configs_reference` for the full IRISConfig field reference with defaults.

## Benchmark Results

These are the numbers **reported in the paper** (Micheli et al., ICLR 2023,
Table 1), averaged over 5 seeds on 8× A100 40GB. They are the target this
implementation aims at, not measurements of this codebase — reproduce them
yourself before citing them as such.

To be comparable, use the number `train()` prints at the end: the mean over
`eval_episodes` episodes collected **after** training finishes (§3.2), averaged
over 5 seeds. The periodic evaluations printed during training are for
monitoring only — taking the best of them is the maximum of a noisy quantity and
reads higher than the agent actually is.

| Metric | IRIS (paper) | SPR | DrQ | CURL | SimPLe |
|--------|--------------|-----|-----|------|--------|
| Mean HNS | **1.046** | 0.616 | 0.465 | 0.261 | 0.332 |
| Superhuman games | **10/26** | 6/26 | 3/26 | 2/26 | 1/26 |

## Checkpoint compatibility

`IRISAgent.CHECKPOINT_FORMAT` is bumped whenever the module layout changes in a
way that makes older weights unmappable. `IRISAgent.load` detects a stale
checkpoint and raises, rather than failing with a list of missing keys.

| Format | Change |
|---|---|
| v1 → v2 | Transformer moved from `nn.TransformerEncoder` to GPT-2 blocks with per-layer key/value caches |
| v2 → v3 | Per-layer residual stacks in encoder/decoder, decoder widened to 64 channels, actor-critic conv block moved to conv + max-pool |
| v3 → v4 | Decoder self-attention at 8/16, attention blocks moved into an `attentions` ModuleDict, categorical reward head over {-1, 0, +1} |

Retrain, or check out the earlier revision to use an old checkpoint.

## Paper conformance

Every value in Appendix A Tables 2–6 is implemented as stated — 45/45 checked
programmatically — and `tests/models/test_iris_paper_alignment.py` pins the
structural details a numeric audit cannot catch: the actor-critic conv block's
op pattern, residual blocks per layer, self-attention resolutions, loss
weighting, the Freeway sampling temperature, and reward handling.

Where the paper leaves a choice open, the configuration exposes it:

- **Reward loss.** §2.2 permits "a mean-squared error loss or a cross-entropy
  loss for the reward predictor, depending on the reward function". Atari returns
  unbounded integer rewards, so the defaults are `reward_transform: sign` (making
  the target categorical over {-1, 0, +1}) with `reward_loss: cross_entropy`. Use
  `reward_transform: none` + `reward_loss: mse` for environments with meaningful
  continuous rewards.
- **Perceptual loss weights.** A.1 inherits VQGAN's LPIPS. The calibrated
  per-channel linear weights are fetched into the torch hub cache on first use;
  `perceptual_linear_weights` overrides the location. Without network access the
  loss falls back to uniform channel weights — still LPIPS in structure — and
  logs that it has done so.
- **Actor-critic channel widths.** A.3 fixes the layer pattern and LSTM hidden
  size but not the convolution channel counts; this implementation uses
  32 → 64 → 128 → 256.

## Using the components directly

Every piece is exported from the top-level package, so the world model can be
used without the Atari training loop:

```python
from simulusion import (
    IRISAgent,
    IRISConfig,
    IRISEncoder,
    IRISDecoder,
    IRISTransformer,
    IRISWorldModel,
    IRISReplayBuffer,
    LPIPSPerceptualLoss,
    build_perceptual_loss,
    compute_lambda_return,
)

agent = IRISAgent.from_config(IRISConfig(), action_size=6, device="cuda")

# Roll the world model forward without touching an environment.
trajectory = agent.imagine_rollout(frames, horizon=20, burn_in_frames=context)
```

`IRISTransformer` exposes `init_cache` / `prime_cache` / `generate_frame_cached`
for incremental generation, so it can drive imagination in your own loop.

## Training on other environments

`IRISTrainer` builds an Atari environment by default, but accepts any
Gymnasium-style environment with a discrete action space and image observations:

```python
trainer = IRISTrainer(game="MyTask", config=cfg, env=my_env)
```

Observations are resized to `frame_height` × `frame_width`; grayscale, HWC and
CHW inputs are all handled.

### Minecraft (MineRL / MineDojo)

```python
from simulusion.envs import make_minecraft_env
from simulusion.training.train_iris import IRISTrainer

env = make_minecraft_env("MineRLTreechop-v0")  # or backend="minedojo"
trainer = IRISTrainer(game="MineRLTreechop-v0", config=cfg, env=env)
```

MineRL's native action space is a `Dict` of nine binary keypresses plus a
continuous camera delta — 2⁹ × ℝ² — which a categorical policy cannot address.
`MinecraftDiscreteEnv` collapses it to `Discrete(13)` over a curated set covering
navigation, looking, and the two interaction verbs:

```
noop, forward, back, left, right, jump, forward_jump,
attack, use, camera_left, camera_right, camera_up, camera_down
```

Pass `action_set=` to substitute your own — the `Obtain*` tasks additionally need
craft/place/equip actions, and without them an agent cannot progress past what
tool-free play allows. Actions the task does not support become no-ops rather
than errors, so the same set works across Treechop and Navigate.

`minerl` and `minedojo` are **not** installable as Simulusion extras. MineRL 1.x
publishes no release compatible with Python 3.11+, and MineDojo pins
`gym==0.21.0`, whose sdist no longer builds under modern setuptools -- so
`pip install simulusion[minerl]` could only ever fail. Install them yourself in a
Python 3.10 environment alongside Simulusion:

```bash
# Python 3.10 environment, separate from the one Simulusion is developed in.
pip install simulusion
pip install "setuptools<66" wheel        # gym 0.21's sdist needs the old backend
pip install minerl                       # or: pip install minedojo
```

Both need a Java runtime and launch a real Minecraft client, so neither runs in a
headless container without a virtual display.

**Expect to retune.** Minecraft is far outside the Atari 100k regime these
defaults target: episodes are long, rewards are sparse, and each environment step
is orders of magnitude slower. The collection budget, `imagination_horizon`, and
`total_epochs` all need raising.

## Hardware and presets

Appendix G reports 8× A100 40GB, roughly 3.5 days per environment. Two presets
are provided:

| Preset | For | Notes |
|---|---|---|
| `configs/experiments/iris.yaml` | Reproduction | Paper's exact hyperparameters. Needs a large GPU. |
| `configs/experiments/iris_small_gpu.yaml` | Consumer GPUs (4–8GB) | Smaller batches and fewer gradient steps per epoch. Measured ~2.0GB peak on a 4GB card. Expect returns below the published numbers. |

The small-GPU preset keeps every *method* hyperparameter (tokens per frame,
vocabulary, imagination horizon, burn-in length, loss weights) at the paper's
values and reduces only batch sizes, transformer depth, and steps per epoch.

## Common Pitfalls

### Codebook collapse

Most codebook entries go unused, and `perplexity` — the effective codebook size
logged each epoch — trends toward 1. This is the quietest way for IRIS to fail:
the reconstruction loss keeps falling (the decoder learns to emit a constant),
the transformer keeps training, and the policy simply never receives a signal.

**Watch `perplexity`.** If it approaches 1, nothing downstream is meaningful.

Both quantizers re-seed dead codes onto real encoder outputs automatically
(`restart_dead_codes_after`, default 0.01 in units of mean assignments per
step). Set it to `0.0` to disable.

### Blank or double-normalized frames

The replay buffer stores `uint8`. Feeding it frames already scaled to `[0, 1]`
truncates every pixel to zero, and the symptoms mimic a healthy run: the
reconstruction loss converges to ~0 and the policy entropy sits at exactly
`ln(num_actions)`. Preprocessing returns `uint8`; conversion to float happens at
consumption time via `IRISTrainer.to_float_tensor`.

### Transformer memory

Sequence length: (16 tokens + 1 action) × 20 timesteps = 340 positions.

**Fixes:**
- Use gradient checkpointing (`gradient_checkpointing: true`)
- Reduce `transformer_timesteps`

### Slow autoregressive generation

Generating a frame costs K sequential steps. A cache-free implementation reruns
the whole prefix each time, which is O(K · L²) per imagined step and dominates
training time.

The transformer keeps per-layer key/value caches (`init_cache`, `prime_cache`,
`generate_frame_cached`), so each token is a single-position forward pass. If
you add a code path that rolls imagination forward, use those rather than
calling `forward` repeatedly.

## See Also

- {doc}`genie` — extends IRIS with latent actions and video-only training
- {doc}`dreamer` — continuous world model alternative to IRIS

## References

- Micheli, V., Alonso, E., & Fleuret, F. (2023). Transformers are Sample-Efficient World Models. *ICLR 2023.*
- Van Den Oord, A., & Vinyals, O. (2017). Neural Discrete Representation Learning. *NeurIPS 2017.*
