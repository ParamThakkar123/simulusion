# Dreamer: Model-Based RL with Latent Dynamics

Dreamer is a model-based reinforcement learning algorithm that learns a latent dynamics model
from images and trains a behavior policy entirely in the latent space.

Based on papers:
- [Dreamer: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1912.01603) (DreamerV1, Hafner et al., 2019)
- [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193) (DreamerV2, Hafner et al., 2020)

## See it in action

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/dreamer_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: the real environment (DeepMind Control cartpole swing-up). Right: Dreamer's world model, given 5 real frames and then imagining the rest open loop from the policy's actions.</figcaption>
</figure>
```

Trained from scratch in one to two hours on a 4 GB laptop GPU with
`python demos/dreamer_demo.py train`, then recorded with `python demos/dreamer_demo.py record`.
More clips are in the {ref}`gallery <gallery-dreamer>`.

```{contents} Contents
:depth: 3
```

## Overview

Dreamer learns a world model from image observations, then trains an actor-critic
policy entirely in the imagination of that world model. No gradients flow from the
environment to the policy — the world model is the only bridge between real
experience and learned behavior.

The family has two major versions, documented individually below.

---

## DreamerV1

### Theory

#### Recurrent State-Space Model (RSSM) with Gaussian Latents

DreamerV1's RSSM maintains a hybrid state with two components:

**1. Deterministic state** `h_t` — a GRU hidden state that captures temporal
dependencies and deterministic transitions:

```{math}
h_t = \text{GRU}(h_{t-1}, [s_{t-1}, a_{t-1}])
```

**2. Stochastic state** `s_t` — a diagonal Gaussian latent variable with
`stoch_size` means and variances, representing uncertainty.

The model operates in two modes:

**Observe mode** (training — uses real observations):

```{math}
\text{Posterior: } s_t \sim q(s_t | h_t, \text{enc}(x_t))
```

**Imagine mode** (policy training — no observations):

```{math}
\text{Prior: } s_t \sim p(s_t | h_t)
```

#### World Model Loss

The complete world model objective (V1):

```{math}
\mathcal{L}_{\text{WM}} = \mathcal{L}_{\text{pred}} + \beta \cdot \mathcal{L}_{\text{KL}}
```

Where:

```{math}
\begin{aligned}
\mathcal{L}_{\text{pred}} &=
\underbrace{\|x_t - \hat{x}_t\|^2}_{\text{image reconstruction}}
+ \underbrace{\|r_t - \hat{r}_t\|^2}_{\text{reward prediction}} \\
\mathcal{L}_{\text{KL}} &=
D_{\text{KL}}\big(q(s_t | h_t, e_t) \;\|\; p(s_t | h_t)\big)
\end{aligned}
```

V1 applies a single KL coefficient `β` without balancing.

#### Actor-Critic in Imagination

DreamerV1 rolls out imagined trajectories using the prior dynamics and trains
actor-critic purely in latent space:

**Actor loss** (REINFORCE with baseline):

```{math}
\mathcal{L}_{\text{actor}} = -\sum_{t=1}^{T}
\log \pi(a_t | s_t) \cdot \text{sg}(G_t^\lambda - V(s_t))
+ \eta \cdot H[\pi(\cdot | s_t)]
```

**Critic loss**:

```{math}
\mathcal{L}_{\text{critic}} = \sum_{t=1}^{T} \|V(s_t) - G_t^\lambda\|^2
```

**Lambda return** (with fixed `γ`):

```{math}
G_t^\lambda = r_t + \gamma \cdot
\begin{cases}
(1 - \lambda) V(s_{t+1}) + \lambda G_{t+1}^\lambda & \text{if } t < T \\
V(s_T) & \text{if } t = T
\end{cases}
```

### Examples

```python
import simulusion

agent = simulusion.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=5_000_000,
)
agent.train()
```

Explicit V1 config:

```python
from simulusion import DreamerAgent, DreamerConfig

cfg = DreamerConfig()

# Select DreamerV1
cfg.algo = "Dreamerv1"

# Gaussian latent (V1 default)
cfg.stoch_size = 30  # diagonal Gaussian dimensions
cfg.deter_size = 200

# Environment
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.total_steps = 5_000_000

# KL (single coefficient)
cfg.kl_loss_coeff = 1.0

agent = DreamerAgent(cfg)
agent.train()
```

```bash
simulusion train dreamer --env dmc/walker-walk --algo Dreamerv1 --device cuda
```

---

## DreamerV2

### Theory

#### Recurrent State-Space Model (RSSM) with Categorical Latents

DreamerV2's RSSM maintains the same hybrid state structure as V1 but replaces
Gaussian latents with **discrete categorical latents**:

```{math}
h_t = \text{GRU}(h_{t-1}, [s_{t-1}, a_{t-1}])
```

**Stochastic state** `s_t` — a concatenation of `num_categories` one-hot
categorical distributions, each with `classes` categories:

```python
# V2: stack of categoricals (e.g., 32 classes × 32 categories)
self.stoch = torch.cat([one_hot(logits[i]) for i in range(num_categories)], dim=-1)
```

Default: 32 categories × 32 classes = 1024 total latent dimensions.
Discrete latents are better at representing multimodal posteriors and
are critical for handling aleatoric uncertainty in complex environments like Atari.

#### World Model Loss with KL Balancing

V2 introduces **KL balancing** — separate weighting for the prior and posterior
KL terms using stop-gradient (`sg`):

```{math}
\mathcal{L}_{\text{KL}} = \alpha \cdot D_{\text{KL}}[q \| \text{sg}(p)]
+ (1 - \alpha) \cdot D_{\text{KL}}[\text{sg}(q) \| p]
```

where `α` (default 0.8) weights the prior-following term higher. This prevents
the posterior from collapsing to a deterministic point mass.

**Free nats**: A threshold (default 3 nats) below which KL is not penalized.

The full world model objective (V2):

```{math}
\mathcal{L}_{\text{WM}} = \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{KL}}
```

```{math}
\mathcal{L}_{\text{pred}} =
\|x_t - \hat{x}_t\|^2
+ \|r_t - \hat{r}_t\|^2
+ \text{BCE}(\gamma_t, \hat{\gamma}_t)
```

#### Discount Head

V2 adds a learned **discount (termination) head** that predicts episode
continuation probability `γ̂_t` via binary cross-entropy:

```{math}
\mathcal{L}_{\text{disc}} = \text{BCE}(\gamma_t, \hat{\gamma}_t)
```

This is critical for Atari where episodes can end due to life loss, so the
discount factor must be learned rather than fixed.

#### Architecture Improvements

- **Layer normalization** in GRU and MLP layers for training stability
- **SiLU activations** replace ELU throughout
- **Two-hot reward encoding** replaces MSE: discretizes reward into 255 bins
  and predicts a softmax distribution over bins

#### Actor-Critic in Imagination

Same structure as V1 but uses the learned discount `γ̂_t` in λ-returns:

```{math}
G_t^\lambda = r_t + \hat{\gamma}_t \cdot
\begin{cases}
(1 - \lambda) V(s_{t+1}) + \lambda G_{t+1}^\lambda & \text{if } t < T \\
V(s_T) & \text{if } t = T
\end{cases}
```

### Examples

```python
import simulusion

agent = simulusion.create_model(
    "dreamer",
    env_backend="atari",
    env="PongNoFrameskip-v4",
    algo="Dreamerv2",
    total_steps=10_000_000,
)
agent.train()
```

Explicit V2 config:

```python
from simulusion import DreamerAgent, DreamerConfig

cfg = DreamerConfig()

# Select DreamerV2
cfg.algo = "Dreamerv2"

# Categorical latent (V2)
cfg.stoch_size = 32  # number of categorical classes per category
cfg.num_categories = 32  # number of categorical distributions
cfg.deter_size = 200

# Environment
cfg.env_backend = "atari"
cfg.env = "PongNoFrameskip-v4"
cfg.total_steps = 10_000_000

# KL balancing
cfg.kl_alpha = 0.8
cfg.free_nats = 3.0

# Discount (V2 uses learned termination)
cfg.discount = 0.997

agent = DreamerAgent(cfg)
agent.train()
```

```bash
simulusion train dreamer --env atari/PongNoFrameskip-v4 --algo Dreamerv2 --device cuda
```

---

## Differences Between DreamerV1 and DreamerV2

| Aspect | DreamerV1 | DreamerV2 |
|--------|-----------|-----------|
| **Latent type** | Gaussian (continuous) | One-hot categorical (discrete) |
| **Stochastic state** | `stoch_size` diagonal Gaussian | `num_categories` × `classes` categoricals |
| **KL formulation** | Single coefficient `β` | KL balancing with `α` weight + free nats |
| **Discount** | Fixed `γ` throughout episode | Learned termination predictor `γ̂_t` |
| **Reward loss** | MSE | Two-hot discretized cross-entropy |
| **Activations** | ELU | SiLU |
| **Normalization** | None in GRU/MLP | LayerNorm in GRU and MLP layers |
| **Atari performance** | ~40% human-normalized score | ~100% human-normalized score |
| **Key advantage** | Simpler, fewer hyperparameters | Better on complex/discrete environments |

### Categorical vs Gaussian Latents

V1 uses a diagonal Gaussian for the stochastic state. V2 uses a concatenation
of one-hot categorical distributions:

```python
# V1: single Gaussian
self.stoch = torch.distributions.Normal(mean, std)

# V2: stack of categoricals
self.stoch = torch.cat([one_hot(logits[i]) for i in range(num_categories)], dim=-1)
```

Discrete latents better capture multimodal posteriors (e.g., "the robot could
be at door A or door B") and are less prone to posterior collapse.

### KL Balancing

| Formulation | V1 | V2 |
|-------------|----|----|
| KL loss | `β · KL[q ‖ p]` | `α · KL[q ‖ sg(p)] + (1-α) · KL[sg(q) ‖ p]` |
| Stop-gradient | None | On prior in first term, posterior in second |
| Effect | Single trade-off | Prior learns to follow posterior; posterior doesn't collapse |

### Discount Head

| | V1 | V2 |
|---|----|----|
| Discount | Fixed scalar `γ=0.99` | Learned `γ̂_t` from BCE loss |
| Purpose | Simple time discount | Model episode termination (life loss in Atari) |

---

## Shared Architecture

### High-level diagram

<div class="architecture-diagram" aria-label="Dreamer architecture diagram">
  <section class="diagram-section">
    <h3>World Model RSSM</h3>
    <div class="diagram-row">
      <span class="diagram-node info">Encoder CNN 64x64</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">GRU plus stochastic latent model</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Decoder transposed CNN</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Imagination Rollout</h3>
    <div class="diagram-row">
      <span class="diagram-node">State s0</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Action a0</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Imagined future states</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Lambda-return target</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Actor-Critic Learning</h3>
    <div class="diagram-row">
      <span class="diagram-node success">Actor policy</span>
      <span class="diagram-node success">Critic value model</span>
    </div>
  </section>
</div>

### Detailed architecture (RSSM)

```{mermaid}
graph TD
    A["Image x_t"] --> B["ConvEncoder"]
    B --> C["Obs embed e_t"]
    D["Prev state h_{t-1}, s_{t-1}"] --> E["GRU (deter)"]
    F["Prev action a_{t-1}"] --> E
    E --> G["h_t (deterministic)"]
    G --> H["Prior model p(s_t | h_t)"]
    H --> I["s_t (stochastic prior)"]
    C --> J["Posterior model q(s_t | h_t, e_t)"]
    G --> J
    J --> K["s_t (stochastic posterior)"]
    K --> L["ConvDecoder → x̂_t"]
    K --> M["Reward head → r̂_t"]
    K --> N["Discount head → γ̂_t (V2 only)"]
```

### Recurrent State-Space Model (RSSM)

The core of Dreamer is the RSSM, defined in `simulusion.models.dreamer_rssm.RSSM`.
It maintains a hybrid state with two components:

**1. Deterministic state** `h_t` — a GRU hidden state that captures temporal
dependencies and deterministic transitions:

```{math}
h_t = \text{GRU}(h_{t-1}, [s_{t-1}, a_{t-1}])
```

**2. Stochastic state** `s_t` — a latent variable representing uncertainty.
- **V1:** Diagonal Gaussian with `stoch_size` means and variances.
- **V2:** Concatenation of `num_categories` one-hot categoricals, each
  with `classes` categories. Default: 32 classes × 32 categories = 1024 total.

The model operates in two modes:

**Observe mode** (training — uses real observations):

```{math}
\text{Posterior: } s_t \sim q(s_t | h_t, \text{enc}(x_t))
```

**Imagine mode** (policy training — no observations):

```{math}
\text{Prior: } s_t \sim p(s_t | h_t)
```

Key insight: the prior learns to predict the posterior without seeing the
observation. During imagination, the prior serves as the dynamics model.

### CNN Encoder (`simulusion.vision.dreamer_encoder.ConvEncoder`)

Four-layer CNN with increasing channels (32 → 64 → 128 → 256) and ReLU
activations. Strided convolutions (stride 2) halve spatial resolution at
each layer. Output is flattened to `obs_embed_size` (default 1024).

```
Input:  (3, 64, 64)
  └─ Conv2D(3, 32, 4×4, stride 2) → (32, 31, 31)
  └─ Conv2D(32, 64, 4×4, stride 2) → (64, 14, 14)
  └─ Conv2D(64, 128, 4×4, stride 2) → (128, 6, 6)
  └─ Conv2D(128, 256, 4×4, stride 2) → (256, 2, 2)
  └─ Flatten → Linear(1024, embed_size)
Output: embed_size-d vector
```

### CNN Decoder (`simulusion.vision.dreamer_decoder.ConvDecoder`)

Mirrored transposed-CNN structure:

```
Input:  stoch + deter state (e.g. 1030-d)
  └─ Linear(1030, 1024) → reshape to (256, 2, 2)
  └─ ConvT2D(256, 128, 5×5, stride 2) → (128, 6, 6)
  └─ ConvT2D(128, 64, 5×5, stride 2) → (64, 14, 14)
  └─ ConvT2D(64, 32, 6×6, stride 2) → (32, 31, 31)
  └─ ConvT2D(32, 3, 6×6, stride 2) → (3, 64, 64)
Output: reconstructed image
```

### Reward and Discount Heads (`DenseDecoder`)

Two-layer MLPs with ELU activations predicting scalar reward, and in V2,
episode discount (termination probability). The discount head is trained with
binary cross-entropy:

```{math}
\mathcal{L}_{\text{disc}} = \text{BCE}(\gamma_t, \hat{\gamma}_t)
```

### Action Decoder (`ActionDecoder`)

Outputs the policy distribution over actions. For continuous actions, predicts
a tanh-squashed Gaussian. For discrete actions, predicts a categorical
distribution. Uses REINFORCE gradient through the world model.

## Shared Training

### Training Loop

DreamerAgent follows a cyclic training loop:

1. **Collect**: Interact with environment using current policy (+ exploration noise). Store experience in `ReplayBuffer`.
2. **Train world model** (every step): Sample batch of `batch_size` sequences of length `train_seq_len` from buffer. Update encoder, RSSM, decoder, reward head, and discount head.
3. **Train actor-critic** (every step after `seed_steps`): Imagine `imagine_horizon` steps using prior dynamics. Compute λ-returns. Update actor and critic.
4. **Log**: Metrics, video reconstructions, and checkpointing.

```{math}
\begin{aligned}
&\text{for each environment step:} \\
&\quad \text{collect } (x_t, a_t, r_t, \gamma_t) \\
&\quad \text{if } step > seed\_steps: \\
&\qquad \text{sample batch from buffer} \\
&\qquad \text{update world model (encoder, RSSM, decoder, reward)} \\
&\qquad \text{imagine } H \text{ steps} \\
&\qquad \text{update actor, critic} \\
&\quad \text{log every } log\_every \text{ steps}
\end{aligned}
```

## Usage in Simulusion

### Using config directly

```python
from simulusion import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.total_steps = 5_000_000

agent = DreamerAgent(cfg)
agent.train()
```

### Environment backends

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stoch_size` | 30 | Stochastic latent dimensions |
| `deter_size` | 200 | Deterministic hidden size |
| `embed_size` | 1024 | Encoder embedding size |
| `imagine_horizon` | 15 | Imagination rollout length |
| `discount` | 0.99 | Discount factor γ |
| `td_lambda` | 0.95 | λ-return parameter |
| `kl_loss_coeff` | 1.0 | KL divergence weight |

### Learning Objectives

**World Model Loss**:

```{math}
\begin{aligned}
\mathcal{L}_\mathrm{world}
&= \mathcal{L}_\mathrm{reconstruction}
 + \mathcal{L}_\mathrm{reward}
 + \beta \cdot \mathcal{L}_\mathrm{KL}
\end{aligned}
```

**Actor Loss** (REINFORCE):

```{math}
\mathcal{L}_\mathrm{actor}
= -\mathbb{E}\left[\log \pi(\mathbf{a} \mid \mathbf{s}) \cdot (G - V(\mathbf{s}))\right]
```

**Critic Loss** (MSE):

```{math}
\mathcal{L}_\mathrm{critic} = \mathbb{E}[(G - V(\mathbf{s}))^2]
```

## DreamerV2 Enhancements

DreamerV2 introduces several improvements:

1. **Discrete latents**: Categorical latent variables instead of Gaussian
2. **KL balancing**: Separate weighting for prior/posterior KL
3. **Discount model**: Learns to predict episode termination
4. **Layer normalization**: More stable training

## Configuration and Checkpoints

Dreamer configs are serializable, so experiments can be reproduced from the
YAML saved with each run or checkpoint:

```python
from simulusion import DreamerConfig, DreamerAgent

cfg = DreamerConfig()
cfg.env = "walker-walk"
cfg.to_yaml("configs/dreamer_walker.yaml")

agent = DreamerAgent.from_config("configs/dreamer_walker.yaml", seed=7)
agent.train()

# Checkpoints save `config.yaml` beside the weights automatically.
agent.dreamer.save("runs/walker/ckpts/model.pt")
restored = DreamerAgent.from_pretrained("runs/walker/ckpts")
print(restored.summary()["total_parameters"])
```

For lower-level workflows, the core `Dreamer` class also supports
`Dreamer.from_config(...)`, `Dreamer.from_pretrained(...)`,
`Dreamer.summary()`, and `Dreamer.parameter_count()`.

## Environment Support

Dreamer supports multiple backends:

```python
cfg = DreamerConfig()

cfg.env_backend = "dmc"  # DeepMind Control Suite
cfg.env = "walker-walk"

cfg.env_backend = "gym"  # Gym/Gymnasium
cfg.env = "Pendulum-v1"

cfg.env_backend = "dmlab"  # DeepMind Lab
cfg.env = "rooms_collect_good_objects_train"
cfg.dmlab_action_repeat = 4

cfg.env_backend = "mujoco"  # MuJoCo
cfg.env = "Humanoid-v4"

cfg.env_backend = "brax"  # JAX/Brax
cfg.env = "ant"

cfg.env_backend = "procgen"  # Procgen
cfg.env = "coinrun"

cfg.env_backend = "unity_mlagents"  # Unity ML-Agents
cfg.unity_file_name = "env.exe"
```

### CLI

```bash
simulusion train dreamer --env dmc/walker-walk --device cuda
```

See {doc}`configs_reference` for the full DreamerConfig field reference with defaults.

### Key Hyperparameters

#### World Model

| Parameter | V1 Default | V2 Default | Effect |
|-----------|------------|------------|--------|
| `stoch_size` | 30 | 32 × 32 classes | Total stochastic capacity |
| `deter_size` | 200 | 200 | GRU hidden size |
| `model_learning_rate` | 6e-4 | 3e-4 | World model learning rate |
| `train_seq_len` | 50 | 50 | Sequence length per batch |
| `batch_size` | 50 | 16 | Sequences per batch |
| `free_nats` | 3.0 | 3.0 | KL free-nats floor, per Dreamer v1 Appendix A. Raw KL below this contributes zero gradient, so the floor caps how hard the KL compresses the posterior. Single shared field: both variants read `DreamerConfig.free_nats` |
| `reward_scale` | 1.0 | 1.0 | Multiplier on replayed rewards. Raise/lower for envs outside DMC's `[0, 1]` per-step range |

#### Actor-Critic

| Parameter | V1 Default | V2 Default | Effect |
|-----------|------------|------------|--------|
| `actor_learning_rate` | 8e-5 | 8e-5 | Policy learning rate |
| `value_learning_rate` | 8e-5 | 8e-5 | Critic learning rate |
| `imagine_horizon` | 15 | 15 | Imagination rollout length |
| `discount` | 0.99 | 0.997 | Discount factor |
| `td_lambda` | 0.95 | 0.95 | λ-return parameter |
| `kl_loss_coeff` | 1.0 | 1.0 | KL loss coefficient |
| `kl_alpha` | — | 0.8 | KL balancing weight (V2 only) |

#### Environment Interaction

| Parameter | Default | Effect |
|-----------|---------|--------|
| `action_repeat` | 2 | Repeat each action N times |
| `action_noise` | 0.3 | Exploration noise std |
| `seed_steps` | 5000 | Random steps before training |
| `total_steps` | 5e6 | Total environment steps |
| `collect_steps` | 1000 | Steps between model updates |

## Common Pitfalls

### Posterior collapse

If the stochastic state is ignored by the dynamics, the model reduces to a
deterministic RNN. Symptoms: good reconstruction but imagination diverges.

**Fixes:**
- Decrease `kl_loss_coeff` or adjust `kl_alpha` (V2)
- Increase `free_nats`
- Reduce `stoch_size`

Both of the first two reduce the KL pressure that drives the posterior toward
the prior. Free nats exist precisely to prevent collapse: raw KL below the
floor is unpenalized, so raising `free_nats` lets the posterior carry more
information. Lowering it (or raising `kl_loss_coeff`) tightens the squeeze and
makes collapse worse.

### Imagination divergence

The prior predicts states that drift from realistic latents over long horizons.

**Fixes:**
- Keep `imagine_horizon` short (10–15)
- Verify multi-step prediction, not just one-step

### NaN loss during training

**Fixes:**
- Reduce `model_learning_rate` to 1e-4
- Tighten gradient clipping (default 100 → 10)
- Enable layer norm

### Actor never improves

**Fixes:**
- Increase `imagine_horizon` for delayed rewards
- Increase exploration noise via `action_noise`
- Verify critic loss is decreasing

## See Also

- {doc}`modular_rssm_guide` — swap Dreamer's encoder, backbone, or decoder for research
- {doc}`iris` — discrete world model alternative to Dreamer
- {doc}`diamond` — diffusion-based world model alternative

## References

- Hafner, D., Lillicrap, T., Fischer, I., Vuong, Q., Held, D., Haarnoja, T., & Abbeel, P. (2019). Dreamer: Learning Latent Dynamics for Planning from Pixels.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Mastering Atari with Discrete World Models.
- Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). Mastering Atari with Discrete World Models (DreamerV2). *ICLR 2021.*
