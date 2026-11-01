# DreamerV3: Mastering Diverse Domains with Fixed Hyperparameters

DreamerV3 is the third generation of the Dreamer family. Its distinguishing claim is
not a new architecture but *robustness*: a single set of hyperparameters trains
successfully across more than 150 tasks spanning continuous and discrete actions,
image and vector observations, dense and sparse rewards, and reward scales differing
by orders of magnitude.

Based on: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
(Hafner, Pasukonis, Ba & Lillicrap, 2023)

```{contents} Contents
:depth: 3
```

## Quick start

```python
import simulusion

agent = simulusion.create_model(
    "dreamer-v3",
    env="Pendulum-v1",
    env_backend="gym",
    total_steps=100_000,
)
agent.train()
```

Scaling up is a single argument — every other hyperparameter stays fixed:

```python
agent = simulusion.create_model("dreamer-v3", env="walker-walk", model_size="200m")
```

## What changed from DreamerV2

DreamerV2 already used discrete latents and two-hot heads. DreamerV3 adds the
robustness machinery that removes the need for per-domain tuning.

| Area | DreamerV2 (this repo) | DreamerV3 |
| --- | --- | --- |
| Latents | Gaussian (`mean`, `std`) | Categorical, 1% unimix, straight-through |
| KL terms | Single balanced KL | Separate `L_dyn` / `L_rep` with free bits |
| Actor objective | Backprop through dynamics | Reinforce + entropy bonus |
| Return scaling | None | Percentile range with a denominator limit |
| Critic | Two-hot on imagined returns | Adds EMA regularizer and a replay loss |
| Head init | Default | Zero-initialized output weights |
| Optimizer | Adam + norm clipping | LaProp + adaptive gradient clipping |
| Normalization | None | RMSNorm + SiLU throughout |
| Sequence model | Dense GRU | Block-diagonal GRU (8 blocks) |
| Replay | Rejects episode-spanning windows | `is_first` flags, an online queue, stored latent states |

## Theory

### World model

The world model is a Recurrent State-Space Model whose stochastic state is a vector
of categorical variables rather than a Gaussian:

```{math}
\begin{aligned}
\text{Sequence model:} &\quad h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{Encoder:} &\quad z_t \sim q_\phi(z_t \mid h_t, x_t) \\
\text{Dynamics predictor:} &\quad \hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t) \\
\text{Reward predictor:} &\quad \hat{r}_t \sim p_\phi(\hat{r}_t \mid h_t, z_t) \\
\text{Continue predictor:} &\quad \hat{c}_t \sim p_\phi(\hat{c}_t \mid h_t, z_t) \\
\text{Decoder:} &\quad \hat{x}_t \sim p_\phi(\hat{x}_t \mid h_t, z_t)
\end{aligned}
```

The model state is `s_t = {h_t, z_t}`. Samples of `z_t` use straight-through
gradients: the forward pass is a hard one-hot vector, the backward pass sees the
softmax probabilities.

### Free bits and the two KL terms

The world model loss splits the KL into a **dynamics** term (train the prior towards
the posterior) and a **representation** term (train the posterior towards the prior),
weighted differently and each clipped below one nat:

```{math}
\begin{aligned}
\mathcal{L}_{\text{dyn}} &= \max\big(1, \operatorname{KL}[\operatorname{sg}(q_\phi) \,\|\, p_\phi]\big) \\
\mathcal{L}_{\text{rep}} &= \max\big(1, \operatorname{KL}[q_\phi \,\|\, \operatorname{sg}(p_\phi)]\big) \\
\mathcal{L}(\phi) &= \beta_{\text{pred}} \mathcal{L}_{\text{pred}}
                   + \beta_{\text{dyn}} \mathcal{L}_{\text{dyn}}
                   + \beta_{\text{rep}} \mathcal{L}_{\text{rep}}
\end{aligned}
```

with `β_pred = 1`, `β_dyn = 1`, `β_rep = 0.1`.

This resolves a tension that previously forced per-domain tuning. Visually complex 3D
environments need a strong regularizer to keep representations predictable; games
where individual pixels matter need a weak one to preserve detail. Free bits switch
each KL term off once it is already minimized, so a single small `β_rep` works for
both.

Note that `L_dyn` and `L_rep` are numerically identical — `sg` changes gradients, not
values — so seeing them log the same number is expected, not a bug.

### Unimix

Every categorical (the encoder, the dynamics predictor, and a discrete policy) is a
mixture of 99% network output and 1% uniform. A distribution can therefore never
become deterministic, which bounds log-probabilities and keeps the KL terms finite.
This eliminates the KL spikes observed in earlier experiments.

### Two-hot reward and critic heads

Rewards and values vary by orders of magnitude across domains. A squared loss on
large targets can diverge; absolute and Huber losses stagnate; normalizing by running
statistics makes the optimization non-stationary.

Instead, the heads output logits over 255 bins placed at
`symexp(linspace(-20, +20, 255))` and are trained with cross entropy against the
two-hot encoding of the target. Predictions are read out as the probability-weighted
average of the bin locations, so any continuous value in the interval is
representable. Crucially, the loss depends only on the predicted probabilities, so
**gradient magnitudes are decoupled from target magnitudes**.

The `symlog` transform used for vector observations follows the same motivation:

```{math}
\operatorname{symlog}(x) = \operatorname{sign}(x)\ln(|x| + 1), \qquad
\operatorname{symexp}(x) = \operatorname{sign}(x)\big(\exp(|x|) - 1\big)
```

It compresses large magnitudes of either sign while approximating the identity near
the origin, so small targets are unaffected.

### Percentile return normalization

The actor uses a fixed entropy scale `η = 3e-4` across all domains. For that to work,
returns must be brought onto a common scale without destroying the information about
reward frequency that tells the agent whether to explore or exploit:

```{math}
S = \operatorname{EMA}\big(\operatorname{Per}(R^\lambda, 95) - \operatorname{Per}(R^\lambda, 5),\ 0.99\big)
```

Returns are divided by `max(1, S)`. Two details carry the robustness:

- **Percentiles, not min/max**, so a single outlier episode does not shrink every
  other return.
- **The denominator limit**, so returns that are already small pass through
  unchanged. Normalizing by standard deviation fails exactly here: under sparse
  rewards the deviation approaches zero and amplifies noise without bound.

### Actor and critic

The actor maximizes the Reinforce objective with an entropy bonus, for both discrete
and continuous actions:

```{math}
\mathcal{L}(\theta) = -\sum_{t=1}^{T}
  \operatorname{sg}\!\left(\frac{R^\lambda_t - v_\psi(s_t)}{\max(1, S)}\right)
  \log \pi_\theta(a_t \mid s_t)
  + \eta\, \mathcal{H}\!\left[\pi_\theta(a_t \mid s_t)\right]
```

The critic regresses `λ`-returns with the two-hot loss and adds two extras:

- an **EMA regularizer** pulling it towards an exponential moving average of its own
  weights (decay `0.98`), which stabilizes bootstrapping without a stale target
  network; and
- a **replay loss** (scale `0.3`) applying the critic loss to replayed trajectories,
  using the imagination returns at the rollout start states as on-policy value
  annotations. This helps where rewards are hard to predict from imagination alone.

Reward and critic output weights are zero-initialized so that both predict exactly
zero at step 0, which avoids a burst of hallucinated value early in training.

### Optimizer

LaProp normalizes gradients by an RMSProp second-moment estimate *before* applying
momentum, whereas Adam computes both from the raw gradient. That ordering tolerates
`ε = 1e-20` and avoids instabilities seen with Adam.

Adaptive gradient clipping scales each tensor's gradient so its norm stays under 30%
of the norm of the weights it belongs to. Because the threshold is relative to the
weights, it does not need retuning when loss functions or loss scales change.

## Model sizes

Widths derive from a single model dimension `d` (Table 3 of the paper). Larger models
score higher **and** need less environment interaction.

| Parameters | Hidden size `d` | Recurrent units | CNN channels | Codes per latent |
| --- | --- | --- | --- | --- |
| 12M | 256 | 1024 | 16 | 16 |
| 25M | 384 | 3072 | 24 | 24 |
| 50M | 512 | 4096 | 32 | 32 |
| 100M | 768 | 6144 | 48 | 48 |
| 200M | 1024 | 8192 | 64 | 64 |
| 400M | 1536 | 12288 | 96 | 96 |

```python
from simulusion.configs import DreamerV3Config

config = DreamerV3Config(model_size="200m")  # widths filled in automatically
config = DreamerV3Config(model_size="200m", hidden_size=896)  # override one field
```

The number of layers and the number of latents (32) are constant across sizes.

```{note}
The paper labels the recurrent column "8d", which holds for every row except 12M,
listed as 1024 rather than 2048. `MODEL_SIZES` reproduces the table verbatim.
```

## Configuration

`DreamerV3Config` extends `DreamerConfig`, so every environment backend option
carries over. The defaults below reproduce Table 4 and are meant to be used
unchanged.

| Parameter | Default | Description |
| --- | --- | --- |
| `model_size` | `"12m"` | Preset controlling every width |
| `latent_dim` | 32 | Number of categorical variables |
| `unimix` | 0.01 | Uniform mixture fraction |
| `free_nats` | 1.0 | Free bits threshold for both KL terms |
| `beta_pred` / `beta_dyn` / `beta_rep` | 1.0 / 1.0 / 0.1 | World model loss weights |
| `discount` | 0.997 | Discount factor (horizon 333) |
| `td_lambda` | 0.95 | `λ`-return trace decay |
| `imagine_horizon` | 15 | Imagination rollout length |
| `actor_entropy` | 3e-4 | Entropy regularizer scale |
| `critic_ema_decay` | 0.98 | Slow critic decay |
| `critic_replay_loss_scale` | 0.3 | Weight of the replay critic loss |
| `return_norm_limit` | 1.0 | Denominator floor |
| `num_buckets` | 255 | Two-hot bin count |
| `symlog_range` | 20.0 | Half-width of the bin grid in symlog space |
| `learning_rate` | 4e-5 | Shared across all three optimizers |
| `agc_clip` | 0.3 | Adaptive gradient clipping fraction |
| `batch_size` / `train_seq_len` | 16 / 64 | Minibatch shape |
| `replay_ratio` | 32 | Time steps trained per policy step collected |
| `buffer_size` | 5M | Replay capacity, lowered to fit available RAM |
| `online_fraction` | 1.0 | Share of each batch the online queue may fill |
| `replay_latents` | True | Store model states in replay and start sequences from them |
| `obs_keys` | `None` | Observations to encode; `None` picks `image`, else `state`, plus `instr` |
| `mlp_layers` / `reward_layers` / `cont_layers` | 3 / 1 / 1 | Hidden layers of the actor, critic and vector encoder/decoder; of the reward predictor; of the continue predictor |

### Replay ratio

`update_steps` is derived from `replay_ratio` rather than set directly:

The replay ratio counts policy steps, so action repeat is divided out
(`collect_steps` counts environment frames):

```
update_steps = round(collect_steps / action_repeat * replay_ratio / (batch_size * train_seq_len))
```

This reproduces the paper's example: a ratio of 32 with action repeat 4 and 16x64
batches is one gradient step per 128 frames. Higher replay ratios trade compute for
data efficiency, predictably. The paper picks the ratio per benchmark to fit its step
budget (Table 2: 32 for Atari, DMLab and Minecraft, 64 for ProcGen, 128 for
Atari100k, 512 for the control suites, 1024 for BSuite). The default of 32 is the
cheapest of these; use 512 to match the paper's control-suite runs. Set
`auto_update_steps=False` to control `update_steps` yourself.

### Replay layout

Replay holds one record per observation, in the indexing of the world model
equations. Record `t` stores `x_t`, the action `a_{t-1}` that led to it, the reward
`r_t` received on arriving at it, and `is_first` / `is_last` / `is_terminal`. The
reward and continue predictors at `s_t = (h_t, z_t)` are trained on `r_t` and
`c_t = 1 - is_terminal_t`, which `s_t` can predict because `h_t` has already seen
`a_{t-1}`. The last observation of each episode gets its own record, so the continue
predictor sees terminal states. In imagination, the reward and continuation for
acting in `s_t` are therefore read off at `s_{t+1}`.

Collection stores the model state `(h_t, z_t)` with each record. A replayed sequence
starts from the stored state of the record before it rather than from zeros, and
after each world-model update the fresh states of the trained sequences are written
back. Each minibatch is filled first from chunks the model has not trained on yet
(the online queue), then topped up uniformly. Collection also continues the open
episode across calls, so the environment is only reset when an episode ends.

## Observations and actions

Both observation kinds and both action kinds are handled automatically:

- **Image observations** (`(C, H, W)`) use a stride-2 convolution stack down to 4x4,
  a sigmoid decoder output, and targets in `[0, 1]`.
- **Vector observations** (`(D,)`) are symlog transformed and passed through an MLP,
  which prevents large inputs from producing reconstruction gradients that swamp the
  representation loss.
- **Several observations at once.** As in the paper, every image input is stacked
  along channels into one CNN, every vector input is concatenated into one MLP, and
  the two embeddings are concatenated. The decoder reconstructs each input, and its
  loss is the sum of the inputs' losses (logged per input as
  `wm/recon_loss/<key>`). Image inputs must share a spatial size.

`obs_keys` chooses which observations are encoded. Left at `None`, it takes
`image` if the environment has one, otherwise `state`, plus `instr` when present:

| Environment | Encoded by default |
| --- | --- |
| DMC, `dmc_obs_type="pixels"` (Visual Control) | `image` |
| DMC, `dmc_obs_type="state"` (Proprio Control) | `state` |
| DMLab `language_*` levels | `image`, `instr` |

Name the keys to encode something else, such as `obs_keys=["image", "state"]`.
Replay stores each encoded input in its own array (images as `uint8`, vectors as
`float32`), and checkpoints record the inputs so `from_pretrained` rebuilds the same
model. Evaluation videos use the `image` observation, or `env.render()` when there
is none.
- **Continuous actions** use a Normal with a `tanh`-squashed mean and a standard
  deviation bounded to `[0.1, 1.0]`.
- **Discrete actions** are wrapped as one-hot vectors by `make_env` and modeled with a
  one-hot categorical carrying the same 1% unimix.

`actor_dist` defaults to `"auto"`, which resolves from the environment's action space.

## Usage

### From the top-level API

```python
import simulusion

config = simulusion.create_config("dreamer-v3", env="walker-walk", seed=0)
agent = simulusion.create_model("dreamer-v3", config)
agent.train()
rewards, videos, latents = agent.evaluate()
```

### Direct construction

```python
from simulusion.models.dreamer_v3 import DreamerV3
from simulusion.configs import DreamerV3Config

model = DreamerV3(
    DreamerV3Config(actor_dist="normal"),
    obs_shape=(3, 64, 64),
    action_size=6,
    device="cuda",
)
model_loss, actor_loss, value_loss = model.train_one_batch()
```

### Checkpoints

```python
model.save("checkpoints/dreamer_v3.pt")  # writes config.yaml alongside
restored = DreamerV3.from_pretrained("checkpoints")
```

Checkpoints include the slow critic and the return normalizer state, so training
resumes exactly rather than restarting the normalization statistics.

### Components in isolation

Every piece is usable on its own:

```python
from simulusion.models import CategoricalRSSM
from simulusion.optim import LaProp, adaptive_grad_clip_
from simulusion.layers import BlockGRUCell
from simulusion.utils.dreamer_v3_utils import ReturnNormalizer, SymexpTwoHotDist
```

## Diagnostics

`train_one_batch` publishes a metrics dictionary that the training loop logs
automatically:

| Metric | What it tells you |
| --- | --- |
| `wm/recon_loss` | Whether the decoder is fitting observations |
| `wm/dyn_loss`, `wm/rep_loss` | KL magnitude; both pinned at 1.0 means free bits are saturated |
| `wm/reward_loss`, `wm/continue_loss` | Reward and termination prediction quality |
| `actor/entropy` | Exploration level; collapsing to 0 means premature exploitation |
| `actor/return_scale` | Current percentile range `S` |
| `actor/advantage` | Normalized advantage; should stay near unit scale |
| `critic/value`, `critic/target` | A persistent gap suggests the critic is lagging |
| `grad_norm/*` | Pre-clipping gradient norms per component |

## Common pitfalls

### Both KL terms sit exactly at 1.0

Free bits are fully saturated, so the representation is not being shaped by the
dynamics at all. This is normal early on. If it persists, the world model may be
underfitting — check `wm/recon_loss` is still decreasing.

### Entropy collapses to zero

The actor has stopped exploring. Check `actor/return_scale`: if it is very large, the
advantages have been scaled down until the entropy term dominates nothing. Confirm
that reward magnitudes are what you expect.

### Rewards are learned but the policy does not improve

Check `critic/value` against `critic/target`. A persistent gap points at the critic;
try raising `critic_replay_loss_scale` for environments where rewards are hard to
predict from imagination alone.

### Out-of-memory with large model sizes

The replay buffer sizes itself from available RAM and logs a warning when it shrinks.
The dominant GPU cost is `batch_size * train_seq_len`; lower `train_seq_len` before
lowering `model_size`, since sequence length costs less in final performance.

## Deviations from the paper

Documented here so results can be compared fairly:

- `buffer_size` is 5M as in the paper, but shrinks to fit the available RAM (each
  record also holds its stored model state).
- The default `replay_ratio` of 32 is the paper's Atari/DMLab setting; its control
  suite runs use 512 (see Replay ratio above).
- Replayed latent states are stored compactly: `h` as float16 and `z` as one class
  index per categorical.
- The critic replay loss uses the imagination return at each replay start state as
  the value annotation, then accumulates `λ`-returns over the replayed rewards. The
  paper describes this mechanism without fixing an implementation.
- Prioritized replay is not implemented; the paper also opts for uniform replay.
- Image reconstruction uses a sigmoid output against `[0, 1]` targets with a squared
  error, matching the Methods section's description of the decoder.

## See also

- {doc}`dreamer` — DreamerV1 and DreamerV2
- {doc}`modular_rssm_guide` — swapping encoders, backbones, and decoders
- {doc}`iris` — a discrete transformer-based world model alternative

## References

- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models.
- Ziyin, L., Wang, Z. T., & Ueda, M. (2020). LaProp: Separating Momentum and Adaptivity in Adam.
- Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization.
- Webber, J. B. W. (2012). A bi-symmetric log transformation for wide-range data.
