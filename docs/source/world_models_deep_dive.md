# World Models Deep Dive (Ha & Schmidhuber, 2018)

This page is a comprehensive technical reference for the Ha & Schmidhuber World Models
implementation in Simulusion. It covers the architecture, data pipeline, training details,
inference, configuration, and common pitfalls.

```{contents} Contents
:depth: 3
```

## Architecture Overview

The world model consists of three independently trained components:

```{mermaid}
graph LR
    A["Raw pixels 64×64×3"] --> B["V: ConvVAE encoder"]
    B --> C["Latent z (32-d)"]
    C --> D["M: MDN-RNN"]
    D --> E["Hidden h + predicted z"]
    C --> F["C: Linear controller"]
    E --> F
    F --> G["Action a"]
    G --> H["Environment"]
    H --> A
    A --> B
```

| Stage | Component | Function | Trained With | File |
|---|---|---|---|---|
| **V** | ConvVAE | Encodes 64×64 RGB → 32-d latent `z` | Reconstruction loss (MSE + KL) | `simulusion.vision.VAE.ConvVAE` |
| **M** | MDN-RNN | Predicts next latent as Gaussian mixture `p(zₜ₊₁|aₜ,zₜ,hₜ)` | GMM NLL + BCE + MSE | `simulusion.models.mdrnn` |
| **C** | Linear Controller | Maps `(zₜ, hₜ)` → action `aₜ` | CMA-ES (reward maximization) | `simulusion.models.controller` |

### Key design decisions

1. **Latent compression:** The VAE compresses 64×64×3 = 12,288 pixels into 32 floats.
   This makes the MDRNN's GMM output tractable and the controller's parameter count
   tiny (~10³), enabling black-box optimization with CMA-ES.

2. **Gaussian mixture output:** The MDRNN predicts the next latent as a mixture of
   Gaussians rather than a single Gaussian. This captures multimodal futures (e.g.,
   "turn left" vs. "turn right" from the same state).

3. **Hidden state is critical:** The controller receives both the latent `z` and the
   RNN hidden state `h`. The paper shows removing `h` drops CarRacing score from
   **906±21 → 632±251**, confirming that temporal memory is essential.

4. **CMA-ES over backprop:** The controller is trained with evolution strategies
   rather than gradient descent. This avoids differentiating through the environment
   or the world model, and works on sparse/delayed rewards.

---

## Stage 1: Vision — ConvVAE

### Model architecture

The ConvVAE follows a standard convolutional encoder-decoder structure:

```
Encoder:                                                 Decoder:
  3×64×64 input                                     32-d latent
  └─ Conv2D(3, 32, 4, stride=2)                     └─ Linear(32, 1024)
  └─ Conv2D(32, 64, 4, stride=2)                    └─ ConvTranspose2d(64, 64, 5, stride=2)
  └─ Conv2D(64, 128, 4, stride=2)                   └─ ConvTranspose2d(64, 32, 5, stride=2)
  └─ Conv2D(128, 256, 4, stride=2)                  └─ ConvTranspose2d(32, 32, 6, stride=2)
  └─ Flatten → Linear(1024, 2×latent_size)          └─ ConvTranspose2d(32, 3, 6, stride=2)
  └─ Returns (mu, logsigma)                            └─ Sigmoid → 3×64×64 output
```

Key classes in `simulusion.vision.VAE.ConvVAE`:

| Class | Purpose |
|---|---|
| `ConvVAEEncoder` | Encodes images → `(mu, logsigma)` |
| `ConvVAEDecoder` | Decodes latent → reconstructed image |
| `ConvVAE` | Combines encoder + decoder, exposes `forward(x)` → `(recon, mu, logsigma)` |

### Loss function

Defined in `simulusion.losses.convae_loss`:

```{math}
\mathcal{L}_{\text{VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{MSE reconstruction}}
  - \frac{1}{2}\sum_{i=1}^{d}\left(1 + 2\log\sigma_i - \mu_i^2 - e^{2\log\sigma_i}\right)
```

| Term | Description |
|---|---|
| **MSE term** | Measures pixel-level reconstruction quality |
| **KL term** | Regularizes the latent distribution toward a standard normal prior. When `μ=0` and `logσ=0`, the KL term is exactly zero and the loss equals the reconstruction loss alone |
| **`size_average=False`** | MSE is summed (not averaged) over pixels — intentional, since the KL term is also summed over latent dimensions, keeping both terms on similar scales |

### Data pipeline

**Training data** is collected by running the environment with random actions:

```python
from simulusion.training.train_world_model import generate_rollouts

generate_rollouts(
    data_dir="./data/carracing",
    env_name="CarRacing-v2",
    num_rollouts=1000,
    seq_len=1000,
    num_workers=8,
)
```

Each rollout is saved as a `.npz` file containing:
- `observations`: `(T, 64, 64, 3)` uint8 array
- `actions`: `(T, action_size)` float32 array
- `rewards`: `(T,)` float32 array
- `terminals`: `(T,)` float32 array

**Dataset classes** for VAE training:

| Dataset | Purpose |
|---|---|
| `ObservationDataset` | Returns individual frames (not sequences). Used by `train_convvae.py`. Extends `RolloutDataset`, overrides `_get_data()` to return only the observation tensor |
| `RolloutDataset` | Base class that loads `.npz` files, manages a circular buffer of open files, and splits files into train/test sets. Each sample returns `dict(observation, action, reward, terminal)` |
| `SequenceDataset` | Used for MDRNN training with raw images (when precomputed latents are disabled). Returns sequences of observations, actions, rewards, terminals, and next observations |
| `LatentSequenceDataset` | Used for MDRNN training with precomputed latents. Operates on pre-encoded numpy arrays rather than raw images, reducing memory and avoiding repeated VAE encoding |

### Train/test split

The `num_test_files` parameter controls how many `.npz` files are reserved for
validation:

```python
dataset = ObservationDataset(
    root="./data",
    train=True,
    num_test_files=600,  # last 600 files → test set
)
```

When `len(files) > num_test_files` and `num_test_files > 0`, the last
`num_test_files` files are used for test. If there aren't enough files, all
data goes to training.

```{warning}
Passing `num_test_files=0` was historically broken: `files[:-0]` evaluates to
`files[:0]` (empty list) in Python due to `-0 == 0`. This was fixed by guarding
the split with `num_test_files > 0`.
```

### Training loop

See `simulusion.training.train_convvae.train_convae()`:

1. Load pretrained VAE checkpoint if available (`noreload=False`).
2. Create `ObservationDataset` with `A.Compose` transforms (resize + optional flip).
3. Train for `num_epochs` using `Adam(..., lr=learning_rate)`.
4. Validate after each epoch.
5. Reduce LR on plateau via `ReduceLROnPlateau`.
6. Early stop via `EarlyStopping`.
7. Save best checkpoint as `best.tar`, current as `checkpoint.tar`.
8. Generate sample images every `sample_interval` epochs.

```{note}
The VAE is compiled with `torch.compile` when CUDA is available for faster
training. This can be disabled if compatibility issues arise.
```

---

## Stage 2: Memory — MDN-RNN

### Why a Gaussian mixture?

A single Gaussian assumes the next latent follows a unimodal distribution, but
many environments are fundamentally multimodal. From the same state, different
actions lead to different futures, and even the same action may have stochastic
outcomes. The Mixture Density Network (MDN) handles this by predicting:

```{math}
p(z_{t+1} | a_t, z_t, h_t) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mu_k, \sigma_k)
```

where `K` is the number of Gaussian components (typically 5), `π_k` are the
mixture weights, and `(μ_k, σ_k)` are the component parameters.

### MDRNN vs MDRNNCell

The module `simulusion.models.mdrnn` provides two variants:

| Class | RNN Type | Use Case | Forward Signature |
|---|---|---|---|
| `MDRNN` | `nn.LSTM` | Training — processes full sequences at once | `(actions, latents)` → `(mus, sigmas, logpi, rs, ds)` |
| `MDRNNCell` | `nn.LSTMCell` | Inference — one step at a time | `(action, latent, hidden)` → `(mus, sigmas, logpi, r, d, next_hidden)` |

Both share the same `_MDRNNBase` parent, which defines the `gmm_linear` output
layer. The output head maps the RNN hidden state to the GMM parameters:

```python
self.gmm_linear = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)
```

This produces:
- `gaussians × latents` means (mus)
- `gaussians × latents` sigmas (raw, then exponentiated)
- `gaussians` logits (softmax → logpi)
- 1 reward logit
- 1 terminal logit

### Weight transfer for inference

Training uses `MDRNN` (batched LSTM). Inference uses `MDRNNCell` (single-step
LSTMCell). The weights must be copied between the two:

```python
batch_rnn = MDRNN(latents=32, actions=3, hiddens=256, gaussians=5)
batch_rnn.load_state_dict(torch.load("mdrnn_best.tar")["state_dict"])

cell_rnn = MDRNNCell(latents=32, actions=3, hiddens=256, gaussians=5)
cell_rnn.rnn.weight_ih.data.copy_(batch_rnn.rnn.weight_ih_l0.data)
cell_rnn.rnn.weight_hh.data.copy_(batch_rnn.rnn.weight_hh_l0.data)
cell_rnn.rnn.bias_ih.data.copy_(batch_rnn.rnn.bias_ih_l0.data)
cell_rnn.rnn.bias_hh.data.copy_(batch_rnn.rnn.bias_hh_l0.data)
cell_rnn.gmm_linear.load_state_dict(batch_rnn.gmm_linear.state_dict())
```

The LSTM's `weight_ih_l0` maps to LSTMCell's `weight_ih`, `weight_hh_l0` to
`weight_hh`, and similarly for biases. The `gmm_linear` weights are shared
identically via `load_state_dict`.

```{important}
The initial hidden state for `MDRNNCell` must be created via `get_init_hidden()`
and updated on every step. A common bug is to reuse the same initial hidden
state, which causes the cell to "restart" from zeros each time, destroying
temporal memory.
```

### Loss function

Computed in `simulusion.training.train_mdn_rnn.get_loss()`:

```{math}
\mathcal{L}_{\text{MDRNN}} = \frac{1}{d+2}\left(
  \underbrace{\mathcal{L}_{\text{GMM}}(z_{t+1}, \mu, \sigma, \pi)}_{\text{next latent prediction}}
  + \underbrace{\text{BCE}(d_t, \hat{d}_t)}_{\text{terminal prediction}}
  + \underbrace{\text{MSE}(r_t, \hat{r}_t) \cdot \mathbb{1}_{\text{include\_reward}}}_{\text{reward prediction}}
\right)
```

| Component | Description |
|---|---|
| **GMM loss** (`simulusion.losses.gmm_loss`) | Negative log-likelihood of the observed next latent under the predicted Gaussian mixture. Uses numerically stable log-sum-exp: `\mathcal{L}_{\text{GMM}} = -\log\sum_k \pi_k \cdot \mathcal{N}(z_{t+1} \mid \mu_k, \sigma_k)` |
| **BCE loss** | Binary cross-entropy for terminal flag prediction |
| **MSE loss** | Mean squared error for reward prediction (only if `include_reward=True`) |
| **Scaling factor** | Total loss divided by `latent_size + 2` (or `latent_size + 1` if reward excluded) to balance the GMM loss scale |

### Precomputed latents

To avoid encoding every batch through the VAE during MDRNN training (which is
memory-intensive), `train_mdn_rnn.py` supports precomputed latents:

1. **Precomputation:** `precompute_latents()` loads the trained VAE, encodes all
   observations in all rollouts, and saves the result as a single `.npz` file:
   ```python
   latent_data = np.load("data/carracing/latents/latents_32.npz")
   latent_data.keys()  # latents, actions, rewards, terminals
   ```

2. **Training with precomputed latents:** Uses `LatentSequenceDataset` which
   operates directly on the numpy arrays without any VAE encoding during training.

3. **Training without precomputed latents:** Uses `SequenceDataset` which returns
   raw image sequences. The `data_pass()` function encodes them on the fly via
   `to_latent()`. This is slower but requires less disk space.

---

## Stage 3: Controller — CMA-ES

### Linear controller

Defined in `simulusion.models.controller.Controller`:

```python
class Controller(nn.Module):
    def __init__(self, latent_size, hidden_size, action_size):
        self.fc = nn.Linear(latent_size + hidden_size, action_size)

    def forward(self, state):
        return self.fc(state)  # state = concat([z, h])
```

The controller is a single linear layer with no activation, bias included.
Input is the concatenation of the latent vector `z` and RNN hidden state `h`.
Output is the action vector (clamped to `[-1, 1]` by the environment wrapper).

Total parameters: `(latent_size + hidden_size + 1) × action_size`. For
CarRacing (32 + 256 + 1) × 3 = **867 parameters**.

### CMA-ES optimization

The controller is trained with Covariance Matrix Adaptation Evolution Strategy
(CMA-ES) via the `cma` package, not gradient descent:

1. **Initialize:** Start with a mean parameter vector (from random init) and
   a covariance matrix.

2. **Sample:** Generate a population of candidate parameter vectors from the
   current distribution.

3. **Evaluate:** For each candidate, run a full episode rollout in the
   environment and collect the total reward. This is done in parallel across
   multiple worker processes.

4. **Update:** The CMA-ES algorithm adjusts the mean and covariance toward
   regions that produced higher rewards.

5. **Repeat** until the target return is reached or convergence.

### Parallel evaluation

`train_controller.py` uses `torch.multiprocessing` for parallel rollout:

| Role | Responsibility |
|---|---|
| **Master process** | Runs CMA-ES, dispatches parameter vectors to workers |
| **Worker processes (`slave_routine`)** | Each loads the trained VAE + MDRNNCell, creates an environment, and runs rollouts with the received controller parameters |
| **Weight transfer in workers** | Each worker converts the batched `MDRNN` checkpoint to an `MDRNNCell` for recurrent inference |

```python
def _run_rollout(ctrl_params, logdir, env_name, action_size, time_limit, device):
    # Load VAE and MDRNN checkpoints, convert to MDRNNCell
    # Create environment
    # Run episode with controller(h, z) → action → step → update h
    return total_reward
```

```{note}
Each worker has its own copy of the VAE and MDRNNCell on its assigned GPU.
The `torch.multiprocessing` spawn model ensures clean CUDA context per process.
The `flatten_parameters` / `load_parameters` utilities convert between the
Controller's `nn.Parameter` tensors and flat numpy arrays for CMA-ES.
```

### Sampling for robust evaluation

Each candidate controller is evaluated `n_samples` times (default 4) with
different random seeds, and the rewards are averaged. This reduces the variance
from stochastic environment dynamics and the VAE's random sampling (`z = μ +
σ · ε`).

---

## Inference Pipeline

The complete inference loop (from `test_trained_model()` in
`train_world_model.py`):

```python
# 1. Load models
vae = ConvVAE(img_channels=3, latent_size=32)
vae.load_state_dict(torch.load("vae/best.tar")["state_dict"])

cell_rnn = MDRNNCell(latents=32, actions=3, hiddens=256, gaussians=5)
# Transfer weights from trained MDRNN (see weight transfer section above)

ctrl = Controller(latent_size=32, hidden_size=256, action_size=3)
ctrl.load_state_dict(torch.load("ctrl/best.tar")["state_dict"])

# 2. Reset environment and hidden state
obs, _ = env.reset()
h, c = cell_rnn.get_init_hidden(1)  # fresh hidden state per episode

# 3. Rollout loop
for step in range(1000):
    # Encode observation → latent
    mu, logsigma = vae.encoder(preprocess(obs))
    z = mu + logsigma.exp() * torch.randn_like(logsigma)

    # Controller computes action from (hidden, latent)
    action = ctrl(h, z).cpu().numpy().flatten()

    # Step environment
    next_obs, reward, done, _ = env.step(action)

    # Update RNN hidden state with (action, latent)
    _, _, _, _, _, (h, c) = cell_rnn(tensor(action), z, (h, c))

    obs = next_obs
    if done:
        break
```

### Critical: hidden state management

The RNN hidden state `(h, c)` **must** be thread-tight through the loop:

```python
# ✅ CORRECT — h, c updated every step
for _ in range(steps):
    action = ctrl(h, z)
    _, _, _, _, _, (h, c) = cell_rnn(action, z, (h, c))
```

```python
# ❌ WRONG — h, c are never updated, model resets every step
for _ in range(steps):
    action = ctrl(h, z)
    cell_rnn(action, z, (h, c))  # return value discarded!
```

A less obvious variant of this bug is using a list comprehension, which also
fails to propagate state:

```python
# ❌ WRONG — h, c are not updated between comprehension iterations
h, c = cell_rnn.get_init_hidden(bs)
outs = [cell_rnn(action[t], latent[t], (h, c)) for t in range(seq_len)]
```

The return value `(h, c)` from each `cell_rnn` call is stored in `outs`, but
the outer `h, c` variables are never rebound. Every iteration passes the
same initial hidden state. This was a real bug encountered in the test suite.

---

## Configuration Reference

### WMVAEConfig

Fields marked with ✓ have defaults, fields marked with · are required.

| Field | Type | Default | Description |
|---|---|---|---|
| `height` | `int` | — | Input image height (pixels) |
| `width` | `int` | — | Input image width (pixels) |
| `latent_size` | `int` | `32` | Dimensionality of VAE latent space |
| `device` | `str` | `"cuda"` | Training device |
| `train_batch_size` | `int` | `32` | Samples per training batch |
| `num_epochs` | `int` | `10` | Number of training epochs |
| `data_dir` | `str` | `"./data"` | Path to rollout data (.npz files) |
| `learning_rate` | `float` | `1e-3` | Adam learning rate |
| `logdir` | `str` | `"results"` | Checkpoint and log directory |
| `noreload` | `bool` | `False` | Skip loading existing checkpoints |
| `nosamples` | `bool` | `False` | Skip saving sample images |
| `scheduler_patience` | `int` | `5` | LR scheduler patience (epochs) |
| `scheduler_factor` | `float` | `0.5` | LR multiplicative factor |
| `early_stopping_patience` | `int` | `30` | Early stopping patience |
| `sample_interval` | `int` | `5` | Epochs between sample saves |
| `extra` | `dict` | `{}` | Additional custom parameters |

### WMMDNRNNConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `latent_size` | `int` | `32` | Latent space dimensionality |
| `action_size` | `int` | `3` | Action space dimensionality |
| `hidden_size` | `int` | `256` | RNN hidden units |
| `gmm_components` | `int` | `5` | GMM mixture components |
| `device` | `str` | `"cuda"` | Training device |
| `batch_size` | `int` | `16` | Sequences per batch |
| `seq_len` | `int` | `32` | Sequence length per sample |
| `num_epochs` | `int` | `30` | Training epochs |
| `data_dir` | `str` | `"./data"` | Rollout data path |
| `learning_rate` | `float` | `1e-3` | RMSprop learning rate |
| `logdir` | `str` | `"results"` | Checkpoint directory |
| `noreload` | `bool` | `False` | Skip loading checkpoints |
| `include_reward` | `bool` | `True` | Include reward in loss |
| `scheduler_patience` | `int` | `5` | LR scheduler patience |
| `scheduler_factor` | `float` | `0.5` | LR multiplicative factor |
| `early_stopping_patience` | `int` | `30` | Early stopping patience |
| `extra` | `dict` | `{}` | Additional custom parameters |

### WMControllerConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `latent_size` | `int` | `32` | Latent space dimensionality |
| `hidden_size` | `int` | `200` | RNN hidden dimensionality |
| `action_size` | `int` | `3` | Action space dimensionality |
| `env_name` | `str` | `"CarRacing-v2"` | Environment for evaluation |
| `logdir` | `str` | `"results"` | Checkpoint directory |
| `n_samples` | `int` | `4` | Rollout samples per candidate |
| `pop_size` | `int` | `10` | CMA-ES population size |
| `target_return` | `float` | `950.0` | Stop when return ≥ target |
| `max_workers` | `int` | `32` | Max parallel workers |
| `display` | `bool` | `True` | Show progress bars |
| `time_limit` | `int` | `1000` | Max steps per episode |
| `extra` | `dict` | `{}` | Additional custom parameters |

---

## Loss Function Details

### conv_vae_loss_fn

```{math}
\mathcal{L} = \text{MSE}(x, \hat{x}) + \text{KL}(q(z|x) \parallel \mathcal{N}(0, I))
```

Where:

```{math}
\text{KL} = -\frac{1}{2}\sum_{i=1}^{d}\left(1 + 2\log\sigma_i - \mu_i^2 - \sigma_i^2\right)
```

```{note}
Both terms use **sum** reduction (not mean). With `latent_size=32` and
image size 64×64, the KL term sums 32 values, while MSE sums `3×64×64` values.
The reconstruction term therefore dominates numerically.
```

### gmm_loss

```{math}
\mathcal{L}_{\text{GMM}} = -\log\sum_{k=1}^{K} \pi_k \cdot \prod_{j=1}^{d}
  \frac{1}{\sqrt{2\pi}\sigma_{k,j}} \exp\left(-\frac{(x_j - \mu_{k,j})^2}{2\sigma_{k,j}^2}\right)
```

Implementation uses numerically stable log-space computation:

```python
normal_dist = Normal(mus, sigmas)
g_log_probs = logpi + normal_dist.log_prob(latent_next_obs).sum(dim=-1)
max_log_probs = g_log_probs.max(dim=-1, keepdim=True)[0]
g_log_probs = g_log_probs - max_log_probs  # stabilize
log_prob = max_log_probs.squeeze() + torch.log(torch.exp(g_log_probs).sum(dim=-1))
return -log_prob.mean()  # negative log-likelihood
```

The `max_log_probs` subtraction prevents numerical overflow in the exp sum.

---

## Testing Guide

The test suite covers all components with 83 tests across 8 files:

| Test file | Tests | What it covers |
|---|---|---|
| `tests/vision/test_convvae.py` | 10 | ConvVAE forward shapes, gradient flow, reconstruction range, latent size variants |
| `tests/models/test_mdrnn.py` | 16 | MDRNN/MDRNNCell shapes, differentiability, hidden state updates, weight transfer |
| `tests/models/test_controller_wm.py` | 5 | Controller shapes, inference modes, differentiability |
| `tests/configs/test_wm_config.py` | 11 | Config creation, defaults, validation, extra key proxying, serialization |
| `tests/losses/test_convae_loss.py` | 6 | Loss scalar, positivity, reconstruction ordering, differentiability, KL behavior |
| `tests/losses/test_gmm_loss.py` | 6 | Loss scalar, positivity, prediction ordering, differentiability, GMM variants |
| `tests/utils/test_train_utils.py` | 17 | EarlyStopping, ReduceLROnPlateau modes, state dict roundtrip |
| `tests/datasets/test_wm_dataset.py` | 12 | RolloutDataset, ObservationDataset, SequenceDataset, LatentSequenceDataset |

Run all tests:

```bash
python -m pytest tests/vision/test_convvae.py tests/models/test_mdrnn.py \
  tests/models/test_controller_wm.py tests/configs/test_wm_config.py \
  tests/losses/test_convae_loss.py tests/losses/test_gmm_loss.py \
  tests/utils/test_train_utils.py tests/datasets/test_wm_dataset.py -v
```

### Notable test patterns

**Testing differentiable gradient flow:**

```python
def test_differentiable(self, model):
    actions = torch.randn(seq_len, bs, 3, requires_grad=True)
    latents = torch.randn(seq_len, bs, 32, requires_grad=True)
    mus, sigmas, logpi, rs, ds = model(actions, latents)
    loss = mus.sum() + sigmas.sum() + logpi.sum() + rs.sum() + ds.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
```

**Testing weight transfer (MDRNN → MDRNNCell):**

```python
def test_weight_transfer_from_mdrnn(self):
    batch_rnn = MDRNN(latents=32, actions=3, hiddens=256, gaussians=5)
    cell_rnn = MDRNNCell(latents=32, actions=3, hiddens=256, gaussians=5)
    # Copy LSTM weights
    cell_rnn.rnn.weight_ih.data.copy_(batch_rnn.rnn.weight_ih_l0.data)
    cell_rnn.rnn.weight_hh.data.copy_(batch_rnn.rnn.weight_hh_l0.data)
    cell_rnn.rnn.bias_ih.data.copy_(batch_rnn.rnn.bias_ih_l0.data)
    cell_rnn.rnn.bias_hh.data.copy_(batch_rnn.rnn.bias_hh_l0.data)
    cell_rnn.gmm_linear.load_state_dict(batch_rnn.gmm_linear.state_dict())
    # Compare outputs step-by-step (MUST update hidden state!)
    h, c = cell_rnn.get_init_hidden(bs)
    cell_outs = []
    for t in range(seq_len):
        out = cell_rnn(actions[t], latents[t], (h, c))
        cell_outs.append(out)
        _, _, _, _, _, (h, c) = out  # ← critical: update h, c
    assert torch.allclose(cell_outs[t][0], mus_batch[t], atol=1e-4, rtol=1e-3)
```

**Testing non-leaf gradient (gmm_loss differentiable):**

When a loss function takes `logpi` that was computed with `.log_softmax()`, the
`logpi` tensor is not a leaf. To verify gradient flow, create and check a leaf:

```python
def test_differentiable(self):
    logpi_raw = torch.randn(2, 5, 5, requires_grad=True)
    logpi = logpi_raw.log_softmax(dim=-1)
    loss = gmm_loss(batch, mus, sigmas, logpi)
    loss.backward()
    assert logpi_raw.grad is not None  # ✅ leaf tensor
    # logpi.grad would be None ⚠️ (non-leaf)
```

---

## CLI Usage

The unified training script `simulusion.training.train_world_model` provides a
complete CLI:

```bash
# Generate data + train all 3 stages
python -m simulusion.training.train_world_model --env CarRacing-v2

# Train only specific stages
python -m simulusion.training.train_world_model --env CarRacing-v2 --stage vae
python -m simulusion.training.train_world_model --env CarRacing-v2 --stage rnn
python -m simulusion.training.train_world_model --env CarRacing-v2 --stage ctrl

# Generate rollouts only
python -m simulusion.training.train_world_model --env CarRacing-v2 --generate_only

# Test a trained model
python -m simulusion.training.train_world_model --env CarRacing-v2 --test

# With custom directories
python -m simulusion.training.train_world_model \
    --env CarRacing-v2 \
    --data_dir ./data/carracing \
    --logdir ./results/carracing \
    --latent_size 32 \
    --rnn_hidden 256 \
    --vae_epochs 50 \
    --rnn_epochs 30 \
    --ctrl_pop_size 16
```

---

## Common Pitfalls

### 1. Hidden state not updated in list comprehension

```python
# ❌ h, c stays at initial zeros for ALL timesteps
h, c = cell_rnn.get_init_hidden(bs)
outs = [cell_rnn(actions[t], latents[t], (h, c)) for t in range(seq_len)]
```

**Fix:** Use a for loop that explicitly rebinds `h, c`:

```python
h, c = cell_rnn.get_init_hidden(bs)
outs = []
for t in range(seq_len):
    out = cell_rnn(actions[t], latents[t], (h, c))
    outs.append(out)
    _, _, _, _, _, (h, c) = out
```

### 2. `files[:-0]` returns empty list

```python
# ❌ When num_test_files=0:
self.files = self.files[:-num_test_files]  # evaluates to self.files[:0] = []
```

**Fix:** Guard with `num_test_files > 0`.

### 3. Mismatched dataloader parallelism

`ObservationDataset` and `SequenceDataset` use a circular buffer pattern
(`load_next_buffer()`) that is incompatible with `num_workers > 0` in the
DataLoader. Always use `num_workers=0` for these datasets.

`LatentSequenceDataset` does not use a circular buffer and can use
`num_workers=4, pin_memory=True`.

### 4. VAE reconstruction + KL scale imbalance

With `size_average=False` on MSE, the reconstruction loss is summed over
`3 × 64 × 64 = 12,288` pixels while KL sums over `latent_size` = 32 dimensions.
The reconstruction term is ~384× larger. If you add a β-VAE weighting,
use `β = latent_size / (3 * height * width)` to balance the scales.

### 5. MDRNN loss scaling factor

The GMM loss operates on `latent_size` dimensions and produces values on the
order of `latent_size`. The total loss divides by `latent_size + 2` (or
`latent_size + 1` without reward) to keep the scale around 1. If you change
`latent_size`, the loss scale changes proportionally.

### 6. CMA-ES workers and GPU memory

Each worker process loads a full copy of the VAE and MDRNNCell. On GPU, this
can exhaust memory with many workers. The `max_workers` config caps this,
and workers are distributed across available GPUs.

---

## Extending the World Model

### Adding a new environment

1. Ensure the environment conforms to Gymnasium's API (`reset`, `step`,
   `action_space`, `observation_space`).
2. Set `env_name` in `WMControllerConfig`.
3. The `GymImageEnv` wrapper handles image resizing to 64×64 and uint8
   conversion.

### Changing the VAE architecture

Override the `ConvVAEEncoder` / `ConvVAEDecoder` classes. The encoder must
return `(mu, logsigma)` with `logsigma.shape == mu.shape`. The decoder
must return a tensor matching the input image shape.

### Changing the RNN cell

Replace `nn.LSTM` / `nn.LSTMCell` in `MDRNN.__init__` / `MDRNNCell.__init__`.
The hidden state `(h, c)` convention must be preserved for weight transfer.

### Adding a non-linear controller

Replace `nn.Linear` with an MLP in `Controller.__init__`. Note that CMA-ES
scales poorly with parameter count: an MLP with 2 hidden layers of 64 units
adds `(32+256)×64 + 64×64 + 64×3 ≈ 22,000` parameters vs. 867 for linear.
Consider using a smaller population size or switching to backprop-based
controller training for larger networks.

## See Also

- {doc}`modular_rssm_guide` — swappable components for custom world model architectures
- {doc}`vision_guide` — encoders, decoders, and tokenizers used in the pipeline
- {doc}`losses_guide` — conv_vae_loss_fn and gmm_loss used in VAE + MDNRNN training
