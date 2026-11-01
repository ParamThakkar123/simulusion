# Loss Functions

The `simulusion.losses` module provides loss functions for the classic
World Models pipeline (VAE + MDNRNN). Both are importable from the top level:

```python
from simulusion.losses import conv_vae_loss_fn, gmm_loss
```

```{contents} Contents
:depth: 2
```

## Which loss to use

| Loss | Stage | Purpose |
|---|---|---|
| `conv_vae_loss_fn` | VAE training | Reconstruct observations in latent space |
| `gmm_loss` | MDNRNN training | Predict next latent state as a mixture of Gaussians |

These two losses are used sequentially in the three-stage World Models pipeline
(Ha & Schmidhuber, 2018):

1. **VAE** (`conv_vae_loss_fn`) — compress observations into a latent vector
2. **MDNRNN** (`gmm_loss`) — predict the next latent vector given actions
3. **Controller** — train a policy inside the learned latent dynamics

## `conv_vae_loss_fn`

Reconstruction + KL divergence for training a convolutional VAE:

```{math}
\mathcal{L} = \underbrace{\|\hat{x} - x\|^2}_{\text{MSE}} \;+\;
\underbrace{-\frac{1}{2} \sum \left(1 + 2\log\sigma - \mu^2 - \sigma^2\right)}_{\text{KL divergence}}
```

```python
from simulusion.losses import conv_vae_loss_fn

reconst, mu, logsigma = vae(images)
loss = conv_vae_loss_fn(reconst, images, mu, logsigma)
loss.backward()
```

| Parameter | Shape | Description |
|---|---|---|
| `reconst` | `(B, C, H, W)` | Reconstructed images from decoder |
| `x` | `(B, C, H, W)` | Original input images |
| `mu` | `(B, latent_dim)` | Mean of encoder's latent distribution |
| `logsigma` | `(B, latent_dim)` | Log variance of encoder's latent distribution |

Returns a scalar tensor. The KL term regularizes the latent distribution toward
a standard normal prior, and the MSE term drives accurate reconstruction.

## `gmm_loss`

Negative log-likelihood under a Gaussian Mixture Model, used to train the
MDNRNN's mixture predictions:

```{math}
p(x \mid \{\pi_k, \mu_k, \sigma_k\}) = \sum_{k} \pi_k \cdot \mathcal{N}(x \mid \mu_k, \sigma_k)
```

```python
from simulusion.losses import gmm_loss

# MDNRNN outputs mixture parameters for each timestep
latent_next_obs = targets  # (B, T, latent_dim)
mus = mdnrnn_output["mus"]  # (B, T, n_mixtures, latent_dim)
sigmas = mdnrnn_output["sigmas"]  # (B, T, n_mixtures, latent_dim)
logpi = mdnrnn_output["logpi"]  # (B, T, n_mixtures)

loss = gmm_loss(latent_next_obs, mus, sigmas, logpi)
```

| Parameter | Shape | Description |
|---|---|---|
| `latent_next_obs` | `(..., latent_dim)` | Target latent vectors |
| `mus` | `(..., n_mixtures, latent_dim)` | Per-mixture means |
| `sigmas` | `(..., n_mixtures, latent_dim)` | Per-mixture standard deviations |
| `logpi` | `(..., n_mixtures)` | Log mixture weights |
| `reduce` | `bool` | If True (default), returns mean over batch |

Returns a scalar tensor (mean negative log-likelihood) when `reduce=True`,
or a per-sample loss tensor when `reduce=False`.

### Numerical stability

The implementation uses the log-sum-exp trick internally:

```
max_log = max(log_pi_k + log N(x | mu_k, sigma_k))
log_prob = max_log + log(sum_k exp((log_pi_k + log N) - max_log))
```

This avoids underflow when mixture components are far from the target.

## Pipeline example

Both losses are used together in the complete World Models training script:

```python
from simulusion.losses import conv_vae_loss_fn, gmm_loss

# --- Stage 1: Train VAE ---
for images in dataloader:
    reconst, mu, logsigma = vae(images)
    loss = conv_vae_loss_fn(reconst, images, mu, logsigma)
    optimizer_vae.zero_grad()
    loss.backward()
    optimizer_vae.step()

# --- Stage 2: Train MDNRNN ---
for latent_sequences, actions in mdnrnn_loader:
    latent_next_obs = latent_sequences[:, 1:]
    mus, sigmas, logpi, _ = mdnrnn(latent_sequences[:, :-1], actions)
    loss = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    optimizer_mdnrnn.zero_grad()
    loss.backward()
    optimizer_mdnrnn.step()
```

## See Also

- {doc}`vision_guide` — encoders and decoders used with these losses
- {doc}`datasets_guide` — datasets used in the VAE + MDNRNN pipeline
- {doc}`world_models_guide` — full World Models (Ha & Schmidhuber) pipeline
