# Evaluation Guide

Simulusion provides a general-purpose evaluation package for measuring the quality of
generative world models. It implements the three metrics used in the DIAMOND paper
(Alonso et al., NeurIPS 2024, Appendix M):

- **FID** (Fréchet Inception Distance) — perceptual quality of individual generated frames
- **FVD** (Fréchet Video Distance) — temporal coherence of generated video clips
- **LPIPS** (Learned Perceptual Image Patch Similarity) — per-frame perceptual similarity

The metrics are model-agnostic: they work with any image or video generation pipeline,
not just DIAMOND.

---

## Quick Start

### Python API

```python
import torch
from simulusion.evals import FID, FVD, LPIPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your real and generated data
real_images = torch.randn(100, 3, 64, 64)  # [N, C, H, W] in [0, 1]
gen_images = torch.randn(100, 3, 64, 64)

# FID: compares distributions of real vs generated frames
fid = FID(device=device)
print(f"FID: {fid(real_images, gen_images):.2f}")

# LPIPS: pairwise perceptual similarity
lpips = LPIPS(device=device)
print(f"LPIPS: {lpips(real_images, gen_images):.4f}")

# FVD: requires video tensors [B, C, T, H, W]
real_videos = torch.randn(16, 3, 20, 64, 64)
gen_videos = torch.randn(16, 3, 20, 64, 64)

fvd = FVD(device=device)
print(f"FVD: {fvd(real_videos, gen_videos):.2f}")
```

### CLI

```bash
# Evaluate a DIAMOND world model
simulusion eval --model diamond --checkpoint path/to/model.pt --game Breakout-v5

# Specify which metrics to compute
simulusion eval --model diamond --checkpoint model.pt --metrics fid,lpips

# Record real and generated videos
simulusion eval --model diamond --checkpoint model.pt --record eval_video.mp4

# Full control
simulusion eval --model diamond --checkpoint model.pt \
    --game Breakout-v5 \
    --num-videos 512 \
    --trajectory-length 20 \
    --batch-size 32 \
    --seed 42 \
    --metrics fid,fvd,lpips \
    --record eval_video.mp4

# Linear-probe an I-JEPA checkpoint (frozen encoder; no FID/FVD involved)
simulusion eval --model jepa --checkpoint results/jepa/jepa_run-latest.pth.tar \
    --root-path /data/imagenet \
    --model-name vit_base \
    --output probe.json
```

I-JEPA is scored differently from the generative world models: there is no
rollout to compare against the environment, so `--model jepa` freezes the EMA
target-encoder and trains a linear head on its average-pooled patch tokens,
following Appendix A.2 of the paper. See {doc}`jepa` for the protocol and the
Python API (`simulusion.jepa_linear_probe`).

### Interactive Play

```bash
# Watch the agent play in the real environment
simulusion play --model diamond --checkpoint path/to/model.pt --game Breakout-v5

# Switch to dream mode (TAB) to watch the agent inside its imagination
# Press arrow keys / WASD to override the agent's actions
simulusion play --model diamond --checkpoint model.pt

# Record gameplay video
simulusion play --model diamond --checkpoint model.pt --record gameplay.mp4
```

---

## FID: Fréchet Inception Distance

### Definition

FID measures the distance between the distribution of real images and the distribution
of generated images in the feature space of a pretrained InceptionV3 network. Lower
scores indicate that the two distributions are more similar.

```{math}
\text{FID} = \|\mu_r - \mu_g\|^2 + \operatorname{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{\frac{1}{2}}\right)
```

where:

- {math}`\mu_r, \Sigma_r` are the mean and covariance of Inception features for real images
- {math}`\mu_g, \Sigma_g` are the mean and covariance of Inception features for generated images
- {math}`(\Sigma_r \Sigma_g)^{\frac{1}{2}}` is the matrix square root of the product

### Implementation

1. **Feature extraction**: images are resized to 299x299 and passed through InceptionV3
   truncated at the `Mixed_7c` layer, producing 2048-dimensional feature vectors.
2. **Statistics**: mean and covariance are computed from all feature vectors.
3. **Distance**: the Fréchet distance between the two Gaussians is computed using
   `scipy.linalg.sqrtm` with a small diagonal regularizer ({math}`\epsilon = 10^{-6}`)
   for numerical stability.

### Usage

```python
from simulusion.evals import FID

fid = FID(device=device, batch_size=64)

# real_images, gen_images: [N, C, H, W] float tensors in [0, 1]
score = fid(real_images, gen_images)
```

### Interpretation

| FID Score | Quality |
|-----------|---------|
| 0 | Perfect (identical distributions) |
| < 10 | Excellent — nearly indistinguishable |
| 10–50 | Good — visible differences but similar structure |
| 50–200 | Moderate — clear distribution mismatch |
| > 200 | Poor — very different distributions |

**Caveat**: FID is sensitive to the number of samples. Always compare scores computed
with the same number of images.

---

## FVD: Fréchet Video Distance

### Definition

FVD extends the Fréchet distance to video by comparing feature distributions from a
video recognition backbone (R3D-18 pretrained on Kinetics-400). It captures both
per-frame quality and temporal dynamics.

```{math}
\text{FVD} = \|\mu_r - \mu_g\|^2 + \operatorname{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{\frac{1}{2}}\right)
```

The formula is identical to FID, but the features come from a 3D convolutional network
that processes spatiotemporal volumes.

### Implementation

1. **Clip sampling**: input videos of arbitrary length are split into 16-frame clips
   (shorter videos are padded by repeating).
2. **Feature extraction**: each clip is resized to 112x112, normalized with Kinetics
   statistics, and passed through R3D-18 truncated before the final FC layer, producing
   512-dimensional feature vectors.
3. **Statistics and distance**: computed identically to FID.

### Usage

```python
from simulusion.evals import FVD

fvd = FVD(device=device, batch_size=16, clip_length=16)

# real_videos, gen_videos: [N, C, T, H, W] float tensors in [0, 1]
score = fvd(real_videos, gen_videos)
```

### Interpretation

| FVD Score | Quality |
|-----------|---------|
| 0 | Perfect |
| < 50 | Excellent temporal coherence |
| 50–300 | Good |
| 300–1000 | Moderate |
| > 1000 | Poor |

**Caveat**: FVD requires at least as many clips as feature dimensions (512) for a
stable covariance estimate. With very small sample sizes, the matrix square root may
become numerically unstable.

---

## LPIPS: Learned Perceptual Image Patch Similarity

### Definition

LPIPS measures the perceptual distance between pairs of images using deep features
from a pretrained VGG16 network. Unlike FID (which compares distributions), LPIPS
compares individual image pairs — it is a **reference metric**.

Lower scores indicate higher perceptual similarity. LPIPS correlates better with
human judgment than pixel-level metrics like MSE or PSNR.

```{math}
\text{LPIPS}(x, y) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot \left(f^l_{hw}(x) - f^l_{hw}(y)\right) \right\|_2^2
```

where {math}`f^l(x)` are features from layer {math}`l` of VGG16, and {math}`w_l` are
learned per-channel weights (uniform in our implementation).

### Implementation

1. **Feature extraction**: images are passed through VGG16-bn, extracting features
   from layers `relu1_2`, `relu2_2`, `relu3_3`, and `relu4_3`.
2. **Normalization**: inputs are scaled from [0, 1] to [-1, 1] (VGG16 training range).
3. **Distance**: for each layer, the squared L2 difference is spatially averaged and
   summed across channels. Layer scores are averaged to produce the final score.

### Usage

```python
from simulusion.evals import LPIPS

lpips = LPIPS(device=device, batch_size=64)

# images_a, images_b: [N, C, H, W] float tensors in [0, 1]
# Must have the same N — this is pairwise
score = lpips(images_a, images_b)
```

### Interpretation

| LPIPS Score | Perceptual Similarity |
|-------------|----------------------|
| 0 | Identical images |
| < 0.05 | Very similar |
| 0.05–0.2 | Noticeable differences |
| 0.2–0.5 | Clearly different |
| > 0.5 | Very different |

---

## Python API Reference

### ``simulusion.evals`` Package

```
simulusion/evals/
  __init__.py     # Exports FID, FVD, LPIPS
  fid.py          # Fréchet Inception Distance
  fvd.py          # Fréchet Video Distance
  lpips.py        # Learned Perceptual Image Patch Similarity
  diamond_utils.py  # DIAMOND-specific trajectory generation utilities
```

### Classes

| Class | Metric | Input Shape | Feature Backbone | Feature Dim |
|-------|--------|-------------|------------------|-------------|
| `FID` | Fréchet Inception Distance | `[N, C, H, W]` | InceptionV3 (Mixed_7c) | 2048 |
| `FVD` | Fréchet Video Distance | `[N, C, T, H, W]` | R3D-18 (avgpool) | 512 |
| `LPIPS` | Learned Perceptual Similarity | `[N, C, H, W]` (pairs) | VGG16-bn (4 layers) | — |

### Common Parameters

| Parameter | Type | Description |
|---|---|---|
| `device` | `torch.device` | CPU or CUDA device for feature extraction |
| `batch_size` | `int` | Number of images/clips processed at once. Lower if GPU memory is limited |

#### FVD-specific

| Parameter | Type | Default | Description |
|---|---|---|---|
| `clip_length` | `int` | 16 | Number of frames per video clip. R3D-18 was trained on 16-frame clips |

---

## Model-Specific Evaluation Scripts

### DIAMOND (`simulusion/inference/eval_diamond.py`)

The DIAMOND eval script provides a full evaluation pipeline:

1. **Load checkpoint**: reconstructs the diffusion model, EDM preconditioner, and
   Euler sampler from the saved checkpoint.
2. **Collect real trajectories**: runs a random-policy agent in the Atari environment
   to collect ground-truth frame sequences.
3. **Generate imagined trajectories**: conditioned on real frames + actions, the
   diffusion model autoregressively generates future frames using the EDM sampler.
4. **Compute metrics**: runs FID, FVD, and LPIPS comparing real vs generated frames.

### Extending to Other Models

To add evaluation for a new model:

1. Create `simulusion/inference/eval_<model>.py` with a `run_eval()` function matching the
   signature in {doc}`cli`.
2. Register it in `EVAL_MODULES` in `simulusion/cli.py`:

```python
EVAL_MODULES = {
    "diamond": "simulusion.inference.eval_diamond",
    "jepa": "simulusion.training.eval_jepa",
    "my_model": "simulusion.inference.eval_my_model",
}
```

   If the new model needs options the others do not, list them in
   `EVAL_MODEL_OPTIONS` so passing them to a different `--model` is an error
   rather than a silent no-op.

The `simulusion eval` command will then accept `--model my_model`.

### Trajectory Collection Utilities

The `simulusion.evals.diamond_utils` module provides helper functions for DIAMOND evaluation:

- `generate_trajectories()` — autoregressively generates frames from a diffusion model
  conditioned on real trajectory data.
- `collect_real_trajectories_from_env()` — collects ground-truth trajectories from
  an Atari environment using random actions.

These are DIAMOND-specific but serve as a template for other model types.

---

## Video Recording

Both `simulusion eval` and `simulusion play` support the `--record` flag to save videos.

### Eval Recording

```bash
simulusion eval --model diamond --checkpoint model.pt --record results.mp4
```

This saves two files:

- `results_real.mp4` — real environment trajectories
- `results_gen.mp4` — model-generated trajectories

### Play Recording

```bash
simulusion play --model diamond --checkpoint model.pt --record gameplay.mp4 --record-fps 20
```

Records the gameplay window in real-time. The recording includes the HUD overlay
(mode indicator, reward, action).

---

## Tips and Best Practices

### Stable Metric Computation

1. **Use enough samples**: FID and FVD estimate covariances which require
   {math}`N \gg D` (samples much greater than feature dimension).
   - FID: at least 10,000 images for stable estimates
   - FVD: at least 1,000 clips (each clip = 16 frames)
   - LPIPS: pairwise, so stable with fewer samples

2. **Match resolution**: all metrics resize inputs internally, but extreme
   mismatches between training and evaluation resolution can bias results.

3. **Use the same random seed** when comparing different models to control for
   sampling noise:
   ```bash
   simulusion eval --model diamond --checkpoint model_a.pt --seed 42
   simulusion eval --model diamond --checkpoint model_b.pt --seed 42
   ```

### Choosing Metrics

| If you want to measure... | Use |
|---|---|
| Perceptual quality of generated frames | FID |
| Temporal coherence of videos | FVD |
| Per-frame accuracy vs a ground-truth video | LPIPS |
| All three (DIAMOND paper standard) | `--metrics fid,fvd,lpips` |

### Interpreting Combined Results

- **Low FID + high FVD**: frames look good individually but lack temporal
  consistency (flickering, jittering).
- **High FID + low FVD**: frames are distorted but coherent across time (e.g.,
  all frames are blurry in the same way).
- **Low FID + low FVD + low LPIPS**: high-quality world model with good dynamics.
