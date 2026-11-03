# DIAMOND

DIAMOND (**DI**ffusion **A**s a **M**odel **O**f e**N**vironment **D**reams) is Simulusion's diffusion-based world-model agent for pixel-control reinforcement learning. It learns a conditional image generator that predicts the next observation from recent frames and actions, then trains an actor-critic policy inside imagined rollouts from that learned simulator.

Use DIAMOND when you want to study model-based RL with pixel-space generation rather than latent-state prediction. Dreamer learns compact latent dynamics; IRIS predicts discrete visual tokens; DIAMOND keeps the environment model in observation space and uses a denoising diffusion model to synthesize future frames.

## See it in action

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/diamond_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: Breakout in the emulator. Right: the diffusion world model generating every frame from its own previous outputs, following the same actions.</figcaption>
</figure>
```

Trained from scratch in one to two hours on a 4 GB laptop GPU with
`python demos/diamond_demo.py train`, then recorded with `python demos/diamond_demo.py record`.
More clips are in the {ref}`gallery <gallery-diamond>`.

```{contents} Contents
```

## High-level architecture

```{mermaid}
graph TD
    O["Recent observation frames"] --> C["Conditioning stack"]
    A["Recent actions"] --> C
    C --> U["Conditional diffusion U-Net"]
    N["Noisy next frame"] --> U
    S["Noise level σ"] --> U
    U --> X["Denoised next observation"]
    X --> R["Reward / termination model"]
    R --> RT["Predicted reward and done"]
    X --> P["Actor-critic imagination"]
    RT --> P
```

The implementation is split into four main parts:

| Part | Simulusion object | Purpose |
|------|----------------|---------|
| Configuration | `DiamondConfig` | Stores Atari preprocessing, diffusion, reward model, actor-critic, optimization, and logging settings. |
| Environment path | `DiamondAtariWrapper`, `make_diamond_atari_env()` | Applies DIAMOND-compatible Atari preprocessing and returns resized RGB observations. |
| World model | `DiffusionUNet`, `EDMPreconditioner`, `EulerSampler` | Generates next observations with EDM-style preconditioning and fast Euler sampling. |
| Agent and replay | `DiamondAgent`, `ReplayBuffer`, `SequenceDataset` | Collects real experience, trains model components, and performs imagination rollouts. |

## Model components

### Conditional diffusion U-Net

`DiffusionUNet` takes a noisy target frame, a stack of conditioning frames, a sequence of actions, and a diffusion noise level. It concatenates the noisy frame with the conditioning frames along the channel dimension, embeds actions and the diffusion timestep, and injects the combined conditioning through adaptive group normalization in residual blocks.

The most important shape convention is that observations are RGB frames and the conditioning history length is controlled by `num_conditioning_frames`. With the default value of `4`, the U-Net input contains one noisy RGB frame plus four RGB conditioning frames.

### EDM preconditioning

DIAMOND uses EDM-style preconditioning to keep denoising numerically stable across a wide noise range. `EDMPreconditioner` wraps the U-Net with the standard skip, input, output, and noise coefficients derived from `sigma_data` and the current noise level. The config exposes the EDM noise schedule parameters:

| Field | Default | Meaning |
|-------|---------|---------|
| `sigma_data` | `0.5` | Assumed data standard deviation for EDM scaling. |
| `sigma_min` | `0.002` | Minimum sampling noise level. |
| `sigma_max` | `80.0` | Maximum sampling noise level. |
| `rho` | `7` | Controls spacing of the Karras noise schedule. |
| `p_mean` | `-0.4` | Mean for log-normal training noise sampling. |
| `p_std` | `1.2` | Standard deviation for log-normal training noise sampling. |

### Fast Euler sampling

`EulerSampler` generates imagined frames with a small number of denoising steps. The default `num_sampling_steps=3` favors fast policy training over photorealistic samples. Increase it when frame quality is more important than rollout throughput.

### Reward and termination model

The diffusion model predicts pixels only. DIAMOND separately trains `RewardTerminationModel` to predict rewards and episode endings from observation-action sequences. Following Appendix D of the paper, its convolutional trunk is a stack of residual blocks (GroupNorm → SiLU → 3×3 conv, stride 1) with 2×2 max-pool downsampling and **adaptive** group normalization for action conditioning, followed by an LSTM to handle partial observability. The number of residual blocks per stage is set by `reward_res_blocks` (default `2`, i.e. `[2, 2, 2, 2]`). The reward head is a 3-way classifier over `sign(reward) ∈ {-1, 0, +1}`. Before each imagined rollout the LSTM is burned in over the conditioning frames and actions. The actor-critic uses these predictions during imagined rollouts.

### Actor-critic in imagination

`ActorCriticNetwork` learns from trajectories generated by the diffusion world model and reward model. Its trunk mirrors the reward model — residual blocks (GroupNorm → SiLU → 3×3 conv, stride 1) with 2×2 max-pool downsampling and an LSTM — but takes a single frame with no action conditioning (plain group normalization), with `actor_res_blocks` residual blocks per stage (default `1`, i.e. `[1, 1, 1, 1]`). The rollout length is controlled by `imagination_horizon`, and returns use `discount_factor` and `lambda_returns`.

## Quick start

Create a small DIAMOND config and agent:

```python
from simulusion import DiamondConfig
from simulusion.training.train_diamond import DiamondAgent

config = DiamondConfig(
    preset="small",
    game="Breakout-v5",
    obs_size=64,
    seed=1,
)
agent = DiamondAgent(config)
```

For configuration files or dictionaries, use `from_config`:

```python
agent = DiamondAgent.from_config(
    "simulusion/configs/experiments/diamond.yaml",
    preset="small",
    game="Pong-v5",
)
```

## Training from the CLI

Use the unified Simulusion CLI with the starter experiment config:

```bash
simulusion train diamond --config simulusion/configs/experiments/diamond.yaml preset=small seed=1
```

You can also run the training module directly:

```bash
python -m simulusion.training.train_diamond --game Breakout-v5 --preset small
```

Add `--print-config` to inspect the composed config before starting a long run:

```bash
simulusion train diamond --config simulusion/configs/experiments/diamond.yaml preset=small --print-config
```

## Training loop

DIAMOND alternates between real environment interaction, model fitting, and imagined policy optimization:

```text
for each epoch:
    collect Atari transitions with the current policy
    store frames, actions, rewards, and done flags in replay
    train the diffusion model on next-frame denoising
    train the reward/termination model on sequence targets
    roll out imagined trajectories through the learned world model
    update actor and critic from imagined returns
```

This separation is important: if generated frames improve but policy returns do not, inspect the reward/termination model and imagined rollout stability in addition to image metrics.

## Presets

`DiamondConfig(preset=...)` applies a hardware-oriented architecture preset. Manual architecture fields are still available when `preset=None`.

| Preset | Typical use | Diffusion channels | Conditioning dim | LSTM dims |
|--------|-------------|--------------------|------------------|-----------|
| `small` | Local development, smoke tests, smaller GPUs | `[32, 32, 32, 32]` | `128` | `256` |
| `medium` | Default experiments | `[64, 64, 64, 64]` | `256` | `512` |
| `large` | Larger GPUs and higher-capacity runs | `[128, 128, 128, 128]` | `512` | `1024` |

## Important configuration fields

| Category | Fields |
|----------|--------|
| Environment | `game`, `seed`, `obs_size`, `frameskip`, `max_noop`, `terminate_on_life_loss`, `reward_clip` |
| Conditioning | `num_conditioning_frames` |
| Diffusion | `diffusion_channels`, `diffusion_res_blocks`, `diffusion_cond_dim`, `sampling_method`, `num_sampling_steps` |
| Reward model | `reward_channels`, `reward_res_blocks`, `reward_cond_dim`, `reward_lstm_dim`, `burn_in_length` |
| Actor-critic | `actor_channels`, `actor_res_blocks`, `actor_lstm_dim`, `imagination_horizon`, `entropy_weight`, `lambda_returns` |
| Optimization | `learning_rate`, `adam_epsilon`, `weight_decay_diffusion`, `weight_decay_reward`, `weight_decay_actor`, `use_amp` |
| Runtime | `device`, `batch_size`, `data_loader_num_workers`, `pin_memory`, `persistent_workers` |

## Atari preprocessing notes

`make_diamond_atari_env()` builds an Atari environment with frame skip, no-op starts, optional life-loss termination, reward clipping, and image resizing. `DiamondAtariWrapper.reset()` returns `(obs, info)`, while `step()` follows the legacy Gym four-tuple `(obs, reward, done, info)`. Observations are HWC `uint8` RGB frames; transpose or normalize them if you feed components outside the packaged DIAMOND agent.

## Evaluation and benchmarks

The benchmark CLI includes a `diamond` adapter, so you can evaluate a checkpoint alongside other world-model agents:

```bash
python examples/benchmark_run_and_report.py \
  --agent diamond \
  --env Breakout-v5 \
  --checkpoint checkpoints/diamond/breakout.pt \
  --episodes 10 \
  --out-dir results/diamond_breakout
```

For Atari 100k-style reporting, `diamond_config.py` includes the supported benchmark game list and random/human reference scores used for human-normalized returns.

## Troubleshooting

- **Generated frames are blurry or unstable:** increase `num_sampling_steps`, train the diffusion model longer, or try a larger preset.
- **Policy improves in imagination but not in the real environment:** check reward/termination calibration, episode termination handling, and whether imagined rollouts are too long for the current model quality.
- **Training is slow:** start with `preset="small"`, reduce `obs_size`, keep `num_sampling_steps` low, and enable AMP on CUDA.
- **Atari import errors:** install the Gym/Gymnasium Atari extras and ROM dependencies required by your environment backend.
