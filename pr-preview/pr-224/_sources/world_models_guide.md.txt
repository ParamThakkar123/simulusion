# World Models Study Guide

This page is a conceptual and practical map of the model families implemented in Simulusion. It is written as a study guide: start with the shared vocabulary, then compare the model families, then use the API reference for exact constructor and method details.

## What is a world model?

A **world model** is a learned simulator. It compresses observations into a latent state, predicts how that state changes after actions, and optionally predicts rewards, terminal flags, or future pixels/tokens. In reinforcement learning, this lets an agent train or plan in imagination instead of relying only on expensive environment interaction.

A common objective decomposes into reconstruction, dynamics, and task losses:

```{math}
\begin{aligned}
\mathcal{L}
&= \mathcal{L}_{\text{recon}}(x_t, \hat{x}_t) \\
&\quad + \beta\,D_{\mathrm{KL}}\!\left(q(z_t \mid x_{\le t}, a_{<t})\;\middle\|\;p(z_t \mid z_{<t}, a_{<t})\right) \\
&\quad + \mathcal{L}_{\text{reward}}
 + \mathcal{L}_{\text{policy}}.
\end{aligned}
```

The exact terms change by family: Dreamer and PlaNet emphasize latent state-space dynamics, IRIS and Genie emphasize discrete token dynamics, JEPA emphasizes representation prediction without pixel reconstruction, and DiT/DIAMOND emphasize diffusion-based generation.

```{mermaid}
graph LR
    A["Observation"] --> B["Encoder or tokenizer"]
    B --> C["Latent state"]
    D["Action"] --> E["Dynamics model"]
    C --> E
    E --> F["Predicted next latent state"]
    F --> G["Decoder reward and value heads"]
    G --> H["Planning policy learning or generation"]
```

## Quick model chooser

| If you want to study or build... | Start with | Core files |
|---|---|---|
| Latent-dynamics model-based RL from pixels | Dreamer | `simulusion.models.dreamer`, `simulusion.models.dreamer_rssm` |
| Classical latent planning with CEM-style imagination | PlaNet / RSSM | `simulusion.models.planet`, `simulusion.models.rssm` |
| Swappable encoders, decoders, and recurrent backbones | Modular RSSM | `simulusion.models.modular_rssm` |
| Self-supervised visual representations | JEPA + ViT | `simulusion.models.jepa_agent`, `simulusion.models.vit` |
| Sample-efficient Atari with discrete token imagination | IRIS | `simulusion.models.iris_agent`, `simulusion.models.iris_transformer` |
| Unsupervised controllable video-world modeling | Genie | `simulusion.models.genie`, `simulusion.models.latent_action_model`, `simulusion.models.dynamics_model` |
| Diffusion world models and image/video generation | DDPM, DiT, DIAMOND | `simulusion.models.diffusion.*` |

## Shared concepts

### Latent state

The latent state is the compact variable the model predicts through time. It may be continuous, discrete, or hybrid:

- **Continuous latents** are convenient for Gaussian losses and differentiable dynamics.
- **Discrete latents** are useful for token transformers, autoregressive modeling, and categorical planning.
- **Deterministic recurrent state** stores history that is not represented by the current stochastic latent alone.

### Dynamics model

A dynamics model predicts future state conditioned on action:

```{math}
p(z_{t+1} \mid z_t, h_t, a_t), \qquad h_{t+1}=f_\theta(h_t, z_t, a_t).
```

A good dynamics model must be accurate enough over multi-step rollouts, not just one-step predictions. This is why many algorithms mix teacher-forced training with imagined rollout losses or policy objectives.

### Imagination and planning

Once a model predicts futures, it can be used in two ways:

1. **Planning:** search over candidate action sequences and execute the first action.
2. **Policy learning:** train an actor and critic on imagined trajectories.

Dreamer is the canonical policy-learning example; PlaNet is the canonical planning example.

## Dreamer

Dreamer learns a Recurrent State-Space Model (RSSM), then trains an actor-critic in latent imagination. The library exposes high-level training through `DreamerAgent` and the core neural components through `Dreamer`, `RSSM`, Dreamer encoders/decoders, replay memory, and utilities.

### Mental model

```{mermaid}
graph TD
    A["Image observation"] --> B["Convolutional encoder"]
    B --> C["RSSM posterior"]
    D["Previous latent and action"] --> E["RSSM prior"]
    C --> F["Decoder reward and discount heads"]
    E --> G["Imagination rollout"]
    G --> H["Actor"]
    G --> I["Critic"]
    I --> J["Return targets"]
    J --> H
```

### What to study

- **Representation learning:** The encoder maps image observations into embeddings for the RSSM.
- **Posterior vs. prior:** The posterior sees the current observation during training; the prior predicts without seeing it and is used for imagination.
- **KL balancing:** The KL term aligns posterior and prior while avoiding posterior collapse or an over-regularized latent.
- **Actor-critic in imagination:** The policy is optimized using imagined rewards and values rather than direct environment gradients.

The return target commonly used in Dreamer-style agents is the lambda return:

```{math}
G_t^\lambda = r_t + \gamma_t\left((1-\lambda)V(s_{t+1}) + \lambda G_{t+1}^\lambda\right).
```

### When to use it

Use Dreamer when your environment has image observations, actions are known, and you want sample-efficient reinforcement learning that can learn from a replay buffer while improving the policy in latent rollouts.

## PlaNet and RSSM

PlaNet also learns latent dynamics from pixels, but it is usually associated with online planning rather than fully training an actor in imagination. In Simulusion, `Planet` is the high-level entry point and `RecurrentStateSpaceModel` is the PlaNet-style state-space component.

### Study focus

- **Belief state:** The deterministic recurrent state summarizes history.
- **Stochastic state:** A sampled latent captures uncertainty and multimodality.
- **Planner loop:** Candidate action sequences are scored by imagined returns, then the first action is executed.

PlaNet is a good conceptual bridge: it is simpler than Dreamer’s actor-critic stack but introduces the same latent-dynamics ideas.

## Modular RSSM

`ModularRSSM` is designed for experimentation. Instead of hard-coding one encoder, backbone, and decoder, it provides interchangeable components:

- `ConvEncoder`, `MLPEncoder`, and `ViTEncoder` for observations.
- `GRUBackbone`, `LSTMBackbone`, and `TransformerBackbone` for temporal memory.
- `ConvDecoder` and `MLPDecoder` for reconstruction.

Use it when studying ablations: for example, replacing a GRU with a transformer backbone while keeping the same loss and data pipeline.

## JEPA and Vision Transformers

JEPA (Joint Embedding Predictive Architecture) learns by predicting representations rather than pixels. The central idea is to avoid spending capacity on reconstructing every pixel detail and instead learn abstract features that are useful for downstream prediction or control.

```{math}
\mathcal{L}_{\text{JEPA}} = \left\|g_\theta(f_\theta(x_{\text{context}}), m) - \operatorname{sg}\left(f_{\bar{\theta}}(x_{\text{target}})\right)\right\|_2^2.
```

Here, `sg` denotes stop-gradient, `f` is an encoder, `g` is a predictor, and `m` represents masks or target metadata.

### Simulusion pieces

- `JEPAAgent` coordinates representation learning.
- `VisionTransformer` and ViT helper constructors provide patch-based encoders.
- `simulusion.masks.*` contains masking/collation strategies for context-target prediction.

### When to use it

Use JEPA when you primarily want strong latent representations for perception, planning, or future world-model components, especially when pixel-perfect generation is not the goal.

## IRIS

IRIS represents frames as discrete tokens and trains an autoregressive transformer world model. An actor-critic then learns from imagined token rollouts. This model family is useful for studying the connection between language-model-style sequence prediction and model-based RL.

```{mermaid}
graph TD
    A["Frame"] --> B["Discrete autoencoder"]
    B --> C["Token sequence"]
    C --> D["Autoregressive transformer"]
    E["Action"] --> D
    D --> F["Next tokens"]
    D --> G["Reward and terminal heads"]
    F --> H["Imagined frame or state"]
    H --> I["Actor critic"]
```

### Important ideas

- **Vector quantization:** Continuous encoder outputs are mapped to codebook entries.
- **Token dynamics:** The transformer predicts the next discrete visual tokens conditioned on previous tokens and actions.
- **Imagination:** Policy learning uses sampled token futures, decoded states, rewards, and termination predictions.

### Simulusion pieces

- `IRISAgent` combines model and policy behavior.
- `IRISTransformer` and `IRISWorldModel` implement token dynamics.
- `DiscreteAutoencoder`, `IRISEncoder`, `IRISDecoder`, and VQ layers implement visual tokenization.
- `IRISReplayBuffer` and `IRISOnPolicyBuffer` store experience for world-model and policy updates.

## Genie

Genie studies controllable world modeling from videos, including settings where action labels may not be available. It learns a latent action model (LAM) that infers action-like discrete variables from frame transitions, then trains a dynamics model to predict future video tokens conditioned on those latent actions.

### Core pipeline

1. **Video tokenizer:** compresses frames into discrete visual tokens.
2. **Latent Action Model:** infers discrete latent actions from pairs or windows of frames.
3. **Dynamics model:** predicts future tokens using visual tokens and latent actions.
4. **Sampler:** iteratively fills or samples future tokens for interactive generation.

This is useful for studying how controllability can emerge from observation-only video data.

## DDPM, DiT, and DIAMOND-style diffusion models

Diffusion models learn to denoise corrupted data. Instead of predicting a single deterministic next frame, they learn a reverse process from noise to clean samples.

```{math}
\begin{aligned}
q(x_t \mid x_0)
&= \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I\right), \\
\mathcal{L}_{\text{DDPM}}
&= \left\|\epsilon - \epsilon_\theta(x_t, t, c)\right\|_2^2.
\end{aligned}
```

### Simulusion pieces

- `DDPM` implements a denoising diffusion probabilistic model.
- `DiT` implements a diffusion transformer with patch embedding, time conditioning, and transformer blocks.
- `DiffusionUNet`, `EDMPreconditioner`, and `EulerSampler` support DIAMOND-style visual dynamics.
- `RewardTerminationModel` and `ActorCriticNetwork` support RL training around diffusion-generated trajectories.

### When to use it

Use diffusion when high-quality generative rollouts matter and you can afford more expensive sampling. Use transformer diffusion (`DiT`) when patch-token modeling and scalable attention are central to the experiment.

## World Models (Ha & Schmidhuber, 2018)

This family follows the three-component architecture from the *World Models* paper:

```{math}
\begin{aligned}
\text{V:}\quad & z_t \sim \text{VAE}(x_t) \\
\text{M:}\quad & p(z_{t+1} \mid a_t, z_t, h_t) = \sum_k \pi_k\,\mathcal{N}(\mu_k, \sigma_k) \\
\text{C:}\quad & a_t = W_c\,[z_t, h_t] + b_c
\end{aligned}
```

### Mental model

```{mermaid}
graph LR
    A["Raw pixels"] --> B["V: ConvVAE encoder"]
    B --> C["Latent vector z"]
    C --> D["M: MDN-RNN"]
    D --> E["Hidden state h + predicted next z"]
    C --> F["C: Linear controller"]
    E --> F
    F --> G["Action a"]
    G --> H["Environment"]
    H --> A
```

### Three-stage pipeline

| Stage | Component | Training | File |
|---|---|---|---|
| 1 | **V — Vision (ConvVAE)** | Unsupervised reconstruction on random rollouts. Encodes 64×64 RGB frames → latent `z` (typically 32-d). | `simulusion.vision.VAE.ConvVAE` |
| 2 | **M — Memory (MDN-RNN)** | Predicts next latent `z_{t+1}` as a Gaussian mixture conditioned on `(a_t, z_t, h_t)`. Also predicts rewards and terminal flags. | `simulusion.models.mdrnn` |
| 3 | **C — Controller (Linear)** | Maps `(z_t, h_t)` → `a_t`. Trained with CMA-ES (not backprop) to maximize cumulative reward. | `simulusion.models.controller` |

### Key ideas

- **Latent-space planning**: The controller operates on compressed `z` + `h`, not raw pixels. This makes CMA-ES tractable (only ~10³ params).
- **Memory via hidden state**: The RNN hidden state `h_t` encodes temporal context. The paper shows removing it drops CarRacing score from 906→632.
- **Dream training**: The trained M can serve as a differentiable simulator. The controller can be trained entirely inside M's hallucinated latent rollouts and then transferred back to the real environment.
- **Temperature annealing**: During dream training, increasing M's sampling temperature `τ` makes the dream environment harder, preventing the controller from exploiting model imperfections.

### When to use it

Use the Ha & Schmidhuber world model when you want:
- A didactic, minimal world model implementation to study how latent-dynamics + evolution works
- To quickly train a policy on a continuous-control task without GPU-heavy RL backprop
- A baseline to compare model-based RL (Dreamer/PlaNet) with evolution-based controller training

### Simulusion pieces

| Component | Module | Key classes |
|---|---|---|
| VAE | `simulusion.vision.VAE.ConvVAE` | `ConvVAE`, `ConvVAEEncoder`, `ConvVAEDecoder` |
| Dynamics | `simulusion.models.mdrnn` | `MDRNN`, `MDRNNCell` |
| Policy | `simulusion.models.controller` | `Controller` |
| Configs | `simulusion.configs.wm_config` | `WMVAEConfig`, `WMMDNRNNConfig`, `WMControllerConfig` |
| Datasets | `simulusion.datasets.wm_dataset` | `RolloutDataset`, `ObservationDataset`, `SequenceDataset`, `LatentSequenceDataset` |
| Losses | `simulusion.losses.convae_loss` | `conv_vae_loss_fn` |
| | `simulusion.losses.gmm_loss` | `gmm_loss` |
| Training | `simulusion.training.train_world_model` | `run_training_pipeline`, `generate_rollouts`, `test_trained_model` |
| | `simulusion.training.train_convvae` | `train_convae` |
| | `simulusion.training.train_mdn_rnn` | `train_mdn_rnn` |
| | `simulusion.training.train_controller` | `train_controller` |

### Quick-start example

```python
from simulusion.configs.wm_config import WMVAEConfig, WMMDNRNNConfig, WMControllerConfig
from simulusion.training.train_convvae import train_convae
from simulusion.training.train_mdn_rnn import train_mdn_rnn
from simulusion.training.train_controller import train_controller

# Stage 1: Train VAE
vae_config = WMVAEConfig(
    {
        "height": 64,
        "width": 64,
        "latent_size": 32,
        "data_dir": "./data/carracing",
        "logdir": "./results/carracing",
        "num_epochs": 10,
        "learning_rate": 1e-3,
    }
)
train_convae(vae_config)

# Stage 2: Train MDN-RNN
mdrnn_config = WMMDNRNNConfig(
    {
        "latent_size": 32,
        "action_size": 3,
        "hidden_size": 256,
        "gmm_components": 5,
        "data_dir": "./data/carracing",
        "logdir": "./results/carracing",
        "num_epochs": 30,
    }
)
train_mdn_rnn(vae_config, mdrnn_config)

# Stage 3: Train Controller with CMA-ES
ctrl_config = WMControllerConfig(
    {
        "latent_size": 32,
        "hidden_size": 256,
        "action_size": 3,
        "env_name": "CarRacing-v3",
        "logdir": "./results/carracing",
        "pop_size": 10,
        "n_samples": 4,
        "target_return": 950.0,
    }
)
train_controller(ctrl_config)
```

### Testing a trained model

```python
from simulusion.training.train_world_model import test_trained_model

test_trained_model(
    logdir="./results/carracing",
    env_name="CarRacing-v3",
    action_size=3,
    num_episodes=5,
)
```

## Practical study path

1. **Start with RSSMs:** understand deterministic and stochastic latent state.
2. **Study Dreamer:** connect latent dynamics to actor-critic learning.
3. **Study PlaNet:** compare planning against learned policies.
4. **Study World Models (Ha & Schmidhuber):** see how VAE + MDN-RNN + CMA-ES controller forms the simplest complete pipeline.
5. **Study token world models:** IRIS and Genie show how discrete visual tokens enable transformer dynamics.
6. **Study representation-only prediction:** JEPA clarifies why not every useful world model must reconstruct pixels.
7. **Study diffusion:** compare likelihood-style denoising rollouts against autoregressive and RSSM rollouts.
8. **Read the API reference:** map each concept to the exact Simulusion class or function.

## Common failure modes

- **Blurry reconstructions:** decoder or latent bottleneck is too weak, or reconstruction dominates the objective.
- **Good one-step predictions but bad rollouts:** dynamics errors compound; evaluate multi-step imagination.
- **Posterior collapse:** the recurrent state ignores stochastic latents; tune KL weights and capacity.
- **Unstable policy learning:** imagined rewards or values are poorly calibrated; shorten horizon and improve world-model validation first.
- **Token dead codes:** vector quantizer codebook usage collapses; tune commitment loss and EMA updates.
- **Diffusion slow sampling:** reduce denoising steps or use a better sampler/preconditioner.
