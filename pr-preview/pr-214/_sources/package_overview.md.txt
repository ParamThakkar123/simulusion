# Package Overview

Simulusion is organized into focused modules so you can use only the pieces you need.

## Quick Import (Public API)

For applications and examples, import the `simulusion` package (installed from
PyPI as `simulusion`). It mirrors the Simulusion implementation package and
exposes the same lazy public API without importing optional training backends
until you use them.

```python
import simulusion

print(simulusion.list_models())
agent = simulusion.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
env = simulusion.make_env("CartPole-v1", backend="gym")
```

Use `simulusion` for direct component imports as well as factory helpers:

```python
from simulusion import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

### Available Exports

| Category | Exports |
|----------|--------|
| **Friendly factories** | `create_config`, `create_model`, `make_env`, `list_models`, `list_env_backends`, `list_envs` |
| **Models / Agents** | `Dreamer`, `DreamerV1`, `DreamerV2`, `DreamerAgent`, `Planet`, `JEPAAgent`, `IRISAgent`, `Genie`, `create_genie`, `DiT`, `create_dit` |
| **State-space models** | `RSSM`, `RecurrentStateSpaceModel`, `DreamerRSSM`, `ModularRSSM`, `create_modular_rssm` |
| **Vision** | `ConvEncoder`, `CNNEncoder`, `IRISEncoder`, `ConvDecoder`, `CNNDecoder`, `DenseDecoder`, `ActionDecoder`, `IRISDecoder`, `VideoTokenizer`, `create_video_tokenizer` |
| **Quantization** | `VectorQuantizer`, `VectorQuantizerEMA` |
| **Configs** | `DreamerConfig`, `JEPAConfig`, `DiTConfig`, `DiamondConfig`, `IRISConfig`, `GenieConfig`, `GenieSmallConfig`, `STTransformerConfig`, `VideoTokenizerConfig`, `LatentActionModelConfig`, `DynamicsModelConfig` |
| **Environments** | `make_atari_env`, `make_gym_env`, `make_mujoco_env`, `make_robotics_env`, `make_brax_env`, `make_procgen_env`, `GymImageEnv`, `ProcgenImageEnv`, `DeepMindControlEnv`, `DMLabEnv`, `make_dmlab_env`, `UnityMLAgentsEnv`, `TimeLimit`, `ActionRepeat`, wrappers, etc. |
| **Memory** | `ReplayBuffer`, `Memory`, `Episode`, `IRISReplayBuffer`, `IRISOnPolicyBuffer` |
| **Reward / Value** | `RewardModel`, `ValueModel`, `DreamerRewardModel`, `DreamerValueModel` |
| **Controllers** | `RSSMPolicy`, `RolloutGenerator`, `IRISPolicy`, `IRISActor`, `IRISCritic`, `CNNFeatureExtractor` |
| **Transformer blocks** | `STTransformer`, `MultiHeadSelfAttention`, `MultiHeadAttention`, `AdaLNNormalization`, `RMSNorm` |
| **Diffusion** | `DiT`, `DDPM`, `PatchEmbed`, `PatchUnEmbed`, `ActorCriticNetwork`, `RewardTerminationModel` |
| **Genie subcomponents** | `LatentActionModel`, `DynamicsModel`, `create_latent_action_model`, `create_dynamics_model` |
| **Export** | `export_any`, `export_model`, `ExportableAgentMixin` |
| **Registry / plugins** | `register_world_model`, `deregister_world_model`, `register_env_backend`, `deregister_env_backend` |
| **Deprecation** | `deprecated`, `deprecated_class`, `deprecated_function` |
| **Utilities** | `Logger`, `FreezeParameters`, `compute_return`, `preprocess_obs`, `__version__` |

Example usage:

```python
import simulusion

# Training
agent = simulusion.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
agent.train()
```

## Core Modules

The module paths below are the public `simulusion.*` surface
(`from simulusion.models import Dreamer`).

- `simulusion.models`: High-level models and agents (`Dreamer`, `DreamerAgent`, `Planet`, `JEPAAgent`)
- `simulusion.configs`: Configuration containers for Dreamer, JEPA, and diffusion runs
- `simulusion.training`: Script-style training entrypoints for world models (VAE, MDNRNN, Controller, Planet, RSSM, JEPA)

## Environment Integration

- `simulusion.envs`: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents adapters
- `simulusion.envs.wrappers`: Action repeat, normalization, time limits

## World Model Building Blocks

- `simulusion.models.dreamer_rssm`: Recurrent state-space model used by Dreamer
- `simulusion.models.modular_rssm`: Modular RSSM with swappable encoder/decoder/backbone for research experiments
- `simulusion.vision`: Encoders/decoders and action heads for latent dynamics models
- `simulusion.reward`: Reward and value prediction heads
- `simulusion.observations`: Symbolic and visual observation reconstruction modules

## Representation Learning and Diffusion

- `simulusion.models.vit`: Vision Transformer and JEPA predictor components
- `simulusion.models.diffusion`: DDPM scheduler and DiT model implementation
- `simulusion.masks`: Mask collators for JEPA-style context/target masking

## Data and Memory

- `simulusion.datasets`: CIFAR-10, ImageNet-1K, and generic `ImageFolder` dataset loaders
- `simulusion.memory`: Replay buffers for Dreamer and episode-based memory for PlaNet/RSSM

## Utilities

- `simulusion.utils`: Logging, parameter freezing, transforms
- `simulusion.transforms`: Data augmentation pipelines
- `simulusion.benchmarks`: CLI and reporting utilities

## Which API Should I Use?

- End-to-end Dreamer training: `DreamerAgent`
- End-to-end JEPA training: `JEPAAgent`
- World model training scripts: `simulusion.training` modules (e.g., `train_world_model` for VAE+MDNRNN+Controller pipeline)
- Low-level model experimentation: `Dreamer`, `RSSM`, decoder/encoder modules
- Custom world model architectures: `ModularRSSM` with swappable encoder/decoder/backbone
- Custom data pipelines: `make_cifar10`, `make_imagenet1k`, `make_imagefolder`
