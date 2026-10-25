# Exported Names by Category

This page lists every public class, function, and constant exported from the
`simulusion` top-level package, grouped by category. For signatures and docstrings
generated from source, see {doc}`api_reference`.

```{contents} Contents
:depth: 2
```

## Factory helpers

| Name | Description |
|---|---|
| `create_config(model, **overrides)` | Return a default config dict for a model family, with optional overrides applied. Models: `dreamer`, `jepa`, `iris`, `dit`, `genie`, `planet`. |
| `create_model(model, config=None, **overrides)` | Instantiate a model or high-level agent by canonical name (see {doc}`public_api`). |
| `make_env(env_id, backend="auto", **kwargs)` | Create a Simulusion environment through a named backend. Backends: `dmc`, `gym`, `atari`, `mujoco`, `robotics`, `procgen`, `brax`, `bsuite`, `unity`. |
| `list_models()` | Return canonical model names accepted by `create_model`. |
| `list_env_backends()` | Return backend names accepted by `make_env`. |
| `list_envs(model=None)` | Return known environment IDs, optionally filtered by model family. |
| `get_model_spec(name)` | Return metadata ({py:class}`ModelSpec`) for a model name or alias. |
| `get_env_backend_spec(name)` | Return metadata ({py:class}`EnvBackendSpec`) for an environment backend. |

## Data classes

| Name | Description |
|---|---|
| `ModelSpec` | Named tuple describing a registered model (name, import_path, config_path, aliases, description). |
| `EnvBackendSpec` | Named tuple describing a registered environment backend (name, factory_path, aliases, description). |

## World model agents

| Name | Source | Description |
|---|---|---|
| `Dreamer` | `simulusion.models.dreamer` | Base Dreamer world model (RSSM-based, V1-style). |
| `DreamerV1` | `simulusion.models.dreamer_v1` | DreamerV1 (alias for base Dreamer). |
| `DreamerV2` | `simulusion.models.dreamer_v2` | DreamerV2 (symlog two-hot heads, balanced KL). |
| `DreamerAgent` | `simulusion.models.dreamer` | High-level Dreamer agent with train/evaluate helpers. |
| `DreamerV1Agent` | `simulusion.models.dreamer_v1` | `DreamerAgent` that trains DreamerV1 (`create_model("dreamer-v1")`). |
| `DreamerV2Agent` | `simulusion.models.dreamer_v2` | `DreamerAgent` that trains DreamerV2 (`create_model("dreamer-v2")`). |
| `Planet` | `simulusion.models.planet` | PlaNet: Deep Planning Network. |
| `JEPAAgent` | `simulusion.models.jepa_agent` | I-JEPA agent for self-supervised visual representation learning. |
| `IRISAgent` | `simulusion.models.iris_agent` | IRIS agent for sample-efficient RL with Transformers. |
| `Genie` | `simulusion.models.genie` | Genie generative interactive environment. |
| `create_genie` | `simulusion.models.genie` | Create a Genie model with specified parameters. |
| `create_genie_small` | `simulusion.models.genie` | Create Genie-small variant (~50M params). |
| `create_genie_large` | `simulusion.models.genie` | Create Genie-large variant (~11B params). |

## Genie subcomponents

| Name | Source | Description |
|---|---|---|
| `LatentActionModel` | `simulusion.models.latent_action_model` | Learns latent actions from pairs of video frames. |
| `DynamicsModel` | `simulusion.models.dynamics_model` | Transformer-based dynamics for future token prediction. |
| `create_latent_action_model` | `simulusion.models.latent_action_model` | Factory for LatentActionModel. |
| `create_dynamics_model` | `simulusion.models.dynamics_model` | Factory for DynamicsModel. |

## State-space models

| Name | Source | Description |
|---|---|---|
| `RSSM` | `simulusion.models.rssm` | Recurrent State-Space Model (standalone). |
| `RecurrentStateSpaceModel` | `simulusion.models.rssm` | Alias for RSSM. |
| `DreamerRSSM` | `simulusion.models.dreamer_rssm` | RSSM variant used in Dreamer training loop. |
| `ModularRSSM` | `simulusion.models.modular_rssm` | Modular RSSM with swappable encoder/backbone/decoder. |
| `create_modular_rssm` | `simulusion.models.modular_rssm` | Factory for ModularRSSM. |

## Diffusion models

| Name | Source | Description |
|---|---|---|
| `DiT` | `simulusion.models.diffusion` | Diffusion Transformer model. |
| `create_dit` | `simulusion.models.diffusion` | Factory for DiT. |
| `PatchEmbed` | `simulusion.models.diffusion` | Image-to-patch embedding layer. |
| `PatchUnEmbed` | `simulusion.models.diffusion` | Patch-to-image un-embedding layer. |
| `DDPM` | `simulusion.models.diffusion` | Denoising Diffusion Probabilistic Model. |
| `ActorCriticNetwork` | `simulusion.models.diffusion` | Actor-critic head for DIAMOND-style RL. |
| `RewardTerminationModel` | `simulusion.models.diffusion` | Reward + termination predictor for DIAMOND. |
| `sinusoidal_time_embedding` | `simulusion.models.diffusion` | Time-step embedding for diffusion. |

## Vision components

| Name | Source | Description |
|---|---|---|
| `ConvEncoder` | `simulusion.vision.dreamer_encoder` | Dreamer convolutional encoder (image → embedding). |
| `CNNEncoder` | `simulusion.vision.planet_encoder` | PlaNet CNN encoder (image → embedding). |
| `IRISEncoder` | `simulusion.vision.iris_encoder` | IRIS encoder (image → discrete tokens). |
| `ConvDecoder` | `simulusion.vision.dreamer_decoder` | Dreamer convolutional decoder (latent → image distribution). |
| `CNNDecoder` | `simulusion.vision.planet_decoder` | PlaNet CNN decoder. |
| `DenseDecoder` | `simulusion.vision.dreamer_decoder` | MLP decoder for reward/value/discount. |
| `ActionDecoder` | `simulusion.vision.dreamer_decoder` | Dreamer policy head (latent → tanh-squashed action). |
| `IRISDecoder` | `simulusion.vision.iris_decoder` | IRIS decoder (tokens → image). |
| `VideoTokenizer` | `simulusion.vision.video_tokenizer` | Genie VQ-VAE video tokenizer. |
| `create_video_tokenizer` | `simulusion.vision.video_tokenizer` | Factory for VideoTokenizer. |
| `VectorQuantizer` | `simulusion.vision.vq_layer` | VQ-VAE vector quantization layer. |
| `VectorQuantizerEMA` | `simulusion.vision.vq_layer` | VQ-VAE with EMA codebook updates. |
| `TanhBijector` | `simulusion.vision.dreamer_decoder` | Tanh bijection for action squashing. |
| `SampleDist` | `simulusion.vision.dreamer_decoder` | MC-sampled distribution statistics. |

## Config classes

| Name | Source | Description |
|---|---|---|
| `DreamerConfig` | `simulusion.configs.dreamer_config` | Dreamer hyperparameter config. |
| `JEPAConfig` | `simulusion.configs.jepa_config` | JEPA hyperparameter config. |
| `DiTConfig` | `simulusion.configs.dit_config` | DiT hyperparameter config. |
| `get_dit_config` | `simulusion.configs.dit_config` | Factory for DiTConfig with presets. |
| `DiamondConfig` | `simulusion.configs.diamond_config` | DIAMOND hyperparameter config. |
| `IRISConfig` | `simulusion.configs.iris_config` | IRIS hyperparameter config. |
| `GenieConfig` | `simulusion.configs.genie_config` | Genie hyperparameter config. |
| `GenieSmallConfig` | `simulusion.configs.genie_config` | Genie-small preset config. |
| `STTransformerConfig` | `simulusion.configs.st_transformer_config` | ST-Transformer config. |
| `VideoTokenizerConfig` | `simulusion.configs.video_tokenizer_config` | Video tokenizer config. |
| `LatentActionModelConfig` | `simulusion.configs.lam_config` | Latent action model config. |
| `DynamicsModelConfig` | `simulusion.configs.dynamics_config` | Dynamics model config. |

## Constants

| Name | Description |
|---|---|
| `MODEL_SPECS` | Dict of all built-in model specs (name → {py:class}`ModelSpec`). |
| `ENV_BACKEND_SPECS` | Dict of all built-in environment backend specs. |
| `ATARI_100K_GAMES` | List of Atari 100K benchmark game names. |
| `HUMAN_SCORES` | Dict of human baseline scores for Atari 100K. |
| `RANDOM_SCORES` | Dict of random baseline scores for Atari 100K. |

## Memory / replay buffers

| Name | Source | Description |
|---|---|---|
| `ReplayBuffer` | `simulusion.memory.dreamer_memory` | Dreamer ring buffer (transitions → sequences). |
| `Memory` | `simulusion.memory.planet_memory` | Episode-based memory for PlaNet. |
| `Episode` | `simulusion.memory.planet_memory` | Single episode recording. |
| `IRISReplayBuffer` | `simulusion.memory.iris_memory` | Ring buffer for IRIS (uint8 images). |
| `IRISOnPolicyBuffer` | `simulusion.memory.iris_memory` | On-policy buffer for episode collection. |

## Environments and wrappers

| Name | Source | Description |
|---|---|---|
| `DeepMindControlEnv` | `simulusion.envs.dmc_env` | DeepMind Control Suite adapter. |
| `DMLabEnv` | `simulusion.envs.dmlab_env` | DeepMind Lab adapter. |
| `make_dmlab_env` | `simulusion.envs.dmlab_env` | Factory for DeepMind Lab. |
| `DMLAB_LEVELS` | `simulusion.envs.dmlab_env` | Available DMLab level names. |
| `GymImageEnv` | `simulusion.envs.gym_env` | Gymnasium image adapter. |
| `make_gym_env` | `simulusion.envs.gym_env` | Factory for GymImageEnv. |
| `MuJoCoImageEnv` | `simulusion.envs.mujoco_env` | MuJoCo image adapter. |
| `make_mujoco_env` | `simulusion.envs.mujoco_env` | Factory for MuJoCo environments. |
| `MujocoEnv` | `simulusion.envs.mujoco_env` | Alias of `MuJoCoImageEnv`. |
| `BraxImageEnv` | `simulusion.envs.brax_env` | Brax image adapter. |
| `make_brax_env` | `simulusion.envs.brax_env` | Factory for BraxImageEnv. |
| `BSuiteImageEnv` | `simulusion.envs.bsuite_env` | BSuite image adapter. |
| `make_bsuite_env` | `simulusion.envs.bsuite_env` | Factory for BSuiteImageEnv. |
| `list_available_bsuite_ids` | `simulusion.envs.bsuite_env` | List BSuite environment IDs. |
| `make_atari_env` | `simulusion.envs.atari_env` | Factory for Atari ALE environments. |
| `list_available_atari_envs` | `simulusion.envs.atari_env` | List available Atari game IDs. |
| `make_atari_vector_env` | `simulusion.envs.atari_env` | Factory for vectorized Atari. |
| `make_diamond_atari_env` | `simulusion.envs.diamond_atari` | DIAMOND-style Atari preprocessing. |
| `make_procgen_env` | `simulusion.envs.procgen_env` | Factory for Procgen environments. |
| `make_robotics_env` | `simulusion.envs.robotics_env` | Factory for Gymnasium Robotics. |
| `register_gymnasium_robotics_envs` | `simulusion.envs.robotics_env` | Register Robotics envs. |
| `list_gymnasium_robotics_envs` | `simulusion.envs.robotics_env` | List installed Robotics envs. |
| `UnityMLAgentsEnv` | `simulusion.envs.unity_env` | Unity ML-Agents adapter. |
| `make_unity_mlagents_env` | `simulusion.envs.unity_env` | Factory for UnityMLAgentsEnv. |
| `DreamEnv` | `simulusion.envs.dream` | Any `WorldModel` as a Gymnasium environment, to train agents inside it. |
| `DreamVectorEnv` | `simulusion.envs.dream` | Many dreams in one batched model call, as a Gymnasium vector environment. |
| `WorldModelEnv` | `simulusion.envs.world_model_env` | Environment inside a learned world model. |
| `make_world_model_env` | `simulusion.envs.world_model_env` | Factory for WorldModelEnv. |
| `TimeLimit` | `simulusion.envs.wrappers` | Episode time limit wrapper. |
| `ActionRepeat` | `simulusion.envs.wrappers` | Action repeat wrapper. |
| `NormalizeActions` | `simulusion.envs.wrappers` | Action normalization to [-1, 1]. |
| `ObsDict` | `simulusion.envs.wrappers` | Observation-to-dict conversion. |
| `OneHotAction` | `simulusion.envs.wrappers` | Discrete to one-hot action conversion. |
| `RewardObs` | `simulusion.envs.wrappers` | Reward observation injection. |
| `ResizeImage` | `simulusion.envs.wrappers` | Image resizing wrapper. |
| `RenderImage` | `simulusion.envs.wrappers` | Render-based image observation. |
| `SelectAction` | `simulusion.envs.wrappers` | Action selection wrapper. |

## Controllers and policies

| Name | Source | Description |
|---|---|---|
| `RSSMPolicy` | `simulusion.controller` | RSSM-based policy for Dreamer. |
| `RolloutGenerator` | `simulusion.controller` | Policy rollouts in the environment. |
| `IRISActor` | `simulusion.controller` | IRIS actor head. |
| `IRISCritic` | `simulusion.controller` | IRIS critic head. |
| `IRISPolicy` | `simulusion.controller` | IRIS combined actor-critic policy. |
| `CNNFeatureExtractor` | `simulusion.controller` | CNN feature extractor for policy inputs. |

## Export

| Name | Source | Description |
|---|---|---|
| `export_any(obj, path, format, ...)` | `simulusion.export` | Export a model or agent (`exported_program`, `aoti`, ONNX, TensorRT, TorchScript). |
| `export_model(module, path, format, ...)` | `simulusion.export` | Export a raw nn.Module. |
| `load_exported(path, format=None)` | `simulusion.export` | Load any exported artifact as a callable. |
| `verify_export(module, exported, example_inputs)` | `simulusion.export` | Check an artifact against its eager module; returns max abs error. |
| `ExportableAgentMixin` | `simulusion.export` | Mixin that adds `.export()` to custom agents. |

## Inference and deployment

See {doc}`deployment_guide`.

| Name | Source | Description |
|---|---|---|
| `optimize_for_inference(module, ...)` | `simulusion.inference` | Wrap a module with a precision policy and optional `torch.compile` / CUDA graphs. |
| `InferenceModel` | `simulusion.inference` | The wrapper returned by `optimize_for_inference`. |
| `inference_context(device, precision)` | `simulusion.inference` | `inference_mode` plus autocast at one resolved precision. |
| `make_stepper(agent)` | `simulusion.inference` | Uniform `init_state` / `observe` / `act` / `imagine` step interface. |
| `DreamerStepper`, `IRISStepper` | `simulusion.inference` | Steppers for Dreamer and IRIS. |
| `DreamerStepModule` | `simulusion.inference` | One Dreamer observe+act step as a pure, exportable module. |
| `benchmark_step(fn, *args)` | `simulusion.inference` | Latency percentiles, throughput and peak memory. |
| `rollout_drift(reference, candidate, state, inputs)` | `simulusion.inference` | Closed-loop divergence of an optimized step from a reference. |
| `quantize_weights(module)` | `simulusion.inference` | Weight-only int8 quantization of `nn.Linear` layers. |
| `save_bundle(dir, module, ...)` / `load_bundle(dir)` | `simulusion.inference` | Deployment bundles: weights, config, artifacts, manifest. |
| `STKVCache` | `simulusion.blocks.st_transformer` | Temporal KV cache for incremental Genie / ST-transformer generation. |

## World model protocol

See {doc}`world_model_protocol`.

## Training

See {doc}`trainer_guide`.

| Name | Source | Description |
|---|---|---|
| `Trainer` | `simulusion.training.trainer` | Runs any training task to its budget on one device: precision, logging, evaluation, early stopping, checkpoints and resume, world model export. |
| `TrainerConfig` | `simulusion.training.trainer` | Settings shared by every run (the `trainer:` section of a run config). |
| `TrainResult` | `simulusion.training.trainer` | What `Trainer.fit` returns: run directory, progress, final metrics, checkpoint and export paths. |
| `TrainingTask` | `simulusion.training.task` | Base class of a family's training (networks, optimizers, data, one step). |
| `register_task` / `list_tasks` | `simulusion.training.task` | Register a task under a name / list the registered ones. |
| `run_training(config, overrides)` | `simulusion.training.run` | Train from a YAML run config (or mapping) plus `key=value` overrides. |
| `MixedPrecision` | `simulusion.training.precision` | Autocast dtype and loss scaling for one precision setting. |
| `GradientUpdate` | `simulusion.training.precision` | One optimizer's step: backward, unscale, clip, step, with accumulation and gradient all-reduce. |

## Data

See {doc}`episodes_guide`.

| Name | Source | Description |
|---|---|---|
| `EpisodeFolder` | `simulusion.data` | A folder of episodes (`.npz`, videos, `.npy`), the dataset every task reads. |
| `EpisodeWriter` | `simulusion.data` | Append episodes to a folder and record its `meta.json`. |
| `LeRobotEpisodes` | `simulusion.data` | Local LeRobot datasets (v2.x and v3.0) as episodes. |

## Agents

| Name | Source | Description |
|---|---|---|
| `CEMPlanner` | `simulusion.agents` | Cross-entropy-method planning over any world model's predicted returns. |
| `PlanningAgent` | `simulusion.agents` | Acts in a real environment by planning in a world model each step. |

| Name | Source | Description |
|---|---|---|
| `as_world_model(model)` | `simulusion.world_model` | Wrap a trained Dreamer, PlaNet, IRIS, DIAMOND, Genie or (ConvVAE, MDN-RNN) model in the common protocol. |
| `load_world_model(path, family=None)` | `simulusion.world_model` | Load a saved world model, or a family's own training checkpoint. |
| `WorldModel` | `simulusion.world_model` | The protocol: `encode` / `step` / `rollout` / `decode` / `save`. |
| `WorldModelSpec` | `simulusion.world_model` | Observation shape, action space, context length and which heads a model has. |
| `list_world_model_families()` | `simulusion.world_model` | The family names `load_world_model` accepts. |

## Reward and value models

| Name | Source | Description |
|---|---|---|
| `RewardModel` | `simulusion.reward` | Base reward model. |
| `ValueModel` | `simulusion.reward` | Base value model. |
| `DreamerRewardModel` | `simulusion.reward` | Dreamer reward predictor. |
| `DreamerValueModel` | `simulusion.reward` | Dreamer value function. |

## Transformer blocks

| Name | Source | Description |
|---|---|---|
| `STTransformer` | `simulusion.blocks` | Spatiotemporal Transformer (Genie). |
| `MultiHeadSelfAttention` | `simulusion.blocks` | Multi-head self-attention. |
| `MultiHeadAttention` | `simulusion.blocks` | Multi-head cross-attention. |
| `AdaLNNormalization` | `simulusion.blocks` | Adaptive layer norm for diffusion. |
| `RMSNorm` | `simulusion.blocks` | Root mean square layer norm. |

## Plugin registry

| Name | Source | Description |
|---|---|---|
| `register_world_model(name, import_path, ...)` | `simulusion.registry` | Register a custom world model architecture. |
| `deregister_world_model(name)` | `simulusion.registry` | Remove a registered model. |
| `get_registered_model_spec(name)` | `simulusion.registry` | Look up a registered model spec. |
| `list_registered_models()` | `simulusion.registry` | List all externally registered model names. |
| `register_env_backend(name, factory_path, ...)` | `simulusion.registry` | Register a custom environment backend. |
| `deregister_env_backend(name)` | `simulusion.registry` | Remove a registered env backend. |
| `list_registered_env_backends()` | `simulusion.registry` | List all registered env backends. |

## Deprecation utilities

| Name | Source | Description |
|---|---|---|
| `deprecated(version, reason)` | `simulusion.utils.deprecation` | Decorator to mark functions/classes as deprecated. |
| `deprecated_class(version, alternative)` | `simulusion.utils.deprecation` | Shortcut for deprecating a class. |
| `deprecated_function(version, alternative)` | `simulusion.utils.deprecation` | Shortcut for deprecating a function. |

## General utilities

| Name | Source | Description |
|---|---|---|
| `Logger` | `simulusion.utils` | Logging utility. |
| `FreezeParameters` | `simulusion.utils` | Context manager to freeze model parameters. |
| `compute_return(rewards, values, gamma, lambda_)` | `simulusion.utils` | Compute GAE or λ-return. |
| `preprocess_obs(obs)` | `simulusion.utils` | Observation preprocessing (resize, normalize). |

## Version

| Name | Description |
|---|---|
| `__version__` | Package version string (semver). |
