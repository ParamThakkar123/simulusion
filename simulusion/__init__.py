"""Simulusion public API.

This package keeps imports lightweight while still exposing a friendly top-level
surface.  Common workflows can use the small factory helpers::

    import simulusion

    cfg = simulusion.create_config("dreamer", env="walker-walk")
    agent = simulusion.create_model("dreamer", cfg)
    env = simulusion.make_env("CartPole-v1", backend="gym")

Lower-level research components remain available as lazy top-level exports, for
example ``from simulusion import DreamerAgent, ConvEncoder, ReplayBuffer``, and
every implementation submodule is reachable directly::

    from simulusion.models import Dreamer
    from simulusion.training.eval_jepa import jepa_linear_probe
    import simulusion.envs
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from simulusion._version import __version__  # noqa: F401

_API_EXPORTS = {
    "EnvBackendSpec": "simulusion.api",
    "ModelSpec": "simulusion.api",
    "MODEL_SPECS": "simulusion.api",
    "ENV_BACKEND_SPECS": "simulusion.api",
    "create_config": "simulusion.api",
    "create_model": "simulusion.api",
    "get_env_backend_spec": "simulusion.api",
    "get_model_spec": "simulusion.api",
    "list_env_backends": "simulusion.api",
    "list_envs": "simulusion.api",
    "list_models": "simulusion.api",
    "make_env": "simulusion.api",
    "export_any": "simulusion.export",
    "export_model": "simulusion.export",
    "load_exported": "simulusion.export",
    "verify_export": "simulusion.export",
    "ExportableAgentMixin": "simulusion.export",
}

_LAZY_EXPORTS: dict[str, str] = {
    # Agents and high-level models.
    "Dreamer": "simulusion.models",
    "DreamerV1": "simulusion.models",
    "DreamerV2": "simulusion.models",
    "DreamerAgent": "simulusion.models",
    "DreamerV1Agent": "simulusion.models",
    "DreamerV2Agent": "simulusion.models",
    "Planet": "simulusion.models",
    "JEPAAgent": "simulusion.models",
    "IRISAgent": "simulusion.models",
    "IRISTransformer": "simulusion.models",
    "IRISWorldModel": "simulusion.models",
    "LPIPSPerceptualLoss": "simulusion.vision",
    "build_perceptual_loss": "simulusion.vision",
    "compute_lambda_return": "simulusion.models",
    "VisionTransformer": "simulusion.models",
    "ModularRSSM": "simulusion.models",
    "create_modular_rssm": "simulusion.models",
    "Genie": "simulusion.models",
    "LatentActionModel": "simulusion.models",
    "DynamicsModel": "simulusion.models",
    "create_genie": "simulusion.models",
    "create_genie_small": "simulusion.models",
    "create_genie_large": "simulusion.models",
    "create_latent_action_model": "simulusion.models",
    "create_dynamics_model": "simulusion.models",
    # State-space models.
    "RSSM": "simulusion.models",
    "RecurrentStateSpaceModel": "simulusion.models",
    # ``dreamer_rssm`` defines the class as ``RSSM``; ``simulusion.models``
    # is what exposes it under the ``DreamerRSSM`` alias.
    "DreamerRSSM": "simulusion.models",
    # Vision components.
    "ConvEncoder": "simulusion.vision",
    "CNNEncoder": "simulusion.vision",
    "ConvDecoder": "simulusion.vision",
    "CNNDecoder": "simulusion.vision",
    "DenseDecoder": "simulusion.vision",
    "ActionDecoder": "simulusion.vision",
    "TanhBijector": "simulusion.vision",
    "SampleDist": "simulusion.vision",
    "IRISEncoder": "simulusion.vision",
    "IRISDecoder": "simulusion.vision",
    "VideoTokenizer": "simulusion.vision",
    "create_video_tokenizer": "simulusion.vision",
    "VectorQuantizer": "simulusion.vision",
    "VectorQuantizerEMA": "simulusion.vision",
    # Memory.
    "ReplayBuffer": "simulusion.memory",
    "Memory": "simulusion.memory",
    "Episode": "simulusion.memory",
    "IRISReplayBuffer": "simulusion.memory",
    "IRISOnPolicyBuffer": "simulusion.memory",
    # Diffusion models.
    # ``DiT`` and ``DDPM`` name both a class and a sibling module. Importing the
    # module binds it over the class on the package, so which one
    # ``simulusion.models.diffusion.DiT`` returns depends on import order -
    # point at the defining modules to make it deterministic.
    "DiT": "simulusion.models.diffusion.DiT",
    "create_dit": "simulusion.models.diffusion",
    "PatchEmbed": "simulusion.models.diffusion",
    "PatchUnEmbed": "simulusion.models.diffusion",
    "DDPM": "simulusion.models.diffusion.DDPM",
    "ActorCriticNetwork": "simulusion.models.diffusion",
    "RewardTerminationModel": "simulusion.models.diffusion",
    "sinusoidal_time_embedding": "simulusion.models.diffusion",
    # Transformer blocks and layers.
    "STTransformer": "simulusion.blocks",
    "MultiHeadSelfAttention": "simulusion.blocks",
    "MultiHeadAttention": "simulusion.blocks",
    "AdaLNNormalization": "simulusion.blocks",
    "RMSNorm": "simulusion.blocks",
    # Controllers and policies.
    "RSSMPolicy": "simulusion.controller",
    "RolloutGenerator": "simulusion.controller",
    "IRISActor": "simulusion.controller",
    "IRISCritic": "simulusion.controller",
    "IRISPolicy": "simulusion.controller",
    "CNNFeatureExtractor": "simulusion.controller",
    # Configs.
    "DreamerConfig": "simulusion.configs",
    "JEPAConfig": "simulusion.configs",
    "DiTConfig": "simulusion.configs",
    "dit_preset_config": "simulusion.configs",
    "list_dit_presets": "simulusion.configs",
    "get_dit_config": "simulusion.configs",
    "DiamondConfig": "simulusion.configs",
    "IRISConfig": "simulusion.configs",
    "GenieConfig": "simulusion.configs",
    "GenieSmallConfig": "simulusion.configs",
    "STTransformerConfig": "simulusion.configs",
    "VideoTokenizerConfig": "simulusion.configs",
    "LatentActionModelConfig": "simulusion.configs",
    "DynamicsModelConfig": "simulusion.configs",
    "ATARI_100K_GAMES": "simulusion.configs",
    "HUMAN_SCORES": "simulusion.configs",
    "RANDOM_SCORES": "simulusion.configs",
    # Environments and wrappers.
    "BSuiteImageEnv": "simulusion.envs",
    "make_bsuite_env": "simulusion.envs",
    "list_available_bsuite_ids": "simulusion.envs",
    "make_atari_env": "simulusion.envs",
    "list_available_atari_envs": "simulusion.envs",
    "make_atari_vector_env": "simulusion.envs",
    "make_diamond_atari_env": "simulusion.envs.diamond_atari",
    "MuJoCoImageEnv": "simulusion.envs",
    "make_mujoco_env": "simulusion.envs",
    "make_mujoco_env_from_config": "simulusion.envs",
    "list_gymnasium_robotics_envs": "simulusion.envs",
    "make_robotics_env": "simulusion.envs",
    "register_gymnasium_robotics_envs": "simulusion.envs",
    "GymImageEnv": "simulusion.envs",
    "make_gym_env": "simulusion.envs",
    "WorldModelEnv": "simulusion.envs",
    "make_world_model_env": "simulusion.envs",
    "BraxImageEnv": "simulusion.envs",
    "make_brax_env": "simulusion.envs",
    "DeepMindControlEnv": "simulusion.envs",
    "DMLabEnv": "simulusion.envs",
    "make_dmlab_env": "simulusion.envs",
    "DMLAB_LEVELS": "simulusion.envs",
    "UnityMLAgentsEnv": "simulusion.envs",
    "make_unity_mlagents_env": "simulusion.envs",
    "MujocoEnv": "simulusion.envs",
    "TimeLimit": "simulusion.envs",
    "ActionRepeat": "simulusion.envs",
    "NormalizeActions": "simulusion.envs",
    "ObsDict": "simulusion.envs",
    "OneHotAction": "simulusion.envs",
    "RewardObs": "simulusion.envs",
    "ResizeImage": "simulusion.envs",
    "RenderImage": "simulusion.envs",
    "SelectAction": "simulusion.envs",
    # I-JEPA evaluation (paper Appendix A.2).
    "jepa_linear_probe": "simulusion.training.eval_jepa",
    "load_jepa_encoder": "simulusion.training.eval_jepa",
    # Reward/value models.
    "RewardModel": "simulusion.reward",
    "ValueModel": "simulusion.reward",
    "DreamerRewardModel": "simulusion.reward",
    "DreamerValueModel": "simulusion.reward",
    # Registry / plugin system.
    "register_world_model": "simulusion.registry",
    "deregister_world_model": "simulusion.registry",
    "get_registered_model_spec": "simulusion.registry",
    "list_registered_models": "simulusion.registry",
    "register_env_backend": "simulusion.registry",
    "deregister_env_backend": "simulusion.registry",
    "list_registered_env_backends": "simulusion.registry",
    # Deprecation helpers.
    "deprecated": "simulusion.utils.deprecation",
    "deprecated_class": "simulusion.utils.deprecation",
    "deprecated_function": "simulusion.utils.deprecation",
    # Utilities.
    "Logger": "simulusion.utils",
    "FreezeParameters": "simulusion.utils",
    "compute_return": "simulusion.utils",
    "preprocess_obs": "simulusion.utils",
    # Performance measurement and tuning.
    "ThroughputMeter": "simulusion.utils.throughput",
    "measure_steps": "simulusion.utils.throughput",
    "tensor_nbytes": "simulusion.utils.throughput",
    "enable_performance_defaults": "simulusion.utils.memory_utils",
    "maybe_compile": "simulusion.utils.memory_utils",
    "to_channels_last": "simulusion.utils.memory_utils",
    # Efficient inference and deployment.
    "InferenceModel": "simulusion.inference",
    "optimize_for_inference": "simulusion.inference",
    "inference_context": "simulusion.inference",
    "make_stepper": "simulusion.inference",
    "DreamerStepper": "simulusion.inference",
    "DreamerStepModule": "simulusion.inference",
    "IRISStepper": "simulusion.inference",
    "benchmark_step": "simulusion.inference",
    "rollout_drift": "simulusion.inference",
    "quantize_weights": "simulusion.inference",
    "save_bundle": "simulusion.inference",
    "load_bundle": "simulusion.inference",
    "STKVCache": "simulusion.blocks.st_transformer",
    # One protocol over every world model family.
    "WorldModel": "simulusion.world_model",
    "WorldModelSpec": "simulusion.world_model",
    "as_world_model": "simulusion.world_model",
    "load_world_model": "simulusion.world_model",
    "list_world_model_families": "simulusion.world_model",
    # One trainer for every family; declarative runs.
    "Trainer": "simulusion.training.trainer",
    "TrainerConfig": "simulusion.training.trainer",
    "TrainResult": "simulusion.training.trainer",
    "TrainingTask": "simulusion.training.task",
    "register_task": "simulusion.training.task",
    "list_tasks": "simulusion.training.task",
    "run_training": "simulusion.training.run",
    "MixedPrecision": "simulusion.training.precision",
    "GradientUpdate": "simulusion.training.precision",
    # One episode format for video, RL and robot data.
    "EpisodeFolder": "simulusion.data",
    "EpisodeWriter": "simulusion.data",
    "LeRobotEpisodes": "simulusion.data",
    # Agents that act through any world model.
    "CEMPlanner": "simulusion.agents",
    "PlanningAgent": "simulusion.agents",
}

_EXPORTS = {**_API_EXPORTS, **_LAZY_EXPORTS}


# Submodules that are part of the public surface and so must resolve as
# attributes of the package, not only via ``import simulusion.<name>``.
_SUBMODULE_EXPORTS = ("api",)


def __getattr__(name: str) -> Any:
    """Lazily import public symbols on first access."""

    if name in _SUBMODULE_EXPORTS:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["__version__", *_SUBMODULE_EXPORTS, *_EXPORTS]
