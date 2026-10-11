"""
Diffusion sub-module - Diffusion model components for world models.

Exported Components:
    - DiT: Diffusion Transformer model
    - PatchEmbed: Image patch embedding
    - PatchUnEmbed: Patch unembedding (decode tokens to image)
    - DDPM: Denoising Diffusion Probabilistic Model implementation
    - ActorCriticNetwork: DIAMOND actor-critic network
    - RewardTerminationModel: Reward/termination prediction model
    - sinusoidal_time_embedding: Time embedding for diffusion models
"""

__all__ = [
    "DiT",
    "create_dit",
    "PatchEmbed",
    "PatchUnEmbed",
    "DDPM",
    "ActorCriticNetwork",
    "RewardTerminationModel",
    "sinusoidal_time_embedding",
]


from typing import Any


def __getattr__(name: str) -> Any:
    if name == "DiT":
        from .DiT import DiT

        return DiT
    if name == "create_dit":
        from .DiT import create_dit

        return create_dit
    if name == "PatchEmbed":
        from .DiT import PatchEmbed

        return PatchEmbed
    if name == "PatchUnEmbed":
        from .DiT import PatchUnEmbed

        return PatchUnEmbed
    if name == "DDPM":
        from .DDPM import DDPM

        return DDPM
    if name == "ActorCriticNetwork":
        from .actor_critic import ActorCriticNetwork

        return ActorCriticNetwork
    if name == "RewardTerminationModel":
        from .reward_termination import RewardTerminationModel

        return RewardTerminationModel
    if name == "sinusoidal_time_embedding":
        from .DiT import sinusoidal_time_embedding

        return sinusoidal_time_embedding

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
