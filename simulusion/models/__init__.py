"""
Models sub-module - Core world model implementations.

Exported Components:
    Agents (High-level training wrappers):
        - DreamerAgent: High-level Dreamer training API
        - JEPAAgent: JEPA agent for self-supervised learning
        - Planet: PlaNet planning agent
        - VisionTransformer: Vision Transformer for image encoding
        - ModularRSSM: Modular RSSM with swappable components
        - Genie: Generative Interactive Environment model

    Core Models:
        - Dreamer: Core Dreamer implementation with RSSM, actor, critic
        - RSSM: Recurrent State-Space Model (Dreamer-style)
        - RecurrentStateSpaceModel: PlaNet-style RSSM
        - LatentActionModel: Latent action learning for Genie
        - DynamicsModel: Future frame prediction for Genie

    Factory Functions:
        - create_genie, create_genie_small, create_genie_large
        - create_modular_rssm
        - create_latent_action_model, create_dynamics_model
"""

__all__ = [
    # Agents
    "Dreamer",
    "DreamerV1",
    "DreamerV2",
    "DreamerAgent",
    "DreamerV1Agent",
    "DreamerV2Agent",
    "Planet",
    "JEPAAgent",
    "IRISAgent",
    "IRISTransformer",
    "IRISWorldModel",
    "compute_lambda_return",
    "VisionTransformer",
    "ModularRSSM",
    "create_modular_rssm",
    "Genie",
    "create_genie",
    "create_genie_small",
    "create_genie_large",
    # RSSM Variants
    "RSSM",
    "RecurrentStateSpaceModel",
    "DreamerRSSM",
    # Genie Sub-components
    "LatentActionModel",
    "DynamicsModel",
    "create_latent_action_model",
    "create_dynamics_model",
]


from typing import Any


def __getattr__(name: str) -> Any:
    # Agents
    if name == "Dreamer":
        from .dreamer import Dreamer

        return Dreamer
    if name == "DreamerV1":
        from .dreamer_v1 import DreamerV1

        return DreamerV1
    if name == "DreamerV2":
        from .dreamer_v2 import DreamerV2

        return DreamerV2
    if name == "DreamerV1Agent":
        from .dreamer_v1 import DreamerV1Agent

        return DreamerV1Agent
    if name == "DreamerV2Agent":
        from .dreamer_v2 import DreamerV2Agent

        return DreamerV2Agent
    if name == "DreamerAgent":
        from .dreamer import DreamerAgent

        return DreamerAgent
    if name == "Planet":
        from .planet import Planet

        return Planet
    if name == "JEPAAgent":
        from .jepa_agent import JEPAAgent

        return JEPAAgent
    if name == "IRISAgent":
        from .iris_agent import IRISAgent

        return IRISAgent
    if name == "IRISTransformer":
        from .iris_transformer import IRISTransformer

        return IRISTransformer
    if name == "IRISWorldModel":
        from .iris_transformer import IRISWorldModel

        return IRISWorldModel
    if name == "compute_lambda_return":
        from .iris_agent import compute_lambda_return

        return compute_lambda_return
    if name == "VisionTransformer":
        from .vit import VisionTransformer

        return VisionTransformer
    if name == "ModularRSSM":
        from .modular_rssm import ModularRSSM

        return ModularRSSM
    if name == "create_modular_rssm":
        from .modular_rssm import create_modular_rssm

        return create_modular_rssm
    if name == "Genie":
        from .genie import Genie

        return Genie
    if name == "create_genie":
        from .genie import create_genie

        return create_genie
    if name == "create_genie_small":
        from .genie import create_genie_small

        return create_genie_small
    if name == "create_genie_large":
        from .genie import create_genie_large

        return create_genie_large

    # RSSM Variants
    if name == "RSSM":
        from .dreamer_rssm import RSSM

        return RSSM
    if name == "DreamerRSSM":
        from .dreamer_rssm import RSSM

        return RSSM
    if name == "RecurrentStateSpaceModel":
        from .rssm import RecurrentStateSpaceModel

        return RecurrentStateSpaceModel

    # Genie Sub-components
    if name == "LatentActionModel":
        from .latent_action_model import LatentActionModel

        return LatentActionModel
    if name == "DynamicsModel":
        from .dynamics_model import DynamicsModel

        return DynamicsModel
    if name == "create_latent_action_model":
        from .latent_action_model import create_latent_action_model

        return create_latent_action_model
    if name == "create_dynamics_model":
        from .dynamics_model import create_dynamics_model

        return create_dynamics_model

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
