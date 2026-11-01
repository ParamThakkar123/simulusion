"""One interface for every world model in Simulusion.

.. code-block:: python

    import torch
    from simulusion.world_model import as_world_model, load_world_model

    model = as_world_model(trained_agent)          # or load_world_model(path)
    state = model.encode(frames, actions)          # real context
    traj = model.rollout(state, planned_actions)   # imagined future
    traj.frames, traj.rewards, traj.terminated

    model.save("runs/my_model/world_model")        # any family, same format
    model = load_world_model("runs/my_model/world_model")

:class:`WorldModel` is the protocol; see :mod:`simulusion.world_model.protocol`
for the conventions (frame layout and range, action layout, state, seeding)
shared by every family. Supported families:

========================  =================================================
``dreamer``               :class:`DreamerWorldModel` (Dreamer V1 and V2)
``planet``                :class:`PlanetWorldModel`
``iris``                  :class:`IRISTokenWorldModel`
``diamond``               :class:`DiamondWorldModel`
``genie``                 :class:`GenieWorldModel` (latent actions)
``world-models``          :class:`MDRNNWorldModel` (ConvVAE + MDN-RNN)
========================  =================================================

JEPA and DiT are not world models in this sense: I-JEPA learns image
representations without actions, and DiT generates images from a class
label, so neither predicts what an action does. They have no adapter.

The protocol covers the model, not the agent: policies, planners and
trainers stay with each family. Steppers
(:mod:`simulusion.inference.steppers`) remain the acting interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import torch

from simulusion.world_model.protocol import (
    MANIFEST_NAME,
    ActionSpec,
    Rollout,
    State,
    Transition,
    WorldModel,
    WorldModelBase,
    WorldModelSpec,
    read_manifest,
)

if TYPE_CHECKING:
    from simulusion.world_model.diamond import DiamondWorldModel
    from simulusion.world_model.dreamer import DreamerWorldModel
    from simulusion.world_model.genie import GenieWorldModel
    from simulusion.world_model.iris import IRISTokenWorldModel
    from simulusion.world_model.planet import PlanetWorldModel
    from simulusion.world_model.vae_mdrnn import MDRNNWorldModel

_FAMILIES: Mapping[str, tuple[str, str]] = {
    "dreamer": ("simulusion.world_model.dreamer", "DreamerWorldModel"),
    "planet": ("simulusion.world_model.planet", "PlanetWorldModel"),
    "iris": ("simulusion.world_model.iris", "IRISTokenWorldModel"),
    "diamond": ("simulusion.world_model.diamond", "DiamondWorldModel"),
    "genie": ("simulusion.world_model.genie", "GenieWorldModel"),
    "world-models": ("simulusion.world_model.vae_mdrnn", "MDRNNWorldModel"),
}

#: Accepted spellings of each family name.
_ALIASES = {
    "dreamer-v1": "dreamer",
    "dreamerv1": "dreamer",
    "dreamer-v2": "dreamer",
    "dreamerv2": "dreamer",
    "worldmodels": "world-models",
    "vae-mdrnn": "world-models",
    "mdrnn": "world-models",
}

WORLD_MODEL_FAMILIES: tuple[str, ...] = tuple(_FAMILIES)

_NOT_WORLD_MODELS = {
    "JEPAAgent": "I-JEPA learns image representations without actions",
    "DiT": "DiT generates images from a class label, not from actions",
}


def list_world_model_families() -> list[str]:
    """The family names :func:`load_world_model` and :func:`world_model_class` accept."""
    return list(WORLD_MODEL_FAMILIES)


def world_model_class(family: str) -> type[WorldModelBase]:
    """The :class:`WorldModelBase` subclass for ``family``."""
    from importlib import import_module

    key = family.strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    if key not in _FAMILIES:
        raise ValueError(
            f"Unknown world model family {family!r}; choose one of "
            f"{', '.join(WORLD_MODEL_FAMILIES)}."
        )
    module, name = _FAMILIES[key]
    cls: type[WorldModelBase] = getattr(import_module(module), name)
    return cls


def as_world_model(model: Any, *args: Any, **kwargs: Any) -> WorldModel:
    """Wrap a trained model of any supported family as a :class:`WorldModel`.

    Accepts the agent or the bare model: a ``Dreamer``/``DreamerAgent``,
    ``Planet`` or its RSSM, ``IRISAgent``, ``DiamondAgent``, ``Genie``, or a
    ``(ConvVAE, MDRNN)`` pair. Something that already implements
    :class:`WorldModel` is returned as is. Extra arguments go to the family's
    constructor, e.g. ``temperature=0`` for Genie.
    """
    if isinstance(model, WorldModel):
        return model
    # Each family's constructor takes its own arguments.
    if isinstance(model, (tuple, list)) and len(model) == 2:
        pair: Any = world_model_class("world-models")
        return pair(*model, *args, **kwargs)
    cls: Any = world_model_class(_detect_family(model))
    return cls(model, *args, **kwargs)


def load_world_model(
    path: str | Path,
    family: Optional[str] = None,
    *,
    device: str | torch.device | None = None,
    **options: Any,
) -> WorldModel:
    """Load a world model saved by :meth:`WorldModel.save`, or a training checkpoint.

    A directory written by ``save`` names its family in ``world_model.json``.
    Anything else is read as ``family``'s own checkpoint (a file, a run
    directory, or for most families a Hugging Face Hub repo id), so
    ``family`` is required then. ``options`` override the saved constructor
    options, e.g. ``temperature=0.5``.
    """
    manifest = read_manifest(path)
    if manifest is not None:
        saved = str(manifest["family"])
        if family is not None and world_model_class(family) is not world_model_class(
            saved
        ):
            raise ValueError(f"{path} holds a {saved!r} world model, not {family!r}.")
        family = saved
    if family is None:
        raise ValueError(
            f"{path} has no {MANIFEST_NAME}, so pass the family it was trained "
            f"as: one of {', '.join(WORLD_MODEL_FAMILIES)}."
        )
    model: WorldModel = world_model_class(family).load(path, device=device, **options)
    return model


def _detect_family(model: Any) -> str:
    name = type(model).__name__
    if name in _NOT_WORLD_MODELS:
        raise TypeError(
            f"{name} has no world model adapter: {_NOT_WORLD_MODELS[name]}, so "
            "it does not predict what an action does."
        )
    if hasattr(model, "video_tokenizer") and hasattr(model, "dynamics_model"):
        return "genie"
    if hasattr(model, "diffusion_model") and hasattr(model, "sampler"):
        return "diamond"
    if hasattr(model, "transformer") and hasattr(model, "forward_actor_critic"):
        return "iris"
    inner = getattr(model, "dreamer", model)
    if all(hasattr(inner, a) for a in ("rssm", "obs_encoder", "actor")):
        return "dreamer"
    rssm = getattr(model, "rssm", model)
    if all(hasattr(rssm, a) for a in ("encoder", "decoder", "grucell")):
        return "planet"
    raise TypeError(
        f"No world model adapter for {name}. Supported families: "
        f"{', '.join(WORLD_MODEL_FAMILIES)}; pass a (ConvVAE, MDRNN) pair for "
        "World Models, or implement the WorldModel protocol."
    )


def __getattr__(name: str) -> Any:
    # The family classes import their models lazily, on first use.
    for module, cls_name in _FAMILIES.values():
        if name == cls_name:
            from importlib import import_module

            return getattr(import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: Sequence[str] = [
    "ActionSpec",
    "DiamondWorldModel",
    "DreamerWorldModel",
    "GenieWorldModel",
    "IRISTokenWorldModel",
    "MDRNNWorldModel",
    "PlanetWorldModel",
    "Rollout",
    "State",
    "Transition",
    "WORLD_MODEL_FAMILIES",
    "WorldModel",
    "WorldModelBase",
    "WorldModelSpec",
    "as_world_model",
    "list_world_model_families",
    "load_world_model",
    "world_model_class",
]
