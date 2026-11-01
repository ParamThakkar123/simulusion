"""PlaNet as a :class:`~simulusion.world_model.WorldModel`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from simulusion.world_model.protocol import (
    ActionSpec,
    State,
    Transition,
    WorldModelBase,
    WorldModelSpec,
    module_device,
)


class PlanetWorldModel(WorldModelBase):
    """PlaNet's recurrent state-space model.

    ``encode`` filters the context with the posterior mean, the way PlaNet's
    CEM planner (:class:`~simulusion.controller.rssm_policy.RSSMPolicy`)
    tracks the real episode; ``step`` advances the deterministic state and
    samples the prior, as the planner's candidate rollouts do. Frames are
    quantised to ``bit_depth`` bits first, matching PlaNet's preprocessing
    without its training-time dequantisation noise. PlaNet has a reward head
    but no termination head; its continuous actions are unbounded.

    Args:
        model: A :class:`~simulusion.models.planet.Planet` or its
            :class:`~simulusion.models.rssm.RecurrentStateSpaceModel`.
        bit_depth: Bits per colour channel the model was trained on (PlaNet's
            default is 5).
    """

    family = "planet"

    def __init__(self, model: Any, *, bit_depth: Optional[int] = None) -> None:
        rssm = getattr(model, "rssm", model)
        for attr in ("encoder", "decoder", "grucell", "state_prior"):
            if not hasattr(rssm, attr):
                raise TypeError(f"{type(model).__name__} is not a PlaNet model.")
        rssm.eval()
        self.rssm = rssm
        self.bit_depth = int(
            bit_depth if bit_depth is not None else getattr(model, "bit_depth", 5)
        )
        self.device = module_device(rssm)
        self.spec = WorldModelSpec(
            family=self.family,
            obs_shape=(3, 64, 64),
            action=ActionSpec("continuous", int(rssm.action_size)),
            context_frames=1,
            max_context=None,
            has_reward=True,
            has_termination=False,
        )

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """[0, 1] -> PlaNet's quantised [-0.5, 0.5] (``preprocess_img`` minus noise)."""
        pixels = torch.round(frames * 255.0)  # back to the integers PlaNet saw
        levels = 2**self.bit_depth
        return torch.floor(pixels / 2 ** (8 - self.bit_depth)) / levels - 0.5

    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        B, T = frames.shape[:2]
        rssm = self.rssm
        h: Optional[torch.Tensor] = None
        s: Optional[torch.Tensor] = None
        a: Optional[torch.Tensor] = None
        for t in range(T):
            if t > 0:
                a = (
                    actions[:, t - 1]
                    if actions is not None
                    else torch.zeros(B, rssm.action_size, device=self.device)
                )
            embed = rssm.encoder(self._preprocess(frames[:, t]))
            h, s = rssm.get_init_state(embed, h, s, a)
        assert h is not None and s is not None
        return {"deter": h, "stoch": s}

    def _step(self, state: State, action: torch.Tensor) -> Transition:
        rssm = self.rssm
        h = rssm.deterministic_state_fwd(state["deter"], state["stoch"], action)
        s = rssm.state_prior(h, sample=True)
        assert isinstance(s, torch.Tensor)
        reward = rssm.pred_reward(h, s).reshape(-1)
        return Transition(state={"deter": h, "stoch": s}, reward=reward)

    def _decode(self, state: State) -> torch.Tensor:
        # Positionally (deter, stoch), as train_planet calls it: the decoder's
        # parameter names say (latent, state), but training is what the
        # weights learned.
        return self.rssm.decoder(state["deter"], state["stoch"]) + 0.5

    def _options(self) -> dict[str, Any]:
        return {"bit_depth": self.bit_depth}

    def _save_weights(self, directory: Path) -> dict[str, str]:
        rssm = self.rssm
        torch.save(
            {"sizes": _rssm_sizes(rssm.state_dict()), "rssm": rssm.state_dict()},
            directory / "model.pt",
        )
        return {"model": "model.pt"}

    @classmethod
    def _load_weights(
        cls,
        path: Path,
        manifest: Optional[dict[str, Any]],
        device: torch.device,
        options: dict[str, Any],
    ) -> "PlanetWorldModel":
        from simulusion.models.rssm import RecurrentStateSpaceModel

        file = path / "model.pt" if path.is_dir() else path
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        # Planet.train writes the bare RSSM state dict (ckpt_<epoch>.pth).
        weights = checkpoint.get("rssm", checkpoint)
        sizes: dict[str, Any] = checkpoint.get("sizes") or _rssm_sizes(weights)
        rssm = RecurrentStateSpaceModel(**sizes)
        rssm.load_state_dict(weights)
        return cls(rssm.to(device), bit_depth=options.get("bit_depth"))


def _rssm_sizes(weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Constructor arguments of a RecurrentStateSpaceModel, read off its weights."""
    state_size = int(weights["grucell.weight_hh"].shape[1])
    latent_size = int(weights["fc_prior_m.weight"].shape[0])
    return {
        "action_size": int(weights["lat_act_layer.weight"].shape[1]) - latent_size,
        "state_size": state_size,
        "latent_size": latent_size,
        "hidden_size": int(weights["fc_prior_1.weight"].shape[0]),
        "embed_size": int(weights["fc_posterior_1.weight"].shape[1]) - state_size,
    }
