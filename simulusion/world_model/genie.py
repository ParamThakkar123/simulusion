"""Genie as a :class:`~simulusion.world_model.WorldModel`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from simulusion.world_model.protocol import (
    ActionSpec,
    State,
    Transition,
    WorldModelBase,
    WorldModelSpec,
    module_device,
)


class GenieWorldModel(WorldModelBase):
    """Genie: video tokens in, latent actions drive the next frame.

    Genie learns its actions from unlabelled video, so the action space is
    ``action_vocab_size`` *latent* actions (``spec.action.latent`` is set):
    what each one does is whatever the model discovered. Without ``actions``,
    ``encode`` labels the context's transitions with the actions Genie infers
    itself. Genie has no reward or termination head.

    The dynamics model sees at most ``num_frames`` frames, so the state keeps a
    rolling window of that many: ``tokens`` ``(B, T, N)`` and ``actions``
    ``(B, T - 1)``. ``decode`` decodes the whole window, because the
    tokenizer's decoder attends over time, and returns its last frame.

    Args:
        model: A :class:`~simulusion.models.genie.Genie`.
        temperature: Sampling temperature for next-frame tokens. ``None`` keeps
            the model's (2.0 in the paper); ``0`` takes the most likely token.
    """

    family = "genie"

    def __init__(self, model: Any, *, temperature: Optional[float] = None) -> None:
        if not (hasattr(model, "dynamics_model") and hasattr(model, "video_tokenizer")):
            raise TypeError(f"{type(model).__name__} is not a Genie model.")
        model.eval()
        self.model = model
        self.device = module_device(model)
        self.window = int(model.num_frames)
        self.temperature = float(
            model.sampler.temperature if temperature is None else temperature
        )
        size = int(model.image_size)
        self.spec = WorldModelSpec(
            family=self.family,
            obs_shape=(3, size, size),
            action=ActionSpec("discrete", int(model.action_vocab_size), latent=True),
            context_frames=1,
            max_context=self.window,
            has_reward=False,
            has_termination=False,
            stochastic=self.temperature > 0,
        )

    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        frames = frames[:, -self.window :]
        video = frames.permute(0, 2, 1, 3, 4)  # Genie reads (B, C, T, H, W)
        tokens = self.model.encode_video(video)
        B, T = tokens.shape[:2]
        if actions is not None:
            acts = actions[:, actions.shape[1] - (T - 1) :]
        elif T > 1:
            inferred = self.model.infer_actions(video).long()
            if inferred.dim() == 1:
                # Mean pooling infers one action for the whole clip (the last
                # transition's); every context transition gets it.
                inferred = inferred.unsqueeze(1).expand(B, T - 1)
            acts = inferred[:, : T - 1]
        else:
            acts = torch.zeros(B, 0, dtype=torch.long, device=self.device)
        return {"tokens": tokens, "actions": acts}

    def _step(self, state: State, action: torch.Tensor) -> Transition:
        tokens = state["tokens"]
        actions = torch.cat([state["actions"], action.unsqueeze(1)], dim=1)
        logits = self.model.dynamics_model(tokens, actions, mask_prob=0.0)[:, -1]
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            B, N, V = probs.shape
            next_tokens = torch.multinomial(probs.reshape(B * N, V), 1).reshape(B, N)
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)[:, -self.window :]
        actions = actions[:, actions.shape[1] - (tokens.shape[1] - 1) :]
        return Transition(state={"tokens": tokens, "actions": actions})

    def _decode(self, state: State) -> torch.Tensor:
        tokenizer = self.model.video_tokenizer
        video = tokenizer.decode(tokenizer.decode_indices(state["tokens"]))
        return video[:, :, -1]  # (B, C, T, H, W) -> the window's last frame

    def _options(self) -> dict[str, Any]:
        return {"temperature": self.temperature}

    def _save_weights(self, directory: Path) -> dict[str, str]:
        self.model.save_pretrained(directory / "model.pt")
        return {"model": "model.pt", "config": "config.yaml"}

    @classmethod
    def _load_weights(
        cls,
        path: Path,
        manifest: Optional[dict[str, Any]],
        device: torch.device,
        options: dict[str, Any],
    ) -> "GenieWorldModel":
        from simulusion.models.genie import Genie

        model = Genie.from_pretrained(path, map_location=device).to(device)
        return cls(model, **options)
