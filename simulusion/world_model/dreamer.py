"""Dreamer (V1 and V2) as a :class:`~simulusion.world_model.WorldModel`."""

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
)


class DreamerWorldModel(WorldModelBase):
    """The RSSM of a Dreamer agent, with its decoder, reward and discount heads.

    ``encode`` filters the context through the posterior, ``step`` samples the
    prior, exactly as :class:`~simulusion.inference.steppers.DreamerStepper`
    does (and so as Dreamer's own imagination does). Actions are continuous in
    ``[-1, 1]``. ``terminated`` is the discount head's continue probability
    falling below ``0.5``; models trained without that head have none.

    Args:
        model: A :class:`~simulusion.models.dreamer.Dreamer`, ``DreamerV2`` or
            the ``DreamerAgent`` wrapping one.
    """

    family = "dreamer"

    def __init__(self, model: Any) -> None:
        from simulusion.inference.steppers import DreamerStepper

        self.stepper = DreamerStepper(model)
        self.dreamer = self.stepper.dreamer
        for name in ("rssm", "obs_encoder", "obs_decoder", "reward_model"):
            getattr(self.dreamer, name).eval()
        if self.stepper.discount_model is not None:
            self.stepper.discount_model.eval()
        self.device = self.stepper.device
        self.spec = WorldModelSpec(
            family=self.family,
            obs_shape=tuple(int(d) for d in self.dreamer.obs_encoder.input_shape),  # type: ignore[arg-type]
            action=ActionSpec("continuous", self.stepper.action_size, -1.0, 1.0),
            context_frames=1,
            max_context=None,
            has_reward=True,
            has_termination=self.stepper.discount_model is not None,
        )

    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        return self._encode_steps(frames, actions)[-1]

    def _encode_steps(
        self, frames: torch.Tensor, actions: Optional[torch.Tensor]
    ) -> list[State]:
        B, T = frames.shape[:2]
        stepper = self.stepper
        state = stepper.init_state(B)
        prev = torch.zeros(B, self.spec.action.n, device=self.device)
        states = []
        for t in range(T):
            if t > 0 and actions is not None:
                prev = actions[:, t - 1].clamp(-1.0, 1.0)
            # DreamerStepper.observe takes raw [0, 255] pixels.
            state = stepper.observe(state, frames[:, t] * 255.0, prev)
            states.append(state)
        return states

    def _step(self, state: State, action: torch.Tensor) -> Transition:
        out = self.stepper.imagine(state, action)
        terminated, info = None, {}
        if out.continue_prob is not None:
            info["continue_prob"] = out.continue_prob
            terminated = out.continue_prob < 0.5
        return Transition(
            state=out.state, reward=out.reward, terminated=terminated, info=info
        )

    def _decode(self, state: State) -> torch.Tensor:
        # Dreamer reconstructs in [-0.5, 0.5].
        return self.stepper.decode(state) + 0.5

    def _options(self) -> dict[str, Any]:
        return {"variant": type(self.dreamer).__name__}

    def _save_weights(self, directory: Path) -> dict[str, str]:
        save = getattr(self.dreamer, "save", None)
        if not callable(save):
            raise TypeError(
                f"{type(self.dreamer).__name__} cannot be saved; wrap a Dreamer."
            )
        save(str(directory / "model.pt"))
        return {"model": "model.pt", "config": "config.yaml"}

    @classmethod
    def _load_weights(
        cls,
        path: Path,
        manifest: Optional[dict[str, Any]],
        device: torch.device,
        options: dict[str, Any],
    ) -> "DreamerWorldModel":
        from simulusion.models.dreamer import Dreamer

        model_cls: Any = Dreamer
        variant = options.get("variant") or _variant_of_checkpoint(path)
        if variant == "DreamerV2":
            from simulusion.models.dreamer_v2 import DreamerV2

            model_cls = DreamerV2
        # Dreamer.from_pretrained reads checkpoint files, run directories and
        # Hugging Face Hub repos alike.
        return cls(model_cls.from_pretrained(path, map_location=device))


def _variant_of_checkpoint(path: Path) -> Optional[str]:
    """``"DreamerV2"`` for a V2 training checkpoint (its config says ``algo: Dreamerv2``).

    V2's heads have a different shape, so the class must be right before the
    weights load. ``None`` when the checkpoint cannot be read here (a Hub id).
    """
    from simulusion.models.model_io import resolve_pretrained_file

    if not path.exists():
        return None
    file = resolve_pretrained_file(
        path, ("model.pt", "pytorch_model.bin", "checkpoint.pt", "ckpt.pt")
    )
    if file is None:
        return None
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    algo = str((config or {}).get("algo", "")).lower()
    return "DreamerV2" if algo == "dreamerv2" else None
