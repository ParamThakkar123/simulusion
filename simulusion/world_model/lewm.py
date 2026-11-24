"""LeWorldModel as a :class:`~simulusion.world_model.WorldModel`.

LeWM predicts embeddings, not pixels, rewards or episode ends: its state is
the recent history of frame embeddings and the actions between them, and what
it is for is reaching a goal (:meth:`LeWMWorldModel.goal_cost`, used by
:class:`~simulusion.agents.GoalCEMPlanner`). It has no decoder (the paper
trains one only to visualize the latent space), so :meth:`decode` raises and
rollouts run with ``decode=False``.

One protocol step is one model step: ``frameskip`` raw environment steps, whose
actions are passed concatenated (``frameskip * action_dim`` values).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from simulusion.world_model.protocol import (
    ActionSpec,
    Rollout,
    State,
    Transition,
    WorldModelBase,
    WorldModelSpec,
    module_device,
)


class LeWMWorldModel(WorldModelBase):
    """A trained :class:`~simulusion.models.lewm.LeWM` behind the shared protocol.

    State: ``emb`` ``(B, K, D)``, the last ``K <= history_size`` embeddings,
    and ``act`` ``(B, K - 1, A)``, the raw action blocks taken at all but the
    newest of them. Steps are deterministic.

    Args:
        model: The trained LeWM.
        obs_shape: ``(C, H, W)`` of the frames it reads; the encoder's input
            size by default.
    """

    family = "lewm"

    def __init__(
        self, model: Any, *, obs_shape: Optional[tuple[int, int, int]] = None
    ) -> None:
        self.model = model.eval()
        self.device = module_device(model)
        cfg = model.config
        self.history = int(cfg.history_size)
        shape = obs_shape or (
            int(cfg.in_channels),
            int(cfg.img_size),
            int(cfg.img_size),
        )
        self.spec = WorldModelSpec(
            family=self.family,
            obs_shape=(int(shape[0]), int(shape[1]), int(shape[2])),
            action=ActionSpec("continuous", int(model.action_block_dim)),
            context_frames=1,
            max_context=self.history,
            has_reward=False,
            has_termination=False,
            stochastic=False,
        )

    # ------------------------------------------------------------- latents

    def embedding(self, state: State) -> torch.Tensor:
        """The newest embedding of ``state``, ``(B, D)``."""
        return state["emb"][:, -1]

    def encode_goal(self, frames: Any) -> torch.Tensor:
        """Embeddings ``(B, D)`` of goal frames (``(B, C, H, W)`` or one frame)."""
        return self.embedding(self.encode(frames))

    def goal_cost(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Squared distance ``(B,)`` from ``state``'s embedding to goal embeddings.

        The planning objective of Eq. 4. ``goal``: ``(B, D)`` or ``(D,)``.
        """
        emb = self.embedding(state)
        return (emb - goal.to(emb).expand_as(emb)).pow(2).sum(-1)

    @property
    def action_mean(self) -> torch.Tensor:
        """Per-dimension mean of raw action blocks (the z-score centre)."""
        return self.model.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        """Per-dimension standard deviation of raw action blocks."""
        return self.model.action_std

    # ----------------------------------------------------------- protocol

    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        B, T = frames.shape[:2]
        if actions is None and T > 1:
            # Without the actions between them, earlier frames are unusable.
            frames, T = frames[:, -1:], 1
        frames = frames[:, -self.history :]
        T = frames.shape[1]
        emb = self.model.encode(frames)
        if T > 1:
            assert actions is not None
            act = actions[:, -(T - 1) :].float()
        else:
            act = emb.new_zeros(B, 0, self.spec.action.n)
        return {"emb": emb, "act": act}

    def _step(self, state: State, action: torch.Tensor) -> Transition:
        acts = torch.cat([state["act"], action.unsqueeze(1).float()], dim=1)
        nxt = self.model.predict(state["emb"], acts)[:, -1:]
        emb = torch.cat([state["emb"], nxt], dim=1)[:, -self.history :]
        keep = emb.shape[1] - 1
        act = acts[:, acts.shape[1] - keep :] if keep else acts[:, :0]
        return Transition(state={"emb": emb, "act": act})

    def _decode(self, state: State) -> torch.Tensor:
        raise NotImplementedError(
            "LeWM predicts embeddings and has no pixel decoder; roll out with "
            "decode=False and compare embeddings (goal_cost) instead."
        )

    def rollout(self, state: State, actions: Any, *, decode: bool = False) -> Rollout:
        """Imagine ``H`` steps; frames are never decoded (see :meth:`_decode`)."""
        if decode:
            self._decode(state)
        return super().rollout(state, actions, decode=False)

    def _options(self) -> dict[str, Any]:
        return {"obs_shape": list(self.spec.obs_shape)}

    def _save_weights(self, directory: Path) -> dict[str, str]:
        torch.save(
            {"config": self.model.config.to_dict(), "model": self.model.state_dict()},
            directory / "lewm.pt",
        )
        return {"model": "lewm.pt"}

    @classmethod
    def _load_weights(
        cls,
        path: Path,
        manifest: Optional[dict[str, Any]],
        device: torch.device,
        options: dict[str, Any],
    ) -> "LeWMWorldModel":
        from simulusion.configs.lewm_config import LeWMConfig
        from simulusion.models.lewm import LeWM

        file = path / "lewm.pt" if path.is_dir() else path
        if not file.exists():
            raise FileNotFoundError(f"No LeWM weights at {file}.")
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        model = LeWM(LeWMConfig.from_dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        shape = options.get("obs_shape")
        return cls(
            model.to(device),
            obs_shape=tuple(int(s) for s in shape) if shape else None,  # type: ignore[arg-type]
        )


__all__ = ["LeWMWorldModel"]
