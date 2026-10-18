"""DIAMOND as a :class:`~simulusion.world_model.WorldModel`."""

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


def build_diamond_networks(config: Any, action_dim: int, device: Any) -> tuple:
    """DIAMOND's world-model networks: ``(diffusion_model, sampler, reward_model)``.

    The one place they are built, so a :class:`DiamondAgent` and a
    :class:`DiamondWorldModel` loaded from its checkpoint are the same networks.
    """
    from simulusion.models.diffusion.diamond_diffusion import (
        DiffusionUNet,
        EDMPreconditioner,
        EulerSampler,
    )
    from simulusion.models.diffusion.reward_termination import RewardTerminationModel

    diffusion_model = DiffusionUNet(
        obs_channels=3,
        num_conditioning_frames=config.num_conditioning_frames,
        # config.diffusion_channels lists absolute channel sizes per level;
        # DiffusionUNet takes a base width and multipliers relative to it.
        base_channels=config.diffusion_channels[0],
        channel_multipliers=tuple(
            int(c // config.diffusion_channels[0]) for c in config.diffusion_channels
        ),
        num_res_blocks=config.diffusion_res_blocks,
        cond_dim=config.diffusion_cond_dim,
        action_dim=action_dim,
    ).to(device)
    sampler = EulerSampler(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        num_steps=config.num_sampling_steps,
        edm_precond=EDMPreconditioner(
            sigma_data=config.sigma_data,
            p_mean=config.p_mean,
            p_std=config.p_std,
        ),
    )
    reward_model = RewardTerminationModel(
        obs_channels=3,
        action_dim=action_dim,
        channels=tuple(config.reward_channels),
        lstm_dim=config.reward_lstm_dim,
        cond_dim=config.reward_cond_dim,
        res_blocks=config.reward_res_blocks,
        frame_size=config.obs_size,
    ).to(device)
    return diffusion_model, sampler, reward_model


class DiamondWorldModel(WorldModelBase):
    """DIAMOND's diffusion model and reward/termination model.

    Each frame is sampled by the diffusion model conditioned on the last
    ``L = num_conditioning_frames`` frames and the actions taken at them,
    exactly as in DIAMOND's imagination: the reward/termination model scores
    ``(x_i, a_i)``, then the sampler draws ``x_{i+1}``. ``encode`` needs ``L``
    frames; without ``actions`` the context is taken to have been reached
    with action ``0`` (NOOP in Atari). Rewards are DIAMOND's clipped classes
    ``{-1, 0, 1}``.

    Args:
        model: A :class:`~simulusion.training.train_diamond.DiamondAgent`, or
            any object with its ``config``, ``diffusion_model``, ``sampler``
            and ``reward_model`` (``None`` for a model trained without one).
        sampling_steps: Override the Euler denoising steps per frame. Fewer
            is faster and blurrier; the paper uses 3.
    """

    family = "diamond"

    def __init__(self, model: Any, *, sampling_steps: Optional[int] = None) -> None:
        for attr in ("diffusion_model", "sampler", "config"):
            if not hasattr(model, attr):
                raise TypeError(f"{type(model).__name__} is not a DIAMOND model.")
        self.config = model.config
        self.diffusion_model = model.diffusion_model.eval()
        self.reward_model = getattr(model, "reward_model", None)
        if self.reward_model is not None:
            self.reward_model.eval()
        self.sampler = model.sampler
        self.sampling_steps = sampling_steps
        if sampling_steps is not None:
            from simulusion.models.diffusion.diamond_diffusion import EulerSampler

            self.sampler = EulerSampler(
                sigma_min=self.sampler.sigma_min,
                sigma_max=self.sampler.sigma_max,
                rho=self.sampler.rho,
                num_steps=int(sampling_steps),
                edm_precond=self.sampler.edm_precond,
            )
        self.device = module_device(self.diffusion_model)
        self.context = int(self.config.num_conditioning_frames)
        self.size = int(self.config.obs_size)
        self.spec = WorldModelSpec(
            family=self.family,
            obs_shape=(3, self.size, self.size),
            action=ActionSpec(
                "discrete", int(self.diffusion_model.action_embed.num_embeddings)
            ),
            context_frames=self.context,
            max_context=self.context,
            has_reward=self.reward_model is not None,
            has_termination=self.reward_model is not None,
        )

    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        L = self.context
        B = frames.shape[0]
        window = frames[:, -L:] * 2.0 - 1.0  # DIAMOND works in [-1, 1]
        prev = (
            actions[:, -(L - 1) :]
            if actions is not None and L > 1
            else torch.zeros(B, L - 1, dtype=torch.long, device=self.device)
        )
        state: State = {"obs_window": window, "prev_actions": prev}
        if self.reward_model is not None:
            # Burn the reward model's LSTM in on the context, stopping one
            # frame short: the first step scores the last context frame with
            # the action chosen there.
            h, c = self.reward_model.init_hidden(B, self.device)
            if L > 1:
                _, _, (h, c) = self.reward_model(window[:, :-1], prev, (h, c))
            # Batch on dim 0, like every state tensor; the LSTM wants dim 1.
            state["reward_h"], state["reward_c"] = h.transpose(0, 1), c.transpose(0, 1)
        return state

    def _step(self, state: State, action: torch.Tensor) -> Transition:
        window = state["obs_window"]
        B = window.shape[0]
        reward = terminated = None
        next_state: State = {}
        if self.reward_model is not None:
            hidden = (
                state["reward_h"].transpose(0, 1).contiguous(),
                state["reward_c"].transpose(0, 1).contiguous(),
            )
            reward, terminated, (h, c) = self.reward_model.predict(
                obs=window[:, -1], actions=action, hidden_state=hidden
            )
            reward, terminated = reward.reshape(B), terminated.reshape(B)
            next_state["reward_h"], next_state["reward_c"] = (
                h.transpose(0, 1),
                c.transpose(0, 1),
            )
        actions = torch.cat([state["prev_actions"], action.unsqueeze(1)], dim=1)
        sampled = self.sampler.sample(
            model=self.diffusion_model,
            shape=(B, 3, self.size, self.size),
            device=self.device,
            obs_history=window,
            actions=actions,
        )
        next_state["obs_window"] = torch.cat(
            [window[:, 1:], sampled.unsqueeze(1)], dim=1
        )
        next_state["prev_actions"] = actions[:, 1:]
        return Transition(state=next_state, reward=reward, terminated=terminated)

    def _decode(self, state: State) -> torch.Tensor:
        return (state["obs_window"][:, -1] + 1.0) / 2.0

    def _options(self) -> dict[str, Any]:
        return (
            {}
            if self.sampling_steps is None
            else {"sampling_steps": self.sampling_steps}
        )

    def _save_weights(self, directory: Path) -> dict[str, str]:
        torch.save(
            {
                "config": self.config.to_dict(),
                "action_dim": self.spec.action.n,
                "diffusion_model": self.diffusion_model.state_dict(),
                "reward_model": (
                    None
                    if self.reward_model is None
                    else self.reward_model.state_dict()
                ),
            },
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
    ) -> "DiamondWorldModel":
        from types import SimpleNamespace

        from simulusion.configs.diamond_config import DiamondConfig
        from simulusion.models.model_io import resolve_pretrained_file

        file = resolve_pretrained_file(
            path, ("model.pt", "checkpoint.pt", "diamond.pt", "pytorch_model.bin")
        )
        if file is None:
            raise FileNotFoundError(f"No DIAMOND checkpoint found at {str(path)!r}.")
        # A world-model save or a full DiamondAgent training checkpoint: both
        # hold the config and the two networks' weights.
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        config = DiamondConfig.from_dict(checkpoint["config"])
        weights = checkpoint["diffusion_model"]
        action_dim = int(
            checkpoint.get("action_dim") or weights["action_embed.weight"].shape[0]
        )
        diffusion_model, sampler, reward_model = build_diamond_networks(
            config, action_dim, device
        )
        diffusion_model.load_state_dict(weights)
        if checkpoint.get("reward_model") is not None:
            reward_model.load_state_dict(checkpoint["reward_model"])
        else:
            reward_model = None
        networks = SimpleNamespace(
            config=config,
            diffusion_model=diffusion_model,
            sampler=sampler,
            reward_model=reward_model,
        )
        return cls(networks, **options)
