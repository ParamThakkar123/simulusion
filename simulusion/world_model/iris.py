"""IRIS as a :class:`~simulusion.world_model.WorldModel`."""

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


class IRISTokenWorldModel(WorldModelBase):
    """IRIS's discrete autoencoder and Transformer, one frame at a time.

    The state is the token history: ``tokens`` ``(B, T, K)`` for the last
    ``T`` frames and ``actions`` ``(B, T - 1)`` between them. ``step`` primes a
    fresh key/value cache with that history, then generates the next frame's
    ``K`` tokens and reads the reward and termination heads at the action
    position, through the same Transformer calls
    :meth:`~simulusion.models.iris_agent.IRISAgent.imagine_rollout` makes.
    The history keeps at most :attr:`WorldModelSpec.max_context` frames, the
    window ``imagine_rollout`` falls back to when its cache fills up.

    Rewards come out on the scale IRIS was trained on (sign-transformed by
    default, see ``IRISConfig.reward_transform``).

    The agent is switched to eval mode: its quantizer revives dead codes in
    training mode, which would rewrite the codebook on every call.

    Args:
        agent: An :class:`~simulusion.models.iris_agent.IRISAgent`.
        temperature: Softmax temperature for sampling next-frame tokens.
        sample: Sample tokens (default, as IRIS imagines); ``False`` takes
            the most likely token, which makes ``step`` deterministic.
    """

    family = "iris"

    def __init__(
        self, agent: Any, *, temperature: float = 1.0, sample: bool = True
    ) -> None:
        for attr in ("encoder", "decoder", "transformer", "config"):
            if not hasattr(agent, attr):
                raise TypeError(f"{type(agent).__name__} is not an IRIS agent.")
        agent.eval()
        self.agent = agent
        self.temperature = float(temperature)
        self.sample = bool(sample)
        self.device = torch.device(agent.device)
        config = agent.config
        self.tokens_per_frame = K = int(config.tokens_per_frame)
        self.grid = int(round(K**0.5))
        capacity = (int(agent.transformer.max_seq_len) - K) // (K + 1)
        window = max(1, min(int(config.transformer_timesteps), capacity))
        self.spec = WorldModelSpec(
            family=self.family,
            obs_shape=tuple(int(d) for d in config.get_frame_shape()),  # type: ignore[arg-type]
            action=ActionSpec("discrete", int(agent.action_size)),
            context_frames=1,
            max_context=window,
            has_reward=True,
            has_termination=True,
            stochastic=self.sample,
        )

    def _tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) in [0, 1] -> token indices (B, T, K)."""
        B, T = frames.shape[:2]
        _, tokens, _ = self.agent.encoder(frames.flatten(0, 1))
        return tokens.reshape(B, T, self.tokens_per_frame)

    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        window = int(self.spec.max_context or 1)
        if actions is None:
            # The Transformer conditions on the actions between frames; without
            # them only the last frame is usable context.
            frames = frames[:, -1:]
        tokens = self._tokens(frames[:, -window:])
        B, T = tokens.shape[:2]
        acts = (
            actions[:, actions.shape[1] - (T - 1) :]
            if actions is not None and T > 1
            else torch.zeros(B, 0, dtype=torch.long, device=self.device)
        )
        return {"tokens": tokens, "actions": acts}

    def _step(self, state: State, action: torch.Tensor) -> Transition:
        transformer = self.agent.transformer
        tokens, actions = state["tokens"], state["actions"]
        B, T, K = tokens.shape
        cache = transformer.init_cache(B, self.device, max_len=T * (K + 1) + K)
        pos = transformer.prime_cache(
            tokens, actions if actions.shape[1] else None, cache, start_pos=0
        )
        _, next_tokens, action_hidden, _ = transformer.generate_frame_cached(
            action,
            cache,
            start_pos=pos,
            sample=self.sample,
            temperature=self.temperature,
        )
        reward = transformer.expected_reward(action_hidden).reshape(B)
        term_logits = transformer.termination_head(action_hidden)
        window = int(self.spec.max_context or 1)
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)[:, -window:]
        actions = torch.cat([actions, action.unsqueeze(1)], dim=1)
        actions = actions[:, actions.shape[1] - (tokens.shape[1] - 1) :]
        return Transition(
            state={"tokens": tokens, "actions": actions},
            reward=reward,
            terminated=term_logits.argmax(dim=-1) == 1,
            info={"termination_prob": torch.softmax(term_logits, dim=-1)[:, 1]},
        )

    def _decode(self, state: State) -> torch.Tensor:
        last = state["tokens"][:, -1]
        grid = last.reshape(last.shape[0], self.grid, self.grid)
        return self.agent.decoder(self.agent.encoder.quantizer.decode_indices(grid))

    def _options(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "sample": self.sample}

    def _save_weights(self, directory: Path) -> dict[str, str]:
        self.agent.save(str(directory / "model.pt"))
        return {"model": "model.pt", "config": "config.yaml"}

    @classmethod
    def _load_weights(
        cls,
        path: Path,
        manifest: Optional[dict[str, Any]],
        device: torch.device,
        options: dict[str, Any],
    ) -> "IRISTokenWorldModel":
        from simulusion.models.iris_agent import IRISAgent

        agent = IRISAgent.from_pretrained(path, device=device)
        return cls(agent, **options)
