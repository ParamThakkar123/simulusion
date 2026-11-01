import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from simulusion.blocks.st_transformer import STKVCache, STTransformer
from simulusion.export import ExportableAgentMixin


class MaskGITSampler:
    """MaskGIT sampling for token-based video generation.

    Uses iterative refinement with a mask schedule to progressively
    reveal tokens during generation.
    """

    def __init__(
        self,
        num_steps: int = 25,
        temperature: float = 2.0,
        mask_schedule: str = "cosine",
    ):
        self.num_steps = num_steps
        self.temperature = temperature
        self.mask_schedule = mask_schedule

    def get_mask_prob(self, step: int) -> float:
        """Get mask probability for given step."""
        if self.mask_schedule == "cosine":
            t = step / self.num_steps
            return 1.0 - (1.0 + math.cos(math.pi * t)) / 2.0
        elif self.mask_schedule == "linear":
            return 1.0 - (step + 1) / self.num_steps
        else:
            raise ValueError(f"Unknown mask schedule: {self.mask_schedule}")

    def sample_frame(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a full frame of tokens from per-token logits in one shot.

        Genie samples each frame with a temperature (2.0 in the paper) using
        random sampling. This helper draws one categorical sample per spatial
        position.

        Args:
            logits: (B, N, vocab_size) next-frame token logits.

        Returns:
            tokens: (B, N) sampled token indices.
        """
        B, N, V = logits.shape
        probs = F.softmax(logits / self.temperature, dim=-1)
        tokens = torch.multinomial(probs.reshape(B * N, V), 1)
        return tokens.reshape(B, N)

    def sample(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MaskGIT refinement step over a single frame's tokens.

        Args:
            logits: (B, N, vocab_size) per-position logits for the frame.
            tokens: (B, N) tokens committed so far (values at masked
                positions are placeholders and ignored).
            mask: (B, N) - 1 for positions still to predict, 0 for committed.
            step: Current refinement step in ``[0, num_steps)``.

        Returns:
            new_tokens: (B, N) with newly revealed positions filled in.
            new_mask: (B, N) with newly revealed positions set to 0.
        """
        mask_bool = mask.bool()

        probs = F.softmax(logits / self.temperature, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
        sampled = sampled.reshape(mask.shape)

        # Confidence of each freshly sampled token; only masked positions are
        # eligible to be revealed this step.
        confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        confidence = confidence.masked_fill(~mask_bool, -1.0)

        # Fraction of positions that should remain masked after this step
        # (cosine schedule: reveal more as ``step`` grows).
        keep_masked_frac = self.get_mask_prob(step)
        N = mask.shape[-1]
        num_keep_masked = int(math.floor(keep_masked_frac * N))

        # Reveal the highest-confidence masked positions this step.
        reveal = mask_bool.clone()
        if num_keep_masked < N:
            threshold = torch.topk(confidence, k=N - num_keep_masked, dim=-1).values[
                ..., -1:
            ]
            reveal = mask_bool & (confidence >= threshold)

        new_tokens = torch.where(reveal, sampled, tokens)
        new_mask = mask_bool & ~reveal

        return new_tokens, new_mask.to(mask.dtype)


class DynamicsModel(ExportableAgentMixin, nn.Module):
    """Dynamics Model for action-controllable video generation.

    A decoder-only transformer that predicts future frame tokens given
    past frame tokens and latent actions. Uses MaskGIT for training
    and sampling.

    Based on Genie paper - uses cross-entropy loss with random masking
    during training, and MaskGIT iterative refinement at inference.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        vocab_size: int = 1024,
        embedding_dim: int = 32,
        action_vocab_size: int = 8,
        dim: int = 5120,
        depth: int = 48,
        num_heads: int = 40,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        gradient_checkpointing: bool = True,
        mask_prob_min: float = 0.5,
        mask_prob_max: float = 1.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.action_vocab_size = action_vocab_size
        self.dim = dim
        self.patch_size = patch_size
        # Paper 2.1: "we randomly mask the input tokens z_2:T-1 according to a
        # Bernoulli distribution masking rate sampled uniformly between 0.5 and
        # 1". The range lives here so the model samples it itself rather than
        # relying on every caller to remember.
        self.mask_prob_min = mask_prob_min
        self.mask_prob_max = mask_prob_max

        num_patches = (image_size // patch_size) ** 2

        # One extra row for the MaskGIT [MASK] token. Zeroing an embedding
        # instead would make "masked" indistinguishable from "a token whose
        # embedding happens to be near zero", and gives the model no learnable
        # representation of maskedness to condition on.
        self.mask_token_id = vocab_size
        self.video_embedding = nn.Embedding(vocab_size + 1, dim)
        self.action_embedding = nn.Embedding(action_vocab_size, dim)

        self.video_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, num_patches, dim)
        )
        nn.init.trunc_normal_(self.video_pos_embed, std=0.02)

        self.action_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.action_pos_embed, std=0.02)

        self.dynamics_transformer = STTransformer(
            num_frames=num_frames,
            num_patches_per_frame=num_patches,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(
        self,
        video_tokens: torch.Tensor,
        actions: torch.Tensor,
        mask_prob: float | None = None,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            video_tokens: (B, T, H*W) - token indices for frames 1 to T
            actions: (B, T) - latent action indices for frames 1 to T
            mask_prob: Masking rate. ``None`` (the default) draws it from
                ``U(mask_prob_min, mask_prob_max)`` per call, as the paper
                specifies. Pass ``0.0`` to disable masking, e.g. at inference.

        Returns:
            logits: (B, T, H*W, vocab_size)
        """
        B, T, N = video_tokens.shape

        # ===== Input token masking (paper 2.1) =====
        # Mask z_2:T-1 -- never the first frame (it is the conditioning prompt)
        # nor the last (its tokens are the prediction target).
        if self.training and T > 2:
            if mask_prob is None:
                # A fresh rate each step. A model trained at a single rate never
                # learns to generate from sparse context, which is precisely what
                # MaskGIT's iterative decoding asks of it at inference.
                mask_prob = float(
                    torch.empty(1)
                    .uniform_(self.mask_prob_min, self.mask_prob_max)
                    .item()
                )
            if mask_prob > 0.0:
                mask_2d = (
                    torch.rand(B, T - 2, N, device=video_tokens.device) < mask_prob
                )
                full_mask = torch.zeros(
                    B, T, N, dtype=torch.bool, device=video_tokens.device
                )
                full_mask[:, 1:-1, :] = mask_2d
                video_tokens = torch.where(
                    full_mask,
                    torch.full_like(video_tokens, self.mask_token_id),
                    video_tokens,
                )
        # ===== End masking =====

        video_emb = self.video_embedding(video_tokens)

        video_emb = video_emb.reshape(B, T, N, self.dim)

        video_emb = video_emb + self.video_pos_embed[:, :T, :, :]

        action_emb = self.action_embedding(actions)

        action_emb = action_emb + self.action_pos_embed[:, :T, :]

        action_emb_expanded = action_emb.unsqueeze(2).expand(-1, -1, N, -1)

        x = video_emb + action_emb_expanded

        x = x.reshape(B, T * N, -1)

        x = self.dynamics_transformer(x)

        x = x.reshape(B, T, N, self.dim)

        logits = self.output_proj(x)

        return logits

    def init_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> STKVCache:
        """Allocate a temporal KV cache for :meth:`forward_cached`."""
        return self.dynamics_transformer.init_cache(
            batch_size, self.num_frames, device=device, dtype=dtype
        )

    def forward_cached(
        self,
        video_tokens: torch.Tensor,
        actions: torch.Tensor,
        cache: STKVCache,
        commit: bool = True,
    ) -> torch.Tensor:
        """Inference forward over only the frames after those in ``cache``.

        Equivalent to the matching frames of ``forward(all_tokens, all_actions,
        mask_prob=0.0)`` in eval mode, without recomputing the cached prefix.

        Args:
            video_tokens: (B, T_new, N) tokens of the new frames. ``T_new`` may
                be larger than 1 only while the cache is empty (the prompt).
            actions: (B, T_new) latent actions for those frames.
            cache: From :meth:`init_cache`; its length is the index of the
                first new frame.
            commit: Advance the cache past these frames. False evaluates a
                candidate frame without keeping it.

        Returns:
            logits: (B, T_new, N, vocab_size)
        """
        B, T, N = video_tokens.shape
        start = cache.length
        if start + T > self.num_frames:
            raise ValueError(
                f"Frame {start + T} exceeds the positional capacity {self.num_frames}."
            )
        x = (
            self.video_embedding(video_tokens)
            + self.video_pos_embed[:, start : start + T]
        )
        action_emb = (
            self.action_embedding(actions) + self.action_pos_embed[:, start : start + T]
        )
        x = x + action_emb.unsqueeze(2)
        x = self.dynamics_transformer(
            x.reshape(B, T * N, self.dim), cache=cache, commit=commit
        )
        return self.output_proj(x.reshape(B, T, N, self.dim))

    def sample(
        self,
        prompt_tokens: torch.Tensor,
        prompt_actions: torch.Tensor,
        num_frames: int,
        sampler: Optional[MaskGITSampler] = None,
    ) -> torch.Tensor:
        """Sample future frames using MaskGIT.

        Args:
            prompt_tokens: (B, T_prompt, N) - starting frame tokens
            prompt_actions: (B, T_prompt) - actions for prompt frames
            num_frames: Total number of frames to generate
            sampler: MaskGIT sampler instance

        Returns:
            generated_tokens: (B, num_frames, N)
        """
        if sampler is None:
            sampler = MaskGITSampler()

        B, T_prompt, N = prompt_tokens.shape

        all_tokens = [prompt_tokens]
        all_actions = [prompt_actions]

        T_remaining = num_frames - T_prompt

        current_tokens = prompt_tokens
        current_actions = prompt_actions

        for step in range(T_remaining):
            logits = self.forward(current_tokens, current_actions)

            next_token_logits = logits[:, -1, :, :]

            probs = F.softmax(next_token_logits / sampler.temperature, dim=-1)

            next_tokens = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
            next_tokens = next_tokens.reshape(B, N)

            next_action = torch.randint(
                0, self.action_vocab_size, (B,), device=prompt_tokens.device
            )

            current_tokens = torch.cat(
                [current_tokens, next_tokens.unsqueeze(1)], dim=1
            )
            current_actions = torch.cat(
                [current_actions, next_action.unsqueeze(1)], dim=1
            )

            all_tokens.append(next_tokens.unsqueeze(1))
            all_actions.append(next_action.unsqueeze(1))

        generated_tokens = torch.cat(all_tokens, dim=1)

        return generated_tokens

    def autoregressive_sample(
        self,
        prompt_tokens: torch.Tensor,
        actions: torch.Tensor,
        num_frames: int,
        temperature: float = 1.0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Simple autoregressive sampling (frame by frame).

        Args:
            prompt_tokens: (B, T_prompt, N) - starting frame tokens
            actions: (B, num_frames - 1) - latent actions for each transition.
                The action driving frame ``t -> t+1`` is ``actions[:, t-1]``.
                If fewer actions are supplied than transitions, the remainder
                are sampled at random.
            num_frames: Total number of frames to generate
            temperature: Sampling temperature
            use_cache: Reuse the temporal keys/values of already generated
                frames (:class:`~simulusion.blocks.st_transformer.STKVCache`)
                instead of re-running the transformer over the whole prefix for
                every frame. The logits match the uncached path up to float
                rounding. One difference: when actions must be padded at
                random, the uncached path redraws the padding for *every*
                earlier frame at every step, while the cached path draws each
                frame's padding once.

        Returns:
            generated_tokens: (B, num_frames, N)
        """
        if use_cache:
            return self._autoregressive_sample_cached(
                prompt_tokens, actions, num_frames, temperature
            )
        B, T_prompt, N = prompt_tokens.shape

        current_tokens = prompt_tokens
        num_supplied = actions.shape[1]

        while current_tokens.shape[1] < num_frames:
            t = current_tokens.shape[1]  # frames generated so far

            if num_supplied >= t:
                current_actions = actions[:, :t]
            else:
                # Not enough actions supplied for this transition; pad with random.
                pad = torch.randint(
                    0,
                    self.action_vocab_size,
                    (B, t - num_supplied),
                    device=prompt_tokens.device,
                )
                current_actions = torch.cat([actions[:, :num_supplied], pad], dim=1)

            logits = self.forward(current_tokens, current_actions)

            next_frame_logits = logits[:, -1, :, :]

            probs = F.softmax(next_frame_logits / temperature, dim=-1)

            next_tokens = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
            next_tokens = next_tokens.reshape(B, N)

            current_tokens = torch.cat(
                [current_tokens, next_tokens.unsqueeze(1)], dim=1
            )

        return current_tokens

    def _autoregressive_sample_cached(
        self,
        prompt_tokens: torch.Tensor,
        actions: torch.Tensor,
        num_frames: int,
        temperature: float,
    ) -> torch.Tensor:
        B, T_prompt, N = prompt_tokens.shape
        device = prompt_tokens.device
        needed = max(num_frames - 1, T_prompt)
        if actions.shape[1] < needed:
            pad = torch.randint(
                0,
                self.action_vocab_size,
                (B, needed - actions.shape[1]),
                device=device,
            )
            actions = torch.cat([actions.to(device), pad], dim=1)

        cache = self.init_cache(B, device=device)
        frames = [prompt_tokens]
        logits = self.forward_cached(prompt_tokens, actions[:, :T_prompt], cache)
        t = T_prompt
        while t < num_frames:
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            next_tokens = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
            next_tokens = next_tokens.reshape(B, 1, N)
            frames.append(next_tokens)
            t += 1
            if t < num_frames:
                logits = self.forward_cached(next_tokens, actions[:, t - 1 : t], cache)
        return torch.cat(frames, dim=1)


def create_dynamics_model(
    num_frames: int = 16,
    image_size: int = 64,
    vocab_size: int = 1024,
    embedding_dim: int = 32,
    action_vocab_size: int = 8,
    dim: int = 5120,
    depth: int = 48,
    num_heads: int = 40,
    patch_size: int = 4,
) -> DynamicsModel:
    """Factory function to create a Dynamics Model."""
    return DynamicsModel(
        num_frames=num_frames,
        image_size=image_size,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        action_vocab_size=action_vocab_size,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
    )
