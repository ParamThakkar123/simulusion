import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple
from simulusion.export import ExportableAgentMixin


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Imagination generates one position at a time. Without a cache, producing the
    K tokens of a frame means re-running the whole Transformer over the entire
    prefix K times, which is O(K * L^2) per imagined step. Caching the keys and
    values of every position already processed reduces that to O(K * L).

    Storage is a pre-allocated (B, num_heads, max_len, head_dim) buffer per
    layer, filled left to right; ``length`` marks the valid prefix.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch_size, num_heads, max_len, head_dim)
        self.keys: List[torch.Tensor] = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.values: List[torch.Tensor] = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.max_len = max_len
        self.length = 0

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append this step's keys/values for ``layer`` and return the full prefix.

        The write offset is ``self.length``, which the caller advances once per
        forward pass (after all layers have been visited), so every layer writes
        to the same slice.
        """
        new_len = self.length + k.shape[2]
        if new_len > self.max_len:
            raise ValueError(
                f"KV cache overflow: {new_len} > {self.max_len}. Rebuild the "
                "cache from a trimmed history before generating further -- see "
                "IRISAgent.imagine_rollout."
            )
        self.keys[layer][:, :, self.length : new_len] = k
        self.values[layer][:, :, self.length : new_len] = v
        return (
            self.keys[layer][:, :, :new_len],
            self.values[layer][:, :, :new_len],
        )

    def advance(self, steps: int) -> None:
        """Commit ``steps`` newly written positions."""
        self.length += steps

    # Note: there is deliberately no in-place `trim` method. Positional
    # embeddings here are absolute, so sliding cached entries down would leave
    # them encoding positions they were not computed at. Callers that outgrow
    # the context must re-prime a fresh cache from a trimmed history, which is
    # what IRISAgent.imagine_rollout does.


class CausalSelfAttention(nn.Module):
    """GPT-2 style multi-head causal self-attention with optional KV caching."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, E)
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, E = x.shape

        q, k, v = self.qkv(x).split(E, dim=2)
        # (B, T, E) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if cache is None:
            attn_mask = None
            is_causal = T > 1
        else:
            past_len = cache.length
            k, v = cache.append(layer_idx, k, v)
            if T == 1:
                # A single new query attends to the whole cached prefix.
                attn_mask = None
                is_causal = False
            else:
                # Queries are the last T positions; they see all of the past and
                # each other causally. `is_causal` assumes square q/k, so build
                # the rectangular mask explicitly.
                total = past_len + T
                allowed = torch.ones(T, total, dtype=torch.bool, device=x.device)
                allowed[:, past_len:] = torch.tril(
                    torch.ones(T, T, dtype=torch.bool, device=x.device)
                )
                attn_mask = allowed
                is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.resid_dropout(self.proj(out))


class GPTBlock(nn.Module):
    """Pre-norm GPT-2 block: LN -> attention -> residual, LN -> MLP -> residual."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class IRISTransformer(ExportableAgentMixin, nn.Module):
    """GPT-like autoregressive Transformer for world modeling.

    Models the dynamics of the environment by predicting, autoregressively over
    an interleaved sequence of frame tokens and actions:

    - Next frame tokens (transition model), one token at a time
    - Rewards
    - Episode termination

    The sequence layout for ``S`` frames and ``S - 1`` actions is::

        z_0^1 ... z_0^K, a_0, z_1^1 ... z_1^K, a_1, ..., z_{S-2}^1 ... z_{S-2}^K,
        a_{S-2}, z_{S-1}^1 ... z_{S-1}^K

    A causal (lower-triangular) attention mask is always applied, so every
    position only attends to itself and preceding positions. The tokens of frame
    ``t + 1`` are predicted starting from the *action* position ``a_t`` (which
    sees the whole of frame ``t`` and the action), then autoregressively from
    each previously predicted token of frame ``t + 1``. This matches the paper's

        z_{t+1}^k ~ p(. | z_{<=t}, a_{<=t}, z_{t+1}^{<k})
    """

    def __init__(
        self,
        vocab_size: int = 512,
        tokens_per_frame: int = 16,
        action_size: int = 18,  # Number of Atari actions
        embed_dim: int = 256,
        num_layers: int = 10,
        num_heads: int = 4,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        reward_classes: int = 3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame
        self.action_size = action_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.action_embedding = nn.Embedding(action_size, embed_dim)

        # Positional embeddings
        # Max sequence length: (tokens_per_frame + 1) * timesteps
        # 16 tokens + 1 action per timestep = 17 tokens/timestep
        max_tokens = tokens_per_frame + 1  # tokens + action
        max_seq_len = max_tokens * 50  # Support up to 50 timesteps
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Embedding dropout (paper Table 4).
        self.embed_dropout = nn.Dropout(dropout)

        # GPT-2-style pre-norm blocks (paper A.2). These are used instead of
        # nn.TransformerEncoder because incremental decoding during imagination
        # needs access to per-layer key/value caches, which the stock module
        # does not expose.
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Output heads
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Token prediction head (for next frame tokens)
        self.token_head = nn.Linear(embed_dim, vocab_size)

        # Reward prediction head. Paper 2.2 allows "a mean-squared error loss or
        # a cross-entropy loss for the reward predictor, depending on the reward
        # function". With the Atari convention of sign-transformed rewards the
        # target is categorical over {-1, 0, +1}, so the head emits one logit per
        # class; reward_classes=1 restores a scalar regression head.
        self.reward_classes = reward_classes
        self.reward_head = nn.Linear(embed_dim, reward_classes)

        # Termination prediction head
        self.termination_head = nn.Linear(
            embed_dim, 2
        )  # Binary: 0=continue, 1=terminal

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with GPT-2 scaling (paper A.2, minGPT)."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.action_embedding.weight, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale residual-path projections by 1/sqrt(2 * num_layers) so the
        # variance of the residual stream stays bounded with depth (GPT-2).
        scale = 1.0 / math.sqrt(2 * self.num_layers)
        for block in self.gpt_blocks():
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            mlp_out = block.mlp[2]
            assert isinstance(mlp_out, nn.Linear)
            nn.init.normal_(mlp_out.weight, mean=0.0, std=0.02 * scale)

    def expected_reward(self, action_hidden: torch.Tensor) -> torch.Tensor:
        """Scalar reward prediction from an action-position hidden state.

        With a categorical head this is the expectation under the predicted
        distribution over {-1, 0, +1} rather than an argmax, so the imagined
        return reflects the model's uncertainty instead of committing to the
        modal class.

        Args:
            action_hidden: (..., embed_dim) hidden states at action positions.

        Returns:
            (...) scalar reward predictions.
        """
        logits = self.reward_head(action_hidden)
        if self.reward_classes == 1:
            return logits.squeeze(-1)
        support = torch.arange(
            self.reward_classes, device=logits.device, dtype=logits.dtype
        ) - (self.reward_classes // 2)
        return (torch.softmax(logits, dim=-1) * support).sum(dim=-1)

    def gpt_blocks(self) -> List[GPTBlock]:
        """The transformer blocks, typed (``nn.ModuleList`` erases the element type)."""
        return [block for block in self.blocks if isinstance(block, GPTBlock)]

    def _run_transformer(
        self,
        sequence: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """Run the GPT blocks, checkpointing them during training when enabled.

        Causality is enforced inside :class:`CausalSelfAttention`; no explicit
        mask argument is needed here. When ``cache`` is given, each block appends
        its keys/values and the cache is advanced once at the end so every layer
        writes to the same positions.
        """
        hidden: torch.Tensor = sequence
        use_checkpoint = self.gradient_checkpointing and self.training and cache is None

        for layer_idx, block in enumerate(self.gpt_blocks()):
            if use_checkpoint:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden, cache=cache, layer_idx=layer_idx)

        if cache is not None:
            cache.advance(sequence.shape[1])

        return hidden

    def _embed_interleaved(
        self,
        frame_tokens: torch.Tensor,  # (B, Tc, K)
        actions: torch.Tensor,  # (B, Tc)
        extra_tokens: Optional[torch.Tensor] = None,  # (B, m)
    ) -> torch.Tensor:
        """Build a positionally-embedded interleaved token/action sequence.

        Produces embeddings for ``[z_0, a_0, ..., z_{Tc-1}, a_{Tc-1}]`` and,
        optionally, ``m`` trailing (partial next-frame) tokens with no following
        action. Used both for the training forward pass and for autoregressive
        generation.
        """
        B, Tc, K = frame_tokens.shape
        E = self.embed_dim

        tok_emb = self.token_embedding(frame_tokens.reshape(B, Tc * K)).reshape(
            B, Tc, K, E
        )
        act_emb = self.action_embedding(actions)  # (B, Tc, E)

        # Interleave each frame block with its action: (B, Tc, K + 1, E)
        blocks = torch.cat([tok_emb, act_emb.unsqueeze(2)], dim=2)
        seq = blocks.reshape(B, Tc * (K + 1), E)

        if extra_tokens is not None and extra_tokens.shape[1] > 0:
            extra_emb = self.token_embedding(extra_tokens)  # (B, m, E)
            seq = torch.cat([seq, extra_emb], dim=1)

        L = seq.shape[1]
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds positional-embedding capacity "
                f"{self.max_seq_len}."
            )
        return seq + self.pos_embedding[:, :L, :]

    def forward(
        self,
        tokens: torch.Tensor,  # (B, S, K) - S frame token grids
        actions: torch.Tensor,  # (B, S-1) - actions between frames
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass through the Transformer world model.

        Args:
            tokens: Frame tokens (B, S, K) for S consecutive frames.
            actions: Actions (B, S-1); ``actions[:, t]`` is taken after frame t.

        Returns:
            token_logits: Predictions of frames 1..S-1 (B, S-1, K, vocab_size).
            rewards: Predicted rewards r_0..r_{S-2} (B, S-1).
            terminations: Predicted terminations d_0..d_{S-2} (B, S-1, 2).
        """
        B, S, K = tokens.shape
        if actions.shape[1] != S - 1:
            raise ValueError(
                f"Expected actions of length S-1={S - 1}, got {actions.shape[1]}."
            )

        # Interleave frames 0..S-1 with actions 0..S-2. The sequence ends on the
        # last frame block (z_{S-1}) with no trailing action.
        tok_emb = self.token_embedding(tokens.reshape(B, S * K)).reshape(
            B, S, K, self.embed_dim
        )
        act_emb = self.action_embedding(actions)  # (B, S-1, E)

        head_blocks = torch.cat(
            [tok_emb[:, : S - 1], act_emb.unsqueeze(2)], dim=2
        )  # (B, S-1, K+1, E)
        head_blocks = head_blocks.reshape(B, (S - 1) * (K + 1), self.embed_dim)
        last_frame = tok_emb[:, S - 1]  # (B, K, E)
        sequence = torch.cat([head_blocks, last_frame], dim=1)  # (B, L, E)

        L = sequence.shape[1]
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds positional-embedding capacity "
                f"{self.max_seq_len}."
            )
        sequence = self.embed_dropout(sequence + self.pos_embedding[:, :L, :])

        hidden = self._run_transformer(sequence)
        hidden = self.layer_norm(hidden)

        # For target frame t+1 (t in 0..S-2), the K hidden states that predict its
        # tokens are the contiguous slice starting at action position a_t:
        #   [a_t, z_{t+1}^1, ..., z_{t+1}^{K-1}]  (stride K+1 between t's).
        starts = torch.arange(S - 1, device=tokens.device) * (K + 1) + K  # (S-1,)
        offsets = torch.arange(K, device=tokens.device)  # (K,)
        idx = (starts[:, None] + offsets[None, :]).reshape(-1)  # ((S-1)*K,)

        sel = hidden[:, idx, :].reshape(B, S - 1, K, self.embed_dim)  # (B, S-1, K, E)
        token_logits = self.token_head(sel)  # (B, S-1, K, vocab)

        # Reward / termination are read from the action position a_t, which is the
        # first element of each per-frame slice.
        action_hidden = sel[:, :, 0, :]  # (B, S-1, E)
        rewards = self.reward_head(action_hidden)  # (B, S-1, reward_classes)
        if self.reward_classes == 1:
            rewards = rewards.squeeze(-1)  # (B, S-1) for scalar regression
        terminations = self.termination_head(action_hidden)  # (B, S-1, 2)

        return token_logits, rewards, terminations

    def _normalize_context(
        self, tokens: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coerce a single-step (frame, action) context to (B, 1, K) / (B, 1)."""
        if tokens.dim() == 3 and tokens.shape[1] != self.tokens_per_frame:
            # (B, H, W) grid of tokens -> (B, K)
            B_grid, H, W = tokens.shape
            tokens = tokens.reshape(B_grid, H * W)
        if tokens.dim() == 2:  # (B, K) -> (B, 1, K)
            tokens = tokens.unsqueeze(1)
        if actions.dim() == 1:  # (B,) -> (B, 1)
            actions = actions.unsqueeze(1)
        return tokens, actions

    def init_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> KVCache:
        """Allocate an empty KV cache sized to this model's context capacity."""
        return KVCache(
            num_layers=self.num_layers,
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            max_len=self.max_seq_len,
            device=device,
            dtype=dtype or self.pos_embedding.dtype,
        )

    def _forward_positions(
        self,
        embeddings: torch.Tensor,  # (B, T, E), no positional encoding yet
        start_pos: int,
        cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Add absolute positional embeddings at ``start_pos`` and run the blocks."""
        T = embeddings.shape[1]
        end = start_pos + T
        if end > self.max_seq_len:
            raise ValueError(
                f"Position {end} exceeds positional-embedding capacity "
                f"{self.max_seq_len}."
            )
        seq = self.embed_dropout(embeddings + self.pos_embedding[:, start_pos:end, :])
        return self.layer_norm(self._run_transformer(seq, cache=cache))

    @torch.no_grad()
    def prime_cache(
        self,
        tokens: torch.Tensor,  # (B, Tc, K)
        actions: Optional[torch.Tensor],  # (B, Tc) or (B, Tc-1)
        cache: KVCache,
        start_pos: int = 0,
    ) -> int:
        """Fill a cache with an interleaved (frame, action) history in one pass.

        This is the "conditioning" phase of imagination: the real frames (and the
        actions taken between them) are pushed through the model so that
        subsequent single-position steps attend to the full history, exactly as
        the teacher-forced training sequence does.

        Args:
            tokens: Frame token grids (B, Tc, K).
            actions: Actions following each frame. Pass ``Tc`` actions to end the
                primed sequence on an action (ready to generate the next frame),
                or ``Tc - 1`` / ``None`` to end on the last frame's tokens.
            cache: Cache to fill; must be empty or positioned at ``start_pos``.
            start_pos: Absolute position of the first embedded token.

        Returns:
            The absolute position just past the primed sequence.
        """
        B, Tc, K = tokens.shape
        tok_emb = self.token_embedding(tokens.reshape(B, Tc * K)).reshape(
            B, Tc, K, self.embed_dim
        )

        num_actions = 0 if actions is None else actions.shape[1]
        if num_actions not in (Tc, Tc - 1):
            raise ValueError(
                f"Expected {Tc} or {Tc - 1} actions for {Tc} frames, got {num_actions}."
            )

        if num_actions > 0:
            assert actions is not None
            act_emb = self.action_embedding(actions)  # (B, num_actions, E)
            paired = torch.cat(
                [tok_emb[:, :num_actions], act_emb.unsqueeze(2)], dim=2
            )  # (B, num_actions, K+1, E)
            sequence = paired.reshape(B, num_actions * (K + 1), self.embed_dim)
            if num_actions < Tc:
                # Trailing frame with no action after it.
                sequence = torch.cat([sequence, tok_emb[:, Tc - 1]], dim=1)
        else:
            sequence = tok_emb.reshape(B, Tc * K, self.embed_dim)

        self._forward_positions(sequence, start_pos, cache)
        return start_pos + sequence.shape[1]

    @torch.no_grad()
    def generate_frame_cached(
        self,
        action: torch.Tensor,  # (B,)
        cache: KVCache,
        start_pos: int,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Generate one frame's K tokens incrementally, one position at a time.

        The cache must already hold the history up to and including the current
        frame's tokens. This appends the action, reads the reward/termination
        hidden state from that position, then autoregressively appends the K - 1
        predicted tokens -- K single-position forwards in total, rather than K
        full-sequence passes.

        Returns:
            step_logits: Per-token logits (B, K, vocab).
            generated: Token indices (B, K).
            action_hidden: Hidden state at the action position (B, E).
            next_pos: Absolute position after the generated frame's tokens.
        """
        K = self.tokens_per_frame
        pos = start_pos

        # Step 1: the action position. Its hidden state both predicts the first
        # token of the next frame and feeds the reward / termination heads.
        act_emb = self.action_embedding(action).unsqueeze(1)  # (B, 1, E)
        hidden = self._forward_positions(act_emb, pos, cache)
        pos += 1
        action_hidden = hidden[:, -1, :]

        logits_list: list[torch.Tensor] = []
        generated_list: list[torch.Tensor] = []

        for k in range(K):
            logits = self.token_head(hidden[:, -1, :])  # (B, vocab)
            logits_list.append(logits)

            if sample:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            else:
                next_token = logits.argmax(dim=-1)  # (B,)
            generated_list.append(next_token)

            if k < K - 1:
                # Feed the token back in so the next one is conditioned on it
                # (z_{t+1}^{k+1} ~ p(. | ..., z_{t+1}^{<=k})). The final token
                # needs no feed-forward pass -- nothing is predicted from it here.
                tok_emb = self.token_embedding(next_token).unsqueeze(1)  # (B, 1, E)
                hidden = self._forward_positions(tok_emb, pos, cache)
                pos += 1

        # Push the last generated token so the cache ends on a complete frame,
        # ready for the next action.
        last_emb = self.token_embedding(generated_list[-1]).unsqueeze(1)
        self._forward_positions(last_emb, pos, cache)
        pos += 1

        step_logits = torch.stack(logits_list, dim=1)  # (B, K, vocab)
        generated = torch.stack(generated_list, dim=1)  # (B, K)
        return step_logits, generated, action_hidden, pos

    @torch.no_grad()
    def _generate_frame(
        self,
        context_tokens: torch.Tensor,  # (B, Tc, K)
        context_actions: torch.Tensor,  # (B, Tc)
        sample: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the next frame's K tokens from a (frame, action) history.

        Stateless convenience wrapper: it builds a cache, primes it with the
        given context and generates one frame. Rolling out many steps should use
        :meth:`prime_cache` + :meth:`generate_frame_cached` directly so the cache
        is reused across steps.

        Returns:
            step_logits: Per-token logits used at generation (B, K, vocab).
            generated: Sampled/greedy token indices (B, K).
            action_hidden: Hidden state at the final action position (B, E), used
                for reward / termination prediction.
        """
        B = context_tokens.shape[0]
        cache = self.init_cache(B, context_tokens.device, self.pos_embedding.dtype)

        # Prime with the frames and all but the last action; the last action is
        # consumed by the generation step itself.
        pos = self.prime_cache(
            context_tokens, context_actions[:, :-1], cache, start_pos=0
        )
        step_logits, generated, action_hidden, _ = self.generate_frame_cached(
            context_actions[:, -1],
            cache,
            start_pos=pos,
            sample=sample,
            temperature=temperature,
        )
        return step_logits, generated, action_hidden

    def predict_next_tokens(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedily predict the next frame tokens autoregressively.

        Args:
            tokens: Current frame tokens (B, K) or (B, H, W).
            actions: Actions taken (B,).

        Returns:
            token_logits: Next frame token logits (B, K, vocab_size). Their argmax
                equals the greedily generated tokens.

            action_hidden: Hidden states for reward prediction (B, embed_dim).
        """
        tokens, actions = self._normalize_context(tokens, actions)
        step_logits, _, action_hidden = self._generate_frame(
            tokens, actions, sample=False
        )
        return step_logits, action_hidden

    def imagine_step(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance imagination one step: next frame tokens + the action hidden state.

        Unlike :meth:`predict_next_tokens` this can sample rather than take the
        argmax, and it returns the generated tokens directly so a rollout does
        not have to re-derive them from logits.

        Args:
            tokens: Current frame tokens (B, K) or (B, H, W).
            actions: Actions taken (B,).
            sample: If True, sample tokens from the predicted distribution.
            temperature: Sampling temperature (ignored when ``sample`` is False).

        Returns:
            next_tokens: Generated token indices (B, K).
            action_hidden: Hidden state at the action position (B, embed_dim),
                the input expected by ``reward_head`` / ``termination_head``.
        """
        tokens, actions = self._normalize_context(tokens, actions)
        _, generated, action_hidden = self._generate_frame(
            tokens, actions, sample=sample, temperature=temperature
        )
        return generated, action_hidden

    def sample_next_tokens(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample next tokens autoregressively from the distribution.

        Args:
            tokens: Current frame tokens (B, K) or (B, H, W).
            actions: Actions taken (B,).
            temperature: Sampling temperature (higher = more random).

        Returns:
            sampled_tokens: Sampled token indices (B, K).
            log_probs: Log probabilities of sampled tokens (B, K).
        """
        tokens, actions = self._normalize_context(tokens, actions)
        step_logits, sampled_tokens, _ = self._generate_frame(
            tokens, actions, sample=True, temperature=temperature
        )
        log_probs = F.log_softmax(step_logits / temperature, dim=-1)
        log_probs = torch.gather(log_probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(
            -1
        )
        return sampled_tokens, log_probs


class IRISWorldModel(ExportableAgentMixin, nn.Module):
    """Complete IRIS World Model combining autoencoder and transformer.

    This is the core component that learns environment dynamics entirely
    in the "imaginary" latent space.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        transformer: IRISTransformer,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer

    def decode_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices (B, K) or (B, H, W) to images.

        The lookup goes through the *encoder's* quantizer codebook, which is the
        table the reconstruction and commitment losses actually train.
        ``IRISDecoder.index_to_embedding`` is a separate, never-optimised
        embedding table, so decoding through it returns noise.
        """
        if indices.dim() == 2:
            side = int(round(indices.shape[1] ** 0.5))
            indices = indices.reshape(indices.shape[0], side, side)
        quantizer = getattr(self.encoder, "quantizer")
        embeddings = quantizer.decode_indices(indices)
        return self.decoder(embeddings)

    def forward(
        self,
        observations: torch.Tensor,  # (B, T+1, C, H, W)
        actions: torch.Tensor,  # (B, T)
    ) -> Tuple[dict, dict]:
        """Full world model forward pass.

        Args:
            observations: Image sequence (B, T+1, C, H, W)
            actions: Actions (B, T)

        Returns:
            predictions: Dictionary with predicted tokens, rewards, terminations
            losses: Dictionary with loss components
        """
        B, T_plus_1, C, H, W = observations.shape
        T = T_plus_1 - 1

        # Encode each frame to tokens
        tokens_list = []
        for t in range(T_plus_1):
            obs_t = observations[:, t]  # (B, C, H, W)
            _, indices_t, _ = self.encoder(obs_t)
            tokens_list.append(indices_t.reshape(B, -1))

        # Stack tokens: (B, T+1, K)
        tokens = torch.stack(tokens_list, dim=1)

        # Get transformer predictions over the full frame sequence. Predictions
        # cover frames 1..T (B, T, K, vocab).
        token_logits, rewards_pred, terminations_pred = self.transformer(
            tokens,  # (B, T+1, K)
            actions,  # (B, T)
        )

        # Decode predictions to images (for visualization)
        decoded_frames_list: list[torch.Tensor] = []
        for t in range(T):
            next_tokens_pred = token_logits[:, t, :, :].argmax(dim=-1)  # Greedy
            # These are token *indices*; they have to be looked up in the
            # codebook before the decoder can consume them. Passing them
            # straight to the decoder raised
            # "Input type (__int64) and bias type (float) should be the same".
            decoded_frames_list.append(self.decode_tokens(next_tokens_pred))

        decoded_frames: Optional[torch.Tensor]
        decoded_frames = (
            torch.stack(decoded_frames_list, dim=1) if decoded_frames_list else None
        )

        # Get actual next tokens for loss computation
        next_tokens = tokens[:, 1:]  # (B, T, K)

        # Compute losses
        token_loss = F.cross_entropy(
            token_logits.reshape(-1, self.transformer.vocab_size),
            next_tokens.reshape(-1),
            reduction="mean",
        )

        # Reward and termination losses would be computed with actual labels
        # (These are computed in the training loop)

        predictions = {
            "token_logits": token_logits,
            "rewards": rewards_pred,
            "terminations": terminations_pred,
            "decoded_frames": decoded_frames,
        }

        losses = {
            "token_loss": token_loss,
        }

        return predictions, losses

    def imagine(
        self,
        initial_tokens: torch.Tensor,  # (B, K)
        policy: nn.Module,
        horizon: int = 20,
        temperature: float = 1.0,
    ) -> dict:
        """Generate imagined trajectories.

        Args:
            initial_tokens: Initial frame tokens (B, K)
            policy: Policy network to sample actions
            horizon: Number of steps to imagine
            temperature: Sampling temperature for token prediction

        Returns:
            imagined: Dictionary with imagined trajectories
        """
        B = initial_tokens.shape[0]
        K = self.transformer.tokens_per_frame

        # Lists to store trajectory
        tokens_history = [initial_tokens]
        actions_history = []
        rewards_history = []
        terminations_history = []

        current_tokens = initial_tokens

        # Condition on the full imagined history (paper 2.3) via a KV cache,
        # rather than restarting from a single frame at every step.
        cache = self.transformer.init_cache(B, initial_tokens.device)
        pos = self.transformer.prime_cache(
            current_tokens.reshape(B, 1, K), None, cache, start_pos=0
        )

        for _step in range(horizon):
            # Get action from policy (using decoded frame)
            with torch.no_grad():
                decoded_frame = self.decode_tokens(current_tokens)
                action = getattr(policy, "forward")(decoded_frame)
                actions_history.append(action)

            action_idx = action.squeeze(-1) if action.dim() > 1 else action

            with torch.no_grad():
                _, sampled_tokens, action_hidden, pos = (
                    self.transformer.generate_frame_cached(
                        action_idx,
                        cache,
                        start_pos=pos,
                        sample=True,
                        temperature=temperature,
                    )
                )
                # Keep the batch dimension: reducing to a scalar here would give
                # every trajectory in the batch the same reward.
                reward = self.transformer.expected_reward(action_hidden)  # (B,)
                termination_logits = self.transformer.termination_head(action_hidden)
                termination = torch.softmax(termination_logits, dim=-1)[:, 1]

            tokens_history.append(sampled_tokens)
            rewards_history.append(reward)
            terminations_history.append(termination)

            # Update current tokens
            current_tokens = sampled_tokens

            # Early stopping once every trajectory has predicted an episode end.
            if bool((termination > 0.5).all()):
                break

        return {
            "tokens": torch.stack(tokens_history, dim=1),  # (B, H+1, K)
            "actions": torch.stack(actions_history, dim=1) if actions_history else None,
            "rewards": torch.stack(rewards_history, dim=1) if rewards_history else None,
            "terminations": (
                torch.stack(terminations_history, dim=1)
                if terminations_history
                else None
            ),
        }
