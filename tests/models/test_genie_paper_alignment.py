"""Checks that the Genie implementation matches Bruce et al. (2024).

The properties pinned here are ones that fail *silently*: a non-causal
ST-transformer still trains, a latent action that ignores the future frame still
produces a loss curve, and a fixed masking rate still converges -- they just
produce a model that cannot do what the paper's model does.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from simulusion.blocks.st_transformer import (
    STSpatialAttention,
    STTemporalAttention,
    STTransformer,
    STTransformerBlock,
)
from simulusion.models.dynamics_model import DynamicsModel
from simulusion.models.latent_action_model import LatentActionModel


def _st(**overrides) -> STTransformer:
    kwargs = dict(num_frames=6, num_patches_per_frame=4, dim=32, depth=2, num_heads=4)
    kwargs.update(overrides)
    return STTransformer(**kwargs).eval()


def _lam(**overrides) -> LatentActionModel:
    kwargs = dict(
        num_frames=8,
        image_size=32,
        patch_size=16,
        encoder_dim=64,
        decoder_dim=64,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        vocab_size=8,
        embedding_dim=32,
    )
    kwargs.update(overrides)
    return LatentActionModel(**kwargs).eval()


def _dynamics(**overrides) -> DynamicsModel:
    kwargs = dict(
        num_frames=6,
        image_size=32,
        vocab_size=64,
        action_vocab_size=8,
        dim=32,
        depth=2,
        num_heads=4,
        patch_size=16,
        gradient_checkpointing=False,
    )
    kwargs.update(overrides)
    return DynamicsModel(**kwargs)


class TestSTTransformerStructure:
    """Paper 2: L blocks of spatial attention, causal temporal attention, one FFW."""

    def test_temporal_attention_is_causal(self):
        """Perturbing the last frame must leave every earlier frame untouched.

        This catches head/space/time axis mix-ups in the attention output
        reshape, which change no tensor shape and raise no error but leak
        information backwards through time.
        """
        torch.manual_seed(0)
        st = _st()
        B, T, N, C = 1, 6, 4, 32
        x = torch.randn(B, T * N, C)

        with torch.no_grad():
            base = st(x).reshape(B, T, N, C)
            perturbed_in = x.clone().reshape(B, T, N, C)
            perturbed_in[:, -1] = torch.randn(B, N, C)
            perturbed = st(perturbed_in.reshape(B, T * N, C)).reshape(B, T, N, C)

        delta = (base - perturbed).abs().amax(dim=(0, 2, 3))
        assert float(delta[:-1].max()) == 0.0, (
            f"future frame leaked into earlier outputs: {delta.tolist()}"
        )
        assert float(delta[-1]) > 0.0, "the perturbed frame itself did not change"

    def test_past_frames_do_influence_the_future(self):
        """Sanity check that the causal mask has not simply severed attention."""
        torch.manual_seed(0)
        st = _st()
        B, T, N, C = 1, 6, 4, 32
        x = torch.randn(B, T * N, C)

        with torch.no_grad():
            base = st(x).reshape(B, T, N, C)
            perturbed_in = x.clone().reshape(B, T, N, C)
            perturbed_in[:, 0] = torch.randn(B, N, C)
            perturbed = st(perturbed_in.reshape(B, T * N, C)).reshape(B, T, N, C)

        delta = (base - perturbed).abs().amax(dim=(0, 2, 3))
        assert float(delta.min()) > 0.0

    def test_spatial_attention_does_not_mix_timesteps(self):
        """Spatial attention attends over 1 x H x W within a single time step."""
        torch.manual_seed(0)
        attn = STSpatialAttention(dim=32, num_heads=4).eval()
        x = torch.randn(1, 5, 4, 32)

        with torch.no_grad():
            base = attn(x)
            other = x.clone()
            other[:, 2] = torch.randn(1, 4, 32)
            perturbed = attn(other)

        delta = (base - perturbed).abs().amax(dim=(0, 2, 3))
        moved = [t for t, v in enumerate(delta.tolist()) if v > 0]
        assert moved == [2], f"spatial attention leaked across time: {moved}"

    def test_single_ffw_after_both_attentions(self):
        """Paper 2: "only one FFW after both spatial and temporal components,
        omitting the post-spatial FFW"."""
        block = STTransformerBlock(dim=32, num_heads=4)
        mlps = [m for m in block.modules() if type(m).__name__ == "STMLP"]
        assert len(mlps) == 1, f"expected one FFW per block, found {len(mlps)}"

    def test_block_rejects_ambiguous_flat_input(self):
        """A flat (B, T*N, C) tensor cannot be split without knowing N."""
        block = STTransformerBlock(dim=32, num_heads=4)
        with pytest.raises(ValueError, match="expects"):
            block(torch.randn(1, 24, 32))

    def test_qk_normalisation_present(self):
        """Paper 3: QK norm, cited for stabilising training at scale."""
        for module in (STSpatialAttention(32, 4), STTemporalAttention(32, 4)):
            assert isinstance(module.q_norm, nn.LayerNorm)
            assert isinstance(module.k_norm, nn.LayerNorm)


class TestLatentActionModel:
    """Paper 2.1: the encoder sees x_1:t *and* x_t+1, and emits the action between."""

    def test_action_depends_on_the_future_frame(self):
        """The whole purpose of the LAM.

        "As the decoder only has access to the history and latent action, a_t
        should encode the most meaningful changes between the past and the
        future." An action computed without ever seeing x_t+1 cannot encode that
        change, and the VQ codebook collapses to an uninformative constant.
        """
        torch.manual_seed(0)
        lam = _lam()
        x_prev = torch.rand(2, 3, 4, 32, 32)

        with torch.no_grad():
            _, z_a, _ = lam._encode(x_prev, torch.rand(2, 3, 32, 32))
            _, z_b, _ = lam._encode(x_prev, torch.rand(2, 3, 32, 32))

        assert float((z_a - z_b).abs().max()) > 0.0, (
            "latent actions are independent of the future frame"
        )

    def test_final_action_is_the_one_that_moves(self):
        """Only the action into the changed frame should react to it."""
        torch.manual_seed(0)
        lam = _lam()
        x_prev = torch.rand(2, 3, 4, 32, 32)

        with torch.no_grad():
            _, z_a, _ = lam._encode(x_prev, torch.rand(2, 3, 32, 32))
            _, z_b, _ = lam._encode(x_prev, torch.rand(2, 3, 32, 32))

        per_step = (z_a - z_b).abs().amax(dim=2).amax(dim=0)
        assert float(per_step[-1]) > 0.0
        assert float(per_step[:-1].max()) == 0.0, (
            "actions over unchanged frames should not move"
        )

    def test_one_action_per_transition(self):
        lam = _lam()
        with torch.no_grad():
            out = lam(torch.rand(2, 3, 4, 32, 32), torch.rand(2, 3, 32, 32))
        assert out["latent_actions"].shape == (2, 4)

    def test_codebook_size_is_small_for_playability(self):
        """Paper 2.1 uses |A| = 8 to keep the action space human-playable."""
        lam = _lam()
        assert lam.vocab_size == 8

    def test_takes_pixels_not_tokens(self):
        """Table 2's ablation: pixel-input beats token-input on controllability."""
        lam = _lam()
        with torch.no_grad():
            out = lam(torch.rand(2, 3, 4, 32, 32), torch.rand(2, 3, 32, 32))
        assert out["reconstructed"].shape == (2, 3, 32, 32)


class TestDynamicsMasking:
    """Paper 2.1: MaskGIT training with a Bernoulli rate drawn from U(0.5, 1)."""

    def test_masking_rate_is_sampled_not_fixed(self):
        """A model trained at one rate never learns to decode from sparse context."""
        model = _dynamics().train()
        assert model.mask_prob_min == 0.5 and model.mask_prob_max == 1.0

        tokens = torch.randint(0, 64, (4, 6, 4))
        actions = torch.zeros(4, 6, dtype=torch.long)

        seen = set()
        for _ in range(40):
            torch.manual_seed(len(seen) + 1)
            before = tokens.clone()
            model(tokens, actions)  # mask_prob=None -> sampled internally
            assert torch.equal(tokens, before), "input tensor was mutated"
            seen.add(_count_masked(model, tokens, actions))
        assert len(seen) > 1, "masking appears deterministic across calls"

    def test_uses_a_dedicated_mask_token(self):
        """MaskGIT masks with a learned token, not by zeroing an embedding."""
        model = _dynamics()
        assert model.video_embedding.num_embeddings == model.vocab_size + 1
        assert model.mask_token_id == model.vocab_size

    def test_no_masking_in_eval(self):
        model = _dynamics().eval()
        tokens = torch.randint(0, 64, (2, 6, 4))
        actions = torch.zeros(2, 6, dtype=torch.long)
        with torch.no_grad():
            a = model(tokens, actions)
            b = model(tokens, actions)
        torch.testing.assert_close(a, b)

    def test_explicit_zero_disables_masking(self):
        model = _dynamics().train()
        tokens = torch.randint(0, 64, (2, 6, 4))
        actions = torch.zeros(2, 6, dtype=torch.long)
        torch.manual_seed(0)
        a = model(tokens, actions, mask_prob=0.0)
        torch.manual_seed(0)
        b = model(tokens, actions, mask_prob=0.0)
        torch.testing.assert_close(a, b)

    def test_actions_are_additive_embeddings(self):
        """Paper 2.1 found additive action embeddings beat concatenation."""
        model = _dynamics()
        assert model.action_embedding.embedding_dim == model.dim, (
            "action embeddings must share the token width to be added, not concatenated"
        )

    def test_output_covers_the_token_vocabulary(self):
        model = _dynamics().eval()
        with torch.no_grad():
            logits = model(
                torch.randint(0, 64, (2, 6, 4)), torch.zeros(2, 6, dtype=torch.long)
            )
        assert logits.shape[-1] == 64, "logits must not include the mask token"


def _count_masked(model: DynamicsModel, tokens: torch.Tensor, actions: torch.Tensor):
    """Number of positions replaced by the mask token in one training forward."""
    captured = {}
    original = model.video_embedding.forward

    def spy(ids):
        captured["n"] = int((ids == model.mask_token_id).sum().item())
        return original(ids)

    model.video_embedding.forward = spy  # type: ignore[method-assign]
    try:
        model(tokens, actions)
    finally:
        model.video_embedding.forward = original  # type: ignore[method-assign]
    return captured.get("n", 0)
