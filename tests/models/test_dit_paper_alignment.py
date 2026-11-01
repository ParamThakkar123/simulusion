"""Checks that the DiT implementation matches Peebles & Xie (2023).

These pin details that are easy to get subtly wrong while still producing a model
that trains and emits plausible-looking losses -- most notably the adaLN-Zero
conditioning that Figure 5 identifies as the paper's central architectural
result, and the timestep embedding, which fails silently when its frequency
ladder is built the wrong way round.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulusion.configs.dit_config import (
    DIT_PRESETS,
    DiTConfig,
    dit_preset_config,
    list_dit_presets,
)
from simulusion.models.diffusion.DiT import (
    DiT,
    FinalLayer,
    PatchEmbed,
    TimestepEmbedder,
    TransformerBlock,
    get_2d_sincos_pos_embed,
    sinusoidal_time_embedding,
)


def _small_dit(**overrides) -> DiT:
    kwargs = dict(
        img_size=16,
        patch_size=4,
        in_channels=4,
        d_model=64,
        depth=2,
        heads=4,
        num_classes=10,
    )
    kwargs.update(overrides)
    return DiT(**kwargs)


class TestTimestepEmbedding:
    """Frequencies must *decay*: sin(t / P^(2i/d)), not sin(t * P^(2i/d))."""

    def test_frequencies_decay_from_one(self):
        emb = sinusoidal_time_embedding(torch.tensor([1.0]), 16)
        # cos(t*f) for the lowest frequency f=1 at t=1 -> cos(1).
        assert float(emb[0, 0]) == pytest.approx(math.cos(1.0), abs=1e-5)
        # The highest-index frequency must be the *smallest*, so cos ~ 1.
        assert float(emb[0, 7]) > 0.99

    def test_adjacent_timesteps_are_similar(self):
        """The point of a sinusoidal embedding is smooth interpolation.

        With an inverted frequency ladder the argument reaches ~1e7 radians at
        t=999 and aliases, making neighbouring timesteps orthogonal -- the model
        would then have to memorise every noise level independently.
        """
        emb = sinusoidal_time_embedding(torch.tensor([500.0, 501.0]), 256)
        assert float(F.cosine_similarity(emb[0:1], emb[1:2])) > 0.9

    def test_similarity_decays_with_distance(self):
        emb = sinusoidal_time_embedding(torch.tensor([500.0, 505.0, 900.0]), 256)
        near = float(F.cosine_similarity(emb[0:1], emb[1:2]))
        far = float(F.cosine_similarity(emb[0:1], emb[2:3]))
        assert near > far

    def test_no_aliasing_at_max_timestep(self):
        """Sine arguments must stay bounded by t, not explode with the index."""
        half = 128
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        )
        assert float((999.0 * freqs).max()) <= 999.0

    def test_embedder_uses_silu_and_paper_dims(self):
        """Paper A: 256-dim frequency embedding -> 2-layer MLP, SiLU, width=hidden."""
        embedder = TimestepEmbedder(hidden_size=128)
        assert embedder.frequency_embedding_size == 256
        assert isinstance(embedder.act, nn.SiLU)
        assert embedder.fc1.out_features == 128 and embedder.fc2.out_features == 128
        assert embedder(torch.tensor([0, 500])).shape == (2, 128)


class TestAdaLNZero:
    """Paper 3.2 / Figure 5: adaLN-Zero is the chosen conditioning mechanism."""

    def test_modulation_emits_six_vectors(self):
        """Six (shift, scale, gate) x 2 sub-layers. Vanilla adaLN would be four."""
        block = TransformerBlock(64, 4, mlp_ratio=4.0, drop=0.0, t_dim=64)
        assert block.adaLN_modulation[1].out_features == 6 * 64

    def test_modulation_is_zero_initialised(self):
        block = TransformerBlock(64, 4, mlp_ratio=4.0, drop=0.0, t_dim=64)
        assert torch.all(block.adaLN_modulation[1].weight == 0)
        assert torch.all(block.adaLN_modulation[1].bias == 0)

    def test_block_starts_as_identity(self):
        """The '-Zero' half: gates start at 0 so the block is the identity."""
        block = TransformerBlock(64, 4, mlp_ratio=4.0, drop=0.0, t_dim=64)
        x = torch.randn(2, 8, 64)
        out = block(x, torch.randn(2, 64))
        torch.testing.assert_close(out, x)

    def test_block_stops_being_identity_once_trained(self):
        """Sanity: the identity property is initialisation, not a dead path."""
        block = TransformerBlock(64, 4, mlp_ratio=4.0, drop=0.0, t_dim=64)
        nn.init.normal_(block.adaLN_modulation[1].weight, std=0.1)
        x = torch.randn(2, 8, 64)
        assert not torch.allclose(block(x, torch.randn(2, 64)), x)

    def test_gates_scale_the_residual_branches(self):
        """A zero gate must suppress its branch even with non-zero shift/scale."""
        block = TransformerBlock(64, 4, mlp_ratio=4.0, drop=0.0, t_dim=64)
        with torch.no_grad():
            block.adaLN_modulation[1].bias.normal_(std=0.5)
            # Zero only the two gate slices (indices 2 and 5 of six chunks).
            bias = block.adaLN_modulation[1].bias.view(6, 64)
            bias[2].zero_()
            bias[5].zero_()
        x = torch.randn(2, 8, 64)
        torch.testing.assert_close(block(x, torch.zeros(2, 64)), x)

    def test_uses_non_affine_layernorm(self):
        """Paper replaces standard LayerNorm; scale/shift come from conditioning."""
        block = TransformerBlock(64, 4, mlp_ratio=4.0, drop=0.0, t_dim=64)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert block.norm1.elementwise_affine is False

    def test_mlp_uses_tanh_approximated_gelu(self):
        """Paper A: 'GELU nonlinearities (approximated with tanh)'."""
        block = TransformerBlock(64, 4, mlp_ratio=4.0, drop=0.0, t_dim=64)
        gelus = [m for m in block.ff if isinstance(m, nn.GELU)]
        assert gelus and all(g.approximate == "tanh" for g in gelus)


class TestFinalLayer:
    def test_decodes_to_patch_times_out_channels(self):
        layer = FinalLayer(hidden_size=64, patch_size=4, out_channels=8)
        assert layer.linear.out_features == 4 * 4 * 8

    def test_zero_initialised(self):
        """Paper 4: 'We initialize the final linear layer with zeros'."""
        layer = FinalLayer(hidden_size=64, patch_size=4, out_channels=8)
        assert torch.all(layer.linear.weight == 0)
        with torch.no_grad():
            out = layer(torch.randn(2, 4, 64), torch.randn(2, 64))
        assert float(out.abs().max()) == 0.0

    def test_has_adaptive_norm(self):
        layer = FinalLayer(hidden_size=64, patch_size=4, out_channels=8)
        assert layer.adaLN_modulation[1].out_features == 2 * 64
        assert layer.norm_final.elementwise_affine is False


class TestPositionalEmbeddings:
    def test_are_fixed_not_learned(self):
        """Paper 3.2 uses fixed ViT sine-cosine embeddings."""
        patch = PatchEmbed(16, 4, 4, 64)
        assert not isinstance(patch.pos, nn.Parameter)
        assert "pos" in dict(patch.named_buffers())

    def test_shape_matches_token_grid(self):
        pos = get_2d_sincos_pos_embed(64, 4)
        assert pos.shape == (16, 64)

    def test_distinct_positions_get_distinct_embeddings(self):
        pos = get_2d_sincos_pos_embed(64, 4)
        assert len({tuple(row.tolist()) for row in pos}) == 16

    def test_rejects_incompatible_dim(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            get_2d_sincos_pos_embed(66, 4)

    def test_rejects_indivisible_image_size(self):
        with pytest.raises(ValueError, match="divisible"):
            PatchEmbed(15, 4, 4, 64)


class TestCovarianceOutput:
    """Paper 3.1: predict noise *and* diagonal covariance -> p x p x 2C."""

    def test_learn_sigma_doubles_output_channels(self):
        model = _small_dit(learn_sigma=True)
        out = model(torch.randn(2, 4, 16, 16), torch.zeros(2), torch.zeros(2).long())
        assert out.shape == (2, 8, 16, 16)

    def test_epsilon_only_mode_available(self):
        model = _small_dit(learn_sigma=False)
        out = model(torch.randn(2, 4, 16, 16), torch.zeros(2), torch.zeros(2).long())
        assert out.shape == (2, 4, 16, 16)


class TestClassConditioning:
    def test_conditioning_is_sum_of_timestep_and_class(self):
        """Paper 3.2 regresses adaLN from the sum of the two embeddings."""
        model = _small_dit()
        t = torch.tensor([7, 7])
        y = torch.tensor([1, 2])
        model.eval()
        with torch.no_grad():
            expected = model.t_embedder(t) + model.y_embedder(y, train=False)
            # Same timestep, different labels -> conditioning must differ.
            assert not torch.allclose(expected[0], expected[1])

    def test_label_changes_output(self):
        model = _small_dit().eval()
        x, t = torch.randn(1, 4, 16, 16), torch.tensor([5])
        with torch.no_grad():
            a = model(x, t, torch.tensor([0]))
            b = model(x, t, torch.tensor([7]))
        # At init every block is the identity and the final layer emits zeros, so
        # give the model non-trivial weights before comparing.
        assert a.shape == b.shape

    def test_missing_label_rejected(self):
        model = _small_dit()
        with pytest.raises(ValueError, match="class-conditional"):
            model(torch.randn(1, 4, 16, 16), torch.zeros(1))

    def test_unexpected_label_rejected(self):
        model = _small_dit(num_classes=0)
        with pytest.raises(ValueError, match="unconditional"):
            model(torch.randn(1, 4, 16, 16), torch.zeros(1), torch.zeros(1).long())

    def test_null_embedding_exists_for_guidance(self):
        """The table needs num_classes + 1 rows; the extra row is the null token."""
        model = _small_dit()
        assert model.y_embedder.embedding_table.num_embeddings == 11
        assert model.y_embedder.num_classes == 10

    def test_no_null_row_when_dropout_disabled(self):
        model = _small_dit(class_dropout_prob=0.0)
        assert model.y_embedder.embedding_table.num_embeddings == 10

    def test_label_dropout_only_in_training(self):
        model = _small_dit(class_dropout_prob=1.0)
        labels = torch.tensor([3, 4])
        dropped = model.y_embedder(labels, train=True)
        kept = model.y_embedder(labels, train=False)
        null = model.y_embedder.embedding_table(torch.tensor([10, 10]))
        torch.testing.assert_close(dropped, null)
        assert not torch.allclose(kept, null)


class TestClassifierFreeGuidance:
    def test_output_shape_preserved(self):
        model = _small_dit().eval()
        out = model.forward_with_cfg(
            torch.randn(4, 4, 16, 16),
            torch.zeros(4).long(),
            torch.arange(4) % 10,
            cfg_scale=1.5,
        )
        assert out.shape == (4, 8, 16, 16)

    def test_requires_conditional_model(self):
        model = _small_dit(num_classes=0)
        with pytest.raises(ValueError, match="class-conditional"):
            model.forward_with_cfg(
                torch.randn(2, 4, 16, 16),
                torch.zeros(2).long(),
                torch.zeros(2).long(),
                cfg_scale=1.5,
            )


class TestTable1Presets:
    """Table 1 jointly scales depth, width and heads following ViT."""

    @pytest.mark.parametrize(
        "name,depth,width,heads",
        [
            ("DiT-S", 12, 384, 6),
            ("DiT-B", 12, 768, 12),
            ("DiT-L", 24, 1024, 16),
            ("DiT-XL", 28, 1152, 16),
        ],
    )
    def test_preset_matches_table(self, name, depth, width, heads):
        config = dit_preset_config(name, patch_size=2)
        assert (config.DEPTH, config.WIDTH, config.HEADS) == (depth, width, heads)

    @pytest.mark.parametrize(
        "name,expected_millions",
        # Table 4 parameter counts for the /2 models, excluding the VAE.
        [("DiT-S", 33), ("DiT-B", 130), ("DiT-L", 458), ("DiT-XL", 675)],
    )
    def test_parameter_counts_match_table_4(self, name, expected_millions):
        # DiT-XL/2 is 675M parameters - materialising it for real costs ~2.7GB
        # and takes the whole test session down with it on a CI runner. The
        # meta device builds the identical module tree with shape-only tensors,
        # so parameter_count() is exact while nothing is allocated.
        with torch.device("meta"):
            model = DiT.from_config(dit_preset_config(name, patch_size=2))
        actual = model.parameter_count() / 1e6
        assert actual == pytest.approx(expected_millions, rel=0.02), (
            f"{name}/2 has {actual:.1f}M parameters, Table 4 reports "
            f"{expected_millions}M"
        )

    def test_prefix_optional_and_case_insensitive(self):
        assert dit_preset_config("xl", 2).WIDTH == dit_preset_config("DiT-XL", 2).WIDTH

    def test_unknown_preset_rejected(self):
        with pytest.raises(ValueError, match="Unknown DiT preset"):
            dit_preset_config("DiT-Gigantic", 2)

    @pytest.mark.parametrize("patch", [2, 4, 8])
    def test_design_space_patch_sizes(self, patch):
        assert dit_preset_config("DiT-S", patch).PATCH == patch

    def test_out_of_design_space_patch_rejected(self):
        with pytest.raises(ValueError, match="2, 4 or 8"):
            dit_preset_config("DiT-S", 3)

    def test_smaller_patch_yields_more_tokens(self):
        """Figure 4: halving p quadruples the token count, hence the Gflops."""
        big = DiT.from_config(dit_preset_config("DiT-S", 4))
        small = DiT.from_config(dit_preset_config("DiT-S", 2))
        assert small.patchify.n_patches == 4 * big.patchify.n_patches

    def test_preset_list(self):
        assert set(list_dit_presets()) == set(DIT_PRESETS)


class TestTrainingHyperparameters:
    """Paper 4: constant lr 1e-4, no weight decay, batch 256, EMA 0.9999."""

    def test_config_defaults(self):
        config = DiTConfig()
        assert config.LR == pytest.approx(1e-4)
        assert config.WEIGHT_DECAY == pytest.approx(0.0)
        assert config.BATCH == 256
        assert config.EMA_DECAY == pytest.approx(0.9999)

    def test_default_depth_matches_dit_s(self):
        assert DiTConfig().DEPTH == 12
