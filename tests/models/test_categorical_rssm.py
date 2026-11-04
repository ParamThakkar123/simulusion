"""Tests for the DreamerV3 categorical RSSM and its block-diagonal GRU."""

import pytest
import torch

from simulusion.layers.block_gru import BlockGRUCell, BlockLinear
from simulusion.models.categorical_rssm import CategoricalRSSM


class TestBlockLinear:
    def test_output_shape(self):
        layer = BlockLinear(16, 24, blocks=4)
        assert layer(torch.randn(5, 16)).shape == (5, 24)

    def test_preserves_leading_dimensions(self):
        layer = BlockLinear(16, 8, blocks=4)
        assert layer(torch.randn(3, 7, 16)).shape == (3, 7, 8)

    def test_blocks_do_not_mix(self):
        # The defining property: output block i must be unaffected by input
        # block j for i != j. Output 0 lives in block 0, which reads inputs 0-1.
        layer = BlockLinear(8, 8, blocks=4)
        inputs = torch.zeros(1, 8, requires_grad=True)
        layer(inputs)[0, 0].backward()
        grad = inputs.grad
        assert grad is not None
        assert float(grad[0, :2].abs().sum()) > 0.0
        assert torch.equal(grad[0, 2:], torch.zeros(6))

    def test_uses_a_fraction_of_a_dense_layer_s_parameters(self):
        # This is the point of the block structure: `blocks` times fewer
        # recurrent weights than a dense layer covering the same units.
        size, blocks = 64, 8
        block_layer = BlockLinear(size, size, blocks=blocks)
        assert block_layer.weight.numel() == size * size // blocks
        assert block_layer.weight.numel() < torch.nn.Linear(size, size).weight.numel()

    def test_rejects_indivisible_sizes(self):
        with pytest.raises(ValueError, match="divisible"):
            BlockLinear(10, 8, blocks=4)
        with pytest.raises(ValueError, match="divisible"):
            BlockLinear(8, 10, blocks=4)


class TestBlockGRUCell:
    def test_output_shape(self):
        cell = BlockGRUCell(32, blocks=4)
        state = cell(torch.randn(6, 32), torch.zeros(6, 32))
        assert state.shape == (6, 32)

    def test_is_differentiable(self):
        cell = BlockGRUCell(16, blocks=4)
        inputs = torch.randn(2, 16, requires_grad=True)
        cell(inputs, torch.zeros(2, 16)).sum().backward()
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()

    def test_update_gate_is_biased_towards_retaining_state(self):
        # With zero input the gate offset should keep most of the prior state.
        cell = BlockGRUCell(32, blocks=4)
        previous = torch.ones(1, 32)
        state = cell(torch.zeros(1, 32), previous)
        assert float((state - previous).abs().mean()) < 0.5

    def test_rejects_indivisible_hidden_size(self):
        with pytest.raises(ValueError, match="divisible"):
            BlockGRUCell(30, blocks=8)


@pytest.fixture
def rssm():
    return CategoricalRSSM(
        action_size=3,
        embed_size=64,
        latent_dim=8,
        latent_classes=6,
        deter_size=32,
        hidden_size=16,
        gru_blocks=4,
    )


class TestCategoricalRSSMState:
    def test_init_state_shapes(self, rssm):
        state = rssm.init_state(4, torch.device("cpu"))
        assert state["logit"].shape == (4, 8, 6)
        assert state["stoch"].shape == (4, 8, 6)
        assert state["deter"].shape == (4, 32)

    def test_feature_size(self, rssm):
        assert rssm.feature_size == 8 * 6 + 32
        state = rssm.init_state(4, torch.device("cpu"))
        assert rssm.get_feat(state).shape == (4, rssm.feature_size)

    def test_seq_to_batch_flattens_time(self, rssm):
        state = {
            "logit": torch.zeros(5, 4, 8, 6),
            "stoch": torch.zeros(5, 4, 8, 6),
            "deter": torch.zeros(5, 4, 32),
        }
        flat = rssm.seq_to_batch(state)
        assert flat["deter"].shape == (20, 32)
        assert flat["stoch"].shape == (20, 8, 6)


class TestCategoricalSampling:
    def test_samples_are_one_hot(self, rssm):
        logits = torch.randn(7, 8, 6)
        sample = rssm._sample(logits)
        assert sample.shape == (7, 8, 6)
        assert torch.allclose(sample.sum(-1), torch.ones(7, 8), atol=1e-5)
        # Exactly one entry per categorical is 1 in the forward pass.
        assert int((sample > 0.99).sum()) == 7 * 8

    def test_straight_through_gradients_reach_the_logits(self, rssm):
        logits = torch.randn(4, 8, 6, requires_grad=True)
        rssm._sample(logits).sum().backward()
        assert logits.grad is not None
        assert float(logits.grad.abs().sum()) > 0.0

    def test_unimix_bounds_probabilities_away_from_zero(self, rssm):
        # A hugely peaked logit vector must still leave unimix/classes mass on
        # every class, which is what keeps the KL terms finite.
        extreme = torch.zeros(1, 8, 6)
        extreme[..., 0] = 50.0
        mixed = rssm._apply_unimix(extreme)
        probs = torch.softmax(mixed, dim=-1)
        assert float(probs.min()) >= rssm.unimix / rssm.latent_classes - 1e-6

    def test_unimix_probabilities_still_sum_to_one(self, rssm):
        mixed = rssm._apply_unimix(torch.randn(3, 8, 6))
        probs = torch.softmax(mixed, dim=-1)
        assert torch.allclose(probs.sum(-1), torch.ones(3, 8), atol=1e-5)

    def test_unimix_disabled_is_a_passthrough(self):
        model = CategoricalRSSM(
            action_size=2,
            embed_size=8,
            latent_dim=2,
            latent_classes=3,
            deter_size=8,
            hidden_size=8,
            gru_blocks=2,
            unimix=0.0,
        )
        logits = torch.randn(2, 2, 3)
        assert torch.equal(model._apply_unimix(logits), logits)


class TestCategoricalRSSMSteps:
    def test_imagine_step_shapes(self, rssm):
        state = rssm.init_state(3, torch.device("cpu"))
        prior = rssm.imagine_step(state, torch.zeros(3, 3))
        assert prior["deter"].shape == (3, 32)
        assert prior["stoch"].shape == (3, 8, 6)

    def test_observe_step_shares_the_deterministic_state(self, rssm):
        state = rssm.init_state(3, torch.device("cpu"))
        posterior, prior = rssm.observe_step(
            state, torch.zeros(3, 3), torch.randn(3, 64)
        )
        assert torch.equal(posterior["deter"], prior["deter"])

    def test_posterior_differs_from_prior(self, rssm):
        state = rssm.init_state(3, torch.device("cpu"))
        posterior, prior = rssm.observe_step(
            state, torch.zeros(3, 3), torch.randn(3, 64) * 5
        )
        assert not torch.allclose(posterior["logit"], prior["logit"])

    def test_is_first_resets_the_recurrent_state(self, rssm):
        state = rssm.init_state(2, torch.device("cpu"))
        state["deter"] = torch.randn(2, 32)
        state["stoch"] = torch.rand(2, 8, 6)
        action = torch.randn(2, 3)

        reset = rssm.imagine_step(state, action, torch.ones(2, 1))
        from_zero = rssm.imagine_step(
            rssm.init_state(2, torch.device("cpu")),
            torch.zeros(2, 3),
            torch.zeros(2, 1),
        )
        assert torch.allclose(reset["deter"], from_zero["deter"], atol=1e-6)

    def test_is_first_zero_keeps_history(self, rssm):
        state = rssm.init_state(2, torch.device("cpu"))
        state["deter"] = torch.randn(2, 32)
        kept = rssm.imagine_step(state, torch.zeros(2, 3), torch.zeros(2, 1))
        reset = rssm.imagine_step(state, torch.zeros(2, 3), torch.ones(2, 1))
        assert not torch.allclose(kept["deter"], reset["deter"])


class TestCategoricalRSSMRollouts:
    def test_observe_rollout_shapes(self, rssm):
        seq_len, batch = 5, 3
        posterior, prior = rssm.observe_rollout(
            torch.randn(seq_len, batch, 64),
            torch.zeros(seq_len, batch, 3),
            torch.zeros(seq_len, batch, 1),
            rssm.init_state(batch, torch.device("cpu")),
        )
        for state in (posterior, prior):
            assert state["deter"].shape == (seq_len, batch, 32)
            assert state["logit"].shape == (seq_len, batch, 8, 6)

    def test_imagine_rollout_returns_one_extra_state(self, rssm):
        start = rssm.init_state(4, torch.device("cpu"))
        states, actions = rssm.imagine_rollout(
            lambda feat: torch.zeros(feat.shape[0], 3), start, horizon=6
        )
        # horizon + 1 states so the policy can be scored on the start state.
        assert states["deter"].shape == (7, 4, 32)
        assert actions.shape == (6, 4, 3)

    def test_imagine_rollout_starts_from_the_given_state(self, rssm):
        start = rssm.init_state(2, torch.device("cpu"))
        start["deter"] = torch.randn(2, 32)
        states, _ = rssm.imagine_rollout(
            lambda feat: torch.zeros(feat.shape[0], 3), start, horizon=2
        )
        assert torch.equal(states["deter"][0], start["deter"])


class TestCategoricalKL:
    def test_kl_of_identical_distributions_is_zero(self, rssm):
        logits = torch.randn(4, 8, 6)
        kl = rssm.kl_divergence(logits, logits.clone())
        assert torch.allclose(kl, torch.zeros(4), atol=1e-5)

    def test_kl_is_non_negative(self, rssm):
        kl = rssm.kl_divergence(torch.randn(9, 8, 6), torch.randn(9, 8, 6))
        assert float(kl.min()) >= -1e-5

    def test_kl_shape_sums_over_latent_factors(self, rssm):
        kl = rssm.kl_divergence(torch.randn(5, 3, 8, 6), torch.randn(5, 3, 8, 6))
        assert kl.shape == (5, 3)

    def test_detached_argument_blocks_that_gradient_path(self, rssm):
        posterior = torch.randn(2, 8, 6, requires_grad=True)
        prior = torch.randn(2, 8, 6, requires_grad=True)
        rssm.kl_divergence(posterior.detach(), prior).sum().backward()
        assert posterior.grad is None
        assert prior.grad is not None
