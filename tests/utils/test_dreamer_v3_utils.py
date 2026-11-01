"""Tests for the DreamerV3 robustness primitives."""

import numpy as np
import pytest
import torch

from simulusion.utils.dreamer_utils import symexp, symlog
from simulusion.utils.dreamer_v3_utils import (
    ReturnNormalizer,
    SymexpTwoHotDist,
    free_bits,
    lambda_return,
    twohot,
)


class TestSymlog:
    def test_roundtrip(self):
        values = torch.tensor([-1e6, -12.5, -1.0, 0.0, 1.0, 12.5, 1e6])
        assert torch.allclose(symexp(symlog(values)), values, rtol=1e-4)

    def test_approximates_identity_near_origin(self):
        small = torch.tensor([-0.01, 0.0, 0.01])
        assert torch.allclose(symlog(small), small, atol=1e-4)

    def test_is_sign_preserving(self):
        values = torch.tensor([-5.0, 5.0])
        assert torch.sign(symlog(values)).tolist() == [-1.0, 1.0]


class TestTwoHot:
    @pytest.fixture
    def bins(self):
        return torch.linspace(-2.0, 2.0, 5)

    def test_sums_to_one(self, bins):
        encoded = twohot(torch.tensor([-1.7, 0.0, 0.3, 1.9]), bins)
        assert torch.allclose(encoded.sum(-1), torch.ones(4))

    def test_exactly_on_a_bin_is_one_hot(self, bins):
        encoded = twohot(torch.tensor([1.0]), bins)
        assert encoded[0].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]

    def test_interpolates_between_neighbors(self, bins):
        # 0.25 of the way from bin 0.0 (index 2) to bin 1.0 (index 3).
        encoded = twohot(torch.tensor([0.25]), bins)
        assert encoded[0, 2] == pytest.approx(0.75)
        assert encoded[0, 3] == pytest.approx(0.25)

    def test_expected_value_recovers_the_target(self, bins):
        targets = torch.tensor([-1.3, 0.7, 1.55])
        recovered = (twohot(targets, bins) * bins).sum(-1)
        assert torch.allclose(recovered, targets, atol=1e-5)

    def test_out_of_range_clips_to_boundary(self, bins):
        encoded = twohot(torch.tensor([100.0, -100.0]), bins)
        assert encoded[0, -1] == pytest.approx(1.0)
        assert encoded[1, 0] == pytest.approx(1.0)

    def test_preserves_shape(self, bins):
        encoded = twohot(torch.zeros(3, 4), bins)
        assert encoded.shape == (3, 4, 5)


class TestSymexpTwoHotDist:
    def test_zero_logits_predict_zero(self):
        # Uniform logits over a symmetric grid must average to exactly zero,
        # which is what makes zero-initializing the head heads useful.
        dist = SymexpTwoHotDist(torch.zeros(4, 255), num_bins=255)
        assert torch.allclose(dist.mean, torch.zeros(4), atol=1e-4)

    def test_bins_span_many_orders_of_magnitude(self):
        dist = SymexpTwoHotDist(torch.zeros(1, 255), num_bins=255, symlog_range=20.0)
        assert dist.bins[0] < -1e8
        assert dist.bins[-1] > 1e8
        # Dense near the origin despite the huge span.
        assert dist.bins.abs().min() < 0.1

    def test_bins_are_finite_in_float32(self):
        dist = SymexpTwoHotDist(torch.zeros(1, 255), num_bins=255, symlog_range=20.0)
        assert torch.isfinite(dist.bins).all()

    def test_log_prob_is_maximized_at_the_target(self):
        target = torch.tensor([3.0])
        logits = torch.zeros(1, 255, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.5)
        for _ in range(200):
            optimizer.zero_grad()
            loss = -SymexpTwoHotDist(logits).log_prob(target).mean()
            loss.backward()
            optimizer.step()
        assert SymexpTwoHotDist(logits).mean.item() == pytest.approx(3.0, abs=0.1)

    def test_gradient_scale_is_independent_of_target_scale(self):
        # The central claim of the two-hot loss: because the loss depends only
        # on the predicted probabilities, a target of 1e6 produces the same
        # gradient magnitude as a target of 1.
        def twohot_grad_norm(target_value):
            logits = torch.zeros(1, 255, requires_grad=True)
            loss = (
                -SymexpTwoHotDist(logits).log_prob(torch.tensor([target_value])).mean()
            )
            loss.backward()
            assert logits.grad is not None
            return float(logits.grad.abs().sum())

        assert twohot_grad_norm(1e6) == pytest.approx(twohot_grad_norm(1.0), rel=1e-4)

    def test_squared_error_gradient_does_scale_with_the_target(self):
        # Contrast with the two-hot loss above: this is the failure mode that
        # motivates it, and why a plain squared loss can diverge on large
        # targets while an absolute loss stagnates on small ones.
        def mse_grad_norm(target_value):
            prediction = torch.zeros(1, requires_grad=True)
            loss = 0.5 * (prediction - torch.tensor([target_value])).pow(2).mean()
            loss.backward()
            assert prediction.grad is not None
            return float(prediction.grad.abs().sum())

        assert mse_grad_norm(1e6) == pytest.approx(1e6 * mse_grad_norm(1.0), rel=1e-3)

    def test_rejects_mismatched_logits(self):
        with pytest.raises(ValueError, match="num_bins"):
            SymexpTwoHotDist(torch.zeros(2, 7), num_bins=255)

    def test_cross_entropy_to_self_equals_entropy(self):
        logits = torch.randn(3, 255)
        dist = SymexpTwoHotDist(logits)
        probs = dist.probs
        expected = -(probs * torch.log_softmax(logits, -1)).sum(-1)
        assert torch.allclose(dist.cross_entropy_to(dist), expected, atol=1e-5)


class TestReturnNormalizer:
    def test_large_returns_are_scaled_down(self):
        normalizer = ReturnNormalizer(decay=0.0)
        returns = torch.linspace(0.0, 1000.0, 512)
        scaled = normalizer(returns)
        assert scaled.max() < returns.max()
        assert normalizer.scale > 1.0

    def test_small_returns_pass_through_unchanged(self):
        # Under sparse rewards the percentile range collapses; the limit keeps
        # the denominator at 1 so tiny returns are not amplified.
        normalizer = ReturnNormalizer(decay=0.0, limit=1.0)
        returns = torch.zeros(256)
        returns[0] = 0.01
        scaled = normalizer(returns)
        assert torch.allclose(scaled, returns)

    def test_denominator_never_drops_below_the_limit(self):
        normalizer = ReturnNormalizer(decay=0.0, limit=1.0)
        normalizer.update(torch.zeros(64))
        assert float(normalizer.denominator()) == pytest.approx(1.0)

    def test_is_robust_to_outliers(self):
        # A single enormous return must not dominate the 5-95 percentile range.
        normalizer = ReturnNormalizer(decay=0.0)
        base = torch.ones(1000)
        normalizer.update(base)
        clean = normalizer.scale

        outlier = base.clone()
        outlier[0] = 1e9
        normalizer_b = ReturnNormalizer(decay=0.0)
        normalizer_b.update(outlier)
        assert normalizer_b.scale == pytest.approx(clean, abs=1e-3)

    def test_ema_smooths_across_batches(self):
        normalizer = ReturnNormalizer(decay=0.9)
        normalizer.update(torch.linspace(0, 100, 256))
        first = normalizer.scale
        normalizer.update(torch.zeros(256))
        assert 0.0 < normalizer.scale < first

    def test_state_roundtrip(self):
        normalizer = ReturnNormalizer(decay=0.5, limit=2.0)
        normalizer.update(torch.linspace(0, 10, 128))
        restored = ReturnNormalizer()
        restored.load_state_dict(normalizer.state_dict())
        assert restored.scale == pytest.approx(normalizer.scale)
        assert restored.limit == pytest.approx(2.0)

    def test_rejects_invalid_percentiles(self):
        with pytest.raises(ValueError):
            ReturnNormalizer(low=95.0, high=5.0)


class TestFreeBits:
    def test_clips_below_the_threshold(self):
        assert float(free_bits(torch.tensor(0.2), nats=1.0)) == pytest.approx(1.0)

    def test_passes_through_above_the_threshold(self):
        assert float(free_bits(torch.tensor(3.0), nats=1.0)) == pytest.approx(3.0)

    def test_gradient_vanishes_once_minimized(self):
        kl = torch.tensor(0.3, requires_grad=True)
        free_bits(kl, nats=1.0).backward()
        assert float(kl.grad) == 0.0

    def test_gradient_flows_when_above_the_threshold(self):
        kl = torch.tensor(2.0, requires_grad=True)
        free_bits(kl, nats=1.0).backward()
        assert float(kl.grad) == 1.0


class TestLambdaReturn:
    def test_lambda_one_is_the_monte_carlo_return(self):
        rewards = torch.ones(4, 1)
        values = torch.zeros(4, 1)
        continues = torch.full((4, 1), 0.9)
        returns = lambda_return(rewards, values, continues, torch.zeros(1), lambda_=1.0)
        expected = 1 + 0.9 * (1 + 0.9 * (1 + 0.9 * 1))
        assert float(returns[0]) == pytest.approx(expected)

    def test_lambda_zero_is_the_one_step_target(self):
        rewards = torch.ones(3, 1)
        values = torch.full((3, 1), 5.0)
        continues = torch.full((3, 1), 0.9)
        returns = lambda_return(rewards, values, continues, torch.zeros(1), lambda_=0.0)
        assert float(returns[0]) == pytest.approx(1.0 + 0.9 * 5.0)

    def test_zero_continue_truncates_the_return(self):
        rewards = torch.ones(3, 1)
        values = torch.full((3, 1), 100.0)
        continues = torch.zeros(3, 1)
        returns = lambda_return(
            rewards, values, continues, torch.full((1,), 100.0), lambda_=0.95
        )
        assert torch.allclose(returns, torch.ones(3, 1))

    def test_shape_is_preserved(self):
        returns = lambda_return(
            torch.zeros(6, 4), torch.zeros(6, 4), torch.zeros(6, 4), torch.zeros(4)
        )
        assert returns.shape == (6, 4)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="share a shape"):
            lambda_return(
                torch.zeros(3, 2),
                torch.zeros(4, 2),
                torch.zeros(3, 2),
                torch.zeros(2),
            )

    def test_matches_an_explicit_recursion(self):
        torch.manual_seed(0)
        rewards = torch.randn(5, 2)
        values = torch.randn(5, 2)
        continues = torch.rand(5, 2)
        bootstrap = torch.randn(2)
        lam = 0.7

        expected = torch.zeros(5, 2)
        accumulated = bootstrap
        for t in reversed(range(5)):
            accumulated = rewards[t] + continues[t] * (
                (1 - lam) * values[t] + lam * accumulated
            )
            expected[t] = accumulated

        actual = lambda_return(rewards, values, continues, bootstrap, lambda_=lam)
        assert torch.allclose(actual, expected, atol=1e-5)


class TestNumpyInterop:
    def test_twohot_matches_manual_numpy_computation(self):
        bins = torch.tensor([0.0, 1.0, 2.0])
        encoded = twohot(torch.tensor([0.4]), bins).numpy()
        assert np.allclose(encoded, [[0.6, 0.4, 0.0]], atol=1e-6)
