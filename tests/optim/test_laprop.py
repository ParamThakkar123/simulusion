"""Tests for the LaProp optimizer and adaptive gradient clipping."""

import pytest
import torch

from simulusion.optim import LaProp, adaptive_grad_clip_


class TestAdaptiveGradClip:
    def test_leaves_small_gradients_untouched(self):
        param = torch.nn.Parameter(torch.ones(10))  # norm sqrt(10) ~ 3.16
        param.grad = torch.full((10,), 0.001)
        before = param.grad.clone()
        adaptive_grad_clip_([param], clip=0.3)
        assert torch.equal(param.grad, before)

    def test_clips_large_gradients_to_the_threshold(self):
        param = torch.nn.Parameter(torch.ones(4))  # norm 2.0
        param.grad = torch.full((4,), 100.0)
        adaptive_grad_clip_([param], clip=0.3)
        assert float(param.grad.norm()) == pytest.approx(0.3 * 2.0, rel=1e-4)

    def test_clipping_preserves_gradient_direction(self):
        param = torch.nn.Parameter(torch.ones(3))
        param.grad = torch.tensor([3.0, 6.0, 9.0])
        adaptive_grad_clip_([param], clip=0.1)
        normalized = param.grad / param.grad.norm()
        expected = torch.tensor([3.0, 6.0, 9.0])
        expected = expected / expected.norm()
        assert torch.allclose(normalized, expected, atol=1e-5)

    def test_threshold_scales_with_the_weight_norm(self):
        small = torch.nn.Parameter(torch.full((4,), 0.5))
        large = torch.nn.Parameter(torch.full((4,), 5.0))
        for param in (small, large):
            param.grad = torch.full((4,), 100.0)
        adaptive_grad_clip_([small], clip=0.3)
        adaptive_grad_clip_([large], clip=0.3)
        assert float(large.grad.norm()) == pytest.approx(
            10.0 * float(small.grad.norm()), rel=1e-4
        )

    def test_zero_weights_still_receive_gradient_via_epsilon(self):
        # Zero-initialized output layers must not have their gradients zeroed.
        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = torch.full((4,), 1.0)
        adaptive_grad_clip_([param], clip=0.3, eps=1e-3)
        assert float(param.grad.abs().sum()) > 0.0

    def test_returns_the_pre_clipping_norm(self):
        param = torch.nn.Parameter(torch.ones(4))
        param.grad = torch.full((4,), 3.0)
        total = adaptive_grad_clip_([param], clip=0.3)
        assert float(total) == pytest.approx(6.0, rel=1e-5)

    def test_handles_parameters_without_gradients(self):
        param = torch.nn.Parameter(torch.ones(4))
        assert float(adaptive_grad_clip_([param])) == 0.0

    def test_rejects_non_positive_clip(self):
        with pytest.raises(ValueError, match="clip"):
            adaptive_grad_clip_([], clip=0.0)


class TestLaProp:
    def test_minimizes_a_quadratic(self):
        param = torch.nn.Parameter(torch.tensor([5.0]))
        optimizer = LaProp([param], lr=0.1)
        for _ in range(300):
            optimizer.zero_grad()
            (param**2).sum().backward()
            optimizer.step()
        assert abs(float(param)) < 0.5

    def test_first_step_size_is_bounded_by_the_learning_rate(self):
        # RMSProp normalization makes the first update approximately +-lr,
        # independent of the gradient magnitude.
        for gradient in (1e-3, 1.0, 1e6):
            param = torch.nn.Parameter(torch.zeros(1))
            optimizer = LaProp([param], lr=0.01)
            param.grad = torch.tensor([gradient])
            optimizer.step()
            assert float(param.abs()) == pytest.approx(0.01, rel=1e-3)

    def test_tolerates_a_tiny_epsilon(self):
        param = torch.nn.Parameter(torch.ones(3))
        optimizer = LaProp([param], lr=1e-3, eps=1e-20)
        param.grad = torch.full((3,), 1e-8)
        optimizer.step()
        assert torch.isfinite(param).all()

    def test_zero_gradient_leaves_parameters_unchanged(self):
        param = torch.nn.Parameter(torch.ones(3))
        optimizer = LaProp([param], lr=0.1)
        param.grad = torch.zeros(3)
        optimizer.step()
        assert torch.equal(param.detach(), torch.ones(3))

    def test_state_roundtrip(self):
        param = torch.nn.Parameter(torch.ones(3))
        optimizer = LaProp([param], lr=0.1)
        param.grad = torch.full((3,), 0.5)
        optimizer.step()
        state = optimizer.state_dict()

        restored_param = torch.nn.Parameter(torch.ones(3))
        restored = LaProp([restored_param], lr=0.1)
        restored.load_state_dict(state)
        assert restored.state[restored_param]["step"] == 1

    def test_decoupled_weight_decay_shrinks_parameters(self):
        param = torch.nn.Parameter(torch.ones(3))
        optimizer = LaProp([param], lr=0.1, weight_decay=0.5)
        param.grad = torch.zeros(3)
        optimizer.step()
        assert float(param[0]) < 1.0

    def test_rejects_invalid_hyperparameters(self):
        param = torch.nn.Parameter(torch.ones(1))
        with pytest.raises(ValueError, match="learning rate"):
            LaProp([param], lr=-1.0)
        with pytest.raises(ValueError, match="beta1"):
            LaProp([param], betas=(1.5, 0.99))
        with pytest.raises(ValueError, match="beta2"):
            LaProp([param], betas=(0.9, 1.5))
        with pytest.raises(ValueError, match="epsilon"):
            LaProp([param], eps=-1.0)

    def test_rejects_sparse_gradients(self):
        param = torch.nn.Parameter(torch.ones(3))
        optimizer = LaProp([param], lr=0.1)
        indices = torch.tensor([[0]])
        values = torch.tensor([1.0])
        param.grad = torch.sparse_coo_tensor(indices, values, (3,))
        with pytest.raises(RuntimeError, match="sparse"):
            optimizer.step()

    def test_supports_a_closure(self):
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = LaProp([param], lr=0.01)

        def closure():
            optimizer.zero_grad()
            loss = (param**2).sum()
            loss.backward()
            return loss

        assert float(optimizer.step(closure)) == pytest.approx(4.0)
