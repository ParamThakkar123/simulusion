"""Tests for the ConvVAE loss function."""

import torch
from simulusion.losses.convae_loss import conv_vae_loss_fn


class TestConvVAELoss:
    def test_loss_is_scalar(self):
        x = torch.randn(4, 3, 64, 64)
        reconst = torch.randn(4, 3, 64, 64)
        mu = torch.randn(4, 32)
        logsigma = torch.randn(4, 32)
        loss = conv_vae_loss_fn(reconst, x, mu, logsigma)
        assert loss.ndim == 0

    def test_loss_positive(self):
        x = torch.randn(4, 3, 64, 64)
        reconst = torch.randn(4, 3, 64, 64)
        mu = torch.randn(4, 32)
        logsigma = torch.randn(4, 32)
        loss = conv_vae_loss_fn(reconst, x, mu, logsigma)
        assert loss > 0

    def test_perfect_reconstruction_lower_loss(self):
        x = torch.randn(4, 3, 64, 64)
        mu = torch.zeros(4, 32)
        logsigma = torch.zeros(4, 32)
        perfect_loss = conv_vae_loss_fn(x, x, mu, logsigma)
        bad_reconst = torch.randn(4, 3, 64, 64)
        bad_loss = conv_vae_loss_fn(bad_reconst, x, mu, logsigma)
        assert perfect_loss < bad_loss

    def test_differentiable(self):
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        reconst = torch.randn(2, 3, 64, 64, requires_grad=True)
        mu = torch.randn(2, 32, requires_grad=True)
        logsigma = torch.randn(2, 32, requires_grad=True)
        loss = conv_vae_loss_fn(reconst, x, mu, logsigma)
        loss.backward()
        assert x.grad is not None
        assert reconst.grad is not None
        assert mu.grad is not None
        assert logsigma.grad is not None

    def test_kl_divergence_zero_for_unit_normal(self):
        x = torch.randn(2, 3, 64, 64)
        reconst = x.clone()
        mu = torch.zeros(2, 32)
        logsigma = torch.zeros(2, 32)
        loss = conv_vae_loss_fn(reconst, x, mu, logsigma)
        reconstruction_loss = ((reconst - x) ** 2).sum()
        assert loss >= reconstruction_loss
        assert loss == reconstruction_loss

    def test_larger_batch(self):
        x = torch.randn(16, 3, 64, 64)
        reconst = torch.randn(16, 3, 64, 64)
        mu = torch.randn(16, 32)
        logsigma = torch.randn(16, 32)
        loss = conv_vae_loss_fn(reconst, x, mu, logsigma)
        assert loss.ndim == 0
        assert loss > 0
