"""Tests for the GMM loss function used in MDRNN training."""

import torch
from simulusion.losses.gmm_loss import gmm_loss


class TestGMMLoss:
    def test_loss_is_scalar(self):
        batch = torch.randn(4, 10, 32)
        mus = torch.randn(4, 10, 5, 32)
        sigmas = torch.exp(torch.randn(4, 10, 5, 32))
        logpi = torch.randn(4, 10, 5).log_softmax(dim=-1)
        loss = gmm_loss(batch, mus, sigmas, logpi)
        assert loss.ndim == 0

    def test_loss_positive(self):
        batch = torch.randn(4, 10, 32)
        mus = torch.randn(4, 10, 5, 32)
        sigmas = torch.exp(torch.randn(4, 10, 5, 32))
        logpi = torch.randn(4, 10, 5).log_softmax(dim=-1)
        loss = gmm_loss(batch, mus, sigmas, logpi)
        assert loss > 0

    def test_perfect_prediction_lower_loss(self):
        batch = torch.randn(4, 10, 32)
        mus = batch.unsqueeze(-2) + torch.randn(4, 10, 1, 32) * 0.01
        sigmas = torch.ones(4, 10, 5, 32) * 0.1
        logpi = torch.zeros(4, 10, 5).log_softmax(dim=-1)
        good_loss = gmm_loss(batch, mus, sigmas, logpi)

        bad_mus = torch.randn(4, 10, 5, 32)
        bad_sigmas = torch.ones(4, 10, 5, 32)
        bad_loss = gmm_loss(batch, bad_mus, bad_sigmas, logpi)
        assert good_loss < bad_loss

    def test_differentiable(self):
        batch = torch.randn(2, 5, 32)
        mus = torch.randn(2, 5, 5, 32, requires_grad=True)
        sigmas_raw = torch.randn(2, 5, 5, 32, requires_grad=True)
        sigmas = torch.exp(sigmas_raw)
        logpi_raw = torch.randn(2, 5, 5, requires_grad=True)
        logpi = logpi_raw.log_softmax(dim=-1)
        loss = gmm_loss(batch, mus, sigmas, logpi)
        loss.backward()
        assert mus.grad is not None
        assert sigmas_raw.grad is not None
        assert logpi_raw.grad is not None

    def test_single_gaussian(self):
        batch = torch.randn(4, 32)
        mus = torch.randn(4, 1, 32)
        sigmas = torch.exp(torch.randn(4, 1, 32))
        logpi = torch.zeros(4, 1).log_softmax(dim=-1)
        loss = gmm_loss(batch, mus, sigmas, logpi)
        assert loss.ndim == 0

    def test_different_gmm_components(self):
        batch = torch.randn(2, 10)
        for n_gauss in [1, 3, 5, 10]:
            mus = torch.randn(2, n_gauss, 10)
            sigmas = torch.exp(torch.randn(2, n_gauss, 10))
            logpi = torch.randn(2, n_gauss).log_softmax(dim=-1)
            loss = gmm_loss(batch, mus, sigmas, logpi)
            assert loss.ndim == 0
