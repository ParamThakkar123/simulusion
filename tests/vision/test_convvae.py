"""Tests for the ConvVAE model (ConvVAEEncoder, ConvVAEDecoder, ConvVAE)."""

import torch
import pytest
from simulusion.vision.VAE.ConvVAE import ConvVAEEncoder, ConvVAEDecoder, ConvVAE


class TestConvVAEEncoder:
    @pytest.fixture
    def encoder(self):
        return ConvVAEEncoder(img_channels=3, latent_size=32)

    def test_forward_shape(self, encoder):
        x = torch.randn(4, 3, 64, 64)
        mu, logsigma = encoder(x)
        assert mu.shape == (4, 32)
        assert logsigma.shape == (4, 32)

    def test_forward_gradient_flow(self, encoder):
        x = torch.randn(2, 3, 64, 64)
        mu, logsigma = encoder(x)
        loss = mu.sum() + logsigma.sum()
        loss.backward()
        for param in encoder.parameters():
            assert param.grad is not None


class TestConvVAEDecoder:
    @pytest.fixture
    def decoder(self):
        return ConvVAEDecoder(latent_size=32, img_channels=3)

    def test_forward_shape(self, decoder):
        z = torch.randn(4, 32)
        out = decoder(z)
        assert out.shape == (4, 3, 64, 64)

    def test_output_range(self, decoder):
        z = torch.randn(4, 32)
        out = decoder(z)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_forward_gradient_flow(self, decoder):
        z = torch.randn(2, 32)
        out = decoder(z)
        out.sum().backward()
        for param in decoder.parameters():
            assert param.grad is not None


class TestConvVAE:
    @pytest.fixture
    def vae(self):
        return ConvVAE(img_channels=3, latent_size=32)

    def test_forward_shape(self, vae):
        x = torch.randn(4, 3, 64, 64)
        recon, mu, logsigma = vae(x)
        assert recon.shape == (4, 3, 64, 64)
        assert mu.shape == (4, 32)
        assert logsigma.shape == (4, 32)

    def test_reconstruction_output_range(self, vae):
        x = torch.randn(4, 3, 64, 64)
        recon, mu, logsigma = vae(x)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0

    def test_differentiable(self, vae):
        x = torch.randn(2, 3, 64, 64)
        recon, mu, logsigma = vae(x)
        loss = ((recon - x) ** 2).mean() + mu.pow(2).mean()
        loss.backward()
        for name, param in vae.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_reconstruction_differs_per_call(self, vae):
        x = torch.randn(4, 3, 64, 64)
        recon1, _, _ = vae(x)
        recon2, _, _ = vae(x)
        assert not torch.allclose(recon1, recon2)

    def test_different_latent_sizes(self):
        for latent_size in [16, 32, 64]:
            vae = ConvVAE(img_channels=3, latent_size=latent_size)
            x = torch.randn(2, 3, 64, 64)
            recon, mu, logsigma = vae(x)
            assert mu.shape[-1] == latent_size
            assert logsigma.shape[-1] == latent_size
            assert recon.shape == (2, 3, 64, 64)
