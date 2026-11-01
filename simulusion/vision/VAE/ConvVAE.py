"""Convolutional Variational Autoencoder (ConvVAE) implementation.

This module provides the ConvVAE model architecture for encoding and decoding
images in the World Models framework. The VAE uses a convolutional encoder
and decoder with a variational latent space.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvVAEEncoder(nn.Module):
    """Convolutional encoder for VAE.

    This encoder takes images and produces the parameters (mean and log variance)
    of a Gaussian distribution in the latent space.

    Attributes:
        latent_size: Dimensionality of the latent space.
        img_channels: Number of input image channels.

    Example:
        >>> encoder = ConvVAEEncoder(img_channels=3, latent_size=32)
        >>> mu, logsigma = encoder(images)
    """

    def __init__(self, img_channels: int, latent_size: int):
        """Initialize the ConvVAE encoder.

        Args:
            img_channels: Number of channels in input images (e.g., 3 for RGB).
            latent_size: Dimensionality of the latent space.
        """
        super(ConvVAEEncoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_logsigma = nn.Linear(2 * 2 * 256, latent_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Tuple of (mu, logsigma) where:
                - mu: Mean of the latent distribution
                - logsigma: Log variance of the latent distribution
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_sigma = self.fc_logsigma(x)

        return mu, log_sigma


class ConvVAEDecoder(nn.Module):
    """Convolutional decoder for VAE.

    This decoder takes latent vectors and reconstructs images.

    Attributes:
        latent_size: Dimensionality of the input latent space.
        img_channels: Number of output image channels.
    """

    def __init__(self, latent_size: int, img_channels: int):
        """Initialize the ConvVAE decoder.

        Args:
            latent_size: Dimensionality of the latent space.
            img_channels: Number of channels in output images.
        """
        super(ConvVAEDecoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images.

        Args:
            z: Latent vector of shape (batch, latent_size).

        Returns:
            Reconstructed image tensor of shape (batch, channels, height, width).
        """
        x = F.relu(self.fc(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.sigmoid(self.deconv4(x))
        return x


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder.

    The ConvVAE is a generative model that encodes images into a latent
    distribution and reconstructs them. It uses the reparameterization trick
    to enable backpropagation through the sampling process.

    Attributes:
        encoder: ConvVAEEncoder that encodes images to latent parameters.
        decoder: ConvVAEDecoder that decodes latent vectors to images.

    Example:
        >>> vae = ConvVAE(img_channels=3, latent_size=32)
        >>> recon_x, mu, logsigma = vae(images)
        >>> # Training loss combines reconstruction and KL divergence
    """

    def __init__(self, img_channels: int, latent_size: int):
        """Initialize the ConvVAE.

        Args:
            img_channels: Number of channels in input/output images.
            latent_size: Dimensionality of the latent space.
        """
        super(ConvVAE, self).__init__()
        self.encoder = ConvVAEEncoder(img_channels, latent_size)
        self.decoder = ConvVAEDecoder(latent_size, img_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and decode an image.

        Args:
            x: Input image tensor of shape (batch, channels, height, width).

        Returns:
            Tuple of (recon_x, mu, logsigma):
                - recon_x: Reconstructed image
                - mu: Mean of latent distribution
                - logsigma: Log variance of latent distribution
        """
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, log_sigma
