"""Loss functions for World Models training.

This module provides loss functions for training VAE and other world model components.
"""

import torch.nn.functional as F
import torch


def conv_vae_loss_fn(
    reconst: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logsigma: torch.Tensor
) -> torch.Tensor:
    """Compute the ConvVAE loss function.

    The loss combines:
    1. Reconstruction loss (MSE) between input and reconstructed images
    2. KL divergence between learned latent distribution and prior (standard normal)

    The total loss is: BCE + KLD

    Args:
        reconst: Reconstructed images from the VAE decoder.
        x: Original input images.
        mu: Mean of the latent distribution.
        logsigma: Log variance of the latent distribution.

    Returns:
        Scalar tensor containing the total VAE loss.

    Example:
        >>> recon_x, mu, logsigma = vae(images)
        >>> loss = conv_vae_loss_fn(recon_x, images, mu, logsigma)
        >>> loss.backward()
    """
    bce = F.mse_loss(reconst, x, size_average=False)
    kld = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return bce + kld
