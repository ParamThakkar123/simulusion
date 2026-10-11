"""Fréchet Inception Distance (FID) metric.

Computes the FID between real and generated image distributions
using features from a pretrained InceptionV3 network.

Reference:
    Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge
    to a Local Nash Equilibrium", NeurIPS 2017.
"""

from functools import lru_cache

import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 truncated at the Mixed_7c layer for 2048-dim features."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        # Remove the classification head; keep up to Mixed_7c
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # InceptionV3 expects 299x299 inputs; the network downsamples by 8x
        # to the lowest resolution feature map, so minimum input is 75x75.
        self.required_size = 299
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim features.

        Args:
            x: Input images in [0, 1] range, shape [B, C, H, W].

        Returns:
            Features of shape [B, 2048].
        """
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        # Resize to 299x299 if necessary (InceptionV3 expected input size)
        if x.shape[-1] != self.required_size or x.shape[-2] != self.required_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.required_size, self.required_size),
                mode="bilinear",
                align_corners=False,
            )
        # Scale to [0, 255]
        if x.max() <= 1.0:
            x = x * 255.0
        if x.min() < 0:
            x = (x + 1.0) / 2.0 * 255.0
        features = self.blocks(x)
        return features.view(x.size(0), -1)


def _compute_statistics(
    features: torch.Tensor, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of features.

    Args:
        features: Feature tensor [N, D].
        eps: Small diagonal regularizer for numerical stability.

    Returns:
        Mean [D] and covariance [D, D].
    """
    arr = features.detach().cpu().numpy()
    mu = np.mean(arr, axis=0)
    sigma = np.cov(arr, rowvar=False)
    # Regularize diagonal to ensure positive-definiteness
    sigma += np.eye(sigma.shape[0]) * eps
    return mu, sigma


def _sqrt_psd(m: np.ndarray) -> np.ndarray:
    """Square root of a symmetric positive semi-definite matrix."""
    w, v = np.linalg.eigh(m)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def _frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Compute the Fréchet distance between two Gaussians.

    Only the trace of ``sqrtm(sigma1 @ sigma2)`` enters the distance, and it
    equals the trace of ``sqrtm(S @ sigma2 @ S)`` with ``S = sqrtm(sigma1)``.
    That matrix is symmetric PSD, so its eigenvalues are real and non-negative
    and ``eigh`` gives the trace directly. ``scipy.linalg.sqrtm`` on the
    non-symmetric product instead runs a recursive Schur decomposition, which
    on a 2048x2048 rank-deficient covariance (any FID over fewer samples than
    feature dimensions) can take minutes.
    """
    diff = mu1 - mu2
    s1 = _sqrt_psd(sigma1)
    m = s1 @ sigma2 @ s1
    eig = np.linalg.eigvalsh((m + m.T) / 2.0)
    tr_covmean = np.sqrt(np.clip(eig, 0.0, None)).sum()
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


@lru_cache(maxsize=None)
def _cached_extractor(device_str: str) -> "InceptionFeatureExtractor":
    """Return a shared, frozen InceptionFeatureExtractor for ``device_str``.

    The backbone is a frozen ImageNet network in eval mode: it has no state that
    a caller can perturb, so every metric instance on a device can share one.
    Rebuilding it per instance re-ran the torchvision weight load (and the
    download, on a cold cache) and held a second copy of the network in memory.
    """
    extractor = InceptionFeatureExtractor(torch.device(device_str))
    extractor.requires_grad_(False)
    return extractor


class FID:
    """Fréchet Inception Distance.

    Usage:
        fid = FID(device=device)
        score = fid(real_images, generated_images)
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.extractor = _cached_extractor(str(device))

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor, desc: str = "") -> torch.Tensor:
        """Extract Inception features for a batch of images."""
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size].to(self.device)
            features = self.extractor(batch)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    def __call__(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> float:
        """Compute FID.

        Args:
            real_images: Real images [N, C, H, W] in [0, 1].
            generated_images: Generated images [M, C, H, W] in [0, 1].

        Returns:
            FID score (lower is better).
        """
        real_feat = self._extract_features(real_images, "real")
        gen_feat = self._extract_features(generated_images, "gen")

        mu_r, sigma_r = _compute_statistics(real_feat)
        mu_g, sigma_g = _compute_statistics(gen_feat)

        return _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

    def __repr__(self) -> str:
        return f"FID(device={self.device})"


class _FIDReference:
    """Constructor alias kept for API consistency."""

    pass
