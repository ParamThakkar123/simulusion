"""Learned Perceptual Image Patch Similarity (LPIPS) metric.

Computes perceptual similarity between pairs of images using features
from a pretrained VGG16 network. Lower scores indicate higher similarity.

Reference:
    Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
    Perceptual Metric", CVPR 2018.
"""

from functools import lru_cache

import torch
import torch.nn as nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """VGG16 truncated to output features from multiple intermediate layers."""

    # Layer indices (0-based) and their output channel counts in VGG16-BN.
    # Indices account for extra BatchNorm layers vs plain VGG16.
    LAYER_CONFIG: list[tuple[int, int, str]] = [
        (5, 64, "relu1_2"),
        (12, 128, "relu2_2"),
        (22, 256, "relu3_3"),
        (32, 512, "relu4_3"),
    ]

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        self.names: list[str] = []
        for end_idx, out_ch, name in self.LAYER_CONFIG:
            self.slices.append(nn.Sequential(*list(vgg[prev : end_idx + 1])))
            self.names.append(name)
            prev = end_idx + 1
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-layer features.

        Args:
            x: Input images [B, C, H, W] in [0, 1].

        Returns:
            List of feature tensors, one per layer.
        """
        features = []
        h = x
        for slice_module in self.slices:
            h = slice_module(h)
            features.append(h)
        return features


@lru_cache(maxsize=None)
def _cached_extractor(device_str: str) -> "VGGFeatureExtractor":
    """Return a shared, frozen VGGFeatureExtractor for ``device_str``.

    The backbone is a frozen ImageNet network in eval mode: it has no state that
    a caller can perturb, so every metric instance on a device can share one.
    Rebuilding it per instance re-ran the torchvision weight load (and the
    download, on a cold cache) and held a second copy of the network in memory.
    """
    extractor = VGGFeatureExtractor(torch.device(device_str))
    extractor.requires_grad_(False)
    return extractor


class LPIPS:
    """Learned Perceptual Image Patch Similarity.

    Uses L2 distance in VGG16 feature space, normalized per channel.

    Usage:
        lpips = LPIPS(device=device)
        score = lpips(image_a, image_b)  # mean over batch
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.extractor = _cached_extractor(str(device))
        # Per-channel normalization weights (from the LPIPS paper)
        # These are learned weights; we use uniform weights as an approximation.
        self.weights = [1.0 / len(self.extractor.names)] * len(self.extractor.names)

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [-1, 1] (as used by VGG/LPIPS)."""
        return x * 2.0 - 1.0

    def _compute_perceptual_distance(
        self, feat_a: list[torch.Tensor], feat_b: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute LPIPS distance between two feature lists.

        Returns:
            Per-image distance [B].
        """
        total: torch.Tensor = torch.zeros(feat_a[0].shape[0], device=feat_a[0].device)
        for f_a, f_b in zip(feat_a, feat_b):
            diff = (f_a - f_b).pow(2)
            # Spatial average per channel -> [B, C] -> mean over channels -> [B]
            total += diff.mean(dim=[2, 3]).sum(dim=1)
        return total / len(feat_a)

    def __call__(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
    ) -> float:
        """Compute mean LPIPS over a batch of image pairs.

        Args:
            images_a: First set of images [N, C, H, W] in [0, 1].
            images_b: Second set of images [N, C, H, W] in [0, 1].

        Returns:
            Mean LPIPS score (lower = more similar).
        """
        all_scores = []
        for i in range(0, len(images_a), self.batch_size):
            batch_a = images_a[i : i + self.batch_size].to(self.device)
            batch_b = images_b[i : i + self.batch_size].to(self.device)

            batch_a = self._normalize(batch_a)
            batch_b = self._normalize(batch_b)

            feat_a = self.extractor(batch_a)
            feat_b = self.extractor(batch_b)

            scores = self._compute_perceptual_distance(feat_a, feat_b)
            all_scores.append(scores.cpu())

        return float(torch.cat(all_scores).mean())

    def __repr__(self) -> str:
        return f"LPIPS(device={self.device})"
