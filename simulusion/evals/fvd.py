"""Fréchet Video Distance (FVD) metric.

Computes the FVD between real and generated video clip distributions
using features from a pretrained 3D ResNet (R3D-18) video backbone.

Reference:
    Unterthiner et al., "Towards Accurate Generative Models of Video:
    A New Metric & Challenges", arXiv 2018.
"""

from functools import lru_cache

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from typing import Optional
import numpy as np

from simulusion.evals.fid import _frechet_distance


class VideoFeatureExtractor(nn.Module):
    """R3D-18 truncated at the avgpool layer for 512-dim video features."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        try:
            model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        except Exception:
            model = r3d_18(pretrained=True)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim video features.

        Args:
            x: Video clips [B, C, T, H, W] with T=16 frames, values in [0, 1].

        Returns:
            Features of shape [B, 512].
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [B, C, T, H, W], got shape {x.shape}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        # Resize to 112x112 if needed (R3D-18 expects 112x112 or larger)
        if x.shape[-1] != 112 or x.shape[-2] != 112:
            x = torch.nn.functional.interpolate(
                x, size=(x.shape[2], 112, 112), mode="trilinear", align_corners=False
            )
        # Normalize: R3D-18 expects ImageNet normalization
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=x.device).view(
            1, 3, 1, 1, 1
        )
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=x.device).view(
            1, 3, 1, 1, 1
        )
        x = (x - mean) / std
        features = self.backbone(x)
        return features.view(x.size(0), -1)


def _sample_clips(
    videos: torch.Tensor, clip_length: int = 16, num_clips: Optional[int] = None
) -> torch.Tensor:
    """Sample fixed-length clips from videos.

    Args:
        videos: [B, C, T, H, W] video tensors.
        clip_length: Number of frames per clip (default 16).
        num_clips: Number of clips to sample per video (default: all possible).

    Returns:
        Clips [N, C, clip_length, H, W].
    """
    B, C, T, H, W = videos.shape
    clips = []
    if T <= clip_length:
        # Pad or repeat if video is shorter than clip_length
        repeat = (clip_length + T - 1) // T
        padded = videos.repeat(1, 1, repeat, 1, 1)[:, :, :clip_length, :, :]
        clips.append(padded)
    else:
        step = max(1, (T - clip_length) // (num_clips or 1))
        for start in range(0, T - clip_length + 1, step):
            clips.append(videos[:, :, start : start + clip_length, :, :])
            if num_clips is not None and len(clips) >= num_clips:
                break
        if not clips:
            clips.append(videos[:, :, :clip_length, :, :])
    return torch.cat(clips, dim=0)


def _compute_statistics(features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of features."""
    arr = features.detach().cpu().numpy()
    mu = np.mean(arr, axis=0)
    sigma = np.cov(arr, rowvar=False)
    return mu, sigma


@lru_cache(maxsize=None)
def _cached_extractor(device_str: str) -> "VideoFeatureExtractor":
    """Return a shared, frozen VideoFeatureExtractor for ``device_str``.

    The backbone is a frozen ImageNet network in eval mode: it has no state that
    a caller can perturb, so every metric instance on a device can share one.
    Rebuilding it per instance re-ran the torchvision weight load (and the
    download, on a cold cache) and held a second copy of the network in memory.
    """
    extractor = VideoFeatureExtractor(torch.device(device_str))
    extractor.requires_grad_(False)
    return extractor


class FVD:
    """Fréchet Video Distance.

    Usage:
        fvd = FVD(device=device)
        score = fvd(real_videos, generated_videos)
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 16,
        clip_length: int = 16,
    ):
        self.device = device
        self.batch_size = batch_size
        self.clip_length = clip_length
        self.extractor = _cached_extractor(str(device))

    @torch.no_grad()
    def _extract_features(self, videos: torch.Tensor, desc: str = "") -> torch.Tensor:
        """Extract video features for a batch of clips."""
        all_features = []
        for i in range(0, len(videos), self.batch_size):
            batch = videos[i : i + self.batch_size].to(self.device)
            features = self.extractor(batch)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    def __call__(
        self,
        real_videos: torch.Tensor,
        generated_videos: torch.Tensor,
    ) -> float:
        """Compute FVD.

        Args:
            real_videos: Real videos [N, C, T, H, W] in [0, 1].
            generated_videos: Generated videos [M, C, T, H, W] in [0, 1].

        Returns:
            FVD score (lower is better).
        """
        # Sample clips of fixed length
        real_clips = _sample_clips(real_videos, self.clip_length)
        gen_clips = _sample_clips(generated_videos, self.clip_length)

        real_feat = self._extract_features(real_clips, "real")
        gen_feat = self._extract_features(gen_clips, "gen")

        mu_r, sigma_r = _compute_statistics(real_feat)
        mu_g, sigma_g = _compute_statistics(gen_feat)

        return _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

    def __repr__(self) -> str:
        return f"FVD(device={self.device})"
