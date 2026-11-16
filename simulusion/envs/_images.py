"""Image conversion shared by the image-based environment adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def to_hwc_uint8(frame: Any) -> np.ndarray:
    """Convert an RGB frame (HWC or CHW, uint8 or float in [0, 1]) to HWC uint8."""
    image = np.asarray(frame)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D RGB frame, got shape {image.shape}.")
    if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        # Float frames in [0, 1] (Craftax, many JAX envs) scale to [0, 255].
        if image.size and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return image


def resize_hwc(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an HWC uint8 image to ``size = (height, width)`` with bilinear filtering."""
    height, width = int(size[0]), int(size[1])
    if image.shape[0] == height and image.shape[1] == width:
        return image
    resized = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized)


def to_chw_uint8(frame: Any, size: tuple[int, int]) -> np.ndarray:
    """Convert and resize a frame to the ``(3, H, W)`` uint8 layout adapters emit."""
    return resize_hwc(to_hwc_uint8(frame), size).transpose(2, 0, 1).copy()


__all__ = ["resize_hwc", "to_chw_uint8", "to_hwc_uint8"]
