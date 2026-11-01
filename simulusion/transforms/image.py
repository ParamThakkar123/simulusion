from typing import Any
from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms

logger = getLogger()


def make_transforms(
    crop_size: int = 224,
    crop_scale: tuple[float, float] = (0.3, 1.0),
    color_jitter: float = 1.0,
    horizontal_flip: bool = False,
    color_distortion: bool = False,
    gaussian_blur: bool = False,
    normalization: tuple[tuple[float, ...], tuple[float, ...]] = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
) -> Any:
    """Compose image augmentations and normalization for vision model training.

    Supports random crops, optional flip/color distortion/blur, and returns a
    `torchvision.transforms.Compose` pipeline.
    """
    logger.info("making imagenet data transforms")

    def get_color_distortion(s: float = 1.0) -> Any:
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    """Probabilistic Gaussian blur augmentation for PIL images.

    Applies blur with random radius in a configurable range when sampled.
    """

    def __init__(
        self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ) -> None:
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Any) -> Any:
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1).item() * (
            self.radius_max - self.radius_min
        )
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
