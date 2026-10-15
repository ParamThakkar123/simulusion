import pytest
import torch
from PIL import Image
import numpy as np

from simulusion.transforms.image import make_transforms, GaussianBlur


class TestMakeTransforms:
    def test_make_transforms_default(self):
        transform = make_transforms()
        assert transform is not None
        # torchvision Compose is not a torch.nn.Module
        assert hasattr(transform, "transforms")

    def test_make_transforms_with_options(self):
        transform = make_transforms(
            crop_size=128,
            horizontal_flip=True,
            color_distortion=True,
            gaussian_blur=True,
        )
        assert transform is not None

    def test_transform_applies_to_image(self):
        transform = make_transforms(crop_size=64)
        # Create a dummy PIL image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # Channels
        assert result.shape[1] == 64  # Height
        assert result.shape[2] == 64  # Width


class TestGaussianBlur:
    @pytest.fixture
    def blur(self):
        return GaussianBlur(p=1.0, radius_min=0.5, radius_max=1.0)

    def test_gaussian_blur_init(self, blur):
        assert blur.prob == 1.0
        assert blur.radius_min == 0.5
        assert blur.radius_max == 1.0

    def test_gaussian_blur_call(self, blur):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = blur(img)
        assert isinstance(result, Image.Image)

    def test_gaussian_blur_no_apply(self):
        blur = GaussianBlur(p=0.0)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = blur(img)
        # Since p=0, should return original
        assert result is img


def test_transform_package_exports_image_helpers():
    from simulusion.transforms import GaussianBlur as ExportedGaussianBlur
    from simulusion.transforms import make_transforms as exported_make_transforms

    assert exported_make_transforms is make_transforms
    assert ExportedGaussianBlur is GaussianBlur
