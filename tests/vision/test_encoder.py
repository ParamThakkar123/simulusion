import pytest
import torch

from simulusion.vision.dreamer_encoder import ConvEncoder
from simulusion.vision.planet_encoder import CNNEncoder


class TestConvEncoder:
    @pytest.fixture
    def encoder(self):
        return ConvEncoder(input_shape=(3, 64, 64), embed_size=512, activation="relu")

    def test_initialization(self, encoder):
        assert encoder.input_shape == (3, 64, 64)
        assert encoder.embed_size == 512
        assert encoder.depth == 32
        assert len(encoder.kernels) == 4

    def test_forward_shape(self, encoder):
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        assert out.shape == (2, 512)

    def test_forward_single_batch(self, encoder):
        x = torch.randn(1, 3, 64, 64)
        out = encoder(x)
        assert out.shape == (1, 512)

    def test_forward_no_embed(self):
        encoder = ConvEncoder(
            input_shape=(3, 64, 64), embed_size=1024, activation="relu"
        )
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        assert out.shape == (2, 1024)


class TestCNNEncoder:
    @pytest.fixture
    def encoder(self):
        return CNNEncoder(embedding_size=256, activation_function="relu")

    def test_initialization(self, encoder):
        assert encoder.embedding_size == 256
        assert encoder.conv1.in_channels == 3
        assert encoder.conv1.out_channels == 32

    def test_forward_shape(self, encoder):
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_forward_single_batch(self, encoder):
        x = torch.randn(1, 3, 64, 64)
        out = encoder(x)
        assert out.shape == (1, 256)

    def test_forward_no_embed(self):
        encoder = CNNEncoder(embedding_size=1024, activation_function="relu")
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        assert out.shape == (2, 1024)
