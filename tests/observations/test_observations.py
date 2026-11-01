import pytest
import torch

from simulusion.observations.dreamer_v1_obs import (
    SymbolicObservationModel,
    VisualObservationModel,
    ObservationModel,
)


class TestSymbolicObservationModel:
    @pytest.fixture
    def model(self):
        return SymbolicObservationModel(
            observation_size=10,
            belief_size=32,
            state_size=16,
            embedding_size=64,
            activation_function="relu",
        )

    def test_initialization(self, model):
        assert model.fc1.in_features == 32 + 16
        assert model.fc1.out_features == 64
        assert model.fc3.out_features == 10

    def test_forward_shape(self, model):
        belief = torch.randn(2, 32)
        state = torch.randn(2, 16)
        out = model(belief, state)
        assert out.shape == (2, 10)

    def test_forward_single_batch(self, model):
        belief = torch.randn(1, 32)
        state = torch.randn(1, 16)
        out = model(belief, state)
        assert out.shape == (1, 10)


class TestVisualObservationModel:
    @pytest.fixture
    def model(self):
        return VisualObservationModel(
            belief_size=32,
            state_size=16,
            embedding_size=64,
            activation_function="relu",
        )

    def test_initialization(self, model):
        assert model.embedding_size == 64
        assert model.fc1.in_features == 32 + 16
        assert model.conv4.out_channels == 3

    def test_forward_shape(self, model):
        belief = torch.randn(2, 32)
        state = torch.randn(2, 16)
        out = model(belief, state)
        # Output shape: (batch, 3, 64, 64)
        assert out.shape == (2, 3, 64, 64)

    def test_forward_single_batch(self, model):
        belief = torch.randn(1, 32)
        state = torch.randn(1, 16)
        out = model(belief, state)
        assert out.shape == (1, 3, 64, 64)


class TestObservationModel:
    def test_symbolic_model(self):
        model = ObservationModel(
            symbolic=True,
            observation_size=5,
            belief_size=20,
            state_size=10,
            embedding_size=40,
        )
        assert isinstance(model, SymbolicObservationModel)

    def test_visual_model(self):
        model = ObservationModel(
            symbolic=False,
            observation_size=5,  # ignored for visual
            belief_size=20,
            state_size=10,
            embedding_size=40,
        )
        assert isinstance(model, VisualObservationModel)
