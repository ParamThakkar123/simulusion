import pytest
import torch
from simulusion.vision.dreamer_decoder import ConvDecoder, DenseDecoder, ActionDecoder


class TestConvDecoder:
    def test_initialization(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (3, 64, 64)
        activation = "relu"
        depth = 32
        decoder = ConvDecoder(stoch_size, deter_size, output_shape, activation, depth)
        assert decoder.output_shape == output_shape
        assert decoder.depth == depth
        assert isinstance(decoder.dense, torch.nn.Linear)
        assert isinstance(decoder.convtranspose, torch.nn.Sequential)

    def test_forward_pass(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (3, 64, 64)
        activation = "relu"
        decoder = ConvDecoder(stoch_size, deter_size, output_shape, activation)
        batch_size = 2
        features = torch.randn(batch_size, stoch_size + deter_size)
        dist = decoder(features)
        assert dist.batch_shape == (batch_size,)
        assert dist.event_shape == output_shape
        mean = dist.mean
        assert mean.shape == (batch_size, *output_shape)

    def test_forward_pass_different_batch(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (3, 64, 64)
        activation = "relu"
        decoder = ConvDecoder(stoch_size, deter_size, output_shape, activation)
        features = torch.randn(1, stoch_size + deter_size)
        dist = decoder(features)
        assert dist.batch_shape == (1,)
        assert dist.event_shape == output_shape


class TestDenseDecoder:
    def test_initialization(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (10,)
        n_layers = 2
        units = 64
        activation = "relu"
        dist = "normal"
        decoder = DenseDecoder(
            stoch_size, deter_size, output_shape, n_layers, units, activation, dist
        )
        assert decoder.input_size == stoch_size + deter_size
        assert decoder.output_shape == output_shape
        assert decoder.n_layers == n_layers
        assert decoder.units == units
        assert decoder.dist == dist
        assert isinstance(decoder.model, torch.nn.Sequential)

    def test_forward_pass_normal(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (10,)
        n_layers = 2
        units = 64
        activation = "relu"
        dist = "normal"
        decoder = DenseDecoder(
            stoch_size, deter_size, output_shape, n_layers, units, activation, dist
        )
        batch_size = 2
        features = torch.randn(batch_size, stoch_size + deter_size)
        dist_out = decoder(features)
        assert dist_out.batch_shape == (batch_size,)
        assert dist_out.event_shape == output_shape
        mean = dist_out.mean
        assert mean.shape == (batch_size, *output_shape)

    def test_forward_pass_binary(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (10,)
        n_layers = 2
        units = 64
        activation = "relu"
        dist = "binary"
        decoder = DenseDecoder(
            stoch_size, deter_size, output_shape, n_layers, units, activation, dist
        )
        batch_size = 2
        features = torch.randn(batch_size, stoch_size + deter_size)
        dist_out = decoder(features)
        assert dist_out.batch_shape == (batch_size,)
        assert dist_out.event_shape == output_shape

    def test_forward_pass_none(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (10,)
        n_layers = 2
        units = 64
        activation = "relu"
        dist = "none"
        decoder = DenseDecoder(
            stoch_size, deter_size, output_shape, n_layers, units, activation, dist
        )
        batch_size = 2
        features = torch.randn(batch_size, stoch_size + deter_size)
        out = decoder(features)
        assert out.shape == (batch_size, *output_shape)

    def test_invalid_dist(self):
        stoch_size = 30
        deter_size = 200
        output_shape = (10,)
        n_layers = 2
        units = 64
        activation = "relu"
        dist = "invalid"
        decoder = DenseDecoder(
            stoch_size, deter_size, output_shape, n_layers, units, activation, dist
        )
        features = torch.randn(2, stoch_size + deter_size)
        with pytest.raises(NotImplementedError):
            decoder(features)


class TestActionDecoder:
    def test_initialization(self):
        action_size = 2
        stoch_size = 30
        deter_size = 200
        n_layers = 2
        units = 64
        activation = "relu"
        decoder = ActionDecoder(
            action_size, stoch_size, deter_size, n_layers, units, activation
        )
        assert decoder.action_size == action_size
        assert decoder.stoch_size == stoch_size
        assert decoder.deter_size == deter_size
        assert decoder.units == units
        assert decoder.n_layers == n_layers
        assert isinstance(decoder.action_model, torch.nn.Sequential)

    def test_forward_pass_stochastic(self):
        action_size = 2
        stoch_size = 30
        deter_size = 200
        n_layers = 2
        units = 64
        activation = "relu"
        decoder = ActionDecoder(
            action_size, stoch_size, deter_size, n_layers, units, activation
        )
        batch_size = 2
        features = torch.randn(batch_size, stoch_size + deter_size)
        action = decoder(features, deter=False)
        assert action.shape == (batch_size, action_size)
        assert torch.all(action >= -1) and torch.all(action <= 1)

    def test_forward_pass_deterministic(self):
        action_size = 2
        stoch_size = 30
        deter_size = 200
        n_layers = 2
        units = 64
        activation = "relu"
        decoder = ActionDecoder(
            action_size, stoch_size, deter_size, n_layers, units, activation
        )
        batch_size = 2
        features = torch.randn(batch_size, stoch_size + deter_size)
        action = decoder(features, deter=True)
        assert action.shape == (batch_size, action_size)
        assert torch.all(action >= -1) and torch.all(action <= 1)

    def test_add_exploration(self):
        action_size = 2
        stoch_size = 30
        deter_size = 200
        n_layers = 2
        units = 64
        activation = "relu"
        decoder = ActionDecoder(
            action_size, stoch_size, deter_size, n_layers, units, activation
        )
        action = torch.randn(2, action_size)
        noisy_action = decoder.add_exploration(action)
        assert noisy_action.shape == action.shape
        assert torch.all(noisy_action >= -1) and torch.all(noisy_action <= 1)

    def test_forward_pass_with_single_action(self):
        action_size = 1
        stoch_size = 30
        deter_size = 200
        n_layers = 2
        units = 64
        activation = "relu"
        decoder = ActionDecoder(
            action_size, stoch_size, deter_size, n_layers, units, activation
        )
        batch_size = 2
        features = torch.randn(batch_size, stoch_size + deter_size)
        action = decoder(features, deter=False)
        assert action.shape == (batch_size, action_size)
        assert torch.all(action >= -1) and torch.all(action <= 1)
