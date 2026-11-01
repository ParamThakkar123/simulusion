import pytest
import torch
from unittest.mock import Mock, patch

from simulusion.models.planet import Planet


class TestPlanet:
    @pytest.fixture
    def config(self):
        return {
            "env": "CartPole-v1",
            "bit_depth": 5,
            "state_size": 200,
            "latent_size": 30,
            "embedding_size": 1024,
            "memory_size": 10,
            "device": torch.device("cpu"),
        }

    @patch("simulusion.models.planet.TorchImageEnvWrapper")
    def test_initialization_with_string_env(self, mock_wrapper, config):
        mock_env = Mock()
        mock_env.action_size = 2
        mock_wrapper.return_value = mock_env

        planet = Planet(**config)

        assert planet.env == mock_env
        assert planet.device == config["device"]
        assert planet.bit_depth == config["bit_depth"]
        assert isinstance(planet.rssm, torch.nn.Module)
        assert planet.memory is not None

    @patch("simulusion.models.planet.TorchImageEnvWrapper")
    def test_initialization_with_custom_env(self, mock_wrapper, config):
        mock_env = Mock()
        mock_env.action_size = 2

        config["env"] = mock_env
        planet = Planet(**config)

        # Should not call wrapper since env has action_size
        mock_wrapper.assert_not_called()
        assert planet.env == mock_env

    @patch("simulusion.models.planet.TorchImageEnvWrapper")
    def test_warmup(self, mock_wrapper, config):
        mock_env = Mock()
        mock_env.action_size = 2
        mock_wrapper.return_value = mock_env

        planet = Planet(**config)
        planet.rollout_gen = Mock()
        planet.memory = Mock()

        planet.warmup(n_episodes=2, random_policy=True)

        planet.rollout_gen.rollout_n.assert_called_once_with(n=2, random_policy=True)
        planet.memory.append.assert_called_once()
