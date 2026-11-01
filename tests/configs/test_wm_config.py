"""Tests for World Models configuration classes."""

import pytest
from simulusion.configs.wm_config import (
    WMVAEConfig,
    WMMDNRNNConfig,
    WMControllerConfig,
)


class TestWMVAEConfig:
    def test_default_creation(self):
        config = WMVAEConfig(height=64, width=64, latent_size=32)
        assert config.height == 64
        assert config.width == 64
        assert config.latent_size == 32
        assert config.train_batch_size == 32

    def test_custom_values(self):
        config = WMVAEConfig(
            height=128,
            width=128,
            latent_size=64,
            train_batch_size=64,
            num_epochs=50,
            learning_rate=1e-4,
            logdir="custom_logs",
        )
        assert config.height == 128
        assert config.width == 128
        assert config.latent_size == 64
        assert config.train_batch_size == 64
        assert config.num_epochs == 50
        assert config.learning_rate == 1e-4
        assert config.logdir == "custom_logs"

    def test_to_dict(self):
        config = WMVAEConfig(height=64, width=64, latent_size=32)
        d = config.to_dict()
        assert d["height"] == 64
        assert d["latent_size"] == 32

    def test_validation_passes(self):
        config = WMVAEConfig(height=64, width=64, latent_size=32)
        assert config.validate() is True

    def test_validation_fails(self):
        with pytest.raises(Exception):
            WMVAEConfig(height=0, width=0, latent_size=0)


class TestWMMDNRNNConfig:
    def test_default_creation(self):
        config = WMMDNRNNConfig()
        assert config.latent_size == 32
        assert config.action_size == 3
        assert config.hidden_size == 256
        assert config.gmm_components == 5

    def test_custom_values(self):
        config = WMMDNRNNConfig(
            latent_size=64,
            action_size=5,
            hidden_size=512,
            gmm_components=10,
            batch_size=32,
            num_epochs=50,
        )
        assert config.latent_size == 64
        assert config.action_size == 5
        assert config.hidden_size == 512
        assert config.gmm_components == 10
        assert config.batch_size == 32
        assert config.num_epochs == 50

    def test_include_reward_default(self):
        config = WMMDNRNNConfig()
        assert config.include_reward is True

    def test_to_dict(self):
        config = WMMDNRNNConfig(latent_size=32, action_size=3)
        d = config.to_dict()
        assert d["latent_size"] == 32
        assert d["action_size"] == 3


class TestWMControllerConfig:
    def test_default_creation(self):
        config = WMControllerConfig()
        assert config.latent_size == 32
        assert config.hidden_size == 200
        assert config.action_size == 3
        assert config.env_name == "CarRacing-v3"

    def test_custom_values(self):
        config = WMControllerConfig(
            latent_size=64,
            hidden_size=512,
            action_size=5,
            env_name="Pendulum-v1",
            pop_size=20,
            n_samples=8,
        )
        assert config.latent_size == 64
        assert config.hidden_size == 512
        assert config.action_size == 5
        assert config.env_name == "Pendulum-v1"
        assert config.pop_size == 20
        assert config.n_samples == 8

    def test_to_dict(self):
        config = WMControllerConfig(latent_size=32, action_size=3)
        d = config.to_dict()
        assert d["latent_size"] == 32
