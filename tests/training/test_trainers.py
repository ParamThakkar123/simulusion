import pytest
import torch
from unittest.mock import Mock

from simulusion.training.rl_harness import ActorCritic, PPOTrainer


class TestActorCritic:
    @pytest.fixture
    def model(self):
        obs_shape = (3, 64, 64)
        action_dim = 4
        return ActorCritic(obs_shape, action_dim)

    def test_initialization(self, model):
        assert model.obs_shape == (3, 64, 64)
        assert model.action_dim == 4
        assert isinstance(model.cnn, torch.nn.Sequential)
        assert isinstance(model.actor, torch.nn.Sequential)
        assert isinstance(model.critic, torch.nn.Sequential)

    def test_forward(self, model):
        obs = torch.randn(2, 3, 64, 64)
        logits, value = model(obs)
        assert logits.shape == (2, 4)
        assert value.shape == (2, 1)

    def test_get_action(self, model):
        obs = torch.randn(1, 3, 64, 64)
        action, log_prob, value = model.get_action(obs)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)
        assert action.dtype == torch.long


class TestPPOTrainer:
    @pytest.fixture
    def mock_vec_env(self):
        env = Mock()
        env.total_envs = 2
        env.action_space.n = 4
        env.observation_space = {"image": Mock(shape=(3, 64, 64))}
        env.reset_batch.return_value = {"obs": {"image": torch.randn(2, 3, 64, 64)}}
        env.step_batch.return_value = {
            "obs": {"image": torch.randn(2, 3, 64, 64)},
            "reward": torch.randn(2),
            "done": torch.zeros(2, dtype=torch.bool),
        }
        return env

    @pytest.fixture
    def trainer(self, mock_vec_env):
        return PPOTrainer(mock_vec_env, device="cpu")

    def test_initialization(self, trainer, mock_vec_env):
        assert trainer.vec_env == mock_vec_env
        assert trainer.device == "cpu"
        assert isinstance(trainer.policy, ActorCritic)
        assert trainer.policy.obs_shape == (3, 64, 64)
        assert trainer.policy.action_dim == 4

    def test_compute_gae(self, trainer):
        rewards = torch.randn(10, 2)
        values = torch.randn(10, 2)
        dones = torch.zeros(10, 2, dtype=torch.bool)
        advantages = trainer.compute_gae(rewards, values, dones)
        assert advantages.shape == (10, 2)
