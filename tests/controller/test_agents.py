import pytest
import torch
from unittest.mock import Mock

from simulusion.controller.rssm_policy import RSSMPolicy


class TestRSSMPolicy:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.action_size = 2
        model.state_size = 30
        model.latent_size = 10
        return model

    @pytest.fixture
    def policy(self, mock_model):
        return RSSMPolicy(
            model=mock_model,
            planning_horizon=5,
            num_candidates=100,
            num_iterations=3,
            top_candidates=10,
            device=torch.device("cpu"),
        )

    def test_initialization(self, policy, mock_model):
        assert policy.rssm == mock_model
        assert policy.N == 100
        assert policy.K == 10
        assert policy.T == 3
        assert policy.H == 5
        assert policy.d == 2
        assert policy.state_size == 30
        assert policy.latent_size == 10

    def test_reset(self, policy):
        policy.reset()
        assert policy.h.shape == (1, 30)
        assert policy.s.shape == (1, 10)
        assert policy.a.shape == (1, 2)
        assert (policy.h == 0).all()
        assert (policy.s == 0).all()
        assert (policy.a == 0).all()
