import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from simulusion.controller.rssm_policy import RSSMPolicy

pytest.importorskip("cv2")
from simulusion.controller.rollout_generator import RolloutGenerator
from simulusion.memory.planet_memory import Episode


class TestRSSMPolicy:
    @pytest.fixture
    def mock_rssm(self):
        rssm = Mock()
        rssm.action_size = 2
        rssm.state_size = 200
        rssm.latent_size = 30
        rssm.encoder = Mock(return_value=torch.randn(1, 1024))
        rssm.get_init_state = Mock(
            return_value=(torch.zeros(1, 200), torch.zeros(1, 30))
        )
        rssm.deterministic_state_fwd = Mock(return_value=torch.randn(1, 200))
        rssm.state_prior = Mock(
            return_value=(torch.zeros(1, 30), torch.ones(1, 30) * 0.1)
        )
        rssm.pred_reward = Mock(return_value=torch.zeros(1, 1))
        return rssm

    def test_init(self, mock_rssm):
        policy = RSSMPolicy(
            model=mock_rssm,
            planning_horizon=20,
            num_candidates=100,
            num_iterations=10,
            top_candidates=10,
            device=torch.device("cpu"),
        )

        assert policy.N == 100
        assert policy.K == 10
        assert policy.T == 10
        assert policy.H == 20
        assert policy.d == 2

    def test_reset(self, mock_rssm):
        policy = RSSMPolicy(
            model=mock_rssm,
            planning_horizon=20,
            num_candidates=100,
            num_iterations=10,
            top_candidates=10,
            device=torch.device("cpu"),
        )

        policy.reset()

        assert policy.h.shape == (1, 200)
        assert policy.s.shape == (1, 30)
        assert policy.a.shape == (1, 2)

    def test_poll(self, mock_rssm):
        policy = RSSMPolicy(
            model=mock_rssm,
            planning_horizon=5,
            num_candidates=10,
            num_iterations=2,
            top_candidates=3,
            device=torch.device("cpu"),
        )

        policy.reset()

        obs = torch.randn(3, 64, 64)
        action = policy.poll(obs, explore=False)

        assert action.shape == (1, 2)

    def test_poll_with_explore(self, mock_rssm):
        policy = RSSMPolicy(
            model=mock_rssm,
            planning_horizon=5,
            num_candidates=10,
            num_iterations=2,
            top_candidates=3,
            device=torch.device("cpu"),
        )

        policy.reset()

        obs = torch.randn(3, 64, 64)
        action = policy.poll(obs, explore=True)

        assert action.shape == (1, 2)


class TestRolloutGenerator:
    @pytest.fixture
    def mock_env(self):
        env = Mock()
        env.max_episode_steps = 10
        env.reset = Mock(return_value=torch.randn(3, 64, 64))
        env.step = Mock(return_value=(torch.randn(3, 64, 64), 1.0, False, {}))
        env.sample_random_action = Mock(return_value=torch.tensor([0.0, 0.0]))
        return env

    @pytest.fixture
    def mock_policy(self):
        policy = Mock()
        policy.reset = Mock()
        policy.poll = Mock(return_value=torch.tensor([[0.0, 0.0]]))
        policy.h = torch.zeros(1, 200)
        policy.s = torch.zeros(1, 30)
        policy.rssm = Mock()
        policy.rssm.decoder = Mock(
            return_value=Mock(
                squeeze=Mock(
                    return_value=Mock(
                        cpu=Mock(return_value=Mock(clamp_=Mock(return_value=Mock())))
                    )
                )
            )
        )
        policy.rssm.pred_reward = Mock(
            return_value=Mock(
                cpu=Mock(
                    return_value=Mock(
                        flatten=Mock(return_value=Mock(item=Mock(return_value=1.0)))
                    )
                )
            )
        )
        return policy

    def test_init(self, mock_env):
        gen = RolloutGenerator(
            env=mock_env, device=torch.device("cpu"), max_episode_steps=5
        )

        assert gen.env == mock_env
        assert gen.max_episode_steps == 5

    def test_init_default_episode(self, mock_env):
        gen = RolloutGenerator(env=mock_env, device=torch.device("cpu"))

        assert gen.episode_gen == Episode

    def test_init_with_custom_episode_gen(self, mock_env):
        gen = RolloutGenerator(
            env=mock_env, device=torch.device("cpu"), episode_gen=Episode
        )

        assert gen.episode_gen == Episode

    @patch("simulusion.controller.rollout_generator.trange")
    def test_rollout_once_random_policy(self, mock_trange, mock_env):
        mock_trange.return_value = range(5)

        gen = RolloutGenerator(
            env=mock_env, device=torch.device("cpu"), max_episode_steps=5
        )

        with patch.object(gen, "episode_gen", return_value=Mock()):
            gen.rollout_once(random_policy=True)

            mock_env.sample_random_action.assert_called()
            mock_env.reset.assert_called_once()

    @patch("simulusion.controller.rollout_generator.trange")
    def test_rollout_n(self, mock_trange, mock_env):
        mock_trange.return_value = [0, 1, 2]

        gen = RolloutGenerator(
            env=mock_env, device=torch.device("cpu"), max_episode_steps=5
        )

        gen.episode_gen = Mock(return_value=Mock())

        episodes = gen.rollout_n(n=3, random_policy=True)

        assert len(episodes) == 3

    @patch("simulusion.controller.rollout_generator.trange")
    def test_rollout_eval_n(self, mock_trange, mock_env, mock_policy):
        mock_trange.return_value = range(5)

        gen = RolloutGenerator(
            env=mock_env,
            device=torch.device("cpu"),
            policy=mock_policy,
            max_episode_steps=5,
        )

        # Four values, matching the real rollout_eval: (episode, frames,
        # metrics, latents). The mock previously returned three, so this test
        # passed while every real caller raised ValueError.
        with patch.object(
            gen,
            "rollout_eval",
            return_value=(Mock(), np.zeros((5, 3, 64, 64)), {}, None),
        ):
            episodes, frames, metrics = gen.rollout_eval_n(n=2)

            assert len(episodes) == 2
