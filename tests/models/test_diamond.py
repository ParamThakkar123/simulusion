import importlib.util

import pytest

if (
    importlib.util.find_spec("gymnasium") is None
    and importlib.util.find_spec("gym") is None
):
    pytest.skip("gym/gymnasium is not installed", allow_module_level=True)

import torch
import numpy as np
from simulusion.utils.gym_compat import gym
from unittest.mock import MagicMock, patch

from simulusion.configs.diamond_config import (
    DiamondConfig,
    ATARI_100K_GAMES,
    HUMAN_SCORES,
    RANDOM_SCORES,
)
from simulusion.envs.diamond_atari import DiamondAtariWrapper, make_diamond_atari_env
from simulusion.datasets.diamond_dataset import ReplayBuffer
from simulusion.models.diffusion.diamond_diffusion import (
    DiffusionUNet,
    EDMPreconditioner,
    EulerSampler,
    AdaptiveGroupNorm,
    TimestepEmbedding,
)
from simulusion.models.diffusion.reward_termination import (
    RewardTerminationModel,
    RewardTerminationLoss,
)
from simulusion.models.diffusion.actor_critic import (
    ActorCriticNetwork,
    RLLoss,
)


class TestDiamondConfig:
    def test_default_config(self):
        config = DiamondConfig()
        assert config.game == "Breakout-v5"
        assert config.obs_size == 64
        assert config.frameskip == 4
        assert config.max_noop == 30
        assert config.num_conditioning_frames == 4
        assert config.imagination_horizon == 15
        assert config.num_sampling_steps == 3

    def test_custom_config(self):
        config = DiamondConfig(
            game="Pong-v5",
            obs_size=32,
            learning_rate=1e-3,
            num_sampling_steps=5,
        )
        assert config.game == "Pong-v5"
        assert config.obs_size == 32
        assert config.learning_rate == 1e-3
        assert config.num_sampling_steps == 5

    def test_atari_100k_games_list(self):
        assert len(ATARI_100K_GAMES) == 26
        assert "Breakout-v5" in ATARI_100K_GAMES
        assert "Pong-v5" in ATARI_100K_GAMES
        assert "Asterix-v5" in ATARI_100K_GAMES

    def test_human_random_scores(self):
        assert len(HUMAN_SCORES) == 26
        assert len(RANDOM_SCORES) == 26
        for game in ATARI_100K_GAMES:
            assert game in HUMAN_SCORES
            assert game in RANDOM_SCORES
            assert HUMAN_SCORES[game] > RANDOM_SCORES[game]


class TestDiamondAtariWrapper:
    @pytest.fixture(autouse=True)
    def _require_cv2(self):
        pytest.importorskip("cv2")

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.__class__ = gym.Env
        env.action_space.n = 18
        env.observation_space.shape = (210, 160, 3)
        env.reset.return_value = (np.zeros((210, 160, 3), dtype=np.uint8), {})
        env.step.return_value = (
            np.zeros((210, 160, 3), dtype=np.uint8),
            1.0,
            False,
            False,
            {},
        )
        env.ale.lives.return_value = 0
        return env

    def test_wrapper_creation(self, mock_env):
        wrapper = DiamondAtariWrapper(
            env=mock_env,
            frameskip=4,
            max_noop=30,
            resize=(64, 64),
        )
        assert wrapper.frameskip == 4
        assert wrapper.max_noop == 30
        assert wrapper.observation_space.shape == (64, 64, 3)

    def test_wrapper_step(self, mock_env):
        wrapper = DiamondAtariWrapper(
            env=mock_env,
            frameskip=2,
            resize=(64, 64),
        )
        obs, reward, done, info = wrapper.step(1)
        assert obs.shape == (64, 64, 3)
        assert mock_env.step.call_count == 2

    def test_wrapper_reset(self, mock_env):
        wrapper = DiamondAtariWrapper(
            env=mock_env,
            resize=(64, 64),
        )
        obs, info = wrapper.reset()
        assert obs.shape == (64, 64, 3)

    def test_reward_clipping(self, mock_env):
        mock_env.step.return_value = (
            np.zeros((210, 160, 3), dtype=np.uint8),
            10.0,
            False,
            False,
            {},
        )
        wrapper = DiamondAtariWrapper(
            env=mock_env,
            reward_clip=True,
            resize=(64, 64),
        )
        _, reward, _, _ = wrapper.step(1)
        assert reward == 1.0

    def test_make_diamond_atari_env(self):
        # `make_diamond_atari_env` imports `gym` locally via
        # `from simulusion.utils.gym_compat import gym`, so patch `gym.make`
        # where it is actually looked up rather than on the env module (which
        # has no module-level `gym` attribute).
        with patch("simulusion.utils.gym_compat.gym.make") as mock_make:
            mock_make.return_value = MagicMock()
            mock_make.return_value.__class__ = gym.Env
            env = make_diamond_atari_env("Breakout-v5", seed=42)
            assert env is not None


class TestReplayBuffer:
    @pytest.fixture
    def buffer(self):
        return ReplayBuffer(
            capacity=1000,
            obs_shape=(64, 64, 3),
            action_dim=1,
            device="cpu",
        )

    def test_buffer_initialization(self, buffer):
        assert buffer.capacity == 1000
        assert buffer.size == 0

    def test_buffer_add(self, buffer):
        obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        next_obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        buffer.add(obs=obs, action=0, reward=1.0, done=False, next_obs=next_obs)

        assert buffer.size == 1

    def test_buffer_sample(self, buffer):
        for i in range(32):
            obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            next_obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            buffer.add(
                obs=obs, action=i % 5, reward=float(i), done=False, next_obs=next_obs
            )

        batch = buffer.sample(batch_size=16)

        assert batch["obs"].shape == (16, 3, 64, 64)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)

    def test_buffer_is_ready(self, buffer):
        assert not buffer.is_ready(10)
        for i in range(15):
            obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            next_obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            buffer.add(obs=obs, action=0, reward=0.0, done=False, next_obs=next_obs)
        assert buffer.is_ready(10)


class TestEDMPreconditioner:
    def test_preconditioner_creation(self):
        precond = EDMPreconditioner(sigma_data=0.5, p_mean=-0.4, p_std=1.2)
        assert precond.sigma_data == 0.5
        assert precond.p_mean == -0.4
        assert precond.p_std == 1.2

    def test_get_preconditioners(self):
        precond = EDMPreconditioner()
        sigma = torch.tensor([0.1, 0.5, 1.0])
        result = precond.get_preconditioners(sigma)

        assert "c_skip" in result
        assert "c_out" in result
        assert "c_in" in result
        assert "c_noise" in result

    def test_sample_noise_level(self):
        precond = EDMPreconditioner()
        sigma = precond.sample_noise_level(32, torch.device("cpu"))
        assert sigma.shape == (32,)
        assert (sigma > 0).all()


class TestEulerSampler:
    def test_sampler_creation(self):
        sampler = EulerSampler(
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7,
            num_steps=3,
        )
        assert sampler.num_steps == 3
        assert sampler.t_steps.shape == (3,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sampler_sample(self):
        sampler = EulerSampler(num_steps=3)

        model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=4,
            base_channels=32,
            channel_multipliers=(1, 1, 1),
            num_res_blocks=1,
            cond_dim=128,
            action_dim=18,
        ).to("cuda")

        shape = (2, 3, 64, 64)
        obs_history = torch.randn(2, 4, 3, 64, 64).to("cuda")
        actions = torch.randint(0, 18, (2, 4)).to("cuda")

        with torch.no_grad():
            result = sampler.sample(
                model=model,
                shape=shape,
                device=torch.device("cuda"),
                obs_history=obs_history,
                actions=actions,
            )

        assert result.shape == shape
        # Frames live in [-1, 1]: that is the domain EDM's sigma_data = 0.5
        # (Appendix C) is calibrated for.
        assert (result >= -1).all() and (result <= 1).all()


class TestDiffusionUNet:
    def test_unet_creation(self):
        model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=4,
            base_channels=32,
            channel_multipliers=(1, 1, 1, 1),
            num_res_blocks=2,
            cond_dim=256,
            action_dim=18,
        )
        assert model is not None

    def test_unet_forward(self):
        model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=4,
            base_channels=32,
            channel_multipliers=(1, 1, 1),
            num_res_blocks=1,
            cond_dim=128,
            action_dim=18,
        )

        B = 2
        x = torch.randn(B, 3, 64, 64)
        t = torch.randn(B)
        obs_history = torch.randn(B, 4, 3, 64, 64)
        actions = torch.randint(0, 18, (B, 4))

        output = model(x, t, obs_history, actions)

        assert output.shape == (B, 3, 64, 64)


class TestAdaptiveGroupNorm:
    def test_adaptive_group_norm(self):
        norm = AdaptiveGroupNorm(num_groups=8, num_channels=32, cond_dim=64)

        x = torch.randn(2, 32, 16, 16)
        cond = torch.randn(2, 64)

        output = norm(x, cond)

        assert output.shape == x.shape


class TestTimestepEmbedding:
    def test_timestep_embedding(self):
        embed = TimestepEmbedding(dim=128)

        t = torch.tensor([0.0, 0.5, 1.0])
        output = embed(t)

        assert output.shape == (3, 128)

    def test_timestep_embedding_single(self):
        embed = TimestepEmbedding(dim=128)

        t = torch.tensor([0.5])
        output = embed(t)

        assert output.shape == (1, 128)


class TestRewardTerminationModel:
    def test_reward_model_creation(self):
        model = RewardTerminationModel(
            obs_channels=3,
            action_dim=18,
            channels=(32, 32, 32, 32),
            lstm_dim=256,
            cond_dim=128,
        )
        assert model is not None

    def test_reward_model_forward(self):
        model = RewardTerminationModel(
            obs_channels=3,
            action_dim=18,
            channels=(16, 16),
            lstm_dim=128,
            cond_dim=64,
        )

        B, T = 2, 5
        obs = torch.randn(B, T, 3, 64, 64)
        actions = torch.randint(0, 18, (B, T))

        reward_logits, term_logits, hidden = model(obs, actions)

        assert reward_logits.shape == (B, T, 3)
        assert term_logits.shape == (B, T, 2)
        assert hidden[0].shape == (1, B, 128)

    def test_reward_model_predict(self):
        model = RewardTerminationModel(
            obs_channels=3,
            action_dim=18,
            lstm_dim=128,
        )

        obs = torch.randn(2, 3, 64, 64)
        actions = torch.randint(0, 18, (2,))

        reward, terminated, hidden = model.predict(obs, actions)

        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert hidden[0].shape == (1, 2, 128)

    def test_reward_model_init_hidden(self):
        model = RewardTerminationModel(lstm_dim=256)
        h, c = model.init_hidden(4, torch.device("cpu"))

        assert h.shape == (1, 4, 256)
        assert c.shape == (1, 4, 256)


class TestRewardTerminationLoss:
    def test_loss_creation(self):
        loss_fn = RewardTerminationLoss()
        assert loss_fn is not None

    def test_loss_forward(self):
        loss_fn = RewardTerminationLoss()

        B, T = 2, 5
        reward_logits = torch.randn(B, T, 3)
        term_logits = torch.randn(B, T, 2)
        rewards = torch.tensor([[0, 1, -1, 0, 1], [1, 0, 0, -1, 1]])
        terminated = torch.tensor(
            [[False, True, False, False, True], [False, False, True, False, False]]
        )

        total_loss, reward_loss, term_loss = loss_fn(
            reward_logits, term_logits, rewards, terminated
        )

        assert total_loss.numel() == 1
        assert reward_loss.numel() == 1
        assert term_loss.numel() == 1


class TestActorCriticNetwork:
    def test_actor_critic_creation(self):
        model = ActorCriticNetwork(
            obs_channels=3,
            action_dim=18,
            channels=(32, 32, 64, 64),
            lstm_dim=512,
        )
        assert model is not None

    def test_actor_critic_forward(self):
        model = ActorCriticNetwork(
            obs_channels=3,
            action_dim=18,
            channels=(16, 16, 32),
            lstm_dim=128,
        )

        B, T = 2, 5
        obs = torch.randn(B, T, 3, 64, 64)

        policy_logits, values, hidden = model(obs)

        assert policy_logits.shape == (B, T, 18)
        assert values.shape == (B, T, 1)
        assert hidden[0].shape == (1, B, 128)

    def test_actor_critic_get_action(self):
        model = ActorCriticNetwork(
            obs_channels=3,
            action_dim=4,
            lstm_dim=64,
        )

        obs = torch.randn(1, 3, 64, 64)

        action, hidden = model.get_action(obs, deterministic=False)

        assert isinstance(action, int)
        assert hidden[0].shape == (1, 1, 64)

    def test_actor_critic_init_hidden(self):
        model = ActorCriticNetwork(lstm_dim=256)
        h, c = model.init_hidden(4, torch.device("cpu"))

        assert h.shape == (1, 4, 256)
        assert c.shape == (1, 4, 256)


class TestRLLoss:
    def test_rl_loss_creation(self):
        loss_fn = RLLoss(
            discount_factor=0.985,
            lambda_returns=0.95,
            entropy_weight=0.001,
        )
        assert loss_fn.discount_factor == 0.985
        assert loss_fn.lambda_returns == 0.95

    def test_compute_lambda_returns(self):
        loss_fn = RLLoss(discount_factor=0.9, lambda_returns=0.5)

        B, T = 2, 5
        rewards = torch.randn(B, T)
        values = torch.randn(B, T + 1)
        dones = torch.zeros(B, T, dtype=torch.bool)

        lambda_returns = loss_fn.compute_lambda_returns(rewards, values, dones)

        assert lambda_returns.shape == (B, T)

    def test_policy_loss(self):
        loss_fn = RLLoss(entropy_weight=0.01)

        B, T, A = 2, 5, 4
        policy_logits = torch.randn(B, T, A)
        actions = torch.randint(0, A, (B, T))
        lambda_returns = torch.randn(B, T)
        values = torch.randn(B, T + 1)

        loss = loss_fn.policy_loss(policy_logits, actions, lambda_returns, values)

        assert loss.numel() == 1

    def test_value_loss(self):
        loss_fn = RLLoss()

        B, T = 2, 5
        values = torch.randn(B, T + 1)
        lambda_returns = torch.randn(B, T)

        loss = loss_fn.value_loss(values, lambda_returns)

        assert loss.numel() == 1


@pytest.mark.integration
class TestIntegration:
    def test_config_human_normalized_score(self):
        human = HUMAN_SCORES["Breakout-v5"]
        random = RANDOM_SCORES["Breakout-v5"]

        hns_random = (random - random) / (human - random)
        hns_human = (human - random) / (human - random)
        hns_super = (human * 2 - random) / (human - random)

        assert hns_random == 0.0
        assert hns_human == 1.0
        assert hns_super == pytest.approx(2.0, abs=0.1)

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_forward_pass(self):
        config = DiamondConfig(obs_size=32)

        diffusion_model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=config.num_conditioning_frames,
            base_channels=32,
            channel_multipliers=(1, 1),
            num_res_blocks=1,
            cond_dim=64,
            action_dim=18,
        ).to("cuda")

        # The conv trunk is flattened into the LSTM (Appendix D), so the frame
        # size determines the cell's input width and has to be declared.
        reward_model = RewardTerminationModel(
            obs_channels=3,
            action_dim=18,
            lstm_dim=64,
            cond_dim=32,
            frame_size=config.obs_size,
        ).to("cuda")

        actor_critic = ActorCriticNetwork(
            obs_channels=3,
            action_dim=18,
            lstm_dim=64,
            frame_size=config.obs_size,
        ).to("cuda")

        B = 2
        obs_history = torch.randn(B, 4, 3, 32, 32).to("cuda")
        actions = torch.randint(0, 18, (B, 4)).to("cuda")

        obs_seq = torch.randn(B, 5, 3, 32, 32).to("cuda")
        action_seq = torch.randint(0, 18, (B, 5)).to("cuda")

        reward_logits, term_logits, _ = reward_model(obs_seq, action_seq)
        assert reward_logits.shape == (B, 5, 3)

        policy_logits, values, _ = actor_critic(obs_seq)
        assert policy_logits.shape == (B, 5, 18)

        sampler = EulerSampler(num_steps=3)
        with torch.no_grad():
            next_obs = sampler.sample(
                model=diffusion_model,
                shape=(B, 3, 32, 32),
                device=torch.device("cuda"),
                obs_history=obs_history,
                actions=actions,
            )
        assert next_obs.shape == (B, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
