"""Conformance tests for DIAMOND (Alonso et al., NeurIPS 2024).

Each test pins a specific claim of the paper -- Algorithm 1's imagination
ordering, the diffusion target's one-step-ahead definition (eq. 5), the
Appendix B DDPM formulation, and the Table 3 environment settings -- so a
regression shows up as a named failure rather than as a slightly worse score.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from simulusion.datasets.diamond_dataset import (
    ReplayBuffer,
    SequenceDataset,
    to_model_domain,
)
from simulusion.models.diffusion.DDPM import DDPM
from simulusion.models.diffusion.actor_critic import ActorCriticNetwork
from simulusion.models.diffusion.diamond_diffusion import TimestepEmbedding
from simulusion.models.diffusion.reward_termination import RewardTerminationModel
from simulusion.envs.diamond_atari import DiamondAtariWrapper


class TestDiffusionTargetIsOneStepAhead:
    """Eq. 5: D_theta(x^tau_{t+1}, tau, x^0_{<=t}, a_{<=t}) predicts x^0_{t+1}."""

    @staticmethod
    def _buffer(n: int = 30) -> ReplayBuffer:
        """A buffer where frame i is a constant image of value i."""
        buffer = ReplayBuffer(capacity=n * 2, obs_shape=(4, 4, 3))
        for i in range(n):
            buffer.add(
                obs=np.full((4, 4, 3), i, dtype=np.uint8),
                action=i % 3,
                reward=0.0,
                done=False,
                next_obs=np.full((4, 4, 3), i + 1, dtype=np.uint8),
            )
        return buffer

    @staticmethod
    def _frame_id(frame: torch.Tensor) -> int:
        """Recover the constant pixel value a frame was built with.

        Frames reach the model in [-1, 1] (Appendix C's sigma_data = 0.5 assumes
        centred data), so undo that mapping rather than assuming [0, 1].
        """
        return int(round((float(frame.flatten()[0]) + 1.0) * 127.5))

    def test_target_immediately_follows_the_conditioning_window(self):
        dataset = SequenceDataset(self._buffer(), sequence_length=5, burn_in=4)
        item = dataset[0]

        conditioning = [self._frame_id(f) for f in item["obs_seq"]]
        assert conditioning == [0, 1, 2, 3, 4]
        # Not 6: a two-step-ahead target makes the conditioning actions describe
        # a transition the model is not being asked to predict.
        assert self._frame_id(item["next_obs"]) == 5

    def test_last_index_stays_in_bounds(self):
        buffer = self._buffer()
        dataset = SequenceDataset(buffer, sequence_length=5, burn_in=4)
        last = dataset[len(dataset) - 1]
        assert (
            self._frame_id(last["next_obs"]) == self._frame_id(last["obs_seq"][-1]) + 1
        )


class TestObservationDomain:
    """Appendix C: sigma_data = 0.5 is the data's standard deviation."""

    def test_frames_are_centred_on_zero(self):
        pixels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
        model_domain = to_model_domain(pixels.clone())
        assert model_domain.min() == pytest.approx(-1.0, abs=1e-5)
        assert model_domain.max() == pytest.approx(1.0, abs=1e-2)
        # Centred: a [0, 1] scaling would put the mean at ~0.5, and every
        # preconditioner in eqs. 9-12 assumes zero-mean data.
        assert abs(float(model_domain.mean())) < 1e-2

    def test_dataset_and_buffer_agree_on_the_domain(self):
        buffer = TestDiffusionTargetIsOneStepAhead._buffer(n=12)
        dataset = SequenceDataset(buffer, sequence_length=4, burn_in=3)
        from_dataset = dataset[0]["obs_seq"][0]
        from_buffer = buffer.sample(batch_size=1)["obs"]
        for tensor in (from_dataset, from_buffer):
            assert tensor.min() >= -1.0 and tensor.max() <= 1.0


class TestEpisodeBoundaries:
    """Sequences must not splice two episodes together."""

    @staticmethod
    def _buffer_with_terminal_at(step: int, n: int = 20) -> ReplayBuffer:
        buffer = ReplayBuffer(capacity=n * 2, obs_shape=(4, 4, 3))
        for i in range(n):
            buffer.add(
                obs=np.full((4, 4, 3), i, dtype=np.uint8),
                action=0,
                reward=0.0,
                done=(i == step),
                next_obs=np.full((4, 4, 3), i + 1, dtype=np.uint8),
            )
        return buffer

    def test_no_window_spans_a_terminal(self):
        length = 5
        buffer = self._buffer_with_terminal_at(10)
        dataset = SequenceDataset(buffer, sequence_length=length, burn_in=4)
        assert len(dataset) > 0
        for start in dataset._starts:
            # The terminal may only be the window's final transition.
            interior = buffer.dones[start : start + length - 1]
            assert not interior.any(), f"window at {start} crosses a terminal"

    def test_truncations_also_block_a_window(self):
        length = 4
        n = 16
        buffer = ReplayBuffer(capacity=n * 2, obs_shape=(4, 4, 3))
        for i in range(n):
            buffer.add(
                obs=np.full((4, 4, 3), i, dtype=np.uint8),
                action=0,
                reward=0.0,
                done=False,
                next_obs=np.full((4, 4, 3), i + 1, dtype=np.uint8),
                truncated=(i == 8),
            )
        dataset = SequenceDataset(buffer, sequence_length=length, burn_in=2)
        # A truncation is not a terminal -- it must not become a label for the
        # termination head -- but the environment still reset, so no sequence
        # may span it.
        assert not buffer.dones.any()
        for start in dataset._starts:
            assert not buffer.truncations[start : start + length - 1].any()


class TestConvTrunkFeedsTheLSTM:
    """Appendix D: the convolutional trunk feeds the LSTM cell."""

    def test_actor_critic_keeps_spatial_features(self):
        model = ActorCriticNetwork(channels=(16, 16, 32, 32), frame_size=64)
        # 64 -> 4x4 after four 2x2 pools; 32 channels x 16 positions.
        assert model.feature_size == 32 * 4 * 4
        assert model.lstm.input_size == model.feature_size

    def test_reward_model_keeps_spatial_features(self):
        model = RewardTerminationModel(channels=(32, 32, 32, 32), frame_size=64)
        assert model.feature_size == 32 * 4 * 4
        assert model.lstm.input_size == model.feature_size

    def test_frame_size_changes_the_feature_width(self):
        small = ActorCriticNetwork(channels=(16, 16, 32, 32), frame_size=32)
        assert small.feature_size == 32 * 2 * 2
        logits, values, _ = small(torch.randn(2, 3, 3, 32, 32))
        assert logits.shape[:2] == (2, 3)
        assert values.shape == (2, 3, 1)

    def test_position_changes_the_policy_output(self):
        """A spatially pooled trunk cannot tell these two frames apart."""
        torch.manual_seed(0)
        model = ActorCriticNetwork(channels=(8, 8, 16, 16), frame_size=64).eval()
        left = torch.zeros(1, 1, 3, 64, 64)
        left[..., 8, 8] = 1.0
        right = torch.zeros(1, 1, 3, 64, 64)
        right[..., 56, 56] = 1.0
        with torch.no_grad():
            a, _, _ = model(left)
            b, _, _ = model(right)
        assert not torch.allclose(a, b, atol=1e-6)


class TestNoiseLevelEmbedding:
    """c_noise = log(sigma)/4 (eq. 11) must reach the network as usable features."""

    def test_output_shape(self):
        embed = TimestepEmbedding(dim=128)
        assert embed(torch.tensor([0.0, 0.5, 1.0])).shape == (3, 128)
        assert embed(torch.tensor([0.5])).shape == (1, 128)

    def test_expansion_is_sinusoidal_and_multi_frequency(self):
        embed = TimestepEmbedding(dim=64, freq_dim=32)
        features = embed._sinusoidal(torch.tensor([0.7]))
        assert features.shape == (1, 32)
        assert features.abs().max() <= 1.0 + 1e-6
        # A Linear(1, n) expansion produces n features that are all exact
        # multiples of the same scalar; a Fourier expansion does not.
        assert features.unique().numel() > 2

    def test_handles_negative_noise_levels(self):
        """log(sigma)/4 is negative for every sigma below 1."""
        embed = TimestepEmbedding(dim=32)
        out = embed(torch.tensor([-3.0, -0.5, 0.0, 2.0]))
        assert torch.isfinite(out).all()
        assert not torch.allclose(out[0], out[3])


class TestDDPMSchedule:
    """Appendix B: the discrete variance-preserving Markov chain of Ho et al."""

    def test_constructs_and_registers_buffers(self):
        ddpm = DDPM(timesteps=20, beta_start=1e-4, beta_end=0.02)
        names = {name for name, _ in ddpm.named_buffers()}
        assert {
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        } <= names

    def test_schedule_terms_are_consistent(self):
        ddpm = DDPM(timesteps=20, beta_start=1e-4, beta_end=0.02)
        alphas = 1.0 - ddpm.betas
        assert torch.allclose(ddpm.alphas, alphas)
        assert torch.allclose(ddpm.alphas_cumprod, torch.cumprod(alphas, dim=0))
        assert ddpm.alphas_cumprod_prev[0] == pytest.approx(1.0)
        assert torch.allclose(ddpm.alphas_cumprod_prev[1:], ddpm.alphas_cumprod[:-1])

    def test_q_sample_interpolates_signal_and_noise(self):
        ddpm = DDPM(timesteps=20, beta_start=1e-4, beta_end=0.02)
        x0 = torch.ones(2, 3, 8, 8)
        noise = torch.zeros_like(x0)
        t = torch.tensor([0, 19])
        noised = ddpm.q_sample(x0, t, noise)
        # With zero noise, q_sample is a pure rescaling by sqrt(alpha_bar_t),
        # which decays as t grows.
        assert noised[0].mean() > noised[1].mean()

    def test_buffers_follow_a_dtype_cast(self):
        ddpm = DDPM(timesteps=5, beta_start=1e-4, beta_end=0.02).to(torch.float64)
        assert ddpm.betas.dtype == torch.float64

    def test_reverse_sampling_runs(self):
        class Eps(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, x, t):  # noqa: ANN001 - test double
                return self.conv(x)

        ddpm = DDPM(timesteps=5, beta_start=1e-4, beta_end=0.02)
        out = ddpm.sample(Eps(), n=2, img_size=8, channels=3)
        assert out.shape == (2, 3, 8, 8)
        assert torch.isfinite(out).all()


class _NoopCountingEnv:
    """Minimal env that records which actions it was stepped with."""

    class _Space:
        n = 4

        def sample(self) -> int:
            return 3  # never NOOP, so a broken reset issues no no-ops at all

    def __init__(self) -> None:
        self.action_space = self._Space()
        self.stepped_actions: list[int] = []

    def reset(self, **kwargs):  # noqa: ANN003
        self.stepped_actions.clear()
        return np.zeros((64, 64, 3), dtype=np.uint8), {}

    def step(self, action):  # noqa: ANN001
        self.stepped_actions.append(int(action))
        return np.zeros((64, 64, 3), dtype=np.uint8), 0.0, False, False, {}


class TestNoopStarts:
    """Table 3: "Max noop 30" -- reset issues a random number of NOOP steps."""

    def test_reset_steps_only_noops(self):
        env = _NoopCountingEnv()
        wrapper = DiamondAtariWrapper(
            env, max_noop=30, terminate_on_life_loss=False, resize=None, seed=0
        )
        wrapper.reset()
        assert env.stepped_actions, "reset performed no no-op steps at all"
        assert set(env.stepped_actions) == {0}
        assert 1 <= len(env.stepped_actions) <= 30

    def test_noop_count_varies_across_resets(self):
        env = _NoopCountingEnv()
        wrapper = DiamondAtariWrapper(
            env, max_noop=30, terminate_on_life_loss=False, resize=None, seed=0
        )
        counts = set()
        for _ in range(20):
            wrapper.reset()
            counts.add(len(env.stepped_actions))
        assert len(counts) > 1, "no-op start is not randomised"


@pytest.mark.integration
class TestImaginationOrdering:
    """Algorithm 1: a_i ~ pi(.|x_i); r_i, d_i ~ R(x_i, a_i); x_{i+1} ~ D(x_<=i, a_<=i)."""

    @staticmethod
    def _agent():
        from simulusion.configs.diamond_config import DiamondConfig
        from simulusion.training.train_diamond import DiamondAgent

        gym = pytest.importorskip("gymnasium")
        del gym
        config = DiamondConfig(
            game="Breakout-v5",
            device="cpu",
            obs_size=32,
            preset="small",
            num_conditioning_frames=4,
            burn_in_length=4,
            imagination_horizon=3,
            num_sampling_steps=1,
            use_amp=False,
        )
        try:
            return DiamondAgent(config)
        except Exception as exc:  # noqa: BLE001 - ALE ROMs are an optional extra
            pytest.skip(f"DIAMOND Atari environment unavailable: {exc}")

    def test_trajectory_has_one_more_frame_than_actions(self):
        agent = self._agent()
        B, L = 2, agent.config.burn_in_length
        H = agent.config.imagination_horizon
        size = agent.config.obs_size

        obs_history = torch.rand(B, L, 3, size, size)
        action_history = torch.randint(0, agent.action_dim, (B, L))
        hidden = agent.reward_model.init_hidden(B, agent.device)

        obs, rewards, dones, actions, _ = agent._imagine_trajectory(
            obs_history, action_history, hidden
        )

        # x_0..x_H, so V(x_H) exists to bootstrap the lambda-return (eq. 14).
        assert obs.shape[:2] == (B, H + 1)
        assert rewards.shape == (B, H)
        assert dones.shape == (B, H)
        assert actions.shape == (B, H)
        # Imagination starts *at* the last conditioning frame, not one step past
        # it, so r_0 belongs to the action the policy chose there.
        assert torch.equal(obs[:, 0], obs_history[:, -1])

    def test_reward_model_sees_the_same_pairing_it_is_trained_on(self):
        """R_psi is trained on (x_i, a_i) -> r_i; imagination must match."""
        agent = self._agent()
        B, L = 2, agent.config.burn_in_length
        size = agent.config.obs_size

        obs_history = torch.rand(B, L, 3, size, size)
        action_history = torch.randint(0, agent.action_dim, (B, L))
        hidden = agent.reward_model.init_hidden(B, agent.device)

        seen: list[torch.Tensor] = []
        original = agent.reward_model.predict

        def spy(obs, actions, hidden_state=None):  # noqa: ANN001
            seen.append(obs.detach().clone())
            return original(obs, actions, hidden_state)

        agent.reward_model.predict = spy  # type: ignore[method-assign]
        obs, *_ = agent._imagine_trajectory(obs_history, action_history, hidden)

        # The first query must be on x_0 (the last conditioning frame), i.e. the
        # frame the policy acted on -- not on the freshly generated x_1.
        assert torch.equal(seen[0], obs_history[:, -1])
        for step, queried in enumerate(seen):
            assert torch.equal(queried, obs[:, step])

    def test_diffusion_conditions_on_the_actions_taken_at_each_frame(self):
        """D is trained on (x_{t-L+1..t}, a_{t-L+1..t}) -> x_{t+1}, with a_j the
        action taken at x_j (``SequenceDataset``). The first imagined frame must
        see a_0..a_{L-2} from the history plus the policy's own a_{L-1}.
        """
        agent = self._agent()
        B, L = 2, agent.config.burn_in_length
        size = agent.config.obs_size

        obs_history = torch.rand(B, L, 3, size, size)
        action_history = torch.randint(0, agent.action_dim, (B, L))
        hidden = agent.reward_model.init_hidden(B, agent.device)

        seen: list[torch.Tensor] = []
        original = agent.sampler.sample

        def spy(*args, **kwargs):  # noqa: ANN002, ANN003
            seen.append(kwargs["actions"].detach().clone())
            return original(*args, **kwargs)

        agent.sampler.sample = spy  # type: ignore[method-assign]
        _, _, _, actions, _ = agent._imagine_trajectory(
            obs_history, action_history, hidden
        )

        assert torch.equal(seen[0][:, :-1], action_history[:, :-1])
        assert torch.equal(seen[0][:, -1], actions[:, 0])
        # Each later step slides the window forward by one imagined action.
        assert torch.equal(seen[1][:, -2], actions[:, 0])
        assert torch.equal(seen[1][:, -1], actions[:, 1])
