"""Regression tests for the IRIS data pipeline.

These cover failure modes that are silent -- training runs to completion and
reports falling losses while the agent learns nothing -- so they are worth
pinning down explicitly.
"""

import numpy as np
import pytest
import torch

from simulusion.configs.iris_config import IRISConfig
from simulusion.memory.iris_memory import IRISReplayBuffer
from simulusion.models.iris_agent import IRISAgent


@pytest.fixture
def buffer():
    return IRISReplayBuffer(
        size=200, obs_shape=(3, 64, 64), action_size=4, seq_len=4, batch_size=2
    )


class TestReplayBufferDtype:
    def test_frames_survive_a_round_trip(self, buffer):
        """Frames must reach the buffer as uint8, not as floats in [0, 1].

        Assigning a float array into the uint8 store truncates every pixel below
        1.0 to zero, so the world model trains on black images while the
        reconstruction loss happily converges to ~0.
        """
        frame = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        buffer.add(frame, np.zeros(4, dtype=np.float32), 0.0, False)

        stored = buffer.observations[0]
        assert stored.dtype == np.uint8
        np.testing.assert_array_equal(stored, frame)
        assert stored.max() > 1, "frames collapsed to near-zero on insertion"

    def test_sampled_sequences_are_not_blank(self, buffer):
        for _ in range(50):
            frame = np.random.randint(64, 256, (3, 64, 64), dtype=np.uint8)
            buffer.add(frame, np.zeros(4, dtype=np.float32), 0.0, False)

        obs, _, _, _ = buffer.sample_sequence()
        assert obs.dtype == np.uint8
        assert obs.min() >= 64
        assert obs.mean() > 1.0


class TestBurnInSampling:
    def test_shapes_and_dtype(self, buffer):
        for _ in range(60):
            frame = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
            buffer.add(frame, np.zeros(4, dtype=np.float32), 0.0, False)

        start, burn_in = buffer.sample_with_burn_in(batch_size=5, burn_in=8)
        assert start.shape == (5, 3, 64, 64)
        assert burn_in.shape == (5, 8, 3, 64, 64)
        assert burn_in.dtype == np.uint8

    def test_zero_burn_in_returns_empty_window(self, buffer):
        for _ in range(10):
            buffer.add(
                np.zeros((3, 64, 64), dtype=np.uint8),
                np.zeros(4, dtype=np.float32),
                0.0,
                False,
            )
        start, burn_in = buffer.sample_with_burn_in(batch_size=3, burn_in=0)
        assert start.shape == (3, 3, 64, 64)
        assert burn_in.shape == (3, 0, 3, 64, 64)

    def test_burn_in_does_not_cross_episode_boundaries(self, buffer):
        """Context from a previous episode must not leak into the next one."""
        # Episode A: constant value 10, ending with a terminal.
        for i in range(10):
            buffer.add(
                np.full((3, 64, 64), 10, dtype=np.uint8),
                np.zeros(4, dtype=np.float32),
                0.0,
                i == 9,
            )
        # Episode B: constant value 200.
        for _ in range(5):
            buffer.add(
                np.full((3, 64, 64), 200, dtype=np.uint8),
                np.zeros(4, dtype=np.float32),
                0.0,
                False,
            )

        # Draw many samples; none may mix the two episodes' pixel values.
        for _ in range(30):
            start, burn_in = buffer.sample_with_burn_in(batch_size=8, burn_in=6)
            for i in range(8):
                if start[i].flat[0] == 200:
                    assert not (burn_in[i] == 10).any(), (
                        "burn-in window leaked frames from the previous episode"
                    )


class TestPolicyInputDomain:
    """Paper A.1: the policy consumes D(E(x)), in both imagination and reality."""

    @pytest.fixture
    def agent(self):
        config = IRISConfig()
        config.vocab_size = 32
        config.token_embedding_dim = 64
        config.encoder_channels = 16
        config.decoder_depth = 8
        config.transformer_layers = 2
        config.transformer_embed_dim = 64
        config.perceptual_weight = 0.0  # keep the test light and offline
        return IRISAgent(config, action_size=4, device=torch.device("cpu"))

    def test_reconstruct_preserves_shape_and_range(self, agent):
        frames = torch.rand(3, 3, 64, 64)
        with torch.no_grad():
            recon = agent.reconstruct(frames)
        assert recon.shape == frames.shape
        assert float(recon.min()) >= 0.0 and float(recon.max()) <= 1.0

    def test_reconstruct_handles_time_dimension(self, agent):
        frames = torch.rand(2, 5, 3, 64, 64)
        with torch.no_grad():
            recon = agent.reconstruct(frames)
        assert recon.shape == frames.shape


class TestTransformerUpdateIsolation:
    """The transformer update must not reach into the autoencoder."""

    @pytest.fixture
    def agent(self):
        config = IRISConfig()
        config.vocab_size = 32
        config.token_embedding_dim = 64
        config.encoder_channels = 16
        config.decoder_depth = 8
        config.transformer_layers = 2
        config.transformer_embed_dim = 64
        config.transformer_timesteps = 4
        config.perceptual_weight = 0.0
        return IRISAgent(config, action_size=4, device=torch.device("cpu"))

    @staticmethod
    def _batch(agent, one_hot: bool):
        b, t = 2, agent.config.transformer_timesteps
        actions = torch.randint(0, agent.action_size, (b, t))
        if one_hot:
            actions = torch.nn.functional.one_hot(actions, agent.action_size).float()
        else:
            # The replay buffer stores float32; a 2-D float tensor must not be
            # handed straight to nn.Embedding.
            actions = actions.float()
        return (
            torch.rand(b, t + 1, 3, 64, 64),
            actions,
            torch.zeros(b, t),
            torch.zeros(b, t, dtype=torch.long),
        )

    @pytest.mark.parametrize("one_hot", [True, False])
    def test_accepts_one_hot_and_scalar_float_actions(self, agent, one_hot):
        metrics = agent.update_transformer(*self._batch(agent, one_hot))
        assert "token_loss" in metrics

    def test_does_not_train_the_encoder(self, agent):
        """Token-prediction loss must not backpropagate into the encoder.

        The autoencoder has its own objective and its own optimizer; gradients
        arriving here would be wasted work and would leave stale ``.grad`` on
        parameters the transformer optimizer does not own.
        """
        before = [p.detach().clone() for p in agent.encoder.parameters()]

        agent.update_transformer(*self._batch(agent, one_hot=True))

        assert all(p.grad is None for p in agent.encoder.parameters()), (
            "transformer loss leaked gradients into the encoder"
        )
        assert all(
            torch.equal(p, b) for p, b in zip(agent.encoder.parameters(), before)
        ), "encoder weights moved during a transformer update"


class TestRecurrentPolicyState:
    @pytest.fixture
    def agent(self):
        config = IRISConfig()
        config.vocab_size = 32
        config.token_embedding_dim = 64
        config.encoder_channels = 16
        config.decoder_depth = 8
        config.transformer_layers = 2
        config.transformer_embed_dim = 64
        config.perceptual_weight = 0.0
        return IRISAgent(config, action_size=4, device=torch.device("cpu"))

    def test_act_returns_and_advances_hidden_state(self, agent):
        """Paper A.3: the policy is recurrent, so state must thread across steps."""
        frame = torch.rand(2, 3, 64, 64)

        out = agent.act(frame, return_hidden=True)
        assert isinstance(out, tuple)
        _, hidden = out
        h, c = hidden
        assert h.shape == (agent.config.actor_layers, 2, agent.config.actor_hidden_size)

        out2 = agent.act(frame, hidden=hidden, return_hidden=True)
        assert isinstance(out2, tuple)
        _, hidden2 = out2
        # Feeding the same frame again with carried state must change the state,
        # otherwise the LSTM is being silently reset.
        assert not torch.allclose(hidden2[0], h)

    def test_act_without_return_hidden_is_backwards_compatible(self, agent):
        actions = agent.act(torch.rand(2, 3, 64, 64))
        assert isinstance(actions, torch.Tensor)
        assert actions.shape == (2,)

    def test_burn_in_produces_nonzero_state(self, agent):
        frames = torch.rand(2, 6, 3, 64, 64)
        hidden = agent.burn_in(frames)
        assert hidden is not None
        assert float(hidden[0].abs().sum()) > 0.0

    def test_burn_in_with_no_frames_is_none(self, agent):
        assert agent.burn_in(torch.rand(2, 0, 3, 64, 64)) is None

    def test_act_restores_training_mode(self, agent):
        agent.train()
        agent.act(torch.rand(1, 3, 64, 64))
        assert agent.training, "act() left the module in eval mode"
