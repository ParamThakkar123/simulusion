from unittest.mock import patch

import pytest
import torch

from simulusion.models.iris_agent import IRISAgent, compute_lambda_return
from simulusion.configs.iris_config import IRISConfig


class TestComputeLambdaReturn:
    def test_compute_lambda_return_shapes(self):
        B, T = 4, 10
        rewards = torch.randn(B, T)
        values = torch.randn(B, T + 1)
        discounts = torch.full((B, T), 0.99)
        lambda_coef = 0.95

        result = compute_lambda_return(rewards, values, discounts, lambda_coef)

        assert result.shape == (B, T)

    def test_compute_lambda_return_single_batch(self):
        B, T = 1, 5
        rewards = torch.randn(B, T)
        values = torch.randn(B, T + 1)
        discounts = torch.full((B, T), 0.99)
        lambda_coef = 0.95

        result = compute_lambda_return(rewards, values, discounts, lambda_coef)

        assert result.shape == (B, T)

    def test_compute_lambda_return_matches_expected(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        values = torch.tensor([[10.0, 8.0, 6.0, 4.0], [5.0, 4.0, 3.0, 2.0]])
        discounts = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        lambda_coef = 0.0

        result = compute_lambda_return(rewards, values, discounts, lambda_coef)

        expected_last = rewards[:, -1] + discounts[:, -1] * values[:, -1]
        assert torch.allclose(result[:, -1], expected_last)


class TestIRISAgentImagineRollout:
    @pytest.fixture
    def config(self):
        config = IRISConfig()
        config.vocab_size = 32
        config.tokens_per_frame = 16
        config.token_embedding_dim = 128
        config.frame_channels = 3
        config.encoder_channels = 32
        config.decoder_channels = 32
        config.frame_shape = (3, 64, 64)
        config.transformer_layers = 2
        config.transformer_heads = 4
        config.transformer_embed_dim = 128
        config.discount = 0.99
        return config

    @pytest.fixture
    def agent(self, config):
        device = torch.device("cpu")
        agent = IRISAgent(config, action_size=4, device=device)
        return agent

    def test_imagine_rollout_reward_shape(self, agent):
        B, C, H, W = 2, 3, 64, 64
        initial_frame = torch.rand(B, C, H, W)
        horizon = 5

        with torch.no_grad():
            trajectory = agent.imagine_rollout(
                initial_frame, horizon=horizon, stop_on_termination=False
            )

        assert trajectory["rewards"].shape == (B, horizon)

    def test_imagine_rollout_frames_shape(self, agent):
        B, C, H, W = 2, 3, 64, 64
        initial_frame = torch.rand(B, C, H, W)
        horizon = 5

        with torch.no_grad():
            trajectory = agent.imagine_rollout(
                initial_frame, horizon=horizon, stop_on_termination=False
            )

        assert trajectory["frames"].shape == (B, horizon + 1, C, H, W)

    def test_imagine_rollout_actions_shape(self, agent):
        B, C, H, W = 2, 3, 64, 64
        initial_frame = torch.rand(B, C, H, W)
        horizon = 5

        with torch.no_grad():
            trajectory = agent.imagine_rollout(
                initial_frame, horizon=horizon, stop_on_termination=False
            )

        assert trajectory["actions"].shape == (B, horizon)

    def test_imagine_rollout_stops_early_on_predicted_termination(self, agent):
        """Paper 2.3: imagination stops if an episode end is predicted.

        With the termination head forced to always predict "terminal", the very
        first step should end the rollout, and every returned tensor must agree
        on that shortened length.
        """
        B, C, H, W = 2, 3, 64, 64
        initial_frame = torch.rand(B, C, H, W)

        with torch.no_grad():
            # logits [-inf-ish, +big] => argmax is class 1 (terminal).
            agent.transformer.termination_head.weight.zero_()
            agent.transformer.termination_head.bias.copy_(torch.tensor([-10.0, 10.0]))
            trajectory = agent.imagine_rollout(initial_frame, horizon=5)

        steps = trajectory["actions"].shape[1]
        assert steps == 1, f"expected to stop after one step, got {steps}"
        assert trajectory["rewards"].shape == (B, steps)
        assert trajectory["continues"].shape == (B, steps)
        assert trajectory["frames"].shape == (B, steps + 1, C, H, W)
        # continues = 1 - P(terminal), which is ~0 for a confident termination.
        assert torch.all(trajectory["continues"] < 1e-3)

    @pytest.mark.parametrize("capacity_steps", [3, 5, 8])
    def test_imagine_rollout_survives_context_overflow(self, agent, capacity_steps):
        """A rollout longer than the Transformer's context must not crash.

        When the KV cache fills, imagination rebuilds it from the most recent
        timesteps. The retained window has to leave room for at least one more
        (action + K tokens) block, or the rebuild overflows immediately.
        """
        K = agent.config.tokens_per_frame
        agent.transformer.max_seq_len = K + capacity_steps * (K + 1)
        # Never terminate, so the full horizon is always attempted.
        agent.transformer.termination_head.bias.data.copy_(torch.tensor([10.0, -10.0]))

        horizon = 14
        with torch.no_grad():
            trajectory = agent.imagine_rollout(
                torch.rand(2, 3, 64, 64), horizon=horizon, stop_on_termination=False
            )

        assert trajectory["actions"].shape == (2, horizon)
        assert torch.isfinite(trajectory["rewards"]).all()

    def test_imagine_rollout_conditions_on_full_history(self, agent):
        """The world model must see the whole imagined trajectory, not one frame.

        Paper 2.3 conditions each new frame on (z_0, a_0, ..., z_t, a_t). The KV
        cache should therefore hold one action plus K tokens per imagined step on
        top of the initial frame's K tokens.
        """
        B = 2
        K = agent.config.tokens_per_frame
        horizon = 4

        with torch.no_grad():
            # Never terminate, so the rollout runs the full horizon.
            agent.transformer.termination_head.weight.zero_()
            agent.transformer.termination_head.bias.copy_(torch.tensor([10.0, -10.0]))
            cache = agent.transformer.init_cache(B, torch.device("cpu"))
            tokens = torch.randint(0, agent.config.vocab_size, (B, 1, K))
            pos = agent.transformer.prime_cache(tokens, None, cache, start_pos=0)
            assert pos == K

            for step in range(horizon):
                action = torch.zeros(B, dtype=torch.long)
                _, _, _, pos = agent.transformer.generate_frame_cached(
                    action, cache, start_pos=pos, sample=False
                )
                assert pos == K + (step + 1) * (K + 1)
                assert cache.length == pos

    def test_rollout_cache_sized_to_horizon_matches_full_cache(self, agent):
        """Sizing the KV cache to the rollout must not change the rollout."""
        agent.transformer.termination_head.bias.data.copy_(torch.tensor([10.0, -10.0]))
        initial_frame = torch.rand(2, 3, 64, 64)
        sized = agent.transformer.init_cache

        def full(batch_size, device, dtype=None, max_len=None):
            return sized(batch_size, device, dtype)

        results = []
        for init_cache in (full, sized):
            with patch.object(agent.transformer, "init_cache", init_cache):
                torch.manual_seed(0)
                with torch.no_grad():
                    results.append(
                        agent.imagine_rollout(
                            initial_frame, horizon=4, stop_on_termination=False
                        )
                    )
        for key in ("frames", "actions", "rewards", "continues"):
            assert torch.allclose(results[0][key].float(), results[1][key].float())

    def test_init_cache_max_len_never_exceeds_context(self, agent):
        transformer = agent.transformer
        cpu = torch.device("cpu")
        assert transformer.init_cache(1, cpu).max_len == transformer.max_seq_len
        assert transformer.init_cache(1, cpu, max_len=20).max_len == 20
        too_long = transformer.max_seq_len + 50
        assert (
            transformer.init_cache(1, cpu, max_len=too_long).max_len
            == transformer.max_seq_len
        )

    def test_act_with_epsilon_samples_from_the_policy(self, agent):
        """Paper Table 5: epsilon-greedy is combined with sampling, not argmax.

        With near-uniform logits and a small epsilon, collection must still
        spread over the actions; an argmax would repeat the top one.
        """
        batch = 512
        logits = torch.zeros(batch, 1, agent.action_size)
        logits[..., 0] = 0.1  # argmax would always pick action 0

        def fake_forward(frames, hidden=None):
            return logits, torch.zeros(batch, 1), hidden

        with patch.object(agent, "forward_actor_critic", fake_forward):
            torch.manual_seed(0)
            actions = agent.act(torch.rand(batch, 3, 64, 64), epsilon=0.01)

        assert actions.shape == (batch,)
        assert (actions == 0).float().mean() < 0.5
        assert len(actions.unique()) == agent.action_size

    def test_reconstruct_in_chunks_matches_one_pass(self, agent):
        frames = torch.rand(2, 5, 3, 64, 64)
        with torch.no_grad():
            agent.reconstruct_chunk_size = 1000
            whole = agent.reconstruct(frames)
            agent.reconstruct_chunk_size = 3
            chunked = agent.reconstruct(frames)
        assert chunked.shape == frames.shape
        assert torch.allclose(whole, chunked, atol=1e-5)

    def test_update_actor_critic_with_imagined_trajectory(self, agent):
        B, T, C, H, W = 2, 5, 3, 64, 64
        frames = torch.randn(B, T + 1, C, H, W)
        actions = torch.randint(0, 4, (B, T))
        rewards = torch.randn(B, T)

        imagined = {
            "frames": frames,
            "actions": actions,
            "rewards": rewards,
        }

        metrics = agent.update_actor_critic(imagined)

        assert "actor_loss" in metrics
        assert "value_loss" in metrics
        assert "total_loss" in metrics


class TestGraphedDecodeStep:
    """Imagination's CUDA-graph fast path must compute what the eager path does."""

    @staticmethod
    def _transformer(device):
        from simulusion.models.iris_transformer import IRISTransformer

        torch.manual_seed(0)
        return (
            IRISTransformer(
                vocab_size=32,
                tokens_per_frame=16,
                action_size=4,
                embed_dim=64,
                num_layers=2,
                num_heads=4,
                dropout=0.0,
            )
            .to(device)
            .eval()
        )

    def test_reset_cache_reprimes_like_a_fresh_one(self):
        transformer = self._transformer("cpu")
        tokens = torch.randint(0, 32, (2, 1, 16))
        action = torch.tensor([1, 2])

        reused = transformer.init_cache(2, torch.device("cpu"))
        pos = transformer.prime_cache(tokens, None, reused, start_pos=0)
        transformer.generate_frame_cached(action, reused, pos, sample=False)
        reused.reset()

        results = []
        for cache in (reused, transformer.init_cache(2, torch.device("cpu"))):
            pos = transformer.prime_cache(tokens, None, cache, start_pos=0)
            results.append(
                transformer.generate_frame_cached(action, cache, pos, sample=False)
            )
        assert torch.equal(results[0][1], results[1][1])
        assert torch.allclose(results[0][0], results[1][0])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_graphed_decode_matches_eager_decode(self):
        from simulusion.models.iris_transformer import GraphedDecodeStep

        device = torch.device("cuda")
        transformer = self._transformer(device)
        tokens = torch.randint(0, 32, (3, 1, 16), device=device)
        actions = torch.randint(0, 4, (4, 3), device=device)

        outputs = []
        for graphed in (False, True):
            cache = transformer.init_cache(3, device)
            step = GraphedDecodeStep(transformer, cache, 3) if graphed else None
            pos = transformer.prime_cache(tokens, None, cache, start_pos=0)
            frames = []
            for action in actions:
                logits, generated, hidden, pos = transformer.generate_frame_cached(
                    action, cache, pos, sample=False, decode_step=step
                )
                frames.append((logits, generated, hidden))
            outputs.append(frames)

        for eager, graphed in zip(*outputs):
            assert torch.equal(eager[1], graphed[1])
            assert torch.allclose(eager[0], graphed[0], atol=1e-5)
            assert torch.allclose(eager[2], graphed[2], atol=1e-5)


class TestIRISAgentCheckpointSecurity:
    @pytest.fixture
    def config(self):
        config = IRISConfig()
        config.vocab_size = 32
        config.tokens_per_frame = 16
        config.token_embedding_dim = 128
        config.frame_channels = 3
        config.encoder_channels = 32
        config.decoder_channels = 32
        config.frame_shape = (3, 64, 64)
        config.transformer_layers = 2
        config.transformer_heads = 4
        config.transformer_embed_dim = 128
        return config

    @pytest.fixture
    def agent(self, config):
        return IRISAgent(config, action_size=4, device=torch.device("cpu"))

    def test_save_stores_config_as_weights_only_safe_dict(self, agent, tmp_path):
        path = tmp_path / "iris.pt"

        agent.save(str(path))
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        assert isinstance(checkpoint["config"], dict)

    def test_load_uses_weights_only_deserialization(self, agent):
        checkpoint = {
            "checkpoint_format": IRISAgent.CHECKPOINT_FORMAT,
            "encoder": agent.encoder.state_dict(),
            "decoder": agent.decoder.state_dict(),
            "transformer": agent.transformer.state_dict(),
            "cnn": agent.cnn.state_dict(),
            "lstm": agent.lstm.state_dict(),
            "actor_head": agent.actor_head.state_dict(),
            "critic_head": agent.critic_head.state_dict(),
            "autoencoder_opt": agent.autoencoder_opt.state_dict(),
            "transformer_opt": agent.transformer_opt.state_dict(),
            "ac_opt": agent.ac_opt.state_dict(),
            "global_step": 7,
            "epoch": 3,
        }

        with patch(
            "simulusion.models.iris_agent.torch.load", return_value=checkpoint
        ) as mock_load:
            agent.load("checkpoint.pt")

        mock_load.assert_called_once_with(
            "checkpoint.pt",
            map_location=agent.device,
            weights_only=True,
        )
        assert agent.global_step == 7
        assert agent.current_epoch == 3

    def test_save_load_round_trip(self, agent, tmp_path):
        path = tmp_path / "iris.pt"
        agent.global_step = 11
        agent.save(str(path))
        agent.load(str(path))
        assert agent.global_step == 11

    def test_load_rejects_stale_checkpoint_format(self, agent):
        """Pre-GPT-block checkpoints must fail loudly, not with missing keys.

        The Transformer's module layout changed when KV caching was added, so
        old weights cannot be mapped across. A checkpoint with no
        ``checkpoint_format`` key predates the field and is therefore v1.
        """
        stale = {"encoder": agent.encoder.state_dict()}

        with patch("simulusion.models.iris_agent.torch.load", return_value=stale):
            with pytest.raises(RuntimeError, match="checkpoint format v1"):
                agent.load("old.pt")
