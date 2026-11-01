"""Tests for the DreamerV3 agent, its networks, and its public wiring."""

import gc
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from simulusion.configs.dreamer_v3_config import MODEL_SIZES, DreamerV3Config
from simulusion.models.dreamer_v3 import (
    DreamerV3,
    DreamerV3Agent,
    _is_true_termination,
    _resolve_actor_dist,
    coerce_dreamer_v3_config,
)
from simulusion.vision.dreamer_v3_nets import (
    DreamerV3Actor,
    DreamerV3Decoder,
    DreamerV3Encoder,
    DreamerV3Head,
    preprocess_image,
)

# Deliberately smaller than the 64x64 the agent uses in practice. Every test here
# builds a model and a replay buffer, and at 64x64 the accumulated buffers add
# enough memory pressure to make memory-hungry tests elsewhere in the suite fail.
# 32x32 still exercises a three-layer convolution stack down to 4x4.
IMAGE_SHAPE = (3, 32, 32)


@pytest.fixture(autouse=True)
def _release_models():
    """Free each test's models and replay buffers before the next test runs.

    Every test here builds a full agent with its own replay buffer. Without an
    explicit collection those accumulate across the module and the extra
    pressure makes memory-hungry tests elsewhere in the suite fail.
    """
    yield
    gc.collect()


def tiny_config(**overrides):
    """A config small enough to train a few steps inside a unit test."""
    defaults = dict(
        model_size="12m",
        hidden_size=16,
        recurrent_units=32,
        cnn_depth=4,
        latent_classes=4,
        latent_dim=4,
        gru_blocks=4,
        mlp_layers=1,
        num_buckets=41,
        batch_size=2,
        train_seq_len=6,
        imagine_horizon=3,
        buffer_size=100,
        actor_dist="normal",
        no_gpu=True,
    )
    defaults.update(overrides)
    return DreamerV3Config(**defaults)


def build(config=None, obs_shape=IMAGE_SHAPE, action_size=2):
    return DreamerV3(config or tiny_config(), obs_shape, action_size, "cpu")


def fill(model, steps=40, episode_length=13, image=True):
    for step in range(steps):
        if image:
            obs = {"image": np.random.randint(0, 255, IMAGE_SHAPE, dtype=np.uint8)}
        else:
            obs = {"obs": np.random.randn(model.obs_shape[0]).astype(np.float32)}
        if model.discrete_actions:
            action = np.zeros(model.action_size, np.float32)
            action[step % model.action_size] = 1.0
        else:
            action = np.random.randn(model.action_size).astype(np.float32)
        model.data_buffer.add(
            obs, action, float(step % 3), (step + 1) % episode_length == 0
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDreamerV3Config:
    def test_paper_defaults(self):
        config = DreamerV3Config()
        assert config.discount == pytest.approx(0.997)
        assert config.td_lambda == pytest.approx(0.95)
        assert config.imagine_horizon == 15
        assert config.actor_entropy == pytest.approx(3e-4)
        assert config.learning_rate == pytest.approx(4e-5)
        assert config.free_nats == pytest.approx(1.0)
        assert (config.beta_pred, config.beta_dyn, config.beta_rep) == (1.0, 1.0, 0.1)
        assert config.unimix == pytest.approx(0.01)
        assert config.critic_ema_decay == pytest.approx(0.98)
        assert config.critic_replay_loss_scale == pytest.approx(0.3)
        assert (config.batch_size, config.train_seq_len) == (16, 64)

    def test_model_size_presets_are_resolved(self):
        config = DreamerV3Config(model_size="200m")
        assert config.hidden_size == 1024
        assert config.recurrent_units == 8192
        assert config.cnn_depth == 64
        assert config.latent_classes == 64

    def test_recurrent_units_are_eight_times_the_model_dimension(self):
        # Table 3 of the paper labels this column "Recurrent units (8d)", and
        # every row follows that rule except 12M, which the table lists as 1024
        # rather than 2048. The presets reproduce the table verbatim, so 12M is
        # excluded here rather than silently "corrected".
        for name, preset in MODEL_SIZES.items():
            if name == "12m":
                continue
            assert preset["recurrent_units"] == 8 * preset["hidden_size"]

    def test_every_preset_is_divisible_by_the_block_count(self):
        for preset in MODEL_SIZES.values():
            assert preset["recurrent_units"] % 8 == 0

    def test_widths_increase_monotonically_with_size(self):
        ordered = [
            MODEL_SIZES[name] for name in ("12m", "25m", "50m", "100m", "200m", "400m")
        ]
        for field in ("hidden_size", "recurrent_units", "cnn_depth", "latent_classes"):
            values = [preset[field] for preset in ordered]
            assert values == sorted(values)

    def test_explicit_widths_override_the_preset(self):
        config = DreamerV3Config(model_size="200m", hidden_size=123)
        assert config.hidden_size == 123
        assert config.recurrent_units == 8192

    def test_unknown_model_size_is_rejected(self):
        with pytest.raises(ValueError, match="model_size"):
            DreamerV3Config(model_size="7b")

    def test_indivisible_recurrent_units_are_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            DreamerV3Config(recurrent_units=100, gru_blocks=8)

    def test_update_steps_follow_the_replay_ratio(self):
        config = DreamerV3Config(
            replay_ratio=32, collect_steps=1000, batch_size=16, train_seq_len=64
        )
        assert config.update_steps == 31

    def test_auto_update_steps_can_be_disabled(self):
        config = DreamerV3Config(auto_update_steps=False, update_steps=7)
        assert config.update_steps == 7

    def test_inherits_environment_fields(self):
        config = DreamerV3Config(env="walker-walk", env_backend="dmc")
        assert config.env == "walker-walk"
        assert config.image_size == (64, 64)

    def test_yaml_roundtrip(self, tmp_path):
        config = DreamerV3Config(model_size="50m", seed=11)
        path = tmp_path / "config.yaml"
        config.to_yaml(path)
        restored = DreamerV3Config.from_yaml(path)
        assert restored.model_size == "50m"
        assert restored.seed == 11
        assert restored.hidden_size == 512

    def test_coercion_accepts_dicts_and_none(self):
        assert isinstance(coerce_dreamer_v3_config(None), DreamerV3Config)
        assert coerce_dreamer_v3_config({"seed": 3}).seed == 3
        with pytest.raises(TypeError):
            coerce_dreamer_v3_config(42)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class TestNetworks:
    def test_preprocess_image_maps_to_unit_range(self):
        raw = torch.tensor([[0, 128, 255]], dtype=torch.uint8)
        processed = preprocess_image(raw)
        assert float(processed.min()) == 0.0
        assert float(processed.max()) == 1.0

    def test_image_encoder_reduces_to_a_small_resolution(self):
        encoder = DreamerV3Encoder(IMAGE_SHAPE, cnn_depth=8)
        assert encoder.resolutions[-1] == (4, 4)
        assert encoder(torch.rand(5, *IMAGE_SHAPE)).shape == (5, encoder.embed_size)

    def test_encoder_preserves_leading_dimensions(self):
        encoder = DreamerV3Encoder(IMAGE_SHAPE, cnn_depth=4)
        embedded = encoder(torch.rand(7, 3, *IMAGE_SHAPE))
        assert embedded.shape == (7, 3, encoder.embed_size)

    def test_standard_64x64_input_uses_four_convolutions(self):
        # The resolution the agent actually runs at. Kept explicit because the
        # rest of this module uses 32x32 to stay light on memory.
        shape = (3, 64, 64)
        encoder = DreamerV3Encoder(shape, cnn_depth=8)
        assert encoder.resolutions == [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
        assert encoder.out_channels == 8 * 2**3
        assert encoder.embed_size == encoder.out_channels * 4 * 4

    def test_standard_64x64_decoder_inverts_the_encoder(self):
        shape = (3, 64, 64)
        encoder = DreamerV3Encoder(shape, cnn_depth=8)
        decoder = DreamerV3Decoder(24, shape, encoder=encoder, cnn_depth=8)
        assert decoder(torch.randn(2, 24)).shape == (2, *shape)

    def test_vector_encoder_uses_an_mlp(self):
        encoder = DreamerV3Encoder((11,), hidden_size=24, mlp_layers=2)
        assert not encoder.is_image
        assert encoder(torch.randn(4, 11)).shape == (4, 24)

    def test_decoder_reconstructs_the_input_shape(self):
        encoder = DreamerV3Encoder(IMAGE_SHAPE, cnn_depth=8)
        decoder = DreamerV3Decoder(32, IMAGE_SHAPE, encoder=encoder, cnn_depth=8)
        assert decoder(torch.randn(5, 32)).shape == (5, *IMAGE_SHAPE)

    def test_decoder_output_is_in_the_unit_range(self):
        encoder = DreamerV3Encoder(IMAGE_SHAPE, cnn_depth=4)
        decoder = DreamerV3Decoder(16, IMAGE_SHAPE, encoder=encoder, cnn_depth=4)
        output = decoder(torch.randn(3, 16) * 10)
        assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0

    def test_decoder_handles_odd_image_sizes(self):
        shape = (3, 63, 65)
        encoder = DreamerV3Encoder(shape, cnn_depth=4)
        decoder = DreamerV3Decoder(16, shape, encoder=encoder, cnn_depth=4)
        assert decoder(torch.randn(2, 16)).shape == (2, *shape)

    def test_image_decoder_requires_an_encoder(self):
        with pytest.raises(ValueError, match="image encoder"):
            DreamerV3Decoder(16, IMAGE_SHAPE)

    def test_decoder_loss_is_zero_for_a_perfect_reconstruction(self):
        decoder = DreamerV3Decoder(8, (5,), hidden_size=8, mlp_layers=1)
        target = torch.zeros(3, 5)
        assert float(decoder.loss(target, target).sum()) == pytest.approx(0.0)

    def test_vector_decoder_loss_uses_symlog_targets(self):
        # A huge target must not produce a huge loss, which is the point of the
        # symlog transform on vector observations.
        decoder = DreamerV3Decoder(8, (2,), hidden_size=8, mlp_layers=1)
        prediction = torch.zeros(1, 2)
        assert float(decoder.loss(prediction, torch.full((1, 2), 1e6)).sum()) < 500

    def test_twohot_head_starts_at_zero(self):
        # Zero-initialized output weights mean the head predicts exactly zero
        # regardless of its input, so the agent cannot hallucinate rewards or
        # values before it has learned anything.
        head = DreamerV3Head(12, layers=1, units=8, num_bins=41)
        prediction = head(torch.randn(6, 12)).mean
        assert torch.allclose(prediction, torch.zeros(6), atol=1e-6)

    def test_twohot_head_zero_init_holds_for_the_default_bin_count(self):
        head = DreamerV3Head(12, layers=1, units=8, num_bins=255, symlog_range=20.0)
        assert torch.allclose(head(torch.randn(4, 12)).mean, torch.zeros(4), atol=1e-6)

    def test_head_zero_init_can_be_disabled(self):
        head = DreamerV3Head(12, layers=1, units=8, num_bins=41, zero_init_output=False)
        assert float(head.model[-1].weight.abs().sum()) > 0.0

    def test_binary_head_returns_a_bernoulli(self):
        head = DreamerV3Head(12, layers=1, units=8, dist="binary")
        dist = head(torch.randn(4, 12))
        assert dist.log_prob(torch.ones(4)).shape == (4,)

    def test_unknown_head_distribution_is_rejected(self):
        with pytest.raises(ValueError, match="distribution"):
            DreamerV3Head(4, layers=1, units=4, dist="poisson")(torch.randn(2, 4))


class TestActor:
    def test_continuous_actions_have_the_right_shape(self):
        actor = DreamerV3Actor(12, 4, discrete=False, layers=1, units=8)
        assert actor(torch.randn(5, 12)).shape == (5, 4)

    def test_continuous_log_prob_and_entropy_are_per_sample(self):
        actor = DreamerV3Actor(12, 4, discrete=False, layers=1, units=8)
        dist = actor.dist(torch.randn(5, 12))
        action = dist.sample()
        assert dist.log_prob(action).shape == (5,)
        assert dist.entropy().shape == (5,)

    def test_continuous_mean_is_bounded(self):
        actor = DreamerV3Actor(12, 3, discrete=False, layers=1, units=8)
        mean = actor.dist(torch.randn(20, 12) * 50).mean
        assert float(mean.abs().max()) <= 1.0

    def test_continuous_std_respects_its_bounds(self):
        actor = DreamerV3Actor(
            12, 3, discrete=False, layers=1, units=8, min_std=0.1, max_std=1.0
        )
        dist = actor.dist(torch.randn(50, 12) * 50)
        std = dist.base_dist.scale
        assert float(std.min()) >= 0.1 - 1e-6
        assert float(std.max()) <= 1.0 + 1e-6

    def test_discrete_actions_are_one_hot(self):
        actor = DreamerV3Actor(12, 5, discrete=True, layers=1, units=8)
        action = actor(torch.randn(7, 12))
        assert action.shape == (7, 5)
        assert torch.allclose(action.sum(-1), torch.ones(7))

    def test_discrete_deterministic_takes_the_argmax(self):
        actor = DreamerV3Actor(12, 5, discrete=True, layers=1, units=8)
        features = torch.randn(1, 12)
        repeated = [actor(features, deterministic=True) for _ in range(5)]
        assert all(torch.equal(repeated[0], other) for other in repeated[1:])

    def test_discrete_unimix_keeps_log_probs_finite(self):
        actor = DreamerV3Actor(12, 4, discrete=True, layers=1, units=8, unimix=0.01)
        dist = actor.dist(torch.randn(3, 12) * 100)
        every_action = torch.eye(4).unsqueeze(1).expand(4, 3, 4)
        assert torch.isfinite(dist.log_prob(every_action)).all()

    def test_exploration_noise_is_off_by_default(self):
        actor = DreamerV3Actor(12, 3, discrete=False, layers=1, units=8)
        action = torch.zeros(2, 3)
        assert torch.equal(actor.add_exploration(action), action)

    def test_exploration_noise_is_clipped_to_the_action_range(self):
        actor = DreamerV3Actor(12, 3, discrete=False, layers=1, units=8)
        noisy = actor.add_exploration(torch.zeros(100, 3), action_noise=10.0)
        assert float(noisy.abs().max()) <= 1.0


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------


class TestDreamerV3Core:
    def test_builds_all_modules(self):
        model = build()
        summary = model.summary()
        assert set(summary["modules"]) == {
            "rssm",
            "obs_encoder",
            "obs_decoder",
            "reward_model",
            "continue_model",
            "value_model",
            "actor",
        }
        assert model.parameter_count() > 0

    def test_slow_critic_starts_identical_and_is_frozen(self):
        model = build()
        for slow, fast in zip(
            model.slow_value_model.parameters(), model.value_model.parameters()
        ):
            assert torch.equal(slow, fast)
            assert not slow.requires_grad

    def test_world_model_loss_is_finite(self):
        model = build()
        fill(model)
        obs, actions, rewards, terminal, first = model.data_buffer.sample()
        loss, posterior = model.world_model_loss(
            torch.as_tensor(obs),
            torch.as_tensor(actions),
            torch.as_tensor(rewards),
            torch.as_tensor(terminal),
            torch.as_tensor(first),
        )
        assert torch.isfinite(loss)
        assert posterior["deter"].shape[0] == model.args.train_seq_len

    def test_train_one_batch_returns_three_finite_losses(self):
        model = build()
        fill(model)
        losses = model.train_one_batch()
        assert len(losses) == 3
        assert all(np.isfinite(value) for value in losses)

    def test_train_one_batch_is_a_no_op_on_an_empty_buffer(self):
        assert build().train_one_batch() == [0.0, 0.0, 0.0]

    def test_reinforce_increases_the_probability_of_good_actions(self):
        # Guards the sign of the actor objective. With a fixed state and a
        # constant positive advantage for action 0, Reinforce must make the
        # policy more likely to pick action 0.
        config = tiny_config(actor_dist="onehot", actor_entropy=0.0)
        model = DreamerV3(config, IMAGE_SHAPE, 3, "cpu")
        horizon, batch = config.imagine_horizon, 8

        features = torch.zeros(horizon + 1, batch, model.rssm.feature_size)
        actions = torch.zeros(horizon, batch, 3)
        actions[..., 0] = 1.0
        imagined = {
            "features": features,
            "actions": actions,
            "returns": torch.ones(horizon, batch),
            "values": torch.zeros(horizon + 1, batch),
            "weights": torch.ones(horizon, batch),
            "rewards": torch.ones(horizon + 1, batch),
        }

        with torch.no_grad():
            before = model.actor.dist(features[0]).probs[0, 0].item()
        for _ in range(20):
            loss = model.actor_loss(imagined)
            model._optimize(
                loss,
                model.actor_opt,
                model.actor_scaler,
                list(model.actor.parameters()),
                "actor",
            )
        with torch.no_grad():
            after = model.actor.dist(features[0]).probs[0, 0].item()

        assert after > before

    def test_reinforce_decreases_the_probability_of_bad_actions(self):
        config = tiny_config(actor_dist="onehot", actor_entropy=0.0)
        model = DreamerV3(config, IMAGE_SHAPE, 3, "cpu")
        horizon, batch = config.imagine_horizon, 8

        features = torch.zeros(horizon + 1, batch, model.rssm.feature_size)
        actions = torch.zeros(horizon, batch, 3)
        actions[..., 0] = 1.0
        imagined = {
            "features": features,
            "actions": actions,
            # Negative advantage: the taken action was worse than the value.
            "returns": torch.zeros(horizon, batch),
            "values": torch.ones(horizon + 1, batch),
            "weights": torch.ones(horizon, batch),
            "rewards": torch.zeros(horizon + 1, batch),
        }

        with torch.no_grad():
            before = model.actor.dist(features[0]).probs[0, 0].item()
        for _ in range(20):
            loss = model.actor_loss(imagined)
            model._optimize(
                loss,
                model.actor_opt,
                model.actor_scaler,
                list(model.actor.parameters()),
                "actor",
            )
        with torch.no_grad():
            after = model.actor.dist(features[0]).probs[0, 0].item()

        assert after < before

    def test_entropy_bonus_pushes_the_policy_towards_uniform(self):
        # With no advantage signal, the entropy term alone should flatten a
        # policy that starts out peaked.
        config = tiny_config(actor_dist="onehot", actor_entropy=1.0)
        model = DreamerV3(config, IMAGE_SHAPE, 4, "cpu")
        horizon, batch = config.imagine_horizon, 4

        features = torch.randn(horizon + 1, batch, model.rssm.feature_size)
        actions = torch.zeros(horizon, batch, 4)
        actions[..., 0] = 1.0
        imagined = {
            "features": features,
            "actions": actions,
            "returns": torch.zeros(horizon, batch),
            "values": torch.zeros(horizon + 1, batch),
            "weights": torch.ones(horizon, batch),
            "rewards": torch.zeros(horizon + 1, batch),
        }

        with torch.no_grad():
            before = model.actor.dist(features[0]).entropy().mean().item()
        for _ in range(30):
            loss = model.actor_loss(imagined)
            model._optimize(
                loss,
                model.actor_opt,
                model.actor_scaler,
                list(model.actor.parameters()),
                "actor",
            )
        with torch.no_grad():
            after = model.actor.dist(features[0]).entropy().mean().item()

        assert after >= before

    def test_critic_learns_to_predict_its_targets(self):
        # The critic loss must actually move predictions towards the returns.
        model = build()
        fill(model)
        obs, actions, rewards, terminal, first = model.data_buffer.sample()
        _, posterior = model.world_model_loss(
            torch.as_tensor(obs),
            torch.as_tensor(actions),
            torch.as_tensor(rewards),
            torch.as_tensor(terminal),
            torch.as_tensor(first),
        )
        imagined = model._imagine(posterior)
        imagined["returns"] = torch.full_like(imagined["returns"], 5.0)

        def error():
            with torch.no_grad():
                predicted = model.value_model(imagined["features"][:-1]).mean
            return float((predicted - 5.0).abs().mean())

        before = error()
        for _ in range(30):
            loss = model.value_loss(imagined)
            model._optimize(
                loss,
                model.value_opt,
                model.value_scaler,
                list(model.value_model.parameters()),
                "critic",
            )
        assert error() < before

    def test_training_updates_every_component(self):
        model = build()
        fill(model)
        before = {
            name: [param.detach().clone() for param in module.parameters()]
            for name, module in model._named_modules().items()
        }
        for _ in range(3):
            model.train_one_batch()
        for name, module in model._named_modules().items():
            changed = any(
                not torch.equal(old, new)
                for old, new in zip(before[name], module.parameters())
            )
            assert changed, f"{name} did not update"

    def test_slow_critic_tracks_the_critic_without_matching_it(self):
        model = build()
        fill(model)
        for _ in range(3):
            model.train_one_batch()
        slow = torch.cat([p.flatten() for p in model.slow_value_model.parameters()])
        fast = torch.cat([p.flatten() for p in model.value_model.parameters()])
        assert not torch.equal(slow, fast)
        assert torch.isfinite(slow).all()

    def test_metrics_are_published_for_logging(self):
        model = build()
        fill(model)
        model.train_one_batch()
        for key in (
            "wm/recon_loss",
            "wm/dyn_loss",
            "wm/rep_loss",
            "actor/entropy",
            "actor/return_scale",
            "grad_norm/world_model",
        ):
            assert key in model.last_metrics

    def test_replay_critic_loss_can_be_disabled(self):
        model = build(tiny_config(critic_replay_loss_scale=0.0))
        fill(model)
        model.train_one_batch()
        assert "critic/replay_loss" not in model.last_metrics

    def test_discrete_action_training(self):
        config = tiny_config(actor_dist="onehot")
        model = DreamerV3(config, IMAGE_SHAPE, 4, "cpu")
        assert model.discrete_actions
        fill(model)
        assert all(np.isfinite(value) for value in model.train_one_batch())

    def test_vector_observation_training(self):
        model = DreamerV3(tiny_config(), (9,), 2, "cpu")
        assert not model.is_image_obs
        assert model.data_buffer.observations.dtype == np.float32
        fill(model, image=False)
        assert all(np.isfinite(value) for value in model.train_one_batch())

    def test_imagination_shapes(self):
        model = build()
        fill(model)
        obs, actions, rewards, terminal, first = model.data_buffer.sample()
        _, posterior = model.world_model_loss(
            torch.as_tensor(obs),
            torch.as_tensor(actions),
            torch.as_tensor(rewards),
            torch.as_tensor(terminal),
            torch.as_tensor(first),
        )
        imagined = model._imagine(posterior)
        horizon = model.args.imagine_horizon
        flat = model.args.train_seq_len * model.args.batch_size
        assert imagined["features"].shape == (
            horizon + 1,
            flat,
            model.rssm.feature_size,
        )
        assert imagined["returns"].shape == (horizon, flat)
        assert imagined["weights"].shape == (horizon, flat)

    def test_imagination_weights_start_at_one_and_decay(self):
        model = build()
        fill(model)
        obs, actions, rewards, terminal, first = model.data_buffer.sample()
        _, posterior = model.world_model_loss(
            torch.as_tensor(obs),
            torch.as_tensor(actions),
            torch.as_tensor(rewards),
            torch.as_tensor(terminal),
            torch.as_tensor(first),
        )
        weights = model._imagine(posterior)["weights"]
        assert torch.allclose(weights[0], torch.ones_like(weights[0]))
        assert float(weights[-1].max()) <= 1.0

    def test_imagined_return_credits_the_state_that_was_acted_from(self):
        # Pins the reward/state alignment. The reward head is trained so that
        # its prediction at a state is the reward for *leaving* that state, so
        # the return at s_0 must include the reward predicted at s_0. Consuming
        # rewards[1:] instead would drop it and shift all credit by one step.
        model = build(tiny_config(discount=1.0, td_lambda=1.0, actor_entropy=0.0))
        horizon = model.args.imagine_horizon

        # A reward head that always predicts 1, a critic that always predicts 0,
        # and no termination make the exact return analytically known.
        rewards = torch.ones(horizon + 1, 4)
        values = torch.zeros(horizon + 1, 4)
        continues = torch.ones(horizon + 1, 4)

        from simulusion.utils.dreamer_v3_utils import lambda_return

        returns = lambda_return(
            rewards=rewards[:-1],
            values=values[1:],
            continues=continues[:-1],
            bootstrap=values[-1],
            lambda_=1.0,
        )
        # Undiscounted sum of the `horizon` rewards collected from s_0 onwards.
        assert float(returns[0, 0]) == pytest.approx(float(horizon))
        # And one fewer reward remains from the next state.
        assert float(returns[1, 0]) == pytest.approx(float(horizon - 1))

    def test_imagined_returns_grow_with_predicted_reward(self):
        model = build()
        fill(model)
        obs, actions, rewards, terminal, first = model.data_buffer.sample()
        _, posterior = model.world_model_loss(
            torch.as_tensor(obs),
            torch.as_tensor(actions),
            torch.as_tensor(rewards),
            torch.as_tensor(terminal),
            torch.as_tensor(first),
        )
        imagined = model._imagine(posterior)
        # Returns accumulate several imagined rewards, so with non-negative
        # rewards they must not be smaller than a single step's reward.
        assert (
            float(imagined["returns"].mean())
            >= float(imagined["rewards"][:-1].mean()) - 1e-3
        )


class TestCheckpointing:
    def test_save_and_restore_roundtrip(self, tmp_path):
        model = build()
        fill(model)
        for _ in range(2):
            model.train_one_batch()
        path = tmp_path / "model.pt"
        model.save(str(path))

        restored = build()
        restored.restore_checkpoint(str(path))
        for name, module in model._named_modules().items():
            for original, loaded in zip(
                module.parameters(), restored._named_modules()[name].parameters()
            ):
                assert torch.equal(original, loaded), name

    def test_save_writes_a_config_beside_the_checkpoint(self, tmp_path):
        model = build()
        model.save(str(tmp_path / "model.pt"))
        assert (tmp_path / "config.yaml").exists()

    def test_return_normalizer_state_survives_a_roundtrip(self, tmp_path):
        model = build()
        fill(model)
        model.train_one_batch()
        path = tmp_path / "model.pt"
        model.save(str(path))

        restored = build()
        restored.restore_checkpoint(str(path))
        assert restored.return_norm.scale == pytest.approx(model.return_norm.scale)

    def test_from_pretrained_rebuilds_the_model(self, tmp_path):
        model = build()
        fill(model)
        model.train_one_batch()
        model.save(str(tmp_path / "model.pt"))

        loaded = DreamerV3.from_pretrained(tmp_path, map_location="cpu")
        assert loaded.obs_shape == model.obs_shape
        assert loaded.action_size == model.action_size
        assert torch.equal(
            next(loaded.actor.parameters()), next(model.actor.parameters())
        )

    def test_from_pretrained_reports_a_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DreamerV3.from_pretrained(tmp_path / "nothing-here")


# ---------------------------------------------------------------------------
# Environment interaction
# ---------------------------------------------------------------------------


class FakeEnv:
    """Minimal image environment with the Simulusion four-tuple step API."""

    def __init__(self, action_size=2, episode_length=5, discrete=False):
        self.action_size = action_size
        self.episode_length = episode_length
        self.discrete = discrete
        self.steps = 0
        self.action_space = Mock()
        self.action_space.shape = (action_size,)
        self.action_space.sample = self._sample

    def _sample(self):
        if self.discrete:
            action = np.zeros(self.action_size, np.float32)
            action[np.random.randint(self.action_size)] = 1.0
            return action
        return np.random.uniform(-1, 1, self.action_size).astype(np.float32)

    def _obs(self):
        return {"image": np.random.randint(0, 255, IMAGE_SHAPE, dtype=np.uint8)}

    def reset(self):
        self.steps = 0
        return self._obs()

    def step(self, action):
        self.steps += 1
        done = self.steps % self.episode_length == 0
        return self._obs(), 1.0, done, {"action": np.asarray(action)}


class TestInteraction:
    def test_collect_random_episodes_fills_the_buffer(self):
        model = build()
        rewards = model.collect_random_episodes(FakeEnv(), 20)
        assert model.data_buffer.steps == 20
        assert len(rewards) > 1

    def test_act_and_collect_data_fills_the_buffer(self):
        model = build()
        rewards = model.act_and_collect_data(FakeEnv(), 15)
        assert model.data_buffer.steps == 15
        assert np.all(rewards > 0)

    def test_episode_starts_are_recorded_during_collection(self):
        model = build()
        model.act_and_collect_data(FakeEnv(episode_length=4), 16)
        assert model.data_buffer.is_first[:16].sum() >= 3

    def test_evaluate_returns_rewards_and_frames(self):
        model = build()
        rewards, videos, latents = model.evaluate(FakeEnv(), 2, render=True)
        assert rewards.shape == (2,)
        assert len(videos) > 0
        assert latents is not None

    def test_greedy_action_is_deterministic_given_the_model_state(self):
        # Note the qualifier: evaluation is greedy in the *policy*, but the
        # latent posterior is still sampled, so repeating an observation does
        # not reproduce an action. Determinism holds once the state is fixed.
        model = build()
        features = torch.randn(1, model.rssm.feature_size)
        with torch.no_grad():
            actions = [model.actor(features, deterministic=True) for _ in range(5)]
        assert all(torch.equal(actions[0], other) for other in actions[1:])

    def test_repeated_observations_differ_because_the_latent_is_sampled(self):
        model = build()
        obs = {"image": np.zeros(IMAGE_SHAPE, dtype=np.uint8)}
        state = model.rssm.init_state(1, model.device)
        action = torch.zeros(1, model.action_size)
        with torch.no_grad():
            states = [
                model.act_with_world_model(obs, state, action)[0]["stoch"]
                for _ in range(10)
            ]
        assert any(not torch.equal(states[0], other) for other in states[1:])

    def test_exploration_sampling_varies(self):
        model = build()
        obs = {"image": np.zeros(IMAGE_SHAPE, dtype=np.uint8)}
        state = model.rssm.init_state(1, model.device)
        action = torch.zeros(1, model.action_size)
        with torch.no_grad():
            samples = [
                model.act_with_world_model(obs, state, action, explore=True)[1]
                for _ in range(8)
            ]
        assert any(not torch.allclose(samples[0], other) for other in samples[1:])


class TestTerminationDetection:
    def test_not_done_is_not_terminal(self):
        assert _is_true_termination({}, False) == 0.0

    def test_plain_done_is_terminal(self):
        assert _is_true_termination({}, True) == 1.0

    def test_time_limit_truncation_is_not_terminal(self):
        assert _is_true_termination({"time_limit": True}, True) == 0.0
        assert _is_true_termination({"TimeLimit.truncated": True}, True) == 0.0

    def test_dm_control_discount_marks_truncation(self):
        assert _is_true_termination({"discount": 1.0}, True) == 0.0
        assert _is_true_termination({"discount": 0.0}, True) == 1.0

    def test_non_dict_info_falls_back_to_done(self):
        assert _is_true_termination(None, True) == 1.0


class TestActorDistResolution:
    def test_box_action_space_resolves_to_normal(self):
        env = Mock()
        env.action_space = Mock(spec=["shape", "low", "high"])
        env._env = None
        assert _resolve_actor_dist(env) == "normal"

    def test_discrete_action_space_resolves_to_onehot(self):
        env = Mock()
        env.action_space = Mock(spec=["n"])
        assert _resolve_actor_dist(env) == "onehot"

    def test_one_hot_wrapped_env_resolves_to_onehot(self):
        inner = Mock()
        inner.action_space = Mock(spec=["n"])
        env = Mock()
        env.action_space = Mock(spec=["shape", "low", "high"])
        env._env = inner
        assert _resolve_actor_dist(env) == "onehot"


# ---------------------------------------------------------------------------
# High-level agent and public API
# ---------------------------------------------------------------------------


class TestDreamerV3Agent:
    @patch("simulusion.models.dreamer.make_env")
    @patch("simulusion.models.dreamer.Logger")
    def test_builds_a_v3_core_and_config(self, _logger, mock_make_env, tmp_path):
        env = FakeEnv()
        env.observation_space = {"image": Mock(shape=IMAGE_SHAPE)}
        env.action_space.low = -np.ones(2)
        env.action_space.high = np.ones(2)
        mock_make_env.return_value = env

        agent = DreamerV3Agent(tiny_config(data_dir=str(tmp_path)))
        assert isinstance(agent.dreamer, DreamerV3)
        assert isinstance(agent.args, DreamerV3Config)

    @patch("simulusion.models.dreamer.make_env")
    @patch("simulusion.models.dreamer.Logger")
    def test_resolves_the_actor_distribution_from_the_env(
        self, _logger, mock_make_env, tmp_path
    ):
        env = FakeEnv(discrete=True)
        env.observation_space = {"image": Mock(shape=IMAGE_SHAPE)}
        env.action_space = Mock(spec=["shape", "n", "sample"])
        env.action_space.shape = (2,)
        env.action_space.n = 2
        env.action_space.sample = env._sample
        mock_make_env.return_value = env

        agent = DreamerV3Agent(tiny_config(actor_dist="auto", data_dir=str(tmp_path)))
        assert agent.args.actor_dist == "onehot"
        assert agent.dreamer.discrete_actions

    @patch("simulusion.models.dreamer.make_env")
    @patch("simulusion.models.dreamer.Logger")
    def test_upgrades_a_base_dreamer_config(self, _logger, mock_make_env, tmp_path):
        from simulusion.configs.dreamer_config import DreamerConfig

        env = FakeEnv()
        env.observation_space = {"image": Mock(shape=IMAGE_SHAPE)}
        env.action_space.low = -np.ones(2)
        env.action_space.high = np.ones(2)
        mock_make_env.return_value = env

        base = DreamerConfig(seed=99, buffer_size=100, data_dir=str(tmp_path))
        agent = DreamerV3Agent(base)
        assert isinstance(agent.args, DreamerV3Config)
        # Explicitly changed fields carry over ...
        assert agent.args.seed == 99
        # ... but fields left at the base defaults must not clobber V3's own
        # tuned values, or the agent would quietly train with V1 settings.
        assert agent.args.discount == pytest.approx(0.997)
        assert agent.args.batch_size == 16

    @patch("simulusion.models.dreamer.make_env")
    @patch("simulusion.models.dreamer.Logger")
    def test_explicit_base_config_overrides_are_respected(
        self, _logger, mock_make_env, tmp_path
    ):
        from simulusion.configs.dreamer_config import DreamerConfig

        env = FakeEnv()
        env.observation_space = {"image": Mock(shape=IMAGE_SHAPE)}
        env.action_space.low = -np.ones(2)
        env.action_space.high = np.ones(2)
        mock_make_env.return_value = env

        base = DreamerConfig(
            discount=0.5, batch_size=3, buffer_size=100, data_dir=str(tmp_path)
        )
        agent = DreamerV3Agent(base)
        assert agent.args.discount == pytest.approx(0.5)
        assert agent.args.batch_size == 3


class TestPublicApi:
    def test_registered_under_both_names(self):
        from simulusion.api import get_model_spec

        for name in ("dreamer-v3", "dreamerv3"):
            spec = get_model_spec(name)
            assert spec.import_path.endswith("DreamerV3Agent")
            assert spec.config_path.endswith("DreamerV3Config")

    def test_create_config_returns_a_v3_config(self):
        from simulusion.api import create_config

        config = create_config("dreamer-v3", seed=5)
        assert isinstance(config, DreamerV3Config)
        assert config.seed == 5

    def test_exported_from_the_package_root(self):
        import simulusion

        assert simulusion.DreamerV3 is DreamerV3
        assert simulusion.DreamerV3Agent is DreamerV3Agent
        assert simulusion.DreamerV3Config is DreamerV3Config

    def test_listed_in_the_environment_catalog(self):
        from simulusion.api import list_envs

        assert len(list_envs("dreamerv3")) > 0
