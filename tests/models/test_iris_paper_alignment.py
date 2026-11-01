"""Checks that the IRIS implementation matches the paper's specification.

These pin architectural details from Micheli et al. (ICLR 2023) Appendix A that
are easy to drift from silently, because a "reasonable" substitute still trains
and still produces plausible loss curves.
"""

import pytest
import torch
import torch.nn as nn

from simulusion.configs.iris_config import IRISConfig
from simulusion.controller.iris_policy import CNNFeatureExtractor
from simulusion.vision.iris_decoder import IRISDecoder
from simulusion.vision.iris_encoder import IRISEncoder
from simulusion.training.train_iris import (
    FREEWAY_COLLECT_TEMPERATURE,
    default_collect_temperature,
)


class TestActorCriticConvBlock:
    """Paper A.3: 4x [3x3 conv stride 1 pad 1 -> ReLU -> 2x2 max-pool stride 2]."""

    def test_layer_pattern(self):
        cnn = CNNFeatureExtractor()
        ops = [type(m).__name__ for m in cnn.conv]
        assert ops == ["Conv2d", "ReLU", "MaxPool2d"] * 4, ops

    def test_convolutions_are_stride_one(self):
        """Downsampling comes from max-pooling, not from strided convolution.

        A stride-2 convolution can step over a two-pixel sprite entirely; the
        max-pool keeps the strongest activation in each window instead.
        """
        cnn = CNNFeatureExtractor()
        for module in cnn.conv:
            if isinstance(module, nn.Conv2d):
                assert module.stride == (1, 1)
                assert module.padding == (1, 1)
                assert module.kernel_size == (3, 3)
            if isinstance(module, nn.MaxPool2d):
                assert module.kernel_size == 2
                assert module.stride == 2

    def test_output_shape(self):
        cnn = CNNFeatureExtractor(output_size=512)
        assert cnn(torch.rand(2, 3, 64, 64)).shape == (2, 512)


class TestAutoencoderTable2:
    """Table 2: 4 layers, 2 residual blocks per layer, 64 channels, both halves."""

    def test_encoder_has_residuals_per_layer(self):
        encoder = IRISEncoder(num_residual_blocks=2)
        assert len(encoder.conv_blocks) == 4
        assert len(encoder.layer_residuals) == 4
        assert all(len(stack) == 2 for stack in encoder.layer_residuals)

    def test_decoder_has_residuals_per_layer(self):
        decoder = IRISDecoder(base_channels=64, num_residual_blocks=2)
        assert len(decoder.upsample_blocks) == 4
        assert len(decoder.layer_residuals) == 4
        assert all(len(stack) == 2 for stack in decoder.layer_residuals)

    def test_decoder_channels_default_to_64(self):
        assert IRISConfig().decoder_depth == 64

    def test_encoder_width_is_constant(self):
        """Table 2 lists one "Channels in convolutions: 64" for the whole stack.

        Doubling per layer (64/128/256/512) is the VQGAN default with a
        non-trivial ``ch_mult``; IRIS's is all ones.
        """
        encoder = IRISEncoder(base_channels=64)
        widths = [block[0].out_channels for block in encoder.conv_blocks]
        assert widths == [64, 64, 64, 64]

    def test_decoder_width_is_constant(self):
        """ "the same ones apply for the decoder" -- including the bottleneck.

        The stack that runs before upsampling sits at the convolutional width,
        not at the 512-d token embedding; running it at 512 put ~5M parameters
        into two residual blocks.
        """
        decoder = IRISDecoder(embedding_dim=512, base_channels=64)
        assert decoder.input_proj.out_channels == 64
        widths = [block.block[1].out_channels for block in decoder.upsample_blocks]
        assert widths == [64, 64, 64, 64]

    def test_autoencoder_halves_are_comparable_in_size(self):
        """A constant-width decoder should not dwarf its encoder."""
        encoder = IRISEncoder(base_channels=64, embedding_dim=512)
        decoder = IRISDecoder(base_channels=64, embedding_dim=512)
        enc_params = sum(p.numel() for p in encoder.parameters())
        dec_params = sum(p.numel() for p in decoder.parameters())
        assert dec_params < 2 * enc_params, (enc_params, dec_params)

    @staticmethod
    def _attention_resolutions(module, forward):
        """Record the spatial size each self-attention block actually receives.

        Reads the live module rather than trusting attribute names, so this keeps
        testing the real behaviour if the blocks are reorganised.
        """
        seen: dict[str, tuple[int, int]] = {}

        def hook(name):
            def record(_module, inputs, _output):
                seen[name] = tuple(inputs[0].shape[-2:])

            return record

        handles = [
            block.register_forward_hook(hook(name))
            for name, block in module.attentions.items()
        ]
        assert handles, "module exposes no self-attention blocks"
        try:
            forward()
        finally:
            for handle in handles:
                handle.remove()
        return set(seen.values())

    def test_encoder_self_attention_at_8_and_16(self):
        encoder = IRISEncoder(vocab_size=32, embedding_dim=64, base_channels=16)
        resolutions = self._attention_resolutions(
            encoder, lambda: encoder(torch.rand(2, 3, 64, 64))
        )
        assert resolutions == {(8, 8), (16, 16)}, resolutions

    def test_decoder_self_attention_at_8_and_16(self):
        """Table 2's "same ones apply for the decoder" includes self-attention."""
        decoder = IRISDecoder(embedding_dim=64, base_channels=16)
        resolutions = self._attention_resolutions(
            decoder, lambda: decoder(torch.rand(2, 64, 4, 4))
        )
        assert resolutions == {(8, 8), (16, 16)}, resolutions

    def test_encoder_decoder_round_trip(self):
        config = IRISConfig()
        encoder = IRISEncoder(
            vocab_size=32,
            embedding_dim=64,
            base_channels=16,
            num_residual_blocks=config.encoder_residual_blocks,
        )
        decoder = IRISDecoder(
            embedding_dim=64,
            base_channels=16,
            num_residual_blocks=config.encoder_residual_blocks,
        )
        z_q, indices, _ = encoder(torch.rand(2, 3, 64, 64))
        assert indices.shape == (2, 4, 4)
        assert decoder(z_q).shape == (2, 3, 64, 64)


class TestFreewayExploration:
    """Appendix H: the collection sampling temperature drops to 0.01 on Freeway."""

    def test_freeway_lowers_temperature(self):
        assert (
            default_collect_temperature("ALE/Freeway-v5", 1.0)
            == FREEWAY_COLLECT_TEMPERATURE
        )

    def test_other_games_unaffected(self):
        assert default_collect_temperature("ALE/Pong-v5", 1.0) == 1.0

    def test_explicit_setting_wins(self):
        """An explicit config value must not be silently overridden."""
        assert default_collect_temperature("ALE/Freeway-v5", 0.7) == 0.7


class TestTransformerLossWeighting:
    """Paper 2.2 lists the three world-model losses without relative weights."""

    def test_losses_are_summed_unweighted(self):
        config = IRISConfig()
        config.vocab_size = 32
        config.token_embedding_dim = 64
        config.encoder_channels = 16
        config.decoder_depth = 8
        config.transformer_layers = 2
        config.transformer_embed_dim = 64
        config.transformer_timesteps = 4
        config.perceptual_weight = 0.0

        from simulusion.models.iris_agent import IRISAgent

        agent = IRISAgent(config, action_size=4, device=torch.device("cpu"))
        b, t = 2, config.transformer_timesteps
        metrics = agent.update_transformer(
            torch.rand(b, t + 1, 3, 64, 64),
            torch.nn.functional.one_hot(torch.randint(0, 4, (b, t)), 4).float(),
            torch.zeros(b, t),
            torch.zeros(b, t, dtype=torch.long),
        )
        expected = metrics["token_loss"] + metrics["reward_loss"] + metrics["term_loss"]
        assert metrics["total_loss"] == pytest.approx(expected, rel=1e-5)


class TestRewardHandling:
    """Paper 2.2: MSE *or* cross-entropy "depending on the reward function".

    Atari returns unbounded integer rewards, so IRIS signs them, making the
    target categorical over {-1, 0, +1} and cross-entropy the right loss.
    """

    @staticmethod
    def _agent(**overrides):
        from simulusion.models.iris_agent import IRISAgent

        config = IRISConfig()
        config.vocab_size = 32
        config.token_embedding_dim = 64
        config.encoder_channels = 16
        config.decoder_depth = 16
        config.transformer_layers = 2
        config.transformer_embed_dim = 64
        config.transformer_timesteps = 4
        config.perceptual_weight = 0.0
        for key, value in overrides.items():
            setattr(config, key, value)
        return IRISAgent(config, action_size=4, device=torch.device("cpu"))

    def test_sign_transform_bounds_rewards(self):
        agent = self._agent()
        raw = torch.tensor([[0.0, 5.0, -3.0, 1200.0]])
        assert agent.transform_reward(raw).tolist() == [[0.0, 1.0, -1.0, 1.0]]

    def test_transform_can_be_disabled(self):
        agent = self._agent(reward_transform="none", reward_loss="mse")
        raw = torch.tensor([[0.0, 5.0, -3.0]])
        assert torch.equal(agent.transform_reward(raw), raw)

    def test_unknown_transform_rejected(self):
        agent = self._agent()
        agent.config.reward_transform = "bogus"
        with pytest.raises(ValueError, match="reward_transform"):
            agent.transform_reward(torch.zeros(1, 2))

    def test_categorical_head_has_three_classes(self):
        assert self._agent().transformer.reward_classes == 3

    def test_scalar_head_when_using_mse(self):
        agent = self._agent(reward_transform="none", reward_loss="mse")
        assert agent.transformer.reward_classes == 1

    def test_imagined_rewards_stay_in_support(self):
        """The expectation over {-1,0,+1} can never leave [-1, 1]."""
        agent = self._agent()
        with torch.no_grad():
            trajectory = agent.imagine_rollout(
                torch.rand(2, 3, 64, 64), horizon=4, stop_on_termination=False
            )
        rewards = trajectory["rewards"]
        assert float(rewards.min()) >= -1.0 and float(rewards.max()) <= 1.0

    def test_large_raw_rewards_do_not_blow_up_the_loss(self):
        """A reward of 1200 must not produce a huge loss the way MSE would."""
        agent = self._agent()
        b, t = 2, agent.config.transformer_timesteps
        rewards = torch.full((b, t), 1200.0)
        metrics = agent.update_transformer(
            torch.rand(b, t + 1, 3, 64, 64),
            torch.nn.functional.one_hot(torch.randint(0, 4, (b, t)), 4).float(),
            rewards,
            torch.zeros(b, t, dtype=torch.long),
        )
        # Cross-entropy over 3 classes starts near ln(3) regardless of scale.
        assert metrics["reward_loss"] < 10.0


class TestPerceptualLossStructure:
    """A.1 inherits VQGAN's LPIPS perceptual term."""

    @pytest.fixture(scope="class")
    def loss(self):
        from simulusion.vision.perceptual_loss import build_perceptual_loss

        module = build_perceptual_loss(enabled=True, num_blocks=5)
        if module is None:
            pytest.skip("torchvision VGG16 weights unavailable")
        return module

    def test_uses_five_vgg_blocks(self, loss):
        assert len(loss.blocks) == 5

    def test_zero_on_identical_images(self, loss):
        images = torch.rand(2, 3, 64, 64)
        assert float(loss(images, images)) == pytest.approx(0.0, abs=1e-6)

    def test_positive_and_differentiable(self, loss):
        target = torch.rand(2, 3, 64, 64)
        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        value = loss(target, pred)
        value.backward()
        assert float(value.detach()) > 0.0
        assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0

    def test_backbone_contributes_no_trainable_parameters(self, loss):
        """The frozen VGG must not end up in the autoencoder optimizer."""
        assert sum(1 for p in loss.parameters() if p.requires_grad) == 0

    def test_stays_in_eval_mode(self, loss):
        loss.train(True)
        assert not loss.training
