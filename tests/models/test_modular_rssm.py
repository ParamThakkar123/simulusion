import pytest
import torch
from simulusion.models.modular_rssm import (
    ModularRSSM,
    create_modular_rssm,
    ConvEncoder,
    MLPEncoder,
    ViTEncoder,
    ConvDecoder,
    MLPDecoder,
    GRUBackbone,
    LSTMBackbone,
    TransformerBackbone,
)


class TestEncoders:
    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def obs_image(self, batch_size):
        return torch.randn(batch_size, 3, 64, 64)

    @pytest.fixture
    def obs_state(self, batch_size):
        return torch.randn(batch_size, 10)

    def test_conv_encoder_forward(self, obs_image):
        encoder = ConvEncoder(input_shape=(3, 64, 64), embed_size=256)
        embed = encoder(obs_image)
        assert embed.shape == (4, 256)

    def test_conv_encoder_embed_size(self):
        encoder = ConvEncoder(input_shape=(3, 64, 64), embed_size=512)
        assert encoder.embed_size == 512

    def test_mlp_encoder_forward(self, obs_state):
        encoder = MLPEncoder(input_dim=10, embed_size=64)
        embed = encoder(obs_state)
        assert embed.shape == (4, 64)

    def test_vit_encoder_forward(self, obs_image):
        encoder = ViTEncoder(
            input_shape=(3, 64, 64), embed_size=128, patch_size=8, depth=2
        )
        embed = encoder(obs_image)
        assert embed.shape == (4, 128)


class TestDecoders:
    @pytest.fixture
    def features(self):
        return torch.randn(4, 32 + 200)

    def test_conv_decoder_forward(self, features):
        decoder = ConvDecoder(stoch_size=32, deter_size=200, output_shape=(3, 64, 64))
        dist = decoder(features)
        # For Independent distributions, batch_shape excludes event dimensions
        assert dist.batch_shape == torch.Size([4])
        assert dist.event_shape == torch.Size([3, 64, 64])

    def test_mlp_decoder_forward(self, features):
        decoder = MLPDecoder(stoch_size=32, deter_size=200, output_dim=1)
        out = decoder(features)
        assert out.batch_shape == (4,)


class TestBackbones:
    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def state(self, batch_size):
        return {
            "mean": torch.randn(batch_size, 32),
            "std": torch.ones(batch_size, 32) * 0.1,
            "stoch": torch.randn(batch_size, 32),
            "deter": torch.randn(batch_size, 200),
        }

    @pytest.fixture
    def action(self, batch_size):
        return torch.randn(batch_size, 6)

    def test_gru_backbone_init_state(self, batch_size):
        backbone = GRUBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=200,
            hidden_size=200,
            embed_size=1024,
        )
        state = backbone.init_state(batch_size, torch.device("cpu"))
        assert state["mean"].shape == (batch_size, 32)
        assert state["deter"].shape == (batch_size, 200)

    def test_gru_backbone_forward(self, state, action):
        backbone = GRUBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=200,
            hidden_size=200,
            embed_size=1024,
        )
        prior, posterior = backbone.forward(state, action, obs_embed=None)
        assert prior["stoch"].shape == (4, 32)
        assert posterior["stoch"].shape == (4, 32)

    def test_lstm_backbone_init_state(self, batch_size):
        backbone = LSTMBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=200,
            hidden_size=200,
            embed_size=1024,
        )
        state = backbone.init_state(batch_size, torch.device("cpu"))
        assert "cell" in state
        assert state["cell"].shape == (batch_size, 200)

    def test_lstm_backbone_forward(self, state, action):
        backbone = LSTMBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=200,
            hidden_size=200,
            embed_size=1024,
        )
        prior, posterior = backbone.forward(state, action, obs_embed=None)
        assert prior["stoch"].shape == (4, 32)

    def test_transformer_backbone_init_state(self, batch_size):
        backbone = TransformerBackbone(
            action_size=6, stoch_size=32, deter_size=200, embed_size=128
        )
        state = backbone.init_state(batch_size, torch.device("cpu"))
        assert state["mean"].shape == (batch_size, 32)

    def test_transformer_backbone_forward(self, state, action):
        backbone = TransformerBackbone(
            action_size=6, stoch_size=32, deter_size=200, embed_size=128, num_layers=2
        )
        prior, posterior = backbone.forward(state, action, obs_embed=None)
        assert prior["stoch"].shape == (4, 32)


class TestModularRSSM:
    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def obs(self, batch_size):
        return torch.randn(batch_size, 3, 64, 64)

    @pytest.fixture
    def actions(self):
        return torch.randn(8, 6)

    @pytest.fixture
    def nonterms(self):
        return torch.ones(8)

    @pytest.fixture
    def rssm(self):
        encoder = ConvEncoder(input_shape=(3, 64, 64), embed_size=256)
        decoder = ConvDecoder(stoch_size=32, deter_size=128, output_shape=(3, 64, 64))
        backbone = GRUBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=128,
            hidden_size=128,
            embed_size=256,
        )
        reward_decoder = MLPDecoder(stoch_size=32, deter_size=128, output_dim=1)
        return ModularRSSM(encoder, decoder, backbone, reward_decoder)

    def test_rssm_properties(self, rssm):
        assert rssm.stoch_size == 32
        assert rssm.deter_size == 128

    def test_rssm_init_state(self, rssm, batch_size):
        state = rssm.init_state(batch_size, torch.device("cpu"))
        assert state["mean"].shape == (batch_size, 32)

    def test_rssm_observe_step(self, rssm, batch_size):
        obs = torch.randn(batch_size, 3, 64, 64)
        prev_state = rssm.init_state(batch_size, torch.device("cpu"))
        action = torch.randn(batch_size, 6)
        prior, posterior = rssm.observe_step(prev_state, action, obs)
        assert prior["stoch"].shape == (batch_size, 32)
        assert posterior["stoch"].shape == (batch_size, 32)

    def test_rssm_imagine_step(self, rssm, batch_size):
        prev_state = rssm.init_state(batch_size, torch.device("cpu"))
        action = torch.randn(batch_size, 6)
        prior = rssm.imagine_step(prev_state, action)
        assert prior["stoch"].shape == (batch_size, 32)

    def test_rssm_observe_rollout(self, rssm, batch_size):
        obs = torch.randn(8, batch_size, 3, 64, 64)
        actions = torch.randn(8, batch_size, 6)
        nonterms = torch.ones(8, batch_size)
        prev_state = rssm.init_state(batch_size, torch.device("cpu"))

        priors, posteriors = rssm.observe_rollout(obs, actions, nonterms, prev_state, 8)
        assert priors["stoch"].shape == (8, batch_size, 32)

    def test_rssm_decode_observation(self, rssm, batch_size):
        features = torch.randn(batch_size, 32 + 128)
        dist = rssm.decode_observation(features)
        # For Independent distributions, batch_shape excludes event dimensions
        assert dist.batch_shape == torch.Size([batch_size])
        assert dist.event_shape == torch.Size([3, 64, 64])

    def test_rssm_decode_reward(self, rssm, batch_size):
        features = torch.randn(batch_size, 32 + 128)
        dist = rssm.decode_reward(features)
        assert dist.batch_shape == (batch_size,)

    def test_rssm_get_dist(self, rssm):
        mean = torch.randn(4, 32)
        std = torch.ones(4, 32) * 0.1
        dist = rssm.get_dist(mean, std)
        assert dist.batch_shape == (4,)

    def test_rssm_detach_state(self, rssm, batch_size):
        state = rssm.init_state(batch_size, torch.device("cpu"))
        detached = rssm.detach_state(state)
        assert detached["mean"].requires_grad is False


class TestCreateModularRSSM:
    def test_create_conv_gru(self):
        rssm = create_modular_rssm(
            encoder_type="conv",
            decoder_type="conv",
            backbone_type="gru",
            obs_shape=(3, 64, 64),
            action_size=6,
            stoch_size=32,
            deter_size=128,
            embed_size=256,
        )
        assert isinstance(rssm, ModularRSSM)

    def test_create_mlp_lstm(self):
        rssm = create_modular_rssm(
            encoder_type="mlp",
            decoder_type="mlp",
            backbone_type="lstm",
            obs_shape=(10,),
            action_size=6,
            stoch_size=32,
            deter_size=128,
            embed_size=64,
        )
        assert isinstance(rssm, ModularRSSM)

    def test_create_vit_transformer(self):
        rssm = create_modular_rssm(
            encoder_type="vit",
            decoder_type="conv",
            backbone_type="transformer",
            obs_shape=(3, 64, 64),
            action_size=6,
            stoch_size=32,
            deter_size=128,
            embed_size=128,
            num_heads=4,
            num_layers=2,
        )
        assert isinstance(rssm, ModularRSSM)

    def test_invalid_encoder_type(self):
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_modular_rssm(
                encoder_type="invalid",
                decoder_type="conv",
                backbone_type="gru",
                obs_shape=(3, 64, 64),
                action_size=6,
            )

    def test_invalid_backbone_type(self):
        with pytest.raises(ValueError, match="Unknown backbone type"):
            create_modular_rssm(
                encoder_type="conv",
                decoder_type="conv",
                backbone_type="invalid",
                obs_shape=(3, 64, 64),
                action_size=6,
            )


class TestModularRSSMWithActor:
    @pytest.fixture
    def rssm_with_mock_actor(self):
        encoder = ConvEncoder(input_shape=(3, 64, 64), embed_size=256)
        decoder = ConvDecoder(stoch_size=32, deter_size=128, output_shape=(3, 64, 64))
        backbone = GRUBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=128,
            hidden_size=128,
            embed_size=256,
        )
        reward_decoder = MLPDecoder(stoch_size=32, deter_size=128, output_dim=1)
        return ModularRSSM(encoder, decoder, backbone, reward_decoder)

    def test_imagine_rollout_with_actor(self, rssm_with_mock_actor):
        batch_size = 4
        horizon = 5

        class MockActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(32 + 128, 6)

            def forward(self, features):
                return torch.tanh(self.net(features))

        actor = MockActor()
        prev_state = rssm_with_mock_actor.init_state(batch_size, torch.device("cpu"))

        imagined_states = rssm_with_mock_actor.imagine_rollout(
            actor, prev_state, horizon
        )
        assert imagined_states["stoch"].shape == (horizon, batch_size, 32)


class TestModularRSSMSwapComponents:
    def test_swap_conv_to_vit_encoder(self):
        encoder_vit = ViTEncoder(
            input_shape=(3, 64, 64), embed_size=256, patch_size=8, depth=2
        )
        decoder = ConvDecoder(stoch_size=32, deter_size=128, output_shape=(3, 64, 64))
        backbone = GRUBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=128,
            hidden_size=128,
            embed_size=256,
        )

        rssm = ModularRSSM(encoder_vit, decoder, backbone)

        obs = torch.randn(4, 3, 64, 64)
        state = rssm.init_state(4, torch.device("cpu"))
        action = torch.randn(4, 6)

        prior, posterior = rssm.observe_step(state, action, obs)
        assert prior["stoch"].shape == (4, 32)

    def test_swap_gru_to_lstm_backbone(self):
        encoder = ConvEncoder(input_shape=(3, 64, 64), embed_size=256)
        decoder = ConvDecoder(stoch_size=32, deter_size=128, output_shape=(3, 64, 64))
        backbone_lstm = LSTMBackbone(
            action_size=6,
            stoch_size=32,
            deter_size=128,
            hidden_size=128,
            embed_size=256,
        )

        rssm = ModularRSSM(encoder, decoder, backbone_lstm)

        state = rssm.init_state(4, torch.device("cpu"))
        action = torch.randn(4, 6)
        obs = torch.randn(4, 3, 64, 64)

        prior, posterior = rssm.observe_step(state, action, obs)
        assert "cell" in posterior
        assert posterior["cell"].shape == (4, 128)
