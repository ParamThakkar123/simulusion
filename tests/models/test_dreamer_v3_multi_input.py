"""DreamerV3 with several observation inputs (images and vectors together)."""

import gc

import numpy as np
import pytest
import torch

gym = pytest.importorskip("gymnasium")

from simulusion.configs.dreamer_v3_config import DreamerV3Config  # noqa: E402
from simulusion.models.dreamer_v3 import DreamerV3, select_obs_space  # noqa: E402
from simulusion.utils.dreamer_utils import symlog  # noqa: E402
from simulusion.vision.dreamer_v3_nets import (  # noqa: E402
    DreamerV3MultiDecoder,
    DreamerV3MultiEncoder,
    split_obs_space,
)

IMAGE = (3, 16, 16)
MIXED = {"image": IMAGE, "instr": (8,)}


@pytest.fixture(autouse=True)
def _release_models():
    yield
    gc.collect()


def tiny_config(**overrides):
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


def box(shape, dtype=np.float32, low=-np.inf, high=np.inf):
    return gym.spaces.Box(low, high, shape, dtype=dtype)


def dict_space(**shapes):
    spaces = {}
    for key, shape in shapes.items():
        if len(shape) == 3:
            spaces[key] = box(shape, np.uint8, 0, 255)
        else:
            spaces[key] = box(shape)
    return gym.spaces.Dict(spaces)


def observation(space, rng):
    obs = {}
    for key, shape in space.items():
        if len(shape) == 3:
            obs[key] = rng.integers(0, 255, shape, dtype=np.uint8)
        else:
            obs[key] = rng.normal(0, 50, shape).astype(np.float32)
    return obs


class FakeEnv:
    """Environment producing every key of ``space`` plus an unused extra."""

    def __init__(self, space, episode_length=5):
        self.space = space
        self.episode_length = episode_length
        self.steps = 0
        self.rng = np.random.default_rng(0)
        self.action_space = box((2,), low=-1.0, high=1.0)
        self.renders = 0

    def _obs(self):
        obs = observation(self.space, self.rng)
        obs["unused"] = np.zeros(4, np.float32)
        return obs

    def reset(self):
        self.steps = 0
        return self._obs()

    def step(self, action):
        self.steps += 1
        done = self.steps % self.episode_length == 0
        info = {"action": np.asarray(action), "terminated": done, "truncated": False}
        return self._obs(), 1.0, done, info

    def render(self):
        self.renders += 1
        return np.zeros((8, 8, 3), np.uint8)


class TestSelectObsSpace:
    def test_prefers_the_image(self):
        space = dict_space(image=IMAGE, state=(5,), position=(2,))
        assert select_obs_space(space) == {"image": IMAGE}

    def test_falls_back_to_the_state_vector(self):
        space = dict_space(state=(5,), position=(2,), velocity=(3,))
        assert select_obs_space(space) == {"state": (5,)}

    def test_adds_the_language_instruction(self):
        space = dict_space(image=IMAGE, instr=(1024,))
        assert select_obs_space(space) == {"image": IMAGE, "instr": (1024,)}

    def test_explicit_keys_are_used_in_order(self):
        space = dict_space(image=IMAGE, state=(5,))
        assert list(select_obs_space(space, ["state", "image"])) == ["state", "image"]
        assert select_obs_space(space, "state, image") == {
            "state": (5,),
            "image": IMAGE,
        }

    def test_missing_keys_are_reported(self):
        with pytest.raises(ValueError, match="instr"):
            select_obs_space(dict_space(image=IMAGE), ["instr"])

    def test_unsupported_ranks_are_rejected(self):
        space = gym.spaces.Dict({"grid": box((4, 4))})
        with pytest.raises(ValueError, match="grid"):
            select_obs_space(space, ["grid"])

    def test_a_single_box_is_one_input(self):
        assert select_obs_space(box((7,))) == {"state": (7,)}

    def test_images_must_share_a_spatial_size(self):
        with pytest.raises(ValueError, match="spatial"):
            split_obs_space({"a": (3, 16, 16), "b": (3, 8, 8)})


class TestMultiEncoderDecoder:
    def make(self, obs_space):
        encoder = DreamerV3MultiEncoder(
            obs_space, cnn_depth=4, hidden_size=16, mlp_layers=1
        )
        decoder = DreamerV3MultiDecoder(
            20, encoder, cnn_depth=4, hidden_size=16, mlp_layers=1
        )
        return encoder, decoder

    def test_embedding_concatenates_image_and_vector_parts(self):
        encoder, _ = self.make({"image": IMAGE, "a": (3,), "b": (5,)})
        assert encoder.image_encoder is not None
        assert encoder.vector_encoder is not None
        assert encoder.vector_encoder.obs_shape == (8,)  # 3 + 5 entries
        assert encoder.embed_size == (
            encoder.image_encoder.embed_size + encoder.vector_encoder.embed_size
        )
        obs = {
            "image": torch.rand(4, 2, *IMAGE),
            "a": torch.randn(4, 2, 3),
            "b": torch.randn(4, 2, 5),
        }
        assert encoder(obs).shape == (4, 2, encoder.embed_size)

    def test_image_inputs_are_stacked_along_channels(self):
        encoder, _ = self.make({"rgb": (3, 16, 16), "depth": (1, 16, 16)})
        assert encoder.image_encoder.obs_shape == (4, 16, 16)
        assert encoder.vector_encoder is None

    def test_decoder_predicts_every_input(self):
        encoder, decoder = self.make({"image": IMAGE, "a": (3,), "b": (5,)})
        out = decoder(torch.randn(4, 2, 20))
        assert out["image"].shape == (4, 2, *IMAGE)
        assert out["a"].shape == (4, 2, 3)
        assert out["b"].shape == (4, 2, 5)
        assert float(out["image"].min()) >= 0.0 and float(out["image"].max()) <= 1.0

    def test_loss_is_zero_for_a_perfect_reconstruction(self):
        _, decoder = self.make({"image": IMAGE, "a": (3,)})
        target = {"image": torch.rand(2, *IMAGE), "a": torch.randn(2, 3) * 100}
        prediction = {"image": target["image"], "a": symlog(target["a"])}
        terms = decoder.loss_terms(prediction, target)
        assert set(terms) == {"image", "a"}
        assert torch.allclose(decoder.loss(prediction, target), torch.zeros(2))

    def test_loss_adds_the_inputs_errors(self):
        _, decoder = self.make({"image": IMAGE, "a": (3,)})
        target = {"image": torch.zeros(1, *IMAGE), "a": torch.zeros(1, 3)}
        prediction = {"image": torch.ones(1, *IMAGE), "a": torch.ones(1, 3)}
        terms = decoder.loss_terms(prediction, target)
        assert float(terms["image"]) == pytest.approx(0.5 * np.prod(IMAGE))
        assert float(terms["a"]) == pytest.approx(0.5 * 3)
        assert float(decoder.loss(prediction, target)) == pytest.approx(
            0.5 * np.prod(IMAGE) + 1.5
        )

    def test_vectors_are_symlog_transformed_on_input(self):
        encoder, _ = self.make({"a": (3,)})
        huge = encoder({"a": torch.full((1, 3), 1e6)})
        assert torch.isfinite(huge).all()


class TestMultiInputCore:
    def build(self, obs_space=MIXED, **config):
        return DreamerV3(tiny_config(**config), obs_space, 2, "cpu")

    def test_replay_stores_each_input_with_its_dtype(self):
        model = self.build()
        buffer = model.data_buffer
        assert buffer.obs_space == MIXED
        assert getattr(buffer, "obs__image").dtype == np.uint8
        assert getattr(buffer, "obs__instr").dtype == np.float32
        assert "obs__instr" in buffer.replay_arrays

    def test_collection_records_only_the_encoded_inputs(self):
        model = self.build()
        model.act_and_collect_data(FakeEnv(MIXED), 6)
        sample = model.data_buffer.sample() if model.data_buffer.can_sample else None
        assert len(model.data_buffer) > 0
        stored = getattr(model.data_buffer, "obs__instr")[: len(model.data_buffer)]
        assert np.abs(stored).sum() > 0
        assert sample is None or set(sample["observations"]) == {"image", "instr"}

    def test_training_reconstructs_every_input(self):
        model = self.build()
        model.collect_random_episodes(FakeEnv(MIXED), 30)
        losses = model.train_one_batch()
        assert all(np.isfinite(value) for value in losses)
        assert "wm/recon_loss/image" in model.last_metrics
        assert "wm/recon_loss/instr" in model.last_metrics
        assert model.last_metrics["wm/recon_loss"] == pytest.approx(
            model.last_metrics["wm/recon_loss/image"]
            + model.last_metrics["wm/recon_loss/instr"],
            rel=1e-4,
        )

    def test_every_input_changes_the_embedding(self):
        model = self.build()
        rng = np.random.default_rng(1)
        obs = observation(MIXED, rng)
        base = model.obs_encoder(model.preprocess({k: v[None] for k, v in obs.items()}))
        changed = dict(obs, instr=obs["instr"] + 100.0)
        other = model.obs_encoder(
            model.preprocess({k: v[None] for k, v in changed.items()})
        )
        assert not torch.allclose(base, other)

    def test_state_only_model_trains_and_renders_videos(self):
        space = {"state": (9,)}
        model = self.build(space)
        assert not model.is_image_obs
        env = FakeEnv(space)
        model.collect_random_episodes(env, 30)
        assert all(np.isfinite(value) for value in model.train_one_batch())
        rewards, videos, _ = model.evaluate(env, 1, render=True)
        assert env.renders > 0
        assert np.asarray(videos[0][0]).shape == (8, 8, 3)

    def test_checkpoint_keeps_the_observation_space(self, tmp_path):
        model = self.build()
        model.collect_random_episodes(FakeEnv(MIXED), 30)
        model.train_one_batch()
        path = tmp_path / "model.pt"
        model.save(str(path))
        restored = DreamerV3.from_pretrained(tmp_path, map_location="cpu")
        assert restored.obs_space == MIXED
        for (name, a), b in zip(
            model.obs_encoder.state_dict().items(),
            restored.obs_encoder.state_dict().values(),
        ):
            assert torch.equal(a, b), name

    def test_single_shape_models_still_accept_their_usual_dicts(self):
        model = DreamerV3(tiny_config(), (9,), 2, "cpu")
        assert model.obs_space == {"state": (9,)}
        tensor = model._obs_to_tensor({"obs": np.ones(9, np.float32)})
        assert tensor["state"].shape == (1, 9)


class TestAgentObservationChoice:
    def test_dreamer_v3_agent_encodes_the_selected_keys(self):
        from simulusion.models.dreamer_v3 import DreamerV3Agent

        agent = DreamerV3Agent.__new__(DreamerV3Agent)
        agent.args = tiny_config(obs_keys=["state"])
        env = type("Env", (), {})()
        env.observation_space = dict_space(image=IMAGE, state=(5,))
        assert agent._observation_shape(env) == {"state": (5,)}

    def test_dreamer_v1_v2_agents_still_encode_the_image(self):
        from simulusion.models.dreamer import DreamerAgent

        agent = DreamerAgent.__new__(DreamerAgent)
        env = type("Env", (), {})()
        env.observation_space = dict_space(image=IMAGE, state=(5,))
        assert tuple(agent._observation_shape(env)) == IMAGE
