"""Every family behind one WorldModel protocol, with one set of guarantees.

Each family is built tiny and untrained (CPU, seconds). The contract tests run
on all of them; the family tests check the adapter against the family's own
code path, so the protocol never changes a model's numerics.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulusion.world_model import (  # noqa: E402
    WORLD_MODEL_FAMILIES,
    Rollout,
    Transition,
    WorldModel,
    as_world_model,
    load_world_model,
    world_model_class,
)
from simulusion.world_model.protocol import MANIFEST_NAME, ActionSpec  # noqa: E402

B, HORIZON = 2, 3


# ---------------------------------------------------------------- tiny models


def _dreamer():
    from simulusion.configs.dreamer_config import DreamerConfig
    from simulusion.models.dreamer import Dreamer

    config = DreamerConfig()
    for key, value in dict(
        buffer_size=64,
        batch_size=2,
        train_seq_len=4,
        imagine_horizon=3,
        obs_embed_size=32,
        num_units=16,
        deter_size=16,
        stoch_size=4,
        perf_defaults=False,
        use_amp=False,
        restore=False,
        use_disc_model=True,
    ).items():
        setattr(config, key, value)
    return Dreamer(config, (3, 64, 64), 2, "cpu")


def _planet():
    from simulusion.models.rssm import RecurrentStateSpaceModel

    return RecurrentStateSpaceModel(
        action_size=2, state_size=16, latent_size=4, hidden_size=16, embed_size=32
    )


def _iris_config():
    from simulusion.configs.iris_config import IRISConfig

    config = IRISConfig()
    for key, value in dict(
        vocab_size=32,
        tokens_per_frame=16,
        token_embedding_dim=32,
        encoder_channels=16,
        decoder_depth=16,
        frame_shape=(3, 64, 64),
        transformer_layers=1,
        transformer_heads=2,
        transformer_embed_dim=32,
        transformer_timesteps=4,
        perceptual_weight=0.0,
    ).items():
        setattr(config, key, value)
    return config


def _iris():
    from simulusion.models.iris_agent import IRISAgent

    return IRISAgent(_iris_config(), action_size=4, device=torch.device("cpu"))


def _diamond_config():
    from simulusion.configs.diamond_config import DiamondConfig

    return DiamondConfig(
        obs_size=16,
        num_conditioning_frames=3,
        diffusion_channels=[32, 32],  # the UNet's GroupNorms use 32 groups
        diffusion_res_blocks=1,
        diffusion_cond_dim=16,
        num_sampling_steps=2,
        reward_channels=[8, 8],
        reward_res_blocks=1,
        reward_cond_dim=16,
        reward_lstm_dim=16,
        device="cpu",
    )


def _diamond():
    from simulusion.world_model.diamond import build_diamond_networks

    config = _diamond_config()
    diffusion_model, sampler, reward_model = build_diamond_networks(config, 4, "cpu")
    return SimpleNamespace(
        config=config,
        diffusion_model=diffusion_model,
        sampler=sampler,
        reward_model=reward_model,
    )


def _genie():
    from simulusion.models.genie import Genie

    return Genie(
        num_frames=4,
        image_size=16,
        tokenizer_vocab_size=32,
        tokenizer_encoder_dim=32,
        tokenizer_decoder_dim=32,
        action_encoder_dim=32,
        action_decoder_dim=32,
        dynamics_dim=32,
        dynamics_depth=1,
        dynamics_num_heads=2,
        encoder_depth=1,
        decoder_depth=1,
        latent_action_depth=1,
        tokenizer_num_heads=2,
        action_num_heads=2,
    )


def _world_models():
    from simulusion.models.mdrnn import MDRNN
    from simulusion.vision.VAE.ConvVAE import ConvVAE

    return ConvVAE(img_channels=3, latent_size=8), MDRNN(8, 3, 16, 2)


BUILDERS = {
    "dreamer": _dreamer,
    "planet": _planet,
    "iris": _iris,
    "diamond": _diamond,
    "genie": _genie,
    "world-models": _world_models,
}


def _build(family, **kwargs):
    torch.manual_seed(0)
    return as_world_model(BUILDERS[family](), **kwargs)


def _frames(model, steps=None, seed=0):
    """Random uint8 context, channel-last, at twice the model's size."""
    c, h, w = model.spec.obs_shape
    t = steps or max(model.spec.context_frames, 3)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (B, t, 2 * h, 2 * w, c), dtype=np.uint8)


def _actions(spec: ActionSpec, *shape, seed=0):
    rng = np.random.default_rng(seed)
    if spec.kind == "discrete":
        return torch.from_numpy(rng.integers(0, spec.n, shape))
    return torch.from_numpy(rng.uniform(-1, 1, (*shape, spec.n)).astype(np.float32))


@pytest.fixture(params=sorted(BUILDERS))
def family(request):
    return request.param


# -------------------------------------------------------------- the contract


def test_every_family_has_a_builder():
    assert set(BUILDERS) == set(WORLD_MODEL_FAMILIES)


def test_implements_the_protocol(family):
    model = _build(family)
    assert isinstance(model, WorldModel)
    assert model.spec.family == family
    assert type(model) is world_model_class(family)


def test_encode_step_decode(family):
    model = _build(family)
    spec = model.spec
    frames = _frames(model)
    state = model.encode(frames, _actions(spec.action, B, frames.shape[1] - 1))
    assert all(v.shape[0] == B for v in state.values())

    out = model.step(state, _actions(spec.action, B))
    assert isinstance(out, Transition)
    if spec.has_reward:
        assert out.reward is not None and out.reward.shape == (B,)
    else:
        assert out.reward is None
    if spec.has_termination:
        assert out.terminated is not None and out.terminated.dtype == torch.bool
        assert out.terminated.shape == (B,)
    else:
        assert out.terminated is None

    image = model.decode(out.state)
    assert image.shape == (B, *spec.obs_shape)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0


def test_encode_without_actions(family):
    model = _build(family)
    state = model.encode(_frames(model))
    assert model.decode(state).shape == (B, *model.spec.obs_shape)


def _same_state(a, b):
    assert set(a) == set(b)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-5, rtol=1e-4)


def test_encode_steps_match_encode_on_every_prefix(family):
    model = _build(family)
    T = model.spec.context_frames + 2
    frames = _frames(model, steps=T)
    actions = _actions(model.spec.action, B, T - 1)

    torch.manual_seed(1)
    states = model.encode_steps(frames, actions)
    assert len(states) == T
    first = model.spec.context_frames - 1
    assert all(state is None for state in states[:first])
    for t in range(first, T):
        # Same seed: Dreamer's stochastic posterior draws the same noise.
        torch.manual_seed(1)
        _same_state(states[t], model.encode(frames[:, : t + 1], actions[:, :t]))


def test_encode_steps_with_a_short_context_use_windows():
    model = _build("world-models")
    frames = _frames(model, steps=4)
    actions = _actions(model.spec.action, B, 3)

    single = model.encode_steps(frames, actions, context=1)
    for t, state in enumerate(single):
        # One frame of context: z of that frame and a fresh LSTM state.
        alone = model.encode(frames[:, t : t + 1])
        _same_state(state, alone)
    with pytest.raises(ValueError, match="context"):
        _build("diamond").encode_steps(_frames(_build("diamond"), steps=4), context=1)


def test_rollout_is_repeated_steps(family):
    model = _build(family)
    spec = model.spec
    state = model.encode(_frames(model))
    actions = _actions(spec.action, B, HORIZON, seed=1)

    torch.manual_seed(5)
    traj = model.rollout(state, actions)
    assert isinstance(traj, Rollout)
    assert traj.frames is not None
    assert traj.frames.shape == (B, HORIZON, *spec.obs_shape)

    torch.manual_seed(5)
    current, frames, rewards = state, [], []
    for t in range(HORIZON):
        out = model.step(current, actions[:, t])
        current = out.state
        frames.append(model.decode(current))
        if out.reward is not None:
            rewards.append(out.reward)
    torch.testing.assert_close(traj.frames, torch.stack(frames, 1))
    if spec.has_reward:
        assert traj.rewards is not None
        torch.testing.assert_close(traj.rewards, torch.stack(rewards, 1))
    else:
        assert traj.rewards is None


def test_rollout_without_decoding(family):
    model = _build(family)
    state = model.encode(_frames(model))
    traj = model.rollout(state, _actions(model.spec.action, B, 2), decode=False)
    assert traj.frames is None
    assert model.batch_size(traj.state) == B


def test_seed_makes_steps_repeatable(family):
    model = _build(family)
    state = model.encode(_frames(model))
    action = _actions(model.spec.action, B)
    torch.manual_seed(3)
    a = model.decode(model.step(state, action).state)
    torch.manual_seed(3)
    b = model.decode(model.step(state, action).state)
    torch.testing.assert_close(a, b)


def test_save_and_load_round_trip(family, tmp_path):
    model = _build(family)
    saved = model.save(tmp_path / "wm")
    assert (saved / MANIFEST_NAME).exists()

    loaded = load_world_model(saved, device="cpu")
    assert type(loaded) is type(model)
    assert loaded.spec == model.spec

    frames = _frames(model)
    actions = _actions(model.spec.action, B, HORIZON, seed=2)
    results = []
    for m in (model, loaded):
        torch.manual_seed(7)
        results.append(m.rollout(m.encode(frames), actions))
    torch.testing.assert_close(results[0].frames, results[1].frames)


def test_single_action_is_broadcast(family):
    model = _build(family)
    state = model.encode(_frames(model))
    spec = model.spec.action
    action = 1 if spec.kind == "discrete" else [0.5] * spec.n
    out = model.step(state, action)
    assert model.batch_size(out.state) == B


# ------------------------------------------------------------- input handling


def test_wrong_number_of_context_actions_is_rejected():
    model = _build("planet")
    frames = _frames(model, steps=3)
    with pytest.raises(ValueError, match="2 actions between 3 context frames"):
        model.encode(frames, _actions(model.spec.action, B, 3))


def test_too_few_context_frames_is_rejected():
    model = _build("diamond")
    with pytest.raises(ValueError, match="at least 3 context frames"):
        model.encode(_frames(model, steps=2))


def test_discrete_actions_out_of_range_are_rejected():
    model = _build("iris")
    state = model.encode(_frames(model))
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        model.step(state, torch.tensor([0, 4]))


def test_continuous_actions_are_clipped_to_bounds():
    spec = ActionSpec("continuous", 2, -1.0, 1.0)
    tensor = spec.as_tensor([[3.0, -3.0]], 1, torch.device("cpu"))
    assert tensor.tolist() == [[1.0, -1.0]]


def test_float_frames_and_uint8_frames_agree():
    model = _build("planet")
    frames = _frames(model)
    a = model.as_frames(frames)
    b = model.as_frames(frames.astype(np.float32) / 255.0)
    torch.testing.assert_close(a, b)


# ----------------------------------------------------------------- discovery


def test_agents_and_inner_models_are_both_accepted():
    # Shaped like Planet and DreamerAgent: the model sits on an attribute.
    planet = SimpleNamespace(rssm=_planet(), bit_depth=4)
    assert type(as_world_model(planet)).__name__ == "PlanetWorldModel"
    assert as_world_model(planet).bit_depth == 4
    dreamer = _dreamer()
    agent = SimpleNamespace(dreamer=dreamer)
    assert as_world_model(agent).dreamer is dreamer


def test_a_world_model_is_passed_through():
    model = _build("genie")
    assert as_world_model(model) is model


@pytest.mark.parametrize("name", ["JEPAAgent", "DiT"])
def test_models_without_action_dynamics_are_refused(name):
    impostor = type(name, (), {})()
    with pytest.raises(TypeError, match="does not predict what an action does"):
        as_world_model(impostor)


def test_unknown_objects_are_refused():
    with pytest.raises(TypeError, match="No world model adapter"):
        as_world_model(object())


def test_loading_a_bare_checkpoint_needs_the_family(tmp_path):
    torch.save({"weights": 1}, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="pass the family"):
        load_world_model(tmp_path)


def test_family_names_and_aliases():
    assert world_model_class("DreamerV2") is world_model_class("dreamer")
    assert world_model_class("vae_mdrnn") is world_model_class("world-models")
    with pytest.raises(ValueError, match="Unknown world model family"):
        world_model_class("muzero")


# ------------------------------------------------- against each family's code


def test_dreamer_step_is_the_stepper_imagine():
    from simulusion.inference.steppers import DreamerStepper

    model = _build("dreamer")
    state = model.encode(_frames(model))
    action = _actions(model.spec.action, B)
    torch.manual_seed(1)
    out = model.step(state, action)
    torch.manual_seed(1)
    with torch.no_grad():
        ref = DreamerStepper(model.dreamer).imagine(state, action)
    torch.testing.assert_close(out.state["deter"], ref.state["deter"])
    torch.testing.assert_close(out.reward, ref.reward)
    assert model.spec.has_termination


def test_planet_encode_matches_the_planner():
    """The CEM planner filters with get_init_state(encoder(obs), h, s, a)."""
    model = _build("planet")
    frames = _frames(model, steps=2)
    actions = _actions(model.spec.action, B, 1)
    state = model.encode(frames, actions)
    rssm = model.rssm
    x = model._preprocess(model.as_frames(frames))
    with torch.no_grad():
        h, s = rssm.get_init_state(rssm.encoder(x[:, 0]))
        h, s = rssm.get_init_state(rssm.encoder(x[:, 1]), h, s, actions[:, 0])
    torch.testing.assert_close(state["deter"], h)
    torch.testing.assert_close(state["stoch"], s)


def test_planet_decodes_with_the_argument_order_training_uses():
    """train_planet calls ``bottle(rssm.decoder, states, latents)``: (h, s)."""
    model = _build("planet")
    state = model.encode(_frames(model))
    with torch.no_grad():
        expected = model.rssm.decoder(state["deter"], state["stoch"]) + 0.5
    torch.testing.assert_close(model.decode(state), expected.clamp(0, 1))


def test_planet_loads_the_bare_training_checkpoint(tmp_path):
    """Planet.train saves only the RSSM state dict, as ckpt_<epoch>.pth."""
    torch.manual_seed(0)
    rssm = _planet()
    torch.save(rssm.state_dict(), tmp_path / "ckpt_25.pth")
    model = load_world_model(tmp_path / "ckpt_25.pth", "planet", device="cpu")
    assert model.spec.action.n == 2
    for name, value in rssm.state_dict().items():
        torch.testing.assert_close(model.rssm.state_dict()[name], value)


def test_iris_rollout_matches_imagine_rollout():
    """Greedy tokens, the agent's own actions: the same frames and rewards."""
    model = _build("iris", sample=False)
    agent = model.agent
    frame = torch.rand(B, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    with torch.no_grad():
        ref = agent.imagine_rollout(
            frame, horizon=HORIZON, sample_tokens=False, stop_on_termination=False
        )
    state = model.encode(frame.unsqueeze(1))
    traj = model.rollout(state, ref["actions"])
    torch.testing.assert_close(traj.frames, ref["frames"][:, 1:], atol=1e-4, rtol=0)
    torch.testing.assert_close(traj.rewards, ref["rewards"], atol=1e-4, rtol=0)


def test_iris_history_stays_inside_the_window():
    model = _build("iris", sample=False)
    window = model.spec.max_context
    state = model.encode(_frames(model))
    traj = model.rollout(state, _actions(model.spec.action, B, window + 2))
    assert traj.state["tokens"].shape[1] == window
    assert traj.state["actions"].shape[1] == window - 1


def test_diamond_loads_a_training_checkpoint(tmp_path):
    """A DiamondAgent checkpoint also holds optimizers and hidden states."""
    nets = _diamond()
    torch.save(
        {
            "config": nets.config.to_dict(),
            "diffusion_model": nets.diffusion_model.state_dict(),
            "reward_model": nets.reward_model.state_dict(),
            "actor_critic": {},
            "diffusion_opt": {},
            "last_policy_hidden": None,
        },
        tmp_path / "checkpoint.pt",
    )
    model = load_world_model(tmp_path, "diamond", device="cpu")
    assert model.spec.action.n == 4  # read off the action embedding
    assert model.spec.context_frames == 3


def test_diamond_sampling_steps_override_keeps_the_preconditioner():
    model = _build("diamond", sampling_steps=1)
    nets = _diamond()
    assert model.sampler.num_steps == 1
    assert model.sampler.edm_precond.sigma_data == nets.sampler.edm_precond.sigma_data


def test_genie_actions_are_latent_and_inferred():
    model = _build("genie", temperature=0)
    assert model.spec.action.latent and not model.spec.has_reward
    state = model.encode(_frames(model, steps=3))
    assert state["actions"].shape == (B, 2)
    assert not model.spec.stochastic


def test_world_models_cell_matches_the_sequence_mdrnn():
    """Encoding steps the LSTM exactly as the trained sequence MDRNN runs it."""
    torch.manual_seed(0)
    vae, mdrnn = _world_models()
    model = as_world_model((vae, mdrnn))
    frames = _frames(model, steps=4)
    actions = _actions(model.spec.action, B, 3)
    state = model.encode(frames, actions)
    with torch.no_grad():
        x = model.as_frames(frames)
        mu, _ = vae.encoder(x.flatten(0, 1))
        latents = mu.reshape(B, 4, -1)
        ins = torch.cat([actions, latents[:, :3]], dim=-1).transpose(0, 1)
        _, (h, c) = mdrnn.rnn(ins)
    torch.testing.assert_close(state["h"], h[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(state["c"], c[0], atol=1e-5, rtol=0)


def test_world_models_load_the_training_checkpoints(tmp_path):
    from simulusion.world_model.vae_mdrnn import MDRNNWorldModel

    torch.manual_seed(0)
    vae, mdrnn = _world_models()
    torch.save({"state_dict": vae.state_dict(), "epoch": 3}, tmp_path / "vae.tar")
    torch.save({"state_dict": mdrnn.state_dict()}, tmp_path / "mdrnn.tar")
    model = MDRNNWorldModel.from_checkpoints(
        tmp_path / "vae.tar", tmp_path / "mdrnn.tar", device="cpu"
    )
    assert model.spec.action.n == 3 and model.cell.gaussians == 2
    with pytest.raises(FileNotFoundError, match="from_checkpoints"):
        load_world_model(tmp_path / "vae.tar", "world-models")
