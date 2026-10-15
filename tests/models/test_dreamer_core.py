"""Unmocked regression tests for the core ``Dreamer`` model.

These construct a tiny real ``Dreamer`` (no environment, no gym) so the acting,
evaluation, checkpointing and discount-head paths actually execute.
"""

import numpy as np
import pytest
import torch

from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.models.dreamer import Dreamer

OBS_SHAPE = (3, 64, 64)
ACTION_SIZE = 2


def _tiny_config(**overrides):
    config = DreamerConfig()
    config.buffer_size = 64
    config.batch_size = 2
    config.train_seq_len = 4
    config.imagine_horizon = 3
    config.obs_embed_size = 32
    config.num_units = 16
    config.deter_size = 16
    config.stoch_size = 4
    config.perf_defaults = False
    config.use_amp = False
    config.restore = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _dreamer(**overrides):
    torch.manual_seed(0)
    return Dreamer(_tiny_config(**overrides), OBS_SHAPE, ACTION_SIZE, "cpu")


def _obs(value=0):
    return {"image": np.full(OBS_SHAPE, value, dtype=np.uint8)}


class _FakeEnv:
    """Old-gym style env returning Dreamer's ``{"image": ...}`` observations."""

    def __init__(self, episode_length=3):
        self.episode_length = episode_length
        self.t = 0
        self.action_space = self

    def sample(self):
        return np.random.uniform(-1.0, 1.0, size=ACTION_SIZE).astype(np.float32)

    def reset(self):
        self.t = 0
        return _obs(self.t)

    def step(self, action):
        self.t += 1
        return _obs(self.t * 40), 1.0, self.t >= self.episode_length, {}


def test_act_with_world_model_carries_posterior_not_prior():
    dreamer = _dreamer()
    captured = {}
    original = dreamer.rssm.observe_step

    def spy(*args, **kwargs):
        captured["result"] = original(*args, **kwargs)
        return captured["result"]

    dreamer.rssm.observe_step = spy
    state = dreamer.rssm.init_state(1, dreamer.device)
    action = torch.zeros(1, ACTION_SIZE)
    with torch.no_grad():
        returned_state, _ = dreamer.act_with_world_model(_obs(200), state, action)

    posterior, prior = captured["result"]
    assert returned_state is posterior
    assert not torch.equal(posterior["mean"], prior["mean"])


def test_act_depends_on_current_observation():
    dreamer = _dreamer()
    state = dreamer.rssm.init_state(1, dreamer.device)
    action = torch.zeros(1, ACTION_SIZE)
    with torch.no_grad():
        dark, _ = dreamer.act_with_world_model(_obs(0), state, action)
        bright, _ = dreamer.act_with_world_model(_obs(255), state, action)
    # The prior ignores the frame; only the posterior mean can differ.
    assert not torch.allclose(dark["mean"], bright["mean"])


def test_evaluate_with_render_runs_unmocked():
    dreamer = _dreamer()
    rewards, videos, latents = dreamer.evaluate(_FakeEnv(), 2, render=True)
    assert rewards.tolist() == [3.0, 3.0]
    expected = dreamer.args.stoch_size + dreamer.args.deter_size
    assert latents.shape == (6, 1, expected)
    assert videos.shape[:2] == (2, 3)


def test_save_restore_round_trips_the_critic(tmp_path):
    source = _dreamer()
    with torch.no_grad():
        for param in source.value_model.parameters():
            param.add_(1.0)
    path = tmp_path / "dreamer.pt"
    source.save(str(path))

    target = _dreamer()
    target.restore_checkpoint(path)
    for expected, actual in zip(
        source.value_model.state_dict().values(),
        target.value_model.state_dict().values(),
    ):
        assert torch.equal(expected, actual)


def test_restore_accepts_checkpoint_without_critic(tmp_path):
    source = _dreamer()
    path = tmp_path / "legacy.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["value_model"]
    torch.save(checkpoint, path)

    target = _dreamer()
    before = [p.clone() for p in target.value_model.parameters()]
    target.restore_checkpoint(path)
    for old, new in zip(before, target.value_model.parameters()):
        assert torch.equal(old, new)


def test_restore_argument_is_honoured(monkeypatch):
    calls = []
    monkeypatch.setattr(Dreamer, "restore_checkpoint", lambda self, p: calls.append(p))
    Dreamer(_tiny_config(restore=True), OBS_SHAPE, ACTION_SIZE, "cpu", restore=False)
    assert calls == []


@pytest.mark.parametrize("use_disc_model", [False, True])
def test_train_one_batch_with_discount_model(use_disc_model):
    dreamer = _dreamer(use_disc_model=use_disc_model)
    env = _FakeEnv(episode_length=5)
    dreamer.collect_random_episodes(env, 40)
    losses = dreamer.train_one_batch()
    assert len(losses) == 3
    assert all(np.isfinite(losses))
    assert losses != [0.0, 0.0, 0.0]


def test_discount_model_discounts_are_scaled_by_gamma():
    dreamer = _dreamer(use_disc_model=True, discount=0.5)
    dreamer.collect_random_episodes(_FakeEnv(episode_length=5), 40)
    dreamer.train_one_batch()
    # cumprod of (gamma * P(continue)) can never exceed gamma ** k.
    horizon = dreamer.discounts.shape[0]
    bounds = torch.tensor([0.5**k for k in range(horizon)]).view(-1, 1, 1)
    assert torch.all(dreamer.discounts <= bounds + 1e-6)


def test_dreamer_v2_trains_one_batch():
    """DreamerV2's two-hot reward/value heads must expose `.mean` as a tensor,
    like the torch distributions the rest of Dreamer reads it from."""
    from simulusion.models.dreamer_v2 import DreamerV2

    torch.manual_seed(0)
    dreamer = DreamerV2(_tiny_config(), OBS_SHAPE, ACTION_SIZE, "cpu")
    dreamer.collect_random_episodes(_FakeEnv(episode_length=5), 40)
    losses = dreamer.train_one_batch()
    assert all(np.isfinite(losses))


@pytest.mark.parametrize(
    ("name", "core", "algo"),
    [
        ("dreamer-v1", "DreamerV1", "Dreamerv1"),
        ("dreamer-v2", "DreamerV2", "Dreamerv2"),
    ],
)
def test_versioned_model_names_build_agents_for_that_version(name, core, algo):
    """`create_model("dreamer-v2")` used to resolve to the bare DreamerV2 core,
    whose constructor needs (args, obs_shape, action_size, device), so it raised."""
    import simulusion.models as models
    from simulusion.api import _load_object, get_model_spec

    agent_cls = _load_object(get_model_spec(name).import_path)
    assert issubclass(agent_cls, models.DreamerAgent)
    assert agent_cls.core_cls is getattr(models, core)
    assert agent_cls.algo_name == algo


def test_dreamer_v2_actor_maximises_the_lambda_returns_themselves():
    """DreamerV2 Sec. 2.4, continuous actions (rho = 0): L = -E[V_lambda].

    A symlog on the returns here made the actor saturate on cartpole swing-up
    (every action pinned at -1, actor gradient ~1e-8)."""
    from simulusion.models.dreamer_v2 import DreamerV2

    torch.manual_seed(0)
    dreamer = DreamerV2(_tiny_config(), OBS_SHAPE, ACTION_SIZE, "cpu")
    returns = torch.tensor([[[5.0]], [[50.0]]], requires_grad=True)
    discounts = torch.tensor([[[1.0]], [[0.5]]])

    loss = dreamer._compute_actor_loss(returns, discounts)
    loss.backward()

    assert loss.item() == pytest.approx(-(5.0 * 1.0 + 50.0 * 0.5) / 2)
    # Every unit of return is worth the same to the actor, however large.
    assert torch.allclose(returns.grad.flatten(), torch.tensor([-0.5, -0.25]))
