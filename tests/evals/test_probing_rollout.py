"""Tests for probing imagined states in simulusion.evals.probing.rollout."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulusion.data.episode import Episode  # noqa: E402
from simulusion.envs.dmc import PhysicsStateSpec  # noqa: E402
from simulusion.evals.probing import (  # noqa: E402
    ProbeDataset,
    extract_latents,
    fit_linear_probe,
    rollout_probe,
)
from simulusion.world_model.protocol import (  # noqa: E402
    ActionSpec,
    Transition,
    WorldModelSpec,
)
from tests.evals.test_probing_latents import _action_fn, _data  # noqa: E402
from tests.test_world_model_protocol import _build  # noqa: E402

T = 30
_SPEC = PhysicsStateSpec(("slider",), ("slide",), ("slider",), ("slide",), ())


class _Cart:
    """A world model that is right about everything, or drifts by ``bias``.

    The frame's brightness is the cart position; the state is that position and
    each step adds the action, exactly as the episodes below were made.
    """

    family = "cart"

    def __init__(self, bias=0.0):
        self.bias = bias
        self.spec = WorldModelSpec(
            family="cart",
            obs_shape=(3, 8, 8),
            action=ActionSpec("continuous", 1),
            context_frames=1,
            max_context=None,
        )

    def encode(self, frames, actions=None):
        frames = torch.as_tensor(np.asarray(frames))
        return {"x": frames[:, -1, 0, 0, :1].float()}

    def step(self, state, action):
        action = torch.as_tensor(action, dtype=torch.float32).reshape(-1, 1)
        return Transition(state={"x": state["x"] + action + self.bias})


def _cart_data(episodes=3):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(episodes):
        actions = rng.integers(-2, 3, (T - 1, 1)).astype(np.float32)
        # Starts at 100 and moves at most 2 a step, so it fits a uint8 pixel.
        x = np.concatenate([[100.0], 100.0 + np.cumsum(actions[:, 0])])
        frames = np.repeat(x.astype(np.uint8)[:, None, None, None], 8, 1)
        frames = np.repeat(np.repeat(frames, 8, 2), 3, 3)
        out.append(
            Episode(
                frames=frames,
                actions=actions,
                observations={
                    "qpos": x[:, None].astype(np.float32),
                    "qvel": np.zeros((T, 1), np.float32),
                    "act": np.zeros((T, 0), np.float32),
                },
            )
        )
    return ProbeDataset("cartpole-swingup", _SPEC, out)


def _fitted(model, data):
    latents = extract_latents(model, data)
    targets = latents.match(data.targets(("qpos",)))
    z = latents.features["x"]
    return fit_linear_probe(z, targets.values, z, targets.values)


def test_exact_dynamics_keep_the_state_at_every_horizon():
    data = _cart_data()
    model = _Cart()
    probe = _fitted(model, data)

    scores = rollout_probe(
        model,
        data,
        probe,
        key="x",
        targets=data.targets(("qpos",)),
        horizons=(10, 1, 5),
        starts_every=4,
    )

    assert scores.horizons == (1, 5, 10)
    assert scores.names == ("qpos/slider",)
    assert scores.r2.shape == scores.encoded_r2.shape == (3, 1)
    np.testing.assert_allclose(scores.r2, 1.0, atol=1e-6)
    np.testing.assert_allclose(scores.encoded_r2, 1.0, atol=1e-6)
    # Starts at 0, 4, ..., 16 in each of 3 episodes leave room for 10 steps.
    assert scores.rollouts == 3 * 5


def test_drifting_dynamics_lose_the_state_with_the_horizon():
    data = _cart_data()
    probe = _fitted(_Cart(), data)

    scores = rollout_probe(
        _Cart(bias=0.5),
        data,
        probe,
        key="x",
        targets=data.targets(("qpos",)),
        horizons=(1, 5, 10),
        starts_every=2,
    )

    assert np.all(np.diff(scores.mean()) < 0)
    np.testing.assert_allclose(scores.encoded_r2, 1.0, atol=1e-6)


@pytest.mark.parametrize("family", ["planet", "dreamer", "iris", "diamond"])
def test_real_families_imagine_and_are_scored(family):
    model = _build(family)
    data = _data(episodes=2)
    latents = extract_latents(model, data, action_fn=_action_fn(model))
    key = sorted(latents.features)[0]
    targets = latents.match(data.targets())
    z = latents.features[key]
    probe = fit_linear_probe(z, targets.values, z, targets.values)

    scores = rollout_probe(
        model,
        data,
        probe,
        key=key,
        targets=data.targets(),
        horizons=(1,),  # IRIS's 4-frame window leaves one step in 5 frames
        starts_every=1,
        action_fn=_action_fn(model),
    )

    assert scores.r2.shape == (1, len(targets.names))
    # Targets too flat over these few frames are NaN in both, never only one.
    np.testing.assert_array_equal(np.isnan(scores.r2), np.isnan(scores.encoded_r2))
    assert np.isfinite(scores.r2).any() and scores.rollouts > 0


def test_rollouts_need_actions_horizons_and_long_enough_episodes():
    data = _cart_data(episodes=1)
    targets = data.targets(("qpos",))
    probe = _fitted(_Cart(), data)

    with pytest.raises(ValueError, match="discrete actions"):
        rollout_probe(_build("iris"), _data(1), probe, key="tokens", targets=targets)
    with pytest.raises(ValueError, match="positive"):
        rollout_probe(_Cart(), data, probe, key="x", targets=targets, horizons=(0,))
    with pytest.raises(ValueError, match="long enough"):
        rollout_probe(_Cart(), data, probe, key="x", targets=targets, horizons=(T,))
    with pytest.raises(ValueError, match="No feature"):
        rollout_probe(_Cart(), data, probe, key="nope", targets=targets)
