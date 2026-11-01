"""Tests for latent extraction in simulusion.evals.probing.latents."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simulusion.data.episode import Episode  # noqa: E402
from simulusion.envs.dmc import PhysicsStateSpec  # noqa: E402
from simulusion.evals.probing import (  # noqa: E402
    ProbeDataset,
    extract_latents,
    pixel_features,
    reinitialized,
)
from tests.test_world_model_protocol import _build  # noqa: E402

T = 5
_SPEC = PhysicsStateSpec(
    qpos_names=("slider", "hinge_1"),
    qpos_kinds=("slide", "hinge"),
    qvel_names=("slider", "hinge_1"),
    qvel_kinds=("slide", "hinge"),
    act_names=(),
)

# Every family's own feature names, from the tiny models' state layouts.
_EXPECTED = {
    "dreamer": {"mean", "std", "stoch", "deter", "features"},
    "planet": {"deter", "stoch", "features"},
    "world-models": {"z", "h", "c", "features"},
    "iris": {"tokens", "tokens_pooled"},
    "genie": {"tokens", "tokens_pooled"},
    "diamond": {"pixels", "reward_h", "reward_c"},
}


def _data(episodes=2, seed=0):
    rng = np.random.default_rng(seed)
    return ProbeDataset(
        task="cartpole-swingup",
        spec=_SPEC,
        episodes=[
            Episode(
                frames=rng.integers(0, 256, (T, 32, 32, 3), dtype=np.uint8),
                actions=rng.uniform(-1, 1, (T - 1, 2)).astype(np.float32),
                observations={
                    "qpos": rng.normal(size=(T, 2)).astype(np.float32),
                    "qvel": rng.normal(size=(T, 2)).astype(np.float32),
                    "act": np.zeros((T, 0), dtype=np.float32),
                },
            )
            for _ in range(episodes)
        ],
    )


def _action_fn(model):
    if model.spec.action.kind == "discrete":
        return lambda a: (a[:, 0] > 0).astype(np.int64)
    n = model.spec.action.n
    return lambda a: np.pad(a, ((0, 0), (0, max(0, n - a.shape[1]))))[:, :n]


@pytest.fixture(params=sorted(_EXPECTED))
def family(request):
    return request.param


def test_every_family_gives_features_aligned_with_targets(family):
    model = _build(family)
    data = _data()

    latents = extract_latents(model, data, action_fn=_action_fn(model))

    assert set(latents.features) == _EXPECTED[family]
    first = model.spec.context_frames - 1  # DIAMOND needs 3 frames of context
    assert len(latents) == 2 * (T - first)
    np.testing.assert_array_equal(latents.step[: T - first], np.arange(first, T))
    for name, value in latents.features.items():
        assert value.dtype == np.float32 and value.shape[0] == len(latents), name
        assert np.isfinite(value).all(), name
    targets = latents.match(data.targets())
    assert len(targets) == len(latents)
    np.testing.assert_array_equal(targets.step, latents.step)
    assert latents.source == family and latents.info["actions"] is True


def test_recurrent_latents_are_the_filtered_state_at_each_frame():
    model = _build("planet")
    data = _data(episodes=1)

    latents = extract_latents(model, data)

    episode = data[0]
    state = model.encode(episode.frames[None, :3], episode.actions[None, :2])
    np.testing.assert_allclose(
        latents.features["deter"][2], state["deter"][0].numpy(), atol=1e-5
    )
    single = extract_latents(model, data, context=1, keys=["deter"])
    assert set(single.features) == {"deter"}
    assert not np.allclose(single.features["deter"][2], latents.features["deter"][2])


def test_actions_are_checked_or_skipped_by_action_space():
    data = _data(episodes=1)

    with pytest.warns(UserWarning, match="discrete actions"):
        latents = extract_latents(_build("iris"), data)
    assert latents.info["actions"] is False
    with pytest.raises(ValueError, match="action_fn"):
        extract_latents(_build("world-models"), data)  # takes 3-d actions
    assert extract_latents(_build("world-models"), data, actions=False).info == {
        "context": None,
        "actions": False,
        "task": "cartpole-swingup",
    }
    with pytest.raises(ValueError, match="No features named"):
        extract_latents(_build("planet"), data, keys=["nope"])


def test_models_without_encode_steps_fall_back_to_encode():
    model = _build("planet")

    class Plain:
        spec = model.spec

        def encode(self, frames, actions=None):
            return model.encode(frames, actions)

    data = _data(episodes=1)
    plain = extract_latents(Plain(), data)
    fast = extract_latents(model, data)
    np.testing.assert_allclose(
        plain.features["deter"], fast.features["deter"], atol=1e-5
    )


def test_match_rejects_targets_from_another_dataset():
    latents = extract_latents(_build("planet"), _data(episodes=2))
    with pytest.raises(ValueError, match="same dataset"):
        latents.match(_data(episodes=1).targets())


def test_pixel_features_downsample_every_frame():
    data = _data()
    pixels = pixel_features(data, size=8)

    assert pixels.features["pixels"].shape == (2 * T, 3 * 8 * 8)
    assert pixels.source == "pixels"
    expected = data[1].frames[3].astype(np.float32).mean() / 255.0
    np.testing.assert_allclose(
        pixels.features["pixels"][T + 3].mean(), expected, rtol=1e-4
    )


def test_reinitialized_draws_new_weights_and_leaves_the_original():
    model = _build("planet")
    before = {k: v.clone() for k, v in model.rssm.state_dict().items()}

    fresh = reinitialized(model, seed=1)

    after = model.rssm.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    changed = [
        k
        for k, v in fresh.rssm.state_dict().items()
        if v.is_floating_point() and not torch.equal(v, before[k])
    ]
    assert changed
    data = _data(episodes=1)
    assert not np.allclose(
        extract_latents(fresh, data).features["deter"],
        extract_latents(model, data).features["deter"],
    )
