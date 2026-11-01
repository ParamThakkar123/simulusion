"""Tests for the probe datasets in simulusion.evals.probing.data."""

import numpy as np
import pytest

from simulusion.data.episode import Episode
from simulusion.data.folder import EpisodeFolder
from simulusion.envs.dmc import PhysicsStateSpec
from simulusion.evals.probing import (
    ProbeDataset,
    collect_probe_episodes,
    unobservable_dims,
)

# A walker-like body: a tracked slide (rootx), a hinge, and a free-joint
# quaternion, so every target transform has something to act on.
_SPEC = PhysicsStateSpec(
    qpos_names=("rootx", "rooty", "ball/qw", "ball/qx", "ball/qy", "ball/qz"),
    qpos_kinds=("slide", "hinge", "free_quat", "free_quat", "free_quat", "free_quat"),
    qvel_names=("rootx", "rooty"),
    qvel_kinds=("slide", "hinge"),
    act_names=(),
)


def _episode(T, offset=0.0):
    qpos = np.zeros((T, 6), dtype=np.float32)
    qpos[:, 0] = offset + np.arange(T)
    qpos[:, 1] = np.linspace(-np.pi, 3 * np.pi, T)
    qpos[:, 2:] = [-0.5, 0.5, -0.5, 0.5]
    return Episode(
        frames=np.zeros((T, 4, 4, 3), dtype=np.uint8),
        actions=np.zeros((T - 1, 1), dtype=np.float32),
        observations={
            "qpos": qpos,
            "qvel": np.full((T, 2), offset, dtype=np.float32),
            "act": np.zeros((T, 0), dtype=np.float32),
            "task.height": np.full((T, 1), 2.0, dtype=np.float32),
        },
    )


def _dataset(n=4, T=5):
    return ProbeDataset(
        task="walker-walk",
        spec=_SPEC,
        episodes=[_episode(T, offset=10.0 * i) for i in range(n)],
    )


def test_targets_wrap_hinges_fix_quaternion_sign_and_drop_tracked_position():
    targets = _dataset().targets()

    assert targets.names == (
        "qpos/rooty/sin",
        "qpos/rooty/cos",
        "qpos/ball/qw",
        "qpos/ball/qx",
        "qpos/ball/qy",
        "qpos/ball/qz",
        "qvel/rootx",
        "qvel/rooty",
    )
    assert targets.values.dtype == np.float32 and targets.values.shape == (20, 8)
    angle = np.linspace(-np.pi, 3 * np.pi, 5)
    np.testing.assert_allclose(targets.values[:5, 0], np.sin(angle), atol=1e-6)
    np.testing.assert_allclose(targets.values[:5, 1], np.cos(angle), atol=1e-6)
    # -q and q are one rotation; the stored -0.5 for qw flips the whole quaternion.
    np.testing.assert_allclose(targets.values[:, 2:6], [[0.5, -0.5, 0.5, -0.5]] * 20)
    np.testing.assert_array_equal(targets.episode, np.repeat(np.arange(4), 5))
    np.testing.assert_array_equal(targets.step, np.tile(np.arange(5), 4))


def test_targets_options_select_groups_and_dimensions():
    data = _dataset(n=1)

    raw = data.targets(("qpos",), unobservable=None, wrap_angles=False)
    assert raw.names[:2] == ("qpos/rootx", "qpos/rooty")
    np.testing.assert_allclose(raw.values[:, 0], np.arange(5))

    task = data.targets(("task", "act"))
    assert task.names == ("task/height",)
    np.testing.assert_allclose(task.values, 2.0)

    assert "qpos/ball/qw" not in data.targets(("qpos",), unobservable=["ball/qw"]).names

    with pytest.raises(ValueError, match="Unknown target groups"):
        data.targets(("pixels",))
    with pytest.raises(ValueError, match="not qpos names"):
        data.targets(unobservable=["nope"])


def test_unobservable_dims_only_apply_to_tracking_cameras():
    assert unobservable_dims("walker", 0) == ("rootx",)
    assert unobservable_dims("quadruped", 2) == ("root/x", "root/y")
    assert unobservable_dims("quadruped", 3) == ()  # the egocentric camera
    assert unobservable_dims("cartpole", 0) == ()
    fixed = ProbeDataset("walker-walk", _SPEC, [_episode(3)], camera=5)
    assert "qpos/rootx" in fixed.targets().names


def test_split_is_by_episode_disjoint_and_seeded():
    data = _dataset(n=10)

    train, val = data.split(0.3, seed=1)
    again, _ = data.split(0.3, seed=1)
    assert (len(train), len(val)) == (7, 3)
    offsets = lambda d: {float(ep.observations["qvel"][0, 0]) for ep in d}  # noqa: E731
    assert not offsets(train) & offsets(val)
    assert offsets(train) | offsets(val) == offsets(data)
    assert offsets(train) == offsets(again)
    assert train.task == data.task and train.spec == data.spec

    with pytest.raises(ValueError, match="at least two"):
        _dataset(n=1).split(0.5)
    with pytest.raises(ValueError, match="val_fraction"):
        data.split(1.0)


def test_save_and_load_round_trip_through_an_episode_folder(tmp_path):
    data = _dataset(n=3)
    data.camera, data.action_repeat = 1, 2

    data.save(tmp_path / "probe")
    loaded = ProbeDataset.load(tmp_path / "probe")

    assert isinstance(loaded.episodes, EpisodeFolder)
    assert (loaded.task, loaded.camera, loaded.action_repeat) == ("walker-walk", 1, 2)
    assert loaded.spec == _SPEC
    np.testing.assert_allclose(loaded.targets().values, data.targets().values)
    train, val = loaded.split(0.34)
    assert isinstance(train.episodes, EpisodeFolder)
    assert len(train) + len(val) == 3


def test_load_rejects_a_folder_without_a_spec(tmp_path):
    _episode(3).save(tmp_path / "episode_000000.npz")
    with pytest.raises(ValueError, match="physics state spec"):
        ProbeDataset.load(tmp_path)


def test_collect_records_true_state_aligned_with_frames():
    pytest.importorskip("dm_control")

    seen = []

    def policy(obs):
        seen.append(obs["image"].shape)
        return np.full(1, 0.5, dtype=np.float32)

    data = collect_probe_episodes(
        "cartpole-swingup",
        2,
        policy=policy,
        size=(16, 16),
        action_repeat=2,
        max_steps=6,
    )

    assert data.task == "cartpole-swingup" and data.action_repeat == 2
    assert data.spec.qpos_names == ("slider", "hinge_1")
    assert len(data) == 2 and len(seen) == 12
    episode = data[0]
    assert episode.frames.shape == (7, 16, 16, 3)
    assert episode.actions.shape == (6, 1)
    assert episode.observations["qpos"].shape == (7, 2)
    assert episode.observations["task.position"].shape == (7, 3)
    # Pushing right the whole time moves the cart right.
    assert episode.observations["qpos"][-1, 0] > episode.observations["qpos"][0, 0]
    # cartpole's own observation is (cart x, cos pole, sin pole): the same state.
    position = episode.observations["task.position"]
    qpos = episode.observations["qpos"]
    np.testing.assert_allclose(position[:, 0], qpos[:, 0], atol=1e-5)
    np.testing.assert_allclose(position[:, 1], np.cos(qpos[:, 1]), atol=1e-5)
    assert data.targets().values.shape == (14, 5)


def test_collect_random_policy_is_seeded_and_validates_arguments():
    pytest.importorskip("dm_control")

    first = collect_probe_episodes(
        "pendulum-swingup", 1, seed=3, size=(8, 8), max_steps=4
    )
    second = collect_probe_episodes(
        "pendulum-swingup", 1, seed=3, size=(8, 8), max_steps=4
    )
    np.testing.assert_array_equal(first[0].actions, second[0].actions)
    np.testing.assert_array_equal(
        first[0].observations["qpos"], second[0].observations["qpos"]
    )

    with pytest.raises(ValueError, match="policy"):
        collect_probe_episodes("pendulum-swingup", 1, policy="greedy")
    with pytest.raises(ValueError, match="episodes"):
        collect_probe_episodes("pendulum-swingup", 0)
