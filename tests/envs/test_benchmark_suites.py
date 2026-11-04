"""BSuite and DMLab-30 adapters follow the benchmark protocol of DreamerV3's paper.

The BSuite tests that load real tasks run when the optional ``bsuite`` package
is installed. DeepMind Lab only builds on Linux, so its tests use a fake
``deepmind_lab`` module with the native API.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from simulusion.configs.dreamer_config import DreamerConfig  # noqa: E402
from simulusion.envs.bsuite_env import (  # noqa: E402
    BSuiteImageEnv,
    list_bsuite_settings,
    normalize_bsuite_id,
)

# --------------------------------------------------------------------------- #
# BSuite
# --------------------------------------------------------------------------- #


class _Spec:
    num_values = 3


class _Step:
    def __init__(self, observation, reward=0.0, last=False):
        self.observation = observation
        self.reward = reward
        self.discount = 0.0 if last else 1.0
        self._last = last

    def last(self):
        return self._last


class _FakeBSuite:
    """Two-step episodes, a budget of two episodes."""

    bsuite_num_episodes = 2

    def __init__(self):
        self.t = 0

    def action_spec(self):
        return _Spec()

    def reset(self):
        self.t = 0
        return _Step(np.zeros((2, 3), dtype=np.float32))

    def step(self, action):
        self.t += 1
        return _Step(np.ones((2, 3), dtype=np.float32), 1.0, last=self.t >= 2)


def test_bare_bsuite_names_mean_the_first_setting():
    assert normalize_bsuite_id("catch") == "catch/0"
    assert normalize_bsuite_id("deep_sea/12") == "deep_sea/12"
    env = BSuiteImageEnv("catch", env=_FakeBSuite())
    assert env.bsuite_id == "catch/0"


def test_state_observations_are_the_flat_vector_only():
    env = BSuiteImageEnv("catch/0", env=_FakeBSuite(), obs_type="state")
    assert set(env.observation_space.spaces) == {"state"}
    obs = env.reset()
    assert set(obs) == {"state"} and obs["state"].shape == (6,)
    both = BSuiteImageEnv("catch/0", env=_FakeBSuite(), obs_type="both")
    assert set(both.reset()) == {"image", "state"}
    with pytest.raises(ValueError, match="obs_type"):
        BSuiteImageEnv("catch/0", env=_FakeBSuite(), obs_type="rgb")


def test_the_episode_budget_is_tracked():
    env = BSuiteImageEnv("catch/0", env=_FakeBSuite(), obs_type="state")
    assert env.num_episodes == 2 and not env.budget_exhausted
    action = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    for _ in range(2):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(action)
    assert env.episodes_completed == 2 and env.budget_exhausted
    # One episode bounds the steps; the run's length is the episode budget.
    assert env.max_episode_steps == 1000


def test_the_action_space_is_marked_one_hot():
    env = BSuiteImageEnv("catch/0", env=_FakeBSuite(), obs_type="state")
    assert env.action_space.one_hot and env.action_space.shape == (3,)


def test_dreamer_stops_once_the_bsuite_budget_is_spent():
    from simulusion.training.tasks.dreamer import DreamerTask

    env = BSuiteImageEnv("catch/0", env=_FakeBSuite(), obs_type="state")
    task = DreamerTask.__new__(DreamerTask)
    task.train_env = env
    assert not task.should_stop()
    env.episodes_completed = 2
    assert task.should_stop()
    task.train_env = types.SimpleNamespace()  # other backends never stop early
    assert not task.should_stop()


def test_training_counts_environment_frames():
    # DMLab repeats each action 4 Lab frames; budgets and logs count frames.
    from simulusion.training.tasks.dreamer import DreamerTask

    task = DreamerTask.__new__(DreamerTask)
    task.config = types.SimpleNamespace(model=DreamerConfig(action_repeat=1))
    task.train_env = types.SimpleNamespace(native_action_repeat=4)
    assert task._frames_per_step() == 4
    task.config.model.action_repeat = 2
    task.train_env = types.SimpleNamespace()
    assert task._frames_per_step() == 2


def _require_bsuite():
    return pytest.importorskip("bsuite")


def _bsuite_environments():
    from bsuite import sweep

    return sorted({bsuite_id.split("/")[0] for bsuite_id in sweep.SWEEP})


def test_every_bsuite_environment_runs_from_its_vector():
    _require_bsuite()
    names = _bsuite_environments()
    assert len(names) == 23
    for name in names:
        env = BSuiteImageEnv(name, obs_type="state")
        obs = env.reset()
        assert obs["state"].ndim == 1 and obs["state"].dtype == np.float32
        for _ in range(3):
            obs, reward, done, info = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            assert obs["state"].shape == env.observation_space["state"].shape
            if done:
                env.reset()
        assert env.num_episodes in (1000, 10_000), name


def test_bsuite_has_468_settings():
    _require_bsuite()
    names = _bsuite_environments()
    settings = [s for name in names for s in list_bsuite_settings(name)]
    assert len(settings) == len(set(settings)) == 468
    assert list_bsuite_settings("catch")[:2] == ["catch/0", "catch/1"]
    with pytest.raises(ValueError, match="Unknown BSuite environment"):
        list_bsuite_settings("not_an_env")


def test_bsuite_records_results_for_scoring(tmp_path):
    _require_bsuite()
    env = BSuiteImageEnv("bandit/0", obs_type="state", results_dir=str(tmp_path))
    for _ in range(3):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
    assert env.episodes_completed == 3 and env.num_episodes == 10_000
    assert list(tmp_path.glob("*.csv"))


# --------------------------------------------------------------------------- #
# DMLab
# --------------------------------------------------------------------------- #


class _FakeLab:
    instances: list["_FakeLab"] = []

    def __init__(self, level, observations, config=None, renderer=None):
        self.level = level
        self.observation_names = list(observations)
        self.config = dict(config or {})
        self.actions: list[np.ndarray] = []
        self.steps = 0
        _FakeLab.instances.append(self)

    def observation_spec(self):
        return [
            {"name": "RGB_INTERLEAVED", "shape": (64, 64, 3), "dtype": np.uint8},
            {"name": "INSTR", "shape": (), "dtype": str},
        ]

    def reset(self, seed=None):
        self.steps = 0

    def observations(self):
        out = {"RGB_INTERLEAVED": np.zeros((64, 64, 3), dtype=np.uint8)}
        if "INSTR" in self.observation_names:
            out["INSTR"] = "Pick the red balloon."
        return out

    def step(self, action, num_steps=1):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        return 1.0

    def is_running(self):
        return self.steps < 3

    def close(self):
        pass


@pytest.fixture
def fake_lab(monkeypatch):
    module = types.ModuleType("deepmind_lab")
    module.Lab = _FakeLab
    monkeypatch.setitem(sys.modules, "deepmind_lab", module)
    _FakeLab.instances.clear()
    return _FakeLab


def test_popart_is_the_15_action_set(fake_lab):
    from simulusion.envs.dmlab import POPART_ACTION_SET, DMLabEnv

    env = DMLabEnv(
        "contributed/dmlab30/explore_goal_locations_small", action_set="popart"
    )
    assert env.action_space.shape == (15,) and env.action_space.one_hot
    env.reset()
    env.step(np.eye(15, dtype=np.float32)[6])  # large look left
    assert fake_lab.instances[-1].actions[-1].tolist() == POPART_ACTION_SET[6].tolist()
    assert POPART_ACTION_SET[6].tolist() == [-60, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="Unknown DMLab action set"):
        DMLabEnv("rooms_watermaze", action_set="r2d3")


def test_language_levels_add_the_embedded_instruction(fake_lab):
    from simulusion.envs.dmlab import DMLabEnv, embed_instruction

    env = DMLabEnv("contributed/dmlab30/language_select_described_object")
    assert "INSTR" in fake_lab.instances[-1].observation_names
    assert env.observation_space["instr"].shape == (1024,)
    assert "INSTR" not in env.observation_space.spaces
    obs = env.reset()
    assert obs["instr"].shape == (1024,) and obs["instr"].dtype == np.float32
    np.testing.assert_array_equal(
        obs["instr"], embed_instruction("Pick the red balloon.")
    )
    # Navigation levels have no instruction.
    plain = DMLabEnv("contributed/dmlab30/rooms_watermaze")
    assert "instr" not in plain.reset()


def test_instruction_embedding_matches_the_reference():
    from simulusion.envs.dmlab import embed_instruction

    vec = embed_instruction("Pick the red balloon.")
    tokens = vec.reshape(32, 32)
    # "pick", "the", "red", "balloon", "." then padding with bucket 0.
    assert not np.allclose(tokens[4], tokens[5])
    np.testing.assert_array_equal(tokens[5], tokens[31])
    np.testing.assert_array_equal(embed_instruction("PICK"), embed_instruction("pick"))


def test_train_and_eval_modes(fake_lab):
    from simulusion.envs.dmlab import DMLabEnv

    DMLabEnv("contributed/dmlab30/rooms_collect_good_objects_test", mode="train")
    lab = fake_lab.instances[-1]
    assert lab.level.endswith("rooms_collect_good_objects_train")
    assert "allowHoldOutLevels" not in lab.config

    DMLabEnv("contributed/dmlab30/rooms_collect_good_objects_test", mode="eval")
    lab = fake_lab.instances[-1]
    assert lab.level.endswith("_test")
    assert lab.config["allowHoldOutLevels"] == "true"
    assert lab.config["mixerSeed"] == str(0x600D5EED)


def test_dmlab30_names_are_found_in_their_folder(fake_lab):
    from simulusion.envs.dmlab import DMLAB30_LEVELS, DMLabEnv, resolve_dmlab_level

    assert len(DMLAB30_LEVELS) == 32  # 30 tasks, two with train and test levels
    assert len(set(DMLAB30_LEVELS)) == 32
    DMLabEnv("explore_goal_locations_small")
    assert fake_lab.instances[-1].level == (
        "contributed/dmlab30/explore_goal_locations_small"
    )
    # Full paths and other levels are left alone.
    assert resolve_dmlab_level("contributed/dmlab30/rooms_watermaze") == (
        "contributed/dmlab30/rooms_watermaze"
    )
    assert resolve_dmlab_level("seekavoid_arena_01") == "seekavoid_arena_01"


def test_make_env_follows_the_dmlab30_protocol_by_default(fake_lab):
    # Reference: popart actions, train mode, 4 Lab frames per step, no
    # outer action repeat and no time limit (levels end their own episodes).
    from simulusion.models.dreamer import make_env

    cfg = DreamerConfig(
        env_backend="dmlab",
        env="rooms_collect_good_objects_test",
        action_repeat=1,
        time_limit=2,
    )
    env = make_env(cfg)
    lab = fake_lab.instances[-1]
    assert lab.level == "contributed/dmlab30/rooms_collect_good_objects_train"
    assert env.action_space.shape == (15,)
    assert env.native_action_repeat == 4
    env.reset()
    dones = [env.step(env.action_space.sample())[2] for _ in range(3)]
    assert dones == [False, False, True]  # the level's end, not time_limit=2


def test_make_env_passes_the_dmlab_options(fake_lab):
    from simulusion.models.dreamer import make_env

    cfg = DreamerConfig(
        env_backend="dmlab",
        env="language_execute_random_task",
        action_repeat=1,
        dmlab_action_set="impala",
        dmlab_action_repeat=2,
        dmlab_mode="eval",
        dmlab_renderer="software",
    )
    env = make_env(cfg)
    assert env.action_space.shape == (9,)
    assert env.native_action_repeat == 2
    assert {"image", "instr"} <= set(env.observation_space.spaces)
    lab = fake_lab.instances[-1]
    assert "INSTR" in lab.observation_names
    assert lab.config["allowHoldOutLevels"] == "true"


@pytest.mark.parametrize(
    ("backend", "module", "cls"),
    [
        ("dmlab", "simulusion.envs.dmlab", "DMLabEnv"),
        ("bsuite", "simulusion.envs.bsuite_env", "BSuiteImageEnv"),
        ("minecraft", "simulusion.envs.minecraft_diamond", "MinecraftDiamondEnv"),
    ],
)
def test_benchmarks_reject_an_outer_action_repeat(monkeypatch, backend, module, cls):
    # The reference repeats inside the env (DMLab) or not at all.
    import importlib

    from simulusion.models.dreamer import make_env

    monkeypatch.setattr(
        importlib.import_module(module), cls, lambda *a, **k: types.SimpleNamespace()
    )
    with pytest.raises(ValueError, match="action_repeat must be 1"):
        make_env(DreamerConfig(env_backend=backend, action_repeat=2))


def test_make_env_passes_the_bsuite_options(tmp_path):
    _require_bsuite()
    from simulusion.models.dreamer import make_env

    cfg = DreamerConfig(
        env_backend="bsuite",
        env="catch",
        action_repeat=1,
        bsuite_obs_type="state",
        bsuite_results_dir=str(tmp_path),
    )
    env = make_env(cfg)
    assert set(env.observation_space.spaces) == {"state"}
    assert set(env.reset()) == {"state"}


def test_only_the_training_env_records_bsuite_results(tmp_path):
    # Train and test envs writing one CSV would overwrite each other's episodes.
    _require_bsuite()
    from simulusion.models.dreamer import make_env

    cfg = DreamerConfig(
        env_backend="bsuite",
        env="bandit",
        action_repeat=1,
        bsuite_results_dir=str(tmp_path),
    )
    train_env, test_env = make_env(cfg), make_env(cfg, evaluation=True)
    for env, episodes in ((train_env, 3), (test_env, 2)):
        for _ in range(episodes):
            env.reset()
            done = False
            while not done:
                _, _, done, _ = env.step(env.action_space.sample())
    (csv,) = tmp_path.glob("*.csv")
    import pandas as pd

    assert pd.read_csv(csv)["episode"].max() == 3
