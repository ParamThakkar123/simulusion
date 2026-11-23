"""DreamerV3 trains on BSuite, DMLab-30 and Minecraft Diamond as in the paper.

The environments are fakes of the native APIs (see tests/envs); the BSuite
tests that load real tasks run when the optional ``bsuite`` package is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from simulusion.configs.dreamer_v3_benchmarks import (  # noqa: E402
    BSUITE_ENVIRONMENTS,
    benchmark_config,
    bsuite_settings,
)
from simulusion.envs.bsuite_env import BSuiteImageEnv  # noqa: E402
from simulusion.models.dreamer import make_env  # noqa: E402
from simulusion.models.dreamer_v3 import (  # noqa: E402
    _resolve_actor_dist,
    select_obs_space,
)
from tests.envs.test_benchmark_suites import _FakeBSuite, fake_lab  # noqa: E402,F401
from tests.envs.test_minecraft_diamond import FakeMineRL  # noqa: E402


def test_one_hot_backends_get_a_categorical_policy():
    from simulusion.envs.wrappers import ActionRepeat, NormalizeActions, TimeLimit

    env = BSuiteImageEnv("catch/0", env=_FakeBSuite(), obs_type="state")
    wrapped = TimeLimit(NormalizeActions(ActionRepeat(env, 1)), 100)
    assert _resolve_actor_dist(wrapped) == "onehot"


@pytest.mark.parametrize("name", BSUITE_ENVIRONMENTS)
def test_every_bsuite_environment_runs_from_the_preset(name):
    pytest.importorskip("bsuite")
    cfg = benchmark_config("bsuite", name)
    assert cfg.bsuite_obs_type == "state" and cfg.env == f"{name}/0"
    env = BSuiteImageEnv(cfg.env, obs_type=cfg.bsuite_obs_type)
    obs = env.reset()
    assert obs["state"].ndim == 1 and obs["state"].dtype == np.float32


def test_the_bsuite_preset_covers_468_settings():
    pytest.importorskip("bsuite")
    settings = [s for name in BSUITE_ENVIRONMENTS for s in bsuite_settings(name)]
    assert len(settings) == len(set(settings)) == 468


def test_dreamer_v3_builds_dmlab_from_the_preset(fake_lab):  # noqa: F811
    cfg = benchmark_config("dmlab", "language_execute_random_task")
    assert cfg.dmlab_action_set == "popart" and cfg.dmlab_mode == "train"
    env = make_env(cfg)
    assert _resolve_actor_dist(env) == "onehot"
    assert env.action_space.shape == (15,)
    assert list(select_obs_space(env.observation_space)) == ["image", "instr"]


def test_dreamer_v3_sees_minecraft_as_categorical_with_image_and_vectors(
    monkeypatch,
):
    from simulusion.envs import minecraft_diamond

    backend = FakeMineRL()
    monkeypatch.setattr(
        minecraft_diamond, "make_minerl_backend", lambda size, speed: backend
    )
    cfg = benchmark_config("minecraft", model_size="12m")
    env = make_env(cfg)
    assert _resolve_actor_dist(env) == "onehot"
    obs_space = select_obs_space(env.observation_space, cfg.obs_keys)
    assert list(obs_space) == cfg.obs_keys
    env.reset()
    obs, _, done, _ = env.step(env.action_space.sample())
    assert set(cfg.obs_keys) <= set(obs) and not done
