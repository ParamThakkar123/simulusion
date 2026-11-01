import os
import sys
from importlib import metadata

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _protobuf_version() -> tuple[int, int]:
    try:
        major, minor = metadata.version("protobuf").split(".")[:2]
        return int(major), int(minor)
    except (metadata.PackageNotFoundError, ValueError):
        return (0, 0)


def _assert_image_observation(
    obs: dict[str, np.ndarray], shape: tuple[int, int, int]
) -> None:
    assert set(obs) >= {"image"}
    assert obs["image"].shape == shape
    assert obs["image"].dtype == np.uint8
    assert int(obs["image"].min()) >= 0
    assert int(obs["image"].max()) <= 255


def test_real_brax_image_env_smoke():
    pytest.importorskip("brax")
    from simulusion.envs.brax_env import BraxImageEnv

    env = BraxImageEnv(
        "inverted_pendulum",
        seed=0,
        size=(16, 16),
        jit=False,
        include_state=True,
    )
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 16, 16))
        assert obs["state"].dtype == np.float32
        assert env.observation_space.contains(obs)

        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 16, 16))
        assert next_obs["state"].dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
        assert "action" in info
        assert "vector_observation" in info

        frame = env.render()
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
    finally:
        env.close()


def test_real_robotics_env_smoke():
    pytest.importorskip("gymnasium_robotics")
    from simulusion.envs.robotics_env import (
        list_gymnasium_robotics_envs,
        make_robotics_env,
    )

    env_ids = list_gymnasium_robotics_envs()
    assert env_ids, "Gymnasium Robotics did not register any environments."

    preferred_ids = (
        "FetchReach-v4",
        "FetchReachDense-v4",
        "AdroitHandDoor-v1",
    )
    env_id = next(
        (candidate for candidate in preferred_ids if candidate in env_ids), env_ids[0]
    )
    env = make_robotics_env(env_id, seed=0, size=(16, 16), render_mode="rgb_array")
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 16, 16))
        assert env.observation_space.contains(obs)

        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 16, 16))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
        assert "action" in info
        assert "vector_observation" in info

        frame = env.render()
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
    finally:
        env.close()


# The `ml-agents` extra deliberately resolves protobuf >= 5.29.6 (see the
# comment above it in pyproject.toml). mlagents-envs 0.28's gencode predates
# protoc 3.19 and cannot load on any protobuf >= 3.21: the C++/upb backends
# raise TypeError and the pure-Python backend raises AttributeError.
@pytest.mark.xfail(
    _protobuf_version() >= (3, 21),
    reason="mlagents-envs 0.28 gencode is incompatible with protobuf >= 3.21",
    raises=(TypeError, AttributeError),
    strict=True,
)
def test_real_mlagents_sdk_importable():
    sdk = pytest.importorskip("mlagents_envs")
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )

    assert UnityEnvironment is not None
    assert EngineConfigurationChannel is not None
    assert getattr(sdk, "__version__", "")


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="Upstream Procgen wheels currently support Python 3.10 and below.",
)
def test_real_procgen_backend_if_supported_python():
    pytest.importorskip("procgen")
    from simulusion.envs.procgen_env import ProcgenImageEnv

    env = ProcgenImageEnv("coinrun", seed=0, size=(16, 16), num_levels=1)
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 16, 16))
        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 16, 16))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
    finally:
        env.close()


@pytest.mark.skipif(
    not os.getenv("SIMULUSION_UNITY_BINARY")
    or not os.getenv("SIMULUSION_UNITY_BEHAVIOR"),
    reason="Set SIMULUSION_UNITY_BINARY and SIMULUSION_UNITY_BEHAVIOR to run against a real Unity build.",
)
def test_real_unity_backend_if_binary_provided():
    from simulusion.envs.unity_env import UnityMLAgentsEnv

    env = UnityMLAgentsEnv(
        file_name=os.environ["SIMULUSION_UNITY_BINARY"],
        behavior_name=os.environ["SIMULUSION_UNITY_BEHAVIOR"],
        seed=0,
        size=(32, 32),
        no_graphics=True,
        include_state=True,
    )
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 32, 32))
        assert env.observation_space.contains(obs)

        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 32, 32))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
        assert "action" in info
    finally:
        env.close()


def test_real_dmc_backend_smoke():
    """DeepMind Control through the dmc backend, when it is installed.

    `pip install simulusion[dmc]` covers Python <= 3.12; on 3.13 the backend is
    installed by `python -m simulusion.install_dmc`, which supplies the `labmaze`
    module through the pure-Python `labmaze-new`.
    """
    pytest.importorskip("dm_control")

    import simulusion

    env = simulusion.make_env("walker-walk", backend="dmc", seed=0, size=(32, 32))
    obs = env.reset()
    _assert_image_observation(obs, (3, 32, 32))
    # A rendered frame, not a blank one: offscreen GL actually worked.
    assert int(obs["image"].max()) > 0

    next_obs, reward, done, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    _assert_image_observation(next_obs, (3, 32, 32))
    assert isinstance(float(reward), float)
    assert done is False
    assert {"discount", "terminated", "truncated"} <= set(info)


def test_real_dmc_physics_state_matches_simulator():
    """``physics_state`` and its spec line up with MuJoCo's own layout."""
    pytest.importorskip("dm_control")

    from simulusion.envs.dmc import DeepMindControlEnv

    env = DeepMindControlEnv("walker-walk", seed=0, size=(8, 8))
    env.reset()
    _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    physics = env._env.physics
    np.testing.assert_allclose(info["physics_state"]["qpos"], physics.data.qpos)
    np.testing.assert_allclose(info["physics_state"]["qvel"], physics.data.qvel)

    spec = env.physics_state_spec()
    assert spec.qpos_names == tuple(physics.named.data.qpos.axes.row.names)
    assert spec.qpos_names[:3] == ("rootz", "rootx", "rooty")
    assert spec.qpos_kinds[:3] == ("slide", "slide", "hinge")
    assert spec.act_names == ()

    # Quadruped has a free root joint (7 qpos, 6 qvel) and activated actuators.
    quadruped = DeepMindControlEnv("quadruped-walk", seed=0, size=(8, 8))
    state, spec = quadruped.physics_state(), quadruped.physics_state_spec()
    assert len(spec.qpos_names) == state["qpos"].shape[0] == 23
    assert len(spec.qvel_names) == state["qvel"].shape[0] == 22
    assert len(spec.act_names) == state["act"].shape[0] == 12
    assert spec.qpos_names[:7] == (
        "root/x",
        "root/y",
        "root/z",
        "root/qw",
        "root/qx",
        "root/qy",
        "root/qz",
    )
    assert spec.qpos_kinds[3:7] == ("free_quat",) * 4
    assert spec.qvel_kinds[:6] == ("free_lin",) * 3 + ("free_ang",) * 3


def test_dmc_installer_never_resolves_upstream_labmaze():
    from simulusion.install_dmc import DM_CONTROL_DEPS, install_steps

    steps = install_steps()
    assert any("labmaze-new" in arg for step in steps for arg in step)
    dm_control_step = next(
        step for step in steps if any("dm-control" in a for a in step)
    )
    assert "--no-deps" in dm_control_step
    # Upstream `labmaze` must never be requested: it has no CPython 3.13 wheel
    # and its sdist builds with Bazel.
    for step in steps:
        for arg in step:
            assert not arg.startswith("labmaze>") and arg != "labmaze"
    assert not any(dep.startswith("labmaze") for dep in DM_CONTROL_DEPS)
