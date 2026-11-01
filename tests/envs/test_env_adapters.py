from typing import Any

import pytest
import numpy as np
from unittest.mock import Mock, patch

import torch

gym = pytest.importorskip("gymnasium")

from simulusion.configs.dreamer_config import DreamerConfig  # noqa: E402
from simulusion.envs.gym_env import GymImageEnv  # noqa: E402
from simulusion.envs.wrappers import FrameStack  # noqa: E402
from simulusion.models.dreamer import make_env  # noqa: E402
from simulusion.utils.utils import TorchImageEnvWrapper  # noqa: E402


class _FakeDiscreteEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.last_action = None
        self.spec = type("Spec", (), {"max_episode_steps": 5})()

    def reset(self, seed=None):
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), {}

    def step(self, action):
        self.last_action = action
        obs = np.array([0.2, 0.1, 0.0, -0.1], dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        pass


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.dmc.DeepMindControlEnv")
def test_make_env_dmc_backend(
    mock_dmc,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "dmc"
    cfg.env = "walker-walk"
    cfg.image_size = (64, 64)

    env = Mock()
    mock_dmc.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_dmc.assert_called_once_with(cfg.env, cfg.seed, size=cfg.image_size)


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.FrameStack")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.gym_env.GymImageEnv")
def test_make_env_gym_backend(
    mock_gym_env,
    mock_repeat,
    mock_normalize,
    mock_frame_stack,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "gym"
    cfg.env = "Pendulum-v1"
    cfg.image_size = (64, 64)
    cfg.gym_render_mode = "rgb_array"
    cfg.frame_stack = 4

    env = Mock()
    mock_gym_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_frame_stack.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_frame_stack.assert_called_once_with(env, 4)
    mock_gym_env.assert_called_once_with(
        cfg.env,
        seed=cfg.seed,
        size=cfg.image_size,
        render_mode=cfg.gym_render_mode,
    )


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.unity_env.UnityMLAgentsEnv")
def test_make_env_unity_backend(
    mock_unity_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "unity_mlagents"
    cfg.unity_file_name = "fake.exe"
    cfg.unity_behavior_name = "Behavior"

    env = Mock()
    mock_unity_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    assert mock_unity_env.call_count == 1
    call_kwargs = mock_unity_env.call_args.kwargs
    assert call_kwargs["file_name"] == cfg.unity_file_name
    assert call_kwargs["behavior_name"] == cfg.unity_behavior_name


def test_unity_mlagents_env_contract_with_fake_sdk(monkeypatch):
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType, SimpleNamespace

    from simulusion.envs.unity_env import UnityMLAgentsEnv

    class _FakeActionTuple:
        def __init__(self, continuous):
            self.continuous = np.asarray(continuous, dtype=np.float32)

    class _FakeEngineConfigurationChannel:
        def __init__(self):
            self.params = None

        def set_configuration_parameters(self, **kwargs):
            self.params = kwargs

    class _FakeActionSpec:
        continuous_size = 2

        @staticmethod
        def is_continuous():
            return True

    class _FakeSteps:
        def __init__(self, agent_ids=(), obs=None, reward=None, interrupted=None):
            self.agent_id = np.asarray(agent_ids, dtype=np.int64)
            self.obs = list(obs or [])
            self.reward = np.asarray(
                reward
                if reward is not None
                else np.zeros(len(self.agent_id), dtype=np.float32),
                dtype=np.float32,
            )
            self.interrupted = np.asarray(
                interrupted
                if interrupted is not None
                else np.zeros(len(self.agent_id), dtype=bool),
                dtype=bool,
            )

    class _FakeUnityEnvironment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.behavior_specs = {
                "Behavior": SimpleNamespace(
                    action_spec=_FakeActionSpec(),
                    observation_specs=[
                        SimpleNamespace(shape=(8, 8, 3)),
                        SimpleNamespace(shape=(4,)),
                    ],
                )
            }
            self._step_count = 0
            self.last_action = None
            self.closed = False

        def reset(self):
            self._step_count = 0

        def get_steps(self, behavior_name):
            assert behavior_name == "Behavior"
            if self._step_count == 0:
                decision = _FakeSteps(
                    agent_ids=[11],
                    obs=[
                        np.full((1, 8, 8, 3), 32, dtype=np.uint8),
                        np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                    ],
                    reward=[0.0],
                )
                terminal = _FakeSteps()
                return decision, terminal
            decision = _FakeSteps()
            terminal = _FakeSteps(
                agent_ids=[11],
                obs=[
                    np.full((1, 8, 8, 3), 96, dtype=np.uint8),
                    np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                ],
                reward=[1.5],
                interrupted=[True],
            )
            return decision, terminal

        def set_actions(self, behavior_name, action_tuple):
            assert behavior_name == "Behavior"
            self.last_action = np.asarray(action_tuple.continuous, dtype=np.float32)

        def step(self):
            self._step_count += 1

        def close(self):
            self.closed = True

    root = ModuleType("mlagents_envs")
    root.__spec__ = ModuleSpec("mlagents_envs", loader=None)
    base_env = ModuleType("mlagents_envs.base_env")
    base_env.__spec__ = ModuleSpec("mlagents_envs.base_env", loader=None)
    base_env.ActionTuple = _FakeActionTuple
    environment = ModuleType("mlagents_envs.environment")
    environment.__spec__ = ModuleSpec("mlagents_envs.environment", loader=None)
    environment.UnityEnvironment = _FakeUnityEnvironment
    side_channel = ModuleType("mlagents_envs.side_channel")
    side_channel.__spec__ = ModuleSpec("mlagents_envs.side_channel", loader=None)
    engine = ModuleType("mlagents_envs.side_channel.engine_configuration_channel")
    engine.__spec__ = ModuleSpec(
        "mlagents_envs.side_channel.engine_configuration_channel", loader=None
    )
    engine.EngineConfigurationChannel = _FakeEngineConfigurationChannel

    monkeypatch.setitem(sys.modules, "mlagents_envs", root)
    monkeypatch.setitem(sys.modules, "mlagents_envs.base_env", base_env)
    monkeypatch.setitem(sys.modules, "mlagents_envs.environment", environment)
    monkeypatch.setitem(sys.modules, "mlagents_envs.side_channel", side_channel)
    monkeypatch.setitem(
        sys.modules,
        "mlagents_envs.side_channel.engine_configuration_channel",
        engine,
    )

    env = UnityMLAgentsEnv(
        file_name="fake.exe",
        behavior_name="Behavior",
        seed=5,
        size=(8, 8),
        include_state=True,
        no_graphics=False,
    )

    obs = env.reset()
    assert set(obs) == {"image", "state"}
    assert obs["image"].shape == (3, 8, 8)
    assert obs["image"].dtype == np.uint8
    assert obs["state"].shape == (4,)
    assert env.observation_space.contains(obs)
    assert np.array_equal(env.render(), obs["image"].transpose(1, 2, 0))

    next_obs, reward, done, info = env.step(np.array([2.0, -3.0], dtype=np.float32))

    assert next_obs["image"].shape == (3, 8, 8)
    assert next_obs["state"].shape == (4,)
    assert reward == 1.5
    assert done is True
    assert np.array_equal(
        env._env.last_action, np.array([[1.0, -1.0]], dtype=np.float32)
    )
    assert np.array_equal(info["action"], np.array([1.0, -1.0], dtype=np.float32))
    assert np.array_equal(
        info["vector_observation"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
    assert info["discount"] == np.array(0.0, dtype=np.float32)
    assert info["terminated"] is False
    assert info["truncated"] is True
    assert info["interrupted"] is True
    assert env.observation_space.contains(next_obs)

    env.close()
    assert env._env.closed is True


def test_gym_image_env_discrete_action_mapping():
    wrapped = GymImageEnv(_FakeDiscreteEnv(), seed=1, size=(64, 64))
    obs = wrapped.reset()
    assert obs["image"].shape == (3, 64, 64)

    action = np.array([-0.2, 0.7, 0.1], dtype=np.float32)
    _, reward, done, info = wrapped.step(action)

    assert wrapped._env.last_action == 1
    assert reward == 1.0
    assert done is False
    assert info["action"].shape == (3,)
    assert np.array_equal(info["action"], np.array([-1.0, 1.0, -1.0], dtype=np.float32))
    assert info["executed_action"] == 1


class _FakeRenderingVectorEnv(_FakeDiscreteEnv):
    """Vector observations, but render() draws a real scene."""

    def render(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[8:24, 14:18] = (200, 30, 30)  # a "pole"
        return frame


def test_gym_image_env_prefers_render_frame_over_vector_bands():
    wrapped = GymImageEnv(_FakeRenderingVectorEnv(), seed=1, size=(32, 32))
    obs = wrapped.reset()
    expected = _FakeRenderingVectorEnv().render().transpose(2, 0, 1)
    assert np.array_equal(obs["image"], expected)

    next_obs, _, _, _ = wrapped.step(np.array([1.0, -1.0, -1.0], dtype=np.float32))
    assert np.array_equal(next_obs["image"], expected)


class _FakeNonRenderingVectorEnv(_FakeDiscreteEnv):
    """Like a gymnasium env made without render_mode: render() gives None."""

    def render(self, *args, **kwargs) -> Any:
        return None


def test_gym_image_env_warns_once_when_falling_back_to_bands():
    wrapped = GymImageEnv(_FakeNonRenderingVectorEnv(), seed=1, size=(8, 8))
    with pytest.warns(UserWarning, match="grey bands"):
        obs = wrapped.reset()
    assert obs["image"].shape == (3, 8, 8)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wrapped.step(np.array([1.0, -1.0, -1.0], dtype=np.float32))


def test_gym_image_env_renders_pendulum_pixels():
    env = GymImageEnv("Pendulum-v1", seed=0, size=(64, 64))
    image = env.reset()["image"]
    env.close()
    # Pendulum draws a red pole on a white background; bands would be grey.
    red = image[0].astype(int) - image[1].astype(int)
    assert red.max() > 100


def test_normalize_actions_wrapper_reports_model_and_executed_actions():
    class _ContinuousEnv:
        def __init__(self):
            self.action_space = gym.spaces.Box(
                low=np.array([-2.0, -4.0], dtype=np.float32),
                high=np.array([2.0, 4.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0, high=255, shape=(3, 2, 2), dtype=np.uint8
                    )
                }
            )
            self.last_action = None

        def reset(self):
            return {"image": np.zeros((3, 2, 2), dtype=np.uint8)}

        def step(self, action):
            self.last_action = np.asarray(action, dtype=np.float32)
            return (
                {"image": np.ones((3, 2, 2), dtype=np.uint8)},
                0.5,
                False,
                {"action": self.last_action.copy()},
            )

    from simulusion.envs.wrappers import NormalizeActions

    wrapped = NormalizeActions(_ContinuousEnv())
    _, reward, done, info = wrapped.step(np.array([2.0, -2.0], dtype=np.float32))

    assert reward == 0.5
    assert done is False
    assert np.array_equal(info["action"], np.array([1.0, -1.0], dtype=np.float32))
    assert np.array_equal(
        info["executed_action"], np.array([2.0, -4.0], dtype=np.float32)
    )
    assert np.array_equal(
        wrapped._env.last_action, np.array([2.0, -4.0], dtype=np.float32)
    )


def test_frame_stack_wrapper_shifts_frames_for_gym_image_env():
    wrapped = FrameStack(
        GymImageEnv(_FakeDiscreteEnv(), seed=1, size=(4, 4)), num_frames=2
    )

    obs = wrapped.reset()
    assert obs["image"].shape == (6, 4, 4)
    assert np.array_equal(obs["image"][0:3], obs["image"][3:6])

    next_obs, reward, done, info = wrapped.step(
        np.array([-0.2, 0.7, 0.1], dtype=np.float32)
    )
    assert reward == 1.0
    assert done is False
    assert isinstance(info, dict)
    assert next_obs["image"].shape == (6, 4, 4)
    assert np.array_equal(next_obs["image"][0:3], obs["image"][3:6])


class _FakeDictObsEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.spec = type("Spec", (), {"max_episode_steps": 8})()

    def reset(self):
        return {"image": np.zeros((3, 64, 64), dtype=np.uint8)}

    def step(self, action):
        obs = {"image": np.ones((3, 64, 64), dtype=np.uint8) * 32}
        return obs, 0.25, False, {}

    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        pass


def test_torch_image_wrapper_accepts_env_instances_and_dict_obs():
    wrapper = TorchImageEnvWrapper(_FakeDictObsEnv(), bit_depth=5)
    obs = wrapper.reset()
    assert torch.is_tensor(obs)
    assert obs.shape == (3, 64, 64)

    nobs, reward, done, info = wrapper.step(np.array([0.1, -0.2], dtype=np.float32))
    assert torch.is_tensor(nobs)
    assert nobs.shape == (3, 64, 64)
    assert reward == 0.5  # action_repeats defaults to 2
    assert done is False
    assert isinstance(info, dict)


class _FakeMjModel:
    nu = 2
    nq = 1
    nv = 1

    def __init__(self):
        self.actuator_ctrlrange = np.array([[-2.0, 2.0], [-0.5, 0.5]], dtype=np.float32)
        self.actuator_ctrllimited = np.array([True, True])

    @staticmethod
    def from_xml_string(xml, assets=None):
        assert "<mujoco" in xml
        return _FakeMjModel()

    @staticmethod
    def from_xml_path(path):
        return _FakeMjModel()

    @staticmethod
    def from_binary_path(path):
        return _FakeMjModel()


class _FakeMjData:
    def __init__(self, model):
        self.ctrl = np.zeros((model.nu,), dtype=np.float32)
        self.qpos = np.zeros((model.nq,), dtype=np.float64)
        self.qvel = np.zeros((model.nv,), dtype=np.float64)
        self.time = 0.0


class _FakeRenderer:
    def __init__(self, model, height, width):
        self.height = height
        self.width = width
        self.closed = False

    def update_scene(self, data, camera=None):
        self.camera = camera

    def render(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8) + 7

    def close(self):
        self.closed = True


def test_native_mujoco_image_env_with_mocked_bindings(monkeypatch):
    fake_mujoco = type("FakeMujoco", (), {})()
    fake_mujoco.MjModel = _FakeMjModel
    fake_mujoco.MjData = _FakeMjData
    fake_mujoco.Renderer = _FakeRenderer
    fake_mujoco.mj_resetData = lambda model, data: None
    fake_mujoco.mj_forward = lambda model, data: None

    def fake_step(model, data, nstep=1):
        data.time += 0.01 * nstep

    fake_mujoco.mj_step = fake_step
    monkeypatch.setitem(__import__("sys").modules, "mujoco", fake_mujoco)

    from simulusion.envs.mujoco_env import MuJoCoImageEnv

    env = MuJoCoImageEnv(
        xml_string="<mujoco/>",
        size=(32, 40),
        camera="track",
        frame_skip=3,
        reward_fn=lambda model, data, action, info: float(action.sum()),
        terminal_fn=lambda model, data, info: data.time > 0.02,
    )
    obs = env.reset(seed=123)
    assert obs["image"].shape == (3, 32, 40)
    assert env.action_space.shape == (2,)
    obs, reward, done, info = env.step(np.array([3.0, 0.25], dtype=np.float32))
    assert obs["image"].shape == (3, 32, 40)
    assert reward == 2.25
    assert done is True
    assert np.array_equal(info["action"], np.array([2.0, 0.25], dtype=np.float32))
    assert np.array_equal(
        info["executed_action"], np.array([2.0, 0.25], dtype=np.float32)
    )


def test_list_gymnasium_robotics_envs_uses_registered_package_envs(monkeypatch):
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType, SimpleNamespace

    import simulusion.envs.robotics_env as robotics_env

    fake_robotics = ModuleType("gymnasium_robotics")
    fake_robotics.__spec__ = ModuleSpec("gymnasium_robotics", loader=None)
    monkeypatch.setitem(sys.modules, "gymnasium_robotics", fake_robotics)
    monkeypatch.setattr(
        robotics_env.gym, "register_envs", lambda module: None, raising=False
    )
    monkeypatch.setattr(
        robotics_env.gym,
        "registry",
        {
            "FetchReachDense-v4": SimpleNamespace(
                entry_point="gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv"
            ),
            "AntMaze_UMazeDense-v5": SimpleNamespace(
                entry_point="gymnasium_robotics.envs.maze.ant_maze_v5:AntMazeEnv"
            ),
            "CartPole-v1": SimpleNamespace(
                entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv"
            ),
        },
    )

    assert robotics_env.list_gymnasium_robotics_envs() == [
        "AntMaze_UMazeDense-v5",
        "FetchReachDense-v4",
    ]


def test_environment_catalog_exposes_robotics_to_online_simulusion(monkeypatch):
    import simulusion.catalog as catalog

    monkeypatch.setattr(
        catalog, "_list_available_robotics_envs", lambda: ["FetchReachDense-v4"]
    )
    monkeypatch.setattr(catalog, "_list_available_atari_envs", lambda: ["ALE/Pong-v5"])

    envs_by_model = catalog._build_env_catalog()

    for model_name in (
        "dreamer",
        "dreamerv1",
        "dreamerv2",
        "planet",
        "rssm",
        "iris",
        "diamond",
        "genie",
        "dit",
    ):
        assert "FetchReachDense-v4" in envs_by_model[model_name]

    assert "jepa" not in envs_by_model
    assert "ijepa" not in envs_by_model


@patch("simulusion.envs.mujoco_env.GymImageEnv")
@patch("simulusion.envs.robotics_env.gym.make")
def test_make_mujoco_env_supports_generic_gymnasium_mujoco_task(
    mock_gym_make,
    mock_gym_image_env,
):
    from simulusion.envs.mujoco_env import make_mujoco_env

    base_env = Mock()
    wrapped_env = Mock()
    mock_gym_make.return_value = base_env
    mock_gym_image_env.return_value = wrapped_env

    out_env = make_mujoco_env(
        "Humanoid-v4",
        seed=7,
        size=(32, 32),
        render_mode="rgb_array",
        forward_reward_weight=1.5,
    )

    assert out_env is wrapped_env
    mock_gym_make.assert_called_once_with(
        "Humanoid-v4",
        render_mode="rgb_array",
        forward_reward_weight=1.5,
    )
    mock_gym_image_env.assert_called_once_with(
        base_env,
        seed=7,
        size=(32, 32),
        render_mode="rgb_array",
    )


def test_make_mujoco_env_from_config_builds_xml_string_env(monkeypatch):
    fake_mujoco = type("FakeMujoco", (), {})()
    fake_mujoco.MjModel = _FakeMjModel
    fake_mujoco.MjData = _FakeMjData
    fake_mujoco.Renderer = _FakeRenderer
    fake_mujoco.mj_resetData = lambda model, data: None
    fake_mujoco.mj_forward = lambda model, data: None
    fake_mujoco.mj_step = lambda model, data, nstep=1: None
    monkeypatch.setitem(__import__("sys").modules, "mujoco", fake_mujoco)

    from simulusion.envs.mujoco_env import make_mujoco_env_from_config

    cfg = DreamerConfig()
    cfg.mujoco_xml_string = "<mujoco/>"
    cfg.mujoco_camera = "track"
    cfg.mujoco_frame_skip = 2
    env = make_mujoco_env_from_config(cfg, size=(16, 24))

    assert env.observation_space["image"].shape == (3, 16, 24)
    assert env._camera == "track"
    assert env._frame_skip == 2


class _FakeJaxRandom:
    @staticmethod
    def PRNGKey(seed):
        return np.array([seed], dtype=np.int64)

    @staticmethod
    def split(key):
        base = int(np.asarray(key).reshape(-1)[0])
        return np.array([base + 1], dtype=np.int64), np.array(
            [base + 2], dtype=np.int64
        )


class _FakeJax:
    random = _FakeJaxRandom()

    @staticmethod
    def jit(fn):
        return fn

    @staticmethod
    def device_get(value):
        return value


class _FakeBraxState:
    def __init__(self, obs, reward=0.0, done=0.0, metrics=None, info=None):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.reward = np.asarray(reward, dtype=np.float32)
        self.done = np.asarray(done, dtype=np.float32)
        self.metrics = metrics or {}
        self.info = info or {}


class _FakeBraxEnv:
    action_size = 2
    episode_length = 7

    def __init__(self):
        self.last_action = None

    def reset(self, rng):
        return _FakeBraxState(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    def step(self, state, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return _FakeBraxState(
            np.array([1.0, 0.0, -1.0], dtype=np.float32),
            reward=1.25,
            done=0.0,
            metrics={"metric": np.asarray(3.0, dtype=np.float32)},
        )


def test_brax_image_env_adapts_functional_brax_api(monkeypatch):
    from simulusion.envs.brax_env import BraxImageEnv

    monkeypatch.setattr(
        "simulusion.envs.brax_env._require_module",
        lambda module_name, install_hint, **kwargs: {
            "jax": _FakeJax,
            "jax.numpy": np,
            "brax.envs": Mock(get_environment=lambda *args, **kwargs: _FakeBraxEnv()),
        }[module_name],
    )

    wrapped = BraxImageEnv(
        "ant",
        seed=0,
        size=(32, 32),
        backend="generalized",
        jit=True,
        include_state=True,
    )
    obs = wrapped.reset()
    render0 = wrapped.render()
    assert obs["image"].shape == (3, 32, 32)
    assert obs["state"].shape == (3,)
    assert wrapped.action_space.shape == (2,)
    assert wrapped.max_episode_steps == 7
    assert render0.shape == (32, 32, 3)
    assert np.array_equal(render0, wrapped.render())

    next_obs, reward, done, info = wrapped.step(np.array([2.0, -2.0], dtype=np.float32))

    assert next_obs["image"].shape == (3, 32, 32)
    assert next_obs["state"].shape == (3,)
    assert reward == 1.25
    assert done is False
    assert np.array_equal(info["action"], np.array([1.0, -1.0], dtype=np.float32))
    assert np.array_equal(
        info["executed_action"], np.array([1.0, -1.0], dtype=np.float32)
    )
    assert info["vector_observation"].shape == (3,)
    assert "discount" in info


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.mujoco_env.make_mujoco_env_from_config")
def test_make_env_native_mujoco_backend(
    mock_make_mujoco_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "mujoco"
    cfg.env = "model.xml"
    cfg.image_size = (64, 64)
    cfg.mujoco_camera = "fixed"
    cfg.mujoco_frame_skip = 4
    cfg.mujoco_reset_noise_scale = 0.01

    env = Mock()
    mock_make_mujoco_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_make_mujoco_env.assert_called_once_with(cfg, cfg.image_size)


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.brax_env.BraxImageEnv")
def test_make_env_brax_backend(
    mock_brax_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "brax"
    cfg.env = "ant"
    cfg.image_size = (64, 64)
    cfg.brax_backend = "mjx"
    cfg.brax_jit = False
    cfg.brax_auto_reset = False
    cfg.time_limit = 100

    env = Mock()
    mock_brax_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_brax_env.assert_called_once_with(
        cfg.env,
        seed=cfg.seed,
        size=cfg.image_size,
        backend=cfg.brax_backend,
        episode_length=cfg.time_limit,
        auto_reset=cfg.brax_auto_reset,
        jit=cfg.brax_jit,
        suppress_warp_warnings=cfg.brax_suppress_warp_warnings,
    )


def test_require_module_filters_warp_messages(monkeypatch, capsys):
    import importlib

    from simulusion.envs import brax_env as be

    # Make find_spec always report modules exist.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)

    def fake_import(name):
        # Simulate noisy import-time prints from MuJoCo/MJX shim.
        print("Some other message")
        print("Failed to import warp: No module named 'warp'")
        print("Failed to import mujoco_warp: No module named 'mujoco_warp'")

        class M:
            pass

        return M

    monkeypatch.setattr(importlib, "import_module", fake_import)

    # When suppression is enabled, only the non-warp line should be replayed.
    be._require_module("brax.envs", "hint", suppress_warp_warnings=True)
    captured = capsys.readouterr()
    assert "Some other message" in captured.out
    assert "Failed to import warp:" not in captured.out
    assert "Failed to import mujoco_warp:" not in captured.out


@patch("simulusion.envs.robotics_env.GymImageEnv")
@patch("simulusion.envs.robotics_env.gym.make")
def test_make_robotics_env_registers_gymnasium_robotics(
    mock_gym_make,
    mock_gym_image_env,
    monkeypatch,
):
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    import simulusion.envs.robotics_env as robotics_env
    from simulusion.envs.robotics_env import make_robotics_env

    fake_robotics = ModuleType("gymnasium_robotics")
    fake_robotics.__spec__ = ModuleSpec("gymnasium_robotics", loader=None)
    monkeypatch.setitem(sys.modules, "gymnasium_robotics", fake_robotics)
    monkeypatch.setattr(
        robotics_env.gym, "register_envs", lambda module: None, raising=False
    )
    base_env = Mock()
    wrapped_env = Mock()
    mock_gym_make.return_value = base_env
    mock_gym_image_env.return_value = wrapped_env

    out_env = make_robotics_env(
        "HalfCheetah-v2",
        seed=3,
        size=(32, 32),
        render_mode="rgb_array",
        reset_noise_scale=0.2,
    )

    assert out_env is wrapped_env
    mock_gym_make.assert_called_once_with(
        "HalfCheetah-v2",
        render_mode="rgb_array",
        reset_noise_scale=0.2,
    )
    mock_gym_image_env.assert_called_once_with(
        base_env,
        seed=3,
        size=(32, 32),
        render_mode="rgb_array",
    )


@patch("simulusion.envs.mujoco_env.GymImageEnv")
@patch("simulusion.envs.robotics_env.gym.make")
def test_make_mujoco_env_falls_back_to_gymnasium_robotics_for_legacy_ids(
    mock_gym_make,
    mock_gym_image_env,
    monkeypatch,
):
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    import simulusion.envs.robotics_env as robotics_env
    from simulusion.envs.mujoco_env import make_mujoco_env

    fake_robotics = ModuleType("gymnasium_robotics")
    fake_robotics.__spec__ = ModuleSpec("gymnasium_robotics", loader=None)
    monkeypatch.setitem(sys.modules, "gymnasium_robotics", fake_robotics)
    monkeypatch.setattr(
        robotics_env.gym, "register_envs", lambda module: None, raising=False
    )
    base_env = Mock()
    wrapped_env = Mock()
    mock_gym_make.side_effect = [
        ImportError(
            "The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project."
        ),
        base_env,
    ]
    mock_gym_image_env.return_value = wrapped_env

    out_env = make_mujoco_env("HalfCheetah-v2", seed=5, size=(16, 16))

    assert out_env is wrapped_env
    assert mock_gym_make.call_count == 2
    mock_gym_make.assert_called_with("HalfCheetah-v2", render_mode="rgb_array")
    mock_gym_image_env.assert_called_once_with(
        base_env,
        seed=5,
        size=(16, 16),
        render_mode="rgb_array",
    )


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.robotics_env.make_robotics_env")
def test_make_env_robotics_backend(
    mock_robotics_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "robotics"
    cfg.env = "HalfCheetah-v2"
    cfg.image_size = (64, 64)
    cfg.gym_render_mode = "rgb_array"

    env = Mock()
    mock_robotics_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_robotics_env.assert_called_once_with(
        "HalfCheetah-v2",
        seed=cfg.seed,
        size=cfg.image_size,
        render_mode="rgb_array",
    )


def test_catalog_queries_gymnasium_registry_at_runtime(monkeypatch):
    import sys
    from types import SimpleNamespace

    import simulusion.catalog as catalog

    fake_gym = SimpleNamespace(
        envs=SimpleNamespace(
            registry={
                "CartPole-v1": SimpleNamespace(
                    id="CartPole-v1",
                    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
                    namespace=None,
                ),
                "ExampleControl-v9": SimpleNamespace(
                    id="ExampleControl-v9",
                    entry_point="example_package:ExampleEnv",
                    namespace=None,
                ),
                "ALE/Pong-v5": SimpleNamespace(
                    id="ALE/Pong-v5",
                    entry_point="ale_py.env:PongEnv",
                    namespace="ALE",
                ),
                "FetchReachDense-v4": SimpleNamespace(
                    id="FetchReachDense-v4",
                    entry_point="gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv",
                    namespace=None,
                ),
            }
        )
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

    assert catalog._list_available_gymnasium_envs() == [
        "CartPole-v1",
        "ExampleControl-v9",
    ]


def test_environment_catalog_uses_runtime_gymnasium_envs(monkeypatch):
    import simulusion.catalog as catalog

    monkeypatch.setattr(
        catalog, "_list_available_gymnasium_envs", lambda: ["DynamicEnv-v9"]
    )
    monkeypatch.setattr(
        catalog, "_list_available_robotics_envs", lambda: ["FetchReach-v4"]
    )
    monkeypatch.setattr(catalog, "_list_available_atari_envs", lambda: ["ALE/Pong-v5"])

    envs_by_model = catalog._build_env_catalog()

    assert "DynamicEnv-v9" in envs_by_model["dreamer"]
    assert "DynamicEnv-v9" in envs_by_model["planet"]
    assert "DynamicEnv-v9" not in envs_by_model["iris"]
    assert "ALE/Pong-v5" in envs_by_model["iris"]
    assert "FetchReach-v4" in envs_by_model["diamond"]


class _FakeDeepMindLabInstance:
    def __init__(self, level, observations, config=None, renderer="hardware", **kwargs):
        self.level = level
        self.observations_requested = observations
        self.config = config
        self.renderer = renderer
        self.kwargs = kwargs
        self.running = False
        self.last_action = None
        self.last_num_steps = None
        self.seed = None

    def reset(self, seed=None):
        self.running = True
        self.seed = seed

    def observations(self):
        return {
            "RGB_INTERLEAVED": np.full((16, 16, 3), 64, dtype=np.uint8),
            "VEL.TRANS": np.array([1.0, 0.0, -1.0], dtype=np.float64),
        }

    def step(self, action, num_steps=1):
        self.last_action = np.asarray(action)
        self.last_num_steps = num_steps
        self.running = False
        return 2.5

    def is_running(self):
        return self.running

    def close(self):
        self.closed = True


class _FakeDeepMindLabModule:
    instances = []

    @staticmethod
    def Lab(*args, **kwargs):
        instance = _FakeDeepMindLabInstance(*args, **kwargs)
        _FakeDeepMindLabModule.instances.append(instance)
        return instance


def test_dmlab_env_adapts_deepmind_lab_api(monkeypatch):
    import sys

    from simulusion.envs.dmlab import DMLabEnv

    _FakeDeepMindLabModule.instances = []
    monkeypatch.setitem(sys.modules, "deepmind_lab", _FakeDeepMindLabModule)

    env = DMLabEnv(
        "rooms_collect_good_objects_train",
        seed=7,
        size=(32, 32),
        action_repeat=3,
        observations=["VEL.TRANS"],
        config={"fps": 30, "episode_length_seconds": 10},
    )
    obs = env.reset()

    assert obs["image"].shape == (3, 32, 32)
    assert obs["VEL.TRANS"].shape == (3,)
    assert env.max_episode_steps == 100
    assert env.action_space.shape == (9,)

    action = -np.ones(env.action_space.shape, dtype=np.float32)
    action[3] = 1.0
    obs, reward, done, info = env.step(action)

    lab = _FakeDeepMindLabModule.instances[-1]
    assert lab.level == "rooms_collect_good_objects_train"
    assert lab.observations_requested == ["RGB_INTERLEAVED", "VEL.TRANS"]
    assert lab.config["width"] == "32"
    assert lab.config["height"] == "32"
    assert lab.seed == 7
    assert lab.last_num_steps == 3
    assert np.array_equal(lab.last_action, np.array([0, 0, 0, 1, 0, 0, 0]))
    assert obs["image"].shape == (3, 32, 32)
    assert reward == 2.5
    assert done is True
    assert info["action"].shape == env.action_space.shape
    assert np.array_equal(info["executed_action"], np.array([0, 0, 0, 1, 0, 0, 0]))
    assert info["discount"] == np.array(0.0, dtype=np.float32)


def test_procgen_env_name_normalization_and_list():
    from simulusion.envs.procgen_env import (
        list_procgen_envs,
        normalize_procgen_env_name,
    )

    assert "coinrun" in list_procgen_envs()
    assert normalize_procgen_env_name("coinrun") == "coinrun"
    assert normalize_procgen_env_name("procgen-coinrun-v0") == "coinrun"
    assert normalize_procgen_env_name("procgen:procgen-coinrun-v0") == "coinrun"


def test_procgen_image_env_wraps_single_vector_env(monkeypatch):
    import sys
    import types

    from simulusion.envs import procgen_env
    from simulusion.envs.procgen_env import ProcgenImageEnv

    class _FakeProcgenEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.action_space = gym.spaces.Discrete(4)
            self.last_action = None
            self.closed = False

        def reset(self):
            return {"rgb": np.zeros((1, 64, 64, 3), dtype=np.uint8)}

        def step(self, action):
            self.last_action = action
            obs = {"rgb": np.ones((1, 64, 64, 3), dtype=np.uint8) * 127}
            return obs, np.array([1.5], dtype=np.float32), np.array([False]), [{}]

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        procgen_env.importlib.util,
        "find_spec",
        lambda name: object() if name == "procgen" else None,
    )
    monkeypatch.setitem(
        sys.modules,
        "procgen",
        types.SimpleNamespace(ProcgenEnv=_FakeProcgenEnv),
    )

    env = ProcgenImageEnv(
        "procgen:procgen-coinrun-v0",
        seed=7,
        size=(32, 32),
        distribution_mode="hard",
        num_levels=100,
        max_episode_steps=123,
    )

    assert env.env_name == "coinrun"
    assert env.max_episode_steps == 123
    assert env.action_space.shape == (4,)
    assert env._env.kwargs["start_level"] == 7
    assert env._env.kwargs["distribution_mode"] == "hard"
    assert env._env.kwargs["num_levels"] == 100

    obs = env.reset()
    assert obs["image"].shape == (3, 32, 32)
    assert obs["image"].dtype == np.uint8

    next_obs, reward, done, info = env.step(np.array([-1.0, -0.5, 0.9, 0.1]))
    assert env._env.last_action.tolist() == [2]
    assert next_obs["image"].shape == (3, 32, 32)
    assert reward == 1.5
    assert done is False
    expected_action = np.array([-1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    assert np.array_equal(info["action"], expected_action)
    assert info["executed_action"] == 2
    assert np.asarray(info["discount"]).item() == 1.0

    frame = env.render()
    assert frame.shape == (32, 32, 3)
    env.close()
    assert env._env.closed is True


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.dmlab.DMLabEnv")
def test_make_env_dmlab_backend(
    mock_dmlab,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "dmlab"
    cfg.env = "rooms_collect_good_objects_train"
    cfg.image_size = (64, 64)
    cfg.dmlab_action_repeat = 5
    cfg.dmlab_observations = ["VEL.TRANS"]
    cfg.dmlab_config = {"fps": 30}
    cfg.dmlab_renderer = "software"

    env = Mock()
    mock_dmlab.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_dmlab.assert_called_once_with(
        cfg.env,
        seed=cfg.seed,
        size=cfg.image_size,
        action_repeat=cfg.dmlab_action_repeat,
        action_set=cfg.dmlab_action_set,
        observations=cfg.dmlab_observations,
        config=cfg.dmlab_config,
        renderer=cfg.dmlab_renderer,
    )


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.procgen_env.ProcgenImageEnv")
def test_make_env_procgen_backend(
    mock_procgen_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "procgen"
    cfg.env = "coinrun"
    cfg.image_size = (64, 64)
    cfg.procgen_distribution_mode = "hard"
    cfg.procgen_num_levels = 200
    cfg.procgen_start_level = 5

    env = Mock()
    mock_procgen_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_procgen_env.assert_called_once_with(
        cfg.env,
        seed=cfg.seed,
        size=cfg.image_size,
        distribution_mode=cfg.procgen_distribution_mode,
        num_levels=cfg.procgen_num_levels,
        start_level=cfg.procgen_start_level,
        max_episode_steps=cfg.time_limit,
    )


@patch("simulusion.envs.wrappers.TimeLimit")
@patch("simulusion.envs.wrappers.NormalizeActions")
@patch("simulusion.envs.wrappers.ActionRepeat")
@patch("simulusion.envs.bsuite_env.BSuiteImageEnv")
def test_make_env_bsuite_backend(
    mock_bsuite_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "bsuite"
    cfg.env = "catch/0"
    cfg.image_size = (64, 64)

    env = Mock()
    mock_bsuite_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_bsuite_env.assert_called_once_with(cfg.env, seed=cfg.seed, size=cfg.image_size)


class _FakeActionSpec:
    num_values = 3


class _FakeTimeStep:
    def __init__(self, observation, reward=0.0, discount=1.0, last=False):
        self.observation = observation
        self.reward = reward
        self.discount = discount
        self._last = last

    def last(self):
        return self._last


class _FakeBSuiteEnv:
    bsuite_num_episodes = 2

    def __init__(self):
        self.last_action = None

    def action_spec(self):
        return _FakeActionSpec()

    def reset(self):
        return _FakeTimeStep({"state": np.array([0.0, 1.0], dtype=np.float32)})

    def step(self, action):
        self.last_action = action
        return _FakeTimeStep(
            np.array([1.0, 0.0], dtype=np.float32), reward=1.5, last=True
        )

    def close(self):
        self.closed = True


def test_bsuite_image_env_wraps_dm_env_discrete_task():
    from simulusion.envs.bsuite_env import BSuiteImageEnv

    base_env = _FakeBSuiteEnv()
    env = BSuiteImageEnv(
        "catch/0", seed=7, size=(16, 16), env=base_env, include_state=True
    )

    obs = env.reset()
    render0 = env.render()
    assert obs["image"].shape == (3, 16, 16)
    assert obs["state"].shape == (2,)
    assert env.action_space.shape == (3,)
    assert render0.shape == (16, 16, 3)
    assert np.array_equal(render0, env.render())

    next_obs, reward, done, info = env.step(
        np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    )
    assert next_obs["image"].shape == (3, 16, 16)
    assert next_obs["state"].shape == (2,)
    assert reward == 1.5
    assert done is True
    assert base_env.last_action == 1
    assert info["bsuite_id"] == "catch/0"
    assert info["vector_observation"].shape == (2,)
    assert np.array_equal(info["action"], np.array([-1.0, 1.0, -1.0], dtype=np.float32))


def test_gym_image_env_reset_seed_replays_initial_observation_and_action_samples():
    from simulusion.envs.gym_env import GymImageEnv

    class _SeedReplayEnv:
        def __init__(self):
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(8, 8, 3), dtype=np.uint8
            )
            self._rng = np.random.default_rng(0)

        def reset(self, seed=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            return self._rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8), {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

        def render(self):
            return self._rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)

    env = GymImageEnv(_SeedReplayEnv(), seed=5, size=(8, 8))
    first = env.reset(seed=123)
    first_action = env.action_space.sample()
    env.step(first_action)
    second = env.reset(seed=123)
    second_action = env.action_space.sample()

    assert np.array_equal(first["image"], second["image"])
    assert np.array_equal(first_action, second_action)


def test_dmc_reset_seed_rebuilds_backend_and_replays_initial_state(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from simulusion.envs.dmc import DeepMindControlEnv

    class _SeededPhysics:
        def __init__(self, seed):
            self._seed = seed

        def render(self, height, width, camera_id=0):
            del camera_id
            return np.full((height, width, 3), self._seed % 255, dtype=np.uint8)

    class _SeededTimeStep:
        def __init__(self, seed, last=False):
            self.observation = {
                "position": np.array([seed, seed + 1], dtype=np.float32)
            }
            self.reward = float(seed)
            self.discount = 0.0 if last else 1.0
            self._last = last

        def last(self):
            return self._last

    class _SeededDMCEnv:
        def __init__(self, seed):
            self.seed = seed
            self.physics = _SeededPhysics(seed)

        def observation_spec(self):
            return {"position": SimpleNamespace(shape=(2,))}

        def action_spec(self):
            return SimpleNamespace(
                minimum=np.array([-1.0, -1.0], dtype=np.float32),
                maximum=np.array([1.0, 1.0], dtype=np.float32),
            )

        def reset(self):
            return _SeededTimeStep(self.seed)

        def step(self, action):
            return _SeededTimeStep(self.seed + int(np.asarray(action).sum()), last=True)

    dm_control = ModuleType("dm_control")
    dm_control.suite = SimpleNamespace(
        load=lambda domain, task, task_kwargs: _SeededDMCEnv(task_kwargs["random"])
    )
    monkeypatch.setitem(sys.modules, "dm_control", dm_control)

    env = DeepMindControlEnv("cartpole-swingup", seed=1, size=(8, 8))
    first = env.reset(seed=17)
    second = env.reset(seed=17)
    third = env.reset(seed=18)

    assert np.array_equal(first["position"], second["position"])
    assert np.array_equal(first["image"], second["image"])
    assert not np.array_equal(first["position"], third["position"])


def test_brax_reset_seed_replays_initial_state_and_action_space_sampling(monkeypatch):
    from simulusion.envs.brax_env import BraxImageEnv

    class _SeededBraxEnv:
        action_size = 2
        episode_length = 7
        observation_size = 3

        def reset(self, rng):
            seed = float(np.asarray(rng).reshape(-1)[0])
            return _FakeBraxState(
                np.array([seed, seed + 1.0, seed + 2.0], dtype=np.float32)
            )

        def step(self, state, action):
            return _FakeBraxState(state.obs, reward=0.0, done=0.0)

    monkeypatch.setattr(
        "simulusion.envs.brax_env._require_module",
        lambda module_name, install_hint, **kwargs: {
            "jax": _FakeJax,
            "jax.numpy": np,
            "brax.envs": Mock(get_environment=lambda *args, **kwargs: _SeededBraxEnv()),
        }[module_name],
    )

    env = BraxImageEnv("ant", seed=0, size=(8, 8), jit=True, include_state=True)
    first = env.reset(seed=9)
    first_action = env.action_space.sample()
    second = env.reset(seed=9)
    second_action = env.action_space.sample()

    assert np.array_equal(first["state"], second["state"])
    assert np.array_equal(first["image"], second["image"])
    assert np.array_equal(first_action, second_action)


def test_dmlab_reset_seed_restarts_episode_sequence_and_action_sampling(monkeypatch):
    import sys

    from simulusion.envs.dmlab import DMLabEnv

    _FakeDeepMindLabModule.instances = []
    monkeypatch.setitem(sys.modules, "deepmind_lab", _FakeDeepMindLabModule)

    env = DMLabEnv("rooms_collect_good_objects_train", seed=5, size=(16, 16))
    env.reset(seed=21)
    first_seed = _FakeDeepMindLabModule.instances[-1].seed
    first_action = env.action_space.sample()
    env.reset()
    second_seed = _FakeDeepMindLabModule.instances[-1].seed
    env.reset(seed=21)
    third_seed = _FakeDeepMindLabModule.instances[-1].seed
    second_action = env.action_space.sample()

    assert first_seed == 21
    assert second_seed == 22
    assert third_seed == 21
    assert np.array_equal(first_action, second_action)


def test_procgen_reset_seed_rebuilds_backend_and_reseeds_action_sampling(monkeypatch):
    import sys
    import types

    from simulusion.envs import procgen_env
    from simulusion.envs.procgen_env import ProcgenImageEnv

    instances = []

    class _SeededProcgenEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.action_space = gym.spaces.Discrete(4)
            self.closed = False
            instances.append(self)

        def reset(self):
            value = self.kwargs["start_level"] % 255
            return {"rgb": np.full((1, 8, 8, 3), value, dtype=np.uint8)}

        def step(self, action):
            return (
                self.reset(),
                np.array([0.0], dtype=np.float32),
                np.array([False]),
                [{}],
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        procgen_env.importlib.util,
        "find_spec",
        lambda name: object() if name == "procgen" else None,
    )
    monkeypatch.setitem(
        sys.modules,
        "procgen",
        types.SimpleNamespace(ProcgenEnv=_SeededProcgenEnv),
    )

    env = ProcgenImageEnv("coinrun", seed=3, size=(8, 8))
    first = env.reset(seed=14)
    first_action = env.action_space.sample()
    old_env = instances[-1]
    second = env.reset(seed=14)
    second_action = env.action_space.sample()

    assert old_env.closed is True
    assert env._env.kwargs["start_level"] == 14
    assert np.array_equal(first["image"], second["image"])
    assert np.array_equal(first_action, second_action)


def test_unity_reset_seed_rebuilds_backend_and_replays_initial_observation(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    class _SeededChannel:
        def set_configuration_parameters(self, **kwargs):
            self.params = kwargs

    class _SeededActionSpec:
        continuous_size = 2

        @staticmethod
        def is_continuous():
            return True

    class _SeededSteps:
        def __init__(self, agent_ids=(), obs=None, reward=None, interrupted=None):
            self.agent_id = np.asarray(agent_ids, dtype=np.int64)
            self.obs = list(obs or [])
            self.reward = np.asarray(
                reward if reward is not None else [0.0], dtype=np.float32
            )
            self.interrupted = np.asarray(
                interrupted if interrupted is not None else [False], dtype=bool
            )

    class _SeededUnityEnvironment:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.behavior_specs = {
                "Behavior": SimpleNamespace(
                    action_spec=_SeededActionSpec(),
                    observation_specs=[SimpleNamespace(shape=(8, 8, 3))],
                )
            }
            self.closed = False
            _SeededUnityEnvironment.instances.append(self)

        def reset(self):
            return None

        def get_steps(self, behavior_name):
            assert behavior_name == "Behavior"
            value = self.kwargs["seed"] % 255
            decision = _SeededSteps(
                agent_ids=[1],
                obs=[np.full((1, 8, 8, 3), value, dtype=np.uint8)],
                reward=[0.0],
            )
            return decision, _SeededSteps(agent_ids=[], obs=[], reward=[])

        def set_actions(self, behavior_name, action_tuple):
            del behavior_name, action_tuple

        def step(self):
            return None

        def close(self):
            self.closed = True

    root = ModuleType("mlagents_envs")
    base_env_mod = ModuleType("mlagents_envs.base_env")
    env_mod = ModuleType("mlagents_envs.environment")
    channel_mod = ModuleType("mlagents_envs.side_channel.engine_configuration_channel")

    class _SeededActionTuple:
        def __init__(self, continuous=None):
            self.continuous = continuous

    base_env_mod.ActionTuple = _SeededActionTuple
    env_mod.UnityEnvironment = _SeededUnityEnvironment
    channel_mod.EngineConfigurationChannel = _SeededChannel

    monkeypatch.setitem(sys.modules, "mlagents_envs", root)
    monkeypatch.setitem(sys.modules, "mlagents_envs.base_env", base_env_mod)
    monkeypatch.setitem(sys.modules, "mlagents_envs.environment", env_mod)
    monkeypatch.setitem(
        sys.modules,
        "mlagents_envs.side_channel.engine_configuration_channel",
        channel_mod,
    )

    from simulusion.envs.unity_env import UnityMLAgentsEnv

    env = UnityMLAgentsEnv(file_name="fake.exe", seed=4, size=(8, 8), no_graphics=True)
    first = env.reset(seed=12)
    old_env = _SeededUnityEnvironment.instances[-1]
    second = env.reset(seed=12)

    assert old_env.closed is True
    assert env._env.kwargs["seed"] == 12
    assert np.array_equal(first["image"], second["image"])


class _StatefulEnv:
    """Tracks internal counter across episodes to test reset isolation."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 1, (4,), dtype=np.float32)
        self._counter = 0

    def reset(self, seed=None):
        self._counter = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._counter += 1
        return np.full(4, self._counter, dtype=np.float32), 1.0, False, {}


def test_reset_does_not_carry_hidden_state():
    """Verify reset() clears environment-internal state between episodes."""
    raw = _StatefulEnv()

    _, _ = raw.reset(seed=42)
    for _ in range(5):
        _, _, done, _ = raw.step(np.array([0.5, 0.5], dtype=np.float32))
        if done:
            break

    obs, info = raw.reset()
    assert np.array_equal(obs, np.zeros(4, dtype=np.float32)), (
        "reset() should restore initial observation, not carry counter from previous episode"
    )
    assert raw._counter == 0, "Internal state counter should be reset to 0"


def test_render_graceful_error_on_unsupported_env():
    """Calling render on an env that does not support it should raise a useful error."""
    raw_env = _StatefulEnv()
    env = GymImageEnv(raw_env, seed=0, size=(8, 8))

    with pytest.raises((Exception, RuntimeError, AttributeError)) as excinfo:
        env.render()
    assert (
        "render" in str(excinfo.value).lower()
        or "rgb_array" in str(excinfo.value).lower()
    ), f"Expected render error message, got: {excinfo.value}"
