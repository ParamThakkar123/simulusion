"""DeepMind Control observation types (pixels / proprioceptive state)."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from simulusion.configs.dreamer_config import DreamerConfig  # noqa: E402
from simulusion.envs.dmc import DeepMindControlEnv, make_dmc_env  # noqa: E402
from simulusion.models.dreamer import image_obs_shape, make_env  # noqa: E402


class _Spec:
    def __init__(self, shape):
        self.shape = shape


class _TimeStep:
    def __init__(
        self,
        observation: dict,
        reward: float | None = 1.0,
        discount: float | None = 1.0,
        last: bool = False,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.discount = discount
        self._last = last

    def last(self):
        return self._last


class _Physics:
    def __init__(self):
        self.renders = 0

    def render(self, height, width, camera_id=0):
        self.renders += 1
        return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeSuiteEnv:
    """Two state entries, ending after ``length`` steps with ``final_discount``."""

    def __init__(self, length=3, final_discount=1.0):
        self.physics = _Physics()
        self.length = length
        self.final_discount = final_discount
        self.steps = 0

    def observation_spec(self):
        return {"position": _Spec((2,)), "velocity": _Spec((3,))}

    def action_spec(self):
        return SimpleNamespace(
            minimum=np.full(1, -1.0, np.float32), maximum=np.full(1, 1.0, np.float32)
        )

    def _obs(self):
        value = float(self.steps)
        return {
            "position": np.array([value, value + 0.5]),
            "velocity": np.array([10.0, 20.0, 30.0]) + value,
        }

    def reset(self):
        self.steps = 0
        return _TimeStep(self._obs(), reward=None, discount=None)

    def step(self, action):
        self.steps += 1
        last = self.steps >= self.length
        discount = self.final_discount if last else 1.0
        return _TimeStep(self._obs(), discount=discount, last=last)


@pytest.fixture
def fake_suite(monkeypatch):
    envs = []
    settings: dict = {}

    def load(domain, task, task_kwargs):
        env = _FakeSuiteEnv(**settings)
        envs.append(env)
        return env

    dm_control = ModuleType("dm_control")
    dm_control.suite = SimpleNamespace(load=load)
    monkeypatch.setitem(sys.modules, "dm_control", dm_control)
    return SimpleNamespace(envs=envs, configure=settings.update)


class TestStateObservations:
    def test_state_mode_returns_the_flat_vector_without_rendering(self, fake_suite):
        env = DeepMindControlEnv("walker-walk", seed=0, obs_type="state")
        obs = env.reset()
        assert set(obs) == {"position", "velocity", "state"}
        assert "image" not in env.observation_space.spaces
        assert env.observation_space["state"].shape == (5,)
        assert env.observation_space.contains(obs)
        next_obs, _, _, info = env.step(np.zeros(1, np.float32))
        assert env.observation_space.contains(next_obs)
        assert fake_suite.envs[-1].physics.renders == 0
        assert np.array_equal(info["vector_observation"], next_obs["state"])

    def test_state_vector_follows_the_observation_spec_order(self, fake_suite):
        env = DeepMindControlEnv("walker-walk", seed=0, obs_type="state")
        obs = env.reset()
        expected = np.array([0.0, 0.5, 10.0, 20.0, 30.0], dtype=np.float32)
        assert obs["state"].dtype == np.float32
        assert np.array_equal(obs["state"], expected)

    def test_pixel_mode_also_carries_the_state_vector(self, fake_suite):
        env = DeepMindControlEnv("walker-walk", seed=0, size=(8, 8))
        obs = env.reset()
        assert set(obs) == {"position", "velocity", "state", "image"}
        assert obs["image"].shape == (3, 8, 8)
        assert env.observation_space.contains(obs)

    def test_render_still_works_in_state_mode(self, fake_suite):
        env = DeepMindControlEnv("walker-walk", seed=0, size=(8, 8), obs_type="state")
        env.reset()
        assert env.render().shape == (8, 8, 3)

    def test_unknown_obs_type_is_rejected(self, fake_suite):
        with pytest.raises(ValueError, match="obs_type"):
            DeepMindControlEnv("walker-walk", seed=0, obs_type="depth")

    def test_make_dmc_env_forwards_the_obs_type(self, fake_suite):
        assert make_dmc_env("walker-walk", obs_type="state").obs_type == "state"


class TestEpisodeEnds:
    def test_time_limit_end_is_a_truncation(self, fake_suite):
        # dm_control ends every episode at its time limit with discount 1.
        fake_suite.configure(length=2, final_discount=1.0)
        env = DeepMindControlEnv("walker-walk", seed=0, obs_type="state")
        env.reset()
        env.step(np.zeros(1, np.float32))
        _, _, done, info = env.step(np.zeros(1, np.float32))
        assert done
        assert info["terminated"] is False
        assert info["truncated"] is True

    def test_zero_discount_is_a_termination(self, fake_suite):
        fake_suite.configure(length=1, final_discount=0.0)
        env = DeepMindControlEnv("walker-walk", seed=0, obs_type="state")
        env.reset()
        _, _, done, info = env.step(np.zeros(1, np.float32))
        assert done
        assert info["terminated"] is True
        assert info["truncated"] is False

    def test_mid_episode_steps_are_neither(self, fake_suite):
        fake_suite.configure(length=5)
        env = DeepMindControlEnv("walker-walk", seed=0, obs_type="state")
        env.reset()
        _, _, done, info = env.step(np.zeros(1, np.float32))
        assert not done
        assert info["terminated"] is False and info["truncated"] is False


class TestDreamerWiring:
    @patch("simulusion.envs.wrappers.TimeLimit")
    @patch("simulusion.envs.wrappers.NormalizeActions")
    @patch("simulusion.envs.wrappers.ActionRepeat")
    @patch("simulusion.envs.dmc.DeepMindControlEnv")
    def test_make_env_passes_the_state_obs_type(
        self, mock_dmc, mock_repeat, mock_normalize, mock_time_limit
    ):
        for mock in (mock_repeat, mock_normalize, mock_time_limit):
            mock.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
        cfg = DreamerConfig(env_backend="dmc", env="walker-walk", dmc_obs_type="state")
        mock_dmc.return_value = Mock()
        make_env(cfg)
        mock_dmc.assert_called_once_with(
            "walker-walk", cfg.seed, size=cfg.image_size, obs_type="state"
        )

    def test_config_defaults_to_pixels(self):
        assert DreamerConfig().dmc_obs_type == "pixels"

    def test_image_models_reject_state_only_environments(self, fake_suite):
        env = DeepMindControlEnv("walker-walk", seed=0, obs_type="state")
        with pytest.raises(ValueError, match="dmc_obs_type='pixels'"):
            image_obs_shape(env)

    def test_image_models_read_the_image_shape(self, fake_suite):
        env = DeepMindControlEnv("walker-walk", seed=0, size=(8, 8))
        assert image_obs_shape(env) == (3, 8, 8)


class TestRealSuite:
    @pytest.mark.parametrize(
        "name, size", [("cheetah-run", 17), ("walker-walk", 24), ("cup-catch", 8)]
    )
    def test_state_mode_on_dm_control(self, name, size):
        pytest.importorskip("dm_control")
        env = DeepMindControlEnv(name, seed=0, obs_type="state")
        obs = env.reset()
        assert obs["state"].shape == (size,)
        assert env.observation_space.contains(obs)
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs["state"].shape == (size,)
        assert np.isfinite(obs["state"]).all()
        assert not done
