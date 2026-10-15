import importlib.util

import pytest
from types import ModuleType, SimpleNamespace

if (
    importlib.util.find_spec("gymnasium") is None
    and importlib.util.find_spec("gym") is None
):
    pytest.skip("gym/gymnasium is not installed", allow_module_level=True)

import numpy as np

from simulusion.envs.dmc import DeepMindControlEnv


class _FakePhysics:
    def render(self, height, width, camera_id=0):
        assert camera_id in (0, 2)
        return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeTimeStep:
    def __init__(self, observation, reward=1.0, discount=0.99, last=False):
        self.observation = observation
        self.reward = reward
        self.discount = discount
        self._last = last

    def last(self):
        return self._last


class _FakeDMCEnv:
    def __init__(self):
        self.physics = _FakePhysics()
        self.step_calls = 0

    def observation_spec(self):
        return {"position": SimpleNamespace(shape=(2,))}

    def action_spec(self):
        return SimpleNamespace(
            minimum=np.array([-1.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 1.0], dtype=np.float32),
        )

    def reset(self):
        return _FakeTimeStep({"position": np.array([0.0, 0.5], dtype=np.float32)})

    def step(self, action):
        self.step_calls += 1
        return _FakeTimeStep(
            {"position": np.asarray(action, dtype=np.float32)},
            reward=None,
            discount=1.0,
            last=self.step_calls >= 2,
        )


def _install_fake_dm_control(monkeypatch):
    fake_env = _FakeDMCEnv()
    suite = SimpleNamespace(load=lambda domain, task, task_kwargs: fake_env)
    dm_control = ModuleType("dm_control")
    dm_control.suite = suite
    monkeypatch.setitem(__import__("sys").modules, "dm_control", dm_control)
    return fake_env


def test_deepmind_control_env_smoke_with_fake_suite(monkeypatch):
    fake_env = _install_fake_dm_control(monkeypatch)

    env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(16, 24))

    assert env.observation_space["image"].shape == (3, 16, 24)
    assert env.action_space.shape == (2,)

    obs = env.reset()
    assert obs["image"].shape == (3, 16, 24)
    assert obs["position"].shape == (2,)

    next_obs, reward, done, info = env.step(np.array([0.25, -0.25], dtype=np.float32))
    assert next_obs["image"].shape == (3, 16, 24)
    assert reward == 0
    assert done is False
    assert info["discount"].dtype == np.float32
    assert fake_env.step_calls == 1


def test_deepmind_control_env_domain_alias_and_render_validation(monkeypatch):
    calls = []
    fake_env = _FakeDMCEnv()

    def load(domain, task, task_kwargs):
        calls.append((domain, task, task_kwargs))
        return fake_env

    dm_control = ModuleType("dm_control")
    dm_control.suite = SimpleNamespace(load=load)
    monkeypatch.setitem(__import__("sys").modules, "dm_control", dm_control)

    env = DeepMindControlEnv(name="cup-catch", seed=7, size=(8, 8))
    assert calls == [("ball_in_cup", "catch", {"random": 7})]

    try:
        env.render(mode="human")
    except ValueError as exc:
        assert "rgb_array" in str(exc)
    else:
        raise AssertionError("DeepMindControlEnv.render should reject non-RGB modes")
