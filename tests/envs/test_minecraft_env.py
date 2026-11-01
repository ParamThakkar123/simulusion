"""Tests for the Minecraft (MineRL / MineDojo) adapter.

Neither backend is installable in CI -- MineRL needs a JDK and launches a real
Minecraft client -- so these drive fakes that reproduce the parts of each API the
adapter actually depends on: the ``Dict`` action space with a continuous camera,
the observation dict, and the pre-Gymnasium 4-tuple step.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("gymnasium") is None:
    pytest.skip("gymnasium is not installed", allow_module_level=True)

import gymnasium as gym

from simulusion.envs.minecraft_env import (
    MINECRAFT_ACTION_SET,
    MinecraftDiscreteEnv,
    _extract_pov,
    list_minecraft_actions,
    list_minecraft_envs,
    make_minecraft_env,
)


class FakeMineRLEnv:
    """Mimics MineRL: dict obs with 'pov', Dict action space, 4-tuple step."""

    def __init__(self, pov_shape=(64, 64, 3), supports_use=True):
        self._pov_shape = pov_shape
        spaces = {
            "forward": gym.spaces.Discrete(2),
            "back": gym.spaces.Discrete(2),
            "left": gym.spaces.Discrete(2),
            "right": gym.spaces.Discrete(2),
            "jump": gym.spaces.Discrete(2),
            "attack": gym.spaces.Discrete(2),
            "camera": gym.spaces.Box(-180.0, 180.0, (2,), dtype=np.float32),
        }
        if supports_use:
            spaces["use"] = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Dict(spaces)
        self.observation_space = gym.spaces.Dict(
            {"pov": gym.spaces.Box(0, 255, pov_shape, dtype=np.uint8)}
        )
        self.received: list[dict] = []
        self.t = 0
        self.closed = False

    def _obs(self):
        return {
            "pov": np.full(self._pov_shape, self.t % 255, dtype=np.uint8),
            "inventory": {"log": self.t},
        }

    def reset(self):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.received.append(action)
        self.t += 1
        return self._obs(), 1.0, self.t >= 5, {"episode_step": self.t}

    def close(self):
        self.closed = True


class FakeMineDojoEnv(FakeMineRLEnv):
    """Mimics MineDojo: channels-first 'rgb' key and a Gymnasium 5-tuple step."""

    def __init__(self):
        super().__init__(pov_shape=(3, 96, 96))

    def _obs(self):
        return {"rgb": np.full((3, 96, 96), self.t % 255, dtype=np.uint8)}

    def reset(self):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.received.append(action)
        self.t += 1
        return self._obs(), 1.0, self.t >= 5, False, {}


class TestActionTranslation:
    def test_discrete_space_matches_action_set(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        assert env.action_space.n == len(MINECRAFT_ACTION_SET)
        assert env.action_names == list_minecraft_actions()

    def test_noop_presses_nothing(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        native = env.translate_action(env.action_names.index("noop"))
        assert all(v == 0 for k, v in native.items() if k != "camera"), native
        assert np.allclose(native["camera"], 0.0)

    def test_movement_sets_only_its_own_key(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        native = env.translate_action(env.action_names.index("forward"))
        assert native["forward"] == 1
        assert native["back"] == 0 and native["jump"] == 0

    def test_compound_action_sets_both_keys(self):
        """forward_jump must press both, not just the last one written."""
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        native = env.translate_action(env.action_names.index("forward_jump"))
        assert native["forward"] == 1 and native["jump"] == 1

    @pytest.mark.parametrize(
        "name,axis,sign",
        [
            ("camera_left", 1, -1),
            ("camera_right", 1, 1),
            ("camera_up", 0, -1),
            ("camera_down", 0, 1),
        ],
    )
    def test_camera_actions_move_the_right_axis(self, name, axis, sign):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        camera = env.translate_action(env.action_names.index(name))["camera"]
        assert np.sign(camera[axis]) == sign
        assert camera[1 - axis] == 0.0

    def test_unsupported_key_becomes_noop_not_a_crash(self):
        """Treechop has no 'use' action; selecting it must not raise."""
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv(supports_use=False))
        native = env.translate_action(env.action_names.index("use"))
        assert "use" not in native

    def test_out_of_range_action_rejected(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        with pytest.raises(ValueError, match="out of range"):
            env.translate_action(len(MINECRAFT_ACTION_SET))


class TestObservations:
    def test_minerl_pov_passes_through_as_hwc_uint8(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        obs, _ = env.reset()
        assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8

    def test_minedojo_chw_is_transposed(self):
        env = MinecraftDiscreteEnv(env=FakeMineDojoEnv())
        obs, _ = env.reset()
        assert obs.shape == (96, 96, 3), "channels-first obs was not transposed"

    def test_float_images_are_rescaled(self):
        frame = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = _extract_pov({"rgb": frame})
        assert out.dtype == np.uint8 and 120 <= int(out.max()) <= 135

    def test_missing_image_key_is_reported(self):
        with pytest.raises(KeyError, match="pov"):
            _extract_pov({"inventory": {}})

    def test_observation_space_matches_actual_frames(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        obs, _ = env.reset()
        assert env.observation_space.shape == obs.shape


class TestStepAPI:
    def test_returns_gymnasium_five_tuple(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        env.reset()
        result = env.step(1)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.dtype == np.uint8
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "discount" in info

    def test_episode_terminates(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        env.reset()
        done = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            if done:
                break
        assert done

    def test_actions_reach_the_backend(self):
        backend = FakeMineRLEnv()
        env = MinecraftDiscreteEnv(env=backend)
        env.reset()
        env.step(env.action_names.index("attack"))
        assert backend.received[-1]["attack"] == 1

    def test_close_propagates(self):
        backend = FakeMineRLEnv()
        MinecraftDiscreteEnv(env=backend).close()
        assert backend.closed


class TestBackendSelection:
    def test_missing_package_gives_install_guidance(self):
        with pytest.raises(ImportError, match="minerl"):
            make_minecraft_env("MineRLTreechop-v0", backend="minerl")

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            MinecraftDiscreteEnv(env_id="x", backend="notarealbackend")

    def test_env_list_is_nonempty(self):
        envs = list_minecraft_envs()
        assert "MineRLTreechop-v0" in envs


class TestIRISCompatibility:
    """The adapter must satisfy what IRISTrainer requires of an env."""

    def test_discrete_action_space_with_n(self):
        env = MinecraftDiscreteEnv(env=FakeMineRLEnv())
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert int(env.action_space.n) == len(MINECRAFT_ACTION_SET)

    def test_frames_survive_iris_preprocessing(self):
        """A MineDojo CHW frame must reach the replay buffer as 64x64 uint8 CHW."""
        pytest.importorskip("cv2")
        from simulusion.configs.iris_config import IRISConfig
        from simulusion.training.train_iris import IRISTrainer

        env = MinecraftDiscreteEnv(env=FakeMineDojoEnv())
        obs, _ = env.reset()

        config = IRISConfig()
        # Borrow the unbound method: constructing a trainer would build models.
        processed = IRISTrainer.preprocess_frame.__get__(
            type("S", (), {"config": config})()
        )(obs)
        assert processed.shape == (3, 64, 64)
        assert processed.dtype == np.uint8
