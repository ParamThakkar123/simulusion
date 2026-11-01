import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from simulusion.envs._actions import clip_box_action, encode_discrete_action  # noqa: E402
from simulusion.envs.wrappers import (  # noqa: E402
    ObsDict,
    OneHotAction,
    RenderImage,
    ResizeImage,
    RewardObs,
    SelectAction,
    UUID,
)


# ---------------------------------------------------------------------------
# clip_box_action
# ---------------------------------------------------------------------------


class TestClipBoxAction:
    def test_clips_to_range(self):
        result = clip_box_action(
            np.array([2.0, -2.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        )
        expected = np.array([1.0, -1.0], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_preserves_valid(self):
        result = clip_box_action(
            np.array([0.5, -0.3]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        )
        expected = np.array([0.5, -0.3], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_returns_float32(self):
        result = clip_box_action(np.array([0, 0]), np.array([-1, -1]), np.array([1, 1]))
        assert result.dtype == np.float32

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            clip_box_action(
                np.array([np.nan, 0.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            clip_box_action(
                np.array([1.0, 2.0, 3.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
            )


# ---------------------------------------------------------------------------
# encode_discrete_action
# ---------------------------------------------------------------------------


class TestEncodeDiscreteAction:
    def test_returns_index_and_one_hot(self):
        index, encoded = encode_discrete_action(1, 4)
        assert index == 1
        expected = np.array([-1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        assert np.array_equal(encoded, expected)

    def test_clips_to_valid_range(self):
        index, _ = encode_discrete_action(-5, 4)
        assert index == 0
        index, _ = encode_discrete_action(100, 4)
        assert index == 3

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            encode_discrete_action(np.nan, 4)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            encode_discrete_action(np.array([]), 4)

    def test_invalid_num_actions(self):
        with pytest.raises(ValueError, match=">= 1"):
            encode_discrete_action(0, 0)

    def test_argmax_for_one_hot_input(self):
        index, encoded = encode_discrete_action(np.array([0.1, 0.9, 0.0, 0.0]), 4)
        assert index == 1
        assert encoded[1] == 1.0


# ---------------------------------------------------------------------------
# OneHotAction
# ---------------------------------------------------------------------------


class _DiscreteEnv:
    def __init__(self, n=3):
        self.action_space = gym.spaces.Discrete(n)
        self._obs = np.zeros((4,), dtype=np.float32)

    def reset(self, seed=None):
        return self._obs.copy(), {}

    def step(self, action):
        return self._obs.copy(), 1.0, action == 0, {}


class TestOneHotAction:
    @pytest.fixture
    def env(self):
        return OneHotAction(_DiscreteEnv(n=3))

    def test_action_space_is_box(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (3,)
        assert env.action_space.dtype == np.float32

    def test_valid_one_hot_step(self, env):
        one_hot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        obs, reward, done, info = env.step(one_hot)
        assert reward == 1.0

    def test_invalid_one_hot_raises(self, env):
        bad = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="one-hot"):
            env.step(bad)

    def test_sample_one_hot(self, env):
        action = env.action_space.sample()
        assert action.shape == (3,)
        assert np.abs(action).sum() == 1.0
        assert action.dtype == np.float32

    def test_reset_delegates(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)


# ---------------------------------------------------------------------------
# ObsDict
# ---------------------------------------------------------------------------


class _FlatObsEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, seed=None):
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def step(self, action):
        return np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32), 0.5, False, {}


class TestObsDict:
    @pytest.fixture
    def env(self):
        return ObsDict(_FlatObsEnv(), key="obs")

    def test_observation_space_is_dict(self, env):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "obs" in env.observation_space.spaces

    def test_action_space_passthrough(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)

    def test_reset_returns_dict(self, env):
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "obs" in obs
        assert obs["obs"].shape == (4,)

    def test_step_returns_dict(self, env):
        obs, reward, done, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
        assert isinstance(obs, dict)
        assert "obs" in obs
        assert reward == 0.5


# ---------------------------------------------------------------------------
# RewardObs
# ---------------------------------------------------------------------------


class _DictObsEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (3, 4, 4), dtype=np.uint8),
            }
        )

    def reset(self, seed=None):
        return {"image": np.zeros((3, 4, 4), dtype=np.uint8)}

    def step(self, action):
        return {"image": np.ones((3, 4, 4), dtype=np.uint8)}, 1.5, False, {}


class TestRewardObs:
    @pytest.fixture
    def env(self):
        return RewardObs(_DictObsEnv())

    def test_observation_space_includes_reward(self, env):
        assert "reward" in env.observation_space.spaces
        assert isinstance(env.observation_space.spaces["reward"], gym.spaces.Box)

    def test_reset_injects_reward(self, env):
        obs = env.reset()
        assert obs["reward"] == 0.0

    def test_step_injects_reward(self, env):
        obs, reward, done, info = env.step(None)
        assert obs["reward"] == 1.5


# ---------------------------------------------------------------------------
# ResizeImage
# ---------------------------------------------------------------------------


class _ObsSpaceEnv:
    def __init__(self):
        self.obs_space = {
            "image": gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "state": gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
        }

    def reset(self, seed=None):
        return {
            "image": np.full((8, 8, 3), 64, dtype=np.uint8),
            "state": np.zeros(4, dtype=np.float32),
        }

    def step(self, action):
        return {
            "image": np.full((8, 8, 3), 128, dtype=np.uint8),
            "state": np.ones(4, dtype=np.float32),
        }

    def render(self, *args, **kwargs):
        return np.full((8, 8, 3), 64, dtype=np.uint8)

    def close(self):
        pass


class TestResizeImage:
    @pytest.fixture
    def env(self):
        return ResizeImage(_ObsSpaceEnv(), size=(4, 4))

    def test_resizes_image_key(self, env):
        obs = env.reset()
        assert obs["image"].shape == (4, 4, 3)

    def test_preserves_non_image_keys(self, env):
        obs = env.reset()
        assert obs["state"].shape == (4,)

    def test_obs_space_updated(self, env):
        space = env.obs_space
        assert space["image"].shape == (4, 4, 3)
        assert space["state"].shape == (4,)

    def test_step_resizes(self, env):
        obs = env.step(None)
        assert obs["image"].shape == (4, 4, 3)


# ---------------------------------------------------------------------------
# RenderImage
# ---------------------------------------------------------------------------


class TestRenderImage:
    @pytest.fixture
    def env(self):
        base = _ObsSpaceEnv()
        return RenderImage(base, key="image")

    def test_obs_space_includes_render(self, env):
        space = env.obs_space
        assert "image" in space
        assert space["image"].shape == (8, 8, 3)

    def test_reset_injects_render(self, env):
        obs = env.reset()
        assert "image" in obs
        assert obs["image"].shape == (8, 8, 3)

    def test_step_injects_render(self, env):
        obs = env.step(None)
        assert "image" in obs
        assert obs["image"].shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# SelectAction
# ---------------------------------------------------------------------------


class _DictActionEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 1, (4,), dtype=np.float32)
        self._obs = np.zeros((4,), dtype=np.float32)

    def reset(self, seed=None):
        return self._obs.copy(), {}

    def step(self, action):
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        return self._obs.copy(), 0.5, False, {}


class TestSelectAction:
    @pytest.fixture
    def env(self):
        return SelectAction(_DictActionEnv(), key="move")

    def test_selects_key_from_dict(self, env):
        action = {
            "move": np.array([0.5, -0.5], dtype=np.float32),
            "other": np.array([1.0]),
        }
        obs, reward, done, info = env.step(action)
        assert reward == 0.5


# ---------------------------------------------------------------------------
# UUID
# ---------------------------------------------------------------------------


class _SimpleEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 1, (2,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.ones(2, dtype=np.float32), 1.0, False, {}


class TestUUID:
    @pytest.fixture
    def env(self):
        return UUID(_SimpleEnv())

    def test_id_is_set_on_init(self, env):
        assert env.id is not None
        assert isinstance(env.id, str)
        assert "-" in env.id

    def test_id_changes_on_reset(self, env):
        first = env.id
        env.reset()
        assert env.id != first

    def test_delegates_step(self, env):
        obs, reward, done, info = env.step(0)
        assert reward == 1.0


# ---------------------------------------------------------------------------
# ActionRepeat - early termination
# ---------------------------------------------------------------------------


class _ActionRepeatBaseEnv:
    def __init__(self, max_steps: int = 5):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 1, (4,), dtype=np.float32)
        self._max_steps = max_steps
        self._step = 0

    def reset(self, seed=None):
        self._step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        done = self._step >= self._max_steps
        return (
            np.ones(4, dtype=np.float32) * self._step,
            1.0,
            done,
            {
                "terminated": done,
                "truncated": False,
                "discount": np.array(0.0 if done else 1.0, dtype=np.float32),
            },
        )


class TestActionRepeat:
    def test_repeats_action_specified_amount(self):
        from simulusion.envs.wrappers import ActionRepeat

        env = ActionRepeat(_ActionRepeatBaseEnv(max_steps=10), amount=3)
        env.reset()
        _, reward, done, info = env.step(0)
        assert reward == 3.0
        assert done is False
        assert info["action_repeat"] == 3

    def test_early_termination_reduces_repeat_count(self):
        from simulusion.envs.wrappers import ActionRepeat

        env = ActionRepeat(_ActionRepeatBaseEnv(max_steps=2), amount=5)
        env.reset()
        _, reward, done, info = env.step(0)
        assert reward == 2.0
        assert done is True
        assert info["action_repeat"] == 2
