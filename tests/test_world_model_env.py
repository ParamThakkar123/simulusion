import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from simulusion.envs import WorldModelEnv, make_world_model_env  # noqa: E402


class _CountingWorldModel:
    def reset(self, seed=None, options=None):
        state = {"count": 0}
        obs = np.array([0.0, 0.0], dtype=np.float32)
        return state, obs, {"seed": seed}

    def step(self, state, action):
        count = state["count"] + 1
        action_value = float(np.asarray(action)[0])
        next_state = {"count": count}
        obs = np.array([count, action_value], dtype=np.float32)
        return {
            "state": next_state,
            "observation": obs,
            "reward": count + action_value,
            "terminated": count >= 2,
            "info": {"action_shape": np.asarray(action).shape},
        }


def test_world_model_env_is_gymnasium_compliant_and_rolls_model_state():
    env = WorldModelEnv(
        _CountingWorldModel(),
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        max_episode_steps=3,
        torch_actions=False,
    )

    obs, info = env.reset(seed=7)
    assert np.array_equal(obs, np.array([0.0, 0.0], dtype=np.float32))
    assert info == {"seed": 7}
    assert env.state == {"count": 0}

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert np.array_equal(obs, np.array([1.0, 0.5], dtype=np.float32))
    assert reward == 1.5
    assert terminated is False
    assert truncated is False
    assert info["action_shape"] == (1,)
    assert info["model_state"] == {"count": 1}
    assert info["elapsed_steps"] == 1

    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False


def test_make_world_model_env_and_adapter_callables_support_dict_observations():
    def reset_adapter(model, seed, options):
        return {
            "state": np.array([0.0], dtype=np.float32),
            "observation": {"latent": np.array([0.0, 1.0], dtype=np.float32)},
            "info": {"source": model["name"]},
        }

    def transition_adapter(model, state, action):
        next_state = state + np.asarray(action, dtype=np.float32)
        return {
            "next_state": next_state,
            "obs": {"latent": np.repeat(next_state, 2)},
        }

    def reward_adapter(model, state, obs, action):
        return state.sum()

    def terminal_adapter(model, state, obs, action):
        return bool(state.item() >= 1.0)

    env = make_world_model_env(
        {"name": "fake"},
        observation_space=gym.spaces.Dict(
            {"latent": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)}
        ),
        action_space=gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        reset_fn=reset_adapter,
        transition_fn=transition_adapter,
        reward_fn=reward_adapter,
        terminal_fn=terminal_adapter,
        max_episode_steps=1,
        torch_actions=False,
    )

    obs, info = env.reset()
    assert np.array_equal(obs["latent"], np.array([0.0, 1.0], dtype=np.float32))
    assert info == {"source": "fake"}

    obs, reward, terminated, truncated, info = env.step(
        np.array([1.0], dtype=np.float32)
    )
    assert np.array_equal(obs["latent"], np.array([1.0, 1.0], dtype=np.float32))
    assert reward == 1.0
    assert terminated is True
    assert truncated is True
    assert np.array_equal(info["model_state"], np.array([1.0], dtype=np.float32))


def test_world_model_env_supports_action_transform_and_dict_tuple_reset():
    def reset_adapter(model, seed, options):
        return {"latent": np.array([0.0], dtype=np.float32)}, {"episode": 0}

    def action_transform(model, action):
        return int(action)

    def transition_adapter(model, state, action):
        assert isinstance(action, int)
        next_state = {"latent": np.array([float(action)], dtype=np.float32)}
        return next_state, 2.0, False, False, {"model_state": next_state}

    env = WorldModelEnv(
        object(),
        observation_space=gym.spaces.Dict(
            {"latent": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)}
        ),
        action_space=gym.spaces.Discrete(3),
        reset_fn=reset_adapter,
        transition_fn=transition_adapter,
        action_transform_fn=action_transform,
    )

    obs, info = env.reset()
    assert np.array_equal(obs["latent"], np.array([0.0], dtype=np.float32))
    assert info == {"episode": 0}

    obs, reward, terminated, truncated, info = env.step(2)
    assert np.array_equal(obs["latent"], np.array([2.0], dtype=np.float32))
    assert reward == 2.0
    assert terminated is False
    assert truncated is False
    assert np.array_equal(
        info["model_state"]["latent"], np.array([2.0], dtype=np.float32)
    )


def test_world_model_env_discount_preserved_when_provided_by_model():
    class _DiscountModel:
        def reset(self, seed=None, options=None):
            return {"count": 0}, np.array([0.0], dtype=np.float32), {}

        def step(self, state, action):
            count = state["count"] + 1
            return {
                "state": {"count": count},
                "observation": np.array([float(count)], dtype=np.float32),
                "reward": 1.0,
                "terminated": count >= 3,
                "info": {"discount": np.array(0.5, dtype=np.float32)},
            }

    env = WorldModelEnv(
        _DiscountModel(),
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        torch_actions=False,
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert "discount" in info
    assert np.asarray(info["discount"]).item() == 0.5


def test_world_model_env_no_discount_when_model_omits_it():
    class _NoDiscountModel:
        def reset(self, seed=None, options=None):
            return {}, np.array([0.0], dtype=np.float32), {}

        def step(self, state, action):
            return {
                "state": {},
                "observation": np.array([1.0], dtype=np.float32),
                "reward": 1.0,
                "terminated": False,
                "info": {},
            }

    env = WorldModelEnv(
        _NoDiscountModel(),
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        torch_actions=False,
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert "discount" not in info


def test_world_model_env_requires_reset_before_step():
    env = WorldModelEnv(
        object(),
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        transition_fn=lambda model, state, action: np.array([0.0], dtype=np.float32),
        torch_actions=False,
    )

    with pytest.raises(RuntimeError, match="Call reset"):
        env.step(np.array([0.0], dtype=np.float32))
