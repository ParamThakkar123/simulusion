import numpy as np
import pytest

from simulusion.testing import MockBoxSpace, MockDiscreteSpace, MockImageEnv


def test_mock_discrete_space_is_deterministic_and_validates_membership():
    space = MockDiscreteSpace(n=3, sample_value=2)

    assert space.sample() == 2
    assert space.contains(0)
    assert not space.contains(3)


def test_mock_box_space_samples_midpoint_and_checks_bounds():
    space = MockBoxSpace(shape=(2, 2), low=-1.0, high=1.0)
    sample = space.sample()

    assert sample.shape == (2, 2)
    assert np.all(sample == 0.0)
    assert space.contains(sample)
    assert not space.contains(np.full((2, 2), 2.0))


def test_mock_image_env_follows_gymnasium_step_api():
    env = MockImageEnv(image_shape=(4, 4, 3), action_dim=2, episode_length=2)

    observation, info = env.reset()
    assert observation.shape == (4, 4, 3)
    assert info == {}

    observation, reward, terminated, truncated, info = env.step(1)
    assert observation.shape == (4, 4, 3)
    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert info == {}

    _, _, terminated, _, _ = env.step(0)
    assert terminated

    with pytest.raises(ValueError):
        env.step(2)

    env.close()
    assert env.closed
