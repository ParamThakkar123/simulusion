"""The one-hot action space shared by every discrete adapter."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from simulusion.envs._spaces import OneHotActionSpace  # noqa: E402


def test_samples_are_one_hot_vectors_in_the_space():
    space = OneHotActionSpace(5, seed=0)
    assert space.shape == (5,) and space.n == 5 and space.one_hot
    for _ in range(20):
        action = space.sample()
        assert space.contains(action)
        assert (action == 1.0).sum() == 1 and (action == -1.0).sum() == 4


def test_seeded_samples_match_the_previous_per_backend_spaces():
    # The BSuite, DMLab, Minecraft and Procgen copies all drew
    # rng.integers(n) from default_rng(seed); seeded runs must not change.
    rng = np.random.default_rng(7)
    expected = [int(rng.integers(0, 6)) for _ in range(10)]
    space = OneHotActionSpace(6, seed=7)
    assert [int(np.argmax(space.sample())) for _ in range(10)] == expected


def test_reseeding_restarts_the_sequence():
    space = OneHotActionSpace(4, seed=1)
    assert space.seed(3) == [3]
    first = [space.sample() for _ in range(5)]
    space.seed(3)
    np.testing.assert_array_equal(first, [space.sample() for _ in range(5)])


def test_seed_without_an_argument_is_deterministic():
    a, b = OneHotActionSpace(4, seed=2), OneHotActionSpace(4, seed=2)
    assert a.seed() == b.seed()
    np.testing.assert_array_equal(a.sample(), b.sample())


def test_rejects_an_empty_action_set():
    with pytest.raises(ValueError, match="n >= 1"):
        OneHotActionSpace(0)


@pytest.mark.parametrize(
    "module",
    [
        "bsuite_env",
        "dmlab",
        "minecraft_diamond",
        "procgen_env",
        "crafter_env",
        "memory_maze_env",
        "atari100k_env",
        "craftax_env",
    ],
)
def test_every_discrete_adapter_uses_the_shared_space(module):
    import importlib

    adapter = importlib.import_module(f"simulusion.envs.{module}")
    assert getattr(adapter, "OneHotActionSpace", OneHotActionSpace) is OneHotActionSpace
    private = [
        name
        for name, value in vars(adapter).items()
        if isinstance(value, type)
        and issubclass(value, OneHotActionSpace.__mro__[1])
        and value is not OneHotActionSpace
        and name.endswith("ActionSpace")
    ]
    assert private == []
