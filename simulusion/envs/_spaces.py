"""Gymnasium spaces shared by the discrete image adapters.

Kept apart from ``_actions`` because importing this module imports gymnasium,
an optional dependency; ``_actions`` must stay importable without it.
"""

from __future__ import annotations

from typing import Any

from simulusion.utils.gym_compat import gym
import numpy as np


class OneHotActionSpace(gym.spaces.Box):
    """A ``Box[-1, 1]^n`` space whose samples are one-hot-like discrete actions.

    Simulusion's discrete image adapters expose actions this way. ``n`` is the
    number of discrete actions, and the adapters accept either such a vector or
    an integer index, decoded by
    :func:`simulusion.envs._actions.encode_discrete_action`.
    """

    # Marks the Box as a one-hot discrete choice, so agents that pick their
    # policy from the action space use a categorical one, not a Gaussian (as
    # the BSuite, DMLab and Minecraft spaces do).
    one_hot = True

    def __init__(self, n: int, seed: int | None = None):
        self.n = int(n)
        self._rng = np.random.default_rng(seed)
        super().__init__(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        self._rng = np.random.default_rng(int(seed))
        return [int(seed)]

    def sample(self, mask: Any = None, probability: Any = None) -> np.ndarray:
        del mask, probability
        action = -np.ones((self.n,), dtype=np.float32)
        action[int(self._rng.integers(0, self.n))] = 1.0
        return action


def image_space(size: tuple[int, int]) -> Any:
    """The ``(3, H, W)`` uint8 image space the adapters emit."""
    return gym.spaces.Box(0, 255, (3, int(size[0]), int(size[1])), dtype=np.uint8)


__all__ = ["OneHotActionSpace", "image_space"]
