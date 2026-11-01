"""Reusable lightweight test doubles for environment-facing tests.

The helpers in this module intentionally avoid importing Gym/Gymnasium so tests
can build deterministic mock environments without optional runtime backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class MockDiscreteSpace:
    """Small stand-in for Gym's discrete action space."""

    n: int
    sample_value: int = 0

    def sample(self) -> int:
        """Return a deterministic action inside the space."""
        if not 0 <= self.sample_value < self.n:
            raise ValueError("sample_value must be in [0, n)")
        return self.sample_value

    def contains(self, value: Any) -> bool:
        """Return whether ``value`` is a valid discrete action."""
        return isinstance(value, (int, np.integer)) and 0 <= int(value) < self.n


@dataclass(slots=True)
class MockBoxSpace:
    """Small stand-in for Gym's Box space with deterministic samples."""

    shape: tuple[int, ...]
    low: float = 0.0
    high: float = 1.0
    dtype: type[np.generic] | np.dtype = np.float32

    def sample(self) -> np.ndarray:
        """Return the midpoint of the box for deterministic tests."""
        midpoint = (self.low + self.high) / 2.0
        return np.full(self.shape, midpoint, dtype=self.dtype)

    def contains(self, value: Any) -> bool:
        """Return whether ``value`` has this shape and lies within bounds."""
        arr = np.asarray(value)
        return (
            arr.shape == self.shape
            and bool(np.all(arr >= self.low))
            and bool(np.all(arr <= self.high))
        )


class MockImageEnv:
    """Deterministic image-observation environment for unit tests.

    The class mimics the small subset of the Gym API used throughout the test
    suite: ``reset()``, ``step()``, ``render()``, ``close()``, and the
    ``observation_space``/``action_space`` attributes.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (64, 64, 3),
        action_dim: int = 2,
        episode_length: int = 5,
        reward: float = 1.0,
    ) -> None:
        self.image_shape = image_shape
        self.episode_length = episode_length
        self.reward = reward
        self.observation_space = MockBoxSpace(
            image_shape, low=0, high=255, dtype=np.uint8
        )
        self.action_space = MockDiscreteSpace(action_dim)
        self.steps = 0
        self.closed = False

    def _observation(self) -> np.ndarray:
        return np.full(self.image_shape, self.steps % 256, dtype=np.uint8)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return a Gymnasium-style tuple."""
        del seed, options
        self.steps = 0
        return self._observation(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one step and return a Gymnasium-style transition."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action for MockImageEnv: {action!r}")
        self.steps += 1
        terminated = self.steps >= self.episode_length
        return self._observation(), self.reward, terminated, False, {}

    def render(self) -> np.ndarray:
        """Return the current observation as an RGB array."""
        return self._observation()

    def close(self) -> None:
        """Mark the environment as closed."""
        self.closed = True


__all__ = ["MockBoxSpace", "MockDiscreteSpace", "MockImageEnv"]
