"""Replay buffer for DreamerV3.

Three differences from :class:`~simulusion.memory.dreamer_memory.ReplayBuffer`
matter for DreamerV3:

* **Sequences may span episode boundaries.** Instead of rejecting such windows,
  the buffer records an ``is_first`` flag at every episode start and the world
  model resets its recurrent state where that flag is set. This removes the
  rejection-sampling loop and keeps short episodes usable.
* **An online queue.** Each minibatch is filled first from recently collected,
  non-overlapping trajectories and only then topped up with uniform samples.
  This keeps the world model current with the behavior policy.
* **Terminal versus truncation.** ``is_terminal`` marks true environment
  terminations, which is what the continue predictor is trained on; time-limit
  truncations end a trajectory without implying zero future value.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Sequence

import numpy as np

__all__ = ["DreamerV3ReplayBuffer"]


class DreamerV3ReplayBuffer:
    """Ring buffer over transitions, sampled as fixed-length sequences.

    Args:
        size: Maximum number of transitions retained.
        obs_shape: Shape of a single observation.
        action_size: Dimension of the action vector.
        seq_len: Length of the sampled sequences.
        batch_size: Number of sequences per batch.
        obs_dtype: Storage dtype for observations. ``uint8`` for images keeps
            the buffer four times smaller than ``float32``.
        online_fraction: Fraction of each batch drawn from the online queue.
    """

    def __init__(
        self,
        size: int,
        obs_shape: Sequence[int],
        action_size: int,
        seq_len: int,
        batch_size: int,
        obs_dtype: Any = np.uint8,
        online_fraction: float = 0.5,
    ) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be at least 2, got {seq_len}")
        if not 0.0 <= online_fraction <= 1.0:
            raise ValueError(
                f"online_fraction must be in [0, 1], got {online_fraction}"
            )

        self.size = int(size)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_size = int(action_size)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.online_fraction = float(online_fraction)

        self.observations = np.empty((self.size, *self.obs_shape), dtype=obs_dtype)
        self.actions = np.empty((self.size, self.action_size), dtype=np.float32)
        self.rewards = np.empty((self.size,), dtype=np.float32)
        self.is_terminal = np.empty((self.size,), dtype=np.float32)
        self.is_first = np.empty((self.size,), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0
        self._next_is_first = True
        self._online: Deque[int] = deque(maxlen=max(1, self.batch_size * 4))
        self._since_online_start = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs: dict | np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool | float,
        is_terminal: bool | float | None = None,
    ) -> None:
        """Append one transition.

        Args:
            obs: Observation, either a raw array or a dict with an ``"image"``
                (or ``"obs"``) key.
            action: Action taken from this observation.
            reward: Reward received for that action.
            done: Whether the episode ended here, for any reason.
            is_terminal: Whether the episode ended by true termination rather
                than by a time limit. Defaults to ``done``, which is the correct
                choice only when the environment does not truncate.
        """
        array = self._extract_obs(obs)
        self.observations[self.idx] = array
        self.actions[self.idx] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.idx] = float(reward)
        self.is_terminal[self.idx] = float(done if is_terminal is None else is_terminal)
        self.is_first[self.idx] = float(self._next_is_first)

        # Register a fresh online window every `seq_len` steps so the queued
        # trajectories do not overlap.
        if self._since_online_start % self.seq_len == 0:
            self._online.append(self.idx)
        self._since_online_start += 1

        self._next_is_first = bool(done)
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        if done:
            self.episodes += 1

    def _extract_obs(self, obs: dict | np.ndarray) -> np.ndarray:
        if isinstance(obs, dict):
            for key in ("image", "obs"):
                if key in obs:
                    return np.asarray(obs[key])
            raise KeyError(
                f"Observation dict has no 'image' or 'obs' key: {sorted(obs)}"
            )
        return np.asarray(obs)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size if self.full else self.idx

    @property
    def can_sample(self) -> bool:
        return len(self) > self.seq_len + 1

    def _valid_range(self) -> int:
        """Number of positions a sequence may start at without crossing the head."""
        return len(self) - self.seq_len

    def _sample_start(self) -> int:
        upper = self._valid_range()
        if upper <= 0:
            return 0
        if self.full:
            # Start anywhere except a window that would run over the write head.
            offset = int(np.random.randint(0, self.size - self.seq_len))
            return (self.idx + offset) % self.size
        return int(np.random.randint(0, upper))

    def _pop_online_start(self) -> int | None:
        while self._online:
            start = self._online[0]
            if self._window_is_complete(start):
                self._online.popleft()
                return start
            return None
        return None

    def _window_is_complete(self, start: int) -> bool:
        """Whether ``seq_len`` transitions after ``start`` have been written."""
        written = (self.idx - start) % self.size
        if not self.full and start >= self.idx:
            return False
        return written >= self.seq_len

    def sample(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw a batch of sequences.

        Returns:
            Tuple ``(observations, actions, rewards, is_terminal, is_first)``
            with leading dimensions ``(seq_len, batch_size)``.
        """
        if not self.can_sample:
            raise RuntimeError(
                f"Replay buffer holds {len(self)} transitions, which is not "
                f"enough for sequences of length {self.seq_len}."
            )

        starts: list[int] = []
        target_online = int(round(self.batch_size * self.online_fraction))
        while len(starts) < target_online:
            start = self._pop_online_start()
            if start is None:
                break
            starts.append(start)
        while len(starts) < self.batch_size:
            starts.append(self._sample_start())

        index = (
            np.asarray(starts, dtype=np.int64)[None, :]
            + np.arange(self.seq_len, dtype=np.int64)[:, None]
        ) % self.size

        return (
            self.observations[index],
            self.actions[index],
            self.rewards[index],
            self.is_terminal[index],
            self.is_first[index],
        )
