"""Replay buffer for DreamerV3.

The buffer stores one record per observation, in the layout the DreamerV3 world
model is defined over (Hafner et al., 2023, Eq. 1). Record ``t`` holds

* ``x_t`` -- the observation,
* ``a_{t-1}`` -- the action that led to it (zero at an episode start), which is
  what the sequence model ``h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`` consumes,
* ``r_t`` -- the reward received on arriving at ``x_t``, which is what the
  reward predictor ``p(r_t | h_t, z_t)`` is trained on (zero at a start),
* ``is_first`` / ``is_last`` / ``is_terminal`` -- episode boundaries. The final
  observation of an episode gets its own record, so the continue predictor
  ``p(c_t | h_t, z_t)`` sees terminal states. ``is_last`` without
  ``is_terminal`` marks a truncation (a time limit, or an abandoned episode),
  after which values are still bootstrapped.

Following the paper's implementation section:

* **Latent states.** The model state ``(h_t, z_t)`` computed while collecting is
  stored with each record. A replayed sequence starts from the stored state of
  the record before it, instead of from zeros, and the trainer writes the fresh
  states of each training rollout back with :meth:`update_latents`.
* **Online queue.** Each minibatch is filled first from non-overlapping
  trajectories that have not been trained on yet, then topped up with uniformly
  sampled trajectories.
* Sequences may span episode boundaries; the world model resets its state
  wherever ``is_first`` is set.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, Deque, Sequence

import numpy as np

__all__ = ["DreamerV3ReplayBuffer"]


class DreamerV3ReplayBuffer:
    """Ring buffer over observation records, sampled as fixed-length sequences.

    Args:
        size: Maximum number of records retained.
        obs_shape: Shape of a single observation, or a mapping from observation
            key to shape for several inputs (``{"image": (3, 64, 64), "instr":
            (1024,)}``). With a mapping, each key is stored in its own array,
            :meth:`add` reads those keys from the observation dict, and
            :meth:`sample` returns ``"observations"`` as a dict.
        action_size: Dimension of the action vector.
        seq_len: Length of the sampled sequences.
        batch_size: Number of sequences per batch.
        obs_dtype: Storage dtype for observations. ``uint8`` for images keeps
            the buffer four times smaller than ``float32``. With several inputs,
            a mapping from key to dtype; keys left out store rank-3 (image)
            observations as ``uint8`` and others as ``float32``.
        online_fraction: Largest fraction of each batch drawn from the online
            queue. The paper fills the batch from the queue first, which is
            ``1.0``; lower values reserve slots for uniform samples.
        deter_size: Size of the recurrent state ``h`` to store per record, or
            ``0`` to not store latent states.
        latent_dim: Number of categorical latents in ``z``.
        latent_classes: Classes per categorical latent.
    """

    #: Arrays written by :func:`simulusion.training.tasks.common.save_replay`.
    replay_arrays: tuple[str, ...] = (
        "observations",
        "prev_actions",
        "rewards",
        "is_first",
        "is_last",
        "is_terminal",
        "deter",
        "stoch",
        "has_latent",
    )

    def __init__(
        self,
        size: int,
        obs_shape: Sequence[int] | Mapping[str, Sequence[int]],
        action_size: int,
        seq_len: int,
        batch_size: int,
        obs_dtype: Any = np.uint8,
        online_fraction: float = 1.0,
        deter_size: int = 0,
        latent_dim: int = 0,
        latent_classes: int = 0,
    ) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be at least 2, got {seq_len}")
        if not 0.0 <= online_fraction <= 1.0:
            raise ValueError(
                f"online_fraction must be in [0, 1], got {online_fraction}"
            )
        if int(size) <= seq_len:
            raise ValueError(f"size={size} must exceed seq_len={seq_len}")
        if deter_size and not (latent_dim and latent_classes):
            raise ValueError("Storing latents needs latent_dim and latent_classes.")
        if latent_classes > np.iinfo(np.uint8).max + 1:
            raise ValueError(f"latent_classes={latent_classes} does not fit in uint8")

        self.size = int(size)
        #: Key -> shape when storing several inputs; None for a single array.
        self.obs_space: dict[str, tuple[int, ...]] | None = None
        if isinstance(obs_shape, Mapping):
            if not obs_shape:
                raise ValueError("obs_shape mapping must name at least one input")
            self.obs_space = {
                str(key): tuple(int(dim) for dim in shape)
                for key, shape in obs_shape.items()
            }
            self.obs_shape: tuple[int, ...] = ()
        else:
            self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_size = int(action_size)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.online_fraction = float(online_fraction)
        self.deter_size = int(deter_size)
        self.latent_dim = int(latent_dim)
        self.latent_classes = int(latent_classes)
        self.stores_latents = self.deter_size > 0

        self.replay_arrays = type(self).replay_arrays
        self._obs_arrays: dict[str, np.ndarray] = {}
        if self.obs_space is None:
            self.observations = np.empty((self.size, *self.obs_shape), dtype=obs_dtype)
        else:
            names = []
            for key, shape in self.obs_space.items():
                array = np.empty(
                    (self.size, *shape), dtype=_obs_dtype(key, shape, obs_dtype)
                )
                self._obs_arrays[key] = array
                # Exposed as attributes so the generic replay checkpointing
                # (which reads `replay_arrays` by name) saves every input.
                setattr(self, _obs_attr(key), array)
                names.append(_obs_attr(key))
            self.replay_arrays = tuple(
                name for name in self.replay_arrays if name != "observations"
            ) + tuple(names)
        self.prev_actions = np.zeros((self.size, self.action_size), dtype=np.float32)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.is_first = np.zeros((self.size,), dtype=np.float32)
        self.is_last = np.zeros((self.size,), dtype=np.float32)
        self.is_terminal = np.zeros((self.size,), dtype=np.float32)
        # Latents are stored compactly: h as float16, z as one class index per
        # categorical. `has_latent` is false until a state has been written.
        latent_rows = self.size if self.stores_latents else 0
        self.deter = np.zeros((latent_rows, self.deter_size), dtype=np.float16)
        self.stoch = np.zeros((latent_rows, self.latent_dim), dtype=np.uint8)
        self.has_latent = np.zeros((self.size,), dtype=bool)

        self.idx = 0
        self.full = False
        #: Environment transitions recorded (one per ``env.step``).
        self.steps = 0
        #: Episodes that have ended (terminated, truncated or abandoned).
        self.episodes = 0
        #: Records ever written; the online queue holds absolute record numbers
        #: so wrapped-around (overwritten) chunks can be told apart.
        self.records = 0
        self._online: Deque[int] = deque(maxlen=max(1, self.batch_size * 16))

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @staticmethod
    def bytes_per_record(
        obs_shape: Sequence[int] | Mapping[str, Sequence[int]],
        obs_dtype: Any,
        action_size: int,
        deter_size: int = 0,
        latent_dim: int = 0,
    ) -> int:
        """Memory one record takes, for sizing the buffer to available RAM."""
        if isinstance(obs_shape, Mapping):
            obs = sum(
                int(np.prod(shape))
                * np.dtype(_obs_dtype(key, tuple(shape), obs_dtype)).itemsize
                for key, shape in obs_shape.items()
            )
        else:
            obs = int(np.prod(obs_shape)) * np.dtype(obs_dtype).itemsize
        # prev action + reward + three flags (float32) + has_latent.
        scalars = action_size * 4 + 4 * 4 + 1
        return obs + scalars + int(deter_size) * 2 + int(latent_dim)

    def add(
        self,
        obs: dict | np.ndarray,
        prev_action: np.ndarray,
        reward: float,
        is_first: bool,
        is_last: bool = False,
        is_terminal: bool = False,
        latent: dict | None = None,
    ) -> None:
        """Append the record for one observation.

        Args:
            obs: Observation ``x_t``, a raw array or a dict with an ``"image"``
                (or ``"obs"``) key.
            prev_action: Action ``a_{t-1}`` that led to ``x_t``. Ignored (stored
                as zeros) when ``is_first``.
            reward: Reward ``r_t`` received on arriving at ``x_t``. Ignored
                (stored as zero) when ``is_first``.
            is_first: ``x_t`` starts an episode.
            is_last: ``x_t`` is the final observation of its episode.
            is_terminal: The episode genuinely ended at ``x_t`` (the environment
                terminated), as opposed to being truncated. Implies ``is_last``.
            latent: Optional model state for ``x_t``: ``{"deter": (deter_size,),
                "stoch": (latent_dim, latent_classes) one-hot}``, as tensors or
                arrays.
        """
        if is_terminal and not is_last:
            raise ValueError("A terminal record must also be the last of its episode.")
        if not is_first and self._last_record_ended_episode():
            raise ValueError(
                "The previous record ended an episode; the next one must be "
                "marked is_first."
            )
        if is_first and len(self) > 0 and not self._last_record_ended_episode():
            # The caller started a new episode without ending the last one, for
            # example by resetting the environment mid-episode. Close it as a
            # truncation so no sequence model ever runs across the reset.
            self.end_episode()

        index = self.idx
        if self.obs_space is None:
            self.observations[index] = self._extract_obs(obs)
        else:
            if not isinstance(obs, Mapping):
                raise TypeError(
                    "This buffer stores several inputs; pass the observation dict."
                )
            missing = [key for key in self.obs_space if key not in obs]
            if missing and len(self.obs_space) == 1:
                # One input stored under another usual name ("image"/"obs"),
                # accepted as the single-array buffer always did.
                (array,) = self._obs_arrays.values()
                array[index] = self._extract_obs(obs)
            elif missing:
                raise KeyError(
                    f"Observation is missing {missing}; it has {sorted(obs)}."
                )
            else:
                for key, array in self._obs_arrays.items():
                    array[index] = np.asarray(obs[key])
        if is_first:
            self.prev_actions[index] = 0.0
            self.rewards[index] = 0.0
        else:
            self.prev_actions[index] = np.asarray(
                prev_action, dtype=np.float32
            ).reshape(-1)
            self.rewards[index] = float(reward)
        self.is_first[index] = float(bool(is_first))
        self.is_last[index] = float(bool(is_last))
        self.is_terminal[index] = float(bool(is_terminal))
        self.has_latent[index] = False
        if latent is not None and self.stores_latents:
            self._write_latent(index, latent["deter"], latent["stoch"])

        if self.records % self.seq_len == 0:
            self._online.append(self.records)
        self.records += 1

        if not is_first:
            self.steps += 1
        if is_last:
            self.episodes += 1
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def end_episode(self) -> None:
        """Mark the newest record as the last of its episode (a truncation).

        Used when collection stops following an episode before it finishes,
        so the next record can start a fresh one. A no-op if the newest record
        already ends an episode.
        """
        if len(self) == 0 or self._last_record_ended_episode():
            return
        self.is_last[(self.idx - 1) % self.size] = 1.0
        self.episodes += 1

    @property
    def episode_open(self) -> bool:
        """Whether the newest record belongs to an episode that has not ended."""
        return len(self) > 0 and not self._last_record_ended_episode()

    def _last_record_ended_episode(self) -> bool:
        if len(self) == 0:
            return True
        return bool(self.is_last[(self.idx - 1) % self.size])

    def _extract_obs(self, obs: dict | np.ndarray) -> np.ndarray:
        if isinstance(obs, dict):
            for key in ("image", "obs"):
                if key in obs:
                    return np.asarray(obs[key])
            raise KeyError(
                f"Observation dict has no 'image' or 'obs' key: {sorted(obs)}"
            )
        return np.asarray(obs)

    def _write_latent(self, index: Any, deter: Any, stoch: Any) -> None:
        deter_np = _to_numpy(deter).reshape(-1, self.deter_size)
        stoch_np = _to_numpy(stoch).reshape(-1, self.latent_dim, self.latent_classes)
        self.deter[index] = deter_np.astype(np.float16).reshape(self.deter[index].shape)
        self.stoch[index] = (
            stoch_np.argmax(axis=-1).astype(np.uint8).reshape(self.stoch[index].shape)
        )
        self.has_latent[index] = True

    def update_latents(self, indices: np.ndarray, deter: Any, stoch: Any) -> None:
        """Write fresh model states back for the records at ``indices``.

        Args:
            indices: ``(T, B)`` record indices, as returned by :meth:`sample`.
            deter: ``(T, B, deter_size)`` recurrent states.
            stoch: ``(T, B, latent_dim, latent_classes)`` one-hot latents.
        """
        if not self.stores_latents:
            return
        flat = np.asarray(indices).reshape(-1)
        self._write_latent(flat, deter, stoch)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size if self.full else self.idx

    @property
    def can_sample(self) -> bool:
        return len(self) > self.seq_len + 1

    @property
    def _oldest(self) -> int:
        return self.idx if self.full else 0

    def _sample_start(self) -> int:
        """Uniform start whose window stays within the written records."""
        if self.full:
            offset = int(np.random.randint(0, self.size - self.seq_len + 1))
            return (self.idx + offset) % self.size
        return int(np.random.randint(0, self.idx - self.seq_len + 1))

    def _pop_online_start(self) -> int | None:
        """Oldest queued chunk that is fully written and not yet overwritten."""
        oldest_kept = self.records - len(self)
        while self._online:
            number = self._online[0]
            if number + self.seq_len > self.records:
                return None  # Still being written; later chunks are newer.
            self._online.popleft()
            if number >= oldest_kept:
                return number % self.size
        return None

    def _initial_state(
        self, starts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stored model state of the record before each window start."""
        batch = len(starts)
        deter = np.zeros((batch, self.deter_size), dtype=np.float32)
        stoch = np.zeros((batch, self.latent_dim, self.latent_classes), np.float32)
        valid = np.zeros((batch,), dtype=np.float32)
        if not self.stores_latents:
            return deter, stoch, valid
        for row, start in enumerate(starts):
            if self.is_first[start] or start == self._oldest:
                continue  # Episode start, or the predecessor was overwritten.
            prev = (start - 1) % self.size
            if not self.has_latent[prev]:
                continue
            deter[row] = self.deter[prev].astype(np.float32)
            stoch[row, np.arange(self.latent_dim), self.stoch[prev]] = 1.0
            valid[row] = 1.0
        return deter, stoch, valid

    def sample(self) -> dict[str, np.ndarray]:
        """Draw a batch of sequences.

        Returns:
            Dict of ``(seq_len, batch_size, ...)`` arrays: ``observations``,
            ``prev_actions``, ``rewards``, ``is_first``, ``is_last``,
            ``is_terminal`` and the record ``indices``; plus the ``(batch_size,
            ...)`` starting model state ``init_deter`` / ``init_stoch`` and
            ``init_valid``, which is 0 where no stored state was available and
            the model should start from zeros.
        """
        if not self.can_sample:
            raise RuntimeError(
                f"Replay buffer holds {len(self)} records, which is not "
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

        start_arr = np.asarray(starts, dtype=np.int64)
        index = (
            start_arr[None, :] + np.arange(self.seq_len, dtype=np.int64)[:, None]
        ) % self.size
        init_deter, init_stoch, init_valid = self._initial_state(start_arr)

        observations: np.ndarray | dict[str, np.ndarray]
        if self.obs_space is None:
            observations = self.observations[index]
        else:
            observations = {
                key: array[index] for key, array in self._obs_arrays.items()
            }
        return {
            "observations": observations,
            "prev_actions": self.prev_actions[index],
            "rewards": self.rewards[index],
            "is_first": self.is_first[index],
            "is_last": self.is_last[index],
            "is_terminal": self.is_terminal[index],
            "indices": index,
            "init_deter": init_deter,
            "init_stoch": init_stoch,
            "init_valid": init_valid,
        }


def _obs_attr(key: str) -> str:
    return f"obs__{key}"


def _obs_dtype(key: str, shape: Sequence[int], obs_dtype: Any) -> Any:
    """Storage dtype for one input: an explicit entry, else by rank."""
    if isinstance(obs_dtype, Mapping):
        if key in obs_dtype:
            return obs_dtype[key]
        return np.uint8 if len(tuple(shape)) == 3 else np.float32
    if len(tuple(shape)) == 3:
        return obs_dtype
    return np.float32


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
