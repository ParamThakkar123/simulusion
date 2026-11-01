import numpy as np
from typing import Tuple, Optional, List


class IRISReplayBuffer:
    """Replay buffer for IRIS (Imagination with auto-Regression over an Inner Speech) training.

    Stores (observation, action, reward, terminal) tuples in a ring buffer
    and supports sampling contiguous sequences for world model training.

    Features:
        - Ring buffer with fixed capacity (FIFO eviction when full)
        - Stores uint8 images for memory efficiency
        - Samples sequences with validation to avoid episode boundaries
        - Supports sequence sampling for temporal learning

    Memory Layout:
        - observations: (capacity, C, H, W) uint8
        - actions: (capacity, action_size) float32
        - rewards: (capacity,) float32
        - terminals: (capacity,) float32

    Args:
        size (int): Maximum number of transitions to store.
        obs_shape (tuple): Shape of observations as (C, H, W).
        action_size (int): Dimension of actions.
        seq_len (int): Length of sequences to sample (default: 20).
        batch_size (int): Number of sequences per batch (default: 64).

    Attributes:
        size (int): Buffer capacity.
        obs_shape (tuple): Observation shape.
        action_size (int): Action dimension.
        seq_len (int): Sequence length.
        batch_size (int): Batch size.
        steps (int): Total transitions added.
        episodes (int): Number of episode terminations observed.
    """

    def __init__(
        self,
        size: int,
        obs_shape: Tuple[int, int, int],
        action_size: int,
        seq_len: int = 20,
        batch_size: int = 64,
    ):
        self.size = size
        self.obs_shape = obs_shape  # (C, H, W)
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

        self.observations = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((size, action_size), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.terminals = np.zeros((size,), dtype=np.float32)

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: float, terminal: bool
    ) -> None:
        """Add a transition to the buffer.

        Args:
            obs: Observation array with shape (C, H, W).
            action: Action array with shape (action_size,).
            reward: Scalar reward value.
            terminal: Boolean indicating if episode terminated.
        """
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = float(terminal)

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes += 1 if terminal else 0

    def sample_sequence(
        self, seq_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of sequences for world model training.

        Returns:
            observations: (batch_size, seq_len+1, C, H, W)
            actions: (batch_size, seq_len, action_size)
            rewards: (batch_size, seq_len)
            terminals: (batch_size, seq_len)
        """
        if seq_len is None:
            seq_len = self.seq_len

        batch_size = self.batch_size
        L = seq_len

        # Sample starting indices
        idxs = self._sample_idxs(L, batch_size)

        # Build sequences
        observations = []
        actions = []
        rewards = []
        terminals = []

        for idx in idxs:
            # Get sequence of observations (L+1 for predicting next frame)
            obs_seq = self.observations[idx : idx + L + 1]
            act_seq = self.actions[idx : idx + L]
            rew_seq = self.rewards[idx : idx + L]
            term_seq = self.terminals[idx : idx + L]

            # Handle wrapping
            if len(obs_seq) < L + 1:
                # Pad by wrapping around
                obs_seq = np.concatenate(
                    [obs_seq, self.observations[: L + 1 - len(obs_seq)]]
                )
                act_seq = np.concatenate([act_seq, self.actions[: L - len(act_seq)]])
                rew_seq = np.concatenate([rew_seq, self.rewards[: L - len(rew_seq)]])
                term_seq = np.concatenate(
                    [term_seq, self.terminals[: L - len(term_seq)]]
                )

            observations.append(obs_seq)
            actions.append(act_seq)
            rewards.append(rew_seq)
            terminals.append(term_seq)

        return (
            np.stack(observations),
            np.stack(actions),
            np.stack(rewards),
            np.stack(terminals),
        )

    def _sample_idxs(self, L: int, n: int) -> np.ndarray:
        """Sample n starting indices whose sequences stay inside one episode.

        A window that straddles a terminal splices the end of one episode onto
        the start of the next; the transformer would then be trained to predict
        a reset frame from the previous episode's final frames.

        The valid starts are enumerated once and drawn from, rather than sampled
        and patched up: the previous version resampled a bad index exactly once
        and never rechecked, so a replacement that also straddled a terminal was
        used anyway.
        """
        # A sequence reads observations[idx : idx + L + 1], so the last valid
        # start for a partially-filled buffer is idx - L - 1; using idx - L let
        # the final sequence read one uninitialised (all-zero) frame.
        valid_start_range = self.size if self.full else self.idx - L - 1

        if valid_start_range <= 0:
            return np.zeros(n, dtype=int)

        candidates = np.arange(valid_start_range)
        if L > 1:
            # terminals[idx + L - 1] is allowed: it is the window's last
            # transition, and its successor frame is the episode's true final
            # observation -- a target the model should learn.
            offsets = np.arange(L - 1)
            window = (candidates[:, None] + offsets[None, :]) % self.size
            candidates = candidates[~(self.terminals[window] > 0).any(axis=1)]

        if candidates.size == 0:
            # Every window crosses a terminal (very short episodes relative to
            # L). Fall back to unfiltered starts rather than failing outright.
            candidates = np.arange(valid_start_range)

        return np.random.choice(candidates, size=n)

    def sample_with_burn_in(
        self, batch_size: int, burn_in: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample start frames together with the frames that precede them.

        IRIS burns in the previous frames to initialise the actor-critic's LSTM
        state before imagining from a given frame (paper A.3). This returns both
        halves so the caller does not have to reason about buffer indexing.

        Episode boundaries are respected: any burn-in frame at or before a
        terminal is replaced by a repeat of the oldest valid frame, so context
        never bleeds across episodes.

        Args:
            batch_size: Number of start frames to draw.
            burn_in: Number of preceding frames to return per start frame.

        Returns:
            start_obs: (batch_size, C, H, W) uint8 frames to imagine from.
            burn_in_obs: (batch_size, burn_in, C, H, W) uint8 preceding frames.
                Empty along axis 1 when ``burn_in`` is 0.
        """
        n = len(self)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        # Start far enough in that a full burn-in window exists where possible.
        # Once the ring buffer has wrapped, also skip starts whose window would
        # straddle the write head: positions just below ``idx`` hold the newest
        # frames while positions just above hold the oldest, so such a window
        # would splice together two distant points in time.
        low = min(burn_in, max(n - 1, 0))
        # Declared unparameterised: the three branches below produce arrays the
        # numpy stubs give different shape types to, and only the values matter.
        starts: np.ndarray
        if low >= n:
            starts = np.full(batch_size, n - 1, dtype=int)
        elif self.full and burn_in > 0:
            candidates: np.ndarray = np.arange(low, n)
            straddles = (candidates - burn_in < self.idx) & (candidates >= self.idx)
            candidates = candidates[~straddles]
            if candidates.size == 0:
                candidates = np.arange(low, n)
            starts = np.random.choice(candidates, size=batch_size)
        else:
            starts = np.random.randint(low, n, size=batch_size)

        start_obs = self.observations[starts]

        if burn_in <= 0:
            empty = np.zeros((batch_size, 0, *self.obs_shape), dtype=np.uint8)
            return start_obs, empty

        burn_in_obs = np.zeros((batch_size, burn_in, *self.obs_shape), dtype=np.uint8)
        for i, start in enumerate(starts):
            first = max(0, start - burn_in)
            window = self.observations[first:start]

            # Truncate at the most recent terminal inside the window so the
            # burn-in stays within one episode.
            terms = self.terminals[first:start]
            terminal_positions = np.nonzero(terms > 0)[0]
            if terminal_positions.size > 0:
                window = window[terminal_positions[-1] + 1 :]

            if window.shape[0] == 0:
                # No valid history: repeat the start frame itself.
                window = start_obs[i][None]

            # Left-pad by repeating the oldest valid frame.
            pad = burn_in - window.shape[0]
            if pad > 0:
                window = np.concatenate([np.repeat(window[:1], pad, axis=0), window])
            burn_in_obs[i] = window

        return start_obs, burn_in_obs

    def sample_single(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Sample a single transition for online updates."""
        idx = np.random.randint(0, self.size if self.full else self.idx)

        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.terminals[idx],
        )

    def __len__(self) -> int:
        return self.size if self.full else self.idx

    @property
    def buffer_capacity(self) -> int:
        """Returns the total capacity of the buffer."""
        return self.size


class IRISOnPolicyBuffer:
    """On-policy buffer for collecting trajectories during environment interaction.

    Used to store the current episode data before adding to the main replay buffer.
    Unlike the main replay buffer, this collects trajectories in a list-based
    structure that's cleared after each episode.

    Useful for:
        - Collecting complete episode trajectories
        - Storing data before batch processing
        - Temporary storage during environment interaction

    Args:
        max_steps (int): Maximum number of steps to store (default: 1000).

    Attributes:
        max_steps (int): Maximum buffer capacity.
        observations (list): List of observations.
        actions (list): List of actions.
        rewards (list): List of rewards.
        terminals (list): List of terminal flags.
    """

    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps
        # Typed lists to satisfy static type checkers
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminals: List[float] = []

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: float, terminal: bool
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(float(terminal))

    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.terminals),
        )

    def __len__(self) -> int:
        return len(self.observations)
