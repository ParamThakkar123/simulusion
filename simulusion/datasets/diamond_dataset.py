import numpy as np
from typing import Dict, Tuple
import torch
from simulusion.utils.device import default_device_name


def to_model_domain(frames: torch.Tensor) -> torch.Tensor:
    """Map uint8 pixels in [0, 255] to the diffusion model's domain, [-1, 1].

    DIAMOND fixes sigma_data = 0.5 (Appendix C), which is the *standard
    deviation of the data distribution*. The EDM preconditioners c_in, c_out and
    c_skip (eqs. 9-12) are derived on the assumption that the data is centred and
    has that spread. Frames scaled to [0, 1] have mean ~0.5 and a standard
    deviation well under 0.5, so every preconditioner is miscalibrated and the
    network's input/output no longer sit at unit variance. Centring on [-1, 1]
    restores the assumption the paper's constant was chosen under.
    """
    return frames.float().div_(127.5).sub_(1.0)


def to_pixel_domain(frames: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_model_domain`: [-1, 1] -> [0, 1] for display/logging."""
    return (frames + 1.0).mul(0.5).clamp(0.0, 1.0)


class ReplayBuffer:
    """
    Replay buffer for storing environment interactions.
    Stores (observation, action, reward, done, next_observation) tuples.
    """

    def __init__(
        self,
        capacity: int = 1000,
        obs_shape: Tuple[int, int, int] = (64, 64, 3),
        action_dim: int = 1,
        device: str = default_device_name(),
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

        self.observations = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        # Separate from `dones`: a time-limit truncation ends the episode (so no
        # sequence may span it) but is *not* an MDP terminal, so it must not
        # become a positive label for the termination head.
        self.truncations = np.zeros((capacity,), dtype=np.bool_)
        self.next_observations = np.zeros((capacity,) + obs_shape, dtype=np.uint8)

        self.position = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
        truncated: bool = False,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            done: True when the environment *terminated* (game over, or a life
                lost when ``terminate_on_life_loss`` is set). This is the target
                for R_psi's termination head.
            truncated: True when the episode ended for an external reason such
                as a time limit. Recorded only so sequence sampling can avoid
                spanning the boundary.
        """
        self.observations[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncations[self.position] = truncated
        self.next_observations[self.position] = next_obs

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def episode_boundaries(self) -> np.ndarray:
        """Transitions after which the environment was reset (terminal or not)."""
        return self.dones | self.truncations

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        obs = to_model_domain(
            torch.from_numpy(self.observations[indices]).to(self.device)
        )
        # observations stored as H,W,C -> convert to C,H,W
        obs = obs.permute(0, 3, 1, 2)

        next_obs = to_model_domain(
            torch.from_numpy(self.next_observations[indices]).to(self.device)
        )
        next_obs = next_obs.permute(0, 3, 1, 2)

        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        if actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": torch.from_numpy(self.rewards[indices]).float().to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).bool().to(self.device),
            "next_obs": next_obs,
        }

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        burn_in: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a sequence of transitions for training.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Total sequence length (burn_in + horizon)
            burn_in: Number of initial frames to use for conditioning

        Returns:
            Dictionary with tensors of shape (batch_size, sequence_length, ...)
        """
        max_start = self.size - sequence_length - 1
        if max_start < 0:
            max_start = 0

        start_indices = np.random.randint(0, max_start + 1, size=batch_size)

        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        next_obs_seq = []

        for i in range(batch_size):
            start = start_indices[i]
            indices = np.arange(start, start + sequence_length + 1)

            obs_seq.append(self.observations[indices[:-1]])
            action_seq.append(self.actions[indices[:-1]])
            reward_seq.append(self.rewards[indices[:-1]])
            done_seq.append(self.dones[indices[:-1]])
            next_obs_seq.append(self.next_observations[indices[:-1]])

        obs = to_model_domain(torch.from_numpy(np.stack(obs_seq)).to(self.device))
        # obs: (B, T, H, W, C) -> (B, T, C, H, W)
        obs = obs.permute(0, 1, 4, 2, 3)

        next_obs = to_model_domain(
            torch.from_numpy(np.stack(next_obs_seq)).to(self.device)
        )
        next_obs = next_obs.permute(0, 1, 4, 2, 3)

        actions = torch.from_numpy(np.stack(action_seq)).long().to(self.device)
        if actions.ndim > 2 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": torch.from_numpy(np.stack(reward_seq)).float().to(self.device),
            "dones": torch.from_numpy(np.stack(done_seq)).bool().to(self.device),
            "next_obs": next_obs,
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= min_size

    def state_dict(self) -> dict:
        """Return a serializable state dict for checkpointing.

        Contains numpy arrays and scalar metadata so it can be saved with
        torch.save or numpy.save.
        """
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "truncations": self.truncations,
            "next_observations": self.next_observations,
            "position": int(self.position),
            "size": int(self.size),
            "capacity": int(self.capacity),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state previously produced by `state_dict()`.

        This will resize internal arrays if the saved capacity differs from the
        current buffer capacity.
        """
        obs = state["observations"]
        actions = state["actions"]
        rewards = state["rewards"]
        dones = state["dones"]
        # Checkpoints written before truncations were tracked simply have none.
        truncations = state.get("truncations")
        if truncations is None:
            truncations = np.zeros_like(dones, dtype=np.bool_)
        next_obs = state["next_observations"]
        pos = int(state.get("position", 0))
        size = int(state.get("size", 0))

        # allocate arrays with saved capacity shapes
        self.capacity = int(state.get("capacity", obs.shape[0]))
        self.observations = np.zeros((self.capacity,) + self.obs_shape, dtype=np.uint8)
        self.next_observations = np.zeros(
            (self.capacity,) + self.obs_shape, dtype=np.uint8
        )
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.truncations = np.zeros((self.capacity,), dtype=np.bool_)

        # copy available data up to saved size
        n = min(size, obs.shape[0], self.capacity)
        if n > 0:
            self.observations[:n] = obs[:n]
            self.next_observations[:n] = next_obs[:n]
            self.actions[:n] = actions[:n]
            self.rewards[:n] = rewards[:n]
            self.dones[:n] = dones[:n]
            self.truncations[:n] = truncations[:n]

        self.position = int(pos) % self.capacity if self.capacity > 0 else 0
        self.size = min(int(size), self.capacity)


class SequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for sampling sequences from the replay buffer.
    Used for training the diffusion world model.

    Sequences never straddle an episode boundary. The replay buffer is a flat
    ring of transitions, so a window spanning a ``done`` splices the end of one
    episode onto the start of the next -- the world model would then be asked to
    predict the first frame of a fresh episode from the last frames of the
    previous one, a transition the environment never produces.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int = 5,  # L (conditioning) + 1 (next frame)
        burn_in: int = 4,
    ):
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self._starts = self._valid_starts()

    def _valid_starts(self) -> np.ndarray:
        """Window starts whose sequence stays inside a single episode.

        A window [start, start + L) is rejected when any of its first L-1
        transitions terminated: dones[start + L - 1] is fine, since that is the
        last transition of the window and its successor frame is still the
        correct target (the episode's final observation).
        """
        size = self.replay_buffer.size
        length = self.sequence_length
        last_start = size - length
        if last_start < 0:
            return np.empty(0, dtype=np.int64)

        candidates = np.arange(last_start + 1, dtype=np.int64)
        if length <= 1 or candidates.size == 0:
            return candidates

        boundaries = self.replay_buffer.episode_boundaries()[:size].astype(bool)
        # cumulative count of boundaries, so "any boundary in [s, s+L-1)" is an
        # O(1) lookup per candidate instead of an O(L) scan.
        cumulative = np.concatenate([[0], np.cumsum(boundaries)])
        interior = cumulative[candidates + length - 1] - cumulative[candidates]
        return candidates[interior == 0]

    def __len__(self) -> int:
        return int(self._starts.size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence starting at the ``idx``-th episode-safe position."""
        start = int(self._starts[idx])
        indices = np.arange(start, start + self.sequence_length)
        # keep numpy arrays separate to avoid mypy inferring ndarray types
        obs_seq_np = self.replay_buffer.observations[indices]
        action_seq_np = self.replay_buffer.actions[indices]
        reward_seq_np = self.replay_buffer.rewards[indices]
        done_seq_np = self.replay_buffer.dones[indices]
        # The diffusion target is the frame that immediately follows the last
        # conditioning frame: next_observations[i] is the successor of
        # observations[i], so the successor of obs_seq[-1] lives at indices[-1],
        # not one slot further on. Reading indices[-1] + 1 (the previous
        # behaviour) trained D_theta to predict x_{t+2} from x_{<=t}, a one-step
        # shift the conditioning actions do not account for.
        next_obs_np = self.replay_buffer.next_observations[indices[-1]]

        # stay on CPU; the training loop batches and transfers to GPU
        obs_seq = to_model_domain(torch.from_numpy(obs_seq_np))
        # (T, H, W, C) -> (T, C, H, W)
        if obs_seq.ndim == 4:
            obs_seq = obs_seq.permute(0, 3, 1, 2)

        next_obs = to_model_domain(torch.from_numpy(next_obs_np))
        # ensure next_obs is (C, H, W)
        if next_obs.ndim == 3:
            next_obs = next_obs.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

        action_seq = torch.from_numpy(action_seq_np).long()
        if action_seq.ndim > 1 and action_seq.shape[-1] == 1:
            action_seq = action_seq.squeeze(-1)

        rewards = torch.from_numpy(reward_seq_np).float()
        dones = torch.from_numpy(done_seq_np).bool()

        return {
            "obs_seq": obs_seq,
            "action_seq": action_seq,
            "actions": action_seq,  # duplicate key for compatibility
            "rewards": rewards,
            "dones": dones,
            "next_obs": next_obs,
        }
