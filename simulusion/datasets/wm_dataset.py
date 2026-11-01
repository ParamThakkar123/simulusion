"""Data generation and dataset classes for World Models.

This module provides utilities for generating rollout data from environments
and PyTorch dataset classes for loading observation sequences.
"""

import numpy as np
from torch.utils.data import Dataset
from albumentations.core.composition import Compose
import glob
import torch
from bisect import bisect
from typing import Any


class RolloutDataset(Dataset):
    """PyTorch Dataset for loading rollout data.

    This dataset loads pre-collected rollout trajectories from disk,
    providing a buffer-based mechanism for efficient data loading.
    It supports train/test splits and custom transforms.

    Attributes:
        root: Root directory containing rollout .npz files.
        transform: Albumentations transform to apply to observations.
        train: If True, use training split; otherwise use test split.
        buffer_size: Maximum number of files to keep in memory.
        num_test_files: Number of files to use for test set.

    Example:
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> dataset = RolloutDataset(
        ...     root='data/carracing',
        ...     transform=transform,
        ...     train=True,
        ...     buffer_size=100,
        ... )
        >>> obs, action, reward, terminal = dataset[0]
    """

    def __init__(
        self,
        root: str,
        transform: Compose,
        train: bool = True,
        buffer_size: int = 1000,
        num_test_files: int = 600,
    ):
        """Initialize the RolloutDataset.

        Args:
            root: Root directory containing rollout .npz files.
            transform: Albumentations transform to apply to observations.
            train: If True, use training split; otherwise use test split.
            buffer_size: Maximum number of files to keep in memory.
            num_test_files: Number of files to use for test set.
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.files = glob.glob(self.root + "/**/*.npz", recursive=True)
        if len(self.files) > num_test_files and num_test_files > 0:
            if train:
                self.files = self.files[:-num_test_files]
            else:
                self.files = self.files[-num_test_files:]
        elif not train:
            self.files = []

        self.cum_size = [0]
        for f in self.files:
            with np.load(f, allow_pickle=False) as data:
                size = len(data["observations"])
            self.cum_size.append(self.cum_size[-1] + size)

        self.buffer = [None] * len(self.files)
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.buffer_fnames: list[str | None] = [None] * len(self.files)

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples across all rollouts.
        """
        return self.cum_size[-1]

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (observation, action, reward, terminal) tensors.
        """
        file_idx = bisect(self.cum_size, idx) - 1
        seq_idx = idx - self.cum_size[file_idx]
        if self.buffer[file_idx] is None:
            self.buffer[file_idx] = np.load(self.files[file_idx], allow_pickle=False)
            self.buffer_fnames[file_idx] = self.files[file_idx]
        data = self.buffer[file_idx]
        return self._get_data(data, seq_idx)

    def _get_data(self, data: Any, idx: int) -> dict:
        """Extract and transform a single data point from rollout.

        Args:
            data: Dictionary containing rollout data arrays.
            idx: Index within the rollout to extract.

        Returns:
            Tuple of (observation, action, reward, terminal) tensors.
        """
        obs = data["observations"][idx]
        action = data["actions"][idx]
        reward = data["rewards"][idx]
        terminal = data["terminals"][idx]

        if self.transform:
            obs = self.transform(image=obs)["image"]

        obs = torch.tensor(obs).permute(2, 0, 1).float() / 255.0
        action = torch.tensor(action).float()
        reward = torch.tensor(reward).float()
        terminal = torch.tensor(terminal).float()

        return dict(observation=obs, action=action, reward=reward, terminal=terminal)

    def load_next_buffer(self) -> None:
        """Load the next batch of rollout files into memory.

        This method implements a circular buffer, loading buffer_size files
        at a time and advancing through the dataset sequentially.
        """
        if self.buffer is None:
            self.buffer = [None] * len(self.files)
        self.buffer_fnames = [None] * len(self.files)

        start_idx = self.buffer_idx
        end_idx = min(start_idx + self.buffer_size, len(self.files))

        for i in range(start_idx, end_idx):
            if self.buffer[i] is None or self.buffer_fnames[i] != self.files[i]:
                self.buffer[i] = np.load(self.files[i], allow_pickle=False)
                self.buffer_fnames[i] = self.files[i]

        self.buffer_idx = end_idx % len(self.files)


class ObservationDataset(RolloutDataset):
    """Dataset for single observation samples (not sequences).

    This dataset extends RolloutDataset to provide individual observations
    rather than sequences, suitable for VAE training.

    Example:
        >>> dataset = ObservationDataset(
        ...     root='data/carracing',
        ...     transform=transform,
        ...     train=True,
        ... )
        >>> obs = dataset[0]
    """

    def _get_data(self, data: Any, idx: int) -> torch.Tensor:  # type: ignore[override]
        obs = data["observations"][idx]
        if self.transform:
            transformed = self.transform(image=obs)
            obs = transformed["image"]
        obs = torch.tensor(obs).permute(2, 0, 1).float()
        obs = obs / 255.0
        return obs


class SequenceDataset(RolloutDataset):
    """Dataset for sequential rollout data.

    This dataset provides sequences of observations, actions, rewards,
    and terminal flags suitable for training recurrent models like MDRNN.

    Attributes:
        seq_len: Length of sequences to return.

    Example:
        >>> dataset = SequenceDataset(
        ...     root='data/carracing',
        ...     transform=transform,
        ...     train=True,
        ...     seq_len=32,
        ... )
        >>> obs, action, reward, terminal, next_obs = dataset[0]
    """

    def __init__(
        self,
        root: str,
        transform: Compose,
        train: bool,
        buffer_size: int,
        num_test_files: int,
        seq_len: int,
    ):
        """Initialize the SequenceDataset.

        Args:
            root: Root directory containing rollout .npz files.
            transform: Albumentations transform to apply to observations.
            train: If True, use training split; otherwise use test split.
            buffer_size: Maximum number of files to keep in memory.
            num_test_files: Number of files to use for test set.
            seq_len: Length of sequences to return.
        """
        super().__init__(root, transform, train, buffer_size, num_test_files)
        self.seq_len = seq_len

    def _get_data(self, data: Any, idx: int) -> tuple:  # type: ignore[override]
        obs_data = data["observations"][idx : idx + self.seq_len]
        if self.transform:
            transformed = [self.transform(image=obs) for obs in obs_data]
            obs_data = [t["image"] for t in transformed]

        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data["actions"][idx + 1 : idx + self.seq_len + 1]
        action = action.astype(np.float32)
        reward = data["rewards"][idx + 1 : idx + self.seq_len + 1]
        terminal = data["terminals"][idx + 1 : idx + self.seq_len + 1].astype(
            np.float32
        )
        return obs, action, reward, terminal, next_obs


class LatentSequenceDataset(Dataset):
    """Dataset for pre-computed latent sequences.

    This dataset uses pre-encoded latent representations instead of raw images,
    which significantly reduces memory usage during RNN training.
    """

    def __init__(
        self,
        latents_arr: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        train: bool,
        buffer_size: int,
        num_test_files: int,
        seq_len: int,
    ):
        super().__init__()
        self.latents_arr = latents_arr
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        total_samples = len(actions)
        test_count = num_test_files * seq_len
        if train:
            self.start_idx = 0
            self.end_idx = max(0, total_samples - test_count)
        else:
            self.start_idx = total_samples - test_count
            self.end_idx = total_samples

        self.seq_len = seq_len
        self.cum_size = list(range(self.start_idx, self.end_idx + 1, seq_len))

    def __len__(self) -> int:
        return len(self.cum_size) - 1

    def __getitem__(self, idx: int) -> tuple:
        start = self.cum_size[idx]
        latent_obs = self.latents_arr[start : start + self.seq_len]
        latent_next_obs = self.latents_arr[start + 1 : start + self.seq_len + 1]
        action = self.actions[start + 1 : start + self.seq_len + 1]
        reward = self.rewards[start + 1 : start + self.seq_len + 1]
        terminal = self.terminals[start + 1 : start + self.seq_len + 1]

        return (
            torch.tensor(latent_obs, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(terminal, dtype=torch.float32),
            torch.tensor(latent_next_obs, dtype=torch.float32),
        )
