"""Tests for World Models dataset classes."""

import numpy as np
import torch
import pytest
from unittest.mock import patch

pytest.importorskip("albumentations")
from simulusion.datasets.wm_dataset import (
    RolloutDataset,
    ObservationDataset,
    SequenceDataset,
    LatentSequenceDataset,
)


class TestRolloutDataset:
    @pytest.fixture
    def mock_npz(self):
        data = {
            "observations": np.random.randint(0, 256, (50, 64, 64, 3), dtype=np.uint8),
            "actions": np.random.randn(50, 3).astype(np.float32),
            "rewards": np.random.randn(50).astype(np.float32),
            "terminals": np.zeros(50, dtype=bool),
        }
        return data

    @pytest.fixture
    def dataset(self, tmp_path, mock_npz):
        data_dir = tmp_path / "rollouts"
        data_dir.mkdir()
        npz_path = str(data_dir / "rollout_0.npz")
        np.savez(npz_path, **mock_npz)
        with patch("glob.glob", return_value=[npz_path]):
            ds = RolloutDataset(
                root=str(data_dir),
                transform=None,
                train=True,
                buffer_size=10,
                num_test_files=0,
            )
        return ds

    def test_len(self, dataset, mock_npz):
        assert len(dataset) == len(mock_npz["observations"])

    def test_get_item_shape(self, dataset):
        item = dataset[0]
        assert "observation" in item
        assert "action" in item
        assert "reward" in item
        assert "terminal" in item
        assert isinstance(item["observation"], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)
        assert isinstance(item["reward"], torch.Tensor)
        assert isinstance(item["terminal"], torch.Tensor)
        assert item["observation"].shape == (3, 64, 64)
        assert item["action"].shape == (3,)

    def test_get_item_dtype(self, dataset):
        item = dataset[0]
        assert item["observation"].dtype == torch.float32
        assert item["action"].dtype == torch.float32
        assert item["reward"].dtype == torch.float32
        assert item["terminal"].dtype == torch.float32


class TestObservationDataset:
    @pytest.fixture
    def mock_npz(self):
        data = {
            "observations": np.random.randint(0, 256, (50, 64, 64, 3), dtype=np.uint8),
            "actions": np.random.randn(50, 3).astype(np.float32),
            "rewards": np.random.randn(50).astype(np.float32),
            "terminals": np.zeros(50, dtype=bool),
        }
        return data

    @pytest.fixture
    def dataset(self, tmp_path, mock_npz):
        data_dir = tmp_path / "obs"
        data_dir.mkdir()
        npz_path = str(data_dir / "rollout_0.npz")
        np.savez(npz_path, **mock_npz)
        with patch("glob.glob", return_value=[npz_path]):
            ds = ObservationDataset(
                root=str(data_dir),
                transform=None,
                train=True,
                buffer_size=10,
                num_test_files=0,
            )
        return ds

    def test_get_item_returns_tensor(self, dataset):
        obs = dataset[0]
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (3, 64, 64)
        assert obs.dtype == torch.float32

    def test_values_normalized(self, dataset):
        obs = dataset[0]
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0


class TestSequenceDataset:
    @pytest.fixture
    def mock_npz(self):
        data = {
            "observations": np.random.randint(0, 256, (100, 64, 64, 3), dtype=np.uint8),
            "actions": np.random.randn(100, 3).astype(np.float32),
            "rewards": np.random.randn(100).astype(np.float32),
            "terminals": np.zeros(100, dtype=bool),
        }
        return data

    @pytest.fixture
    def dataset(self, tmp_path, mock_npz):
        data_dir = tmp_path / "seq"
        data_dir.mkdir()
        npz_path = str(data_dir / "rollout_0.npz")
        np.savez(npz_path, **mock_npz)
        with patch("glob.glob", return_value=[npz_path]):
            ds = SequenceDataset(
                root=str(data_dir),
                transform=None,
                train=True,
                buffer_size=10,
                num_test_files=0,
                seq_len=16,
            )
        return ds

    def test_get_item_returns_5_tuple(self, dataset):
        result = dataset[0]
        assert len(result) == 5
        obs, action, reward, terminal, next_obs = result
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 15

    def test_sequence_length(self, dataset):
        obs, action, reward, terminal, next_obs = dataset[0]
        assert len(obs) == 15
        assert len(next_obs) == 15
        assert action.shape == (16, 3)
        assert reward.shape == (16,)
        assert terminal.shape == (16,)

    def test_next_obs_is_shifted(self, dataset):
        obs, action, reward, terminal, next_obs = dataset[0]
        assert len(obs) == 15
        assert len(next_obs) == 15


class TestLatentSequenceDataset:
    @pytest.fixture
    def arrays(self):
        n = 500
        return {
            "latents": np.random.randn(n, 32).astype(np.float32),
            "actions": np.random.randn(n, 3).astype(np.float32),
            "rewards": np.random.randn(n).astype(np.float32),
            "terminals": np.zeros(n, dtype=np.float32),
        }

    @pytest.fixture
    def dataset(self, arrays):
        return LatentSequenceDataset(
            latents_arr=arrays["latents"],
            actions=arrays["actions"],
            rewards=arrays["rewards"],
            terminals=arrays["terminals"],
            train=True,
            buffer_size=10,
            num_test_files=1,
            seq_len=32,
        )

    def test_len(self, dataset, arrays):
        assert len(dataset) > 0
        assert len(dataset) < len(arrays["actions"])

    def test_get_item_shapes(self, dataset):
        latent_obs, action, reward, terminal, latent_next_obs = dataset[0]
        assert latent_obs.shape == (32, 32)
        assert latent_next_obs.shape == (32, 32)
        assert action.shape == (32, 3)
        assert reward.shape == (32,)
        assert terminal.shape == (32,)

    def test_next_latent_is_shifted(self, dataset):
        latent_obs, action, reward, terminal, latent_next_obs = dataset[0]
        assert torch.allclose(latent_obs[1:], latent_next_obs[:-1])

    def test_train_test_split(self, arrays):
        train_ds = LatentSequenceDataset(
            latents_arr=arrays["latents"],
            actions=arrays["actions"],
            rewards=arrays["rewards"],
            terminals=arrays["terminals"],
            train=True,
            buffer_size=10,
            num_test_files=2,
            seq_len=32,
        )
        test_ds = LatentSequenceDataset(
            latents_arr=arrays["latents"],
            actions=arrays["actions"],
            rewards=arrays["rewards"],
            terminals=arrays["terminals"],
            train=False,
            buffer_size=10,
            num_test_files=2,
            seq_len=32,
        )
        assert len(train_ds) + len(test_ds) <= len(arrays["actions"]) // 32
