import pytest
import numpy as np
import tempfile
from pathlib import Path

pytest.importorskip("cv2")
pytest.importorskip("h5py")
from simulusion.datasets import (
    VideoFolderDataset,
    ImageFolderDataset,
    NumPyDataset,
    create_video_dataloader,
    DatasetConfig,
    VideoDatasetConfig,
)


class TestVideoFolderDataset:
    """Test VideoFolderDataset with synthetic data."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = VideoFolderDataset(
                data_source=tmpdir,
                num_frames=8,
                image_size=32,
            )
            assert dataset.num_frames == 8
            assert dataset.image_size == 32


class TestImageFolderDataset:
    """Test ImageFolderDataset with synthetic data."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_dir = Path(tmpdir) / "seq_0"
            seq_dir.mkdir()
            (seq_dir / "0.png").write_bytes(b"fake")
            (seq_dir / "1.png").write_bytes(b"fake")

            dataset = ImageFolderDataset(
                data_source=tmpdir,
                num_frames=4,
                image_size=32,
            )
            assert dataset.num_frames == 4


class TestNumPyDataset:
    """Test NumPyDataset with synthetic data."""

    def test_from_array(self):
        data = np.random.rand(10, 16, 64, 64, 3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, data)
            try:
                dataset = NumPyDataset(
                    data_source=f.name,
                    num_frames=8,
                    image_size=64,
                )
                sample = dataset[0]
                assert sample.shape[0] == 16  # Full video returned
            finally:
                pass  # Don't delete on Windows while file is open


class TestCreateVideoDataloader:
    """Test the factory function."""

    def test_create_video_folder_dataloader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, loader = create_video_dataloader(
                dataset_type="video_folder",
                data_source=tmpdir,
                num_frames=8,
                image_size=32,
                batch_size=2,
            )
            assert dataset is not None
            assert loader is not None

    def test_create_numpy_dataloader(self):
        data = np.random.rand(10, 16, 32, 32, 3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, data)
            try:
                dataset, loader = create_video_dataloader(
                    dataset_type="numpy",
                    data_source=f.name,
                    num_frames=8,
                    image_size=32,
                    batch_size=2,
                )
                assert dataset is not None
                assert loader is not None
            finally:
                pass


class TestDatasetConfig:
    """Test dataset configuration classes."""

    def test_dataset_config_defaults(self):
        config = DatasetConfig()
        assert config.num_frames == 16
        assert config.image_size == 64
        assert config.batch_size == 4

    def test_video_dataset_config(self):
        config = VideoDatasetConfig(
            dataset_type="video_folder",
            data_source="/path/to/data",
            num_frames=8,
            image_size=32,
            batch_size=2,
        )
        assert config.dataset_type == "video_folder"
        assert config.num_frames == 8
        assert config.batch_size == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
