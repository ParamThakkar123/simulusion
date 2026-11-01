import pytest
import numpy as np
import torch
from pathlib import Path

h5py = pytest.importorskip("h5py")


class TestTinyWorldsDataset:
    """Tests for TinyWorlds dataset loading and shape handling."""

    def _create_mock_h5(self, tmp_path: Path, layout: str, raw_shape: tuple) -> Path:
        """Create a mock .h5 file for testing."""
        path = tmp_path / f"test_{layout}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "frames", data=np.random.randint(0, 255, raw_shape).astype(np.uint8)
            )
        return path

    def test_output_shape_is_CTHW(self, tmp_path):
        """Test that dataset output is (C, T, H, W) per Genie convention."""
        from simulusion.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHWC", (10, 16, 64, 64, 3))

        class TestLoader(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestLoader.__new__(TestLoader)
        dataset.dataset_name = "TEST"
        dataset.config = TestLoader.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        sample = dataset[0]
        assert sample.shape == (3, 16, 64, 64), (
            f"Expected (C,T,H,W)=(3,16,64,64) but got {sample.shape}"
        )

    def test_grayscale_to_rgb(self, tmp_path):
        """Test that grayscale (1 channel) is converted to RGB (3 channels)."""
        from simulusion.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHW", (5, 16, 64, 64))

        class TestGrayDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestGrayDataset.__new__(TestGrayDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestGrayDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        sample = dataset[0]
        assert sample.shape[0] == 3, (
            f"Grayscale should become 3 channels, got shape {sample.shape}"
        )

    def test_nthw_layout_handling(self, tmp_path):
        """Test that NTHW layout (no channel dim) is handled correctly."""
        from simulusion.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHW", (5, 16, 64, 64))

        class TestNTHWDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestNTHWDataset.__new__(TestNTHWDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestNTHWDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        assert dataset.data_layout == "NTHW"
        sample = dataset[0]
        assert sample.shape == (3, 16, 64, 64), (
            f"NTHW should produce (C,T,H,W)=(3,16,64,64) but got {sample.shape}"
        )

    def test_nthwc_layout_handling(self, tmp_path):
        """Test that NTHWC layout is handled correctly."""
        from simulusion.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHWC", (5, 16, 64, 64, 3))

        class TestNTHWCDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestNTHWCDataset.__new__(TestNTHWCDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestNTHWCDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        assert dataset.data_layout == "NTHWC"
        sample = dataset[0]
        assert sample.shape == (3, 16, 64, 64), (
            f"NTHWC should produce (C,T,H,W)=(3,16,64,64) but got {sample.shape}"
        )

    def test_batch_dimensions_with_dataloader(self, tmp_path):
        """Test that DataLoader produces correct batch dimensions."""
        from simulusion.datasets.tinyworlds import TinyWorldsDataset
        from torch.utils.data import DataLoader

        h5_path = self._create_mock_h5(tmp_path, "NTHWC", (20, 16, 64, 64, 3))

        class TestBatchDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestBatchDataset.__new__(TestBatchDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestBatchDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        B, C, T, H, W = batch.shape
        assert B == 4, f"Batch size should be 4, got {B}"
        assert C == 3, f"Channels should be 3 (RGB), got {C}"
        assert T == 16, f"Frames should be 16, got {T}"
        assert H == 64, f"Height should be 64, got {H}"
        assert W == 64, f"Width should be 64, got {W}"


class TestTinyWorldsLayouts:
    """Samples must be real clips: consecutive frames, channels kept intact."""

    @staticmethod
    def _dataset(tmp_path: Path, data: np.ndarray, num_frames: int):
        from simulusion.datasets.tinyworlds import TinyWorldsDataset

        path = tmp_path / "frames.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("frames", data=data)
        return TinyWorldsDataset(
            dataset_name="SONIC",
            num_frames=num_frames,
            image_size=data.shape[-2],
            data_file=str(path),
        )

    @staticmethod
    def _labelled_frames(n: int, size: int = 8) -> np.ndarray:
        """Frame t has value t in R, 100 + t in G and 200 - t in B."""
        t = np.arange(n, dtype=np.uint8)[:, None, None]
        frames = np.empty((n, size, size, 3), dtype=np.uint8)
        frames[..., 0] = t
        frames[..., 1] = 100 + t
        frames[..., 2] = 200 - t
        return frames

    def test_flat_frame_stream_yields_consecutive_windows(self, tmp_path):
        """TinyWorlds stores each game as one (N, H, W, 3) stream of frames."""
        dataset = self._dataset(tmp_path, self._labelled_frames(20), num_frames=4)

        assert dataset.data_layout == "NHWC"
        assert len(dataset) == 17

        clip = dataset[5] * 255.0
        assert clip.shape == (3, 4, 8, 8)
        expected_t = torch.arange(5, 9, dtype=torch.float32)
        assert torch.allclose(clip[0, :, 0, 0], expected_t)
        assert torch.allclose(clip[1, :, 0, 0], 100 + expected_t)
        assert torch.allclose(clip[2, :, 0, 0], 200 - expected_t)

    def test_clip_layout_keeps_channels_and_time_apart(self, tmp_path):
        """(N, T, H, W, C) clips: channel c of frame t must stay channel c."""
        clips = self._labelled_frames(6).reshape(2, 3, 8, 8, 3)
        dataset = self._dataset(tmp_path, clips, num_frames=3)

        clip = dataset[1] * 255.0
        expected_t = torch.arange(3, 6, dtype=torch.float32)
        assert torch.allclose(clip[0, :, 0, 0], expected_t)
        assert torch.allclose(clip[1, :, 0, 0], 100 + expected_t)
        assert torch.allclose(clip[2, :, 0, 0], 200 - expected_t)


class TestTinyWorldsWorkers:
    """The HDF5 handle must never be shared across DataLoader worker processes.

    HDF5 is not fork-safe. The dataset used to open the file in ``__init__`` and
    keep the handle, so forked workers all read through the parent's inherited
    handle; that segfaults, and in a notebook the kernel dies with no traceback.
    Spawned workers (Windows, macOS) failed differently: the handle cannot be
    pickled at all.
    """

    def _dataset(self, tmp_path: Path, num_videos: int = 8):
        from simulusion.datasets.tinyworlds import TinyWorldsDataset

        path = tmp_path / "clips.h5"
        rng = np.random.default_rng(0)
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "frames",
                data=rng.integers(0, 255, (num_videos, 8, 16, 16, 3), dtype=np.uint8),
            )
        return TinyWorldsDataset(
            dataset_name="SONIC",
            num_frames=8,
            image_size=16,
            cache_dir=str(tmp_path),
            download=False,
            data_file=str(path),
        )

    def _batches(self, dataset, num_workers: int, context=None):
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=num_workers,
            multiprocessing_context=context,
        )
        return torch.cat(list(loader))

    def test_construction_leaves_no_open_handle(self, tmp_path):
        dataset = self._dataset(tmp_path)
        assert dataset._data_file is None
        assert len(dataset) == 8

    def test_inherited_handle_is_replaced_not_reused(self, tmp_path):
        import os

        dataset = self._dataset(tmp_path)
        dataset[0]
        inherited = dataset._data_file
        # What a forked worker sees: a live handle opened by another process.
        dataset._data_file_pid = os.getpid() + 1

        dataset[1]

        assert dataset._data_file is not inherited
        assert dataset._data_file_pid == os.getpid()
        # The parent's handle is left alone rather than closed from the child.
        assert inherited.id.valid
        inherited.close()

    @pytest.mark.parametrize("method", ["fork", "spawn"])
    def test_workers_read_same_data_as_main_process(self, tmp_path, method):
        import multiprocessing as mp
        import torch

        if method not in mp.get_all_start_methods():
            pytest.skip(f"{method} start method not available on this platform")

        dataset = self._dataset(tmp_path)
        expected = self._batches(dataset, num_workers=0)
        # Reading in the parent first is the case that crashed: the parent holds
        # an open handle when the workers are forked.
        assert dataset._data_file is not None

        got = self._batches(dataset, num_workers=2, context=mp.get_context(method))

        assert torch.equal(got, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
