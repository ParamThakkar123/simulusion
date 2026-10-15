"""
TinyWorlds Dataset Loaders

Loads pre-processed video datasets from HuggingFace for training Genie-style world models.
Based on: https://github.com/AlmondGod/tinyworlds

Available datasets:
- PICO_DOOM: Minimal Doom gameplay
- PONG: Classic Pong
- ZELDA: Zelda Ocarina of Time (2D)
- POLE_POSITION: Racing game
- SONIC: Sonic the Hedgehog
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError:
    h5py = None
    logger.warning("h5py not installed. Install with: pip install h5py")

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning(
        "huggingface_hub not installed. Install with: pip install huggingface_hub"
    )


@dataclass
class TinyWorldsConfig:
    """Configuration for TinyWorlds datasets."""

    dataset_name: str = "SONIC"
    num_frames: int = 16
    image_size: int = 64
    batch_size: int = 4
    num_workers: int = 4
    cache_dir: Optional[str] = None
    split: str = "train"


class TinyWorldsDataset(Dataset):
    """Dataset for TinyWorlds game video data.

    Loads pre-processed frames from HuggingFace datasets repository.
    """

    DATASET_CONFIGS = {
        "PICO_DOOM": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "picodoom_frames.h5",
            "description": "Minimal Doom gameplay",
        },
        "PONG": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "pong_frames.h5",
            "description": "Classic Pong",
        },
        "ZELDA": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "zelda_frames.h5",
            "description": "Zelda Ocarina of Time (2D)",
        },
        "POLE_POSITION": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "pole_position_frames.h5",
            "description": "Racing game",
        },
        "SONIC": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "sonic_frames.h5",
            "description": "Sonic the Hedgehog",
        },
    }

    def __init__(
        self,
        dataset_name: str = "SONIC",
        num_frames: int = 16,
        image_size: int = 64,
        split: str = "train",
        cache_dir: Optional[str] = None,
        download: bool = True,
        data_file: Optional[str] = None,
    ) -> None:
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_CONFIGS.keys())}"
            )

        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        self.num_frames = num_frames
        self.image_size = image_size
        self.split = split
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.data_file = data_file

        self._data_file: Optional[Any] = None
        self._data_file_pid: Optional[int] = None
        self._data_path: Optional[Path] = None
        self.num_samples = 0
        self.video_length = 0

        if data_file:
            self._load_data(Path(data_file))
        elif download:
            self._download_or_load_data()
        else:
            self._load_data()

    def _get_default_cache_dir(self) -> str:
        cache: str = os.path.expanduser("~/.cache/tinyworlds")
        os.makedirs(cache, exist_ok=True)
        return cache

    def _get_local_path(self) -> Path:
        return Path(self.cache_dir) / self.config["filename"]

    def _download_or_load_data(self) -> None:
        local_path = self._get_local_path()

        if local_path.exists():
            logger.info(f"Found cached dataset at {local_path}")
            self._load_data()
            return

        if not HF_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub not installed. Cannot download datasets.\n"
                "Install with: pip install huggingface_hub"
            )

        logger.info(f"Downloading {self.dataset_name} dataset from HuggingFace...")
        logger.info("This may take several minutes depending on your connection.")

        try:
            downloaded_path = hf_hub_download(
                repo_id=self.config["repo_id"],
                filename=self.config["filename"],
                repo_type="dataset",
                cache_dir=self.cache_dir,
            )
            downloaded_file = Path(downloaded_path)
            if not downloaded_file.exists():
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")

            local_path = self._get_local_path()
            if downloaded_file != local_path and not local_path.exists():
                shutil.copy2(downloaded_file, local_path)

            logger.info(f"Downloaded to: {local_path}")
            self._load_data()
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {self.dataset_name}: {e}\n"
                "You can manually download from: "
                f"https://huggingface.co/datasets/{self.config['repo_id']}/tree/main"
            )

    def _load_data(self, file_path: Optional[Path] = None) -> None:
        if h5py is None:
            raise RuntimeError(
                "h5py not installed. Cannot load dataset files. Install with: pip install h5py"
            )

        local_path = file_path or self._get_local_path()

        if not local_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {local_path}")

        # Only the metadata is read here, and the file is closed again. HDF5 is
        # not fork-safe: a handle opened in this process and inherited by forked
        # DataLoader workers corrupts HDF5's state once several of them read
        # through it, which segfaults -- in a notebook the kernel just dies with
        # no traceback. Each process opens its own handle on first read instead.
        with h5py.File(local_path, "r") as f:
            if "videos" in f:
                self._data_key = "videos"
            elif "frames" in f:
                self._data_key = "frames"
            else:
                available_keys = list(f.keys())
                raise KeyError(
                    f"No 'videos' or 'frames' key found. Available: {available_keys}"
                )
            # h5py dataset objects are dynamically typed; access shape via getattr
            shape = getattr(f[self._data_key], "shape", None)

        self._data_path = local_path
        self._close_data_file()

        if not isinstance(shape, tuple):
            raise ValueError(
                f"Unable to determine data shape for dataset: {self.dataset_name}"
            )

        if len(shape) == 5:
            self.num_samples = shape[0]
            self.video_length = shape[1]
            self.raw_height = shape[2]
            self.raw_width = shape[3]
            self.channels = shape[4]
            self.data_layout = "NTHWC"
        elif len(shape) == 4 and shape[-1] in (1, 3, 4):
            # One continuous stream of frames, which is how TinyWorlds ships
            # every game: (N, H, W, C). Each sample is a window of
            # `num_frames` consecutive frames from it.
            self.total_frames = shape[0]
            self.video_length = self.num_frames
            self.num_samples = max(0, self.total_frames - self.num_frames + 1)
            self.raw_height = shape[1]
            self.raw_width = shape[2]
            self.channels = shape[3]
            self.data_layout = "NHWC"
        elif len(shape) == 4:
            self.num_samples = shape[0]
            self.video_length = shape[1]
            self.raw_height = shape[2]
            self.raw_width = shape[3]
            self.channels = 1
            self.data_layout = "NTHW"
        else:
            raise ValueError(f"Unexpected data shape: {shape}")

        logger.info(
            f"Loaded {self.dataset_name}: {self.num_samples} videos, "
            f"{self.video_length} frames each, {self.raw_height}x{self.raw_width}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def _open_data_file(self) -> Any:
        """Return this process's HDF5 handle, opening it on first use.

        The handle is tagged with the pid that opened it, so a forked DataLoader
        worker that inherited the parent's handle opens a fresh one of its own
        rather than reading through the shared one.
        """
        if self._data_file is None or self._data_file_pid != os.getpid():
            path = getattr(self, "_data_path", None)
            assert path is not None and h5py is not None, "Data file is not loaded"
            # A handle inherited across fork is dropped, never closed: closing it
            # would act on HDF5 state that still belongs to the parent.
            self._data_file = h5py.File(path, "r")
            self._data_file_pid = os.getpid()
        return self._data_file

    def _close_data_file(self) -> None:
        f = getattr(self, "_data_file", None)
        if f is not None and getattr(self, "_data_file_pid", None) == os.getpid():
            f.close()
        self._data_file = None
        self._data_file_pid = None

    def __getstate__(self) -> Dict[str, Any]:
        # Spawned workers (Windows, macOS) receive the dataset by pickling, and
        # an h5py handle cannot be pickled; each worker opens its own on first read.
        state = self.__dict__.copy()
        state["_data_file"] = None
        state["_data_file_pid"] = None
        return state

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self._open_data_file()[self._data_key]
        if self.data_layout == "NHWC":
            video = data[idx : idx + self.num_frames]
        else:
            video = data[idx][:]

        if not isinstance(video, np.ndarray):
            video = np.array(video)

        if self.data_layout in ("NTHWC", "NHWC"):
            pass
        elif self.data_layout == "NTHW":
            video = np.expand_dims(video, axis=-1)
        else:
            raise ValueError(f"Unknown data layout: {self.data_layout}")

        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            video = video[indices]
        else:
            padding = np.tile(video[-1:], (self.num_frames - total_frames, 1, 1, 1))
            video = np.concatenate([video, padding], axis=0)

        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        else:
            video = video.astype(np.uint8)

        # (T, H, W, C) -> (C, T, H, W). This must be a permute: reshaping the
        # flattened (T*C) axis into (C, T) interleaves channels across frames.
        video = torch.from_numpy(video[..., :3]).float()
        video = video.permute(3, 0, 1, 2)
        if video.shape[0] == 1:
            video = video.expand(3, -1, -1, -1)
        video = video.contiguous()

        if self.image_size is not None and (
            video.shape[2] != self.image_size or video.shape[3] != self.image_size
        ):
            resize_transform = transforms.Resize((self.image_size, self.image_size))
            video = resize_transform(video)

        video = video / 255.0

        return video

    def __del__(self) -> None:
        self._close_data_file()

    def get_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        return {
            "name": self.dataset_name,
            "description": self.config["description"],
            "num_samples": self.num_samples,
            "video_length": self.video_length,
            "raw_resolution": f"{self.raw_height}x{self.raw_width}",
            "channels": self.channels,
        }


class TinyWorldsDataLoader:
    """Factory class for creating TinyWorlds dataloaders."""

    DATASET_NAMES = list(TinyWorldsDataset.DATASET_CONFIGS.keys())

    @staticmethod
    def create_dataloader(
        dataset_name: str = "SONIC",
        num_frames: int = 16,
        image_size: int = 64,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True,
        cache_dir: Optional[str] = None,
        download: bool = True,
        data_file: Optional[str] = None,
        val_split: float = 0.0,
        split: str = "train",
        split_seed: int = 0,
    ) -> Tuple[TinyWorldsDataset, DataLoader]:
        dataset = TinyWorldsDataset(
            dataset_name=dataset_name,
            num_frames=num_frames,
            image_size=image_size,
            cache_dir=cache_dir,
            download=download,
            data_file=data_file,
        )

        # ``TinyWorldsDataset`` carries a ``split`` attribute but never divided
        # the clips on it, so every "split" saw the whole dataset. Partition
        # here instead, seeded so that the train and val calls agree and stay
        # disjoint.
        subset: Any = dataset
        if val_split:
            if split not in ("train", "val"):
                raise ValueError(f"split must be 'train' or 'val', got {split!r}")
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            if min(train_size, val_size) == 0:
                raise ValueError(
                    f"val_split={val_split} leaves {train_size} train / {val_size} "
                    f"val clips out of {len(dataset)}; pick a fraction that gives "
                    f"both sides at least one clip"
                )
            generator = torch.Generator().manual_seed(split_seed)
            train_subset, val_subset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=generator
            )
            subset = val_subset if split == "val" else train_subset

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # Pinned staging lets the H2D copy overlap compute instead of
            # blocking on a pageable buffer. Only meaningful with a CUDA target,
            # and pinning costs host memory, so it follows the worker setting.
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=shuffle,
        )

        logger.info(
            f"Created {dataset_name} {split} dataloader: {len(subset)} samples, "
            f"{len(loader)} batches, batch_size={batch_size}"
        )

        return dataset, loader

    @staticmethod
    def list_available_datasets() -> List[str]:
        """List all available dataset names."""
        return TinyWorldsDataLoader.DATASET_NAMES

    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict:
        """Get information about a specific dataset without downloading."""
        if dataset_name not in TinyWorldsDataset.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(TinyWorldsDataset.DATASET_CONFIGS.keys())}"
            )
        return TinyWorldsDataset.DATASET_CONFIGS[dataset_name]


def create_tinyworlds_dataloader(
    dataset_name: str = "SONIC",
    num_frames: int = 16,
    image_size: int = 64,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
    download: bool = True,
    data_file: Optional[str] = None,
    val_split: float = 0.0,
    split: str = "train",
    split_seed: int = 0,
) -> Tuple[TinyWorldsDataset, DataLoader]:
    return TinyWorldsDataLoader.create_dataloader(
        dataset_name=dataset_name,
        num_frames=num_frames,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        cache_dir=cache_dir,
        download=download,
        data_file=data_file,
        val_split=val_split,
        split=split,
        split_seed=split_seed,
    )


def download_all_datasets(cache_dir: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Download all available TinyWorlds datasets.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        Dictionary mapping dataset names to local file paths
    """
    """Download all available TinyWorlds datasets.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        Dictionary mapping dataset names to local file paths
    """
    cache_dir = cache_dir or os.path.expanduser("~/.cache/tinyworlds")
    results: Dict[str, Optional[str]] = {}

    for dataset_name in TinyWorldsDataLoader.list_available_datasets():
        logger.info(f"Downloading {dataset_name}...")
        try:
            dataset, _ = create_tinyworlds_dataloader(
                dataset_name=dataset_name,
                download=True,
                cache_dir=cache_dir,
            )
            results[dataset_name] = str(dataset._get_local_path())
            logger.info(f"  -> {dataset_name} downloaded successfully")
        except Exception as e:
            logger.error(f"  -> {dataset_name} failed: {e}")
            results[dataset_name] = None

    return results
