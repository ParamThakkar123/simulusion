import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, List, Tuple, Union, Any, cast, Sequence
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import logging
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    num_frames: int = 16
    image_size: int = 64
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


class VideoDatasetBase(Dataset):
    """Base class for video datasets.

    All video datasets should inherit from this class and implement
    the _load_video method.
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[str], List[Path]],
        num_frames: int = 16,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        self.data_source: Union[str, Path, List[str], List[Path]] = data_source
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize

        # video_paths is a list of Path-like objects or numeric indices for in-memory datasets
        # Use a union element type so subclasses may return either Path or int indices.
        # video_paths may be a sequence of Path objects (file-based datasets)
        # or integer indices (in-memory datasets). Use Sequence for flexibility.
        self.video_paths: Sequence[Union[Path, int]] = self._get_video_paths()

    def _get_video_paths(self) -> Sequence[Union[Path, int]]:
        """Get list of video file paths. Override in subclass."""
        raise NotImplementedError

    def _load_video(self, idx: int) -> torch.Tensor:
        """Load a single video. Override in subclass."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        video = self._load_video(idx)

        if self.transform is not None:
            video = self.transform(video)

        if self.normalize:
            video = video / 255.0 if video.max() > 1.0 else video

        return video


class VideoFolderDataset(VideoDatasetBase):
    """Dataset that loads videos from a folder.

    Supports common video formats: .mp4, .avi, .mkv, .webm

    **Usage**::

        dataset = VideoFolderDataset(
            data_source="/path/to/videos",
            num_frames=16,
            image_size=64
        )
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[str], List[Path]],
        num_frames: int = 16,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        extensions: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".webm", ".mov"),
        recursive: bool = True,
    ):
        self.extensions = extensions
        self.recursive = recursive
        super().__init__(data_source, num_frames, image_size, transform, normalize)

    def _get_video_paths(self) -> Sequence[Union[Path, int]]:
        if isinstance(self.data_source, (list, tuple)):
            return [Path(p) for p in self.data_source]

        data_path = Path(cast(Union[str, Path], self.data_source))  # type: ignore[redundant-cast]

        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if data_path.is_file():
            return [data_path]

        video_paths: List[Path] = []

        if self.recursive:
            for ext in self.extensions:
                video_paths.extend(list(data_path.rglob(f"*{ext}")))
        else:
            for ext in self.extensions:
                video_paths.extend(list(data_path.glob(f"*{ext}")))

        return sorted(video_paths)

    def _load_video(self, idx: int) -> torch.Tensor:
        video_path = cast(Path, self.video_paths[idx])

        cap = cv2.VideoCapture(str(video_path))
        frames_list: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        frames_arr = np.array(frames_list)

        frames_arr = self._sample_frames(frames_arr)

        frames_pil = [Image.fromarray(frame) for frame in frames_arr]
        frames_resized = [
            f.resize((self.image_size, self.image_size)) for f in frames_pil
        ]

        frames_array = np.stack([np.array(f) for f in frames_resized])
        return torch.from_numpy(frames_array).float()

    def _sample_frames(self, frames: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        # Accept either an ndarray or a list of frames and normalize to ndarray
        if isinstance(frames, list):
            frames = np.array(frames)
        total_frames = int(len(frames))

        if total_frames == self.num_frames:
            return frames

        if total_frames < self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            return frames[indices]

        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        return frames[indices]


class ImageFolderDataset(VideoDatasetBase):
    """Dataset that loads image sequences from folders.

    Each subfolder is treated as a video sequence.

    **Usage**::

        dataset = ImageFolderDataset(
            data_source="/path/to/images",
            num_frames=16,
            image_size=64
        )
    """

    def __init__(
        self,
        data_source: Union[str, Path, List[str], List[Path]],
        num_frames: int = 16,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        image_sort_key: Optional[Callable] = None,
    ):
        self.extensions = extensions
        self.image_sort_key = image_sort_key or (
            lambda x: int(x.stem.split(".")[0]) if x.stem.isdigit() else 0
        )
        super().__init__(data_source, num_frames, image_size, transform, normalize)

    def _get_video_paths(self) -> Sequence[Union[Path, int]]:
        if isinstance(self.data_source, (list, tuple)):
            return [Path(p) for p in self.data_source]

        data_path = Path(self.data_source)

        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if data_path.is_file():
            return [data_path.parent]

        sequences = [d for d in data_path.iterdir() if d.is_dir()]
        return sorted(sequences)

    def _load_video(self, idx: int) -> torch.Tensor:
        seq_path = cast(Path, self.video_paths[idx])

        image_files: List[Path] = []
        for ext in self.extensions:
            image_files.extend(seq_path.glob(f"*{ext}"))

        image_files = sorted(image_files, key=self.image_sort_key)

        if len(image_files) == 0:
            raise ValueError(f"No images found in {seq_path}")

        frames = []
        for img_path in image_files[: self.num_frames]:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            frames.append(np.array(img))

        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        frames_array = np.stack(frames[: self.num_frames])
        return torch.from_numpy(frames_array).float()


class NumPyDataset(VideoDatasetBase):
    """Dataset that loads videos from numpy files.

    Supports .npy and .npz files.

    **Usage**::

        dataset = NumPyDataset(
            data_source="/path/to/videos.npy",
            num_frames=16,
            image_size=64
        )
    """

    def __init__(
        self,
        data_source: Union[str, Path],
        num_frames: int = 16,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        key: Optional[str] = None,
    ):
        self.key = key

        data_path = Path(data_source)

        if data_path.suffix == ".npz":
            self.npz_data = np.load(data_path, allow_pickle=False)
            if self.key is None:
                self.key = list(self.npz_data.keys())[0]
            data = self.npz_data[self.key]
            self.num_samples = data.shape[0] if len(data.shape) >= 4 else 1
            self.is_5d = len(data.shape) == 5
        else:
            self.npz_data = None
            data = np.load(data_path, allow_pickle=False)
            self.num_samples = data.shape[0] if len(data.shape) >= 4 else 1
            self.is_5d = len(data.shape) == 5

        super().__init__(data_source, num_frames, image_size, transform, normalize)

    def _get_video_paths(self) -> Sequence[Union[Path, int]]:
        # For numpy-backed datasets we represent entries as integer indices
        return list(range(self.num_samples))

    def _load_video(self, idx: int) -> torch.Tensor:
        if self.npz_data is not None:
            data = self.npz_data[self.key]
        else:
            data = np.load(str(self.data_source), allow_pickle=False)

        if self.is_5d:
            video = data[idx]
        else:
            video = data

        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video).float()
        else:
            video = torch.tensor(video).float()

        return video


class RLEnvironmentDataset(VideoDatasetBase):
    """Dataset for RL environment recordings.

    Loads trajectories stored as:
    - .npz files with 'observations' and 'actions' keys
    - Directory with episode folders

    **Usage**::

        dataset = RLEnvironmentDataset(
            data_source="/path/to/rl_episodes",
            num_frames=16,
            image_size=64
        )
    """

    def __init__(
        self,
        data_source: Union[str, Path],
        num_frames: int = 16,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        obs_key: str = "observations",
    ):
        self.obs_key = obs_key
        super().__init__(data_source, num_frames, image_size, transform, normalize)

    def _get_video_paths(self) -> Sequence[Union[Path, int]]:
        data_path = Path(cast(Union[str, Path], self.data_source))

        if data_path.is_file() and data_path.suffix == ".npz":
            self.single_file = True
            return [data_path]

        episode_files = list(data_path.rglob("*.npz"))
        return sorted(episode_files)

    def _load_video(self, idx: int) -> torch.Tensor:
        if hasattr(self, "single_file") and self.single_file:
            data = np.load(str(cast(Path, self.video_paths[idx])), allow_pickle=False)
            observations = data[self.obs_key]
        else:
            data = np.load(str(cast(Path, self.video_paths[idx])), allow_pickle=False)
            observations = data[self.obs_key]

        if isinstance(observations, dict):
            if "image" in observations:
                observations = observations["image"]
            elif "pixels" in observations:
                observations = observations["pixels"]
            else:
                observations = list(observations.values())[0]

        observations = np.array(observations)

        if observations.ndim == 3:
            observations = np.expand_dims(observations, axis=-1)

        if observations.shape[-1] in [1, 3] and observations.ndim == 4:
            observations = np.transpose(observations, (0, 3, 1, 2))

        total_frames = observations.shape[0]

        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            observations = observations[indices]
        else:
            padding = np.tile(
                observations[-1:], (self.num_frames - total_frames, 1, 1, 1)
            )
            observations = np.concatenate([observations, padding], axis=0)

        processed = []
        for frame in observations:
            if frame.shape[0] == 1:
                frame = frame[0]
            if frame.shape[-1] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.image_size, self.image_size))
            processed.append(np.array(img))

        return torch.from_numpy(np.stack(processed)).float()


class HDF5Dataset(VideoDatasetBase):
    """Dataset that loads videos from HDF5 files.

    Supports pre-processed video datasets stored in HDF5 format.
    Expected structure: HDF5 file with 'videos' dataset of shape (N, T, H, W, C)
    or (N, T, C, H, W).

    **Usage**::

        dataset = HDF5Dataset(
            data_source="/path/to/videos.h5",
            num_frames=16,
            image_size=64
        )
    """

    def __init__(
        self,
        data_source: Union[str, Path],
        num_frames: int = 16,
        image_size: int = 64,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        key: str = "videos",
        memmap: bool = False,
    ):
        self.key = key
        self.memmap = memmap
        self._h5_file = None

        data_path = Path(data_source)
        if not data_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {data_path}")

        import h5py

        with h5py.File(data_path, "r") as f:
            if key not in f:
                available_keys = list(f.keys())
                raise KeyError(
                    f"Key '{key}' not found in HDF5. Available: {available_keys}"
                )
            data = cast(Any, f[key])
            # mypy doesn't understand h5py dataset types; treat as Any for shape access
            if len(data.shape) == 5:
                self.num_samples = int(data.shape[0])
                self.video_length = int(data.shape[1])
                self.raw_height = int(data.shape[2])
                self.raw_width = int(data.shape[3])
                self.channels = int(data.shape[4])
                self.data_layout = "NCTHW" if data.shape[-1] <= 4 else "NTHWC"
            elif len(data.shape) == 4:
                self.num_samples = int(data.shape[0])
                self.video_length = int(data.shape[1])
                self.raw_height = int(data.shape[2])
                self.raw_width = int(data.shape[3])
                self.channels = 1
                self.data_layout = "NTHW"
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

        super().__init__(data_source, num_frames, image_size, transform, normalize)

    def _get_video_paths(self) -> Sequence[Union[Path, int]]:
        return list(range(self.num_samples))

    def _open_h5(self) -> Any:
        if self._h5_file is None:
            import h5py

            self._h5_file = h5py.File(self.data_source, "r" if self.memmap else "r")
        return self._h5_file

    def _load_video(self, idx: int) -> torch.Tensor:
        f = self._open_h5()
        data = cast(Any, f[self.key])

        if self.memmap:
            video = data[idx]
        else:
            video = data[idx][:]

        if not isinstance(video, np.ndarray):
            video = np.array(video)

        if self.data_layout == "NCTHW":
            video = np.transpose(video, (0, 2, 3, 1))
        elif self.data_layout == "NTHWC":
            video = np.transpose(video, (0, 1, 3, 2))

        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)

        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            video = video[indices]
        else:
            padding = np.tile(video[-1:], (self.num_frames - total_frames, 1, 1, 1))
            video = np.concatenate([video, padding], axis=0)

        if video.shape[1] != self.image_size or video.shape[2] != self.image_size:
            processed = []
            for frame in video:
                if frame.max() <= 1.0 and frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                img = Image.fromarray(frame)
                img = img.resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                processed.append(np.array(img))
            video = np.stack(processed)
        else:
            if video.max() <= 1.0:
                video = (video * 255).astype(np.uint8)

        return torch.from_numpy(video).float()

    def __len__(self) -> int:
        return self.num_samples

    def __del__(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()


def create_video_dataloader(
    dataset_type: str,
    data_source: Union[str, Path, List[str]],
    num_frames: int = 16,
    image_size: int = 64,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs: Any,
) -> Tuple[Dataset, DataLoader]:
    """Factory function to create video dataloaders.

    Args:
        dataset_type: Type of dataset ("video_folder", "image_folder", "numpy", "rl")
        data_source: Path or list of paths to data
        num_frames: Number of frames per video
        image_size: Target image size (height and width)
        batch_size: Batch size for dataloader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        **kwargs: Additional arguments for specific dataset types

    Returns:
        Tuple of (dataset, dataloader)

    **Usage**::

        dataset, loader = create_video_dataloader(
            dataset_type="video_folder",
            data_source="/path/to/videos",
            num_frames=16,
            image_size=64,
            batch_size=4
        )
    """
    dataset_classes = {
        "video_folder": VideoFolderDataset,
        "image_folder": ImageFolderDataset,
        "numpy": NumPyDataset,
        "rl": RLEnvironmentDataset,
    }

    if dataset_type not in dataset_classes:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Available: {list(dataset_classes.keys())}"
        )

    dataset_class = dataset_classes[dataset_type]

    dataset = dataset_class(
        data_source=data_source,
        num_frames=num_frames,
        image_size=image_size,
        **kwargs,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and len(dataset) > 0,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle and len(dataset) >= batch_size,
    )

    logger.info(
        f"Created {dataset_type} dataloader with {len(dataset)} samples, batch_size={batch_size}"
    )

    return dataset, loader


@dataclass
class VideoDatasetConfig(DatasetConfig):
    """Configuration for video datasets."""

    dataset_type: str = "video_folder"
    data_source: str = ""
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mkv")
    recursive: bool = True
    obs_key: str = "observations"


def create_video_dataset_from_config(
    config: VideoDatasetConfig,
) -> Tuple[Dataset, DataLoader]:
    """Create video dataset and dataloader from config."""
    return create_video_dataloader(
        dataset_type=config.dataset_type,
        data_source=config.data_source,
        num_frames=config.num_frames,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        pin_memory=config.pin_memory,
        extensions=config.extensions
        if hasattr(config, "extensions")
        else (".mp4", ".avi", ".mkv"),
        recursive=config.recursive if hasattr(config, "recursive") else True,
        obs_key=config.obs_key if hasattr(config, "obs_key") else "observations",
    )
