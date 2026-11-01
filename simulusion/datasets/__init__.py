__all__ = [
    "VideoDatasetBase",
    "VideoFolderDataset",
    "ImageFolderDataset",
    "NumPyDataset",
    "RLEnvironmentDataset",
    "HDF5Dataset",
    "DatasetConfig",
    "VideoDatasetConfig",
    "create_video_dataloader",
    "create_video_dataset_from_config",
    "TinyWorldsDataset",
    "TinyWorldsDataLoader",
    "TinyWorldsConfig",
    "create_tinyworlds_dataloader",
    "download_all_datasets",
]


from typing import Any


def __getattr__(name: str) -> Any:
    if name == "VideoDatasetBase":
        from .video_datasets import VideoDatasetBase

        return VideoDatasetBase
    if name == "VideoFolderDataset":
        from .video_datasets import VideoFolderDataset

        return VideoFolderDataset
    if name == "ImageFolderDataset":
        from .video_datasets import ImageFolderDataset

        return ImageFolderDataset
    if name == "NumPyDataset":
        from .video_datasets import NumPyDataset

        return NumPyDataset
    if name == "RLEnvironmentDataset":
        from .video_datasets import RLEnvironmentDataset

        return RLEnvironmentDataset
    if name == "HDF5Dataset":
        from .video_datasets import HDF5Dataset

        return HDF5Dataset
    if name == "DatasetConfig":
        from .video_datasets import DatasetConfig

        return DatasetConfig
    if name == "VideoDatasetConfig":
        from .video_datasets import VideoDatasetConfig

        return VideoDatasetConfig
    if name == "create_video_dataloader":
        from .video_datasets import create_video_dataloader

        return create_video_dataloader
    if name == "create_video_dataset_from_config":
        from .video_datasets import create_video_dataset_from_config

        return create_video_dataset_from_config
    if name == "TinyWorldsDataset":
        from .tinyworlds import TinyWorldsDataset

        return TinyWorldsDataset
    if name == "TinyWorldsDataLoader":
        from .tinyworlds import TinyWorldsDataLoader

        return TinyWorldsDataLoader
    if name == "TinyWorldsConfig":
        from .tinyworlds import TinyWorldsConfig

        return TinyWorldsConfig
    if name == "create_tinyworlds_dataloader":
        from .tinyworlds import create_tinyworlds_dataloader

        return create_tinyworlds_dataloader
    if name == "download_all_datasets":
        from .tinyworlds import download_all_datasets

        return download_all_datasets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
