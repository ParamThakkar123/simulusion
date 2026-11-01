import os
import subprocess
import time

import numpy as np

from logging import getLogger
from typing import Any, Tuple

import torch
import torchvision
from torch.utils.data import random_split

logger = getLogger()


def make_imagenet1k(
    transform: Any,
    batch_size: int,
    collator: Any = None,
    pin_mem: bool = True,
    num_workers: int = 8,
    world_size: int = 1,
    rank: int = 0,
    root_path: str | None = None,
    image_folder: str | None = None,
    training: bool = True,
    copy_data: bool = False,
    drop_last: bool = True,
    subset_file: str | None = None,
) -> Tuple[
    torch.utils.data.Dataset,
    torch.utils.data.DataLoader,
    torch.utils.data.distributed.DistributedSampler,
]:
    """Build an ImageNet-1K dataset and dataloader with distributed sampling.

    Factory function that creates an ImageNet dataset and returns a tuple of
    (dataset, dataloader, sampler) for use in JEPA or other self-supervised
    training pipelines.

    Supports:
        - Optional data staging from network storage to local scratch
        - Subset filtering via text file listing allowed image IDs
        - Distributed sampling for multi-GPU training

    Args:
        transform: Transforms to apply to images.
        batch_size (int): Number of samples per batch.
        collator (callable, optional): Custom collate function (e.g., mask collator).
        pin_mem (bool): Whether to pin memory for GPU transfer (default: True).
        num_workers (int): Number of data loading workers (default: 8).
        world_size (int): Number of distributed processes (default: 1).
        rank (int): Rank of current process (default: 0).
        root_path (str, optional): Root path containing ImageNet data.
        image_folder (str, optional): Subfolder containing ImageNet data.
        training (bool): Load train or validation split (default: True).
        copy_data (bool): Copy data locally for faster loading (default: False).
        drop_last (bool): Drop incomplete final batch (default: True).
        subset_file (str, optional): Path to file listing allowed image IDs.

    Returns:
        tuple: (dataset, dataloader, sampler)
            - dataset: ImageNet dataset instance
            - dataloader: DataLoader with distributed sampling
            - sampler: DistributedSampler instance
    """
    # Annotate as Any because we may wrap the ImageNet with ImageNetSubset
    dataset: Any = ImageNet(
        root=root_path,  # type: ignore[arg-type]
        image_folder=image_folder,  # type: ignore[arg-type]
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False,
    )
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info("ImageNet dataset created")
    # Explicitly annotate the distributed sampler variable for mypy.
    dist_sampler: torch.utils.data.distributed.DistributedSampler
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        # Keep workers alive across epochs; respawning them re-imports torch and
        # re-opens the dataset in every child on every epoch boundary.
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    logger.info("ImageNet unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class ImageNet(torchvision.datasets.ImageFolder):
    """ImageNet dataset wrapper with optional local copy/extract workflow.

    Extends torchvision.datasets.ImageFolder to support data staging from
    network storage to local scratch space for faster multi-process training
    on cluster environments (e.g., SLURM).

    Features:
        - Optional data copying from network storage to local /scratch
        - Extracts tar archives automatically on first access
        - Supports train/validation splits
        - Optional target indexing for balanced sampling
    """

    def __init__(
        self,
        root: str,
        image_folder: str = "imagenet_full_size/061417/",
        tar_file: str = "imagenet_full_size-061417.tar.gz",
        transform: Any = None,
        train: bool = True,
        job_id: str | None = None,
        local_rank: int | None = None,
        copy_data: bool = True,
        index_targets: bool = False,
    ) -> None:
        """Initialize ImageNet dataset.

        Args:
            root (str): Root network directory for ImageNet data.
            image_folder (str): Path to images inside root (default: "imagenet_full_size/061417/").
            tar_file (str): Name of tar archive to extract (default: "imagenet_full_size-061417.tar.gz").
            transform (callable, optional): Transform to apply to images.
            train (bool): Load train data if True, validation if False (default: True).
            job_id (str, optional): SLURM job ID for local storage path.
            local_rank (int, optional): Local process rank for coordination.
            copy_data (bool): Copy data from network to local scratch (default: True).
            index_targets (bool): Build index of image IDs per class (default: False).
        """
        suffix = "train/" if train else "val/"
        data_path = None
        if copy_data:
            logger.info("copying data locally")
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank,
            )
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f"data-path {data_path}")

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info("Initialized ImageNet")

        if index_targets:
            self.targets: list[int] = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets_arr = np.array(self.targets)
            self.samples_arr = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets_arr == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f"num-labeled target {t} {len(indices)}")
            logger.info(f"min. labeled indices {mint}")


class ImageNetSubset(object):
    """View over an `ImageNet` dataset filtered by an explicit image-id list.

    The subset file contains target image names; only matching samples are
    kept while preserving transforms and label mapping from the base dataset.
    """

    def __init__(self, dataset: Any, subset_file: str) -> None:
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file: str) -> None:
        """Filter self.dataset to a subset"""
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f"Using {subset_file}")
        with open(subset_file, "r") as rfile:
            for line in rfile:
                class_name = line.split("_")[0]
                target = class_to_idx[class_name]
                img = line.split("\n")[0]
                new_samples.append((os.path.join(root, class_name, img), target))
        self.samples = new_samples

    @property
    def classes(self) -> Any:
        return self.dataset.classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def copy_imgnt_locally(
    root: str,
    suffix: str,
    image_folder: str = "imagenet_full_size/061417/",
    tar_file: str = "imagenet_full_size-061417.tar.gz",
    job_id: str | None = None,
    local_rank: int | None = None,
) -> str | None:
    """Copy and extract ImageNet archives to per-job local scratch storage.

    In SLURM environments this reduces network filesystem pressure by unpacking
    once per job and synchronizing worker processes with a signal file.
    """
    if job_id is None:
        try:
            job_id = os.environ["SLURM_JOBID"]
        except Exception:
            logger.info("No job-id, will load directly from network file")
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ["SLURM_LOCALID"])
        except Exception:
            logger.info("No job-id, will load directly from network file")
            return None

    source_file = os.path.join(root, tar_file)
    target = f"/scratch/slurm_tmpdir/{job_id}/"
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f"{source_file}\n{target}\n{target_file}\n{data_path}")

    tmp_sgnl_file = os.path.join(target, "copy_signal.txt")

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [["tar", "-xf", source_file, "-C", target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f"Executing {cmnd}")
                subprocess.run(cmnd)
                logger.info(f"Cmnd took {(time.time() - start_time) / 60.0} min.")
            with open(tmp_sgnl_file, "+w") as f:
                print("Done copying locally.", file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f"{local_rank}: Checking {tmp_sgnl_file}")

    return data_path


def make_imagefolder(
    transform: Any,
    batch_size: int,
    collator: Any = None,
    pin_mem: bool = True,
    num_workers: int = 8,
    world_size: int = 1,
    rank: int = 0,
    root_path: str | None = None,
    image_folder: str | None = None,
    drop_last: bool = True,
    val_split: float | None = None,
    split: str = "train",
    split_seed: int = 0,
) -> Tuple[
    torch.utils.data.Dataset,
    torch.utils.data.DataLoader,
    torch.utils.data.distributed.DistributedSampler,
]:
    """Create an ImageFolder dataset loader for custom folder-structured datasets.

    Supports optional train/validation split and distributed sampling, making
    it a drop-in replacement for ImageNet loaders in training scripts.

    Args:
        val_split: Fraction held out for validation. ``None`` uses everything.
        split: Which side of that split to return, ``"train"`` or ``"val"``.
            Call twice with the same ``val_split`` and ``split_seed`` to get the
            two disjoint halves.
        split_seed: Seed for the partition, so both calls agree on it.
    """
    # Build a string root path for ImageFolder; coerce None -> empty string
    if image_folder:
        root_arg = os.path.join(root_path or "", image_folder)
    else:
        root_arg = root_path or ""

    dataset = torchvision.datasets.ImageFolder(
        root=root_arg,
        transform=transform,
    )
    if val_split:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        # Seed the split so that the "train" and "val" calls partition the same
        # dataset the same way. Without a fixed generator the two calls would
        # draw independent splits and the validation set would overlap the
        # training set, which makes a held-out loss meaningless.
        generator = torch.Generator().manual_seed(split_seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        dataset = val_dataset if split == "val" else train_dataset
    dist_sampler: torch.utils.data.distributed.DistributedSampler
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        # Keep workers alive across epochs; respawning them re-imports torch and
        # re-opens the dataset in every child on every epoch boundary.
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    logger.info("ImageFolder data loader created")
    return dataset, data_loader, dist_sampler
