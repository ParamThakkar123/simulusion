from typing import Any

import torch
from torchvision.datasets import CIFAR10
from logging import getLogger

logger = getLogger()


def make_cifar10(
    transform: Any,
    batch_size: int,
    collator: Any = None,
    pin_mem: bool = True,
    num_workers: int = 8,
    world_size: int = 1,
    rank: int = 0,
    root_path: str | None = None,
    drop_last: bool = True,
    train: bool = True,
    download: bool = False,
) -> Any:
    """Create CIFAR-10 dataset and distributed dataloader.

    Factory function that creates a CIFAR-10 dataset with the provided transforms
    and returns a tuple of (dataset, dataloader, sampler) for use in JEPA or
    diffusion training pipelines.

    Args:
        transform: Transforms to apply to images (e.g., RandomCrop, ColorJitter).
        batch_size (int): Number of samples per batch.
        collator (callable, optional): Custom collate function for batching
            (e.g., mask collator for JEPA).
        pin_mem (bool): Whether to pin memory for faster GPU transfer (default: True).
        num_workers (int): Number of data loading workers (default: 8).
        world_size (int): Number of distributed processes (default: 1).
        rank (int): Rank of current process in distributed setting (default: 0).
        root_path (str, optional): Path to store/load CIFAR-10 data.
        drop_last (bool): Whether to drop incomplete final batch (default: True).
        train (bool): Whether to load train or test split (default: True).
        download (bool): Whether to download dataset if not present (default: False).

    Returns:
        tuple: (dataset, dataloader, sampler)
            - dataset: torchvision.datasets.CIFAR10 instance
            - dataloader: torch.utils.data.DataLoader with distributed sampling
            - sampler: torch.utils.data.distributed.DistributedSampler

    Example:
        >>> transform = make_transforms(crop_size=224)
        >>> dataset, loader, sampler = make_cifar10(
        ...     transform=transform,
        ...     batch_size=256,
        ...     root_path="./data",
        ...     download=True
        ... )
    """
    dataset = CIFAR10(
        root=root_path,
        train=train,
        download=download,
        transform=transform,
    )
    dist_sampler: torch.utils.data.distributed.DistributedSampler = (
        torch.utils.data.distributed.DistributedSampler(
            dataset=dataset, num_replicas=world_size, rank=rank
        )
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    logger.info("CIFAR10 data loader created")
    return dataset, data_loader, dist_sampler
