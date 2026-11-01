from torch.utils.data import DataLoader, Dataset
from typing import Iterator, Optional
import multiprocessing


def create_efficient_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    """Create a memory-efficient and fast DataLoader."""
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True,
    )


def prefetch_iterator(iterator: Iterator, buffer_size: int = 3) -> Iterator:
    """Add prefetching to any iterator."""
    from collections import deque

    buffer: deque = deque()

    def fill_buffer() -> None:
        for item in iterator:
            buffer.append(item)
            if len(buffer) >= buffer_size:
                break

    fill_buffer()

    while buffer:
        yield buffer.popleft()
        # Prefetch next item
        try:
            next_item = next(iterator)
            buffer.append(next_item)
        except StopIteration:
            pass
