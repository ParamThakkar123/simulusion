"""Pieces several tasks share: where offline batches come from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from simulusion.data.episode import Episode
from simulusion.data.folder import DatasetMeta, EpisodeFolder
from simulusion.data.sequence import SequenceDataset, collate_sequences


@dataclass
class SequenceDataConfig:
    """Where an offline task reads its episodes, and how it cuts them.

    Attributes:
        path: An episode folder, or for ``source: lerobot`` a local LeRobot
            dataset folder.
        source: ``"folder"`` (Simulusion episodes, videos, ``.npy``) or
            ``"lerobot"``.
        camera: The view to train on, for datasets with several.
        sequence_length: Frames per training window; the task's natural
            length (its model's context) when ``None``.
        stride: Frames between window starts.
        image_size: Resize frames to this square size; the model's size when
            ``None``.
        batch_size: Windows per batch (per process under data parallelism).
        num_workers: ``DataLoader`` worker processes.
        val_fraction: Fraction of episodes held out for evaluation; 0 for none.
        val_batches: Batches scored per evaluation.
        episodes: Use only these episode indices.
        preload: Decode every episode into memory up front.
        seed: Seed for the train / validation split.
    """

    path: Optional[str] = None
    source: str = "folder"
    camera: Optional[str] = None
    sequence_length: Optional[int] = None
    stride: int = 1
    image_size: Optional[int] = None
    batch_size: int = 16
    num_workers: int = 0
    val_fraction: float = 0.0
    val_batches: int = 8
    episodes: Optional[List[int]] = None
    preload: bool = False
    seed: int = 0


class _Subset(Sequence[Episode]):
    def __init__(self, base: Any, indices: List[int]) -> None:
        self.base, self.indices = base, indices
        self.lengths = [base.lengths[i] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Episode:  # type: ignore[override]
        return self.base[self.indices[i]]


def open_episodes(config: SequenceDataConfig) -> Tuple[Any, Optional[DatasetMeta]]:
    """The episodes ``config`` points at, and their metadata if known."""
    if not config.path:
        raise ValueError(
            "This task trains on a dataset: set data.path to an episode folder "
            "(or a LeRobot dataset with data.source=lerobot)."
        )
    if config.source == "folder":
        folder = EpisodeFolder(config.path, episodes=config.episodes)
        return folder, folder.meta
    if config.source == "lerobot":
        from simulusion.data.lerobot import LeRobotEpisodes

        dataset = LeRobotEpisodes(
            config.path,
            camera=config.camera,
            episodes=config.episodes,
        )
        return dataset, dataset.meta
    raise ValueError(
        f"Unknown data.source '{config.source}'; use 'folder' or 'lerobot'."
    )


def sequence_loaders(
    config: SequenceDataConfig,
    trainer: Any,
    *,
    sequence_length: int,
    image_size: Optional[int | Tuple[int, int]],
    require_actions: bool = False,
) -> Tuple[Iterator[dict], Optional[DataLoader], Optional[DatasetMeta]]:
    """An endless iterator of training batches, a validation loader, and metadata."""
    episodes, meta = open_episodes(config)
    camera = config.camera if config.source == "folder" else None
    train_eps: Any = episodes
    val_eps: Any = None
    if config.val_fraction > 0:
        order = np.random.default_rng(config.seed).permutation(len(episodes))
        n_val = min(
            max(1, int(round(len(episodes) * config.val_fraction))), len(episodes) - 1
        )
        if n_val < 1:
            raise ValueError("val_fraction needs at least two episodes.")
        val_eps = _Subset(episodes, sorted(order[:n_val].tolist()))
        train_eps = _Subset(episodes, sorted(order[n_val:].tolist()))

    def windows(eps: Any) -> SequenceDataset:
        return SequenceDataset(
            eps,
            config.sequence_length or sequence_length,
            stride=config.stride,
            image_size=config.image_size or image_size,
            main_camera=camera,
            require_actions=require_actions,
            preload=config.preload,
        )

    generator = torch.Generator().manual_seed(config.seed)
    train = DataLoader(
        windows(train_eps),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_sequences,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
    if len(train) == 0:
        raise ValueError(
            f"Only {len(train.dataset)} training windows for batch_size="  # type: ignore[arg-type]
            f"{config.batch_size}; lower data.batch_size or add data."
        )
    val = None
    if val_eps is not None:
        val = DataLoader(
            windows(val_eps),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_sequences,
        )
        val = trainer.prepare_dataloader(val)
    return trainer.infinite(trainer.prepare_dataloader(train)), val, meta


#: Arrays a ring replay buffer (Dreamer's, IRIS's) may hold, row-aligned.
_REPLAY_ARRAYS = ("observations", "actions", "rewards", "terminals", "terminated")


def save_replay(buffer: Any, path: Path) -> None:
    """Write the filled part of a ring replay buffer, plus its write position."""
    n = buffer.size if buffer.full else buffer.idx
    arrays = {
        name: getattr(buffer, name)[:n]
        for name in _REPLAY_ARRAYS
        if hasattr(buffer, name)
    }
    state = np.array([buffer.idx, int(buffer.full), buffer.steps, buffer.episodes])
    np.savez(path, state=state, **arrays)


def load_replay(buffer: Any, path: Path) -> None:
    """Restore what :func:`save_replay` wrote into a buffer of the same size."""
    with np.load(path, allow_pickle=False) as data:
        n = data["observations"].shape[0]
        if n > buffer.size:
            raise ValueError(
                f"The saved replay holds {n} transitions; this buffer holds {buffer.size}."
            )
        for name in _REPLAY_ARRAYS:
            if name in data.files and hasattr(buffer, name):
                getattr(buffer, name)[:n] = data[name]
        idx, full, steps, episodes = (int(v) for v in data["state"])
    buffer.idx, buffer.full, buffer.steps, buffer.episodes = (
        idx,
        bool(full),
        steps,
        episodes,
    )


def mean_metrics(rows: List[dict]) -> dict:
    keys = set().union(*rows) if rows else set()
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)}


__all__ = [
    "SequenceDataConfig",
    "load_replay",
    "mean_metrics",
    "open_episodes",
    "save_replay",
    "sequence_loaders",
]
