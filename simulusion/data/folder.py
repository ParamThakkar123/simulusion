"""A folder of episodes: the on-disk dataset every task reads.

.. code-block:: text

    my_dataset/
        meta.json                 # optional: fps, action space, cameras, source
        episode_000000.npz        # one Episode each (see simulusion.data.episode)
        episode_000001.npz
        ...

Folders without ``meta.json`` work too, as do loose videos (``.mp4`` etc.,
read as frame-only episodes) and ``.npy`` frame arrays, so a directory of
clips is already a dataset. Sub-folders are searched recursively.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from simulusion.data.episode import Episode, npz_length
from simulusion.data.video_io import VIDEO_SUFFIXES, read_video

META_NAME = "meta.json"
META_FORMAT = "simulusion-episodes"
META_VERSION = 1


@dataclass
class DatasetMeta:
    """What ``meta.json`` records about a folder of episodes.

    Attributes:
        action_kind: ``"discrete"``, ``"continuous"`` or ``None`` (video).
        action_dim: Number of discrete actions, or the action vector length.
        action_low / action_high: Bounds of continuous actions, if known.
        fps: Frame rate of the recording, if known.
        cameras: Names of the extra cameras (``frames.<name>`` arrays).
        observation_keys: Names of the low-dimensional observations.
        source: Where the data came from (``"lerobot:lerobot/pusht"``).
        info: Anything else worth keeping.
    """

    action_kind: Optional[str] = None
    action_dim: Optional[int] = None
    action_low: Optional[List[float]] = None
    action_high: Optional[List[float]] = None
    fps: Optional[float] = None
    cameras: List[str] = field(default_factory=list)
    observation_keys: List[str] = field(default_factory=list)
    source: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"format": META_FORMAT, "version": META_VERSION, **asdict(self)}

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> "DatasetMeta":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in values.items() if k in known})

    @classmethod
    def infer(cls, episode: Episode, **extra: Any) -> "DatasetMeta":
        """Fill the action space and keys from one episode."""
        meta = cls(
            cameras=sorted(episode.cameras),
            observation_keys=sorted(episode.observations),
            **extra,
        )
        actions = episode.actions
        if actions is not None and meta.action_kind is None:
            if np.issubdtype(actions.dtype, np.integer) and actions.ndim == 1:
                meta.action_kind = "discrete"
                meta.action_dim = int(actions.max()) + 1 if actions.size else 0
            else:
                meta.action_kind = "continuous"
                meta.action_dim = (
                    int(np.prod(actions.shape[1:])) if actions.ndim > 1 else 1
                )
        return meta


class EpisodeFolder(Sequence[Episode]):
    """Read a folder of episodes; ``folder[i]`` loads episode ``i``.

    Args:
        root: The folder.
        episodes: Indices of the episodes to keep (after sorting by path), for
            train/validation splits.
    """

    def __init__(
        self, root: str | Path, *, episodes: Optional[Iterable[int]] = None
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")
        files = sorted(
            p
            for p in (self.root.rglob("*") if self.root.is_dir() else [self.root])
            if p.is_file() and _is_episode_file(p)
        )
        if episodes is not None:
            files = [files[i] for i in episodes]
        if not files:
            raise ValueError(
                f"No episodes (.npz, .npy or video files) under {self.root}."
            )
        self.files: List[Path] = files
        self.meta = read_meta(self.root if self.root.is_dir() else self.root.parent)
        self._lengths: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Episode:  # type: ignore[override]
        return load_episode(self.files[index])

    def __iter__(self) -> Iterator[Episode]:
        for path in self.files:
            yield load_episode(path)

    @property
    def lengths(self) -> List[int]:
        """Frames per episode, read without loading the frames where possible."""
        if self._lengths is None:
            self._lengths = [_episode_length(p) for p in self.files]
        return self._lengths

    def split(
        self, val_fraction: float, seed: int = 0
    ) -> tuple["EpisodeFolder", "EpisodeFolder"]:
        """Disjoint train / validation folders, split by whole episodes."""
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        if len(self) < 2:
            raise ValueError("Splitting needs at least two episodes.")
        order = np.random.default_rng(seed).permutation(len(self))
        n_val = min(max(1, int(round(len(self) * val_fraction))), len(self) - 1)
        val, train = sorted(order[:n_val].tolist()), sorted(order[n_val:].tolist())
        return self._subset(train), self._subset(val)

    def _subset(self, indices: List[int]) -> "EpisodeFolder":
        sub = object.__new__(EpisodeFolder)
        sub.root = self.root
        sub.files = [self.files[i] for i in indices]
        sub.meta = self.meta
        sub._lengths = [self.lengths[i] for i in indices] if self._lengths else None
        return sub


class EpisodeWriter:
    """Append episodes to a folder, numbering after any already there.

    .. code-block:: python

        with EpisodeWriter("data/pusht", meta=DatasetMeta(fps=10)) as writer:
            for episode in episodes:
                writer.write(episode)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        meta: Optional[DatasetMeta] = None,
        compress: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = meta if meta is not None else read_meta(self.root)
        self.compress = compress
        existing = sorted(self.root.glob("episode_*.npz"))
        self._next = (int(existing[-1].stem.split("_")[-1]) + 1) if existing else 0
        self.written = 0

    def write(self, episode: Episode) -> Path:
        if self.meta is None:
            self.meta = DatasetMeta.infer(episode)
        elif self.meta.action_kind is None and episode.has_actions:
            inferred = DatasetMeta.infer(episode)
            self.meta.action_kind, self.meta.action_dim = (
                inferred.action_kind,
                inferred.action_dim,
            )
        if (
            self.meta.action_kind == "discrete"
            and episode.actions is not None
            and episode.actions.size
        ):
            self.meta.action_dim = max(
                self.meta.action_dim or 0, int(episode.actions.max()) + 1
            )
        path = episode.save(
            self.root / f"episode_{self._next:06d}.npz", compress=self.compress
        )
        self._next += 1
        self.written += 1
        return path

    def close(self) -> None:
        if self.meta is not None:
            write_meta(self.root, self.meta)

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_meta(root: str | Path) -> Optional[DatasetMeta]:
    path = Path(root) / META_NAME
    if not path.exists():
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("format") not in (None, META_FORMAT):
        return None
    return DatasetMeta.from_json(values)


def write_meta(root: str | Path, meta: DatasetMeta) -> Path:
    path = Path(root) / META_NAME
    path.write_text(json.dumps(meta.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


def load_episode(path: str | Path) -> Episode:
    """One episode from an ``.npz``, a ``.npy`` frame array or a video file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return Episode.load(path)
    if suffix == ".npy":
        return Episode(frames=np.load(path, allow_pickle=False))
    if suffix in VIDEO_SUFFIXES:
        return Episode(frames=read_video(path))
    raise ValueError(f"Not an episode file: {path}")


def _is_episode_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in {".npz", ".npy"} or suffix in VIDEO_SUFFIXES


def _episode_length(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return npz_length(path)
    if suffix == ".npy":
        return int(np.load(path, mmap_mode="r").shape[0])
    return len(load_episode(path))


__all__ = [
    "DatasetMeta",
    "EpisodeFolder",
    "EpisodeWriter",
    "load_episode",
    "read_meta",
    "write_meta",
]
