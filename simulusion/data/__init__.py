"""Datasets for world models: one episode format for video, RL and robot data.

* :class:`Episode` - one trajectory: frames, and optionally the actions,
  rewards and episode ends between them, extra cameras, robot state and a
  language instruction. Saved as one ``.npz`` file.
* :class:`EpisodeFolder` / :class:`EpisodeWriter` - a directory of episodes,
  the on-disk dataset every training task reads.
* :class:`SequenceDataset` - fixed-length windows over episodes for a
  ``DataLoader``.
* :class:`LeRobotEpisodes` - local LeRobot datasets (v2.x and v3.0).
* :func:`episode_from_rlds` - RLDS episodes (the Open X-Embodiment format)
  given as Python / NumPy values.
"""

from typing import Any

from simulusion.data.episode import Episode, as_uint8_frames
from simulusion.data.folder import (
    DatasetMeta,
    EpisodeFolder,
    EpisodeWriter,
    load_episode,
    read_meta,
    write_meta,
)
from simulusion.data.sequence import (
    SequenceDataset,
    collate_sequences,
    frames_to_float,
    one_hot_actions,
)

_LAZY = {
    "LeRobotEpisodes": "simulusion.data.lerobot",
    "import_lerobot": "simulusion.data.lerobot",
    "RLDSKeys": "simulusion.data.rlds",
    "episode_from_rlds": "simulusion.data.rlds",
    "read_video": "simulusion.data.video_io",
    "write_video": "simulusion.data.video_io",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetMeta",
    "Episode",
    "EpisodeFolder",
    "EpisodeWriter",
    "LeRobotEpisodes",
    "RLDSKeys",
    "SequenceDataset",
    "as_uint8_frames",
    "collate_sequences",
    "episode_from_rlds",
    "frames_to_float",
    "import_lerobot",
    "load_episode",
    "one_hot_actions",
    "read_meta",
    "read_video",
    "write_meta",
    "write_video",
]
