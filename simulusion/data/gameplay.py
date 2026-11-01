"""Captured gameplay video as episodes: games without an emulator (Terraria).

PC games such as Terraria cannot be run inside a training loop, so their
data is footage: a screen capture (OBS, ShadowPlay, ...) and, for an
action-conditioned model, a log of the inputs held on every frame.
:func:`import_gameplay` turns a folder of such captures into an episode
folder, decoding one frame at a time so hour-long 1080p captures fit in
memory, and cutting them into clips.

.. code-block:: text

    captures/
        session_01.mp4
        session_01.inputs.csv     # optional: one row per video frame
        session_02.mp4
        session_02.inputs.csv

An input log sits next to its video, named ``<video stem>.inputs.csv`` (or
``.inputs.npy``, a ``(frames, inputs)`` array). The CSV has a header naming
its columns, one row per video frame in order, and numeric values: ``0 / 1``
for keys and buttons, positions or deltas for the mouse. A ``frame`` column,
if present, is only checked to be increasing. Row ``t`` is what was held on
frame ``t``, so it becomes ``actions[t]``.

.. code-block:: python

    from simulusion.data import import_gameplay

    import_gameplay("captures/", "data/terraria", game="terraria", clip_length=500)

or ``simulusion datasets import video captures/ data/terraria --game terraria``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from simulusion.data.episode import Episode
from simulusion.data.folder import DatasetMeta, EpisodeWriter
from simulusion.data.games import get_game
from simulusion.data.preprocess import FramePipeline
from simulusion.data.video_io import VIDEO_SUFFIXES, iter_video, video_fps

logger = logging.getLogger(__name__)

INPUT_SUFFIXES = (".inputs.csv", ".inputs.npy")


def import_gameplay(
    source: str | Path,
    out_dir: str | Path,
    *,
    game: Optional[str] = None,
    pipeline: Optional[FramePipeline] = None,
    clip_length: Optional[int] = None,
    inputs: bool = True,
) -> Path:
    """Convert gameplay videos (and their input logs) into an episode folder.

    Args:
        source: A video file, or a folder searched recursively for videos.
        out_dir: Episode folder to write (appended to if it exists).
        game: A preset (``"terraria"``) whose pipeline and frame rate to use.
        pipeline: Preprocessing; overrides the preset's.
        clip_length: Cut each video into episodes of at most this many
            frames (counted after frame skip); a whole video per episode
            when ``None``. A clip shorter than two frames is dropped.
        inputs: Read input logs next to the videos. Either every video has
            one or none does.

    Returns:
        ``out_dir``.
    """
    preset = get_game(game) if game else None
    if preset is not None and preset.source != "video":
        raise ValueError(
            f"'{game}' is an emulated game; record it with "
            "simulusion.data.record_retro, or replay movies with import_retro_movies."
        )
    pipeline = pipeline or (preset.pipeline if preset else FramePipeline())
    if clip_length is not None and clip_length < 2:
        raise ValueError("clip_length must be at least 2 frames.")
    videos = _video_paths(source)
    logs = {v: _input_log(v) for v in videos} if inputs else {v: None for v in videos}
    with_logs = [v for v, log in logs.items() if log is not None]
    if with_logs and len(with_logs) != len(videos):
        missing = [v.name for v, log in logs.items() if log is None]
        raise ValueError(
            "Some videos have an input log and some do not; an action-conditioned "
            f"dataset needs one for each. Missing: {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
            + ". Pass inputs=False (--no-inputs) to import frames only."
        )

    columns: Optional[List[str]] = None
    native_fps = video_fps(videos[0]) or (preset.fps if preset else None)
    meta = DatasetMeta(
        fps=native_fps / pipeline.frame_skip if native_fps else None,
        source=f"gameplay:{source}",
        info={
            "preset": preset.name if preset else None,
            "pipeline": pipeline.to_json(),
        },
    )
    with EpisodeWriter(out_dir, meta=meta) as writer:
        for video in videos:
            log = logs[video]
            actions = None
            if log is not None:
                names, actions = log
                if columns is None:
                    columns = names
                    meta.action_kind = "continuous"
                    meta.action_dim = len(names)
                    meta.info["inputs"] = names
                elif names != columns:
                    raise ValueError(
                        f"{video.name}'s input log has columns {names}; "
                        f"earlier logs have {columns}."
                    )
            clips = 0
            for episode in _clips(video, actions, pipeline, clip_length):
                writer.write(episode)
                clips += 1
            logger.info("%s: %d episode(s)", video.name, clips)
    return Path(out_dir)


def _clips(
    video: Path,
    inputs: Optional[np.ndarray],
    pipeline: FramePipeline,
    clip_length: Optional[int],
) -> Iterator[Episode]:
    """Stream ``video`` and yield episodes of at most ``clip_length`` frames."""
    skip = pipeline.frame_skip
    frames: List[np.ndarray] = []
    kept: List[int] = []  # raw frame index of each kept frame
    total = 0

    def episode() -> Optional[Episode]:
        if len(frames) < 2:
            return None
        actions = None
        if inputs is not None:
            actions = np.stack(
                [
                    _reduce(inputs[r : r + skip], pipeline.action_reduce)
                    for r in kept[:-1]
                ]
            )
        return Episode(frames=np.stack(frames), actions=actions)

    for index, frame in enumerate(iter_video(video)):
        total = index + 1
        if inputs is not None and index >= len(inputs):
            break
        if index % skip:
            continue
        frames.append(pipeline.frame(frame))
        kept.append(index)
        if clip_length is not None and len(frames) == clip_length:
            done = episode()
            if done is not None:
                yield done
            frames.clear()
            kept.clear()
    if inputs is not None and total < len(inputs):
        logger.warning(
            "%s: %d input rows for %d frames; the extra rows are ignored.",
            video.name,
            len(inputs),
            total,
        )
    elif inputs is not None and total > len(inputs):
        logger.warning(
            "%s: only %d input rows; frames after them are dropped.",
            video.name,
            len(inputs),
        )
    done = episode()
    if done is not None:
        yield done


def _reduce(rows: np.ndarray, how: str) -> np.ndarray:
    if how == "first" or len(rows) == 1:
        return rows[0]
    if how == "any":
        return (rows != 0).any(axis=0).astype(np.float32)
    return rows.mean(axis=0).astype(np.float32)


def _input_log(video: Path) -> Optional[Tuple[List[str], np.ndarray]]:
    """A video's input log as (column names, ``(frames, inputs)`` float32)."""
    for suffix in INPUT_SUFFIXES:
        path = video.with_name(video.stem + suffix)
        if not path.exists():
            continue
        if suffix.endswith(".npy"):
            values = np.load(path, allow_pickle=False).astype(np.float32)
            if values.ndim == 1:
                values = values[:, None]
            return [f"input_{i}" for i in range(values.shape[1])], values
        return _read_csv(path)
    return None


def _read_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path} needs a header row and at least one frame.")
    header = [name.strip() for name in rows[0]]
    try:
        values = np.array(
            [[float(v) for v in row] for row in rows[1:] if row], dtype=np.float32
        )
    except ValueError as exc:
        raise ValueError(f"{path} has a non-numeric value: {exc}") from None
    if values.shape[1] != len(header):
        raise ValueError(
            f"{path}: {values.shape[1]} values per row, {len(header)} names."
        )
    if "frame" in header:
        column = header.index("frame")
        if np.any(np.diff(values[:, column]) <= 0):
            raise ValueError(f"{path}: the frame column must increase row by row.")
        values = np.delete(values, column, axis=1)
        header.pop(column)
    if not header:
        raise ValueError(f"{path} has no input columns besides 'frame'.")
    return header, values


def _video_paths(source: str | Path) -> List[Path]:
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"No such video or folder: {root}")
    paths = [root] if root.is_file() else list(root.rglob("*"))
    found = sorted(
        p for p in paths if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not found:
        raise ValueError(
            f"No videos ({', '.join(sorted(VIDEO_SUFFIXES))}) under {root}."
        )
    return found


__all__ = ["INPUT_SUFFIXES", "import_gameplay"]
