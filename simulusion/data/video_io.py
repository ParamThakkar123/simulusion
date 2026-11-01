"""Decode and encode video files for the episode format.

PyAV (``pip install simulusion[lerobot]``) is preferred: it decodes AV1, which
LeRobot datasets use, and seeks by timestamp. OpenCV (``simulusion[viz]``)
is the fallback for whole-file reads of common codecs.
"""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mkv", ".webm", ".mov"})


def read_video(
    path: str | Path,
    *,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> np.ndarray:
    """Frames of a video as ``(T, H, W, 3)`` uint8 RGB.

    Args:
        path: The video file.
        start: First timestamp (seconds) to keep, inclusive. LeRobot v3 packs
            many episodes into one file and locates each by timestamps.
        end: Timestamp (seconds) to stop before, exclusive.
    """
    if importlib.util.find_spec("av") is not None:
        return _read_av(Path(path), start, end)
    if start is not None or end is not None:
        raise ImportError(
            "Reading part of a video needs PyAV: pip install simulusion[lerobot]"
        )
    if importlib.util.find_spec("cv2") is not None:
        return _read_cv2(Path(path))
    raise ImportError(
        f"Decoding {Path(path).name} needs PyAV (pip install simulusion[lerobot]) "
        "or OpenCV (pip install simulusion[viz])."
    )


def iter_video(path: str | Path) -> Iterator[np.ndarray]:
    """Yield a video's frames one at a time as ``(H, W, 3)`` uint8 RGB, for
    footage too long to hold in memory at full resolution."""
    path = Path(path)
    if importlib.util.find_spec("av") is not None:
        import av

        with av.open(str(path)) as container:
            for decoded in container.decode(container.streams.video[0]):
                yield decoded.to_ndarray(format="rgb24")
        return
    if importlib.util.find_spec("cv2") is not None:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise FileNotFoundError(f"Could not open video: {path}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        return
    raise ImportError(
        f"Decoding {path.name} needs PyAV (pip install simulusion[lerobot]) "
        "or OpenCV (pip install simulusion[viz])."
    )


def video_fps(path: str | Path) -> Optional[float]:
    """A video's average frame rate, or ``None`` when it cannot be read."""
    if importlib.util.find_spec("av") is not None:
        import av

        with av.open(str(path)) as container:
            rate = container.streams.video[0].average_rate
            return float(rate) if rate else None
    if importlib.util.find_spec("cv2") is not None:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        return fps or None
    return None


def write_video(path: str | Path, frames: np.ndarray, fps: float = 30.0) -> Path:
    """Encode ``(T, H, W, 3)`` uint8 frames as an H.264 (or MPEG-4) file with PyAV."""
    import av

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    with av.open(str(out), mode="w") as container:
        codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
        stream: Any = container.add_stream(
            codec, rate=Fraction(fps).limit_denominator(1000)
        )
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            stream.options = {"crf": "0", "preset": "ultrafast"}
        for frame in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(frame, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out


def _read_av(path: Path, start: Optional[float], end: Optional[float]) -> np.ndarray:
    import av

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start:
            # Seek lands on the keyframe at or before `start`; frames before it
            # are decoded and dropped below.
            offset = int(start / float(stream.time_base)) if stream.time_base else 0
            container.seek(offset, stream=stream, backward=True, any_frame=False)
        # Half a frame of slack: container timestamps are rounded.
        rate = float(stream.average_rate or 30)
        slack = 0.5 / rate
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if t is not None and start is not None and t < start - slack:
                continue
            if t is not None and end is not None and t >= end - slack:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No frames decoded from {path} in [{start}, {end}).")
    return np.stack(frames)


def _read_cv2(path: Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Could not open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}.")
    return np.stack(frames)


__all__ = ["VIDEO_SUFFIXES", "iter_video", "read_video", "video_fps", "write_video"]
