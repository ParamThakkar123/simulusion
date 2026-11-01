"""Shared utilities for interactive play scripts.

Provides:
  - resolve_checkpoint_path() — resolve checkpoint paths with fallback directories
  - KEY_TO_ACTION / get_action_from_key() — Atari keyboard mapping
  - ActionControl — enum for HUMAN vs AGENT control display
  - DreamBuffer — the last N dream steps, saved on demand
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# OpenCV arrow key codes (platform-independent via unmasked waitKey)
_ARROW_UP = 0x26
_ARROW_DOWN = 0x28
_ARROW_LEFT = 0x25
_ARROW_RIGHT = 0x27

KEY_TO_ACTION: dict[int, int] = {
    ord("z"): 0,  # NOOP
    ord("x"): 1,  # FIRE
    ord(" "): 1,  # SPACE = FIRE
    ord("w"): 2,  # UP
    ord("s"): 5,  # DOWN
    ord("a"): 4,  # LEFT
    ord("d"): 3,  # RIGHT
    # Arrow keys
    _ARROW_UP: 2,
    _ARROW_DOWN: 5,
    _ARROW_LEFT: 4,
    _ARROW_RIGHT: 3,
}


def get_action_from_key(key: int) -> Optional[int]:
    """Map a keyboard code to an Atari action, or None if unmapped.

    Handles both masked (0xFF) and unmasked OpenCV key codes.
    """
    if key == -1:
        return None
    action = KEY_TO_ACTION.get(key)
    if action is not None:
        return action
    masked = key & 0xFF
    if masked != key:
        return KEY_TO_ACTION.get(masked)
    return None


def resolve_checkpoint_path(path: str, model_dir: str = "checkpoints/diamond") -> str:
    """Resolve a checkpoint path, searching common locations.

    Args:
        path: User-provided checkpoint path (bare name or full path).
        model_dir: Fallback directory to search, e.g. ``checkpoints/diamond``.

    Returns:
        Resolved absolute path to the checkpoint file.

    Raises:
        FileNotFoundError: if the path cannot be resolved.
    """
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    alt = Path(model_dir) / p
    if alt.exists():
        return str(alt.resolve())
    raise FileNotFoundError(f"Checkpoint not found at {path} or {alt}")


def init_video_recorder(
    record_path: Optional[str], fps: int = 20, frame_shape: Any = None
) -> Any:
    """Create a StreamingVideoWriter if *record_path* is provided.

    Returns the writer or None.
    """
    if record_path is None:
        return None
    from simulusion.utils.utils import StreamingVideoWriter

    return StreamingVideoWriter(record_path, fps=fps, frame_shape=frame_shape)


class DreamBuffer:
    """The last ``size`` dream steps: frame, action and predicted reward.

    Frame ``k`` is the one the world model imagined after ``actions[k]``, and
    ``rewards[k]`` is the reward it predicted for that action.
    """

    def __init__(self, size: int) -> None:
        self._steps: deque[tuple[np.ndarray, int, float]] = deque(maxlen=size)

    def __len__(self) -> int:
        return len(self._steps)

    def add(self, frame: np.ndarray, action: int, reward: float) -> None:
        """Append one step; ``frame`` is uint8 RGB."""
        self._steps.append((frame, int(action), float(reward)))

    def clear(self) -> None:
        self._steps.clear()

    def save(
        self, directory: str | Path = ".", fps: int = 20, **info: Any
    ) -> tuple[Path, Path]:
        """Write ``dream_<timestamp>.mp4`` and ``.json``; return both paths."""
        from simulusion.utils.utils import StreamingVideoWriter

        if not self._steps:
            raise ValueError("no dream steps to save")

        stem = "dream_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        video = Path(directory) / f"{stem}.mp4"
        data = video.with_suffix(".json")

        writer = StreamingVideoWriter(str(video), fps=fps)
        for frame, _, _ in self._steps:
            writer.write_frame(frame)
        writer.close()

        data.write_text(
            json.dumps(
                {
                    "video": video.name,
                    **info,
                    "fps": fps,
                    "actions": [action for _, action, _ in self._steps],
                    "rewards": [reward for _, _, reward in self._steps],
                },
                indent=2,
            )
        )
        return video, data
