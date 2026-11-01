"""Shared utilities for interactive play scripts.

Provides:
  - resolve_checkpoint_path() — resolve checkpoint paths with fallback directories
  - KEY_TO_ACTION / get_action_from_key() — Atari keyboard mapping
  - ActionControl — enum for HUMAN vs AGENT control display
"""

from pathlib import Path
from typing import Any, Optional

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
