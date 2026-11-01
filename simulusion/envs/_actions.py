from __future__ import annotations

from typing import Any

import numpy as np


def clip_box_action(action: Any, low: Any, high: Any) -> np.ndarray:
    """Return a finite float32 action clipped to a Box range."""
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)
    action_arr = np.asarray(action, dtype=np.float32)
    try:
        reshaped = action_arr.reshape(low_arr.shape)
    except ValueError as exc:
        raise ValueError(
            f"Expected action with shape {low_arr.shape}, got {action_arr.shape}."
        ) from exc
    if not np.isfinite(reshaped).all():
        raise ValueError("Action must contain only finite values.")
    return np.clip(reshaped, low_arr, high_arr).astype(np.float32, copy=False)


def encode_discrete_action(action: Any, num_actions: int) -> tuple[int, np.ndarray]:
    """Convert scalar or vector actions into a clipped discrete index and one-hot-like vector."""
    n = int(num_actions)
    if n < 1:
        raise ValueError("num_actions must be >= 1")
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Discrete action cannot be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("Discrete action must contain only finite values.")
    if arr.size == n and n > 1:
        index = int(np.argmax(arr))
    else:
        index = int(round(float(arr[0])))
    index = int(np.clip(index, 0, n - 1))
    encoded = -np.ones((n,), dtype=np.float32)
    encoded[index] = 1.0
    return index, encoded
