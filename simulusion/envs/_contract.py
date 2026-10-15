from __future__ import annotations

from typing import Any

import numpy as np


def finalize_step_info(
    info: Any,
    *,
    done: bool,
    terminated: bool | None = None,
    truncated: bool | None = None,
    discount: Any | None = None,
) -> dict[str, Any]:
    normalized: dict[str, Any]
    if info is None:
        normalized = {}
    else:
        normalized = dict(info)

    if truncated is None:
        truncated = bool(
            normalized.get("truncated", normalized.get("TimeLimit.truncated", False))
        )
    else:
        truncated = bool(truncated)

    if terminated is None:
        terminated = bool(normalized.get("terminated", bool(done and not truncated)))
    else:
        terminated = bool(terminated)

    normalized["terminated"] = terminated
    normalized["truncated"] = truncated

    if discount is None:
        discount = normalized.get(
            "discount", np.array(0.0 if done else 1.0, dtype=np.float32)
        )
    normalized["discount"] = np.asarray(discount, dtype=np.float32)
    return normalized


__all__ = ["finalize_step_info"]
