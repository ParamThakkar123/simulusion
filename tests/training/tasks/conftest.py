"""Shared fixtures for the training-task tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simulusion.data import Episode, EpisodeWriter


def write_episodes(
    root: Path,
    *,
    count: int = 4,
    length: int = 12,
    size: int = 16,
    actions: Any = None,
    seed: int = 0,
) -> Path:
    """A folder of random episodes; ``actions`` is ``None``, an int (discrete
    count) or ``("continuous", dim)``."""
    rng = np.random.default_rng(seed)
    with EpisodeWriter(root) as writer:
        for _ in range(count):
            kwargs: dict[str, Any] = {}
            if isinstance(actions, int):
                kwargs["actions"] = rng.integers(0, actions, length - 1)
            elif actions is not None:
                kwargs["actions"] = rng.uniform(-1, 1, (length - 1, actions[1])).astype(
                    np.float32
                )
            if actions is not None:
                kwargs["rewards"] = rng.normal(size=length - 1).astype(np.float32)
                kwargs["terminated"] = np.r_[np.zeros(length - 2, bool), True]
            writer.write(
                Episode(
                    frames=rng.integers(
                        0, 255, (length, size, size, 3), dtype=np.uint8
                    ),
                    **kwargs,
                )
            )
    return root


@pytest.fixture
def trainer_overrides(tmp_path: Path) -> list[str]:
    return [
        "trainer.device=cpu",
        f"trainer.out_dir={tmp_path.as_posix()}/runs",
        "trainer.run_name=run",
        "trainer.console=false",
        "trainer.log_every=1",
    ]
