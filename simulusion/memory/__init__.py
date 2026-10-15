"""
Memory sub-module - Experience replay and episode memory systems.

Exported Components:
    - ReplayBuffer: Experience replay buffer for Dreamer
    - Episode: Episode storage used by Planet
    - Memory: Base memory class (deque-based)
    - IRISReplayBuffer: IRIS replay buffer
    - IRISOnPolicyBuffer: IRIS on-policy buffer
"""

from typing import Any

__all__ = [
    "ReplayBuffer",
    "Episode",
    "Memory",
    "IRISReplayBuffer",
    "IRISOnPolicyBuffer",
]


def __getattr__(name: str) -> Any:
    if name == "ReplayBuffer":
        from .dreamer_memory import ReplayBuffer

        return ReplayBuffer
    if name == "Episode":
        from .planet_memory import Episode

        return Episode
    if name == "Memory":
        from .planet_memory import Memory

        return Memory
    if name == "IRISReplayBuffer":
        from .iris_memory import IRISReplayBuffer

        return IRISReplayBuffer
    if name == "IRISOnPolicyBuffer":
        from .iris_memory import IRISOnPolicyBuffer

        return IRISOnPolicyBuffer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
