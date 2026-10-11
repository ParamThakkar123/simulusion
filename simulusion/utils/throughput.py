"""Measure training/inference throughput.

Efficiency work is only meaningful if it is measured. This module provides a
small meter that reports steps per second and, optionally, how many bytes each
step ships to the accelerator - the two numbers that move when the transfer
path, precision, or kernel-launch overhead changes.

Usage::

    meter = ThroughputMeter(device=agent.device)
    for _ in range(steps):
        batch = sample()
        meter.record_transfer(batch)      # optional
        agent.train_one_batch()
        meter.step()
    print(meter.summary())
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List

import torch


def tensor_nbytes(*tensors: Any) -> int:
    """Total bytes backing ``tensors``, skipping anything that is not a tensor.

    ``element_size() * nelement()`` rather than ``untyped_storage().nbytes()``:
    a sliced view would otherwise be charged for the whole base allocation.
    """
    total = 0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            total += tensor.element_size() * tensor.nelement()
        elif isinstance(tensor, (list, tuple)):
            total += tensor_nbytes(*tensor)
        elif isinstance(tensor, dict):
            total += tensor_nbytes(*tensor.values())
    return total


class ThroughputMeter:
    """Steps per second and bytes-to-device, over a sliding window.

    Args:
        device: Device the work runs on. When it is CUDA the meter
            synchronises before reading the clock, because kernel launches are
            asynchronous and an unsynchronised timer measures queueing rather
            than execution.
        window: Number of recent steps the windowed rate is computed over.
    """

    def __init__(
        self, device: torch.device | str | None = None, window: int = 50
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        self._sync = self.device is not None and self.device.type == "cuda"
        self._durations: Deque[float] = deque(maxlen=window)
        self._bytes: Deque[int] = deque(maxlen=window)
        self._pending_bytes = 0
        self.total_steps = 0
        self.total_bytes = 0
        self._started = time.perf_counter()
        self._last = self._started

    def _now(self) -> float:
        if self._sync:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def record_transfer(self, *tensors: Any) -> None:
        """Attribute ``tensors``' bytes to the step currently in flight."""
        self._pending_bytes += tensor_nbytes(*tensors)

    def step(self) -> None:
        """Close out one step."""
        now = self._now()
        self._durations.append(now - self._last)
        self._bytes.append(self._pending_bytes)
        self.total_bytes += self._pending_bytes
        self.total_steps += 1
        self._pending_bytes = 0
        self._last = now

    def reset(self) -> None:
        """Discard history, e.g. after warmup and compilation have settled."""
        self._durations.clear()
        self._bytes.clear()
        self._pending_bytes = 0
        self.total_steps = 0
        self.total_bytes = 0
        self._started = self._last = self._now()

    @property
    def stats(self) -> Dict[str, float]:
        windowed = sum(self._durations)
        elapsed = max(self._last - self._started, 1e-12)
        return {
            "steps": float(self.total_steps),
            "elapsed_s": elapsed,
            "steps_per_s": self.total_steps / elapsed,
            "recent_steps_per_s": (
                len(self._durations) / windowed if windowed > 0 else 0.0
            ),
            "ms_per_step": (
                1000.0 * windowed / len(self._durations) if self._durations else 0.0
            ),
            "mib_to_device_per_step": (
                (sum(self._bytes) / len(self._bytes)) / (1024**2)
                if self._bytes
                else 0.0
            ),
            "total_gib_to_device": self.total_bytes / (1024**3),
        }

    def summary(self) -> str:
        """One-line human-readable report."""
        s = self.stats
        return (
            f"{s['steps']:.0f} steps in {s['elapsed_s']:.1f}s | "
            f"{s['recent_steps_per_s']:.2f} steps/s "
            f"({s['ms_per_step']:.1f} ms/step) | "
            f"{s['mib_to_device_per_step']:.1f} MiB/step to device "
            f"({s['total_gib_to_device']:.2f} GiB total)"
        )


def measure_steps(
    step_fn: Any, iterations: int, warmup: int = 3, device: Any = None
) -> Dict[str, float]:
    """Time ``step_fn`` over ``iterations``, discarding ``warmup`` steps first.

    Warmup matters: the first calls pay for lazy CUDA context creation, cuDNN
    autotuning, and ``torch.compile`` tracing, none of which recur.
    """
    meter = ThroughputMeter(device=device)
    for _ in range(warmup):
        step_fn()
    meter.reset()
    for _ in range(iterations):
        step_fn()
        meter.step()
    return meter.stats


__all__: List[str] = ["ThroughputMeter", "measure_steps", "tensor_nbytes"]
