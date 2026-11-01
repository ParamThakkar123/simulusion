"""Measure inference speed *and* what an optimisation costs in fidelity.

Every inference optimisation trades something. Compilation and CUDA graphs are
exact up to kernel-level float reordering; bf16, quantisation and fewer
sampling steps are not. For a world model a tiny per-step error is not the whole
story, because the state is fed back into itself and errors compound over a
rollout. So this module reports two things:

* :func:`benchmark_step` - latency percentiles, throughput and peak memory of a
  step callable, with proper warm-up and CUDA synchronisation;
* :func:`rollout_drift` - how far a candidate step (compiled, bf16, quantised,
  exported, ...) drifts from a reference step over an N-step closed-loop rollout.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch
from torch.utils._pytree import tree_flatten


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = (len(sorted_values) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


@dataclass
class LatencyReport:
    """Timing summary for one benchmarked callable. Times are milliseconds."""

    name: str
    iterations: int
    batch_size: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    steps_per_sec: float
    items_per_sec: float
    peak_memory_mb: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        mem = (
            f", peak {self.peak_memory_mb:.1f} MB"
            if self.peak_memory_mb is not None
            else ""
        )
        return (
            f"{self.name}: p50 {self.p50_ms:.3f} ms, p99 {self.p99_ms:.3f} ms, "
            f"{self.steps_per_sec:.1f} steps/s, {self.items_per_sec:.1f} items/s "
            f"(batch {self.batch_size}){mem}"
        )


def benchmark_step(
    step: Callable[..., Any],
    *args: Any,
    warmup: int = 10,
    iterations: int = 100,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    name: str = "step",
    **kwargs: Any,
) -> LatencyReport:
    """Time ``step(*args, **kwargs)``.

    Each iteration is timed individually (with a device synchronisation on CUDA,
    because kernel launches are asynchronous) so tail latency is visible, which
    matters for acting in a real-time environment. ``warmup`` iterations run
    first and are discarded: they absorb ``torch.compile`` compilation,
    CUDA-graph capture and cuDNN autotuning.

    Args:
        step: Callable to time. For a stateful loop, pass a closure that
            advances its own state.
        warmup: Untimed iterations.
        iterations: Timed iterations.
        batch_size: Items processed per call, for ``items_per_sec``.
        device: Device to synchronise and read peak memory on. Inferred from
            the first tensor argument when omitted.
        name: Label for the report.
    """
    if device is None:
        tensors = [
            v for v in tree_flatten((args, kwargs))[0] if isinstance(v, torch.Tensor)
        ]
        device = tensors[0].device if tensors else torch.device("cpu")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    with torch.inference_mode():
        for _ in range(warmup):
            step(*args, **kwargs)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            step(*args, **kwargs)
            _sync(device)
            times.append((time.perf_counter() - start) * 1000.0)

    peak = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None
    )
    ordered = sorted(times)
    mean = sum(times) / len(times)
    return LatencyReport(
        name=name,
        iterations=iterations,
        batch_size=batch_size,
        mean_ms=mean,
        p50_ms=_percentile(ordered, 0.5),
        p90_ms=_percentile(ordered, 0.9),
        p99_ms=_percentile(ordered, 0.99),
        min_ms=ordered[0],
        steps_per_sec=1000.0 / mean if mean > 0 else float("inf"),
        items_per_sec=batch_size * 1000.0 / mean if mean > 0 else float("inf"),
        peak_memory_mb=peak,
    )


@dataclass
class DriftReport:
    """Per-step divergence of a candidate rollout from a reference rollout.

    ``max_abs_error[t]`` is the largest absolute difference over every tensor
    in the state after step ``t``; ``rel_error[t]`` is the L2 norm of the
    difference divided by the L2 norm of the reference.
    """

    steps: int
    max_abs_error: list[float] = field(default_factory=list)
    rel_error: list[float] = field(default_factory=list)

    @property
    def final_max_abs_error(self) -> float:
        return self.max_abs_error[-1] if self.max_abs_error else 0.0

    @property
    def worst_rel_error(self) -> float:
        return max(self.rel_error) if self.rel_error else 0.0

    def within(self, *, atol: float = math.inf, rtol: float = math.inf) -> bool:
        """True if every step stays within ``atol`` absolute and ``rtol`` relative error."""
        return all(e <= atol for e in self.max_abs_error) and all(
            e <= rtol for e in self.rel_error
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "max_abs_error": self.max_abs_error,
            "rel_error": self.rel_error,
            "final_max_abs_error": self.final_max_abs_error,
            "worst_rel_error": self.worst_rel_error,
        }

    def summary(self) -> str:
        return (
            f"drift over {self.steps} steps: final max|err| "
            f"{self.final_max_abs_error:.3e}, worst rel err {self.worst_rel_error:.3e}"
        )


def _state_tensors(state: Any) -> list[torch.Tensor]:
    return [
        v.detach().to(torch.float64)
        for v in tree_flatten(state)[0]
        if isinstance(v, torch.Tensor) and v.is_floating_point()
    ]


def rollout_drift(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    initial_state: Any,
    step_inputs: Sequence[Sequence[Any]],
) -> DriftReport:
    """Run both steps closed-loop from ``initial_state`` and compare states.

    Each step is called as ``step(state, *step_inputs[t])`` and must return the
    next state (any pytree of tensors, e.g. a tuple or dict). Each rollout feeds
    back its *own* output, so the report shows compounding error, not only
    one-step error.

    Stochastic models must be given their randomness through ``step_inputs``
    (for example the ``noise`` argument of
    :class:`~simulusion.inference.steppers.DreamerStepModule`): compiled and
    exported graphs do not consume the global RNG the way eager code does, so
    seeding alone does not make the two rollouts comparable.
    """
    report = DriftReport(steps=len(step_inputs))
    ref_state, cand_state = initial_state, initial_state
    with torch.inference_mode():
        for inputs in step_inputs:
            ref_state = reference(ref_state, *inputs)
            cand_state = candidate(cand_state, *inputs)
            ref_t, cand_t = _state_tensors(ref_state), _state_tensors(cand_state)
            if len(ref_t) != len(cand_t):
                raise ValueError(
                    "Reference and candidate states have different structures."
                )
            max_abs, num, den = 0.0, 0.0, 0.0
            for r, c in zip(ref_t, cand_t):
                diff = (r - c.to(r.device)).abs()
                if diff.numel():
                    max_abs = max(max_abs, float(diff.max()))
                num += float((diff**2).sum())
                den += float((r**2).sum())
            report.max_abs_error.append(max_abs)
            report.rel_error.append(math.sqrt(num) / max(math.sqrt(den), 1e-12))
    return report


__all__ = ["DriftReport", "LatencyReport", "benchmark_step", "rollout_drift"]
