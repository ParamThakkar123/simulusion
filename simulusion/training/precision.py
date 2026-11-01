"""Mixed precision and optimizer steps shared by every trainer.

Each family used to carry its own ``use_amp`` flag, a ``GradScaler("cuda")``
per optimizer and the same six lines around ``backward()``. They live here
once, so a run's precision is one setting rather than a flag per model:

.. code-block:: python

    precision = MixedPrecision("bf16", device)
    update = GradientUpdate(optimizer, model.parameters(), precision, max_grad_norm=1.0)

    with precision.autocast():
        loss = model(batch)
    grad_norm = update(loss)

``GradientUpdate`` keeps a loss scaler per optimizer. Algorithms with several
optimizers (a world model, an actor and a critic) step them one after another,
and a scaler shared between them would let an overflow in one skip the others.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn as nn

from simulusion.inference.precision import resolve_precision

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class MixedPrecision:
    """Autocast dtype and loss scaling for one training run.

    Args:
        precision: ``"fp32"``, ``"bf16"``, ``"fp16"`` or ``"auto"`` (bf16 on
            CUDA devices that support it, fp16 on older ones, fp32 elsewhere).
        device: The device the run trains on.

    fp32 enters no autocast region, so it is bit-identical to running without
    one. Only fp16 scales the loss: bf16 has fp32's dynamic range.
    """

    def __init__(
        self,
        precision: str | None,
        device: torch.device | str,
        *,
        scale_loss: Optional[bool] = None,
    ) -> None:
        self.device = torch.device(device)
        self.name = resolve_precision(precision, self.device)
        self.dtype = _DTYPES[self.name]
        self._scale_loss = self.name == "fp16" if scale_loss is None else scale_loss

    @classmethod
    def from_use_amp(
        cls, use_amp: bool, device: torch.device | str
    ) -> "MixedPrecision":
        """The precision the per-family ``use_amp`` flags used to select.

        ``use_amp`` meant autocast at the device's default dtype: fp16 on CUDA
        (with a scaler), bf16 on CPU and fp16 on MPS, both without one: the
        ``GradScaler("cuda")`` they built disabled itself off CUDA.
        """
        device = torch.device(device)
        if not use_amp:
            return cls("fp32", device)
        if device.type == "cpu":
            return cls("bf16", device)
        return cls("fp16", device, scale_loss=device.type == "cuda")

    @property
    def enabled(self) -> bool:
        return self.name != "fp32"

    @property
    def scales_loss(self) -> bool:
        return self._scale_loss

    @contextlib.contextmanager
    def autocast(self, enabled: bool = True) -> Iterator[None]:
        """Autocast at this precision; ``enabled=False`` runs a block in fp32."""
        if not (enabled and self.enabled):
            yield
            return
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            yield

    def grad_scaler(self) -> torch.amp.GradScaler:
        """A fresh loss scaler, enabled only for fp16."""
        if not self.scales_loss:
            return torch.amp.GradScaler("cpu", enabled=False)
        return torch.amp.GradScaler(self.device.type)

    def __repr__(self) -> str:
        return f"MixedPrecision({self.name!r}, device={self.device})"


class GradientUpdate:
    """One optimizer's step: zero, backward, unscale, clip, step.

    Args:
        optimizer: The optimizer to step.
        parameters: The parameters to clip; normally the ones ``optimizer``
            owns.
        precision: Decides whether the loss is scaled.
        max_grad_norm: Clip threshold, or ``None`` not to clip. The norm is
            measured after unscaling, so the threshold means the same thing at
            every precision.
        accumulation_steps: Optimizer steps happen every this many calls;
            the calls in between only add gradients. Each loss is divided by
            it, so the step sees the mean over the accumulated batches.

    Calling it returns the gradient norm before clipping (a tensor), or
    ``None`` on a call that only accumulated. A non-finite norm under fp16
    means the scaler skipped the step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.Tensor],
        precision: MixedPrecision,
        *,
        max_grad_norm: Optional[float] = None,
        accumulation_steps: int = 1,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1.")
        self.optimizer = optimizer
        self.parameters = [p for p in parameters]
        self.precision = precision
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = int(accumulation_steps)
        self.scaler = precision.grad_scaler()
        self._pending = 0

    def __call__(
        self, loss: torch.Tensor, *, retain_graph: bool = False
    ) -> Optional[torch.Tensor]:
        if self._pending == 0:
            self.optimizer.zero_grad(set_to_none=True)
        self._pending += 1
        final = self._pending >= self.accumulation_steps
        scaled = self.scaler.scale(loss / self.accumulation_steps)
        scaled.backward(retain_graph=retain_graph)  # type: ignore[no-untyped-call]
        if not final:
            return None
        self._pending = 0
        self.scaler.unscale_(self.optimizer)
        if self.max_grad_norm is not None:
            norm = nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        else:
            norm = _grad_norm(self.parameters)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return norm

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])


def _grad_norm(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g) for g in grads])
    )


__all__ = ["GradientUpdate", "MixedPrecision"]
