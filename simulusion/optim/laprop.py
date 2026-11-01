"""LaProp optimizer and adaptive gradient clipping, as used by DreamerV3.

LaProp separates momentum from adaptivity: gradients are first normalized by an
RMSProp-style second-moment estimate and only then smoothed by momentum. Adam,
by contrast, computes both the momentum and the normalizer from the raw
gradients. The LaProp ordering tolerates a much smaller epsilon (DreamerV3 uses
``1e-20``) and avoids the occasional instabilities observed with Adam.

Adaptive gradient clipping (AGC) rescales each parameter's gradient when its
norm exceeds a fraction of the norm of the weights it belongs to. Because the
threshold is relative to the weights, it does not need to be retuned when loss
functions or loss scales change.

References:
    Ziyin et al., 2020 - https://arxiv.org/abs/2002.04839 (LaProp)
    Brock et al., 2021 - https://arxiv.org/abs/2102.06171 (AGC)
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch.optim import Optimizer

__all__ = ["LaProp", "adaptive_grad_clip_"]


def adaptive_grad_clip_(
    parameters: Iterable[torch.Tensor],
    clip: float = 0.3,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Clip gradients in place, per tensor, relative to the parameter norm.

    Each parameter's gradient is scaled down so that its L2 norm does not exceed
    ``clip * max(||w||, eps)``, where ``||w||`` is the L2 norm of the
    corresponding parameter tensor. Gradients already below the threshold are
    left untouched.

    Args:
        parameters: Parameters whose ``.grad`` should be clipped.
        clip: Fraction of the weight norm allowed for the gradient norm.
        eps: Floor on the weight norm, so zero-initialized tensors still train.

    Returns:
        The total gradient norm before clipping, for logging.
    """
    if clip <= 0.0:
        raise ValueError(f"clip must be positive, got {clip}")

    pairs = [(param, param.grad) for param in parameters if param.grad is not None]
    if not pairs:
        return torch.zeros(())

    total_sq = torch.zeros((), device=pairs[0][1].device, dtype=torch.float32)
    for param, grad in pairs:
        grad_norm = grad.detach().float().norm(2)
        total_sq = total_sq + grad_norm.pow(2)

        param_norm = param.detach().float().norm(2).clamp(min=eps)
        max_norm = clip * param_norm
        # Clamping the scale at 1 keeps gradients within budget untouched.
        scale = (max_norm / grad_norm.clamp(min=1e-12)).clamp(max=1.0)
        grad.detach().mul_(scale.to(grad.dtype))

    return total_sq.sqrt()


class LaProp(Optimizer):
    """LaProp: RMSProp normalization followed by momentum smoothing.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate.
        betas: ``(beta1, beta2)`` for the momentum and second-moment estimates.
        eps: Term added to the denominator. LaProp tolerates values far smaller
            than Adam's because the epsilon does not interact with momentum.
        weight_decay: Decoupled weight decay coefficient (applied to the
            parameters directly, not to the gradients).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 4e-5,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-20,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> Any:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                # RMSProp normalization first ...
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                normalized = grad / denom

                # ... then momentum on the already-normalized gradient.
                exp_avg.mul_(beta1).add_(normalized, alpha=1.0 - beta1)

                if weight_decay != 0.0:
                    param.add_(param, alpha=-lr * weight_decay)

                param.add_(exp_avg, alpha=-lr / bias_correction1)

        return loss
