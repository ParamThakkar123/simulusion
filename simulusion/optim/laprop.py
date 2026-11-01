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

from typing import Any, Callable, Iterable, cast

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

    params = [param.detach() for param, _ in pairs]
    grads = [grad.detach() for _, grad in pairs]
    # One fused kernel per list instead of a Python loop over tensors.
    grad_norms = torch.stack([norm.float() for norm in torch._foreach_norm(grads, 2)])
    param_norms = torch.stack(
        [norm.float() for norm in torch._foreach_norm(params, 2)]
    ).clamp(min=eps)
    # Clamping the scale at 1 keeps gradients within budget untouched.
    scales = (clip * param_norms / grad_norms.clamp(min=1e-12)).clamp(max=1.0)
    by_dtype: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for grad, scale in zip(grads, scales.unbind()):
        group = by_dtype.setdefault(grad.dtype, ([], []))
        group[0].append(grad)
        group[1].append(scale.to(grad.dtype))
    for group_grads, group_scales in by_dtype.values():
        torch._foreach_mul_(group_grads, group_scales)

    return grad_norms.pow(2).sum().sqrt()


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

            # Parameters are updated together, one fused kernel per operation,
            # in buckets that share a step count, device and dtype.
            buckets: dict[tuple, list[torch.Tensor]] = {}
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                key = (state["step"], param.device, param.dtype)
                buckets.setdefault(key, []).append(param)

            for (step, _, _), params in buckets.items():
                grads = [cast(torch.Tensor, param.grad) for param in params]
                exp_avgs = [self.state[param]["exp_avg"] for param in params]
                exp_avg_sqs = [self.state[param]["exp_avg_sq"] for param in params]

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                # RMSProp normalization first ...
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
                denom = torch._foreach_div(exp_avg_sqs, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)
                normalized = torch._foreach_div(grads, list(denom))

                # ... then momentum on the already-normalized gradient.
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, normalized, alpha=1.0 - beta1)

                if weight_decay != 0.0:
                    torch._foreach_mul_(params, 1.0 - lr * weight_decay)

                torch._foreach_add_(params, exp_avgs, alpha=-lr / bias_correction1)

        return loss
