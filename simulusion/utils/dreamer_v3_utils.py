"""Robustness primitives introduced by DreamerV3.

This module collects the small, self-contained numerical tools that make
DreamerV3 train with fixed hyperparameters across domains:

* :class:`SymexpTwoHotDist` -- a categorical distribution over exponentially
  spaced bins, trained on two-hot encoded targets. This decouples gradient
  magnitudes from target magnitudes for the reward predictor and the critic.
* :class:`ReturnNormalizer` -- percentile-based return scaling with a
  denominator limit, used to keep the actor entropy regularizer on a fixed
  scale regardless of reward magnitude or sparsity.
* :func:`free_bits` -- clipping of the KL terms below one nat.
* :func:`lambda_return` -- bootstrapped ``lambda``-returns.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

import torch

from simulusion.utils.dreamer_utils import symexp, symlog

__all__ = [
    "SymexpTwoHotDist",
    "ReturnNormalizer",
    "free_bits",
    "lambda_return",
    "symexp",
    "symlog",
    "twohot",
]


def twohot(target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot encode ``target`` against a sorted, monotonically increasing grid.

    The two-hot encoding generalizes one-hot encoding to continuous values: all
    entries are zero except at the two bins bracketing the target, whose weights
    sum to one and are linearly interpolated by distance.

    Args:
        target: Tensor of arbitrary shape with real-valued targets.
        bins: 1-D tensor of bin locations, sorted ascending.

    Returns:
        Tensor of shape ``(*target.shape, len(bins))``.
    """
    if bins.ndim != 1:
        raise ValueError(f"bins must be 1-D, got shape {tuple(bins.shape)}")
    num_bins = bins.shape[0]
    if num_bins < 2:
        raise ValueError("bins must contain at least two entries")

    target = target.to(bins.dtype)
    # Tensor bounds, so the clamp does not wait on the GPU.
    clipped = target.clamp(min=bins[0], max=bins[-1])

    # Index of the lower bracketing bin. `bins[below] <= clipped <= bins[above]`.
    below = bins.reshape(*([1] * clipped.ndim), num_bins) <= clipped.unsqueeze(-1)
    below = below.sum(dim=-1) - 1
    below = below.clamp(0, num_bins - 2)
    above = below + 1

    lower = bins[below]
    upper = bins[above]
    span = (upper - lower).clamp(min=torch.finfo(bins.dtype).eps)
    weight_upper = ((clipped - lower) / span).clamp(0.0, 1.0)
    weight_lower = 1.0 - weight_upper

    encoded = torch.zeros(
        *clipped.shape, num_bins, dtype=bins.dtype, device=bins.device
    )
    encoded.scatter_(-1, below.unsqueeze(-1), weight_lower.unsqueeze(-1))
    encoded.scatter_add_(-1, above.unsqueeze(-1), weight_upper.unsqueeze(-1))
    return encoded


class SymexpTwoHotDist:
    """Categorical distribution over exponentially spaced bins.

    The network emits ``num_bins`` logits. Bin locations are
    ``symexp(linspace(-symlog_range, +symlog_range, num_bins))``, so they span
    many orders of magnitude while remaining dense near zero. Predictions are
    read out as the probability-weighted average of the bin locations, which
    lets the head output any continuous value in the covered interval.

    Training minimizes the cross entropy against the two-hot encoded target.
    Because the loss only depends on the predicted probabilities and not on the
    bin locations, gradient magnitudes are decoupled from target magnitudes.

    Args:
        logits: Tensor of shape ``(*batch, num_bins)``.
        num_bins: Number of bins.
        symlog_range: Half-width of the grid in symlog space.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        num_bins: int = 255,
        symlog_range: float = 20.0,
    ) -> None:
        if logits.shape[-1] != int(num_bins):
            raise ValueError(
                f"logits last dimension {logits.shape[-1]} does not match "
                f"num_bins={num_bins}"
            )
        self.logits = logits
        self.num_bins = int(num_bins)
        self.symlog_range = float(symlog_range)
        # Bins are always float32: symexp(20) overflows float16, so the grid
        # must not follow an autocast dtype.
        self.bins = symexp(
            torch.linspace(
                -self.symlog_range,
                self.symlog_range,
                self.num_bins,
                device=logits.device,
                dtype=torch.float32,
            )
        )
        # Bins ascend, so the negative ones are a prefix. Their count is found
        # on the host (symexp keeps signs) instead of by masking GPU tensors.
        grid = torch.linspace(-self.symlog_range, self.symlog_range, self.num_bins)
        self._num_negative = int((grid < 0).sum())

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits.float(), dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """Expected value under the predicted bin probabilities.

        Two precautions keep this accurate across the many orders of magnitude
        the bins span. Positive and negative bins are accumulated separately,
        each from small to large magnitude, and the reduction runs in float64.
        Without the wider accumulator, cancellation between the two halves
        leaves a residual on the order of the largest bin's float32 ulp -- which
        for the default grid means a zero-initialized head would predict a
        reward of about -2 instead of exactly 0.
        """
        probs = self.probs.double()
        bins = self.bins.double()
        weighted = probs * bins
        negative = weighted[..., : self._num_negative]
        positive = weighted[..., self._num_negative :]
        # Reverse the negative half so both halves accumulate small -> large.
        negative_sum = torch.flip(negative, dims=(-1,)).sum(dim=-1)
        positive_sum = positive.sum(dim=-1)
        return (negative_sum + positive_sum).to(self.logits.dtype)

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """Cross entropy against the two-hot encoding of ``target``.

        Args:
            target: Real-valued targets shaped like ``logits`` without the final
                bin dimension.

        Returns:
            Log-probability tensor shaped like ``target``.
        """
        encoded = twohot(target.float(), self.bins)
        return (torch.log_softmax(self.logits.float(), dim=-1) * encoded).sum(dim=-1)

    def cross_entropy_to(self, other: "SymexpTwoHotDist") -> torch.Tensor:
        """Cross entropy of this distribution against ``other``'s probabilities.

        Used for the critic's EMA (slow target) regularizer, where the target is
        a full distribution rather than a scalar.
        """
        target = other.probs.detach()
        return -(torch.log_softmax(self.logits.float(), dim=-1) * target).sum(dim=-1)


class ReturnNormalizer:
    """Percentile-range return normalization with a denominator limit.

    Divides returns by an exponentially smoothed estimate of the range between
    the ``low`` and ``high`` percentiles. Only large return magnitudes are scaled
    down: the denominator is ``max(limit, scale)``, so returns that are already
    small pass through untouched. This preserves information about reward
    frequency (unlike advantage normalization) and does not blow up under sparse
    rewards (unlike standard-deviation normalization).

    Args:
        decay: EMA decay for the smoothed range.
        limit: Lower bound on the denominator.
        low: Lower percentile, in ``[0, 100]``.
        high: Upper percentile, in ``[0, 100]``.
    """

    def __init__(
        self,
        decay: float = 0.99,
        limit: float = 1.0,
        low: float = 5.0,
        high: float = 95.0,
    ) -> None:
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(f"Require 0 <= low < high <= 100, got {low} and {high}")
        self.decay = float(decay)
        self.limit = float(limit)
        self.low = float(low)
        self.high = float(high)
        self._scale: torch.Tensor | None = None

    @property
    def scale(self) -> float:
        """Current smoothed percentile range (before the limit is applied)."""
        return 0.0 if self._scale is None else float(self._scale)

    def scale_tensor(self) -> torch.Tensor:
        """:attr:`scale` as a tensor, read without synchronizing with the GPU."""
        if self._scale is None:
            return torch.zeros(())
        return torch.as_tensor(self._scale).detach()

    def update(self, returns: torch.Tensor) -> torch.Tensor:
        """Update the EMA from a batch of returns and return the denominator."""
        flat = returns.detach().reshape(-1).float()
        quantiles = torch.quantile(
            flat,
            torch.tensor([self.low / 100.0, self.high / 100.0], device=flat.device),
        )
        batch_scale = (quantiles[1] - quantiles[0]).clamp(min=0.0)
        if self._scale is None:
            self._scale = batch_scale
        else:
            self._scale = (
                self.decay * self._scale.to(batch_scale.device)
                + (1.0 - self.decay) * batch_scale
            )
        return self.denominator(device=returns.device, dtype=returns.dtype)

    def denominator(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Return ``max(limit, scale)`` as a tensor."""
        value = 0.0 if self._scale is None else self._scale
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        return torch.clamp(tensor, min=self.limit)

    def __call__(self, returns: torch.Tensor) -> torch.Tensor:
        return returns / self.update(returns)

    def state_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "decay": self.decay,
            "limit": self.limit,
            "low": self.low,
            "high": self.high,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.limit = float(state.get("limit", self.limit))
        self.low = float(state.get("low", self.low))
        self.high = float(state.get("high", self.high))
        scale = state.get("scale")
        self._scale = None if scale is None else torch.as_tensor(float(scale))


def free_bits(kl: torch.Tensor, nats: float = 1.0) -> torch.Tensor:
    """Clip a KL term below ``nats`` so it stops contributing once minimized.

    DreamerV3 applies this to both the dynamics and representation losses, which
    lets the representation loss carry a small weight without collapsing to
    trivially predictable (and uninformative) representations.
    """
    return torch.clamp(kl, min=float(nats))


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute bootstrapped ``lambda``-returns.

    Implements ``R_t = r_t + gamma_t * ((1 - lambda) * v_t + lambda * R_{t+1})``
    with ``R_T = bootstrap``, where ``gamma_t`` is folded into ``continues``
    (that is, ``continues = discount * continue_flag``).

    Args:
        rewards: Rewards ``r_t``, shape ``(T, ...)``.
        values: Value estimates ``v_t`` for the *next* state, shape ``(T, ...)``.
        continues: Discounted continuation flags, shape ``(T, ...)``.
        bootstrap: Value used to terminate the recursion, shape ``(...)``.
        lambda_: Trace decay in ``[0, 1]``.

    Returns:
        Tensor of returns with shape ``(T, ...)``.
    """
    if not rewards.shape == values.shape == continues.shape:
        raise ValueError(
            "rewards, values and continues must share a shape; got "
            f"{tuple(rewards.shape)}, {tuple(values.shape)}, {tuple(continues.shape)}"
        )
    horizon = rewards.shape[0]
    interm = rewards + continues * values * (1.0 - lambda_)
    outputs = []
    accumulated = bootstrap
    for t in range(horizon - 1, -1, -1):
        accumulated = interm[t] + continues[t] * lambda_ * accumulated
        outputs.append(accumulated)
    return torch.flip(torch.stack(outputs, dim=0), dims=(0,))
