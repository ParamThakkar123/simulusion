"""Utilities shared by the Dreamer family (V1, V2 and V3).

Besides the generic helpers (parameter freezing, logging, ``lambda``-returns),
this module holds the robustness primitives DreamerV3 introduced to train with
fixed hyperparameters across domains:

* :func:`symlog` / :func:`symexp` -- symmetric log squashing and its inverse.
* :func:`twohot` -- two-hot encoding against an arbitrary sorted grid.
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

import os
import pickle  # noqa: S403  # restricted unpickler loaded separately
import torch
import numpy as np

from typing import Any, Iterable
from torch.nn import Module
from torch.nn.parameter import Parameter

from types import ModuleType
from typing import Optional

from simulusion.utils.logging_utils import MetricsLogger, get_package_logger

# Optional WandB availability is handled lazily by MetricsLogger.
wandb: Optional[ModuleType] = None


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform used by Dreamer V2/V3 for reward/value targets.

    Defined as ``sign(x) * log(1 + |x|)``. This compresses large positive or
    negative values into a range that is easier to predict with a categorical
    distribution over a bounded set of buckets.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`.

    Defined as ``sign(x) * (exp(|x|) - 1)``.
    """
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class TwoHotEncoder:
    """Two-hot encoding for symlog targets (Dreamer V2 reward/value heads).

    A target value is softly assigned to the two nearest buckets on a uniform
    grid spanning ``[-symlog_range, symlog_range]``. The categorical logits
    produced by a network can then be decoded back into a real value by
    computing the expected bucket center.

    Args:
        num_buckets: Number of buckets in the categorical distribution.
        symlog_range: Maximum absolute value (in symlog space) covered by the
            grid. Values outside the range are clipped to the boundary buckets.
    """

    def __init__(self, num_buckets: int = 255, symlog_range: float = 10.0):
        self.num_buckets = int(num_buckets)
        self.symlog_range = float(symlog_range)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Allocate the bucket-center buffer on CPU. Use :meth:`to` to move."""
        self._bucket_centers = torch.linspace(
            -self.symlog_range, self.symlog_range, self.num_buckets
        )
        self.bucket_centers = self._bucket_centers

    def to(self, device: torch.device) -> "TwoHotEncoder":
        self.bucket_centers = self._bucket_centers.to(device)
        return self

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        """Two-hot encode a real-valued target into soft bucket probabilities.

        Args:
            target: Tensor of arbitrary shape containing real-valued targets.

        Returns:
            Tensor with an extra final dimension of size ``num_buckets``
            containing the soft two-hot distribution. The encoding assumes
            the target is already in symlog space, matching Dreamer V2.
        """
        centers = self._bucket_centers.to(device=target.device, dtype=torch.float32)
        return twohot(target, centers)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode categorical logits into the expected real-valued prediction.

        The logits are first softmaxed and then combined with the bucket
        centers. The output is passed through :func:`symexp` to invert the
        symlog transform.

        Args:
            logits: Tensor with a final dimension of ``num_buckets``.

        Returns:
            Tensor with the same shape as ``logits`` minus the last dimension.
        """
        probs = torch.softmax(logits, dim=-1)
        centers = self.bucket_centers.to(probs.device)
        expectation = (probs * centers).sum(-1, keepdim=True)
        return symexp(expectation)


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


def get_parameters(modules: Iterable[Module]) -> list[Parameter]:
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    """Context manager that temporarily disables gradients for given modules.

    Useful during imagination or target-network forward passes where gradients
    through certain components should be blocked for speed and correctness.
    """

    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
          output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self) -> None:
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


class Logger:
    """Experiment logger for scalars and GIF rollouts using WandB.

    Provides helpers to write scalar metrics, dump pickle snapshots, and save
    video previews during Dreamer training/evaluation.
    """

    def __init__(
        self,
        log_dir: str,
        enable_wandb: bool = False,
        wandb_project: str = "simulusion",
        wandb_entity: str = "",
        video_format: str = "gif",
        video_fps: int = 20,
        enable_tensorboard: bool = False,
        enable_console: bool = True,
        enable_jsonl: bool = True,
        jsonl_filename: str = "metrics.jsonl",
    ) -> None:
        self._log_dir = log_dir
        self._logger = get_package_logger("dreamer")
        self._logger.info("logging outputs to %s", log_dir)
        self._n_logged_samples = 10
        self.enable_wandb = enable_wandb
        self.video_format = video_format
        self.video_fps = video_fps
        self.metrics = MetricsLogger(
            log_dir,
            logger=self._logger,
            enable_console=enable_console,
            enable_jsonl=enable_jsonl,
            jsonl_filename=jsonl_filename,
            enable_tensorboard=enable_tensorboard,
            enable_wandb=enable_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
        self._wandb_run = self.metrics._wandb_run

    def log_scalar(self, scalar: Any, name: str, step_: int) -> None:
        self.metrics.log({name: scalar}, step_)

    def log_scalars(self, scalar_dict: dict[str, Any], step: int) -> None:
        self.metrics.log(scalar_dict, step)
        self.dump_scalars_to_pickle(scalar_dict, step)

    def log_videos(
        self,
        videos: Any,
        step: int,
        max_videos_to_save: int = 1,
        fps: int | None = None,
        video_title: str = "video",
    ) -> None:
        if fps is None:
            fps = self.video_fps
        format = self.video_format
        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, videos.shape[0]])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0] < max_length:
                padding = np.tile(
                    [videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1)
                )
                videos[i] = np.concatenate([videos[i], padding], 0)

            video_u8 = _video_to_uint8(videos[i])
            if format.lower() == "mp4":
                from simulusion.utils.utils import Mp4Writer

                # Mp4Writer takes BGR, like OpenCV
                if video_u8.shape[-1] == 3:
                    video_u8 = video_u8[..., ::-1]
                new_video_title = video_title + "{}_{}".format(step, i) + ".mp4"
                filename = os.path.join(self._log_dir, new_video_title)
                height, width = video_u8.shape[1], video_u8.shape[2]
                out = Mp4Writer(filename, fps, (width, height))
                for frame in np.ascontiguousarray(video_u8):
                    out.write(frame)
                out.release()
            else:  # gif
                new_video_title = video_title + "{}_{}".format(step, i) + ".gif"
                filename = os.path.join(self._log_dir, new_video_title)
                _write_gif(video_u8, filename, fps)

            # Log to WandB
            if self.enable_wandb and self._wandb_run:
                video_array = np.array(videos[i])
                self.metrics.log_video(f"{video_title}_{i}", video_array, step, fps=fps)

    def dump_scalars_to_pickle(
        self, metrics: dict[str, Any], step: int, log_title: str | None = None
    ) -> None:
        log_path = os.path.join(
            self._log_dir, "scalar_data.pkl" if log_title is None else log_title
        )
        with open(log_path, "ab") as f:
            pickle.dump({"step": step, **dict(metrics)}, f)

    def flush(self) -> None:
        self.metrics.flush()


def _video_to_uint8(video: Any) -> np.ndarray:
    """Return ``(T, H, W, C)`` uint8 frames from uint8 or [0, 1] float input.

    Dreamer passes uint8 frames. Scaling those by 255 as if they were floats
    wrapped every pixel around, so only float input is rescaled.
    """
    video = np.asarray(video)
    if np.issubdtype(video.dtype, np.floating):
        return (np.clip(video, 0.0, 1.0) * 255).round().astype(np.uint8)
    return video.astype(np.uint8)


def _write_gif(video_u8: np.ndarray, filename: str, fps: int) -> None:
    """Write ``(T, H, W, C)`` uint8 frames to an animated GIF with Pillow.

    Pillow is already a torchvision dependency. The previous moviepy writer was
    an undeclared dependency, and moviepy's Pillow cap conflicts with the
    patched Pillow release Simulusion requires.
    """
    from PIL import Image

    frames = [
        Image.fromarray(frame[..., 0] if frame.shape[-1] == 1 else frame)
        for frame in video_u8
    ]
    if not frames:
        return
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(1000 / max(1, fps)))),
        loop=0,
    )


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


def compute_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    td_lam: float,
    last_value: torch.Tensor,
) -> torch.Tensor:
    """Compute TD(lambda) returns from imagined rewards, values, and discounts.

    Dreamer V1/V2 convention for :func:`lambda_return`: ``values[t]`` is the
    value of state ``t``, so the next-state values are formed by shifting in
    ``last_value``, which also bootstraps the recursion.
    """
    next_values = torch.cat([values[1:], last_value.unsqueeze(0)], 0)
    return lambda_return(rewards, next_values, discounts, last_value, td_lam)
