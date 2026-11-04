"""Encoder, decoder, and prediction heads for DreamerV3.

Compared to earlier Dreamer generations these networks differ in three ways:

* **RMSNorm + SiLU** replace unnormalized ELU/ReLU stacks.
* **Vector observations are symlog transformed** on the encoder input and on the
  decoder target, which prevents large inputs from producing large
  reconstruction gradients that would swamp the representation loss.
* **Scalar heads emit two-hot logits over exponentially spaced bins** instead of
  Gaussian parameters, and their output weights are zero-initialized so the
  agent does not hallucinate rewards and values before it has learned anything.

Both image (rank-3, ``(C, H, W)``) and vector (rank-1, ``(D,)``) observations are
supported; the right stack is chosen from the observation rank.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions

from simulusion.export import ExportableAgentMixin
from simulusion.layers.rms_norm import RMSNorm
from simulusion.utils.dreamer_utils import SymexpTwoHotDist, symlog

__all__ = [
    "ChannelRMSNorm",
    "DreamerV3Encoder",
    "DreamerV3Decoder",
    "DreamerV3MultiEncoder",
    "DreamerV3MultiDecoder",
    "split_obs_space",
    "DreamerV3Head",
    "DreamerV3Actor",
    "count_parameters",
    "head_input_size",
    "preprocess_image",
]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError as exc:
        options = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"Unknown activation {name!r}. Options: {options}") from exc


def preprocess_image(obs: torch.Tensor) -> torch.Tensor:
    """Scale raw ``uint8`` images to the ``[0, 1]`` range used by DreamerV3.

    The decoder ends in a sigmoid, so targets live in ``[0, 1]`` rather than the
    ``[-0.5, 0.5]`` range used by DreamerV1/V2.
    """
    return obs.to(torch.float32) / 255.0


def _is_image(shape: Sequence[int]) -> bool:
    return len(shape) == 3


class ChannelRMSNorm(ExportableAgentMixin, nn.Module):
    """RMSNorm over the channel dimension of an ``(N, C, H, W)`` tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = RMSNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def _conv_out(size: int, kernel: int = 4, stride: int = 2, padding: int = 1) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class DreamerV3Encoder(ExportableAgentMixin, nn.Module):
    """Encode observations into the embedding consumed by the RSSM posterior.

    Args:
        obs_shape: ``(C, H, W)`` for images or ``(D,)`` for vectors.
        cnn_depth: Channel count of the first convolution; later layers double.
        hidden_size: Width of the MLP used for vector observations.
        mlp_layers: Number of hidden layers in the vector MLP.
        min_resolution: Spatial size at which the convolution stack stops.
        activation: Activation name.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        cnn_depth: int = 32,
        hidden_size: int = 256,
        mlp_layers: int = 3,
        min_resolution: int = 4,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.is_image = _is_image(self.obs_shape)
        self.resolutions: list[tuple[int, int]] = []
        self.out_channels = 0

        if self.is_image:
            channels, height, width = self.obs_shape
            layers: list[nn.Module] = []
            resolutions: list[tuple[int, int]] = [(int(height), int(width))]
            in_ch = channels
            depth = int(cnn_depth)
            while min(height, width) > min_resolution and len(resolutions) <= 6:
                layers.append(nn.Conv2d(in_ch, depth, 4, stride=2, padding=1))
                layers.append(ChannelRMSNorm(depth))
                layers.append(_activation(activation))
                height, width = _conv_out(height), _conv_out(width)
                resolutions.append((height, width))
                in_ch = depth
                depth *= 2
            if len(resolutions) == 1:
                raise ValueError(
                    f"Image observations must be larger than min_resolution="
                    f"{min_resolution}; got {self.obs_shape}."
                )
            self.conv = nn.Sequential(*layers)
            self.resolutions = resolutions
            self.out_channels = in_ch
            self.embed_size = in_ch * height * width
        else:
            layers = []
            in_features = self.obs_shape[0]
            for _ in range(int(mlp_layers)):
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(RMSNorm(hidden_size))
                layers.append(_activation(activation))
                in_features = hidden_size
            self.mlp = nn.Sequential(*layers)
            self.embed_size = hidden_size

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Embed a batch of observations of shape ``(*batch, *obs_shape)``."""
        batch_shape = obs.shape[: obs.ndim - len(self.obs_shape)]
        flat = obs.reshape(-1, *self.obs_shape)
        if self.is_image:
            out = self.conv(flat)
            out = out.reshape(out.shape[0], -1)
        else:
            out = self.mlp(symlog(flat))
        return out.reshape(*batch_shape, self.embed_size)


class DreamerV3Decoder(ExportableAgentMixin, nn.Module):
    """Reconstruct observations from model states.

    Returns the predicted mean rather than a distribution: DreamerV3 trains the
    decoder with a squared error (on symlog targets for vector inputs), so the
    caller computes the loss directly.

    Args:
        feature_size: Size of the concatenated ``(z, h)`` model state.
        obs_shape: ``(C, H, W)`` for images or ``(D,)`` for vectors.
        encoder: Encoder whose resolution schedule should be mirrored. Required
            for image observations so the transposed convolutions invert the
            encoder exactly, including odd spatial sizes.
        cnn_depth: Base channel count, matching the encoder.
        hidden_size: Width of the MLP used for vector observations.
        mlp_layers: Number of hidden layers in the vector MLP.
        activation: Activation name.
    """

    def __init__(
        self,
        feature_size: int,
        obs_shape: Sequence[int],
        encoder: DreamerV3Encoder | None = None,
        cnn_depth: int = 32,
        hidden_size: int = 256,
        mlp_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.is_image = _is_image(self.obs_shape)
        self.feature_size = int(feature_size)

        if self.is_image:
            if encoder is None or not encoder.is_image:
                raise ValueError(
                    "An image encoder must be supplied so the decoder can mirror "
                    "its resolution schedule."
                )
            resolutions = list(encoder.resolutions)
            self.start_resolution = resolutions[-1]
            self.start_channels = encoder.out_channels
            self.linear = nn.Linear(
                self.feature_size,
                self.start_channels
                * self.start_resolution[0]
                * self.start_resolution[1],
            )

            layers: list[nn.Module] = []
            in_ch = self.start_channels
            # Walk the encoder resolutions backwards, doubling spatial size and
            # halving channels at every step.
            for index in range(len(resolutions) - 1, 0, -1):
                target_h, target_w = resolutions[index - 1]
                current_h, current_w = resolutions[index]
                out_pad_h = target_h - 2 * current_h
                out_pad_w = target_w - 2 * current_w
                if out_pad_h not in (0, 1) or out_pad_w not in (0, 1):
                    raise ValueError(
                        "Cannot invert the encoder resolution schedule for "
                        f"{self.obs_shape}; got a mismatch at index {index}."
                    )
                last = index == 1
                out_ch = self.obs_shape[0] if last else max(in_ch // 2, int(cnn_depth))
                layers.append(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        4,
                        stride=2,
                        padding=1,
                        output_padding=(out_pad_h, out_pad_w),
                    )
                )
                if not last:
                    layers.append(ChannelRMSNorm(out_ch))
                    layers.append(_activation(activation))
                in_ch = out_ch
            self.deconv = nn.Sequential(*layers)
        else:
            layers = []
            in_features = self.feature_size
            for _ in range(int(mlp_layers)):
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(RMSNorm(hidden_size))
                layers.append(_activation(activation))
                in_features = hidden_size
            layers.append(nn.Linear(in_features, self.obs_shape[0]))
            self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict observations from features of shape ``(*batch, feature_size)``."""
        batch_shape = features.shape[:-1]
        flat = features.reshape(-1, self.feature_size)
        if self.is_image:
            out = self.linear(flat)
            out = out.reshape(
                -1,
                self.start_channels,
                self.start_resolution[0],
                self.start_resolution[1],
            )
            out = torch.sigmoid(self.deconv(out))
        else:
            out = self.mlp(flat)
        return out.reshape(*batch_shape, *self.obs_shape)

    def loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Squared reconstruction error, summed over observation dimensions.

        Vector targets are symlog transformed first; image targets are already in
        ``[0, 1]`` and are compared directly against the sigmoid output.
        """
        if not self.is_image:
            target = symlog(target)
        error = 0.5 * (prediction - target).pow(2)
        return error.reshape(*error.shape[: error.ndim - len(self.obs_shape)], -1).sum(
            dim=-1
        )


class DreamerV3Head(ExportableAgentMixin, nn.Module):
    """MLP head for reward, value, and continuation prediction.

    Args:
        in_features: Size of the input features.
        layers: Number of hidden layers.
        units: Hidden width.
        dist: One of ``"symexp_twohot"``, ``"binary"``, or ``"symlog_mse"``.
        num_bins: Number of bins for the two-hot distribution.
        symlog_range: Half-width of the two-hot grid in symlog space.
        activation: Activation name.
        zero_init_output: Zero-initialize the output layer. DreamerV3 does this
            for the reward predictor and the critic so their initial predictions
            are exactly zero, which measurably speeds up early learning.
    """

    def __init__(
        self,
        in_features: int,
        layers: int = 3,
        units: int = 256,
        dist: str = "symexp_twohot",
        num_bins: int = 255,
        symlog_range: float = 20.0,
        activation: str = "silu",
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.dist = dist
        self.num_bins = int(num_bins)
        self.symlog_range = float(symlog_range)

        modules: list[nn.Module] = []
        current = int(in_features)
        for _ in range(int(layers)):
            modules.append(nn.Linear(current, units))
            modules.append(RMSNorm(units))
            modules.append(_activation(activation))
            current = int(units)

        out_features = self.num_bins if dist == "symexp_twohot" else 1
        output = nn.Linear(current, out_features)
        if zero_init_output:
            nn.init.zeros_(output.weight)
            if output.bias is not None:
                nn.init.zeros_(output.bias)
        modules.append(output)
        self.model = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> Any:
        out = self.model(features)
        if self.dist == "symexp_twohot":
            return SymexpTwoHotDist(out, self.num_bins, self.symlog_range)
        if self.dist == "binary":
            return distributions.Bernoulli(logits=out.squeeze(-1), validate_args=False)
        if self.dist == "symlog_mse":
            return out.squeeze(-1)
        raise ValueError(f"Unknown head distribution {self.dist!r}")


class DreamerV3Actor(ExportableAgentMixin, nn.Module):
    """Policy head supporting one-hot discrete and continuous actions.

    DreamerV3 uses the Reinforce estimator for both action types, so the head
    always exposes an explicit distribution with ``log_prob`` and ``entropy``.

    * Discrete actions use a one-hot categorical with the same 1% unimix as the
      world model latents, which keeps log-probabilities finite.
    * Continuous actions use a Normal whose mean is squashed by ``tanh`` and
      whose standard deviation is bounded to ``[min_std, max_std]``. Actions are
      clipped to the ``[-1, 1]`` action space by the environment wrapper.

    Args:
        in_features: Size of the model-state features.
        action_size: Number of actions (discrete) or action dimensions.
        discrete: Whether the action space is discrete.
        layers: Number of hidden layers.
        units: Hidden width.
        unimix: Uniform mixture fraction for the discrete distribution.
        min_std: Lower bound on the continuous standard deviation.
        max_std: Upper bound on the continuous standard deviation.
        activation: Activation name.
    """

    def __init__(
        self,
        in_features: int,
        action_size: int,
        discrete: bool,
        layers: int = 3,
        units: int = 256,
        unimix: float = 0.01,
        min_std: float = 0.1,
        max_std: float = 1.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.action_size = int(action_size)
        self.discrete = bool(discrete)
        self.unimix = float(unimix)
        self.min_std = float(min_std)
        self.max_std = float(max_std)

        modules: list[nn.Module] = []
        current = int(in_features)
        for _ in range(int(layers)):
            modules.append(nn.Linear(current, units))
            modules.append(RMSNorm(units))
            modules.append(_activation(activation))
            current = int(units)
        out_features = self.action_size if self.discrete else 2 * self.action_size
        modules.append(nn.Linear(current, out_features))
        self.model = nn.Sequential(*modules)

    def dist(self, features: torch.Tensor) -> distributions.Distribution:
        """Return the action distribution for the given features."""
        out = self.model(features)
        if self.discrete:
            logits = out
            if self.unimix > 0.0:
                probs = torch.softmax(logits, dim=-1)
                uniform = torch.ones_like(probs) / float(self.action_size)
                probs = (1.0 - self.unimix) * probs + self.unimix * uniform
                logits = torch.log(probs)
            return distributions.OneHotCategorical(logits=logits, validate_args=False)
        mean, std = torch.chunk(out, 2, dim=-1)
        mean = torch.tanh(mean)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
        return distributions.Independent(
            distributions.Normal(mean, std, validate_args=False), 1
        )

    def forward(
        self, features: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample (or take the mode of) an action."""
        dist = self.dist(features)
        if deterministic:
            if self.discrete:
                logits = torch.as_tensor(getattr(dist, "logits"))
                index = torch.argmax(logits, dim=-1)
                return torch.nn.functional.one_hot(index, self.action_size).to(
                    logits.dtype
                )
            return dist.mean
        if self.discrete:
            # Inverse-CDF draw of a one-hot action: the distribution of
            # OneHotCategorical.sample without torch.multinomial, which cannot
            # be captured in a CUDA graph.
            probs = torch.as_tensor(getattr(dist, "probs"))
            uniform = torch.rand_like(probs[..., :1])
            index = (probs.cumsum(-1) < uniform).sum(-1).clamp(max=self.action_size - 1)
            return torch.nn.functional.one_hot(index, self.action_size).to(probs.dtype)
        # mean + std * noise samples the same Normal. torch.normal would first
        # check std >= 0 on the host, which a CUDA graph cannot contain.
        normal = getattr(dist, "base_dist")
        with torch.no_grad():
            return normal.loc + normal.scale * torch.randn_like(normal.loc)

    def add_exploration(
        self, action: torch.Tensor, action_noise: float = 0.0
    ) -> torch.Tensor:
        """Optional additive exploration noise for continuous actions.

        DreamerV3 relies on its entropy regularizer rather than injected noise,
        so this is a no-op unless ``action_noise`` is explicitly set.
        """
        if action_noise <= 0.0 or self.discrete:
            return action
        noisy = action + torch.randn_like(action) * action_noise
        return torch.clamp(noisy, -1.0, 1.0)


def split_obs_space(
    obs_space: Mapping[str, Sequence[int]],
) -> tuple[list[str], list[str]]:
    """Split observation keys into image (rank 3) and vector (rank 1) keys.

    Raises:
        ValueError: For other ranks, or images whose spatial sizes differ
            (image inputs are stacked along channels into one CNN).
    """
    images: list[str] = []
    vectors: list[str] = []
    for key, shape in obs_space.items():
        rank = len(tuple(shape))
        if rank == 3:
            images.append(key)
        elif rank == 1:
            vectors.append(key)
        else:
            raise ValueError(
                f"Observation {key!r} has shape {tuple(shape)}; DreamerV3 encodes "
                "images (C, H, W) and vectors (D,)."
            )
    sizes = {tuple(obs_space[key])[1:] for key in images}
    if len(sizes) > 1:
        raise ValueError(
            f"Image observations {images} must share a spatial size; got {sizes}."
        )
    return images, vectors


class DreamerV3MultiEncoder(ExportableAgentMixin, nn.Module):
    """Encode a dict of observations into one embedding.

    As in DreamerV3, image inputs are stacked along channels and encoded by one
    CNN, vector inputs are concatenated, symlog transformed and encoded by one
    MLP, and the two embeddings are concatenated.

    Args:
        obs_space: Observation key -> shape, ``(C, H, W)`` or ``(D,)``.
        cnn_depth, hidden_size, mlp_layers, min_resolution, activation: As for
            :class:`DreamerV3Encoder`.
    """

    def __init__(
        self,
        obs_space: Mapping[str, Sequence[int]],
        cnn_depth: int = 32,
        hidden_size: int = 256,
        mlp_layers: int = 3,
        min_resolution: int = 4,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.obs_space = {
            key: tuple(int(d) for d in shape) for key, shape in obs_space.items()
        }
        if not self.obs_space:
            raise ValueError("obs_space must name at least one observation")
        self.image_keys, self.vector_keys = split_obs_space(self.obs_space)
        kwargs: dict[str, Any] = dict(
            cnn_depth=cnn_depth,
            hidden_size=hidden_size,
            mlp_layers=mlp_layers,
            min_resolution=min_resolution,
            activation=activation,
        )
        self.image_encoder: DreamerV3Encoder | None = None
        self.vector_encoder: DreamerV3Encoder | None = None
        if self.image_keys:
            first = self.obs_space[self.image_keys[0]]
            channels = sum(self.obs_space[key][0] for key in self.image_keys)
            self.image_encoder = DreamerV3Encoder((channels, *first[1:]), **kwargs)
        if self.vector_keys:
            size = sum(self.obs_space[key][0] for key in self.vector_keys)
            self.vector_encoder = DreamerV3Encoder((size,), **kwargs)
        self.embed_size = sum(
            encoder.embed_size
            for encoder in (self.image_encoder, self.vector_encoder)
            if encoder is not None
        )

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Embed ``obs``: images in ``[0, 1]``, vectors raw (symlog is applied here).

        Every tensor has shape ``(*batch, *obs_space[key])``.
        """
        parts = []
        if self.image_encoder is not None:
            parts.append(
                self.image_encoder(torch.cat([obs[k] for k in self.image_keys], -3))
            )
        if self.vector_encoder is not None:
            parts.append(
                self.vector_encoder(torch.cat([obs[k] for k in self.vector_keys], -1))
            )
        return torch.cat(parts, dim=-1)


class DreamerV3MultiDecoder(ExportableAgentMixin, nn.Module):
    """Reconstruct every encoded observation from the model state.

    Mirrors :class:`DreamerV3MultiEncoder`: one transposed CNN emits all image
    channels (sigmoid, compared with ``[0, 1]`` targets), one MLP emits all
    vector entries (compared with symlog targets). Each input's squared error
    is summed over its dimensions, and the inputs' errors are added.

    Args:
        feature_size: Size of the model state ``(z, h)``.
        encoder: The encoder whose inputs and resolution schedule to mirror.
        cnn_depth, hidden_size, mlp_layers, activation: As for
            :class:`DreamerV3Decoder`.
    """

    def __init__(
        self,
        feature_size: int,
        encoder: DreamerV3MultiEncoder,
        cnn_depth: int = 32,
        hidden_size: int = 256,
        mlp_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.obs_space = dict(encoder.obs_space)
        self.image_keys = list(encoder.image_keys)
        self.vector_keys = list(encoder.vector_keys)
        kwargs: dict[str, Any] = dict(
            cnn_depth=cnn_depth,
            hidden_size=hidden_size,
            mlp_layers=mlp_layers,
            activation=activation,
        )
        self.image_decoder: DreamerV3Decoder | None = None
        self.vector_decoder: DreamerV3Decoder | None = None
        if encoder.image_encoder is not None:
            self.image_decoder = DreamerV3Decoder(
                feature_size,
                encoder.image_encoder.obs_shape,
                encoder=encoder.image_encoder,
                **kwargs,
            )
        if encoder.vector_encoder is not None:
            self.vector_decoder = DreamerV3Decoder(
                feature_size, encoder.vector_encoder.obs_shape, **kwargs
            )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict each observation, keyed like the encoder's inputs."""
        out: dict[str, torch.Tensor] = {}
        if self.image_decoder is not None:
            channels = [self.obs_space[k][0] for k in self.image_keys]
            images = self.image_decoder(features).split(channels, dim=-3)
            out.update(zip(self.image_keys, images))
        if self.vector_decoder is not None:
            sizes = [self.obs_space[k][0] for k in self.vector_keys]
            vectors = self.vector_decoder(features).split(sizes, dim=-1)
            out.update(zip(self.vector_keys, vectors))
        return out

    def loss_terms(
        self,
        prediction: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Per-input reconstruction loss, each summed over that input's dimensions.

        Image targets are expected in ``[0, 1]``; vector targets raw (they are
        symlog transformed here, as the decoder predicts in symlog space).
        """
        terms: dict[str, torch.Tensor] = {}
        for key in self.image_keys:
            error = 0.5 * (prediction[key] - target[key]).pow(2)
            terms[key] = error.flatten(-3).sum(-1)
        for key in self.vector_keys:
            error = 0.5 * (prediction[key] - symlog(target[key])).pow(2)
            terms[key] = error.sum(-1)
        return terms

    def loss(
        self,
        prediction: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Total reconstruction loss: the per-input losses added together."""
        terms = list(self.loss_terms(prediction, target).values())
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total


def head_input_size(latent_dim: int, latent_classes: int, deter_size: int) -> int:
    """Feature size produced by concatenating the flattened latent and ``h``."""
    return int(latent_dim) * int(latent_classes) + int(deter_size)


def count_parameters(module: nn.Module) -> int:
    """Total parameter count of a module, for the model-size presets."""
    return int(np.sum([param.numel() for param in module.parameters()]))
