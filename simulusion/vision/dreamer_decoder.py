from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, Bernoulli
from torch.distributions.independent import Independent
import numpy as np
import torch.distributions as distributions
from torch.distributions import constraints
import torch.nn.functional as F
from simulusion.export import ExportableAgentMixin


_str_to_activation = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


class TanhBijector(distributions.Transform):
    """Bijective tanh transform for squashing Gaussian distributions to [-1, 1].

    This transformation is essential for Dreamer's action policy. Raw neural network
    outputs are Gaussian distributions over R^n, but actions in continuous control
    environments are typically bounded in [-1, 1]. The tanh bijector provides:

    1. **Bijective mapping**: tanh is invertible (with atanh as inverse)
    2. **Stable log-det Jacobian**: Computable for gradient-based training
    3. **Clipped actions**: During inference, actions are naturally bounded

    - Forward: y = tanh(x)
    - Inverse: x = atanh(y) = 0.5 * log((1+y)/(1-y))
    - Log-det: log|dy/dx| = 2*(log(2) - x - softplus(-2x))

    **Usage with Dreamer ActionDecoder**::

        dist = TransformedDistribution(
            Normal(mean, std),
            TanhBijector()
        )
        action = dist.sample()  # Bounded to [-1, 1]

    Reference:
        Building a Scalable Deep RL Library by Learning from Mistakes, Haarnoja et al.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)  # type: ignore[no-untyped-call]

    @property
    def sign(self) -> int:
        return 1

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def atanh(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class ConvDecoder(ExportableAgentMixin, nn.Module):
    """Convolutional decoder for reconstructing observations from latent states.

    Part of Dreamer's world model, this decoder reconstructs image observations
    from the combined stochastic (s) and deterministic (h) RSSM states.

    - Input: Concatenated [stoch_state, deter_state], shape (B, stoch+deter)
    - Process: Dense projection + 4 transposed convolutions (upsampling 2x each)
    - Output: Independent Normal distribution over observation pixels

    The decoder mirrors the ConvEncoder's structure but in reverse (transposed convs
    instead of regular convs). This creates a symmetric autoencoder where the encoder
    and decoder can be trained jointly to learn compressed representations.

    Returns ``torch.distributions.Independent(Normal(mean, std), len(shape))``
    allowing log_prob(observation) computation for reconstruction loss.

    **Usage in Dreamer world model**::

        decoder = ConvDecoder(
            stoch_size=30,
            deter_size=200,
            output_shape=(3, 64, 64),  # RGB images
            activation='relu'
        )
        obs_dist = decoder(latent_features)  # Returns distribution
        log_prob = obs_dist.log_prob(target_observation)

    The reconstruction loss is ``-log_prob(observation)``, which encourages
    the RSSM to learn states that capture observation information.
    """

    def __init__(
        self,
        stoch_size: int,
        deter_size: int,
        output_shape: tuple[int, ...],
        activation: str,
        depth: int = 32,
    ) -> None:
        super().__init__()

        self.output_shape = output_shape
        self.depth = depth
        self.kernels = [5, 5, 6, 6]
        self.act_fn = _str_to_activation[activation]

        self.dense = nn.Linear(stoch_size + deter_size, 32 * self.depth)

        layers: list[nn.Module] = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = (
                32 * self.depth
                if i == 0
                else self.depth * (2 ** (len(self.kernels) - 1 - i))
            )
            out_ch = (
                output_shape[0]
                if i == len(self.kernels) - 1
                else self.depth * (2 ** (len(self.kernels) - 2 - i))
            )
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2))
            if i != len(self.kernels) - 1:
                layers.append(self.act_fn)

        self.convtranspose = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Independent:
        out_batch_shape = features.shape[:-1]
        out = self.dense(features)
        out = torch.reshape(out, [-1, 32 * self.depth, 1, 1])
        out = self.convtranspose(out)
        mean = torch.reshape(out, (*out_batch_shape, *self.output_shape))

        out_dist = Independent(Normal(mean, 1), len(self.output_shape))

        return out_dist


class _TwoHotDistribution:
    """Categorical distribution over symlog-spaced buckets (Dreamer V2).

    Stores raw logits and provides ``log_prob`` and ``mode`` / ``mean`` for
    use in the world model and actor-critic losses. Decoding goes through
    :func:`dreamer_utils.symexp` to invert the symlog transform.
    """

    def __init__(
        self, logits: torch.Tensor, num_buckets: int, symlog_range: float
    ) -> None:
        self.logits = logits
        self.num_buckets = int(num_buckets)
        self.symlog_range = float(symlog_range)
        centers = torch.linspace(
            -self.symlog_range, self.symlog_range, self.num_buckets
        )
        self._centers = centers.to(logits.device)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] != self.num_buckets:
            return self._log_prob_symlog(target)
        return torch.log_softmax(self.logits, dim=-1) * target

    def _log_prob_symlog(self, target: torch.Tensor) -> torch.Tensor:
        from simulusion.utils.dreamer_utils import TwoHotEncoder

        encoder = TwoHotEncoder(
            num_buckets=self.num_buckets, symlog_range=self.symlog_range
        ).to(target.device)
        encoded = encoder.encode(target)
        return (torch.log_softmax(self.logits, dim=-1) * encoded).sum(-1)

    @property
    def mean(self) -> torch.Tensor:
        # A property, as on torch distributions: Dreamer reads `dist.mean` for
        # every reward/value head, so a method here made DreamerV2's actor loss
        # receive a bound method instead of a tensor.
        from simulusion.utils.dreamer_utils import symexp

        probs = torch.softmax(self.logits, dim=-1)
        centers = self._centers.to(probs.device)
        # Logits are (..., *output_shape, K); reducing K leaves the head's own
        # output shape, the same as the Gaussian heads' `mean`.
        expectation = (probs * centers).sum(-1)
        return symexp(expectation)


class DenseDecoder(ExportableAgentMixin, nn.Module):
    """MLP decoder for reward/value/discount prediction from latent features.

    Part of Dreamer's world model, this decoder predicts scalar quantities
    (rewards, values, discount factors) from RSSM latent states.

    - Input: [stoch_state, deter_state] concatenated, shape (B, stoch+deter)
    - Process: MLP with configurable layers and hidden units
    - Output: Predicted quantity with distribution (normal, binary, or raw)

    Supports three output types:
    - ``'normal'``: Gaussian distribution for regression (rewards, values)
    - ``'binary'``: Bernoulli distribution for binary classification (discount)
    - ``'none'``: Raw tensor for non-probabilistic outputs

    **Usage**::

        reward_decoder = DenseDecoder(
            stoch_size=30,
            deter_size=200,
            output_shape=(1,),
            n_layers=2,
            units=400,
            activation='elu',
            dist='normal'
        )
        reward_dist = reward_decoder(latent_features)
        reward_loss = -reward_dist.log_prob(target_reward)

    **For discount prediction (binary)**::

        discount_decoder = DenseDecoder(
            stoch_size=30,
            deter_size=200,
            output_shape=(1,),
            n_layers=2,
            units=400,
            activation='elu',
            dist='binary'
        )
    """

    def __init__(
        self,
        stoch_size: int,
        deter_size: int,
        output_shape: tuple[int, ...],
        n_layers: int,
        units: int,
        activation: str,
        dist: str,
        num_buckets: int = 255,
        symlog_range: float = 10.0,
    ) -> None:
        super().__init__()

        self.input_size = stoch_size + deter_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.dist = dist
        self.num_buckets = int(num_buckets)
        self.symlog_range = float(symlog_range)

        layers: list[nn.Module] = []

        for i in range(self.n_layers):
            in_ch = self.input_size if i == 0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        if self.dist == "symlog_twohot":
            out_dim = int(np.prod(self.output_shape)) * self.num_buckets
        else:
            out_dim = int(np.prod(self.output_shape))

        layers.append(nn.Linear(self.units, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Any:
        out = self.model(features)

        if self.dist == "normal":
            return Independent(Normal(out, 1), len(self.output_shape))
        if self.dist == "binary":
            return Independent(Bernoulli(logits=out), len(self.output_shape))
        if self.dist == "none":
            return out
        if self.dist == "symlog_twohot":
            logits = out.reshape(
                *out.shape[:-1], int(np.prod(self.output_shape)), self.num_buckets
            )
            return _TwoHotDistribution(logits, self.num_buckets, self.symlog_range)
        raise NotImplementedError(self.dist)

        raise NotImplementedError(self.dist)


class SampleDist:
    """Distribution wrapper that estimates statistics via Monte Carlo sampling.

    Provides approximated `mean`, `mode`, and `entropy` helpers for transformed
    distributions where analytic forms may be inconvenient.
    """

    def __init__(self, dist: Any, samples: int = 100) -> None:
        self._dist = dist
        self._samples = samples

    @property
    def name(self) -> str:
        return "SampleDist"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)

    def mean(self) -> torch.Tensor:
        sample = self._dist.rsample(self._samples)
        return torch.mean(sample, 0)

    def mode(self) -> torch.Tensor:
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self) -> torch.Tensor:
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self) -> torch.Tensor:
        return self._dist.sample()


class ActionDecoder(ExportableAgentMixin, nn.Module):
    """Dreamer actor head producing squashed continuous actions from latent features.

    Outputs a transformed Gaussian policy with optional deterministic mode and
    utility for additive exploration noise.
    """

    def __init__(
        self,
        action_size: int,
        stoch_size: int,
        deter_size: int,
        n_layers: int,
        units: int,
        activation: str,
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
    ) -> None:
        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.n_layers = n_layers

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        layers: list[nn.Module] = []
        for i in range(self.n_layers):
            in_ch = self.stoch_size + self.deter_size if i == 0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, 2 * self.action_size))
        self.action_model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, deter: bool = False) -> torch.Tensor:
        out = self.action_model(features)
        mean, std = torch.chunk(out, 2, dim=-1)

        raw_init_std = np.log(np.exp(self._init_std) - 1)
        action_mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        action_std = F.softplus(std + raw_init_std) + self._min_std

        raw_dist = Normal(action_mean, action_std)
        transformed = TransformedDistribution(raw_dist, TanhBijector())
        independent = Independent(transformed, 1)
        dist = SampleDist(independent)

        if deter:
            return dist.mode()
        else:
            return dist.rsample()

    def mean_action(self, features: torch.Tensor) -> torch.Tensor:
        """Deterministic action ``tanh(mean)`` of the squashed Gaussian policy.

        ``forward(deter=True)`` estimates the mode by drawing 100 samples and
        keeping the most likely, so it is random and ~100x the cost of one
        sample. This is the usual deployment choice instead: one pass, no
        randomness, and traceable by every exporter. It is not bit-identical to
        the Monte Carlo mode, so evaluation numbers can differ slightly.
        """
        out = self.action_model(features)
        mean, _ = torch.chunk(out, 2, dim=-1)
        return torch.tanh(self._mean_scale * torch.tanh(mean / self._mean_scale))

    def add_exploration(
        self, action: torch.Tensor, action_noise: float = 0.3
    ) -> torch.Tensor:
        return torch.clamp(Normal(action, action_noise).rsample(), -1, 1)
