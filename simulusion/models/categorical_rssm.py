"""Recurrent State-Space Model with categorical latents (DreamerV3).

Where DreamerV1/V2's :class:`~simulusion.models.dreamer_rssm.RSSM` parametrizes
the stochastic state as a diagonal Gaussian, DreamerV3 uses a vector of
independent categorical variables sampled with straight-through gradients. The
model state is ``s_t = {h_t, z_t}``:

.. code-block:: text

    Sequence model:      h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    Encoder (posterior): z_t ~ q(z_t | h_t, x_t)
    Dynamics (prior):    z_t ~ p(z_t | h_t)

Two details matter for stability and are implemented here:

* **Unimix.** Every categorical is a mixture of 99% network output and 1%
  uniform. This makes it impossible for a distribution to become deterministic
  and keeps the KL terms well behaved.
* **Straight-through sampling.** The forward pass uses a hard one-hot sample
  while the backward pass sees the soft probabilities, so gradients reach the
  logits.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.distributions as distributions
import torch.nn as nn

from simulusion.layers.block_gru import BlockGRUCell
from simulusion.layers.rms_norm import RMSNorm

__all__ = ["CategoricalRSSM"]

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


class CategoricalRSSM(nn.Module):
    """DreamerV3 world model dynamics over categorical latent states.

    Args:
        action_size: Dimension of the (one-hot or continuous) action vector.
        embed_size: Size of the observation embedding produced by the encoder.
        latent_dim: Number of categorical variables per latent state.
        latent_classes: Number of classes per categorical variable.
        deter_size: Size of the recurrent state ``h``.
        hidden_size: Width of the MLPs inside the model.
        gru_blocks: Number of block-diagonal groups in the sequence model.
        unimix: Fraction of uniform probability mixed into every categorical.
        activation: Activation used by the internal MLPs.
    """

    def __init__(
        self,
        action_size: int,
        embed_size: int,
        latent_dim: int = 32,
        latent_classes: int = 32,
        deter_size: int = 1024,
        hidden_size: int = 256,
        gru_blocks: int = 8,
        unimix: float = 0.01,
        activation: str = "silu",
    ) -> None:
        super().__init__()

        self.action_size = int(action_size)
        self.embed_size = int(embed_size)
        self.latent_dim = int(latent_dim)
        self.latent_classes = int(latent_classes)
        self.deter_size = int(deter_size)
        self.hidden_size = int(hidden_size)
        self.unimix = float(unimix)
        self.stoch_size = self.latent_dim * self.latent_classes
        self.feature_size = self.stoch_size + self.deter_size

        # Sequence model: embed (z, a, h) densely, then advance the block GRU.
        self.gru_input = nn.Sequential(
            nn.Linear(self.stoch_size + self.action_size + self.deter_size, deter_size),
            RMSNorm(deter_size),
            _activation(activation),
        )
        self.gru = BlockGRUCell(self.deter_size, blocks=gru_blocks)

        # Dynamics predictor p(z_t | h_t).
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_size, self.hidden_size),
            RMSNorm(self.hidden_size),
            _activation(activation),
            nn.Linear(self.hidden_size, self.stoch_size),
        )
        # Encoder posterior q(z_t | h_t, x_t).
        self.posterior_net = nn.Sequential(
            nn.Linear(self.deter_size + self.embed_size, self.hidden_size),
            RMSNorm(self.hidden_size),
            _activation(activation),
            nn.Linear(self.hidden_size, self.stoch_size),
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """Return a zero-initialized model state."""
        return dict(
            logit=torch.zeros(
                batch_size, self.latent_dim, self.latent_classes, device=device
            ),
            stoch=torch.zeros(
                batch_size, self.latent_dim, self.latent_classes, device=device
            ),
            deter=torch.zeros(batch_size, self.deter_size, device=device),
        )

    def get_feat(self, state: dict) -> torch.Tensor:
        """Concatenate the flattened stochastic and deterministic state."""
        stoch = state["stoch"]
        flat = stoch.reshape(*stoch.shape[:-2], self.stoch_size)
        return torch.cat([flat, state["deter"]], dim=-1)

    def detach_state(self, state: dict) -> dict:
        return {key: value.detach() for key, value in state.items()}

    def seq_to_batch(self, state: dict) -> dict:
        """Flatten a ``(T, B, ...)`` state into ``(T * B, ...)``."""
        return {
            key: value.reshape(-1, *value.shape[2:]) for key, value in state.items()
        }

    # ------------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------------

    def _apply_unimix(self, logits: torch.Tensor) -> torch.Tensor:
        """Mix the softmax output with a uniform distribution.

        Returns logits of the mixed distribution, so downstream code can keep
        working with logits rather than probabilities.
        """
        if self.unimix <= 0.0:
            return logits
        probs = torch.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / float(self.latent_classes)
        mixed = (1.0 - self.unimix) * probs + self.unimix * uniform
        return torch.log(mixed)

    def get_dist(self, logits: torch.Tensor) -> distributions.Independent:
        """Return the factorized categorical distribution over the latent."""
        base = distributions.OneHotCategoricalStraightThrough(logits=logits)
        return distributions.Independent(base, 1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Draw a straight-through one-hot sample from the latent distribution."""
        probs = torch.softmax(logits, dim=-1)
        index = torch.multinomial(
            probs.reshape(-1, self.latent_classes), num_samples=1
        ).reshape(*probs.shape[:-1])
        sample = torch.nn.functional.one_hot(index, self.latent_classes).to(probs.dtype)
        # Straight-through: hard sample forward, soft probabilities backward.
        return sample + probs - probs.detach()

    def _stats(self, logits: torch.Tensor) -> torch.Tensor:
        return self._apply_unimix(
            logits.reshape(*logits.shape[:-1], self.latent_dim, self.latent_classes)
        )

    # ------------------------------------------------------------------
    # Transition steps
    # ------------------------------------------------------------------

    def imagine_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        is_first: torch.Tensor | None = None,
    ) -> dict:
        """Advance the state one step using only the dynamics predictor.

        Args:
            prev_state: Previous model state.
            prev_action: Previous action, shape ``(B, action_size)``.
            is_first: Optional ``(B, 1)`` mask; where it is 1 the previous state
                and action are zeroed, which resets the model at episode starts.

        Returns:
            The prior state ``{logit, stoch, deter}``.
        """
        prev_stoch = prev_state["stoch"]
        prev_deter = prev_state["deter"]

        if is_first is not None:
            keep = (1.0 - is_first).to(prev_deter.dtype)
            prev_action = prev_action * keep
            prev_stoch = prev_stoch * keep.unsqueeze(-1)
            prev_deter = prev_deter * keep

        flat_stoch = prev_stoch.reshape(*prev_stoch.shape[:-2], self.stoch_size)
        gru_in = self.gru_input(
            torch.cat([flat_stoch, prev_action, prev_deter], dim=-1)
        )
        deter = self.gru(gru_in, prev_deter)

        logit = self._stats(self.prior_net(deter))
        return dict(logit=logit, stoch=self._sample(logit), deter=deter)

    def observe_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor | None = None,
    ) -> Tuple[dict, dict]:
        """Advance the state one step using an observation embedding.

        Returns:
            ``(posterior, prior)``. Both share the deterministic state because
            the sequence model is advanced exactly once per timestep.
        """
        prior = self.imagine_step(prev_state, prev_action, is_first)
        logit = self._stats(
            self.posterior_net(torch.cat([prior["deter"], embed], dim=-1))
        )
        posterior = dict(logit=logit, stoch=self._sample(logit), deter=prior["deter"])
        return posterior, prior

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def observe_rollout(
        self,
        embed: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        init_state: dict,
    ) -> Tuple[dict, dict]:
        """Run the model over a batch of observed sequences.

        Args:
            embed: Observation embeddings, shape ``(T, B, embed_size)``.
            actions: Actions ``a_{t-1}``, shape ``(T, B, action_size)``.
            is_first: Episode-start flags, shape ``(T, B, 1)``.
            init_state: State to start from, typically ``init_state(B, device)``.

        Returns:
            ``(posterior, prior)`` dicts with tensors of shape ``(T, B, ...)``.
        """
        seq_len = embed.shape[0]
        posteriors: list[dict] = []
        priors: list[dict] = []
        state = init_state

        for t in range(seq_len):
            posterior, prior = self.observe_step(
                state, actions[t], embed[t], is_first[t]
            )
            posteriors.append(posterior)
            priors.append(prior)
            state = posterior

        keys = ("logit", "stoch", "deter")
        if not posteriors:
            empty = {key: init_state[key].unsqueeze(0)[:0] for key in keys}
            return empty, empty

        stacked_post = {
            key: torch.stack([item[key] for item in posteriors], dim=0) for key in keys
        }
        stacked_prior = {
            key: torch.stack([item[key] for item in priors], dim=0) for key in keys
        }
        return stacked_post, stacked_prior

    def imagine_rollout(
        self,
        policy: Any,
        init_state: dict,
        horizon: int,
    ) -> Tuple[dict, torch.Tensor]:
        """Roll the dynamics forward under ``policy`` without observations.

        Args:
            policy: Callable mapping features to an action tensor.
            init_state: Starting state, shape ``(B, ...)``.
            horizon: Number of steps to imagine.

        Returns:
            ``(states, actions)`` where ``states`` holds ``horizon + 1`` entries
            (the starting state followed by each imagined state) and ``actions``
            has shape ``(horizon, B, action_size)``. Including the start state
            lets the caller evaluate the policy on the state it acted from.
        """
        keys = ("logit", "stoch", "deter")
        states: list[dict] = [init_state]
        actions: list[torch.Tensor] = []
        state = init_state

        for _ in range(horizon):
            action = policy(self.get_feat(state))
            state = self.imagine_step(state, action)
            actions.append(action)
            states.append(state)

        stacked = {
            key: torch.stack([item[key] for item in states], dim=0) for key in keys
        }
        return stacked, torch.stack(actions, dim=0)

    def kl_divergence(
        self,
        posterior_logit: torch.Tensor,
        prior_logit: torch.Tensor,
    ) -> torch.Tensor:
        """KL between two factorized categorical latents, summed over factors."""
        post = distributions.Independent(
            distributions.OneHotCategorical(logits=posterior_logit), 1
        )
        prior = distributions.Independent(
            distributions.OneHotCategorical(logits=prior_logit), 1
        )
        return distributions.kl.kl_divergence(post, prior)
