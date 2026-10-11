import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from typing import Optional, Tuple
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


class RSSM(ExportableAgentMixin, nn.Module):
    """Recurrent State-Space Model used by Dreamer for latent dynamics learning.

    The RSSM is the core world model component that learns compact representations
    of environment dynamics. It maintains a hybrid state consisting of:

    1. **Deterministic State (h)** -- A recurrent hidden state updated by a GRU,
       capturing sequential/temporal information and deterministic transitions.

    2. **Stochastic State (s)** -- A latent variable representing stochastic,
       multi-modal uncertainty in the environment (e.g., ambiguous observations).

    The model operates in two modes:

    - **Observe Mode** -- Updates states using actual observations from the environment.
      Uses the representation model: p(s_t | h_t, obs_t)
    - **Imagine Mode** -- Predicts future states without observations.
      Uses the transition/prior model: p(s_t | h_t)

    **Architecture**

    - Input: Previous state (h_{t-1}, s_{t-1}) and action a_{t-1}
    - Process: GRU updates deterministic state, MLP computes stochastic prior/posterior
    - Output: Updated state (h_t, s_t) and distributions

    **State Representation**

    - deter (h): GRU hidden state, captures sequential context
    - stoch (s): Stochastic latent, multi-modal uncertainty
    - mean/std: Parameters of the stochastic distribution

    **Usage with DreamerAgent**::

        rssm = RSSM(
            action_size=action_dim,
            stoch_size=30,      # Stochastic state dimension
            deter_size=200,     # Deterministic (GRU) state dimension
            hidden_size=200,    # MLP hidden layer size
            obs_embed_size=256,  # Observation embedding from encoder
            activation='elu'
        )

        # Observe with actual observation
        posterior, prior = rssm.observe_step(prev_state, prev_action, obs_embed)

        # Imagine future without observation
        prior = rssm.imagine_step(current_state, action)

    **Training**

    The RSSM is trained by maximizing the ELBO (Evidence Lower Bound):

    - KL divergence between prior and posterior encourages the prior to
      capture environment dynamics
    - Reconstruction loss from decoder ensures state captures observation info

    Reference:
        Dreamer: Scalable Reinforcement Learning Using World Models
        Hafner et al., 2020 - https://arxiv.org/abs/1912.01603
    """

    def __init__(
        self,
        action_size: int,
        stoch_size: int,
        deter_size: int,
        hidden_size: int,
        obs_embed_size: int,
        activation: str,
    ) -> None:
        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.embedding_size = obs_embed_size

        self.act_fn = _str_to_activation[activation]
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)

        self.fc_state_action = nn.Linear(
            self.stoch_size + self.action_size, self.deter_size
        )
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior = nn.Linear(self.hidden_size, 2 * self.stoch_size)
        self.fc_embed_posterior = nn.Linear(
            self.embedding_size + self.deter_size, self.hidden_size
        )
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2 * self.stoch_size)

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """Initialize RSSM state with zeros.

        Args:
            batch_size: Number of parallel sequences
            device: torch device for tensors

        Returns:
            Dictionary containing zero-initialized state components:
                - mean, std: Stochastic distribution parameters
                - stoch: Stochastic state sample
                - deter: Deterministic GRU hidden state
        """
        return dict(
            mean=torch.zeros(batch_size, self.stoch_size).to(device),
            std=torch.zeros(batch_size, self.stoch_size).to(device),
            stoch=torch.zeros(batch_size, self.stoch_size).to(device),
            deter=torch.zeros(batch_size, self.deter_size).to(device),
        )

    def get_dist(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> distributions.Independent:
        """Create an Independent Normal distribution from mean and std.

        Args:
            mean: Location parameter
            std: Scale parameter

        Returns:
            Independent Normal distribution with given parameters
        """
        distribution: distributions.Independent = distributions.independent.Independent(
            distributions.Normal(mean, std), 1
        )
        return distribution

    def _gru_input(
        self, prev_state: dict, prev_action: torch.Tensor, nonterm: torch.Tensor
    ) -> torch.Tensor:
        """Project [action, stoch] into the GRU input space and apply nonterm.

        Per the Danijar Dreamer reference, the previous stochastic state is
        masked by `nonterm` before being concatenated with the action, and the
        previous deterministic state is also masked by `nonterm` when fed back
        to the GRU. This ensures that the state is reset (rather than
        propagated) at episode boundaries.
        """
        prev_stoch_masked = prev_state["stoch"] * nonterm
        x = torch.cat([prev_action, prev_stoch_masked], dim=-1)
        x = self.act_fn(self.fc_state_action(x))
        return x

    def observe_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        obs_embed: torch.Tensor,
        nonterm: torch.Tensor = torch.tensor(1.0),
        noise: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[dict, dict]:
        """Update state using actual observation (observe mode).

        In observe mode, the RSSM first computes a transition prior from the
        previous state and action, then refines the stochastic state using the
        actual observation embedding to form the posterior.

        Args:
            prev_state: Dictionary with 'deter' (h_{t-1}) and 'stoch' (s_{t-1})
            prev_action: Previous action a_{t-1}, shape (B, action_size)
            obs_embed: Observation embedding from encoder, shape (B, obs_embed_size)
            nonterm: Termination mask (1.0 = continue, 0.0 = terminal)
            noise: Optional ``(prior_noise, posterior_noise)`` standard-normal
                tensors of shape (B, stoch_size) used for the two samples
                instead of drawing them internally. Passing noise makes the step
                a pure function of its inputs, which exported graphs and
                eager-vs-exported parity checks need.

        Returns:
            A tuple ``(posterior, prior)`` of state dictionaries. The posterior
            incorporates observation information; the prior is the transition
            prediction before observation. Both share the same deterministic
            state because the GRU is only advanced once per timestep.

            The order differs from :meth:`observe_rollout` and from
            ``ModularRSSM.observe_step``, which both return
            ``(prior, posterior)``. The dictionaries have identical keys, so a
            swapped unpacking fails silently: acting on the prior discards the
            current observation. It is kept for backward compatibility.
        """
        prior_noise, posterior_noise = (None, None) if noise is None else noise
        prior = self.imagine_step(prev_state, prev_action, nonterm, noise=prior_noise)
        posterior_embed = self.act_fn(
            self.fc_embed_posterior(torch.cat([obs_embed, prior["deter"]], dim=-1))
        )
        posterior = self.fc_state_posterior(posterior_embed)
        mean, std = torch.chunk(posterior, 2, dim=-1)
        std = F.softplus(std) + 0.1
        if posterior_noise is None:
            posterior_noise = torch.randn_like(mean)
        sample = mean + posterior_noise * std
        posterior_state = dict(mean=mean, std=std, stoch=sample, deter=prior["deter"])
        return posterior_state, prior

    def imagine_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        nonterm: torch.Tensor = torch.tensor(1.0),
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """Predict next state without observation (imagine mode).

        In imagine mode, the RSSM predicts future states using only the prior
        distribution. This is used for planning and policy learning where
        actual observations are not available.

        Args:
            prev_state: Dictionary with 'deter' (h_{t-1}) and 'stoch' (s_{t-1})
            prev_action: Previous action a_{t-1}, shape (B, action_size)
            nonterm: Termination mask (1.0 = continue, 0.0 = terminal)
            noise: Optional standard-normal tensor (B, stoch_size) used for the
                stochastic sample instead of drawing one internally.

        Returns:
            Dictionary with predicted state containing:
                - deter: Predicted deterministic state
                - mean, std, stoch: Prior stochastic state distribution
        """
        x = self._gru_input(prev_state, prev_action, nonterm)
        prior_deter = self.rnn(x, prev_state["deter"] * nonterm)
        prior_embed = self.act_fn(self.fc_embed_prior(prior_deter))
        prior = self.fc_state_prior(prior_embed)
        mean, std = torch.chunk(prior, 2, dim=-1)
        std = F.softplus(std) + 0.1
        if noise is None:
            noise = torch.randn_like(mean)
        sample = mean + noise * std
        return dict(mean=mean, std=std, stoch=sample, deter=prior_deter)

    def get_prior(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        nonterm: torch.Tensor = torch.tensor(1.0),
    ) -> dict:
        return self.imagine_step(prev_state, prev_action, nonterm)

    def get_posterior(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        obs_embed: torch.Tensor,
        nonterm: torch.Tensor = torch.tensor(1.0),
    ) -> dict:
        """Compute posterior distribution over stochastic state.

        The posterior incorporates observation information to produce
        a more accurate state estimate.

        Args:
            prev_state: Previous state dictionary
            prev_action: Previous action
            obs_embed: Observation embedding
            nonterm: Termination mask

        Returns:
            Dictionary with posterior state (observation-informed). Note that
            the previous-state shape ``(B, ...)`` is preserved; the batch
            dimension is not flattened.
        """
        posterior, _ = self.observe_step(prev_state, prev_action, obs_embed, nonterm)
        return posterior

    def detach_state(self, state: dict) -> dict:
        """Detach state tensors from computation graph.

        Used during DreamerV2 training to prevent gradient flow through
        the observation/update pathway.

        Args:
            state: State dictionary with tensor values

        Returns:
            Detached state dictionary
        """
        return {k: v.detach() for k, v in state.items()}

    def seq_to_batch(self, state_dict: dict) -> dict:
        """Convert sequence state to batch format.

        Args:
            state_dict: Dictionary with sequence-dimension tensors (T, B, ...)

        Returns:
            Dictionary with batch-dimension tensors (B*T, ...)
        """
        return {k: v.reshape(-1, *v.shape[2:]) for k, v in state_dict.items()}

    def observe_rollout(
        self,
        obs_embed: torch.Tensor,
        actions: torch.Tensor,
        nonterms: torch.Tensor,
        init_state: dict,
        seq_len: int,
    ) -> Tuple[dict, dict]:
        """Process a sequence of observations (observe mode rollout).

        At each timestep we run ``observe_step`` once to obtain the transition
        prior (the prediction given the previous state and action) and the
        observation-informed posterior. The posterior is then used as the
        previous state for the next step, matching the standard Dreamer
        inference pattern.

        Args:
            obs_embed: Observation embeddings, shape (T+1, B, obs_embed_size)
            actions: Actions, shape (T, B, action_size)
            nonterms: Non-termination flags, shape (T, B, 1)
            init_state: Initial state dictionary
            seq_len: Sequence length T

        Returns:
            prior: Dictionary with prior states stacked along the time axis
            posterior: Dictionary with posterior states stacked along the time axis
        """
        prior_states = []
        posterior_states = []
        state = init_state

        for t in range(seq_len):
            posterior, prior = self.observe_step(
                state, actions[t], obs_embed[t], nonterms[t]
            )
            posterior_states.append(posterior)
            prior_states.append(prior)
            state = posterior

        to_stack = ["mean", "std", "stoch", "deter"]

        if not prior_states:
            empty = {k: init_state[k].unsqueeze(0)[:0] for k in to_stack}
            return empty, empty

        prior = {k: torch.stack([p[k] for p in prior_states], dim=0) for k in to_stack}
        posterior = {
            k: torch.stack([p[k] for p in posterior_states], dim=0) for k in to_stack
        }

        return prior, posterior

    def imagine_rollout(
        self, policy: nn.Module, init_state: dict, horizon: int
    ) -> dict:
        """Generate imagined trajectory using policy (imagine mode rollout).

        Args:
            policy: Actor network that outputs actions from state features
            init_state: Initial state dictionary
            horizon: Number of steps to imagine

        Returns:
            Dictionary with imagined states for each step
        """
        states = []
        state = init_state

        for _ in range(horizon):
            features = torch.cat([state["stoch"], state["deter"]], dim=-1)
            action = policy(features, deter=False)
            state = self.imagine_step(state, action)
            states.append(state)

        to_stack = ["mean", "std", "stoch", "deter"]
        return {k: torch.stack([s[k] for s in states], dim=0) for k in to_stack}

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple:
        """Forward pass for training (computes sequence of states).

        Args:
            x: Observations, shape (B, T+1, C, H, W)
            u: Actions, shape (B, T, action_size)

        Returns:
            states: List of state dictionaries for each timestep
            priors: List of prior distributions (tuples of mean, std)
            posteriors: List of posterior distributions (tuples of mean, std)
        """
        B = x.size(0)
        T = u.size(1)

        priors = []
        posteriors = []

        state = self.init_state(B, x.device)

        for t in range(T):
            prior = self.get_prior(state, u[:, t])
            priors.append((prior["mean"], prior["std"]))

            obs_embed = x[:, t + 1].reshape(B, -1)
            posterior = self.get_posterior(state, u[:, t], obs_embed)
            posteriors.append((posterior["mean"], posterior["std"]))

            state = posterior

        states = None
        return states, priors, posteriors
