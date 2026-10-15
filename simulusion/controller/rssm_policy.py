"""RSSM-based policy for model-predictive control.

This module provides the RSSMPolicy class that implements model-predictive control
using the RSSM (Recurrent State Space Model) latent dynamics model. The policy uses
a Cross-Entropy Method (CEM) for planning actions in latent space.

Reference:
    Ha & Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution.
    https://arxiv.org/abs/1805.11111
"""

from typing import Any

import torch

from simulusion.utils.memory_utils import maybe_compile
from torch.distributions import Normal


class RSSMPolicy:
    """Model-predictive controller using Cross-Entropy Method (CEM) with RSSM.

    Plans actions by optimizing a sequence of future actions in the RSSM's
    latent space. Uses Cross-Entropy Method to refine action sequences based
    on predicted returns.

    The policy uses a Cross-Entropy Method style loop: it samples candidate
    action sequences, rolls them forward in latent space, scores predicted
    returns, and refits a Gaussian proposal to top-performing candidates.

    Algorithm:
        1. Initialize Gaussian distribution over action sequences
        2. Sample N candidate action sequences
        3. Rollout each sequence in RSSM latent space
        4. Score by predicted cumulative rewards
        5. Keep top K candidates, fit Gaussian to them
        6. Repeat for T iterations
        7. Execute first action from best sequence

    Attributes:
        rssm: The RSSM world model.
        N: Number of candidate action sequences to sample.
        K: Number of top candidates to use for updating the proposal.
        T: Number of CEM iterations per planning step.
        H: Planning horizon (number of future steps to consider).
        d: Action dimensionality.
        device: Device to run computations on.
        state_size: Hidden state dimensionality.
        latent_size: Latent state dimensionality.

    Example:
        >>> policy = RSSMPolicy(
        ...     model=rssm,
        ...     planning_horizon=12,
        ...     num_candidates=1000,
        ...     num_iterations=5,
        ...     top_candidates=100,
        ...     device='cuda'
        ... )
        >>> policy.reset()
        >>> action = policy.poll(observation)
    """

    def __init__(
        self,
        model: Any,
        planning_horizon: int,
        num_candidates: int,
        num_iterations: int,
        top_candidates: int,
        device: torch.device | str,
        compile_rollout: bool = False,
    ) -> None:
        """Initialize the RSSM policy.

        Args:
            model: The RSSM world model.
            planning_horizon: Number of future steps to plan ahead.
            num_candidates: Number of candidate action sequences to sample.
            num_iterations: Number of CEM iterations per planning step.
            top_candidates: Number of top candidates to keep for refitting.
            device: Device to run computations on.
            compile_rollout: Compile the candidate-rollout step. Planning runs
                ``num_iterations * planning_horizon`` tiny kernels per env step,
                so it is launch-bound; compiling collapses that overhead. Off by
                default because compilation costs seconds on first call and
                recompiles whenever ``num_candidates`` changes.
        """
        super().__init__()
        self.rssm = model
        self.N = num_candidates
        self.K = top_candidates
        self.T = num_iterations
        self.H = planning_horizon
        self.d = self.rssm.action_size
        self.device = device
        self.state_size = self.rssm.state_size
        self.latent_size = self.rssm.latent_size
        # A plain function, not a module: compiling a module would rewrite
        # state_dict keys, and this one is shared with the trainer.
        self._rollout_step = maybe_compile(
            self._rollout_step_eager, enabled=compile_rollout
        )

    def _rollout_step_eager(
        self, h_t: torch.Tensor, s_t: torch.Tensor, a_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance every candidate one step and score it."""
        h_t = self.rssm.deterministic_state_fwd(h_t, s_t, a_t)
        s_t = self.rssm.state_prior(h_t, sample=True)
        return h_t, s_t, self.rssm.pred_reward(h_t, s_t).squeeze(-1)

    def reset(self) -> None:
        """Reset the policy state.

        Initializes the hidden state, latent state, and action to zeros.
        Should be called at the beginning of each episode.
        """
        self.h = torch.zeros(1, self.state_size).to(self.device)
        self.s = torch.zeros(1, self.latent_size).to(self.device)
        self.a = torch.zeros(1, self.d).to(self.device)

    def _poll(self, obs: torch.Tensor) -> None:
        """Perform CEM planning to select actions.

        This internal method runs the Cross-Entropy Method optimization
        to find the best action sequence given the current observation.

        Args:
            obs: Current observation tensor of shape (channels, height, width).
        """
        self.mu = torch.zeros(self.H, self.d).to(self.device)
        self.stddev = torch.ones(self.H, self.d).to(self.device)
        assert len(obs.shape) == 3, "obs should be [CHW]"
        self.h, self.s = self.rssm.get_init_state(
            self.rssm.encoder(obs[None]), self.h, self.s, self.a
        )
        for _ in range(self.T):
            rwds = torch.zeros(self.N).to(self.device)
            actions: Any = Normal(self.mu, self.stddev).sample((self.N,))  # type: ignore[no-untyped-call]
            h_t = self.h.expand(self.N, -1)
            s_t = self.s.expand(self.N, -1)
            for a_t in torch.unbind(actions, dim=1):
                h_t, s_t, step_reward = self._rollout_step(h_t, s_t, a_t)
                rwds += step_reward
            _, k = torch.topk(rwds, self.K, dim=0, largest=True, sorted=False)
            self.mu = actions[k].mean(dim=0)
            self.stddev = actions[k].std(dim=0, unbiased=False)
        self.a = self.mu[0:1]

    def poll(self, observation: torch.Tensor, explore: bool = False) -> torch.Tensor:
        """Get action for given observation.

        Args:
            observation: Current observation tensor of shape (channels, height, width).
            explore: If True, add exploration noise to the selected action.

        Returns:
            Action tensor of shape (1, action_size).
        """
        # `inference_mode` over `no_grad`: it additionally skips version-counter
        # and view bookkeeping. Safe here because no tensor produced by planning
        # is ever used in an autograd graph - only `self.a` escapes, as data.
        with torch.inference_mode():
            self._poll(observation)
            if explore:
                self.a += torch.randn_like(self.a) * 0.3
            return self.a
