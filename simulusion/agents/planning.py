"""Act through any world model by planning in it.

A :class:`~simulusion.world_model.WorldModel` with a reward head is enough
to choose actions: imagine many action sequences, score them by predicted
return, keep the best. :class:`CEMPlanner` does that with the cross-entropy
method, the planner PlaNet uses, but through the shared protocol only, so it
works on every family's world model: one trained with its agent, or alone
from a dataset (``dreamer-wm``, ``iris-wm``, ``diamond-wm``, ``mdrnn``).

.. code-block:: python

    from simulusion.agents import CEMPlanner, PlanningAgent, run_episode
    from simulusion.world_model import load_world_model

    model = load_world_model("runs/dreamer-wm_.../world_model")
    agent = PlanningAgent(model, CEMPlanner(horizon=12, candidates=500))
    episode_return = run_episode(env, agent)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

import numpy as np
import torch

from simulusion.world_model.protocol import State, WorldModel


@dataclass
class CEMPlanner:
    """Cross-entropy-method planning over a world model's imagined returns.

    Each of ``iterations`` rounds samples ``candidates`` action sequences of
    length ``horizon``, imagines them all in one batched rollout, and refits
    the sampling distribution to the ``elites`` with the highest discounted
    predicted return. Continuous actions are Gaussian (clipped to the action
    bounds); discrete ones are a categorical per step. Imagined returns stop
    accumulating once the model predicts the episode ended.

    Attributes:
        horizon: Steps per imagined sequence.
        candidates: Sequences per round.
        iterations: Refitting rounds.
        elites: Best sequences each refit uses.
        discount: Per-step discount of imagined rewards.
        init_std: Initial standard deviation of continuous actions.
        min_std: Floor on the refitted standard deviation.
        smoothing: Weight of the old distribution in each refit (0 = none).
    """

    horizon: int = 12
    candidates: int = 500
    iterations: int = 5
    elites: int = 50
    discount: float = 0.99
    init_std: float = 1.0
    min_std: float = 0.05
    smoothing: float = 0.0

    @torch.no_grad()
    def plan(self, model: WorldModel, state: State) -> torch.Tensor:
        """The first action of the best plan from ``state`` (a batch of one).

        Returns ``(A,)`` float for continuous actions, a scalar int64 for
        discrete ones.
        """
        spec = model.spec
        if not spec.has_reward:
            raise ValueError(
                f"Planning needs predicted rewards; the {spec.family} world model has no reward head."
            )
        if self.elites > self.candidates:
            raise ValueError("elites cannot exceed candidates.")
        device = model.device
        H, N = self.horizon, self.candidates
        batch = expand_state(state, N)
        weights = self.discount ** torch.arange(H, device=device, dtype=torch.float32)
        action = spec.action
        # Continuous plans are Gaussian (mean, std); discrete ones categorical.
        mean = torch.zeros(H, action.n, device=device)
        if action.low is not None and action.high is not None:
            mean += (action.low + action.high) / 2.0
        std = torch.full((H, action.n), self.init_std, device=device)
        probs = torch.full((H, action.n), 1.0 / action.n, device=device)
        for _ in range(self.iterations):
            if action.kind == "continuous":
                samples = mean + std * torch.randn(N, H, action.n, device=device)
                if action.low is not None or action.high is not None:
                    samples = samples.clamp(action.low, action.high)
            else:
                samples = torch.multinomial(probs.repeat(N, 1), 1).view(N, H)
            returns = self._returns(model, batch, samples, weights)
            top = returns.topk(self.elites).indices
            elite = samples[top]
            if action.kind == "continuous":
                new_mean, new_std = elite.mean(0), elite.std(0).clamp_min(self.min_std)
                mean = self.smoothing * mean + (1 - self.smoothing) * new_mean
                std = self.smoothing * std + (1 - self.smoothing) * new_std
            else:
                counts = torch.nn.functional.one_hot(elite, action.n).float().mean(0)
                probs = self.smoothing * probs + (1 - self.smoothing) * counts
        if action.kind == "continuous":
            return mean[0]
        return probs[0].argmax()

    def _returns(
        self,
        model: WorldModel,
        batch: State,
        samples: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        out = model.rollout(batch, samples, decode=False)
        rewards = out.rewards
        assert rewards is not None
        if out.terminated is not None:
            # Rewards after a predicted ending do not count.
            alive = torch.cumprod(
                torch.cat(
                    [torch.ones_like(out.terminated[:, :1]), ~out.terminated[:, :-1]],
                    dim=1,
                ).float(),
                dim=1,
            )
            rewards = rewards * alive
        return (rewards.float() * weights).sum(dim=1)


class PlanningAgent:
    """An agent that keeps recent frames and plans each action in the world model.

    Every :meth:`act` re-encodes the last ``context`` frames and the actions
    between them (the protocol's :meth:`~WorldModel.encode`), then plans
    from that state. Recurrent models (Dreamer, PlaNet, V + M) carry the
    whole history in their state, so a short context is only an
    approximation for them; models with a fixed window (IRIS, DIAMOND,
    Genie) see exactly what they were trained to see.

    Args:
        model: Any world model with a reward head.
        planner: How actions are chosen; :class:`CEMPlanner` by default.
        context: Frames kept; the model's ``max_context`` (or its
            ``context_frames`` when that is unbounded), at least 1.
    """

    def __init__(
        self,
        model: WorldModel,
        planner: Optional[CEMPlanner] = None,
        *,
        context: Optional[int] = None,
    ) -> None:
        self.model = model
        self.planner = planner if planner is not None else CEMPlanner()
        spec = model.spec
        default = spec.max_context or max(spec.context_frames, 8)
        self.context = max(int(context or default), spec.context_frames, 1)
        self.frames: Deque[np.ndarray] = deque(maxlen=self.context)
        self.actions: Deque[Any] = deque(maxlen=self.context - 1)

    def reset(self, observation: Any) -> None:
        """Start an episode from its first observation."""
        self.frames.clear()
        self.actions.clear()
        frame = _frame(observation)
        # Pad with the first frame until there is enough context, taking
        # the "no-op" action 0 between the copies.
        for _ in range(self.model.spec.context_frames):
            self.frames.append(frame)
        for _ in range(len(self.frames) - 1):
            self.actions.append(_zero_action(self.model))

    def act(self) -> np.ndarray:
        """The action to take now."""
        if not self.frames:
            raise RuntimeError("Call reset(first_observation) before act().")
        frames = np.stack(self.frames)[None]
        actions = None
        if len(self.frames) > 1:
            actions = np.stack(list(self.actions)[-(len(self.frames) - 1) :])[None]
        state = self.model.encode(frames, actions)
        return self.planner.plan(self.model, state).cpu().numpy()

    def observe(self, action: Any, observation: Any) -> None:
        """Record the action just taken and the observation it led to."""
        self.actions.append(np.asarray(action))
        self.frames.append(_frame(observation))


def run_episode(env: Any, agent: PlanningAgent, *, max_steps: int = 1000) -> float:
    """Play one episode of a Gymnasium-style ``env`` with ``agent``; returns its return."""
    observation, _ = env.reset()
    agent.reset(observation)
    total = 0.0
    for _ in range(max_steps):
        action = agent.act()
        env_action = (
            int(action) if agent.model.spec.action.kind == "discrete" else action
        )
        observation, reward, terminated, truncated, _ = env.step(env_action)
        total += float(reward)
        if terminated or truncated:
            break
        agent.observe(action, observation)
    return total


def expand_state(state: State, n: int) -> State:
    """Repeat a batch-of-one state ``n`` times along the batch dimension."""
    out: State = {}
    for key, value in state.items():
        if value.shape[0] != 1:
            raise ValueError(
                f"Planning starts from one state; '{key}' has batch {value.shape[0]}."
            )
        out[key] = value.expand(n, *value.shape[1:]).contiguous()
    return out


def _frame(observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        for key in ("image", "pixels", "rgb"):
            if key in observation:
                observation = observation[key]
                break
    return np.asarray(observation)


def _zero_action(model: WorldModel) -> np.ndarray:
    spec = model.spec.action
    if spec.kind == "discrete":
        return np.asarray(0)
    return np.zeros(spec.n, dtype=np.float32)


__all__ = ["CEMPlanner", "PlanningAgent", "expand_state", "run_episode"]
