"""A uniform, stateful step interface over Simulusion world models.

Deployed world models run as a loop: filter the latest real observation into a
latent state, pick an action from that state, and - for planning or dreaming -
roll the state forward without observations. Each model family spells these
steps differently. A *stepper* gives them one shape:

.. code-block:: python

    state = stepper.init_state(batch_size)
    while running:
        state = stepper.observe(state, obs, prev_action)   # real observation
        action = stepper.act(state)
        step = stepper.imagine(state, action)               # optional
        ...

State is always a flat ``dict[str, Tensor]`` with the batch on dim 0. Keeping it
explicit, rather than hidden inside the model, is what makes the loop batchable
across environments, compilable (the step is a pure function of its inputs),
and exportable: see :meth:`DreamerStepper.step_module`.

Steppers never change a model's numerics: with default options every method
calls the model's own code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from simulusion.export import ExportableAgentMixin

State = Dict[str, torch.Tensor]


@dataclass
class ImagineOutput:
    """Result of one imagined (observation-free) step."""

    state: State
    reward: torch.Tensor
    continue_prob: Optional[torch.Tensor] = None


@runtime_checkable
class WorldModelStepper(Protocol):
    """Protocol implemented by every stepper.

    ``imagine`` and ``decode`` are optional; ``supports_imagination`` says
    whether ``imagine`` is available.
    """

    device: torch.device
    supports_imagination: bool

    def init_state(self, batch_size: int = 1) -> State: ...

    def observe(
        self, state: State, obs: torch.Tensor, prev_action: Optional[torch.Tensor]
    ) -> State: ...

    def act(self, state: State, *, explore: bool = False) -> torch.Tensor: ...


def _dist_mean(dist: Any) -> torch.Tensor:
    if isinstance(dist, torch.Tensor):
        return dist
    mean = dist.mean
    result: torch.Tensor = mean() if callable(mean) else mean
    return result


class DreamerStepper:
    """Stepper for Dreamer (V1) agents.

    Args:
        agent: A :class:`~simulusion.models.dreamer.DreamerAgent` or the inner
            :class:`~simulusion.models.dreamer.Dreamer`.
        action_mode: How ``act(explore=False)`` picks an action.
            ``"mode"`` (default) calls the actor exactly as
            ``Dreamer.act_with_world_model`` does, i.e. a 100-sample Monte Carlo
            mode estimate. ``"mean"`` uses
            :meth:`~simulusion.vision.dreamer_decoder.ActionDecoder.mean_action`:
            deterministic, one pass, and what :meth:`step_module` exports.
    """

    supports_imagination = True

    def __init__(self, agent: Any, *, action_mode: str = "mode") -> None:
        dreamer = getattr(agent, "dreamer", agent)
        for attr in ("rssm", "obs_encoder", "actor", "reward_model"):
            if not hasattr(dreamer, attr):
                raise TypeError(
                    f"DreamerStepper needs a Dreamer agent; {type(agent).__name__} "
                    f"has no {attr!r}."
                )
        if action_mode not in {"mode", "mean"}:
            raise ValueError("action_mode must be 'mode' or 'mean'.")
        self.dreamer = dreamer
        self.action_mode = action_mode
        self.device = torch.device(dreamer.device)
        self.action_size = int(dreamer.action_size)
        self.action_noise = float(getattr(dreamer.args, "action_noise", 0.3))
        self.discount_model = getattr(dreamer, "discount_model", None)

    @staticmethod
    def features(state: State) -> torch.Tensor:
        return torch.cat([state["stoch"], state["deter"]], dim=-1)

    def init_state(self, batch_size: int = 1) -> State:
        state: State = self.dreamer.rssm.init_state(batch_size, self.device)
        return state

    def observe(
        self,
        state: State,
        obs: torch.Tensor,
        prev_action: Optional[torch.Tensor],
        *,
        noise: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> State:
        """Filter a raw ``[0, 255]`` image batch (B, C, H, W) into the state."""
        from simulusion.models.dreamer import preprocess_obs

        if prev_action is None:
            prev_action = torch.zeros(
                obs.shape[0], self.action_size, device=self.device
            )
        embed = self.dreamer.obs_encoder(preprocess_obs(obs.to(self.device)))
        posterior, _ = self.dreamer.rssm.observe_step(
            state, prev_action, embed, noise=noise
        )
        return dict(posterior)

    def act(self, state: State, *, explore: bool = False) -> torch.Tensor:
        features = self.features(state)
        actor = self.dreamer.actor
        if explore:
            action = actor(features, deter=False)
            return actor.add_exploration(action, self.action_noise)
        if self.action_mode == "mean":
            return actor.mean_action(features)
        return actor(features, deter=True)

    def imagine(
        self,
        state: State,
        action: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> ImagineOutput:
        prior = dict(self.dreamer.rssm.imagine_step(state, action, noise=noise))
        features = self.features(prior)
        reward = _dist_mean(self.dreamer.reward_model(features)).squeeze(-1)
        cont = None
        if self.discount_model is not None:
            cont = _dist_mean(self.discount_model(features)).squeeze(-1)
        return ImagineOutput(state=prior, reward=reward, continue_prob=cont)

    def decode(self, state: State) -> torch.Tensor:
        """Reconstructed observation, in Dreamer's ``[-0.5, 0.5]`` space."""
        return _dist_mean(self.dreamer.obs_decoder(self.features(state)))

    def step_module(self) -> "DreamerStepModule":
        """A pure ``nn.Module`` for one observe+act step, ready to export."""
        return DreamerStepModule(self.dreamer)


class DreamerStepModule(ExportableAgentMixin, nn.Module):
    """One Dreamer observe+act step as a pure function of tensors.

    ``forward(deter, stoch, prev_action, obs, prior_noise, posterior_noise)``
    returns ``(deter, stoch, action)``. Sampling noise is an input, so the
    graph contains no RNG: it is exportable by ``torch.export``/ONNX, gives
    identical results eager vs. exported, and the caller controls seeding (pass
    zeros for the posterior mean). The action is the deterministic
    ``tanh(mean)`` (see :meth:`ActionDecoder.mean_action`).
    """

    def __init__(self, dreamer: Any) -> None:
        super().__init__()
        self.rssm = dreamer.rssm
        self.obs_encoder = dreamer.obs_encoder
        self.actor = dreamer.actor

    def forward(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        prev_action: torch.Tensor,
        obs: torch.Tensor,
        prior_noise: torch.Tensor,
        posterior_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.obs_encoder(obs.to(torch.float32) / 255.0 - 0.5)
        prev = {"deter": deter, "stoch": stoch}
        posterior, _ = self.rssm.observe_step(
            prev,
            prev_action,
            embed,
            nonterm=torch.ones((), device=deter.device),
            noise=(prior_noise, posterior_noise),
        )
        features = torch.cat([posterior["stoch"], posterior["deter"]], dim=-1)
        return posterior["deter"], posterior["stoch"], self.actor.mean_action(features)

    def example_inputs(
        self, batch_size: int = 1, obs_shape: Optional[tuple[int, ...]] = None
    ) -> tuple[torch.Tensor, ...]:
        """Zero inputs of the right shapes, e.g. for export or benchmarking."""
        device = next(self.parameters()).device
        rssm = self.rssm
        if obs_shape is None:
            obs_shape = tuple(self.obs_encoder.input_shape)
        return (
            torch.zeros(batch_size, rssm.deter_size, device=device),
            torch.zeros(batch_size, rssm.stoch_size, device=device),
            torch.zeros(batch_size, rssm.action_size, device=device),
            torch.zeros(batch_size, *obs_shape, device=device),
            torch.zeros(batch_size, rssm.stoch_size, device=device),
            torch.zeros(batch_size, rssm.stoch_size, device=device),
        )


class IRISStepper:
    """Stepper for :class:`~simulusion.models.iris_agent.IRISAgent` acting.

    The IRIS policy is a CNN -> LSTM over frames, so ``observe`` advances the
    LSTM on the new frame and caches the action logits, and ``act`` samples
    from them. Imagination in IRIS is a whole-rollout procedure with a KV cache
    and cache rebuilding (paper 2.3); use ``IRISAgent.imagine_rollout`` for it
    rather than a per-step API, so ``supports_imagination`` is False.

    Args:
        agent: The IRIS agent. It is switched to eval mode.
        temperature: Softmax temperature for action sampling; ``0`` with
            ``explore=False`` takes the argmax.
    """

    supports_imagination = False

    def __init__(self, agent: Any, *, temperature: float = 1.0) -> None:
        if not hasattr(agent, "forward_actor_critic"):
            raise TypeError(f"{type(agent).__name__} is not an IRIS agent.")
        agent.eval()
        self.agent = agent
        self.temperature = float(temperature)
        self.device = torch.device(agent.device)

    def init_state(self, batch_size: int = 1) -> State:
        h, c = self.agent._init_lstm_hidden(batch_size)
        return {"h": h, "c": c}

    def observe(
        self, state: State, obs: torch.Tensor, prev_action: Optional[torch.Tensor]
    ) -> State:
        """Advance the policy on a (B, C, H, W) frame batch in ``[0, 1]``."""
        del prev_action  # the IRIS policy conditions on frames only
        logits, values, (h, c) = self.agent.forward_actor_critic(
            obs.to(self.device).unsqueeze(1), hidden=(state["h"], state["c"])
        )
        return {"h": h, "c": c, "logits": logits[:, -1], "value": values[:, -1]}

    def act(self, state: State, *, explore: bool = False) -> torch.Tensor:
        if "logits" not in state:
            raise ValueError("Call observe() before act().")
        logits = state["logits"]
        if not explore and self.temperature == 0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / max(self.temperature, 1e-8), dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


def make_stepper(model: Any, **kwargs: Any) -> WorldModelStepper:
    """Build the stepper matching ``model``'s family."""
    if hasattr(model, "forward_actor_critic"):
        return IRISStepper(model, **kwargs)
    if hasattr(getattr(model, "dreamer", model), "rssm"):
        return DreamerStepper(model, **kwargs)
    raise TypeError(
        f"No stepper for {type(model).__name__}. Implement the WorldModelStepper "
        "protocol (init_state / observe / act) for it."
    )


__all__ = [
    "DreamerStepModule",
    "DreamerStepper",
    "IRISStepper",
    "ImagineOutput",
    "State",
    "WorldModelStepper",
    "make_stepper",
]
