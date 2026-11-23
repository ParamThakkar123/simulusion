"""CUDA graphs for the two sequential loops of a DreamerV3 update.

Both loops of a training update step the recurrent model one time step at a
time: the observe rollout over a replayed ``64 x 16`` batch, and imagination
over ``15`` steps from every replayed state. Each step is ~90 small kernels, so
on a laptop GPU the update is bound by kernel launches and Python overhead, not
by GPU work: a 12M-parameter update took ~2.5 s of which the GPU was busy for a
fraction.

The shapes of both loops are the same every update, so each is captured once as
a CUDA graph and replayed:

* :class:`GraphedObserveRollout` captures the forward *and* backward of the
  observe rollout with :func:`torch.cuda.make_graphed_callables`; gradients flow
  to the observation embedding and to the RSSM parameters exactly as before.
* :class:`GraphedImagination` captures the no-grad imagination rollout.

Parameters are updated in place by the optimizers, so the captured graphs keep
reading the current weights. Random draws inside a graph come from the CUDA
generator's philox state, which advances on every replay, so samples stay fresh.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

__all__ = ["GraphedImagination", "GraphedObserveRollout"]

_STATE_KEYS = ("logit", "stoch", "deter")


class _ObserveRollout(nn.Module):
    """``observe_rollout`` as a module of tensors in, tensors out."""

    def __init__(self, rssm: Any) -> None:
        super().__init__()
        self.rssm = rssm

    def forward(
        self,
        embed: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        deter: torch.Tensor,
        stoch: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        posterior, prior = self.rssm.observe_rollout(
            embed, actions, is_first, {"deter": deter, "stoch": stoch}
        )
        return (
            posterior["logit"],
            posterior["stoch"],
            posterior["deter"],
            prior["logit"],
            prior["stoch"],
        )


class GraphedObserveRollout:
    """Replays the observe rollout's forward and backward as CUDA graphs.

    Args:
        rssm: The :class:`~simulusion.models.categorical_rssm.CategoricalRSSM`.
        embed, actions, is_first, init_state: Example inputs fixing the shapes,
            as passed to ``rssm.observe_rollout``.
    """

    def __init__(
        self,
        rssm: Any,
        embed: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        init_state: dict,
    ) -> None:
        self.shapes = self._shapes(embed, actions, is_first, init_state)
        sample = (
            torch.zeros_like(embed).requires_grad_(True),
            torch.zeros_like(actions),
            torch.zeros_like(is_first),
            torch.zeros_like(init_state["deter"]),
            torch.zeros_like(init_state["stoch"]),
        )
        self._module = _ObserveRollout(rssm)
        self._graphed: Any = torch.cuda.make_graphed_callables(
            self._module, sample, num_warmup_iters=3, allow_unused_input=True
        )

    @staticmethod
    def _shapes(*args: Any) -> tuple:
        embed, actions, is_first, init_state = args
        return (
            tuple(embed.shape),
            tuple(actions.shape),
            tuple(is_first.shape),
            tuple(init_state["deter"].shape),
            tuple(init_state["stoch"].shape),
        )

    def matches(
        self,
        embed: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        init_state: dict,
    ) -> bool:
        """Whether these inputs have the captured shapes and can be replayed."""
        return (
            embed.requires_grad
            and self._shapes(embed, actions, is_first, init_state) == self.shapes
        )

    def __call__(
        self,
        embed: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        init_state: dict,
    ) -> tuple[dict, dict]:
        post_logit, post_stoch, deter, prior_logit, prior_stoch = self._graphed(
            embed,
            actions.detach(),
            is_first.detach(),
            init_state["deter"].detach(),
            init_state["stoch"].detach(),
        )
        posterior = {"logit": post_logit, "stoch": post_stoch, "deter": deter}
        prior = {"logit": prior_logit, "stoch": prior_stoch, "deter": deter}
        return posterior, prior


class GraphedImagination:
    """Replays a no-grad imagination rollout as one CUDA graph.

    Args:
        rssm: The recurrent state-space model.
        policy: Maps features to an action; captured as it is now, so it must
            keep its parameters (updated in place) and code.
        start: Example start state ``{logit, stoch, deter}`` of shape ``(N, ...)``.
        horizon: Number of imagined steps.
    """

    def __init__(
        self,
        rssm: Any,
        policy: Callable[[torch.Tensor], torch.Tensor],
        start: dict,
        horizon: int,
    ) -> None:
        self.horizon = int(horizon)
        self.shapes = {key: tuple(start[key].shape) for key in _STATE_KEYS}
        self._start = {key: start[key].detach().clone() for key in _STATE_KEYS}

        def rollout() -> tuple[dict, torch.Tensor]:
            return rssm.imagine_rollout(policy, self._start, self.horizon)

        # Warm up on a side stream (allocator and library handles), then capture.
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                rollout()
        torch.cuda.current_stream().wait_stream(stream)
        # Capture must not begin while earlier work is still in flight.
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        # Thread-local capture: other threads (autograd's, loggers) may touch
        # CUDA meanwhile without invalidating it.
        with (
            torch.no_grad(),
            torch.cuda.graph(self._graph, capture_error_mode="thread_local"),
        ):
            self._states, self._actions = rollout()

    def matches(self, start: dict, horizon: int) -> bool:
        return int(horizon) == self.horizon and all(
            tuple(start[key].shape) == self.shapes[key] for key in _STATE_KEYS
        )

    def __call__(self, start: dict) -> tuple[dict, torch.Tensor]:
        for key in _STATE_KEYS:
            self._start[key].copy_(start[key])
        self._graph.replay()
        # Outputs live in the graph's memory and are overwritten by the next
        # replay, so hand out copies.
        states = {key: value.clone() for key, value in self._states.items()}
        return states, self._actions.clone()
