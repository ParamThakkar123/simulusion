"""Agents that act through any :class:`~simulusion.world_model.WorldModel`.

A world model is trained by one task and used by another: these agents need
nothing but the shared protocol, so they work with every family's world
model, whether it was trained alongside its own agent or alone from a
dataset.
"""

from simulusion.agents.planning import (
    CEMPlanner,
    PlanningAgent,
    expand_state,
    run_episode,
)

__all__ = ["CEMPlanner", "PlanningAgent", "expand_state", "run_episode"]
