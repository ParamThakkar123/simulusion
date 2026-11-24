"""Goal reaching: start in a logged state, reach a later one within a budget.

The evaluation of reward-free world models in DINO-WM (Zhou et al., 2025)
and LeWorldModel (Maes et al., 2026, App. F.1). Each trial takes an episode
from the offline dataset, starts the environment in one of its states and
sets as goal the state ``goal_offset`` steps later in the same episode, so
the goal is reachable with the dataset's own dynamics. The agent sees its
frames and the goal frame and has ``budget`` environment steps; a trial
succeeds if the environment reports success at any step. The score is the
success rate.

Any agent works that has

* ``reset(goal)``: start a trial; ``goal`` is ``{"image": (3, H, W) uint8}``;
* ``act(observation)``: the next environment action.

A planner that plans several steps at once keeps the rest of its plan and
returns it one action at a time.

.. code-block:: python

    from simulusion.benchmarks.goal_reaching import run_goal_reaching

    result = run_goal_reaching("pusht", agent, "data/pusht", env_kwargs={"size": 224})
    result.success_rate

The datasets need the true state of every frame (``observations["state"]``),
which :func:`simulusion.data.collect.collect_goal_episodes` records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class GoalReachingSetup:
    """How one environment is evaluated.

    Attributes:
        env: Goal environment name (:data:`simulusion.envs.goal.GOAL_ENVS`).
        goal_offset: Steps between the start state and the goal state.
        budget: Environment steps the agent has.
        trials: Trials per evaluation.
    """

    env: str
    goal_offset: int
    budget: int
    trials: int = 50


#: LeWorldModel's protocol (App. F.1): Two-Room goals 100 steps ahead with
#: 150 steps; the others 25 ahead with 50. 50 trials, as in DINO-WM.
GOAL_REACHING: dict[str, GoalReachingSetup] = {
    "two_room": GoalReachingSetup("two_room", goal_offset=100, budget=150),
    "pusht": GoalReachingSetup("pusht", goal_offset=25, budget=50),
    "cube": GoalReachingSetup("cube", goal_offset=25, budget=50),
    "reacher_goal": GoalReachingSetup("reacher_goal", goal_offset=25, budget=50),
}


class GoalAgent(Protocol):
    def reset(self, goal: Mapping[str, np.ndarray]) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> Any: ...


@dataclass
class GoalTrial:
    """One start-goal pair from the dataset."""

    start_state: np.ndarray
    goal_state: np.ndarray
    episode: int
    start: int


@dataclass
class GoalReachingResult:
    """Outcome of an evaluation.

    Attributes:
        successes: Per trial, whether the goal was reached.
        steps: Per trial, the steps taken (the budget when it failed).
        final_distances: Per trial, the environment's goal distance at the end.
    """

    successes: list[bool] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    final_distances: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of trials that reached the goal."""
        return float(np.mean(self.successes)) if self.successes else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "trials": float(len(self.successes)),
            "mean_steps": float(np.mean(self.steps)) if self.steps else 0.0,
            "mean_final_distance": (
                float(np.mean(self.final_distances)) if self.final_distances else 0.0
            ),
        }


def sample_goal_trials(
    episodes: Sequence[Any],
    *,
    goal_offset: int,
    trials: int,
    seed: int = 0,
) -> list[GoalTrial]:
    """Draw ``trials`` start-goal pairs, ``goal_offset`` steps apart, from ``episodes``.

    Episodes are drawn uniformly among those long enough, then a start
    uniformly among the frames that leave room for the goal.
    """
    rng = np.random.default_rng(seed)
    usable = []
    for i, ep in enumerate(episodes):
        states = getattr(ep, "observations", {}).get("state")
        if states is None:
            raise ValueError(
                f"Episode {i} has no observations['state']; goal reaching needs the "
                "true states (collect with simulusion.data.collect)."
            )
        if len(states) > goal_offset:
            usable.append(i)
    if not usable:
        raise ValueError(f"No episode is longer than goal_offset={goal_offset}.")
    out = []
    for _ in range(int(trials)):
        i = int(rng.choice(usable))
        states = np.asarray(episodes[i].observations["state"], np.float64)
        start = int(rng.integers(0, len(states) - goal_offset))
        out.append(GoalTrial(states[start], states[start + goal_offset], i, start))
    return out


def evaluate_goal_reaching(
    env: Any,
    agent: GoalAgent,
    trials: Sequence[GoalTrial],
    *,
    budget: int,
    progress: Optional[Callable[[int, int, bool], None]] = None,
) -> GoalReachingResult:
    """Run ``agent`` on each trial in ``env``; a trial stops at success or after ``budget`` steps."""
    result = GoalReachingResult()
    for n, trial in enumerate(trials):
        obs = env.reset(
            options={"state": trial.start_state, "goal_state": trial.goal_state}
        )
        agent.reset(env.goal_observation())
        success, steps, distance = (
            False,
            0,
            float(env.goal_distance(env.get_state(), trial.goal_state)),
        )
        for steps in range(1, int(budget) + 1):
            obs, _, done, info = env.step(agent.act(obs))
            distance = float(info["goal_distance"])
            if info["success"]:
                success = True
                break
            if done and info.get("terminated"):
                break
        result.successes.append(success)
        result.steps.append(steps)
        result.final_distances.append(distance)
        if progress is not None:
            progress(n + 1, len(trials), success)
    return result


def run_goal_reaching(
    setup: str | GoalReachingSetup,
    agent: GoalAgent,
    dataset: Any,
    *,
    trials: Optional[int] = None,
    seed: int = 0,
    env: Any = None,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    progress: Optional[Callable[[int, int, bool], None]] = None,
) -> GoalReachingResult:
    """Evaluate ``agent`` with a named protocol on a dataset of that environment.

    Args:
        setup: A key of :data:`GOAL_REACHING` or a :class:`GoalReachingSetup`.
        agent: Has ``reset(goal)`` and ``act(observation)``.
        dataset: An episode folder path, or a sequence of episodes.
        trials: Overrides the protocol's number of trials.
        seed: Seeds the choice of trials.
        env: The environment; by default one is made from ``setup.env``.
        env_kwargs: Arguments for that environment (e.g. ``{"size": 224}``);
            it must match the dataset's settings.
    """
    from simulusion.data.folder import EpisodeFolder
    from simulusion.envs.goal import goal_env_name, make_goal_env

    if isinstance(setup, str):
        setup = GOAL_REACHING[goal_env_name(setup)]
    episodes = (
        EpisodeFolder(Path(dataset))
        if isinstance(dataset, str) or hasattr(dataset, "__fspath__")
        else dataset
    )
    trial_list = sample_goal_trials(
        episodes,
        goal_offset=setup.goal_offset,
        trials=setup.trials if trials is None else trials,
        seed=seed,
    )
    environment = env
    if environment is None:
        environment = make_goal_env(
            setup.env,
            **{"max_episode_steps": None, "seed": seed, **dict(env_kwargs or {})},
        )
    return evaluate_goal_reaching(
        environment, agent, trial_list, budget=setup.budget, progress=progress
    )


class RandomGoalAgent:
    """Uniform random actions: the chance level of a goal-reaching protocol."""

    def __init__(self, action_space: Any, seed: Optional[int] = None) -> None:
        self.space = action_space
        self.rng = np.random.default_rng(seed)

    def reset(self, goal: Mapping[str, np.ndarray]) -> None:
        pass

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.rng.uniform(self.space.low, self.space.high).astype(np.float32)


__all__ = [
    "GOAL_REACHING",
    "GoalAgent",
    "GoalReachingResult",
    "GoalReachingSetup",
    "GoalTrial",
    "RandomGoalAgent",
    "evaluate_goal_reaching",
    "run_goal_reaching",
    "sample_goal_trials",
]
