"""Collect offline episodes in a goal environment with a scripted policy.

.. code-block:: python

    from simulusion.data.collect import collect_goal_episodes

    collect_goal_episodes("pusht", "data/pusht", episodes=1000, length=100,
                          env_kwargs={"size": 224})

or ``simulusion datasets collect pusht data/pusht --episodes 1000 --length
100 --size 224``. Each episode has ``length`` frames, ``length - 1`` actions
and the true state of every frame as ``observations["state"]``, which the
goal-reaching benchmark uses to start the environment in a logged state and
to set goals (:mod:`simulusion.benchmarks.goal_reaching`). The folder reads
like any other: every training task takes ``data.path=OUT``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from simulusion.data.episode import Episode
from simulusion.data.folder import DatasetMeta, EpisodeWriter


def collect_goal_episodes(
    env: Any,
    out: str | Path,
    *,
    episodes: int,
    length: int,
    policy: Any = None,
    seed: int = 0,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Run ``policy`` in ``env`` for ``episodes`` episodes and write them to ``out``.

    Args:
        env: A :class:`~simulusion.envs.goal.GoalEnv`, or the name of one
            (``"two_room"``, ``"pusht"``, ``"cube"``, ``"reacher_goal"``).
        out: The episode folder; episodes are appended after any already there.
        episodes: Number of episodes.
        length: Frames per episode (``length - 1`` actions); fewer when the
            policy ends an episode early (:attr:`GoalPolicy.done`).
        policy: A policy from :mod:`simulusion.envs.goal_policies`, its name
            (``"random"``, ``"von_mises"``, ...), or ``None`` for the
            environment's default.
        seed: Seeds the environment (episode ``i`` resets with ``seed + i``)
            and the policy.
        env_kwargs: Arguments for the environment when ``env`` is a name.
        progress: Called as ``progress(done, episodes)`` after each episode.

    Returns:
        The folder.
    """
    from simulusion.envs import goal_policies
    from simulusion.envs.goal import GoalEnv, goal_env_name, make_goal_env

    if length < 2:
        raise ValueError("Episodes need at least 2 frames.")
    name: Optional[str] = None
    if isinstance(env, GoalEnv):
        environment = env
    else:
        name = goal_env_name(str(env))
        kwargs = {"max_episode_steps": None, "seed": seed, **dict(env_kwargs or {})}
        environment = make_goal_env(name, **kwargs)
    if policy is None:
        if name is None:
            name = next(
                (k for k, v in _goal_classes().items() if isinstance(environment, v)),
                None,
            )
        policy_cls = goal_policies.DEFAULT_POLICIES.get(
            name or "", goal_policies.RandomPolicy
        )
        policy = policy_cls(seed=seed)
    elif isinstance(policy, str):
        if policy not in goal_policies.POLICIES:
            raise ValueError(
                f"Unknown policy {policy!r}. Choose from {sorted(goal_policies.POLICIES)}."
            )
        policy = goal_policies.POLICIES[policy](seed=seed)

    space = environment.action_space
    meta = DatasetMeta(
        action_kind="continuous",
        action_dim=int(np.prod(space.shape)),
        action_low=np.asarray(space.low, np.float64).reshape(-1).tolist(),
        action_high=np.asarray(space.high, np.float64).reshape(-1).tolist(),
        observation_keys=["state"],
        source=f"simulusion.collect:{name or type(environment).__name__}",
        info={
            "policy": type(policy).__name__,
            "length": int(length),
            "seed": int(seed),
        },
    )
    with EpisodeWriter(out, meta=meta) as writer:
        for i in range(int(episodes)):
            obs = environment.reset(seed=seed + i)
            policy.reset(environment)
            frames = [environment.render()]
            states = [environment.get_state()]
            actions = []
            for _ in range(int(length) - 1):
                action = np.asarray(policy(environment, obs), np.float32)
                obs, _, _, info = environment.step(action)
                actions.append(np.asarray(info["action"], np.float32))
                frames.append(environment.render())
                states.append(environment.get_state())
                if getattr(policy, "done", False):
                    break
            writer.write(
                Episode(
                    frames=np.stack(frames),
                    actions=np.stack(actions),
                    observations={"state": np.stack(states).astype(np.float32)},
                )
            )
            if progress is not None:
                progress(i + 1, int(episodes))
    return Path(out)


def _goal_classes() -> dict[str, type]:
    import importlib

    from simulusion.envs.goal import GOAL_ENVS

    classes = {}
    for key, (module, cls) in GOAL_ENVS.items():
        try:
            classes[key] = getattr(importlib.import_module(module), cls)
        except ImportError:
            continue
    return classes


__all__ = ["collect_goal_episodes"]
