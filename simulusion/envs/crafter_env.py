"""Crafter adapter for Simulusion image-based agents.

Crafter (Hafner, 2021) is an open-world survival game on a 64x64 grid, built to
measure a broad set of agent abilities in one environment. It is a headline
benchmark of DreamerV3 (Hafner et al., 2023), trained for 1M environment steps.

The benchmark score is not the episode return. Each of Crafter's 22
achievements gets a success rate, the percentage of episodes in which it was
unlocked at least once, and the score is their geometric mean,
``exp(mean(ln(1 + s_i))) - 1``. :func:`crafter_score` computes it from the
``achievements`` info dicts this adapter returns at the end of each episode.

The upstream ``crafter`` package uses the classic Gym step API. The adapter
presents Simulusion's image contract instead: ``{"image": (3, H, W) uint8}``
observations, one-hot-like actions over the 17 discrete actions, and the
``terminated`` / ``truncated`` / ``discount`` info keys. Crafter reports death
with discount 0; the 10000-step time limit keeps discount 1, so it is a
truncation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from simulusion.envs._actions import encode_discrete_action
from simulusion.envs._contract import finalize_step_info
from simulusion.envs.brax_env import _require_module
from simulusion.envs._images import to_chw_uint8, to_hwc_uint8

#: The 22 Crafter achievements, in the order of ``crafter.constants``.
CRAFTER_ACHIEVEMENTS = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)

#: Crafter task names: ``reward`` gives the achievement reward signal,
#: ``noreward`` the same environment with zero reward (unsupervised setting).
CRAFTER_TASKS = ("reward", "noreward")

_TASK_ALIASES = {
    "reward": "reward",
    "crafter": "reward",
    "crafter_reward": "reward",
    "crafter-reward": "reward",
    "crafterreward-v1": "reward",
    "noreward": "noreward",
    "crafter_noreward": "noreward",
    "crafter-noreward": "noreward",
    "crafternoreward-v1": "noreward",
}


def normalize_crafter_task(task: str) -> str:
    """Map a Crafter task name to ``"reward"`` or ``"noreward"``.

    Accepts ``"reward"``, ``"crafter_reward"`` (DreamerV3's naming),
    ``"CrafterReward-v1"`` (the Gym id), and the ``noreward`` equivalents.
    """
    key = str(task).strip().lower()
    if key not in _TASK_ALIASES:
        raise ValueError(
            f"Unknown Crafter task {task!r}. Use 'reward' or 'noreward' "
            "(also accepted: 'crafter_reward', 'CrafterReward-v1', ...)."
        )
    return _TASK_ALIASES[key]


def _require_crafter() -> Any:
    return _require_module(
        "crafter",
        "Install it with `pip install simulusion[crafter]` or `pip install crafter`.",
    )


def crafter_success_rates(
    episode_achievements: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Percentage of episodes in which each achievement was unlocked.

    Args:
        episode_achievements: One mapping per finished episode, from achievement
            name to how often it was unlocked in that episode. This is the
            ``info["achievements"]`` dict of an episode's last step.
    """
    episodes = list(episode_achievements)
    if not episodes:
        raise ValueError("crafter_success_rates needs at least one episode.")
    rates = {}
    for name in CRAFTER_ACHIEVEMENTS:
        unlocked = sum(1 for counts in episodes if float(counts.get(name, 0)) > 0)
        rates[name] = 100.0 * unlocked / len(episodes)
    return rates


def crafter_score(episode_achievements: Iterable[Mapping[str, Any]]) -> float:
    """The Crafter score of a set of episodes.

    Args:
        episode_achievements: The ``info["achievements"]`` dict of each
            finished episode's last step.

    See :func:`crafter_score_from_rates` for the formula.
    """
    return crafter_score_from_rates(crafter_success_rates(episode_achievements))


def crafter_score_from_rates(success_rates: Mapping[str, float]) -> float:
    """The Crafter score: geometric mean of the achievement success rates.

    ``score = exp(mean_i ln(1 + s_i)) - 1`` with each ``s_i`` the success rate of
    achievement ``i`` as a percentage in ``[0, 100]`` (Hafner, 2021).
    """
    missing = [name for name in CRAFTER_ACHIEVEMENTS if name not in success_rates]
    if missing:
        raise ValueError(
            "Success rates must cover all 22 achievements; missing "
            + ", ".join(missing)
        )
    values = np.array(
        [float(success_rates[name]) for name in CRAFTER_ACHIEVEMENTS], np.float64
    )
    if (values < 0).any() or (values > 100).any():
        raise ValueError("Success rates are percentages and must lie in [0, 100].")
    return float(np.exp(np.mean(np.log1p(values))) - 1.0)


class CrafterEnv:
    """Crafter with Simulusion's image observation and one-hot action contract.

    Args:
        task: ``"reward"`` (default) or ``"noreward"``; see
            :func:`normalize_crafter_task` for accepted spellings.
        seed: World-generation seed. Each reset draws a new world from it.
        size: Output image size ``(height, width)``. Crafter renders at this
            size directly; DreamerV3 uses the default 64x64.
        area: World size in cells.
        view: Agent view in cells ``(height, width)``; Crafter's default 9x9
            includes the inventory rows.
        length: Episode length limit in steps (10000 in the benchmark). Pass
            ``None`` for no limit.

    The step info carries Crafter's own keys (``achievements``, ``inventory``,
    ``player_pos``, ``semantic``) next to the contract keys. ``achievements``
    counts unlocks so far in the episode, so the dict at the final step is what
    :func:`crafter_score` consumes.
    """

    def __init__(
        self,
        task: str = "reward",
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        area: tuple[int, int] = (64, 64),
        view: tuple[int, int] = (9, 9),
        length: int | None = 10000,
    ) -> None:
        crafter = _require_crafter()
        from simulusion.envs._spaces import OneHotActionSpace, image_space
        from simulusion.utils.gym_compat import gym

        self.task = normalize_crafter_task(task)
        self._seed = int(seed)
        self._size = (int(size[0]), int(size[1]))
        self._env = crafter.Env(
            area=tuple(area),
            view=tuple(view),
            size=self._size,
            reward=self.task == "reward",
            length=length,
            seed=self._seed,
        )
        self._num_actions = int(self._env.action_space.n)
        self._action_space = OneHotActionSpace(self._num_actions, seed=self._seed)
        self._observation_space = gym.spaces.Dict({"image": image_space(self._size)})
        self._last_frame: np.ndarray | None = None

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def action_names(self) -> list[str]:
        """Names of the discrete actions, indexed like the one-hot vector."""
        return list(_require_crafter().constants.actions)

    def _observe(self, frame: Any) -> dict[str, np.ndarray]:
        self._last_frame = to_hwc_uint8(frame)
        return {"image": to_chw_uint8(self._last_frame, self._size)}

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._seed = int(seed)
            # Crafter draws each world from its own RandomState; reseeding it
            # makes the next reset reproducible.
            self._env._seed = self._seed  # noqa: SLF001 - no public setter
            self._env._episode = 0  # noqa: SLF001
            self._action_space.seed(self._seed)
        return self._observe(self._env.reset())

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        index, model_action = encode_discrete_action(action, self._num_actions)
        frame, reward, done, info = self._env.step(index)
        done = bool(done)
        discount = float(info.get("discount", 0.0 if done else 1.0))
        # Death sets discount 0; the time limit ends with discount 1.
        terminated = done and discount == 0.0
        info = finalize_step_info(
            dict(info),
            done=done,
            terminated=terminated,
            truncated=done and not terminated,
            discount=np.array(discount, np.float32),
        )
        info["action"] = model_action
        info["executed_action"] = index
        return self._observe(frame), float(reward), done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if self._last_frame is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._last_frame.copy()

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


def make_crafter_env(env_id: str = "reward", **kwargs: Any) -> CrafterEnv:
    """Create a Crafter environment, e.g. ``make_crafter_env("reward", seed=0)``."""
    return CrafterEnv(env_id, **kwargs)


__all__ = [
    "CRAFTER_ACHIEVEMENTS",
    "CRAFTER_TASKS",
    "CrafterEnv",
    "crafter_score",
    "crafter_score_from_rates",
    "crafter_success_rates",
    "make_crafter_env",
    "normalize_crafter_task",
]
