"""Benchmark suites: every benchmark's environments, defined once for every model.

A benchmark is a property of the environments, not of the agent: Atari100k is
the same 26 games, 400K frames, no sticky actions and an action repeat of 4
whether DreamerV3, IRIS or DIAMOND plays it. This module records that once:

* the task lists (DMC, Atari 57 and 100k, ProcGen, DMLab-30, BSuite, Minecraft),
* each benchmark's protocol as a :class:`Benchmark` (step budget, action repeat
  and the environment options it implies), in :data:`BENCHMARKS`,
* the Atari human and random reference scores and the human-normalized score.

Environment options use the field names the configs share (``env_backend``,
``gym_make_kwargs``, ``dmc_obs_type``, ``time_limit``, ...), so
:meth:`Benchmark.env_fields` drops straight into any config that has them.
Agent settings (replay ratio, model size, parallel environments) belong to the
agent; DreamerV3's live in :mod:`simulusion.configs.dreamer_v3_benchmarks`.

.. code-block:: python

    from simulusion.benchmarks import get_benchmark, human_normalized_score

    atari = get_benchmark("atari100k")
    fields = atari.env_fields("pong")   # env, action_repeat, total_steps, gym options
    hns = human_normalized_score(18.0, "ALE/Pong-v5")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ATARI100K_GAMES",
    "ATARI57_GAMES",
    "ATARI_HUMAN_SCORES",
    "ATARI_RANDOM_SCORES",
    "BENCHMARKS",
    "BSUITE_ENVIRONMENTS",
    "Benchmark",
    "DMC_PROPRIO_TASKS",
    "DMC_VISUAL_TASKS",
    "DMLAB30_LEVELS",
    "MINECRAFT_OBS_KEYS",
    "PROCGEN_GAMES",
    "atari_env_id",
    "atari_game",
    "bsuite_settings",
    "get_benchmark",
    "human_normalized_score",
]


@dataclass(frozen=True)
class Benchmark:
    """One benchmark's protocol: its tasks, budget and environment options.

    Attributes:
        name: Key in :data:`BENCHMARKS`.
        description: One line on what it measures.
        tasks: Every task the benchmark score covers.
        env_steps: Environment frames per run (the unit of ``total_steps``);
            ``None`` when the benchmark fixes episodes instead (BSuite).
        action_repeat: Frames per agent step.
        overrides: Environment options, keyed by the shared config field names.
    """

    name: str
    description: str
    tasks: tuple[str, ...]
    env_steps: int | None
    action_repeat: int
    overrides: dict[str, Any] = field(default_factory=dict)

    def task_id(self, task: str) -> str:
        """The ``env`` value for ``task``.

        Atari names become ``ALE/<Game>-v5``, DMLab-30 names gain their
        ``contributed/dmlab30/`` directory and BSuite names their first setting
        (``"catch"`` -> ``"catch/0"``); other names pass through.
        """
        backend = self.overrides.get("env_backend")
        if backend == "gym" and not task.startswith("ALE/"):
            return atari_env_id(task)
        if backend == "dmlab" and "/" not in task:
            return f"contributed/dmlab30/{task}"
        if backend == "bsuite" and "/" not in task:
            return f"{task}/0"
        return task

    def env_fields(self, task: str | None = None) -> dict[str, Any]:
        """Config fields that put an agent on ``task`` under this protocol.

        ``env``, ``action_repeat``, ``total_steps`` (when the budget is in
        steps) and every environment option; ``task`` defaults to the first.
        Containers are copies, so editing the result never changes the preset.
        """
        task = self.tasks[0] if task is None else task
        fields: dict[str, Any] = {
            "env": self.task_id(task),
            "action_repeat": self.action_repeat,
        }
        if self.env_steps is not None:
            fields["total_steps"] = self.env_steps
        for key, value in self.overrides.items():
            fields[key] = (
                type(value)(value) if isinstance(value, (dict, list)) else value
            )
        return fields


def atari_env_id(game: str) -> str:
    """``"ms_pacman"`` -> ``"ALE/MsPacman-v5"``."""
    return "ALE/" + "".join(part.capitalize() for part in game.split("_")) + "-v5"


def atari_game(game: str) -> str:
    """The snake-case name of an Atari game, however it is written.

    ``"ALE/MsPacman-v5"``, ``"MsPacman-v5"``, ``"MsPacman"`` and
    ``"ms_pacman"`` all give ``"ms_pacman"``.
    """
    name = game.split("/")[-1]
    name = re.sub(r"-v\d+$", "", name)
    if name.lower() != name:
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return name.lower()


DMC_VISUAL_TASKS = (
    "acrobot-swingup",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "cup-catch",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "hopper-hop",
    "hopper-stand",
    "pendulum-swingup",
    "quadruped-run",
    "quadruped-walk",
    "reacher-easy",
    "reacher-hard",
    "walker-run",
    "walker-stand",
    "walker-walk",
)
# Proprioceptive Control drops the two quadruped tasks (paper, Benchmarks).
DMC_PROPRIO_TASKS = tuple(t for t in DMC_VISUAL_TASKS if not t.startswith("quadruped"))

ATARI100K_GAMES = (
    "alien",
    "amidar",
    "assault",
    "asterix",
    "bank_heist",
    "battle_zone",
    "boxing",
    "breakout",
    "chopper_command",
    "crazy_climber",
    "demon_attack",
    "freeway",
    "frostbite",
    "gopher",
    "hero",
    "jamesbond",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "ms_pacman",
    "pong",
    "private_eye",
    "qbert",
    "road_runner",
    "seaquest",
    "up_n_down",
)

ATARI57_GAMES = (
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank_heist",
    "battle_zone",
    "beam_rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "centipede",
    "chopper_command",
    "crazy_climber",
    "defender",
    "demon_attack",
    "double_dunk",
    "enduro",
    "fishing_derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice_hockey",
    "jamesbond",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "montezuma_revenge",
    "ms_pacman",
    "name_this_game",
    "phoenix",
    "pitfall",
    "pong",
    "private_eye",
    "qbert",
    "riverraid",
    "road_runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space_invaders",
    "star_gunner",
    "surround",
    "tennis",
    "time_pilot",
    "tutankham",
    "up_n_down",
    "venture",
    "video_pinball",
    "wizard_of_wor",
    "yars_revenge",
    "zaxxon",
)

PROCGEN_GAMES = (
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
)

# DMLab-30 (Espeholt et al. 2018), level names relative to contributed/dmlab30.
DMLAB30_LEVELS = (
    "rooms_collect_good_objects_train",
    "rooms_exploit_deferred_effects_train",
    "rooms_select_nonmatching_object",
    "rooms_watermaze",
    "rooms_keys_doors_puzzle",
    "language_select_described_object",
    "language_select_located_object",
    "language_execute_random_task",
    "language_answer_quantitative_question",
    "lasertag_one_opponent_small",
    "lasertag_three_opponents_small",
    "lasertag_one_opponent_large",
    "lasertag_three_opponents_large",
    "natlab_fixed_large_map",
    "natlab_varying_map_regrowth",
    "natlab_varying_map_randomized",
    "skymaze_irreversible_path_hard",
    "skymaze_irreversible_path_varied",
    "psychlab_arbitrary_visuomotor_mapping",
    "psychlab_continuous_recognition",
    "psychlab_sequential_comparison",
    "psychlab_visual_search",
    "explore_object_locations_small",
    "explore_object_locations_large",
    "explore_obstructed_goals_small",
    "explore_obstructed_goals_large",
    "explore_goal_locations_small",
    "explore_goal_locations_large",
    "explore_object_rewards_few",
    "explore_object_rewards_many",
)

# BSuite experiments that DreamerV3 reports (the 23 environments of the suite);
# each runs every one of its settings, e.g. "catch/0" .. "catch/19".
BSUITE_ENVIRONMENTS = (
    "bandit",
    "bandit_noise",
    "bandit_scale",
    "cartpole",
    "cartpole_noise",
    "cartpole_scale",
    "cartpole_swingup",
    "catch",
    "catch_noise",
    "catch_scale",
    "deep_sea",
    "deep_sea_stochastic",
    "discounting_chain",
    "memory_len",
    "memory_size",
    "mnist",
    "mnist_noise",
    "mnist_scale",
    "mountain_car",
    "mountain_car_noise",
    "mountain_car_scale",
    "umbrella_distract",
    "umbrella_length",
)

# Minecraft Diamond inputs (paper, Minecraft appendix): the image plus inventory
# counts, their episode maxima, the equipped item and health/hunger/breath.
MINECRAFT_OBS_KEYS = (
    "image",
    "inventory",
    "inventory_max",
    "equipped",
    "health",
    "hunger",
    "breath",
)


def bsuite_settings(environment: str) -> list[str]:
    """Every setting of a BSuite environment, e.g. ``"catch/0"`` .. ``"catch/19"``.

    Requires the ``bsuite`` package. Across the 23 environments there are 468.
    """
    from simulusion.envs.bsuite_env import list_bsuite_settings

    return list_bsuite_settings(environment)


# ALE ids default to a frame skip of 4. Dreamer applies the action repeat, so


# Atari reference scores (Wang et al. 2016), keyed by :func:`atari_game` names.
ATARI_HUMAN_SCORES: dict[str, float] = {
    "alien": 7127.7,
    "amidar": 1719.5,
    "assault": 742.0,
    "asterix": 8503.3,
    "bank_heist": 753.1,
    "battle_zone": 37187.5,
    "boxing": 12.1,
    "breakout": 30.5,
    "chopper_command": 7387.8,
    "crazy_climber": 35829.4,
    "demon_attack": 1971.0,
    "freeway": 29.6,
    "frostbite": 4334.7,
    "gopher": 2412.5,
    "hero": 30826.4,
    "jamesbond": 302.8,
    "kangaroo": 3035.0,
    "krull": 2665.5,
    "kung_fu_master": 22736.3,
    "ms_pacman": 6951.6,
    "pong": 14.6,
    "private_eye": 69571.3,
    "qbert": 13455.0,
    "road_runner": 7845.0,
    "seaquest": 42054.7,
    "up_n_down": 11693.2,
}

ATARI_RANDOM_SCORES: dict[str, float] = {
    "alien": 227.8,
    "amidar": 5.8,
    "assault": 222.4,
    "asterix": 210.0,
    "bank_heist": 14.2,
    "battle_zone": 2360.0,
    "boxing": 0.1,
    "breakout": 1.7,
    "chopper_command": 811.0,
    "crazy_climber": 10780.5,
    "demon_attack": 152.1,
    "freeway": 0.0,
    "frostbite": 65.2,
    "gopher": 257.6,
    "hero": 1027.0,
    "jamesbond": 29.0,
    "kangaroo": 52.0,
    "krull": 1598.0,
    "kung_fu_master": 258.5,
    "ms_pacman": 307.3,
    "pong": -20.7,
    "private_eye": 24.9,
    "qbert": 163.9,
    "road_runner": 11.5,
    "seaquest": 68.4,
    "up_n_down": 533.4,
}


def human_normalized_score(score: float, game: str) -> float:
    """``(score - random) / (human - random)`` for an Atari game.

    ``game`` may be written any way :func:`atari_game` accepts.

    Raises:
        KeyError: ``game`` has no human or random reference score.
    """
    key = atari_game(game)
    if key not in ATARI_HUMAN_SCORES or key not in ATARI_RANDOM_SCORES:
        raise KeyError(
            f"No human/random reference score for '{game}'; known games: "
            f"{', '.join(sorted(ATARI_HUMAN_SCORES))}."
        )
    human, random = ATARI_HUMAN_SCORES[key], ATARI_RANDOM_SCORES[key]
    if human == random:
        return 0.0
    return (score - random) / (human - random)


# ALE ids default to a frame skip of 4. Agents apply the action repeat, so
# the emulator must step single frames or every action would last 16 frames.
_ALE_FRAMES_PER_EPISODE = 108_000  # 30 minutes at 60 fps, the ALE v5 limit.

BENCHMARKS: dict[str, Benchmark] = {
    benchmark.name: benchmark
    for benchmark in (
        Benchmark(
            name="dmc-proprio",
            description="Proprioceptive Control: DMC from joint states, no pixels.",
            tasks=DMC_PROPRIO_TASKS,
            env_steps=500_000,
            action_repeat=2,
            overrides={
                "env_backend": "dmc",
                "dmc_obs_type": "state",
                "time_limit": 1000,
            },
        ),
        Benchmark(
            name="dmc-visual",
            description="Visual Control: DMC from 64x64 pixels.",
            tasks=DMC_VISUAL_TASKS,
            env_steps=1_000_000,
            action_repeat=2,
            overrides={
                "env_backend": "dmc",
                "dmc_obs_type": "pixels",
                "time_limit": 1000,
            },
        ),
        Benchmark(
            name="atari",
            description="Atari 57 with sticky actions and the full action set.",
            tasks=ATARI57_GAMES,
            env_steps=200_000_000,
            action_repeat=4,
            overrides={
                "env_backend": "gym",
                "gym_make_kwargs": {
                    "frameskip": 1,
                    "repeat_action_probability": 0.25,
                    "full_action_space": True,
                },
                "time_limit": _ALE_FRAMES_PER_EPISODE,
            },
        ),
        Benchmark(
            name="atari100k",
            description=(
                "Atari100k: 26 games, 400K frames, environments as originally "
                "introduced (no sticky actions, minimal action set)."
            ),
            tasks=ATARI100K_GAMES,
            env_steps=400_000,
            action_repeat=4,
            overrides={
                "env_backend": "gym",
                "gym_make_kwargs": {
                    "frameskip": 1,
                    "repeat_action_probability": 0.0,
                    "full_action_space": False,
                },
                "time_limit": _ALE_FRAMES_PER_EPISODE,
            },
        ),
        Benchmark(
            name="procgen",
            description="ProcGen: hard difficulty, unlimited level set.",
            tasks=PROCGEN_GAMES,
            env_steps=50_000_000,
            action_repeat=1,
            overrides={
                "env_backend": "procgen",
                "procgen_distribution_mode": "hard",
                "procgen_num_levels": 0,
                # Each game ends its own episodes; this only bounds a stuck one.
                "time_limit": 100_000,
            },
        ),
        Benchmark(
            name="dmlab",
            description="DMLab-30 with an action repeat of 4.",
            tasks=DMLAB30_LEVELS,
            env_steps=100_000_000,
            action_repeat=4,
            overrides={
                "env_backend": "dmlab",
                # The fixed action space of PopArt and R2D2 (15 actions).
                "dmlab_action_set": "popart",
                # Train on *_train levels; evaluate on held-out ones separately.
                "dmlab_mode": "train",
                # The repeat of 4 happens inside Lab, as in the reference;
                # make_env requires the agent's own action_repeat to be 1 here
                # and applies no time limit (levels end their own episodes).
                # Training still counts Lab frames (native_action_repeat).
                "action_repeat": 1,
                "dmlab_action_repeat": 4,
            },
        ),
        Benchmark(
            name="bsuite",
            description="BSuite: 23 environments, a single environment instance.",
            tasks=BSUITE_ENVIRONMENTS,
            env_steps=None,
            action_repeat=1,
            overrides={
                "env_backend": "bsuite",
                # Learn from BSuite's flattened observation vector.
                "bsuite_obs_type": "state",
                "time_limit": 10_000_000,
            },
        ),
        Benchmark(
            name="minecraft",
            description=(
                "Minecraft Diamond: 12 milestone rewards, 36000-step episodes, "
                "100x block breaking speed."
            ),
            tasks=("diamond",),
            env_steps=100_000_000,
            action_repeat=1,
            overrides={
                "env_backend": "minecraft",
                "minecraft_break_speed": 100.0,
                "obs_keys": list(MINECRAFT_OBS_KEYS),
                "time_limit": 36_000,
            },
        ),
    )
}

_ALIASES = {
    "proprio": "dmc-proprio",
    "proprio-control": "dmc-proprio",
    "visual": "dmc-visual",
    "visual-control": "dmc-visual",
    "atari57": "atari",
    "atari-100k": "atari100k",
    "dmlab30": "dmlab",
    "dmlab-30": "dmlab",
    "minecraft-diamond": "minecraft",
    "diamond": "minecraft",
    "behaviour-suite": "bsuite",
}


def _canonical_name(name: str) -> str:
    """The :data:`BENCHMARKS` key for ``name`` or one of its aliases."""
    key = str(name).strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    if key not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark {name!r}. Options: {', '.join(BENCHMARKS)}."
        )
    return key


def get_benchmark(name: str) -> Benchmark:
    """The :class:`Benchmark` called ``name`` (aliases such as ``"dmlab30"`` work)."""
    return BENCHMARKS[_canonical_name(name)]
