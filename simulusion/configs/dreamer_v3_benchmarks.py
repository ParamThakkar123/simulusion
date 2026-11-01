"""DreamerV3 settings for the benchmarks of the paper (Table 2).

Each :class:`Benchmark` records the paper's protocol for one benchmark: its
tasks, step budget, action repeat, replay ratio and model size, plus the
environment options that follow from the text (sticky actions on Atari, the
hard ProcGen setting with unlimited levels, proprioceptive vs. pixel inputs on
the control suites). :func:`benchmark_config` turns one into a
:class:`~simulusion.configs.dreamer_v3_config.DreamerV3Config`::

    cfg = benchmark_config("dmc-proprio", "walker-walk")
    cfg = benchmark_config("atari100k", "pong", model_size="12m")  # smaller GPU

Every other hyperparameter keeps the paper's fixed defaults.

Step budgets count environment frames, the same unit as ``total_steps``: an
Atari100k run is 400K frames, 100K agent steps after the action repeat of 4.

Two parts of Table 2 are not reproduced. The paper steps 16 environment
instances in parallel (1 for Atari100k and BSuite); this agent steps one, which
the paper reports performance is robust to, and the replay ratio is counted per
policy step either way. Minecraft is not listed because there is no Minecraft
backend for the Dreamer agents.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from simulusion.configs.dreamer_v3_config import DreamerV3Config

__all__ = [
    "BENCHMARKS",
    "Benchmark",
    "atari_env_id",
    "benchmark_config",
]


@dataclass(frozen=True)
class Benchmark:
    """One row of Table 2 and the environment settings it implies."""

    name: str
    description: str
    tasks: tuple[str, ...]
    env_steps: int | None  # None: the benchmark fixes episodes per task (BSuite).
    action_repeat: int
    env_instances: int
    replay_ratio: int
    model_size: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def task_id(self, task: str) -> str:
        """The ``env`` value for ``task``.

        Atari names become ``ALE/<Game>-v5`` and DMLab-30 names gain their
        ``contributed/dmlab30/`` directory; other names pass through.
        """
        backend = self.overrides.get("env_backend")
        if backend == "gym" and not task.startswith("ALE/"):
            return atari_env_id(task)
        if backend == "dmlab" and "/" not in task:
            return f"contributed/dmlab30/{task}"
        return task


def atari_env_id(game: str) -> str:
    """``"ms_pacman"`` -> ``"ALE/MsPacman-v5"``."""
    return "ALE/" + "".join(part.capitalize() for part in game.split("_")) + "-v5"


# Visual Control (Tassa et al. 2018; Hafner et al. 2019): 20 tasks.
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

# ALE ids default to a frame skip of 4. Dreamer applies the action repeat, so
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
            env_instances=16,
            replay_ratio=512,
            model_size="12m",
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
            env_instances=16,
            replay_ratio=512,
            model_size="12m",
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
            env_instances=16,
            replay_ratio=32,
            model_size="200m",
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
            env_instances=1,
            replay_ratio=128,
            model_size="200m",
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
            env_instances=16,
            replay_ratio=64,
            model_size="200m",
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
            env_instances=16,
            replay_ratio=32,
            model_size="200m",
            overrides={
                "env_backend": "dmlab",
                # Repeat in the shared wrapper only, so that total_steps and
                # time_limit count frames as on every other benchmark.
                "dmlab_action_repeat": 1,
                # Levels end their own episodes.
                "time_limit": 10_000_000,
            },
        ),
        Benchmark(
            name="bsuite",
            description="BSuite: 23 environments, a single environment instance.",
            tasks=BSUITE_ENVIRONMENTS,
            env_steps=None,
            action_repeat=1,
            env_instances=1,
            replay_ratio=1024,
            model_size="200m",
            overrides={"env_backend": "bsuite", "time_limit": 10_000_000},
        ),
    )
}


def _canonical(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-")
    aliases = {
        "proprio": "dmc-proprio",
        "proprio-control": "dmc-proprio",
        "visual": "dmc-visual",
        "visual-control": "dmc-visual",
        "atari57": "atari",
        "atari-100k": "atari100k",
        "dmlab30": "dmlab",
        "dmlab-30": "dmlab",
    }
    key = aliases.get(key, key)
    if key not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark {name!r}. Options: {', '.join(BENCHMARKS)}."
        )
    return key


def benchmark_config(
    benchmark: str, task: str | None = None, **overrides: Any
) -> DreamerV3Config:
    """A :class:`DreamerV3Config` following ``benchmark``'s protocol.

    Args:
        benchmark: A key of :data:`BENCHMARKS`, e.g. ``"dmc-proprio"``.
        task: The task to train on; defaults to the benchmark's first task.
            Atari games may be given as ``"pong"`` or ``"ALE/Pong-v5"``.
        **overrides: Any other config fields, applied last. Use them to fit a
            smaller GPU (``model_size="12m"``) or a shorter run
            (``total_steps=...``); the protocol fields are still set.

    Returns:
        The resolved config. ``update_steps`` is derived from the benchmark's
        replay ratio unless ``auto_update_steps=False`` is passed.
    """
    spec = BENCHMARKS[_canonical(benchmark)]
    task = spec.tasks[0] if task is None else task
    fields: dict[str, Any] = {
        "env": spec.task_id(task),
        "action_repeat": spec.action_repeat,
        "replay_ratio": spec.replay_ratio,
        "model_size": spec.model_size,
        **{key: value for key, value in spec.overrides.items()},
    }
    if spec.env_steps is not None:
        fields["total_steps"] = spec.env_steps
    if isinstance(fields.get("gym_make_kwargs"), dict):
        fields["gym_make_kwargs"] = dict(fields["gym_make_kwargs"])
    fields.update(overrides)
    valid = set(DreamerV3Config.__dataclass_fields__)
    unknown = sorted(set(fields) - valid)
    if unknown:
        raise ValueError(f"Invalid config override(s): {', '.join(unknown)}")
    return replace(DreamerV3Config(), **fields)
