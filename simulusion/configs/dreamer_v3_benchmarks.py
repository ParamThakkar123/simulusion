"""DreamerV3 settings for the benchmarks of the paper (Table 2).

The environments of each benchmark (tasks, step budget, action repeat, sticky
actions, observation type, ...) are shared by every model and live in
:mod:`simulusion.benchmarks`. This module adds only what Table 2
fixes for DreamerV3 itself: the replay ratio, the model size and the number of
parallel environment instances. :func:`benchmark_config` combines the two into
a :class:`~simulusion.configs.dreamer_v3_config.DreamerV3Config`::

    cfg = benchmark_config("dmc-proprio", "walker-walk")
    cfg = benchmark_config("atari100k", "pong", model_size="12m")  # smaller GPU

Every other hyperparameter keeps the paper's fixed defaults.

Step budgets count environment frames, the same unit as ``total_steps``: an
Atari100k run is 400K frames, 100K agent steps after the action repeat of 4.

Two parts of Table 2 are not reproduced. The paper steps 16 environment
instances in parallel (1 for Atari100k and BSuite); this agent steps one, which
the paper reports performance is robust to, and the replay ratio is counted per
policy step either way.

BSuite fixes a number of episodes per setting instead of a step budget; the run
stops once :class:`~simulusion.envs.bsuite_env.BSuiteImageEnv` reports the
budget spent. A task such as ``"catch"`` means its first setting, ``"catch/0"``;
the benchmark score covers every setting (:func:`bsuite_settings`).

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any

from simulusion.benchmarks import (  # noqa: F401  (task tables re-exported)
    ATARI57_GAMES,
    ATARI100K_GAMES,
    BSUITE_ENVIRONMENTS,
    DMC_PROPRIO_TASKS,
    DMC_VISUAL_TASKS,
    DMLAB30_LEVELS,
    MINECRAFT_OBS_KEYS,
    PROCGEN_GAMES,
    Benchmark,
    atari_env_id,
    bsuite_settings,
    get_benchmark,
)
from simulusion.configs.dreamer_v3_config import DreamerV3Config

__all__ = [
    "BENCHMARKS",
    "DreamerV3Benchmark",
    "atari_env_id",
    "benchmark_config",
    "bsuite_settings",
]


@dataclass(frozen=True)
class DreamerV3Benchmark(Benchmark):
    """A shared :class:`~simulusion.benchmarks.Benchmark` plus the
    DreamerV3 settings of its Table 2 row.

    ``config`` holds :class:`DreamerV3Config` fields (the replay ratio and model
    size), so they are declared once, on the config. ``env_instances`` is not a
    config field: the agent steps one environment (see the module docstring).
    """

    env_instances: int = 1
    config: Mapping[str, Any] = field(default_factory=dict)


def _check_fields(fields: Mapping[str, Any]) -> None:
    unknown = sorted(set(fields) - set(DreamerV3Config.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Invalid config override(s): {', '.join(unknown)}")


# Table 2: env instances, then the DreamerV3Config fields the row fixes.
_TABLE_2: dict[str, tuple[int, dict[str, Any]]] = {
    "dmc-proprio": (16, {"replay_ratio": 512, "model_size": "12m"}),
    "dmc-visual": (16, {"replay_ratio": 512, "model_size": "12m"}),
    "atari": (16, {"replay_ratio": 32, "model_size": "200m"}),
    "atari100k": (1, {"replay_ratio": 128, "model_size": "200m"}),
    "procgen": (16, {"replay_ratio": 64, "model_size": "200m"}),
    "dmlab": (16, {"replay_ratio": 32, "model_size": "200m"}),
    "bsuite": (1, {"replay_ratio": 1024, "model_size": "200m"}),
    "minecraft": (64, {"replay_ratio": 32, "model_size": "200m"}),
}

for _, _config in _TABLE_2.values():
    _check_fields(_config)

BENCHMARKS: dict[str, DreamerV3Benchmark] = {
    name: DreamerV3Benchmark(
        **asdict(get_benchmark(name)),
        env_instances=instances,
        config=MappingProxyType(config),
    )
    for name, (instances, config) in _TABLE_2.items()
}


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
    spec = BENCHMARKS[get_benchmark(benchmark).name]
    fields: dict[str, Any] = {**spec.config, **spec.env_fields(task), **overrides}
    _check_fields(fields)
    return replace(DreamerV3Config(), **fields)
