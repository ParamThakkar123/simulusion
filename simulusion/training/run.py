"""Run any training task from one YAML file.

A run config names a task and fills its sections:

.. code-block:: yaml

    task: genie
    trainer:            # TrainerConfig: budget, precision, logging, checkpoints
      max_steps: 20000
      precision: bf16
    model:              # the family's own config (GenieConfig here)
      num_frames: 16
    data:               # where batches come from
      path: data/clips

Train it from the command line, with ``key=value`` overrides for any field
(nested keys use dots)::

    simulusion train --config genie.yaml trainer.max_steps=500 model.dynamics_depth=4
    simulusion train genie data.path=data/clips          # defaults for the rest

or from Python with :func:`run`. Every run writes its fully resolved config to
``<run_dir>/config.yaml``; passing that file back with ``--config`` repeats
the run. Example configs for each task live in ``simulusion/configs/runs/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from simulusion.configs.structured import ConfigError, from_plain, merge, to_plain
from simulusion.experiments.config import dotlist_to_dict
from simulusion.training.task import TrainingTask, get_task, list_tasks
from simulusion.training.trainer import TrainResult, Trainer, TrainerConfig

#: Shipped example run configs.
RUN_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "runs"


def run_config_dict(
    task: str, trainer: TrainerConfig, task_config: Any
) -> dict[str, Any]:
    """The YAML-ready form of a run: ``task``, ``trainer`` and the task's sections."""
    sections = to_plain(task_config)
    if not isinstance(sections, dict):
        raise TypeError("A task config must be a dataclass.")
    return {"task": task, "trainer": to_plain(trainer), **sections}


def dump_run_config(config: Mapping[str, Any]) -> str:
    header = (
        "# Resolved Simulusion run config. Repeat this run with:\n"
        "#   simulusion train --config <this file>\n"
    )
    return header + yaml.safe_dump(dict(config), sort_keys=False)


def resolve_run_config(
    config: str | Path | Mapping[str, Any] | None = None,
    overrides: Sequence[str] | Mapping[str, Any] = (),
    *,
    task: Optional[str] = None,
) -> tuple[type[TrainingTask], TrainerConfig, Any, dict[str, Any]]:
    """Compose defaults, a YAML file (or mapping) and overrides into a run.

    Returns ``(task_class, trainer_config, task_config, resolved_dict)``.
    Precedence, lowest first: the task's defaults, ``config``, ``overrides``.
    An explicit ``task`` wins over one named in ``config`` or ``overrides``.
    """
    values = _load(config)
    extra = (
        dict(overrides)
        if isinstance(overrides, Mapping)
        else dotlist_to_dict(list(overrides))
    )
    values = merge(values, extra)
    name = task or values.get("task")
    if not name:
        raise ConfigError(
            "The run config names no task. Add `task: <name>` or pass one of: "
            + ", ".join(list_tasks())
        )
    task_cls = get_task(str(name))
    values["task"] = task_cls.name
    defaults = run_config_dict(task_cls.name, TrainerConfig(), task_cls.config_cls())
    resolved = merge(defaults, values)
    trainer_config = from_plain(TrainerConfig, resolved.get("trainer"), "trainer.")
    sections = {k: v for k, v in resolved.items() if k not in ("task", "trainer")}
    task_config: Any = from_plain(task_cls.config_cls, sections)
    return (
        task_cls,
        trainer_config,
        task_config,
        run_config_dict(task_cls.name, trainer_config, task_config),
    )


def run(
    config: str | Path | Mapping[str, Any] | None = None,
    overrides: Sequence[str] | Mapping[str, Any] = (),
    *,
    task: Optional[str] = None,
) -> TrainResult:
    """Train the task a run config describes; see the module docstring."""
    task_cls, trainer_config, task_config, resolved = resolve_run_config(
        config, overrides, task=task
    )
    trainer = Trainer(trainer_config, run_config=resolved)
    return trainer.fit(task_cls(task_config))


def _load(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(to_plain(config))
    path = Path(config)
    if not path.exists() and not path.suffix:
        shipped = RUN_CONFIG_DIR / f"{path.name}.yaml"
        if shipped.exists():
            path = shipped
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {config}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must hold a mapping.")
    return loaded


def main(argv: Optional[Sequence[str]] = None) -> Optional[TrainResult]:
    """``simulusion train [TASK] [--config FILE] [--print-config] [key=value ...]``."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        prog="simulusion train",
        description="Train any Simulusion task from a run config.",
        epilog="Tasks: " + ", ".join(list_tasks()),
    )
    parser.add_argument(
        "task", nargs="?", help="Task name (or set `task:` in the config)."
    )
    parser.add_argument(
        "--config", "-c", help="Run config YAML, or a shipped config name."
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit without training.",
    )
    parser.add_argument("overrides", nargs="*", help="key=value overrides.")
    args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    overrides = list(args.overrides) + list(unknown)
    task = args.task
    if task is not None and "=" in task:
        overrides.insert(0, task)
        task = None

    try:
        task_cls, trainer_config, task_config, resolved = resolve_run_config(
            args.config, overrides, task=task
        )
    except (KeyError, FileNotFoundError) as exc:
        # Unknown task names and missing config files are config mistakes;
        # report them as such rather than as a traceback.
        raise ConfigError(str(exc.args[0] if exc.args else exc)) from exc
    if args.print_config:
        sys.stdout.write(dump_run_config(resolved))
        return None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    trainer = Trainer(trainer_config, run_config=resolved)
    result = trainer.fit(task_cls(task_config))
    if trainer.is_main_process:
        print(f"Run directory: {result.run_dir}")
        if result.world_model_path is not None:
            print(f"World model:   {result.world_model_path}")
        if result.model_path is not None:
            print(f"Model:         {result.model_path}")
    return result


#: ``simulusion.run_training``: the same as :func:`run`.
run_training = run

__all__ = [
    "RUN_CONFIG_DIR",
    "run_training",
    "dump_run_config",
    "main",
    "resolve_run_config",
    "run",
    "run_config_dict",
]


if __name__ == "__main__":
    import simulusion.training.tasks  # noqa: F401  (registers the built-in tasks)

    main()
