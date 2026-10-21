"""The one path the older per-family training entrypoints take.

``DreamerAgent.train()``, ``IRISTrainer.train()``, ``DiamondAgent.train()``,
``Planet.train()``, ``GenieTrainer.train()``, ``DiT.fit()``,
``JEPAAgent.train()``, ``train_convae``, ``train_mdn_rnn`` and
``train_controller`` keep their signatures, but each now builds its family's :class:`TrainingTask` around the
object it was called on and runs it with :func:`fit_legacy`, on the shared
:class:`~simulusion.training.Trainer`. There is one training loop to
maintain, and the old calls gain what the trainer gives every run: resumable
checkpoints and a record of the run (``run.json``, ``run_config.yaml``,
``train_metrics.jsonl``).

What every one of them now does the same way:

* The run directory is the folder the entrypoint always wrote to (Dreamer's
  ``logdir``, IRIS's ``save_dir``, DIAMOND's ``checkpoint_dir``, ...). Inside
  it the trainer adds its layout (``checkpoints/step_*``, ``run.json``,
  ``run_config.yaml``, ``train_metrics.jsonl``) next to the family's own
  files, which each wrapper still writes.
* The object keeps its own device and precision (its ``use_amp``), so
  results match what the entrypoint produced before.
* Evaluation, early stopping and "best" checkpoints go through the trainer's
  ``eval_every`` / ``early_stopping_metric`` / ``best_metric``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from simulusion.training.task import TrainingTask
from simulusion.training.trainer import TrainResult, Trainer, TrainerConfig

#: The trainer's run files under the old entrypoints. Families write their own
#: ``config.yaml`` (and Dreamer its ``metrics.jsonl``) in the same folders.
RUN_CONFIG_NAME = "run_config.yaml"
METRICS_NAME = "train_metrics.jsonl"


def fit_legacy(
    task: TrainingTask,
    run_dir: str | Path,
    *,
    max_steps: Optional[int] = None,
    device: Any = None,
    precision: str = "fp32",
    seed: int = 0,
    log_every: int = 1,
    eval_every: int = 0,
    checkpoint_every: int = 0,
    keep_checkpoints: int = 1,
    best_metric: Optional[str] = None,
    best_mode: str = "max",
    early_stopping_metric: Optional[str] = None,
    early_stopping_mode: str = "max",
    patience: int = 10,
    min_delta: float = 1e-4,
    export_world_model: bool = False,
    console: bool = True,
) -> TrainResult:
    """Run ``task`` on the shared Trainer in ``run_dir``; see the module docstring.

    Args:
        task: The family's task, built around an existing agent or trainer.
        run_dir: The folder the entrypoint has always written to.
        max_steps: Budget, in the task's progress unit; the task's own
            default when ``None``.
        device: The object's device; ``cpu`` keeps the trainer on the CPU,
            anything else lets Accelerate pick the accelerator.
        precision: The run's precision, for tasks that take it from the
            trainer (JEPA's ``use_bfloat16``); objects that keep their own
            ``use_amp`` ignore it.
        seed: Recorded in ``run.json``. The object has seeded itself already;
            the trainer seeds again with the same value.
        log_every, eval_every, checkpoint_every, keep_checkpoints: As in
            :class:`~simulusion.training.TrainerConfig`.
        best_metric, best_mode: Keep the best evaluation's checkpoint.
        early_stopping_metric, early_stopping_mode, patience, min_delta:
            Early stopping, as the entrypoint's own flag configured it.
        export_world_model: Also write ``<run_dir>/world_model``.
        console: Print metrics as they are logged.
    """
    run_dir = Path(run_dir)
    is_cpu = device is not None and torch.device(device).type == "cpu"
    config = TrainerConfig(
        max_steps=int(max_steps) if max_steps is not None else None,
        seed=int(seed),
        precision=precision,
        device="cpu" if is_cpu else None,
        log_every=max(1, int(log_every)),
        eval_every=max(0, int(eval_every)),
        checkpoint_every=max(0, int(checkpoint_every)),
        keep_checkpoints=max(1, int(keep_checkpoints)),
        out_dir=str(run_dir.parent),
        run_name=run_dir.name,
        best_metric=best_metric,
        best_mode=best_mode,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
        patience=int(patience),
        min_delta=float(min_delta),
        export_world_model=export_world_model,
        console=console,
    )
    trainer = Trainer(
        config, config_filename=RUN_CONFIG_NAME, metrics_filename=METRICS_NAME
    )
    return trainer.fit(task)


__all__ = ["METRICS_NAME", "RUN_CONFIG_NAME", "fit_legacy"]
