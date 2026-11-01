"""The one training loop every Simulusion family runs on.

.. code-block:: python

    from simulusion.training import Trainer, TrainerConfig
    from simulusion.training.tasks import GenieTask, GenieTaskConfig

    result = Trainer(TrainerConfig(max_steps=10_000, precision="bf16")).fit(
        GenieTask(GenieTaskConfig(data=VideoDataConfig(path="data/clips")))
    )
    result.world_model_path  # runs/<name>/world_model, loadable with load_world_model

A run uses one device: a CPU, one GPU, or Apple MPS. Precision is a run
setting (``fp32``, ``bf16``, ``fp16``, ``auto``) applied through
:class:`~simulusion.training.precision.MixedPrecision`, rather than a flag per
model.

Every run gets a directory holding everything needed to reproduce it::

    runs/<run_name>/
        config.yaml          # the fully resolved run config: re-run with --config
        run.json             # git commit, package versions, host, command line
        metrics.jsonl        # every logged metric
        checkpoints/step_000500/   # resumable: trainer.resume=<this dir> or "auto"
        world_model/         # the trained world model (WorldModel protocol)
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, cast

import numpy as np
import torch
import torch.nn as nn

from simulusion.configs.serialization import SerializableConfigMixin
from simulusion.training.precision import GradientUpdate, MixedPrecision
from simulusion.training.task import TrainingTask
from simulusion.utils.device import resolve_device
from simulusion.utils.logging_utils import MetricsLogger
from simulusion.utils.train_utils import EarlyStopping

logger = logging.getLogger("simulusion.trainer")

CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = "best"
_COMPLETE = "checkpoint.json"


@dataclass
class TrainerConfig(SerializableConfigMixin):
    """Settings shared by every training run, whatever the family.

    Attributes:
        max_steps: Budget in the task's progress unit (optimizer steps,
            environment steps or epochs; see the task). ``None`` uses the
            task's default.
        seed: Seeds Python, NumPy and PyTorch.
        deterministic: Ask PyTorch for deterministic kernels. Slower; some
            ops only warn.
        precision: ``fp32``, ``bf16``, ``fp16`` or ``auto``.
        device: ``None`` (or ``"auto"``) picks CUDA, then MPS, else CPU;
            ``"cpu"``, ``"cuda"``, ``"cuda:1"`` or ``"mps"`` choose one. A
            device that is not available falls back with a warning.
        grad_accumulation: Batches per optimizer step, for tasks whose
            updates support it (dataset-driven ones).
        log_every: Log metrics every this many units of progress.
        eval_every: Evaluate every this many units; 0 only at the end.
        checkpoint_every: Checkpoint every this many units; 0 only at the end.
        keep_checkpoints: Periodic checkpoints kept on disk (oldest dropped).
            The final one is always kept.
        out_dir: Parent directory of run directories.
        run_name: Run directory name; a timestamped one when ``None``.
        resume: A checkpoint directory to resume from, or ``"auto"`` for the
            newest complete checkpoint of ``out_dir/run_name``.
        export_world_model: Write the task's world model to
            ``<run>/world_model`` when training ends.
        best_metric: Evaluation metric whose best value so far is kept as a
            checkpoint in ``checkpoints/best`` (rewritten on every
            improvement, never pruned); ``None`` keeps none.
        best_mode: ``"max"`` or ``"min"``: which direction is better.
        early_stopping_metric: Evaluation metric that stops the run once it
            stops improving; ``None`` disables early stopping.
        early_stopping_mode: ``"min"`` or ``"max"``.
        patience: Evaluations without improvement before stopping.
        min_delta: Relative improvement that counts.
        console: Print metrics to the console.
    """

    max_steps: Optional[int] = None
    seed: int = 0
    deterministic: bool = False
    precision: str = "fp32"
    device: Optional[str] = None
    grad_accumulation: int = 1
    log_every: int = 100
    eval_every: int = 0
    checkpoint_every: int = 0
    keep_checkpoints: int = 2
    out_dir: str = "runs"
    run_name: Optional[str] = None
    resume: Optional[str] = None
    export_world_model: bool = True
    best_metric: Optional[str] = None
    best_mode: str = "max"
    early_stopping_metric: Optional[str] = None
    early_stopping_mode: str = "min"
    patience: int = 10
    min_delta: float = 1e-4
    console: bool = True


@dataclass
class TrainResult:
    """What :meth:`Trainer.fit` returns."""

    run_dir: Path
    progress: int
    iterations: int
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint: Optional[Path] = None
    world_model_path: Optional[Path] = None
    model_path: Optional[Path] = None
    best_checkpoint: Optional[Path] = None
    best_value: Optional[float] = None
    stopped_early: bool = False


class Trainer:
    """Runs a :class:`~simulusion.training.task.TrainingTask` to its budget.

    Args:
        config: Run settings; defaults throughout when ``None``.
        run_config: The full run config (all sections) to write as
            ``config.yaml``. :func:`simulusion.training.run.run` passes it;
            without it only the trainer and task sections are written.
        config_filename: Name of the resolved run config in the run directory.
        metrics_filename: Name of the metrics log in the run directory. The
            older entrypoints (:mod:`simulusion.training.legacy`) rename both
            so they never overwrite a family's own ``config.yaml`` or
            ``metrics.jsonl`` in the folder it has always written to.
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        *,
        run_config: Optional[Mapping[str, Any]] = None,
        config_filename: str = "config.yaml",
        metrics_filename: str = "metrics.jsonl",
    ) -> None:
        self.config = config if config is not None else TrainerConfig()
        device = self.config.device
        self.device: torch.device = resolve_device(
            None if device in (None, "auto") else device
        )
        self.precision = MixedPrecision(self.config.precision, self.device)
        self.run_config = dict(run_config) if run_config is not None else None
        self.config_filename = config_filename
        self.metrics_filename = metrics_filename
        self.iteration = 0
        self.run_dir: Path = Path(self.config.out_dir)
        self.task: Optional[TrainingTask] = None
        self._metrics_logger: Optional[MetricsLogger] = None
        self._stopper: Optional[EarlyStopping] = None
        self._best: Optional[float] = None
        self._checkpoints: list[Path] = []

    # ------------------------------------------------------------ for tasks

    def prepare_module(self, module: nn.Module) -> nn.Module:
        """``module`` on the run's device."""
        return module.to(self.device)

    def prepare_dataloader(self, loader: Iterable[Any]) -> "DeviceLoader":
        """``loader`` with every batch's tensors moved to the run's device."""
        return DeviceLoader(loader, self.device)

    def gradient_update(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.Tensor],
        *,
        max_grad_norm: Optional[float] = None,
        accumulate: bool = True,
    ) -> GradientUpdate:
        """A :class:`GradientUpdate` at this run's precision and accumulation."""
        return GradientUpdate(
            optimizer,
            parameters,
            self.precision,
            max_grad_norm=max_grad_norm,
            accumulation_steps=self.config.grad_accumulation if accumulate else 1,
        )

    @staticmethod
    def infinite(loader: Iterable[Any]) -> Iterator[Any]:
        """Cycle ``loader`` forever, re-shuffling each pass."""
        epoch = 0
        while True:
            sampler = getattr(loader, "set_epoch", None)
            if callable(sampler):
                sampler(epoch)
            yielded = False
            for batch in loader:
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("The training data loader yielded no batches.")
            epoch += 1

    # --------------------------------------------------------------- loop

    def fit(self, task: TrainingTask) -> TrainResult:
        """Train ``task`` until its budget is spent or early stopping fires."""
        cfg = self.config
        self.task = task
        task.attach(self)
        resume_dir = self._resolve_resume()
        self.run_dir = (
            resume_dir.parent.parent
            if resume_dir is not None
            else self._new_run_dir(task)
        )
        self._seed(cfg.seed)
        if cfg.deterministic:
            _enable_determinism()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_files(task)
        self._metrics_logger = MetricsLogger(
            str(self.run_dir),
            enable_console=cfg.console,
            jsonl_filename=self.metrics_filename,
            run_name=self.run_dir.name,
        )

        task.setup(self)
        if cfg.early_stopping_metric:
            self._stopper = EarlyStopping(
                mode=cfg.early_stopping_mode,
                patience=cfg.patience,
                threshold=cfg.min_delta,
            )
        if resume_dir is not None:
            self._load_checkpoint(resume_dir)
            logger.info("Resumed from %s at %s", resume_dir, self._progress_text())

        budget = int(cfg.max_steps if cfg.max_steps is not None else task.max_steps())
        logger.info(
            "Training '%s' for %d %s on %s (%s) -> %s",
            task.name,
            budget,
            task.progress_unit,
            self.device,
            self.precision.name,
            self.run_dir,
        )

        stopped_early = False
        last_checkpoint: Optional[Path] = None
        window: dict[str, list[float]] = {}
        started = time.perf_counter()
        progress = task.progress()
        try:
            while progress < budget:
                before = progress
                step_metrics = task.train_step()
                self.iteration += 1
                progress = task.progress()
                for key, value in step_metrics.items():
                    if _is_number(value):
                        window.setdefault(key, []).append(float(value))

                if _crossed(before, progress, cfg.log_every) or progress >= budget:
                    logged = {k: float(np.mean(v)) for k, v in window.items()}
                    logged["time/elapsed_s"] = time.perf_counter() - started
                    self._log(logged, progress, prefix="train")
                    window = {}

                # The last step is evaluated by final_evaluation() below.
                if (
                    cfg.eval_every
                    and progress < budget
                    and _crossed(before, progress, cfg.eval_every)
                ):
                    if self._evaluate(task, progress):
                        stopped_early = True
                        break

                # Likewise the last step's checkpoint, written after the final
                # evaluation so it records that evaluation's outcome.
                if (
                    cfg.checkpoint_every
                    and progress < budget
                    and _crossed(before, progress, cfg.checkpoint_every)
                ):
                    last_checkpoint = self.save_checkpoint()

                if task.should_stop():
                    logger.info("The task finished early at %s.", self._progress_text())
                    stopped_early = True
                    break
        except KeyboardInterrupt:
            logger.warning("Interrupted; writing a checkpoint before exiting.")
            self.save_checkpoint()
            task.close()
            self._close_logger()
            raise

        final = _numbers(task.final_evaluation())
        if final:
            self._log(final, progress, prefix="eval")
            self._keep_best(final, required=False)
        if (
            last_checkpoint is None
            or self._checkpoint_progress(last_checkpoint) != progress
        ):
            last_checkpoint = self.save_checkpoint(final=True)

        world_model_path = None
        if cfg.export_world_model:
            world_model_path = self._export_world_model(task)
        model_path = self._export_model(task)
        task.close()
        self._close_logger()
        return TrainResult(
            run_dir=self.run_dir,
            progress=progress,
            iterations=self.iteration,
            metrics=final,
            checkpoint=last_checkpoint,
            world_model_path=world_model_path,
            model_path=model_path,
            best_checkpoint=self._best_checkpoint(),
            best_value=self._best,
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------- checkpointing

    def save_checkpoint(
        self, *, final: bool = False, name: Optional[str] = None
    ) -> Path:
        """Write a resumable checkpoint of the task and the loop; returns its dir.

        ``name`` writes it to ``checkpoints/<name>`` instead of
        ``checkpoints/step_<progress>``, outside the pruned periodic ones.
        """
        task = self._require_task()
        progress = task.progress()
        root = self.run_dir / CHECKPOINT_DIR
        target = root / (name or f"step_{progress:08d}")
        staging = root / f".tmp_{target.name}"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "task").mkdir(parents=True)
        task.save_checkpoint(staging / "task")
        torch.save(self._loop_state(), staging / "trainer.pt")
        (staging / _COMPLETE).write_text(
            json.dumps(
                {
                    "task": task.name,
                    "progress": progress,
                    "progress_unit": task.progress_unit,
                    "iteration": self.iteration,
                    "time": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        if not final and name is None:
            self._checkpoints.append(target)
            self._prune_checkpoints()
        return target

    def _load_checkpoint(self, directory: Path) -> None:
        task = self._require_task()
        task.load_checkpoint(directory / "task")
        state = torch.load(
            directory / "trainer.pt", map_location="cpu", weights_only=True
        )
        self.iteration = int(state["iteration"])
        self._best = state.get("best")
        if self._stopper is not None and state.get("stopper"):
            self._stopper.load_state_dict(state["stopper"])
        _set_rng_state(state["rng"])

    def _loop_state(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "best": self._best,
            "stopper": self._stopper.state_dict() if self._stopper else None,
            "rng": _rng_state(),
        }

    def _resolve_resume(self) -> Optional[Path]:
        resume = self.config.resume
        if not resume:
            return None
        if resume == "auto":
            if not self.config.run_name:
                raise ValueError("trainer.resume='auto' needs trainer.run_name.")
            found = latest_checkpoint(Path(self.config.out_dir) / self.config.run_name)
            if found is None:
                logger.info("resume='auto': no checkpoint yet, starting fresh.")
            return found
        path = Path(resume)
        if (path / CHECKPOINT_DIR).is_dir():  # a run directory
            found = latest_checkpoint(path)
            if found is None:
                raise FileNotFoundError(f"No complete checkpoint under {path}.")
            return found
        if not (path / _COMPLETE).exists():
            raise FileNotFoundError(f"{path} is not a complete Simulusion checkpoint.")
        return path

    def _prune_checkpoints(self) -> None:
        keep = max(1, self.config.keep_checkpoints)
        while len(self._checkpoints) > keep:
            old = self._checkpoints.pop(0)
            shutil.rmtree(old, ignore_errors=True)

    @staticmethod
    def _checkpoint_progress(path: Path) -> int:
        return int(json.loads((path / _COMPLETE).read_text())["progress"])

    # ---------------------------------------------------------- internals

    def _evaluate(self, task: TrainingTask, progress: int) -> bool:
        """Run an evaluation; True when early stopping says to stop."""
        metrics = _numbers(task.evaluate())
        if not metrics:
            return False
        self._log(metrics, progress, prefix="eval")
        self._keep_best(metrics)
        name = self.config.early_stopping_metric
        if self._stopper is None or not name:
            return False
        name = _metric_name(name, metrics)
        self._stopper.step(metrics[name])
        if self._stopper.stop:
            logger.info(
                "Early stopping at %s: %s has not improved for %d evaluations.",
                self._progress_text(),
                name,
                self.config.patience,
            )
            return True
        return False

    def _keep_best(
        self, metrics: Mapping[str, float], *, required: bool = True
    ) -> None:
        name = self.config.best_metric
        if not name:
            return
        if (
            not required
            and name not in metrics
            and name.removeprefix("eval/") not in metrics
        ):
            return  # a final evaluation that does not report it
        name = _metric_name(name, metrics)
        value = float(metrics[name])
        better = (
            self._best is None
            or (self.config.best_mode == "max" and value > self._best)
            or (self.config.best_mode == "min" and value < self._best)
        )
        if better:
            self._best = value
            self.save_checkpoint(name=BEST_CHECKPOINT)
            logger.info("New best %s=%.4g at %s", name, value, self._progress_text())

    def _best_checkpoint(self) -> Optional[Path]:
        path = self.run_dir / CHECKPOINT_DIR / BEST_CHECKPOINT
        return path if (path / _COMPLETE).exists() else None

    def _export_world_model(self, task: TrainingTask) -> Optional[Path]:
        model = task.world_model()
        if model is None:
            return None
        path = self.run_dir / "world_model"
        model.save(path)
        logger.info("World model written to %s", path)
        return path

    def _export_model(self, task: TrainingTask) -> Optional[Path]:
        path = task.export(self.run_dir / "model")
        if path is not None:
            logger.info("Model written to %s", path)
        return path

    def _log(self, metrics: Mapping[str, float], step: int, prefix: str) -> None:
        if self._metrics_logger is not None and metrics:
            self._metrics_logger.log(metrics, step, prefix=prefix)
            self._metrics_logger.flush()

    def _close_logger(self) -> None:
        if self._metrics_logger is not None:
            self._metrics_logger.close()
            self._metrics_logger = None

    def _new_run_dir(self, task: TrainingTask) -> Path:
        name = self.config.run_name or f"{task.name}_{time.strftime('%Y%m%d-%H%M%S')}"
        return Path(self.config.out_dir) / name

    def _write_run_files(self, task: TrainingTask) -> None:
        from simulusion.training.provenance import collect_provenance
        from simulusion.training.run import dump_run_config, run_config_dict

        config = self.run_config or run_config_dict(task.name, self.config, task.config)
        (self.run_dir / self.config_filename).write_text(
            dump_run_config(config), encoding="utf-8"
        )
        info = collect_provenance()
        info.update(
            {
                "task": task.name,
                "progress_unit": task.progress_unit,
                "device": str(self.device),
                "precision": self.precision.name,
            }
        )
        (self.run_dir / "run.json").write_text(json.dumps(info, indent=2) + "\n")

    def _seed(self, seed: int) -> None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _progress_text(self) -> str:
        task = self._require_task()
        return f"{task.progress()} {task.progress_unit}"

    def _require_task(self) -> TrainingTask:
        if self.task is None:
            raise RuntimeError("No task is being trained.")
        return self.task


class DeviceLoader:
    """Iterate a loader, moving every tensor in each batch to ``device``.

    Batches may be tensors or nested dicts, lists and tuples of them; other
    values pass through. ``len()`` and other attributes (``dataset``,
    ``batch_size``, ...) are the wrapped loader's.
    """

    def __init__(self, loader: Iterable[Any], device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.non_blocking = device.type == "cuda"

    def __iter__(self) -> Iterator[Any]:
        for batch in self.loader:
            yield _to_device(batch, self.device, self.non_blocking)

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes DeviceLoader does not define itself
        # (dataset, batch_size, sampler, set_epoch, ...).
        if name == "loader":
            raise AttributeError(name)
        return getattr(self.loader, name)


def _to_device(batch: Any, device: torch.device, non_blocking: bool) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        moved = {k: _to_device(v, device, non_blocking) for k, v in batch.items()}
        return moved if type(batch) is dict else cast(Any, type(batch))(moved)
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return type(batch)(*(_to_device(v, device, non_blocking) for v in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device, non_blocking) for v in batch)
    return batch


def _numbers(metrics: Mapping[str, Any]) -> dict[str, float]:
    """The scalar entries of ``metrics``, as floats."""
    return {k: float(v) for k, v in metrics.items() if _is_number(v)}


def latest_checkpoint(run_dir: str | Path) -> Optional[Path]:
    """The newest complete checkpoint in a run directory, or ``None``."""
    root = Path(run_dir) / CHECKPOINT_DIR
    if not root.is_dir():
        return None
    complete = [p for p in root.glob("step_*") if (p / _COMPLETE).exists()]
    return max(complete, key=lambda p: p.name) if complete else None


def _metric_name(name: str, metrics: Mapping[str, float]) -> str:
    """``name`` as a key of ``metrics``, accepting the logged ``eval/`` prefix."""
    if name not in metrics and name.startswith("eval/"):
        name = name[len("eval/") :]  # the logged name, as seen in metrics.jsonl
    if name not in metrics:
        raise KeyError(
            f"Metric '{name}' is not among the evaluation metrics: "
            f"{', '.join(sorted(metrics))}"
        )
    return name


def _crossed(before: int, after: int, every: int) -> bool:
    """Progress moved past a multiple of ``every`` (progress may jump by more than 1)."""
    if every <= 0:
        return False
    return after // every > before // every


def _is_number(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.numel() == 1
    return isinstance(value, (bool, int, float, np.number))


def _rng_state() -> dict[str, Any]:
    # Plain lists and tensors only, so trainer.pt loads with weights_only=True.
    version, keys, pos, has_gauss, cached = cast(tuple, np.random.get_state())
    py_version, py_state, py_gauss = random.getstate()
    state: dict[str, Any] = {
        "python": [py_version, list(py_state), py_gauss],
        "numpy": [version, keys.tolist(), int(pos), int(has_gauss), float(cached)],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    py_version, py_state, py_gauss = state["python"]
    random.setstate((py_version, tuple(py_state), py_gauss))
    version, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state(
        (version, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached)
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError:
            logger.warning("Saved CUDA RNG state does not match this machine's GPUs.")


def _enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


__all__ = [
    "DeviceLoader",
    "TrainResult",
    "Trainer",
    "TrainerConfig",
    "latest_checkpoint",
]
