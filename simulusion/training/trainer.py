"""The one training loop every Simulusion family runs on.

.. code-block:: python

    from simulusion.training import Trainer, TrainerConfig
    from simulusion.training.tasks import GenieTask, GenieTaskConfig

    result = Trainer(TrainerConfig(max_steps=10_000, precision="bf16")).fit(
        GenieTask(GenieTaskConfig(data=VideoDataConfig(path="data/clips")))
    )
    result.world_model_path  # runs/<name>/world_model, loadable with load_world_model

The loop is built on `Hugging Face Accelerate`_, so the same code runs on a
CPU, one GPU, or several (``accelerate launch -m simulusion.training.run
...``) without changes. Precision is a run setting (``fp32``, ``bf16``,
``fp16``, ``auto``) applied through :class:`~simulusion.training.precision.
MixedPrecision`, rather than a flag per model.

Every run gets a directory holding everything needed to reproduce it::

    runs/<run_name>/
        config.yaml          # the fully resolved run config: re-run with --config
        run.json             # git commit, package versions, host, command line
        metrics.jsonl        # every logged metric
        checkpoints/step_000500/   # resumable: trainer.resume=<this dir> or "auto"
        world_model/         # the trained world model (WorldModel protocol)

.. _Hugging Face Accelerate: https://huggingface.co/docs/accelerate
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
from simulusion.utils.logging_utils import MetricsLogger
from simulusion.utils.train_utils import EarlyStopping

logger = logging.getLogger("simulusion.trainer")

CHECKPOINT_DIR = "checkpoints"
_COMPLETE = "checkpoint.json"


@dataclass
class TrainerConfig(SerializableConfigMixin):
    """Settings shared by every training run, whatever the family.

    Attributes:
        max_steps: Budget in the task's progress unit (optimizer steps,
            environment steps or epochs; see the task). ``None`` uses the
            task's default.
        seed: Seeds Python, NumPy and PyTorch (offset per process under data
            parallelism, so each shards differently).
        deterministic: Ask PyTorch for deterministic kernels. Slower; some
            ops only warn.
        precision: ``fp32``, ``bf16``, ``fp16`` or ``auto``.
        device: ``None`` picks the accelerator Accelerate finds (CUDA, MPS,
            else CPU); ``"cpu"`` forces CPU.
        grad_accumulation: Batches per optimizer step, for tasks whose
            updates support it (dataset-driven ones).
        find_unused_parameters: Tell DDP some parameters may get no gradient
            in a step (models with optional heads). Only matters with several
            processes; it costs an extra pass over the graph.
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
        early_stopping_metric: Evaluation metric that stops the run once it
            stops improving; ``None`` disables early stopping.
        early_stopping_mode: ``"min"`` or ``"max"``.
        patience: Evaluations without improvement before stopping.
        min_delta: Relative improvement that counts.
        tensorboard: Also log to TensorBoard (``simulusion[ml]``).
        wandb: Also log to Weights & Biases (``simulusion[ml]``).
        wandb_project: W&B project.
        console: Print metrics to the console.
    """

    max_steps: Optional[int] = None
    seed: int = 0
    deterministic: bool = False
    precision: str = "fp32"
    device: Optional[str] = None
    grad_accumulation: int = 1
    find_unused_parameters: bool = False
    log_every: int = 100
    eval_every: int = 0
    checkpoint_every: int = 0
    keep_checkpoints: int = 2
    out_dir: str = "runs"
    run_name: Optional[str] = None
    resume: Optional[str] = None
    export_world_model: bool = True
    early_stopping_metric: Optional[str] = None
    early_stopping_mode: str = "min"
    patience: int = 10
    min_delta: float = 1e-4
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: str = "simulusion"
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
    stopped_early: bool = False


class Trainer:
    """Runs a :class:`~simulusion.training.task.TrainingTask` to its budget.

    Args:
        config: Run settings; defaults throughout when ``None``.
        run_config: The full run config (all sections) to write as
            ``config.yaml``. :func:`simulusion.training.run.run` passes it;
            without it only the trainer and task sections are written.
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        *,
        run_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs

        self.config = config if config is not None else TrainerConfig()
        # Precision is handled by MixedPrecision/GradientUpdate (a loss scaler
        # per optimizer), so Accelerate itself always runs in "no" mode.
        self.accelerator = Accelerator(
            cpu=self.config.device == "cpu",
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=self.config.find_unused_parameters
                )
            ],
        )
        if self.config.device not in (None, "cpu", "auto"):
            wanted = torch.device(str(self.config.device)).type
            if wanted != self.accelerator.device.type:
                raise ValueError(
                    f"device={self.config.device!r} was requested but Accelerate "
                    f"selected {self.accelerator.device}. Leave device unset, or "
                    "use CUDA_VISIBLE_DEVICES / `accelerate config` to choose GPUs."
                )
        self.device: torch.device = self.accelerator.device
        self.precision = MixedPrecision(self.config.precision, self.device)
        self.run_config = dict(run_config) if run_config is not None else None
        self.iteration = 0
        self.run_dir: Path = Path(self.config.out_dir)
        self.task: Optional[TrainingTask] = None
        self._metrics_logger: Optional[MetricsLogger] = None
        self._stopper: Optional[EarlyStopping] = None
        self._best: Optional[float] = None
        self._checkpoints: list[Path] = []

    # ------------------------------------------------------------ for tasks

    @property
    def is_main_process(self) -> bool:
        return bool(self.accelerator.is_main_process)

    @property
    def num_processes(self) -> int:
        return int(self.accelerator.num_processes)

    @property
    def process_index(self) -> int:
        return int(self.accelerator.process_index)

    def prepare_module(self, module: nn.Module) -> nn.Module:
        """Move ``module`` to the device, wrapped for data parallelism if needed."""
        module = module.to(self.device)
        if self.num_processes > 1 and any(p.requires_grad for p in module.parameters()):
            return self.accelerator.prepare_model(module)
        return module

    def broadcast_parameters(self, modules: Iterable[nn.Module]) -> None:
        """Copy process 0's weights and buffers to every process.

        DDP does this when it wraps a module. Modules trained unwrapped (with
        ``gradient_update(..., all_reduce=True)``) must start identical too,
        and each process is seeded differently. A no-op in one process.
        """
        if self.num_processes == 1:
            return
        import torch.distributed as dist

        for module in modules:
            for tensor in [*module.parameters(), *module.buffers()]:
                dist.broadcast(tensor.data, src=0)

    def prepare_dataloader(self, loader: Any) -> Any:
        """Shard ``loader`` across processes and move its batches to the device."""
        return self.accelerator.prepare_data_loader(loader)

    def unwrap(self, module: nn.Module) -> nn.Module:
        """The module inside a data-parallel wrapper (for saving)."""
        return self.accelerator.unwrap_model(module)

    def gradient_update(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.Tensor],
        *,
        max_grad_norm: Optional[float] = None,
        modules: Iterable[nn.Module] = (),
        accumulate: bool = True,
        all_reduce: bool = False,
    ) -> GradientUpdate:
        """A :class:`GradientUpdate` at this run's precision and accumulation.

        Pass the DDP-wrapped ``modules`` the parameters belong to, or
        ``all_reduce=True`` for parameters of modules left unwrapped (see
        :class:`GradientUpdate`).
        """
        return GradientUpdate(
            optimizer,
            parameters,
            self.precision,
            max_grad_norm=max_grad_norm,
            accumulation_steps=self.config.grad_accumulation if accumulate else 1,
            modules=modules,
            all_reduce=all_reduce and self.num_processes > 1,
        )

    def reduce(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Average scalar metrics over processes (a no-op on one process)."""
        values = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        if self.num_processes == 1 or not values:
            return values
        keys = sorted(values)
        tensor = torch.tensor([values[k] for k in keys], device=self.device)
        tensor = self.accelerator.reduce(tensor, reduction="mean")
        return dict(zip(keys, tensor.tolist()))

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
        if self.num_processes > 1 and not task.distributed:
            raise RuntimeError(
                f"The '{task.name}' task collects its own data in one process and "
                "cannot run data-parallel. Launch it as a single process."
            )
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
        if self.is_main_process:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_files(task)
            self._metrics_logger = MetricsLogger(
                str(self.run_dir),
                enable_console=cfg.console,
                enable_tensorboard=cfg.tensorboard,
                enable_wandb=cfg.wandb,
                wandb_project=cfg.wandb_project,
                run_name=self.run_dir.name,
            )
        self.accelerator.wait_for_everyone()

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
            "Training '%s' for %d %s on %s (%s, %d process%s) -> %s",
            task.name,
            budget,
            task.progress_unit,
            self.device,
            self.precision.name,
            self.num_processes,
            "" if self.num_processes == 1 else "es",
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
                    self._log(self.reduce(logged), progress, prefix="train")
                    window = {}

                if cfg.eval_every and _crossed(before, progress, cfg.eval_every):
                    if self._evaluate(task, progress):
                        stopped_early = True
                        break

                if cfg.checkpoint_every and _crossed(
                    before, progress, cfg.checkpoint_every
                ):
                    last_checkpoint = self.save_checkpoint()
        except KeyboardInterrupt:
            logger.warning("Interrupted; writing a checkpoint before exiting.")
            self.save_checkpoint()
            task.close()
            self._close_logger()
            raise

        final = self.reduce(task.final_evaluation())
        if final:
            self._log(final, progress, prefix="eval")
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
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------- checkpointing

    def save_checkpoint(self, *, final: bool = False) -> Path:
        """Write a resumable checkpoint of the task and the loop; returns its dir."""
        task = self._require_task()
        progress = task.progress()
        root = self.run_dir / CHECKPOINT_DIR
        target = root / f"step_{progress:08d}"
        self.accelerator.wait_for_everyone()
        if self.is_main_process:
            staging = root / f".tmp_step_{progress:08d}"
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
            if not final:
                self._checkpoints.append(target)
                self._prune_checkpoints()
        self.accelerator.wait_for_everyone()
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
        _set_rng_state(state["rng"][min(self.process_index, len(state["rng"]) - 1)])

    def _loop_state(self) -> dict[str, Any]:
        # Only the main process writes, so other ranks' RNG is not kept: a
        # resumed data-parallel run is seeded identically but not bit-exact.
        return {
            "iteration": self.iteration,
            "best": self._best,
            "stopper": self._stopper.state_dict() if self._stopper else None,
            "rng": [_rng_state()],
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
        metrics = self.reduce(task.evaluate())
        if not metrics:
            return False
        self._log(metrics, progress, prefix="eval")
        name = self.config.early_stopping_metric
        if self._stopper is None or not name:
            return False
        if name not in metrics and name.startswith("eval/"):
            name = name[len("eval/") :]  # the logged name, as seen in metrics.jsonl
        if name not in metrics:
            raise KeyError(
                f"early_stopping_metric '{name}' is not among the evaluation "
                f"metrics: {', '.join(sorted(metrics))}"
            )
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

    def _export_world_model(self, task: TrainingTask) -> Optional[Path]:
        model = task.world_model()
        if model is None:
            return None
        path = self.run_dir / "world_model"
        if self.is_main_process:
            model.save(path)
            logger.info("World model written to %s", path)
        self.accelerator.wait_for_everyone()
        return path

    def _export_model(self, task: TrainingTask) -> Optional[Path]:
        path: Optional[Path] = None
        if self.is_main_process:
            path = task.export(self.run_dir / "model")
            if path is not None:
                logger.info("Model written to %s", path)
        self.accelerator.wait_for_everyone()
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
        (self.run_dir / "config.yaml").write_text(
            dump_run_config(config), encoding="utf-8"
        )
        info = collect_provenance()
        info.update(
            {
                "task": task.name,
                "progress_unit": task.progress_unit,
                "device": str(self.device),
                "precision": self.precision.name,
                "num_processes": self.num_processes,
            }
        )
        (self.run_dir / "run.json").write_text(json.dumps(info, indent=2) + "\n")

    def _seed(self, seed: int) -> None:
        seed = int(seed) + self.process_index
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


def latest_checkpoint(run_dir: str | Path) -> Optional[Path]:
    """The newest complete checkpoint in a run directory, or ``None``."""
    root = Path(run_dir) / CHECKPOINT_DIR
    if not root.is_dir():
        return None
    complete = [p for p in root.glob("step_*") if (p / _COMPLETE).exists()]
    return max(complete, key=lambda p: p.name) if complete else None


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


__all__ = ["TrainResult", "Trainer", "TrainerConfig", "latest_checkpoint"]
