"""What a family plugs into the shared :class:`~simulusion.training.Trainer`.

A :class:`TrainingTask` is everything that differs between families: which
networks and optimizers exist, where batches come from (a dataset, a replay
buffer filled by an environment), and what one unit of training does. The
trainer owns everything that does not differ: devices and data parallelism,
precision, seeding, the step budget, logging, evaluation cadence, early
stopping, checkpoints and resuming, and exporting the world model at the end.

.. code-block:: python

    @register_task("my-model")
    class MyTask(TrainingTask):
        config_cls = MyTaskConfig        # dataclass; its fields are YAML sections

        def setup(self, trainer):
            self.model = trainer.prepare_module(MyModel(**...))
            self.update = trainer.gradient_update(
                torch.optim.Adam(self.model.parameters()), self.model.parameters()
            )
            self.batches = trainer.infinite(trainer.prepare_dataloader(loader))

        def train_step(self):
            with self.trainer.precision.autocast():
                loss = self.model(next(self.batches).to(self.trainer.device))
            self.update(loss)
            return {"loss": loss.item()}

A task is the "agent" half of a family: it may hold a world model and a policy
and decide how they train together. The world model itself stays a separate
object, reachable through :meth:`TrainingTask.world_model`, so it can be
saved, evaluated or planned with on its own.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Optional

import torch

if TYPE_CHECKING:
    from simulusion.training.trainer import Trainer
    from simulusion.world_model.protocol import WorldModel

Metrics = Mapping[str, float]

#: Task name -> ``"module:Class"``. Imported on first use so that listing tasks
#: does not import every family.
_TASKS: dict[str, str] = {}
_LOADED: dict[str, type["TrainingTask"]] = {}


class TrainingTask(ABC):
    """One family's training, run by :class:`~simulusion.training.Trainer`.

    Subclasses set :attr:`config_cls` (a dataclass whose fields are the
    run config's sections besides ``trainer``) and implement :meth:`setup`
    and :meth:`train_step`.

    Attributes:
        name: Registry name, set by :func:`register_task`.
        config_cls: The task's config dataclass.
        progress_unit: What :meth:`progress` counts, for logs: ``"steps"``
            (calls to :meth:`train_step`), ``"env_steps"`` or ``"epochs"``.
        distributed: The task runs under data parallelism (several
            processes). Tasks that collect from an environment into a local
            replay buffer are single-process.
        default_max_steps: Budget used when the trainer config gives none.
    """

    name: ClassVar[str] = ""
    config_cls: ClassVar[type]
    progress_unit: ClassVar[str] = "steps"
    distributed: ClassVar[bool] = False
    default_max_steps: ClassVar[int] = 1000

    def __init__(self, config: Any = None) -> None:
        self.config = config if config is not None else self.config_cls()
        self._trainer: Optional["Trainer"] = None

    @property
    def trainer(self) -> "Trainer":
        if self._trainer is None:
            raise RuntimeError(f"{type(self).__name__} is not attached to a Trainer.")
        return self._trainer

    def attach(self, trainer: "Trainer") -> None:
        self._trainer = trainer

    def max_steps(self) -> int:
        """The budget, in :attr:`progress_unit`, when the trainer sets none."""
        return self.default_max_steps

    @abstractmethod
    def setup(self, trainer: "Trainer") -> None:
        """Build networks, optimizers and data on ``trainer.device``."""

    @abstractmethod
    def train_step(self) -> Metrics:
        """One unit of training; returns scalar metrics."""

    def progress(self) -> int:
        """How far training is, in :attr:`progress_unit`.

        Defaults to the number of :meth:`train_step` calls. Tasks whose budget
        is counted in environment steps or epochs override it.
        """
        return self.trainer.iteration

    def evaluate(self) -> Metrics:
        """Periodic evaluation metrics (validation loss, episode return)."""
        return {}

    def final_evaluation(self) -> Metrics:
        """Metrics computed once, after the last step. Defaults to :meth:`evaluate`."""
        return self.evaluate()

    def world_model(self) -> Optional["WorldModel"]:
        """The trained world model behind the shared protocol, if the task has one."""
        return None

    def export(self, directory: Path) -> Optional[Path]:
        """Write the trained model in its own reusable form; return ``directory``.

        For models that are not world models (DiT, I-JEPA, a VAE), which
        :meth:`world_model` cannot return. Called once, when training ends,
        with ``<run_dir>/model``. The default writes nothing.
        """
        del directory
        return None

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume: weights, optimizer and scaler states."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state

    def save_checkpoint(self, directory: Path) -> None:
        """Write the task's resumable state into ``directory``.

        The default saves :meth:`state_dict` to ``task.pt``. Tasks with state
        too large or too odd for one file (a replay buffer) extend it.
        """
        torch.save(self.state_dict(), directory / "task.pt")

    def load_checkpoint(self, directory: Path) -> None:
        path = directory / "task.pt"
        if path.exists():
            state = torch.load(
                path, map_location=self.trainer.device, weights_only=True
            )
            self.load_state_dict(state)

    def close(self) -> None:
        """Release environments and other resources."""


def register_task(
    name: str, target: Optional[str] = None
) -> Callable[[type[TrainingTask]], type[TrainingTask]]:
    """Register a task class under ``name``.

    Use as a decorator on the class, or call with ``target="module:Class"``
    to register lazily without importing it.
    """

    def decorate(cls: type[TrainingTask]) -> type[TrainingTask]:
        cls.name = name
        _TASKS[name] = f"{cls.__module__}:{cls.__qualname__}"
        _LOADED[name] = cls
        return cls

    if target is not None:
        _TASKS[name] = target
    return decorate


def _ensure_builtin() -> None:
    import simulusion.training.tasks  # noqa: F401  (registers the built-in tasks)


def get_task(name: str) -> type[TrainingTask]:
    """The task class registered as ``name``."""
    _ensure_builtin()
    key = name.strip().lower()
    if key not in _TASKS:
        known = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown training task '{name}'. Known: {known}")
    if key not in _LOADED:
        module, _, attr = _TASKS[key].partition(":")
        cls = getattr(importlib.import_module(module), attr)
        cls.name = key
        _LOADED[key] = cls
    return _LOADED[key]


def list_tasks() -> list[str]:
    _ensure_builtin()
    return sorted(_TASKS)


__all__ = ["Metrics", "TrainingTask", "get_task", "list_tasks", "register_task"]
