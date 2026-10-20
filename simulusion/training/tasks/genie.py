"""Genie on the shared trainer: a latent-action world model learned from video."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


from simulusion.data.sequence import frames_to_float
from simulusion.training.task import Metrics, TrainingTask, register_task
from simulusion.training.tasks.common import (
    SequenceDataConfig,
    mean_metrics,
    sequence_loaders,
)
from simulusion.training.train_genie import GenieConfig, GenieTrainer, build_genie


@dataclass
class GenieTaskConfig:
    """``model``: Genie's architecture and optimizer; ``data``: the video clips.

    Genie needs no actions or rewards, so any folder of videos or episodes
    trains it; windows are ``model.num_frames`` long.
    """

    model: GenieConfig = field(default_factory=GenieConfig)
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)


@register_task("genie")
class GenieTask(TrainingTask):
    """Train Genie (tokenizer, latent actions and dynamics) on video clips.

    Progress is optimizer steps. Evaluation reports held-out reconstruction
    loss when ``data.val_fraction`` is set.
    """

    config_cls = GenieTaskConfig
    distributed = True

    def max_steps(self) -> int:
        return int(self.config.model.max_steps)

    def setup(self, trainer: Any) -> None:
        cfg = self.config
        model = trainer.prepare_module(build_genie(cfg.model))
        self.genie = GenieTrainer(
            model, cfg.model, trainer.device, precision=trainer.precision
        )
        self.genie.update = trainer.gradient_update(
            self.genie.optimizer, model.parameters(), max_grad_norm=1.0, modules=[model]
        )
        self.batches, self.val, _ = sequence_loaders(
            cfg.data,
            trainer,
            sequence_length=cfg.model.num_frames,
            image_size=cfg.model.image_size,
        )

    def train_step(self) -> Metrics:
        batch = next(self.batches)
        video = frames_to_float(batch["frames"]).permute(0, 2, 1, 3, 4)
        losses = self.genie.train_step(video)
        return {k: float(v) for k, v in losses.items() if v is not None}

    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        rows = []
        for i, batch in enumerate(self.val):
            if i >= self.config.data.val_batches:
                break
            video = frames_to_float(batch["frames"]).permute(0, 2, 1, 3, 4)
            rows.append({k: float(v) for k, v in self.genie.validate(video).items()})
        self.genie.model.train()
        return mean_metrics(rows)

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model.genie import GenieWorldModel

        return GenieWorldModel(self.trainer.unwrap(self.genie.model))

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.trainer.unwrap(self.genie.model).state_dict(),
            "update": self.genie.update.state_dict(),
            "scheduler": self.genie.scheduler.state_dict(),
            "global_step": self.genie.global_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.trainer.unwrap(self.genie.model).load_state_dict(state["model"])
        self.genie.update.load_state_dict(state["update"])
        self.genie.scheduler.load_state_dict(state["scheduler"])
        self.genie.global_step = int(state["global_step"])


__all__ = ["GenieTask", "GenieTaskConfig"]
