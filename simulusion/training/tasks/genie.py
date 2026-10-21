"""Genie on the shared trainer: a latent-action world model learned from video."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


import torch

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

    Args:
        config: The task config.
        genie: Train this :class:`~simulusion.training.train_genie.
            GenieTrainer` (its model, optimizer, schedule and precision)
            instead of building one, on ``train_loader`` / ``val_loader``,
            which yield ``(B, C, T, H, W)`` video tensors. This is how
            :meth:`GenieTrainer.train` runs; with ``checkpoint_dir`` its
            ``genie_step_<N>.pt`` files are also written whenever the trainer
            checkpoints. Single-process only.
    """

    config_cls = GenieTaskConfig

    def __init__(
        self,
        config: Any = None,
        *,
        genie: Optional[GenieTrainer] = None,
        train_loader: Optional[Any] = None,
        val_loader: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._adopted = genie
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._checkpoint_dir = checkpoint_dir

    def max_steps(self) -> int:
        return int(self.config.model.max_steps)

    def progress(self) -> int:
        # The model's own step counter, so a GenieTrainer loaded from a
        # checkpoint continues towards the same budget.
        return int(self.genie.global_step)

    def setup(self, trainer: Any) -> None:
        if self._adopted is not None:
            self.genie = self._adopted
            self.batches = trainer.infinite(self._train_loader)
            self.val = self._val_loader
            return
        cfg = self.config
        model = trainer.prepare_module(build_genie(cfg.model))
        self.genie = GenieTrainer(
            model, cfg.model, trainer.device, precision=trainer.precision
        )
        self.genie.update = trainer.gradient_update(
            self.genie.optimizer, model.parameters(), max_grad_norm=1.0
        )
        self.batches, self.val, _ = sequence_loaders(
            cfg.data,
            trainer,
            sequence_length=cfg.model.num_frames,
            image_size=cfg.model.image_size,
        )

    def _video(self, batch: Any) -> torch.Tensor:
        if self._adopted is not None:  # the caller's (B, C, T, H, W) tensors
            return batch.to(self.genie.device)
        return frames_to_float(batch["frames"]).permute(0, 2, 1, 3, 4)

    def train_step(self) -> Metrics:
        losses = self.genie.train_step(self._video(next(self.batches)))
        return {k: float(v) for k, v in losses.items() if v is not None}

    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        if self._adopted is not None:
            # The whole held-out loader, as GenieTrainer.train scored it.
            return {"val_recon_loss": self.genie.validate_epoch(self.val)}
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

        return GenieWorldModel(self.genie.model)

    def save_checkpoint(self, directory: Path) -> None:
        super().save_checkpoint(directory)
        if self._adopted is not None and self._checkpoint_dir:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            name = f"genie_step_{self.genie.global_step}.pt"
            self.genie.save_checkpoint(os.path.join(self._checkpoint_dir, name))

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.genie.model.state_dict(),
            "update": self.genie.update.state_dict(),
            "scheduler": self.genie.scheduler.state_dict(),
            "global_step": self.genie.global_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.genie.model.load_state_dict(state["model"])
        self.genie.update.load_state_dict(state["update"])
        self.genie.scheduler.load_state_dict(state["scheduler"])
        self.genie.global_step = int(state["global_step"])


__all__ = ["GenieTask", "GenieTaskConfig"]
