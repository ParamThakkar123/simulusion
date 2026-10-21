"""DiT on the shared trainer: class-conditional image diffusion.

Not a world model (DiT generates images from a class label, not from
actions), so the run exports the model itself to ``<run_dir>/model``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from simulusion.configs.dit_config import DiTConfig
from simulusion.training.task import Metrics, TrainingTask, register_task


@dataclass
class DiTTaskConfig:
    """``model``: DiT's config (its UPPER_CASE fields, as in ``train_dit``).

    ``DATASET`` / ``ROOT_PATH`` / ``IMAGE_FOLDER``-style fields pick the data;
    ``EPOCHS`` sets the default budget (epochs x batches per epoch). Held-out
    loss is evaluated when ``val_batches`` > 0 and the dataset has a held-out
    split (CIFAR-10's test set, or an imagefolder with ``VAL_SPLIT``).

    Attributes:
        image_folder: Sub-folder of ``ROOT_PATH`` for ``imagenet`` /
            ``imagefolder`` datasets.
        val_batches: Held-out batches per evaluation; 0 skips evaluation.
    """

    model: DiTConfig = field(default_factory=DiTConfig)
    image_folder: Optional[str] = None
    val_batches: int = 8


@register_task("dit")
class DiTTask(TrainingTask):
    """DiT with DDPM's L_simple, an EMA shadow model and gradient clipping at 1.

    Args:
        config: The task config.
        loaders: Prebuilt ``(train_loader, val_loader)`` to use instead of
            building them from the config (``val_loader`` may be ``None``).
        legacy_workdir: Run as ``DiT.fit`` does: evaluation scores the online
            model over the whole held-out loader, and every checkpoint also
            writes ``dit_model_epoch<N>.pth`` (the EMA weights) here.
    """

    config_cls = DiTTaskConfig
    distributed = True

    def __init__(
        self,
        config: Any = None,
        *,
        loaders: Optional[tuple] = None,
        legacy_workdir: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._loaders = loaders
        self._legacy_workdir = legacy_workdir

    def max_steps(self) -> int:
        return int(self.config.model.EPOCHS) * self._batches_per_epoch

    def setup(self, trainer: Any) -> None:
        from simulusion.models.diffusion.DDPM import DDPM
        from simulusion.models.diffusion.DiT import DiT, dit_loaders

        c = self.config.model
        if self._loaders is not None:
            train, val = self._loaders
        else:
            train, val = dit_loaders(
                c.DATASET,
                batch_size=c.BATCH,
                img_size=c.IMG_SIZE,
                root_path=c.ROOT_PATH,
                image_folder=self.config.image_folder,
                crop_size=c.CROP_SIZE,
                num_workers=c.NUM_WORKERS,
                val_split=c.VAL_SPLIT if c.DATASET.lower() == "imagefolder" else None,
                with_val=self.config.val_batches > 0,
            )
        self._batches_per_epoch = max(1, len(train))
        self.batches = trainer.infinite(trainer.prepare_dataloader(train))
        self.val = trainer.prepare_dataloader(val) if val is not None else None
        self.ddpm = DDPM(
            timesteps=c.TIMESTEPS, beta_start=c.BETA_START, beta_end=c.BETA_END
        ).to(trainer.device)
        net = DiT(
            img_size=c.IMG_SIZE,
            patch_size=c.PATCH,
            in_channels=c.CHANNELS,
            d_model=c.WIDTH,
            depth=c.DEPTH,
            heads=c.HEADS,
            drop=c.DROP,
            t_dim=256,
            num_classes=c.NUM_CLASSES,
            class_dropout_prob=c.CLASS_DROPOUT_PROB,
            learn_sigma=c.LEARN_SIGMA,
        )
        self.ema = copy.deepcopy(net).to(trainer.device).eval() if c.EMA else None
        if self.ema is not None:
            for p in self.ema.parameters():
                p.requires_grad = False
        self.model = trainer.prepare_module(net)
        # Paper 4: constant learning rate, no weight decay, no warmup.
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=c.LR, weight_decay=c.WEIGHT_DECAY
        )
        self.update = trainer.gradient_update(
            optimizer, self.model.parameters(), max_grad_norm=1.0, modules=[self.model]
        )

    def train_step(self) -> Metrics:
        from simulusion.models.diffusion.DiT import dit_ema_update, dit_l_simple

        images, labels = next(self.batches)
        self.model.train()
        with self.trainer.precision.autocast():
            loss = dit_l_simple(
                self.model, self.ddpm, images, labels, self.config.model.TIMESTEPS
            )
        norm = self.update(loss)
        if norm is not None and self.ema is not None:
            dit_ema_update(
                self.trainer.unwrap(self.model), self.ema, self.config.model.EMA_DECAY
            )
        metrics = {"loss": float(loss.detach())}
        if norm is not None:
            metrics["grad_norm"] = float(norm)
        return metrics

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        from simulusion.models.diffusion.DiT import dit_l_simple

        if self.val is None:
            return {}
        legacy = self._legacy_workdir is not None
        # DiT.fit scored the online model; the task scores the EMA it ships.
        model = self.ema if self.ema is not None and not legacy else self.model
        was_training = model.training
        model.eval()
        # Fixed noise per evaluation so successive ones are comparable.
        state = torch.random.get_rng_state()
        torch.manual_seed(0)
        total, count = 0.0, 0
        for i, (images, labels) in enumerate(self.val):
            if not legacy and i >= self.config.val_batches:
                break
            loss = dit_l_simple(
                model, self.ddpm, images, labels, self.config.model.TIMESTEPS
            )
            total += float(loss) * images.size(0)
            count += images.size(0)
        torch.random.set_rng_state(state)
        model.train(was_training)
        return {"val_loss": total / max(count, 1)}

    def save_checkpoint(self, directory: Path) -> None:
        super().save_checkpoint(directory)
        if self._legacy_workdir is not None:
            epoch = self.trainer.iteration // self._batches_per_epoch
            os.makedirs(self._legacy_workdir, exist_ok=True)
            model = (
                self.ema if self.ema is not None else self.trainer.unwrap(self.model)
            )
            torch.save(
                model.state_dict(),
                Path(self._legacy_workdir) / f"dit_model_epoch{epoch}.pth",
            )

    def export(self, directory: Path) -> Optional[Path]:
        model = self.ema if self.ema is not None else self.trainer.unwrap(self.model)
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory / "dit_model.pth")  # type: ignore[operator]
        return directory

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.trainer.unwrap(self.model).state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "update": self.update.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.trainer.unwrap(self.model).load_state_dict(state["model"])
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])
        self.update.load_state_dict(state["update"])


__all__ = ["DiTTask", "DiTTaskConfig"]
