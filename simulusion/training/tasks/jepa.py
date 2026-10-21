"""I-JEPA on the shared trainer: self-supervised image representations.

Not a world model (I-JEPA predicts representations of masked image regions,
not the effect of actions), so the run exports the encoders to
``<run_dir>/model``. Data parallelism, precision and checkpoints come from the
Trainer instead of ``train_jepa``'s own SLURM/NCCL setup; the objective, the
masks, the learning-rate / weight-decay schedules and the EMA target encoder
are the same.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from simulusion.configs.jepa_config import JEPAConfig
from simulusion.training.task import Metrics, TrainingTask, register_task


@dataclass
class JEPATaskConfig:
    """``model``: I-JEPA's config (data, masks, optimization; the paper's by default).

    ``model.epochs`` sets the default budget (epochs x batches per epoch).
    ``model.use_bfloat16`` is superseded by ``trainer.precision``. Held-out
    loss is evaluated when ``val_batches`` > 0 and the dataset has a held-out
    split (CIFAR-10's test set, or an image folder with ``val_split``).
    """

    model: JEPAConfig = field(default_factory=JEPAConfig)
    val_batches: int = 0


@register_task("jepa")
class JEPATask(TrainingTask):
    """I-JEPA: context encoder + predictor regress an EMA target encoder's patches.

    Args:
        config: The task config.
        legacy_folder: Run as ``train_jepa`` does, in this folder: one step is
            a whole epoch (so evaluation, early stopping and checkpoints are
            per epoch), evaluation scores the whole held-out loader, training
            starts from ``model.read_checkpoint`` when ``model.load_checkpoint``
            is set, and every checkpoint also writes ``<write_tag>-latest.pth.tar``
            (and ``<write_tag>-ep<N>.pth.tar`` every 50 epochs). This is how
            :meth:`JEPAAgent.train` runs.
    """

    config_cls = JEPATaskConfig
    distributed = True

    #: train_jepa's epochs between numbered checkpoints.
    LEGACY_CHECKPOINT_FREQ = 50

    def __init__(
        self, config: Any = None, *, legacy_folder: Optional[str] = None
    ) -> None:
        super().__init__(config)
        self._legacy_folder = legacy_folder
        if legacy_folder is not None:
            self.progress_unit = "epochs"  # type: ignore[misc]
        self._epochs_done = 0

    def max_steps(self) -> int:
        if self._legacy_folder is not None:
            return int(self.config.model.epochs)
        return int(self.config.model.epochs) * self.ipe

    def progress(self) -> int:
        if self._legacy_folder is not None:
            return self._epochs_done
        return super().progress()

    def setup(self, trainer: Any) -> None:
        from simulusion.helpers.jepa_helper import init_model, init_opt
        from simulusion.masks.multiblock import MaskCollator
        from simulusion.training.train_jepa import build_loss_fn, make_jepa_loaders
        from simulusion.transforms.image import make_transforms

        c = self.config.model
        args = c.to_train_dict()
        start_lr, lr = c.start_lr, c.lr
        if c.lr_reference_batch_size:
            # The paper's rates are quoted for an effective batch of 2048.
            scale = (c.batch_size * trainer.num_processes) / float(
                c.lr_reference_batch_size
            )
            start_lr, lr = start_lr * scale, lr * scale
        self.loss_fn = build_loss_fn(c.loss_type)
        encoder, predictor = init_model(
            device=trainer.device,
            patch_size=c.patch_size,
            crop_size=c.crop_size,
            pred_depth=c.pred_depth,
            pred_emb_dim=c.pred_emb_dim,
            model_name=c.model_name,
        )
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad = False
        self.masks = MaskCollator(
            input_size=c.crop_size,
            patch_size=c.patch_size,
            pred_mask_scale=c.pred_mask_scale,
            enc_mask_scale=c.enc_mask_scale,
            aspect_ratio=c.aspect_ratio,
            nenc=c.num_enc_masks,
            npred=c.num_pred_masks,
            allow_overlap=c.allow_overlap,
            min_keep=c.min_keep,
        )
        transform = make_transforms(
            crop_size=c.crop_size,
            crop_scale=c.crop_scale,
            gaussian_blur=c.use_gaussian_blur,
            horizontal_flip=c.use_horizontal_flip,
            color_distortion=c.use_color_distortion,
            color_jitter=c.color_jitter_strength,
        )
        # The loaders shard themselves (a DistributedSampler per rank), so
        # they are not passed through trainer.prepare_dataloader.
        train, self.sampler, self.val = make_jepa_loaders(
            args,
            transform=transform,
            mask_collator=self.masks,
            world_size=trainer.num_processes,
            rank=trainer.process_index,
            with_val=self.config.val_batches > 0,
        )
        self.ipe = max(1, len(train))
        self.loader = train
        self._epoch = 0
        self._iter = iter(train)
        optimizer, _, self.lr_schedule, self.wd_schedule = init_opt(
            encoder=encoder,
            predictor=predictor,
            wd=float(c.weight_decay),
            final_wd=float(c.final_weight_decay),
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=c.final_lr,
            iterations_per_epoch=self.ipe,
            warmup=c.warmup,
            num_epochs=c.epochs,
            ipe_scale=c.ipe_scale,
        )
        self.encoder = trainer.prepare_module(encoder)
        self.predictor = trainer.prepare_module(predictor)
        self.update = trainer.gradient_update(
            optimizer,
            [*self.encoder.parameters(), *self.predictor.parameters()],
            modules=[self.encoder, self.predictor],
            accumulate=False,
        )
        self._total = int(self.ipe * c.epochs * c.ipe_scale)
        self.step_index = 0
        if self._legacy_folder is not None and c.load_checkpoint:
            self._load_legacy(trainer, optimizer)

    def _momentum(self) -> float:
        lo, hi = self.config.model.ema
        return lo + min(self.step_index, self._total) * (hi - lo) / max(self._total, 1)

    def _next(self) -> tuple:
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self._epoch)
            self._iter = iter(self.loader)
            return next(self._iter)

    def _loss(self, udata: Any, masks_enc: list, masks_pred: list) -> torch.Tensor:
        from simulusion.utils.jepa_utils import repeat_interleave_batch
        from simulusion.utils.utils import apply_masks

        device = self.trainer.device
        imgs = udata[0].to(device, non_blocking=True)
        masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
        masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
        with torch.no_grad():
            h = self.target(imgs)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, len(imgs), repeat=len(masks_enc))
        z = self.predictor(self.encoder(imgs, masks_enc), masks_enc, masks_pred)
        return self.loss_fn(z, h)

    def train_step(self) -> Metrics:
        if self._legacy_folder is None:
            return self._one_step()
        # One epoch: a pass over the loader, as train_jepa counts them.
        rows = [self._one_step() for _ in range(self.ipe)]
        self._epochs_done += 1
        return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}

    def _one_step(self) -> dict[str, float]:
        udata, masks_enc, masks_pred = self._next()
        new_lr = self.lr_schedule.step()
        new_wd = self.wd_schedule.step()
        self.encoder.train()
        self.predictor.train()
        with self.trainer.precision.autocast():
            loss = self._loss(udata, masks_enc, masks_pred)
        self.update(loss)
        # Momentum update of the target encoder (one fused op, as train_jepa).
        with torch.no_grad():
            m = self._momentum()
            target = list(self.target.parameters())
            online = [
                p.detach() for p in self.trainer.unwrap(self.encoder).parameters()
            ]
            torch._foreach_mul_(target, m)
            torch._foreach_add_(target, online, alpha=1.0 - m)
        self.step_index += 1
        return {
            "loss": float(loss.detach()),
            "lr": float(new_lr),
            "wd": float(new_wd),
            "ema": m,
        }

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        self.encoder.eval()
        self.predictor.eval()
        total, count = 0.0, 0
        for i, (udata, masks_enc, masks_pred) in enumerate(self.val):
            if self._legacy_folder is None and i >= self.config.val_batches:
                break
            with self.trainer.precision.autocast():
                loss = float(self._loss(udata, masks_enc, masks_pred))
            total += loss * len(udata[0])
            count += len(udata[0])
        return {"val_loss": total / max(count, 1)}

    def save_checkpoint(self, directory: Path) -> None:
        super().save_checkpoint(directory)
        if self._legacy_folder is not None:
            self._save_legacy()

    def _legacy_state(self) -> dict[str, Any]:
        c = self.config.model
        return {
            "encoder": self.trainer.unwrap(self.encoder).state_dict(),
            "predictor": self.trainer.unwrap(self.predictor).state_dict(),
            "target_encoder": self.target.state_dict(),
            "opt": self.update.optimizer.state_dict(),
            "scaler": None,
            "epoch": self._epochs_done,
            "batch_size": c.batch_size,
            "world_size": self.trainer.num_processes,
            "lr": c.lr,
        }

    def _save_legacy(self) -> None:
        """train_jepa's own checkpoint files, in its format."""
        folder = Path(str(self._legacy_folder))
        folder.mkdir(parents=True, exist_ok=True)
        tag = self.config.model.write_tag
        state = self._legacy_state()
        torch.save(state, folder / f"{tag}-latest.pth.tar")
        if self._epochs_done and self._epochs_done % self.LEGACY_CHECKPOINT_FREQ == 0:
            torch.save(state, folder / f"{tag}-ep{self._epochs_done}.pth.tar")

    def _load_legacy(self, trainer: Any, optimizer: Any) -> None:
        """Start from a train_jepa checkpoint (``model.read_checkpoint``)."""
        from simulusion.helpers.jepa_helper import load_checkpoint

        c = self.config.model
        folder = Path(str(self._legacy_folder))
        path = (
            folder / c.read_checkpoint
            if c.read_checkpoint
            else folder / f"{c.write_tag}-latest.pth.tar"
        )
        if Path(str(c.read_checkpoint or "")).is_absolute():
            path = Path(str(c.read_checkpoint))
        _, _, _, _, _, start_epoch = load_checkpoint(
            device=trainer.device,
            r_path=str(path),
            encoder=self.trainer.unwrap(self.encoder),
            predictor=self.trainer.unwrap(self.predictor),
            target_encoder=self.target,
            opt=optimizer,
            scaler=None,
        )
        # Bring the schedules and the mask collator to that epoch.
        for _ in range(int(start_epoch) * self.ipe):
            self.lr_schedule.step()
            self.wd_schedule.step()
            self.masks.step()
        self.step_index = int(start_epoch) * self.ipe
        self._epochs_done = int(start_epoch)

    def export(self, directory: Path) -> Optional[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "ijepa.pth.tar"
        torch.save(
            {
                "encoder": self.trainer.unwrap(self.encoder).state_dict(),
                "predictor": self.trainer.unwrap(self.predictor).state_dict(),
                "target_encoder": self.target.state_dict(),
                "config": self.config.model.to_dict(),
            },
            path,
        )
        return directory

    def state_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.trainer.unwrap(self.encoder).state_dict(),
            "predictor": self.trainer.unwrap(self.predictor).state_dict(),
            "target_encoder": self.target.state_dict(),
            "update": self.update.state_dict(),
            "step_index": self.step_index,
            "epoch": self._epoch,
            "epochs_done": self._epochs_done,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.trainer.unwrap(self.encoder).load_state_dict(state["encoder"])
        self.trainer.unwrap(self.predictor).load_state_dict(state["predictor"])
        self.target.load_state_dict(state["target_encoder"])
        self.update.load_state_dict(state["update"])
        self.step_index = int(state["step_index"])
        # Bring the schedules and the mask collator to where they were.
        for _ in range(self.step_index):
            self.lr_schedule.step()
            self.wd_schedule.step()
            self.masks.step()
        self._epoch = int(state["epoch"])
        self._epochs_done = int(state.get("epochs_done", 0))
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self._epoch)
        self._iter = iter(self.loader)


__all__ = ["JEPATask", "JEPATaskConfig"]
