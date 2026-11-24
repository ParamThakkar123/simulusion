"""LeWorldModel on the shared trainer: offline, reward-free, from pixels.

Trains a :class:`~simulusion.models.lewm.LeWM` on episodes of frames and
actions (an episode folder, or a LeRobot dataset such as PushT), with the
two-term objective of the paper (Eq. 3) and its optimization: AdamW, a linear
warmup over 1% of the steps then cosine decay, gradient clipping at 1. Run
with ``trainer.precision=bf16`` for the paper's precision. The run exports a
``lewm`` world model (:class:`~simulusion.world_model.lewm.LeWMWorldModel`).

.. code-block:: bash

    simulusion train lewm data.path=data/pusht trainer.precision=bf16

Each training window spans ``model.window_frames`` raw frames: every
``frameskip``-th frame is kept, and the actions between kept frames are
concatenated into one action block per model step.

The reference steps its scheduler once per epoch while sizing the warmup and
cosine in optimizer steps; this task steps it every optimizer step, which is
what those sizes describe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from simulusion.configs.lewm_config import LeWMConfig
from simulusion.data.sequence import frames_to_float, one_hot_actions
from simulusion.training.task import Metrics, TrainingTask, register_task
from simulusion.training.tasks.common import (
    SequenceDataConfig,
    mean_metrics,
    open_episodes,
    sequence_loaders,
)

logger = logging.getLogger(__name__)


def _default_data() -> SequenceDataConfig:
    return SequenceDataConfig(batch_size=128, val_fraction=0.1)


@dataclass
class LeWMTaskConfig:
    """``model``: LeWM's architecture, objective and optimizer; ``data``: episodes.

    The episodes need actions; rewards are ignored. ``data.batch_size`` is
    128 and ``data.val_fraction`` 0.1 by default, as in the paper's code
    (``train_split: 0.9``). ``data.sequence_length`` is ignored: the window
    is ``model.window_frames``.
    """

    model: LeWMConfig = field(default_factory=LeWMConfig)
    data: SequenceDataConfig = field(default_factory=_default_data)


@register_task("lewm")
class LeWMTask(TrainingTask):
    """Train LeWorldModel: next-embedding prediction plus SIGReg.

    Progress is optimizer steps; the default budget is ``model.epochs``
    passes over the training windows. Evaluation reports held-out
    ``val_pred_loss``, ``val_sigreg_loss`` and the latent ``val_straightness``
    (Eq. 9) when ``data.val_fraction`` is set.
    """

    config_cls = LeWMTaskConfig

    def max_steps(self) -> int:
        return int(self.config.model.epochs) * self._steps_per_epoch

    def setup(self, trainer: Any) -> None:
        from simulusion.models.lewm import LeWM, lr_lambda

        cfg: LeWMConfig = self.config.model
        data: SequenceDataConfig = self.config.data
        episodes, meta = open_episodes(data)
        if meta is not None and meta.action_dim:
            cfg.action_dim = int(meta.action_dim)
        self._discrete = meta is not None and meta.action_kind == "discrete"
        mean, std = self._action_stats(episodes, cfg)
        if not cfg.action_dim:
            cfg.action_dim = int(mean.numel())
        self.batches, self.val, _ = sequence_loaders(
            data,
            trainer,
            sequence_length=cfg.window_frames,
            image_size=cfg.img_size,
            require_actions=True,
        )
        if data.sequence_length not in (None, cfg.window_frames):
            logger.warning(
                "data.sequence_length is ignored; LeWM windows are %d frames.",
                cfg.window_frames,
            )
        windows = sum(
            max(0, (n - cfg.window_frames) // data.stride + 1)
            for n in getattr(episodes, "lengths", [])
        )
        train_windows = windows * (1.0 - data.val_fraction)
        self._steps_per_epoch = max(1, int(train_windows // data.batch_size))

        model = LeWM(cfg)
        model.set_action_stats(mean, std)
        self.net = model.to(trainer.device)
        self.model = trainer.prepare_module(self.net)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        total = trainer.config.max_steps or self.max_steps()
        warmup = max(1, int(cfg.warmup_fraction * total))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda(warmup, total)
        )
        self.update = trainer.gradient_update(
            optimizer, self.net.parameters(), max_grad_norm=cfg.grad_clip
        )

    def _action_stats(
        self, episodes: Any, cfg: LeWMConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension mean and std of raw actions over (a sample of) the episodes."""
        n = len(episodes)
        limit = cfg.action_stats_episodes or n
        if n > limit:
            picks = np.random.default_rng(0).choice(n, size=limit, replace=False)
        else:
            picks = np.arange(n)
        rows = []
        for i in sorted(int(p) for p in picks):
            actions = episodes[i].actions
            if actions is None:
                continue
            tensor = self._actions(torch.as_tensor(np.asarray(actions)), cfg)
            rows.append(tensor.reshape(len(tensor), -1))
        if not rows:
            raise ValueError("LeWM trains on episodes with actions; none have any.")
        flat = torch.cat(rows).float()
        flat = flat[~torch.isnan(flat).any(dim=1)]
        return flat.mean(0), flat.std(0)

    def _actions(self, actions: torch.Tensor, cfg: LeWMConfig) -> torch.Tensor:
        if self._discrete or (
            not torch.is_floating_point(actions) and actions.dim() == 1
        ):
            return one_hot_actions(actions.long(), int(cfg.action_dim))
        return actions.float()

    def _window(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw windows -> frame-skipped frames and action blocks."""
        cfg: LeWMConfig = self.config.model
        k = cfg.frameskip
        device = self.trainer.device
        frames = frames_to_float(batch["frames"][:, ::k].to(device, non_blocking=True))
        actions = self._actions(batch["actions"].to(device, non_blocking=True), cfg)
        B = actions.shape[0]
        steps = actions.shape[1] // k
        blocks = actions[:, : steps * k].reshape(B, steps, k * actions.shape[-1])
        return frames, blocks

    def train_step(self) -> Metrics:
        self.model.train()
        frames, actions = self._window(next(self.batches))
        with self.trainer.precision.autocast():
            out = self.net.loss(frames, actions)
        self.update(out["loss"])
        self.scheduler.step()
        return {
            "loss": float(out["loss"].detach()),
            "pred_loss": float(out["pred_loss"].detach()),
            "sigreg_loss": float(out["sigreg_loss"].detach()),
            "lr": float(self.scheduler.get_last_lr()[0]),
        }

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        from simulusion.models.lewm import temporal_straightness

        self.model.eval()
        rows = []
        for i, batch in enumerate(self.val):
            if i >= self.config.data.val_batches:
                break
            frames, actions = self._window(batch)
            with self.trainer.precision.autocast():
                out = self.net.loss(frames, actions)
                emb = self.net.encode(frames)
            rows.append(
                {
                    "val_loss": float(out["loss"]),
                    "val_pred_loss": float(out["pred_loss"]),
                    "val_sigreg_loss": float(out["sigreg_loss"]),
                    "val_straightness": float(temporal_straightness(emb.float()))
                    if emb.shape[1] >= 3
                    else float("nan"),
                }
            )
        return mean_metrics(rows)

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model.lewm import LeWMWorldModel

        return LeWMWorldModel(self.net)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.net.state_dict(),
            "update": self.update.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state["model"])
        self.update.load_state_dict(state["update"])
        self.scheduler.load_state_dict(state["scheduler"])


__all__ = ["LeWMTask", "LeWMTaskConfig"]
