"""Dreamer on the shared trainer: the agent online, or its world model alone.

* ``dreamer`` / ``dreamer-v1`` / ``dreamer-v2``: the full agent. It collects
  from an environment into a replay buffer and trains the world model, actor
  and critic from it, exactly as :meth:`DreamerAgent.train` does. Progress is
  environment steps (after action repeat).
* ``dreamer-wm``: only the world model (RSSM, encoder, decoder, reward and
  discount heads), from a dataset of episodes. No environment, no policy:
  the result is a :class:`~simulusion.world_model.WorldModel` to evaluate,
  plan with, or hand to an agent.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import torch

from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.data.sequence import one_hot_actions
from simulusion.training.task import Metrics, TrainingTask, register_task
from simulusion.training.tasks.common import (
    SequenceDataConfig,
    load_replay,
    mean_metrics,
    save_replay,
    sequence_loaders,
)

logger = logging.getLogger(__name__)


def _core_cls(name: str) -> Any:
    if name == "dreamer-v2":
        from simulusion.models.dreamer_v2 import DreamerV2

        return DreamerV2
    if name == "dreamer-v1":
        from simulusion.models.dreamer_v1 import DreamerV1

        return DreamerV1
    from simulusion.models.dreamer import Dreamer

    return Dreamer


@dataclass
class DreamerTaskConfig:
    """``model``: Dreamer's config, environment settings included.

    Attributes:
        model: Environment, architecture, replay and optimizer settings. Its
            ``total_steps`` is the default budget, in environment steps.
        save_replay: Keep the replay buffer in checkpoints so a resumed run
            continues on the same data. Costs disk: the buffer holds every
            frame it has seen.
    """

    model: DreamerConfig = field(default_factory=DreamerConfig)
    save_replay: bool = True


@register_task("dreamer")
class DreamerTask(TrainingTask):
    """Dreamer (V1 defaults) collecting from an environment.

    One :meth:`train_step` is one round of :meth:`DreamerAgent.train`:
    ``update_steps`` gradient steps on replay, then ``collect_steps`` of
    environment interaction with the updated policy. The first round also
    collects ``seed_steps`` with a random policy.
    """

    config_cls = DreamerTaskConfig
    progress_unit = "env_steps"
    variant: ClassVar[str] = "dreamer"

    def max_steps(self) -> int:
        return int(self.config.model.total_steps)

    def setup(self, trainer: Any) -> None:
        from simulusion.models.dreamer import make_env

        args = self.config.model
        args.seed = trainer.config.seed
        self.train_env = make_env(args)
        self.test_env = make_env(args)
        obs_shape = self.train_env.observation_space["image"].shape
        action_size = self.train_env.action_space.shape[0]
        self.dreamer = _core_cls(self.variant)(
            args, obs_shape, action_size, trainer.device, restore=False
        )
        self.dreamer.set_precision(trainer.precision)
        self.seeded = False

    def progress(self) -> int:
        return int(self.dreamer.data_buffer.steps * self.config.model.action_repeat)

    def train_step(self) -> Metrics:
        args = self.config.model
        metrics: dict[str, float] = {}
        if not self.seeded:
            rewards = self.dreamer.collect_random_episodes(
                self.train_env, args.seed_steps // args.action_repeat
            )
            metrics["seed_avg_reward"] = float(np.mean(rewards))
            self.seeded = True
        rows = []
        for _ in range(args.update_steps):
            model_loss, actor_loss, value_loss = self.dreamer.train_one_batch()
            rows.append(
                {
                    "model_loss": model_loss,
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                    **self.dreamer.metrics,
                }
            )
        metrics.update(mean_metrics(rows))
        rewards = self.dreamer.act_and_collect_data(
            self.train_env, args.collect_steps // args.action_repeat
        )
        buffer = self.dreamer.data_buffer
        metrics.update(
            {
                "train_avg_reward": float(np.mean(rewards)),
                "train_max_reward": float(np.max(rewards)),
                "replay/fill_ratio": min(1.0, buffer.steps / max(1, buffer.size)),
                "replay/episodes": float(buffer.episodes),
            }
        )
        return metrics

    def evaluate(self) -> Metrics:
        rewards, _, _ = self.dreamer.evaluate(
            self.test_env, self.config.model.test_episodes
        )
        return {
            "eval_avg_reward": float(np.mean(rewards)),
            "eval_std_reward": float(np.std(rewards)),
            "eval_max_reward": float(np.max(rewards)),
            "eval_min_reward": float(np.min(rewards)),
        }

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model import as_world_model

        return as_world_model(self.dreamer)

    def save_checkpoint(self, directory: Path) -> None:
        self.dreamer.save(str(directory / "dreamer.pt"))
        torch.save(
            {
                "seeded": self.seeded,
                "updates": {
                    name: getattr(self.dreamer, f"{name}_update").state_dict()
                    for name in ("world_model", "actor", "value")
                },
            },
            directory / "task.pt",
        )
        if self.config.save_replay:
            save_replay(self.dreamer.data_buffer, directory / "replay.npz")

    def load_checkpoint(self, directory: Path) -> None:
        self.dreamer.restore_checkpoint(
            directory / "dreamer.pt", map_location=self.trainer.device
        )
        state = torch.load(directory / "task.pt", map_location="cpu", weights_only=True)
        for name, update in state["updates"].items():
            getattr(self.dreamer, f"{name}_update").load_state_dict(update)
        if (directory / "replay.npz").exists():
            load_replay(self.dreamer.data_buffer, directory / "replay.npz")
            self.seeded = bool(state["seeded"])
        else:
            logger.warning(
                "The checkpoint has no replay buffer (save_replay was off); "
                "resuming re-collects seed episodes."
            )

    def close(self) -> None:
        for env in (getattr(self, "train_env", None), getattr(self, "test_env", None)):
            close = getattr(env, "close", None)
            if callable(close):
                close()


@register_task("dreamer-v1")
class DreamerV1Task(DreamerTask):
    variant = "dreamer-v1"


@register_task("dreamer-v2")
class DreamerV2Task(DreamerTask):
    """DreamerV2: discrete latents, KL balancing, and its own heads."""

    variant = "dreamer-v2"


# ------------------------------------------------------------- world model


@dataclass
class DreamerWorldModelTaskConfig:
    """``model``: Dreamer's architecture and optimizer; ``data``: the episodes.

    Windows are ``model.train_seq_len`` frames, resized to
    ``model.image_size``; ``data.batch_size`` sets the batch (Dreamer's
    ``model.batch_size`` is replaced by it). Discrete actions are one-hot
    encoded to ``action_dim`` (from the dataset's ``meta.json`` when unset).
    Episodes without rewards train with zero rewards and the reward head
    learns nothing useful; ``terminated`` / ``truncated`` mark boundaries.

    Attributes:
        variant: ``"dreamer"`` (V1) or ``"dreamer-v2"``.
        action_dim: Discrete action count or continuous action length.
    """

    model: DreamerConfig = field(default_factory=DreamerConfig)
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)
    variant: str = "dreamer"
    action_dim: Optional[int] = None


@register_task("dreamer-wm")
class DreamerWorldModelTask(TrainingTask):
    """Dreamer's world model trained offline from a dataset, without a policy.

    Data-parallel: each process trains on its shard and the world model's
    gradients are averaged across processes before every step.
    """

    config_cls = DreamerWorldModelTaskConfig
    distributed = True

    def max_steps(self) -> int:
        return 10_000

    def setup(self, trainer: Any) -> None:
        cfg = self.config
        args = copy.deepcopy(cfg.model)
        args.batch_size = cfg.data.batch_size
        args.buffer_size = 1  # no replay: batches come from the dataset
        self.batches, self.val, meta = sequence_loaders(
            cfg.data,
            trainer,
            sequence_length=args.train_seq_len,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            require_actions=True,
        )
        first = next(self.batches)
        self._first: Optional[dict] = first
        actions = first["actions"]
        self.discrete = not torch.is_floating_point(actions)
        dim = cfg.action_dim or (meta.action_dim if meta else None)
        if dim is None:
            if self.discrete:
                raise ValueError(
                    "Discrete actions need their count: set action_dim, or write "
                    "the dataset with a meta.json (EpisodeWriter does)."
                )
            dim = int(actions.shape[-1])
        self.action_dim = int(dim)
        obs_shape = tuple(int(d) for d in first["frames"].shape[2:])
        self.dreamer = _core_cls(cfg.variant)(
            args, obs_shape, self.action_dim, trainer.device, restore=False
        )
        self.dreamer.set_precision(
            trainer.precision, all_reduce=trainer.num_processes > 1
        )
        trainer.broadcast_parameters(self.dreamer.world_model_modules)
        self.args = args

    def _tensors(self, batch: dict) -> tuple:
        """A dataset batch in the replay buffer's time-major layout.

        Replay row ``t`` holds ``obs_t``, the action taken there, the reward
        it earned and whether the episode ended after it. A window of ``L``
        frames has ``L - 1`` such transitions; the last row's action and
        reward are padding that ``world_model_loss`` never reads (it uses
        ``acs[:-1]`` and ``rews[:-1]``).
        """
        frames = batch["frames"]  # (B, L, C, H, W) uint8
        B, L = frames.shape[:2]
        device = self.trainer.device
        obs = frames.transpose(0, 1).contiguous().to(device)

        def pad(x: torch.Tensor) -> torch.Tensor:  # (B, L-1, ...) -> (L, B, ...)
            x = x.transpose(0, 1).to(device)
            return torch.cat([x, torch.zeros_like(x[:1])], dim=0)

        actions = batch["actions"]
        if self.discrete:
            actions = one_hot_actions(actions, self.action_dim)
        acs = pad(actions.float().reshape(B, L - 1, -1))
        zeros = torch.zeros(B, L - 1)
        rewards = batch.get("rewards", zeros).float()
        reward_scale = float(getattr(self.args, "reward_scale", 1.0))
        rews = pad(rewards * reward_scale).unsqueeze(-1)
        terminated = batch.get("terminated", zeros.bool()).float()
        boundary = torch.maximum(
            terminated, batch.get("truncated", zeros.bool()).float()
        )
        nonterms = (1.0 - pad(boundary)).unsqueeze(-1)
        continues = (1.0 - pad(terminated)).unsqueeze(-1)
        return obs, acs, rews, nonterms, continues

    def train_step(self) -> Metrics:
        batch = self._first if self._first is not None else next(self.batches)
        self._first = None
        for module in self.dreamer.world_model_modules:
            module.train()
        loss, norm = self.dreamer.update_world_model(*self._tensors(batch))
        metrics = {"model_loss": float(loss.detach()), **self.dreamer._model_components}
        if norm is not None:
            metrics["grad_norm/model"] = float(norm)
        return metrics

    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        rows = []
        for module in self.dreamer.world_model_modules:
            module.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val):
                if i >= self.config.data.val_batches:
                    break
                if batch["frames"].shape[0] != self.args.batch_size:
                    continue  # world_model_loss is built for a fixed batch size
                loss = self.dreamer.world_model_loss(*self._tensors(batch))
                rows.append(
                    {"val_model_loss": float(loss)}
                    | {
                        f"val_{k.split('/')[-1]}": v
                        for k, v in self.dreamer._model_components.items()
                    }
                )
        return mean_metrics(rows)

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model import as_world_model

        return as_world_model(self.dreamer)

    def save_checkpoint(self, directory: Path) -> None:
        self.dreamer.save(str(directory / "dreamer.pt"))
        torch.save(
            {"world_model": self.dreamer.world_model_update.state_dict()},
            directory / "task.pt",
        )

    def load_checkpoint(self, directory: Path) -> None:
        self.dreamer.restore_checkpoint(
            directory / "dreamer.pt", map_location=self.trainer.device
        )
        state = torch.load(directory / "task.pt", map_location="cpu", weights_only=True)
        self.dreamer.world_model_update.load_state_dict(state["world_model"])


__all__ = [
    "DreamerTask",
    "DreamerTaskConfig",
    "DreamerV1Task",
    "DreamerV2Task",
    "DreamerWorldModelTask",
    "DreamerWorldModelTaskConfig",
]
