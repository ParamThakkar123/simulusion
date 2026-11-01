"""DIAMOND on the shared trainer: the agent online, or its world model alone.

* ``diamond``: the agent, exactly as :meth:`DiamondAgent.train` runs it
  (Algorithm 1): one step is one epoch of collection then diffusion, reward
  and actor-critic updates. Progress is epochs.
* ``diamond-wm``: only the world model (the EDM denoiser and the
  reward/termination model) from a dataset of episodes with discrete
  actions. No environment, no policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from simulusion.configs.diamond_config import DiamondConfig
from simulusion.training.task import Metrics, TrainingTask, register_task
from simulusion.training.tasks.common import (
    SequenceDataConfig,
    mean_metrics,
    sequence_loaders,
)


@dataclass
class DiamondTaskConfig:
    """``model``: DIAMOND's settings (game, architecture, schedule).

    Attributes:
        model: :class:`~simulusion.configs.diamond_config.DiamondConfig`. Its
            ``num_epochs`` is the default budget. ``device`` and ``seed`` come
            from the trainer config instead.
        eval_episodes: Episodes per evaluation.
    """

    model: DiamondConfig = field(default_factory=DiamondConfig)
    eval_episodes: int = 1


def _for_trainer(config: DiamondConfig, trainer: Any) -> DiamondConfig:
    config.device = str(trainer.device)
    config.seed = int(trainer.config.seed)
    return config


@register_task("diamond")
class DiamondTask(TrainingTask):
    """DIAMOND on Atari (or any env with the DIAMOND wrapper's API, as ``env``).

    Args:
        config: The task config.
        env: The environment to build DiamondAgent with, instead of Atari.
        agent: Train this :class:`~simulusion.training.train_diamond.
            DiamondAgent` instead of building one (its environment, replay
            and networks, at its own precision). This is how
            :meth:`DiamondAgent.train` runs; its ``checkpoint_<epoch>.pt``
            files are then also written whenever the trainer checkpoints.
    """

    config_cls = DiamondTaskConfig
    progress_unit = "epochs"

    def __init__(
        self,
        config: Any = None,
        *,
        env: Optional[Any] = None,
        agent: Optional[Any] = None,
    ) -> None:
        super().__init__(config)
        self._env = env
        self._adopted = agent

    def max_steps(self) -> int:
        return int(self.config.model.num_epochs)

    def setup(self, trainer: Any) -> None:
        from simulusion.training.train_diamond import DiamondAgent

        self.epoch = 0
        if self._adopted is not None:
            self.agent = self._adopted
            return
        self.agent = DiamondAgent(
            _for_trainer(self.config.model, trainer), env=self._env
        )
        self.agent.set_precision(trainer.precision)
        self.epoch = 0

    def progress(self) -> int:
        return self.epoch

    def train_step(self) -> Metrics:
        metrics = self.agent.train_epoch()
        self.epoch += 1
        out: dict[str, float] = {"replay/size": float(self.agent.replay_buffer.size)}
        if metrics is not None:
            out.update(metrics)
        return out

    def evaluate(self) -> Metrics:
        score = self.agent.evaluate(num_episodes=self.config.eval_episodes)
        return {
            "eval_reward": float(score),
            "eval_human_normalized": float(
                self.agent._compute_human_normalized_score(score)
            ),
        }

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model import as_world_model

        return as_world_model(self.agent)

    def save_checkpoint(self, directory: Path) -> None:
        # Writes agent.pt plus agent_replay.npz and agent_obs.npy beside it.
        self.agent.save_checkpoint(directory / "agent.pt")
        torch.save(
            {
                "epoch": self.epoch,
                "updates": {
                    name: getattr(self.agent, f"{name}_update").state_dict()
                    for name in ("diffusion", "reward", "actor")
                },
            },
            directory / "task.pt",
        )
        if self._adopted is not None:
            # Into config.checkpoint_dir, where DiamondAgent.train put them.
            self.agent.save_checkpoint(f"checkpoint_{self.epoch}.pt")

    def load_checkpoint(self, directory: Path) -> None:
        self.agent.load_checkpoint(str(directory / "agent.pt"))
        state = torch.load(
            directory / "task.pt", map_location=self.trainer.device, weights_only=True
        )
        for name, update in state["updates"].items():
            getattr(self.agent, f"{name}_update").load_state_dict(update)
        self.epoch = int(state["epoch"])

    def close(self) -> None:
        if self._adopted is not None:
            return  # the caller's DiamondAgent keeps using its environment
        close = getattr(self.agent.env, "close", None)
        if callable(close):
            close()


# ------------------------------------------------------------- world model


class _NoEnv:
    """Stands in for the environment when DIAMOND trains from a dataset.

    DiamondAgent reads only the action space from its environment while
    building; nothing here is ever stepped.
    """

    def __init__(self, n: int) -> None:
        from gymnasium.spaces import Discrete

        self.action_space: Any = Discrete(n)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("diamond-wm trains from a dataset and has no environment.")

    step = reset


@dataclass
class DiamondWorldModelTaskConfig:
    """``model``: DIAMOND's architecture and optimizer; ``data``: the episodes.

    Windows are ``burn_in_length + imagination_horizon + 1`` frames, the same
    sequences DIAMOND's replay trains on: the denoiser predicts the last frame
    from the ``num_conditioning_frames`` before it and their actions, and the
    reward/termination model reads the whole window. Frames are resized to
    ``obs_size``. Actions must be discrete; episodes without rewards or
    terminations train those heads on zeros.

    Attributes:
        action_dim: Number of discrete actions; the dataset's ``meta.json``
            when unset.
    """

    model: DiamondConfig = field(default_factory=DiamondConfig)
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)
    action_dim: Optional[int] = None


@register_task("diamond-wm")
class DiamondWorldModelTask(TrainingTask):
    """DIAMOND's denoiser and reward model trained offline, without a policy.

    Data-parallel: gradients are averaged across processes before each step.
    """

    config_cls = DiamondWorldModelTaskConfig

    def max_steps(self) -> int:
        return 50_000

    def setup(self, trainer: Any) -> None:
        from simulusion.training.train_diamond import DiamondAgent

        cfg = self.config
        model = _for_trainer(cfg.model, trainer)
        self.batches, self.val, meta = sequence_loaders(
            cfg.data,
            trainer,
            sequence_length=model.burn_in_length + model.imagination_horizon + 1,
            image_size=model.obs_size,
            require_actions=True,
        )
        n = cfg.action_dim or (meta.action_dim if meta else None)
        if n is None:
            raise ValueError(
                "DIAMOND needs the number of discrete actions: set action_dim, or "
                "write the dataset with a meta.json (EpisodeWriter does)."
            )
        self.agent = DiamondAgent(model, env=_NoEnv(int(n)))
        self.agent.set_precision(trainer.precision)

    @staticmethod
    def _batch(batch: dict) -> dict:
        """A window in the layout DIAMOND's replay SequenceDataset produces.

        A window of ``S + 1`` frames holds ``S`` transitions: ``obs_seq`` is
        frames ``0..S-1`` with the actions taken at them, and ``next_obs`` is
        frame ``S``, the successor of the last conditioning frame.
        """
        from simulusion.datasets.diamond_dataset import to_model_domain

        actions = batch["actions"]
        if torch.is_floating_point(actions):
            raise ValueError(
                "DIAMOND needs discrete actions; this dataset's are continuous."
            )
        frames = batch["frames"]  # (B, S+1, C, H, W) uint8
        B, S1 = frames.shape[:2]
        zeros = torch.zeros(B, S1 - 1, device=frames.device)
        return {
            "obs_seq": to_model_domain(frames[:, :-1]),
            "next_obs": to_model_domain(frames[:, -1]),
            "action_seq": actions.long(),
            "rewards": batch.get("rewards", zeros).float(),
            # DIAMOND's termination head models true episode ends.
            "dones": batch.get("terminated", zeros.bool()).bool(),
        }

    def train_step(self) -> Metrics:
        batch = self._batch(next(self.batches))
        diffusion = self.agent._update_diffusion_model(batch)
        reward = self.agent._update_reward_model(batch)
        return {"diffusion_loss": float(diffusion), "reward_loss": float(reward)}

    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        agent = self.agent
        rows = []
        for i, raw in enumerate(self.val):
            if i >= self.config.data.val_batches:
                break
            batch = self._batch(raw)
            with torch.no_grad():
                agent.reward_model.eval()
                logits, term_logits, _ = agent.reward_model(
                    obs=batch["obs_seq"], actions=batch["action_seq"]
                )
                n = max(0, logits.shape[1] - 1)
                total, reward_loss, term_loss = agent.reward_loss_fn(
                    reward_logits=logits[:, :-1],
                    termination_logits=term_logits[:, :-1],
                    rewards=batch["rewards"][:, :n],
                    terminated=batch["dones"][:, :n],
                )
            rows.append({"val_reward_loss": float(total)})
        return mean_metrics(rows)

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model import as_world_model

        return as_world_model(self.agent)

    def state_dict(self) -> dict[str, Any]:
        agent = self.agent
        return {
            "diffusion_model": agent.diffusion_model.state_dict(),
            "reward_model": agent.reward_model.state_dict(),
            "diffusion": agent.diffusion_update.state_dict(),
            "reward": agent.reward_update.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        agent = self.agent
        agent.diffusion_model.load_state_dict(state["diffusion_model"])
        agent.reward_model.load_state_dict(state["reward_model"])
        agent.diffusion_update.load_state_dict(state["diffusion"])
        agent.reward_update.load_state_dict(state["reward"])


__all__ = [
    "DiamondTask",
    "DiamondTaskConfig",
    "DiamondWorldModelTask",
    "DiamondWorldModelTaskConfig",
]
