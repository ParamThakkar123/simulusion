"""IRIS on the shared trainer: the Atari 100k agent, or its world model alone.

* ``iris``: the agent, exactly as :class:`~simulusion.training.train_iris.
  IRISTrainer` runs it (paper Algorithm 1): one step is one epoch of
  collection, then autoencoder, transformer and actor-critic updates on
  their warm-start schedules. Progress is epochs. The final evaluation is the
  paper's benchmark number (mean return over ``eval_episodes`` at the end of
  training); periodic ones are monitoring only.
* ``iris-wm``: only the world model (discrete autoencoder and transformer),
  from a dataset of episodes with discrete actions. Each step updates both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import torch

from simulusion.configs.iris_config import IRISConfig
from simulusion.data.sequence import frames_to_float, one_hot_actions
from simulusion.training.task import Metrics, TrainingTask, register_task
from simulusion.training.tasks.common import (
    SequenceDataConfig,
    load_replay,
    mean_metrics,
    save_replay,
    sequence_loaders,
)

logger = logging.getLogger(__name__)


@dataclass
class IRISTaskConfig:
    """``model``: IRIS's hyperparameters (the paper's by default).

    Attributes:
        model: :class:`~simulusion.configs.iris_config.IRISConfig`. Its
            ``total_epochs`` is the default budget and ``env`` the game.
        game: Environment id; ``model.env`` when unset.
        eval_episodes: Episodes per periodic (monitoring) evaluation;
            ``model.eval_episodes`` when unset. The final evaluation always
            uses ``model.eval_episodes``.
        save_replay: Keep the replay buffer in checkpoints, so a resumed run
            trains on the same data. The environment itself cannot be saved:
            collection resumes from a fresh episode.
    """

    model: IRISConfig = field(default_factory=IRISConfig)
    game: Optional[str] = None
    eval_episodes: Optional[int] = None
    save_replay: bool = True


@register_task("iris")
class IRISTask(TrainingTask):
    """IRIS on Atari (or any image environment passed as ``env``)."""

    config_cls = IRISTaskConfig
    progress_unit = "epochs"

    def __init__(self, config: Any = None, *, env: Optional[Any] = None) -> None:
        super().__init__(config)
        self._env = env

    def max_steps(self) -> int:
        return int(self.config.model.total_epochs)

    def setup(self, trainer: Any) -> None:
        from simulusion.training.train_iris import IRISTrainer

        cfg = self.config
        self.iris = IRISTrainer(
            game=cfg.game or cfg.model.env,
            device=str(trainer.device),
            seed=trainer.config.seed,
            config=cfg.model,
            env=self._env,
        )
        self.iris.agent.set_precision(trainer.precision)
        self.epoch = 0

    def progress(self) -> int:
        return self.epoch

    def train_step(self) -> Metrics:
        metrics = self.iris.train_epoch(self.epoch)
        self.epoch += 1
        return {k: float(v) for k, v in metrics.items()}

    def evaluate(self) -> Metrics:
        # IRISTrainer only evaluates once the policy has started training.
        if self.epoch <= self.config.model.start_actor_critic_after:
            return {}
        episodes = self.config.eval_episodes or self.config.model.eval_episodes
        return cast(dict, self.iris.evaluate(num_episodes=episodes))

    def final_evaluation(self) -> Metrics:
        return cast(
            dict, self.iris.evaluate(num_episodes=self.config.model.eval_episodes)
        )

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model import as_world_model

        return as_world_model(self.iris.agent)

    def save_checkpoint(self, directory: Path) -> None:
        agent = self.iris.agent
        agent.save(str(directory / "agent.pt"))
        hidden = self.iris._collect_hidden
        torch.save(
            {
                "epoch": self.epoch,
                "updates": {
                    name: getattr(agent, f"{name}_update").state_dict()
                    for name in ("autoencoder", "transformer", "ac")
                },
                "env_steps": self.iris.env_steps,
                "last_episode_return": self.iris._last_episode_return,
                "hidden": list(hidden) if hidden is not None else None,
            },
            directory / "task.pt",
        )
        if self.config.save_replay:
            save_replay(self.iris.replay_buffer, directory / "replay.npz")

    def load_checkpoint(self, directory: Path) -> None:
        agent = self.iris.agent
        agent.load(str(directory / "agent.pt"))
        state = torch.load(
            directory / "task.pt", map_location=self.trainer.device, weights_only=True
        )
        for name, update in state["updates"].items():
            getattr(agent, f"{name}_update").load_state_dict(update)
        self.epoch = int(state["epoch"])
        self.iris.env_steps = int(state["env_steps"])
        self.iris._last_episode_return = float(state["last_episode_return"])
        if (directory / "replay.npz").exists():
            load_replay(self.iris.replay_buffer, directory / "replay.npz")
        else:
            logger.warning(
                "The checkpoint has no replay buffer; IRIS resumes with an empty one."
            )
        # The environment's own state is not checkpointed: the next collection
        # starts a new episode (with a fresh policy state) rather than
        # continuing the one that was in flight.
        self.iris._reset_collection_state()

    def close(self) -> None:
        for env in (self.iris.env, self.iris._eval_env):
            close = getattr(env, "close", None)
            if callable(close):
                close()


# ------------------------------------------------------------- world model


@dataclass
class IRISWorldModelTaskConfig:
    """``model``: IRIS's architecture and optimizer; ``data``: the episodes.

    Windows are ``transformer_timesteps + 1`` frames (IRIS's replay samples
    the same), resized to the configured frame size. Actions must be
    discrete. The autoencoder trains on ``autoencoder_batch_size`` frames
    drawn from each batch's windows. Episodes without rewards or terminations
    train those heads on zeros.

    Attributes:
        action_dim: Number of discrete actions; the dataset's ``meta.json``
            when unset.
    """

    model: IRISConfig = field(default_factory=IRISConfig)
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)
    action_dim: Optional[int] = None


@register_task("iris-wm")
class IRISWorldModelTask(TrainingTask):
    """IRIS's tokenizer and transformer trained offline, without a policy.

    Data-parallel: gradients are averaged across processes before each step.
    """

    config_cls = IRISWorldModelTaskConfig
    distributed = True

    def max_steps(self) -> int:
        return 10_000

    def setup(self, trainer: Any) -> None:
        from simulusion.models.iris_agent import IRISAgent

        cfg = self.config
        model = cfg.model
        self.batches, self.val, meta = sequence_loaders(
            cfg.data,
            trainer,
            sequence_length=model.transformer_timesteps + 1,
            image_size=(model.frame_height, model.frame_width),
            require_actions=True,
        )
        n = cfg.action_dim or (meta.action_dim if meta else None)
        if n is None:
            raise ValueError(
                "IRIS needs the number of discrete actions: set action_dim, or "
                "write the dataset with a meta.json (EpisodeWriter does)."
            )
        self.action_dim = int(n)
        self.agent = IRISAgent(
            config=model, action_size=self.action_dim, device=trainer.device
        )
        self.agent.set_precision(
            trainer.precision, all_reduce=trainer.num_processes > 1
        )
        trainer.broadcast_parameters(
            [self.agent.encoder, self.agent.decoder, self.agent.transformer]
        )

    def _tensors(self, batch: dict) -> tuple:
        actions = batch["actions"]
        if torch.is_floating_point(actions):
            raise ValueError(
                "IRIS needs discrete actions; this dataset's are continuous."
            )
        frames = frames_to_float(batch["frames"])  # (B, T+1, C, H, W)
        B, T1 = frames.shape[:2]
        zeros = torch.zeros(B, T1 - 1, device=frames.device)
        acts = one_hot_actions(actions, self.action_dim)
        rewards = batch.get("rewards", zeros).float()
        # IRIS's termination head models true episode ends, not time limits.
        terms = batch.get("terminated", zeros.bool()).long()
        return frames, acts, rewards, terms

    def train_step(self) -> Metrics:
        frames, acts, rewards, terms = self._tensors(next(self.batches))
        flat = frames.flatten(0, 1)
        k = min(self.config.model.autoencoder_batch_size, flat.shape[0])
        pick = torch.randperm(flat.shape[0], device=flat.device)[:k]
        ae = self.agent.update_autoencoder(flat[pick])
        self.trainer.broadcast_quantizers(self.agent.encoder)
        tf = self.agent.update_transformer(frames, acts, rewards, terms)
        return {
            "recon_loss": ae["recon_loss"],
            "vq_loss": ae["vq_loss"],
            "perceptual_loss": ae["perceptual_loss"],
            "perplexity": ae["perplexity"],
            "token_loss": tf["token_loss"],
            "reward_loss": tf["reward_loss"],
            "term_loss": tf["term_loss"],
        }

    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        import torch.nn.functional as F

        agent = self.agent
        rows = []
        modules: tuple[torch.nn.Module, ...] = (
            agent.encoder,
            agent.decoder,
            agent.transformer,
        )
        modes = [m.training for m in modules]
        for m in modules:
            m.eval()
        try:
            with torch.no_grad():
                for i, batch in enumerate(self.val):
                    if i >= self.config.data.val_batches:
                        break
                    frames, acts, _, _ = self._tensors(batch)
                    flat = frames.flatten(0, 1)
                    z_q, indices, _ = agent.encoder(flat)
                    recon = F.l1_loss(agent.decoder(z_q), flat)
                    tokens = indices.reshape(frames.shape[0], frames.shape[1], -1)
                    logits, _, _ = agent.transformer(tokens, acts.argmax(-1))
                    token_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
                    )
                    rows.append(
                        {
                            "val_recon_loss": float(recon),
                            "val_token_loss": float(token_loss),
                        }
                    )
        finally:
            for m, mode in zip(modules, modes):
                m.train(mode)
        return mean_metrics(rows)

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model import as_world_model

        return as_world_model(self.agent)

    def save_checkpoint(self, directory: Path) -> None:
        self.agent.save(str(directory / "agent.pt"))
        torch.save(
            {
                "updates": {
                    name: getattr(self.agent, f"{name}_update").state_dict()
                    for name in ("autoencoder", "transformer")
                }
            },
            directory / "task.pt",
        )

    def load_checkpoint(self, directory: Path) -> None:
        self.agent.load(str(directory / "agent.pt"))
        state = torch.load(
            directory / "task.pt", map_location=self.trainer.device, weights_only=True
        )
        for name, update in state["updates"].items():
            getattr(self.agent, f"{name}_update").load_state_dict(update)


__all__ = [
    "IRISTask",
    "IRISTaskConfig",
    "IRISWorldModelTask",
    "IRISWorldModelTaskConfig",
]
