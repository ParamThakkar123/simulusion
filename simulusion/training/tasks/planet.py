"""PlaNet on the shared trainer: an RSSM learned online, acted on by CEM planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from simulusion.training.task import Metrics, TrainingTask, register_task
from simulusion.training.tasks.common import mean_metrics


@dataclass
class PlanetConfig:
    """PlaNet's environment, model, planner and training settings.

    The defaults are those of :class:`~simulusion.models.planet.Planet` and
    ``Planet.train``.
    """

    env: str = "Pendulum-v1"
    bit_depth: int = 5
    action_repeats: int = 1
    max_episode_steps: Optional[int] = None
    state_size: int = 200
    latent_size: int = 30
    embedding_size: int = 1024
    memory_size: int = 100
    planning_horizon: int = 20
    num_candidates: int = 1000
    num_iterations: int = 10
    top_candidates: int = 100
    warmup_episodes: int = 1
    epochs: int = 100
    steps_per_epoch: int = 150
    batch_size: int = 32
    chunk_length: int = 50
    beta: float = 1.0
    scheduler_type: Optional[str] = "step"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanetTaskConfig:
    """``model``: everything PlaNet needs; ``save_replay`` keeps episodes in checkpoints."""

    model: PlanetConfig = field(default_factory=PlanetConfig)
    save_replay: bool = True


@register_task("planet")
class PlanetTask(TrainingTask):
    """PlaNet, one epoch per step, as ``Planet.train`` runs it.

    An epoch is ``steps_per_epoch`` gradient steps on replayed chunks, then one
    exploration episode with the CEM planner and one evaluation episode, both
    added to the replay. Progress is epochs.
    """

    config_cls = PlanetTaskConfig
    progress_unit = "epochs"

    def max_steps(self) -> int:
        return int(self.config.model.epochs)

    def setup(self, trainer: Any) -> None:
        from simulusion.models.planet import Planet, make_planet_scheduler

        cfg = self.config.model
        self.planet = Planet(
            cfg.env,
            bit_depth=cfg.bit_depth,
            device=trainer.device,
            state_size=cfg.state_size,
            latent_size=cfg.latent_size,
            embedding_size=cfg.embedding_size,
            memory_size=cfg.memory_size,
            policy_cfg={
                "planning_horizon": cfg.planning_horizon,
                "num_candidates": cfg.num_candidates,
                "num_iterations": cfg.num_iterations,
                "top_candidates": cfg.top_candidates,
            },
            headless=True,
            max_episode_steps=cfg.max_episode_steps,
            action_repeats=cfg.action_repeats,
        )
        self.update = trainer.gradient_update(
            self.planet.optimizer, self.planet.rssm.parameters(), max_grad_norm=1000.0
        )
        self.scheduler = make_planet_scheduler(
            self.planet.optimizer, cfg.scheduler_type, cfg.scheduler_kwargs, cfg.epochs
        )
        self.epoch = 0

    def progress(self) -> int:
        return self.epoch

    def train_step(self) -> Metrics:
        from simulusion.training.train_planet import train as planet_train
        from simulusion.utils.utils import flatten_dict

        cfg = self.config.model
        planet = self.planet
        if len(planet.memory.episodes) == 0:
            planet.warmup(n_episodes=cfg.warmup_episodes, random_policy=True)
        rows = []
        for _ in range(cfg.steps_per_epoch):
            out = planet_train(
                planet.memory,
                planet.rssm.train(),
                planet.optimizer,
                self.trainer.device,
                N=cfg.batch_size,
                H=cfg.chunk_length,
                beta=cfg.beta,
                update=self.update,
            )
            rows.append(flatten_dict(out))
        metrics = mean_metrics(rows)
        metrics["learning_rate"] = float(planet.optimizer.param_groups[0]["lr"])
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(
                    sum(v for k, v in metrics.items() if k.startswith("losses/"))
                )
            else:
                self.scheduler.step()
        planet.memory.append([planet.rollout_gen.rollout_once(explore=True)])
        episode, _, eval_metrics, _ = planet.rollout_gen.rollout_eval()
        planet.memory.append([episode])
        metrics.update(
            {
                k: float(v)  # type: ignore[arg-type]
                for k, v in flatten_dict(eval_metrics).items()
                if np.isscalar(v)
            }
        )
        self.epoch += 1
        return metrics

    def evaluate(self) -> Metrics:
        _, _, eval_metrics, _ = self.planet.rollout_gen.rollout_eval()
        from simulusion.utils.utils import flatten_dict

        return {
            k: float(v)  # type: ignore[arg-type]
            for k, v in flatten_dict(eval_metrics).items()
            if np.isscalar(v)
        }

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model import as_world_model

        return as_world_model(self.planet, bit_depth=self.config.model.bit_depth)

    def save_checkpoint(self, directory: Path) -> None:
        torch.save(
            {
                "rssm": self.planet.rssm.state_dict(),
                "update": self.update.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.epoch,
            },
            directory / "task.pt",
        )
        if self.config.save_replay:
            save_planet_memory(self.planet.memory, directory / "replay")

    def load_checkpoint(self, directory: Path) -> None:
        state = torch.load(
            directory / "task.pt", map_location=self.trainer.device, weights_only=True
        )
        self.planet.rssm.load_state_dict(state["rssm"])
        self.update.load_state_dict(state["update"])
        if self.scheduler is not None and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = int(state["epoch"])
        if (directory / "replay").is_dir():
            load_planet_memory(self.planet.memory, directory / "replay")


def save_planet_memory(memory: Any, directory: Path) -> None:
    """Each finished PlaNet episode as ``episode_<i>.npz`` (x, u, r, t)."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, episode in enumerate(memory.episodes):
        np.savez(
            directory / f"episode_{i:06d}.npz",
            x=np.asarray(episode.x),
            u=np.asarray(episode.u),
            r=np.asarray(episode.r),
            t=np.asarray(episode.t),
        )


def load_planet_memory(memory: Any, directory: Path) -> None:
    from simulusion.memory.planet_memory import Episode

    memory.episodes.clear()
    memory.eps_lengths.clear()
    for path in sorted(directory.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            episode = Episode()
            episode.x, episode.u, episode.r, episode.t = (
                data["x"],
                data["u"],
                data["r"],
                data["t"],
            )
            episode._size = int(episode.u.shape[0])
        memory.append([episode])


__all__ = ["PlanetConfig", "PlanetTask", "PlanetTaskConfig"]
