"""Ha & Schmidhuber's World Models on the shared trainer, one stage per task.

The three stages train separately, each from the previous one's output:

1. ``vae``: a ConvVAE on frames from an episode folder (V).
2. ``mdrnn``: an MDN-RNN on the VAE's latents of the same episodes, with
   their actions, rewards and episode ends (M). Exports the V + M world model.
3. ``controller``: a linear controller (C) evolved with CMA-ES, scored in the
   real environment or entirely inside the world model's dream (the paper's
   "training inside of the dream" experiment).

.. code-block:: bash

    simulusion train vae data.path=data/car
    simulusion train mdrnn data.path=data/car vae=runs/vae_.../model/vae.tar
    simulusion train controller world_model=runs/mdrnn_.../world_model mode=dream
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from simulusion.configs.wm_config import (
    WMControllerConfig,
    WMMDNRNNConfig,
    WMVAEConfig,
)
from simulusion.data.sequence import frames_to_float, one_hot_actions
from simulusion.training.task import Metrics, TrainingTask, register_task
from simulusion.training.tasks.common import (
    SequenceDataConfig,
    mean_metrics,
    sequence_loaders,
)

logger = logging.getLogger(__name__)

#: The ConvVAE's fixed input size.
VAE_SIZE = 64


def _vae_config() -> WMVAEConfig:
    return WMVAEConfig(height=VAE_SIZE, width=VAE_SIZE)


@dataclass
class VAETaskConfig:
    """``model``: the VAE's settings; ``data``: episodes whose frames it learns.

    Frames are resized to 64x64 (the ConvVAE's input) and, as in
    ``train_convae``, flipped horizontally half the time when ``flip`` is on.
    Evaluation reports held-out loss when ``data.val_fraction`` is set, and
    drives ``ReduceLROnPlateau`` with it.
    """

    model: WMVAEConfig = field(default_factory=_vae_config)
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)
    flip: bool = True


@register_task("vae")
class VAETask(TrainingTask):
    """World Models' V: a ConvVAE on single frames. Exports ``model/vae.tar``."""

    config_cls = VAETaskConfig
    distributed = True

    def max_steps(self) -> int:
        return 20_000

    def setup(self, trainer: Any) -> None:
        from simulusion.utils.train_utils import ReduceLROnPlateau
        from simulusion.vision.VAE.ConvVAE import ConvVAE

        cfg = self.config.model
        self.batches, self.val, _ = sequence_loaders(
            self.config.data, trainer, sequence_length=1, image_size=VAE_SIZE
        )
        self.vae = trainer.prepare_module(
            ConvVAE(img_channels=3, latent_size=cfg.latent_size)
        )
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=cfg.learning_rate)
        self.update = trainer.gradient_update(
            optimizer, self.vae.parameters(), modules=[self.vae]
        )
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )

    def _loss(self, frames: torch.Tensor) -> torch.Tensor:
        from simulusion.losses.convae_loss import conv_vae_loss_fn

        recon, mu, logvar = self.vae(frames)
        # conv_vae_loss_fn sums over the batch; per-frame keeps the scale
        # independent of the batch size.
        return conv_vae_loss_fn(recon, frames, mu, logvar) / frames.shape[0]

    def train_step(self) -> Metrics:
        frames = frames_to_float(next(self.batches)["frames"][:, 0])
        if self.config.flip:
            flip = torch.rand(frames.shape[0], device=frames.device) < 0.5
            frames = torch.where(flip[:, None, None, None], frames.flip(-1), frames)
        self.vae.train()
        with self.trainer.precision.autocast():
            loss = self._loss(frames)
        self.update(loss)
        return {"loss": float(loss.detach())}

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        self.vae.eval()
        rows = []
        for i, batch in enumerate(self.val):
            if i >= self.config.data.val_batches:
                break
            rows.append(
                {"val_loss": float(self._loss(frames_to_float(batch["frames"][:, 0])))}
            )
        metrics = mean_metrics(rows)
        if metrics:
            self.scheduler.step(metrics["val_loss"])
        return metrics

    def export(self, directory: Path) -> Optional[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "vae.tar"
        torch.save({"state_dict": self.trainer.unwrap(self.vae).state_dict()}, path)
        return directory

    def state_dict(self) -> dict[str, Any]:
        return {
            "vae": self.trainer.unwrap(self.vae).state_dict(),
            "update": self.update.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.trainer.unwrap(self.vae).load_state_dict(state["vae"])
        self.update.load_state_dict(state["update"])
        self.scheduler.load_state_dict(state["scheduler"])


# ----------------------------------------------------------------- MDN-RNN


@dataclass
class MDRNNTaskConfig:
    """``model``: the MDN-RNN's settings; ``data``: the episodes; ``vae``: V.

    Windows are ``model.seq_len + 1`` frames. Each is encoded by the frozen
    VAE, sampling ``z = mu + sigma * eps`` as ``train_mdn_rnn`` does, and the
    MDN-RNN learns ``p(z_{t+1} | z_t, a_t, h_t)`` plus the reward and done
    heads. Discrete actions are one-hot encoded to ``model.action_size``;
    continuous ones set it from the data.

    Attributes:
        vae: The trained VAE: a ``vae.tar`` from the ``vae`` task, or
            ``train_convae``'s ``best.tar``.
    """

    model: WMMDNRNNConfig = field(default_factory=WMMDNRNNConfig)
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)
    vae: Optional[str] = None


@register_task("mdrnn")
class MDRNNTask(TrainingTask):
    """World Models' M. Exports the V + M world model and ``model/mdrnn.tar``."""

    config_cls = MDRNNTaskConfig
    distributed = True

    def max_steps(self) -> int:
        return 20_000

    def setup(self, trainer: Any) -> None:
        from simulusion.models.mdrnn import MDRNN
        from simulusion.utils.train_utils import ReduceLROnPlateau
        from simulusion.vision.VAE.ConvVAE import ConvVAE

        cfg = self.config.model
        if not self.config.vae:
            raise ValueError(
                "mdrnn trains on a VAE's latents: set vae=<path to vae.tar>."
            )
        weights = torch.load(self.config.vae, map_location="cpu", weights_only=True)
        weights = weights.get("state_dict", weights)
        # The latent size is the VAE's, whatever the config says.
        cfg.latent_size = int(weights["encoder.fc_mu.weight"].shape[0])
        self.vae = ConvVAE(img_channels=3, latent_size=cfg.latent_size)
        self.vae.load_state_dict(weights)
        self.vae.to(trainer.device).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.batches, self.val, meta = sequence_loaders(
            self.config.data,
            trainer,
            sequence_length=cfg.seq_len + 1,
            image_size=VAE_SIZE,
            require_actions=True,
        )
        if meta is not None and meta.action_dim:
            cfg.action_size = int(meta.action_dim)
        self.mdrnn = trainer.prepare_module(
            MDRNN(
                latents=cfg.latent_size,
                actions=cfg.action_size,
                hiddens=cfg.hidden_size,
                gaussians=cfg.gmm_components,
            )
        )
        optimizer = torch.optim.RMSprop(
            self.mdrnn.parameters(), lr=cfg.learning_rate, alpha=0.9
        )
        self.update = trainer.gradient_update(
            optimizer, self.mdrnn.parameters(), modules=[self.mdrnn]
        )
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )

    def _losses(self, batch: dict) -> dict[str, torch.Tensor]:
        from simulusion.training.train_mdn_rnn import get_loss

        cfg = self.config.model
        frames = frames_to_float(batch["frames"])  # (B, T+1, C, 64, 64)
        B, T1 = frames.shape[:2]
        with torch.no_grad():
            _, mu, logsigma = self.vae(frames.flatten(0, 1))
            z = (mu + logsigma.exp() * torch.randn_like(logsigma)).reshape(B, T1, -1)
        actions = one_hot_actions(batch["actions"], cfg.action_size).float()
        actions = actions.reshape(B, T1 - 1, -1)
        zeros = torch.zeros(B, T1 - 1, device=frames.device)
        rewards = batch.get("rewards", zeros).float()
        terminal = batch.get("terminated", zeros.bool()).float()
        return get_loss(
            self.mdrnn,
            z[:, :-1],
            actions,
            rewards,
            terminal,
            z[:, 1:],
            cfg.include_reward,
            cfg.latent_size,
        )

    def train_step(self) -> Metrics:
        self.mdrnn.train()
        with self.trainer.precision.autocast():
            losses = self._losses(next(self.batches))
        self.update(losses["loss"])
        return {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        self.mdrnn.eval()
        rows = []
        for i, batch in enumerate(self.val):
            if i >= self.config.data.val_batches:
                break
            rows.append({f"val_{k}": float(v) for k, v in self._losses(batch).items()})
        metrics = mean_metrics(rows)
        if metrics:
            self.scheduler.step(metrics["val_loss"])
        return metrics

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model.vae_mdrnn import MDRNNWorldModel

        return MDRNNWorldModel(self.vae, self.trainer.unwrap(self.mdrnn))

    def export(self, directory: Path) -> Optional[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "mdrnn.tar"
        torch.save({"state_dict": self.trainer.unwrap(self.mdrnn).state_dict()}, path)
        return directory

    def state_dict(self) -> dict[str, Any]:
        return {
            "mdrnn": self.trainer.unwrap(self.mdrnn).state_dict(),
            "update": self.update.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.trainer.unwrap(self.mdrnn).load_state_dict(state["mdrnn"])
        self.update.load_state_dict(state["update"])
        self.scheduler.load_state_dict(state["scheduler"])


# -------------------------------------------------------------- controller


@dataclass
class ControllerTaskConfig:
    """``model``: CMA-ES settings; ``world_model``: the V + M to act through.

    Attributes:
        model: Population size, samples per candidate, episode length
            (``time_limit``), target return and ``env_name``. Latent, hidden
            and action sizes are read from the world model instead.
        world_model: A saved V + M world model (the ``mdrnn`` task's
            ``<run>/world_model``).
        mode: ``"env"`` scores candidates in the real environment through V
            and M's features; ``"dream"`` scores them inside the world model,
            starting from a real first frame, with M's reward and done heads.
        sigma: CMA-ES initial step size.
        temperature: Dream sampling temperature (the paper's tau).
        env_kwargs: Extra ``gymnasium.make`` arguments (e.g.
            ``{continuous: true}`` for CarRacing).
    """

    model: WMControllerConfig = field(default_factory=WMControllerConfig)
    world_model: Optional[str] = None
    mode: str = "env"
    sigma: float = 0.1
    temperature: float = 1.0
    env_kwargs: dict[str, Any] = field(default_factory=dict)


@register_task("controller")
class ControllerTask(TrainingTask):
    """World Models' C: a linear policy on ``[z, h]``, one CMA-ES generation per step.

    A resumed run restarts CMA-ES from the saved mean and step size (the
    covariance estimate is not kept).
    """

    config_cls = ControllerTaskConfig
    progress_unit = "generations"
    default_max_steps = 100

    def __init__(
        self, config: Any = None, *, env_factory: Optional[Any] = None
    ) -> None:
        super().__init__(config)
        self._env_factory = env_factory

    def setup(self, trainer: Any) -> None:
        import cma

        from simulusion.models.controller import Controller
        from simulusion.world_model import load_world_model

        cfg = self.config
        if cfg.mode not in ("env", "dream"):
            raise ValueError("mode must be 'env' or 'dream'.")
        if not cfg.world_model:
            raise ValueError(
                "controller acts through a world model: set world_model=<dir>."
            )
        self.wm: Any = load_world_model(
            cfg.world_model, device=trainer.device, temperature=cfg.temperature
        )
        if self.wm.spec.family != "world-models":
            raise ValueError(
                f"{cfg.world_model} is a {self.wm.spec.family} world model, not V + M."
            )
        cell: Any = getattr(self.wm, "cell")  # MDRNNWorldModel's single-step MDN-RNN
        self.controller = Controller(cell.latents, cell.hiddens, cell.actions).to(
            trainer.device
        )
        self.n_params = sum(p.numel() for p in self.controller.parameters())
        self.es = cma.CMAEvolutionStrategy(
            self._flat(),
            cfg.sigma,
            {"popsize": cfg.model.pop_size, "seed": trainer.config.seed + 1},
        )
        self.generation = 0
        self.best_return = float("-inf")
        self.best_params = self._flat()
        self.env = self._make_env()

    def _make_env(self) -> Any:
        if self._env_factory is not None:
            return self._env_factory()
        import gymnasium as gym

        from simulusion.envs.gym_env import GymImageEnv

        env = gym.make(self.config.model.env_name, **self.config.env_kwargs)
        return GymImageEnv(env=env, size=(VAE_SIZE, VAE_SIZE))

    def _flat(self) -> np.ndarray:
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.controller.parameters()]
        )

    def _load(self, flat: np.ndarray) -> None:
        offset = 0
        for p in self.controller.parameters():
            n = p.numel()
            p.data.copy_(
                torch.as_tensor(flat[offset : offset + n], dtype=p.dtype).view_as(p)
            )
            offset += n

    def progress(self) -> int:
        return self.generation

    @torch.no_grad()
    def _episode(self, flat: np.ndarray) -> float:
        """One episode's return for controller parameters ``flat``."""
        self._load(flat)
        wm, cfg = self.wm, self.config
        obs, _ = self.env.reset()
        frame = torch.as_tensor(
            np.asarray(obs["image"] if isinstance(obs, dict) else obs)
        )
        state = wm.encode(frame.unsqueeze(0))
        total = 0.0
        for _ in range(cfg.model.time_limit):
            action = self.controller(torch.cat([state["z"], state["h"]], dim=-1))
            if cfg.mode == "dream":
                out = wm.step(state, action)
                state = out.state
                total += float(out.reward[0])
                if bool(out.terminated[0]):
                    break
                continue
            # Real environment: M only tracks h from the actions taken.
            next_obs, reward, terminated, truncated, _ = self.env.step(
                action[0].cpu().numpy()
            )
            total += float(reward)
            if terminated or truncated:
                break
            out = wm.step(state, action)
            frame = torch.as_tensor(
                np.asarray(
                    next_obs["image"] if isinstance(next_obs, dict) else next_obs
                )
            )
            z = wm.encode(frame.unsqueeze(0))["z"]
            state = {"z": z, "h": out.state["h"], "c": out.state["c"]}
        return total

    def train_step(self) -> Metrics:
        cfg = self.config.model
        solutions = self.es.ask()
        returns = [
            float(np.mean([self._episode(s) for _ in range(cfg.n_samples)]))
            for s in solutions
        ]
        # CMA-ES minimises.
        self.es.tell(solutions, [-r for r in returns])
        best = int(np.argmax(returns))
        if returns[best] > self.best_return:
            self.best_return = returns[best]
            self.best_params = np.asarray(solutions[best])
        self.generation += 1
        return {
            "return_mean": float(np.mean(returns)),
            "return_max": float(np.max(returns)),
            "best_return": self.best_return,
            "sigma": float(self.es.sigma),
        }

    def evaluate(self) -> Metrics:
        mean = np.asarray(self.es.mean)
        returns = [self._episode(mean) for _ in range(self.config.model.n_samples)]
        return {
            "eval_return": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
        }

    def export(self, directory: Path) -> Optional[Path]:
        self._load(self.best_params)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "controller.tar"
        torch.save(
            {"state_dict": self.controller.state_dict(), "reward": self.best_return},
            path,
        )
        return directory

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": torch.as_tensor(np.asarray(self.es.mean)),
            "sigma": float(self.es.sigma),
            "best_params": torch.as_tensor(self.best_params),
            "best_return": self.best_return,
            "generation": self.generation,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        import cma

        self.es = cma.CMAEvolutionStrategy(
            state["mean"].numpy(),
            float(state["sigma"]),
            {
                "popsize": self.config.model.pop_size,
                "seed": self.trainer.config.seed + 1,
            },
        )
        self.best_params = state["best_params"].numpy()
        self.best_return = float(state["best_return"])
        self.generation = int(state["generation"])

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


__all__ = [
    "ControllerTask",
    "ControllerTaskConfig",
    "MDRNNTask",
    "MDRNNTaskConfig",
    "VAETask",
    "VAETaskConfig",
]
