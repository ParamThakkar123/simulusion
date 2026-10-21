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
from torch.utils.data import DataLoader

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


#: train_convae / train_mdn_rnn hold out this many rollout files for testing.
LEGACY_TEST_FILES = 600


def _vae_config() -> WMVAEConfig:
    return WMVAEConfig(height=VAE_SIZE, width=VAE_SIZE)


class _LegacyEpochs:
    """The older World Models entrypoints' way of running: whole epochs.

    One step is a pass over the training rollouts, evaluation is the loss over
    the held-out ones (the last ``LEGACY_TEST_FILES`` files, or the training
    rollouts when there are too few), and each checkpoint also writes their
    ``checkpoint.tar`` and, after an evaluation that improved, ``best.tar``.
    """

    def _legacy_loaders(
        self, trainer: Any, path: str, sequence_length: int, batch_size: int
    ) -> None:
        from simulusion.data.folder import EpisodeFolder
        from simulusion.data.sequence import SequenceDataset, collate_sequences

        folder = EpisodeFolder(path)
        n = len(folder)
        test = list(range(n - LEGACY_TEST_FILES, n)) if n > LEGACY_TEST_FILES else []
        train = list(range(n - len(test)))

        def loader(indices: list, shuffle: bool) -> Any:
            windows = SequenceDataset(
                EpisodeFolder(path, episodes=indices),
                sequence_length,
                image_size=VAE_SIZE,
            )
            return trainer.prepare_dataloader(
                DataLoader(
                    windows,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=collate_sequences,
                )
            )

        self.train_loader = loader(train, True)
        if not test:
            logger.warning(
                "%d rollouts are too few to hold %d out; validating on the training data.",
                n,
                LEGACY_TEST_FILES,
            )
        self.val = loader(test, False) if test else self.train_loader
        self._epochs_done = 0
        self._last_val: Optional[float] = None
        self._best_val: Optional[float] = None
        self._improved = False

    def _record_val(self, value: float) -> None:
        self._last_val = value
        self._improved = self._best_val is None or value < self._best_val
        if self._improved:
            self._best_val = value

    def _save_legacy(
        self, directory: Path, module: torch.nn.Module, optimizer: Any, scheduler: Any
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": self._epochs_done,
            "state_dict": module.state_dict(),
            "precision": self._last_val,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, directory / "checkpoint.tar")
        if self._improved:
            torch.save(state, directory / "best.tar")

    @staticmethod
    def _reload(
        directory: Path, module: torch.nn.Module, optimizer: Any, scheduler: Any
    ) -> None:
        best = directory / "best.tar"
        if not best.exists():
            return
        state = torch.load(best, map_location="cpu", weights_only=True)
        logger.info(
            "Reloading %s at epoch %s, with test error %s",
            best,
            state["epoch"],
            state["precision"],
        )
        module.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])


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
class VAETask(_LegacyEpochs, TrainingTask):
    """World Models' V: a ConvVAE on single frames. Exports ``model/vae.tar``.

    Args:
        config: The task config.
        legacy_dir: Run as ``train_convae`` does, writing its files here
            (``<logdir>/vae``): whole epochs over ``model.data_dir``, the
            ``checkpoint.tar`` / ``best.tar`` checkpoints, resuming from
            ``best.tar`` unless ``model.noreload``, and a grid of decoded
            samples every ``model.sample_interval`` epochs unless
            ``model.nosamples``.
    """

    config_cls = VAETaskConfig
    progress_unit = "steps"

    def __init__(self, config: Any = None, *, legacy_dir: Optional[str] = None) -> None:
        super().__init__(config)
        self._legacy_dir = Path(legacy_dir) if legacy_dir is not None else None
        if self._legacy_dir is not None:
            self.progress_unit = "epochs"  # type: ignore[misc]

    def max_steps(self) -> int:
        if self._legacy_dir is not None:
            return int(self.config.model.num_epochs)
        return 20_000

    def progress(self) -> int:
        if self._legacy_dir is not None:
            return self._epochs_done
        return super().progress()

    def setup(self, trainer: Any) -> None:
        from simulusion.utils.train_utils import ReduceLROnPlateau
        from simulusion.vision.VAE.ConvVAE import ConvVAE

        cfg = self.config.model
        if self._legacy_dir is not None:
            self._legacy_loaders(trainer, cfg.data_dir, 1, cfg.train_batch_size)
        else:
            self.batches, self.val, _ = sequence_loaders(
                self.config.data, trainer, sequence_length=1, image_size=VAE_SIZE
            )
        self.vae = trainer.prepare_module(
            ConvVAE(img_channels=3, latent_size=cfg.latent_size)
        )
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=cfg.learning_rate)
        self.update = trainer.gradient_update(optimizer, self.vae.parameters())
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
        if self._legacy_dir is not None and not cfg.noreload:
            self._reload(
                self._legacy_dir,
                self.vae,
                optimizer,
                self.scheduler,
            )

    def _loss(self, frames: torch.Tensor) -> torch.Tensor:
        from simulusion.losses.convae_loss import conv_vae_loss_fn

        recon, mu, logvar = self.vae(frames)
        # conv_vae_loss_fn sums over the batch; per-frame keeps the scale
        # independent of the batch size.
        return conv_vae_loss_fn(recon, frames, mu, logvar) / frames.shape[0]

    def train_step(self) -> Metrics:
        if self._legacy_dir is None:
            return self._step(next(self.batches))
        rows = [self._step(batch) for batch in self.train_loader]
        self._epochs_done += 1
        cfg = self.config.model
        if not cfg.nosamples and self._epochs_done % cfg.sample_interval == 0:
            self._save_samples()
        return mean_metrics(rows)

    def _save_samples(self) -> None:
        from torchvision.utils import save_image

        cfg = self.config.model
        assert self._legacy_dir is not None
        samples = self._legacy_dir / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        vae: Any = self.vae
        with torch.no_grad():
            z = torch.randn(16, cfg.latent_size, device=self.trainer.device)
            images = vae.decoder(z).cpu()
        save_image(images, samples / f"sample_epoch_{self._epochs_done}.png")

    def _step(self, batch: dict) -> dict[str, float]:
        frames = frames_to_float(batch["frames"][:, 0])
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
            if self._legacy_dir is None and i >= self.config.data.val_batches:
                break
            rows.append(
                {"val_loss": float(self._loss(frames_to_float(batch["frames"][:, 0])))}
            )
        metrics = mean_metrics(rows)
        if metrics:
            self.scheduler.step(metrics["val_loss"])
            if self._legacy_dir is not None:
                self._record_val(metrics["val_loss"])
        return metrics

    def save_checkpoint(self, directory: Path) -> None:
        super().save_checkpoint(directory)
        if self._legacy_dir is not None:
            self._save_legacy(
                self._legacy_dir,
                self.vae,
                self.update.optimizer,
                self.scheduler,
            )

    def export(self, directory: Path) -> Optional[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "vae.tar"
        torch.save({"state_dict": self.vae.state_dict()}, path)
        return directory

    def state_dict(self) -> dict[str, Any]:
        return {
            "vae": self.vae.state_dict(),
            "update": self.update.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.vae.load_state_dict(state["vae"])
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
class MDRNNTask(_LegacyEpochs, TrainingTask):
    """World Models' M. Exports the V + M world model and ``model/mdrnn.tar``.

    Args:
        config: The task config.
        legacy_dir: Run as ``train_mdn_rnn`` does, writing its files here
            (``<logdir>/mdrnn``): whole epochs over ``model.data_dir``, the
            ``checkpoint.tar`` / ``best.tar`` checkpoints, and resuming from
            ``best.tar`` unless ``model.noreload``.
    """

    config_cls = MDRNNTaskConfig
    progress_unit = "steps"

    def __init__(self, config: Any = None, *, legacy_dir: Optional[str] = None) -> None:
        super().__init__(config)
        self._legacy_dir = Path(legacy_dir) if legacy_dir is not None else None
        if self._legacy_dir is not None:
            self.progress_unit = "epochs"  # type: ignore[misc]

    def max_steps(self) -> int:
        if self._legacy_dir is not None:
            return int(self.config.model.num_epochs)
        return 20_000

    def progress(self) -> int:
        if self._legacy_dir is not None:
            return self._epochs_done
        return super().progress()

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

        meta = None
        if self._legacy_dir is not None:
            self._legacy_loaders(trainer, cfg.data_dir, cfg.seq_len + 1, cfg.batch_size)
        else:
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
        self.update = trainer.gradient_update(optimizer, self.mdrnn.parameters())
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
        if self._legacy_dir is not None and not cfg.noreload:
            self._reload(
                self._legacy_dir,
                self.mdrnn,
                optimizer,
                self.scheduler,
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
        if self._legacy_dir is None:
            return self._step(next(self.batches))
        rows = [self._step(batch) for batch in self.train_loader]
        self._epochs_done += 1
        return mean_metrics(rows)

    def _step(self, batch: dict) -> dict[str, float]:
        self.mdrnn.train()
        with self.trainer.precision.autocast():
            losses = self._losses(batch)
        self.update(losses["loss"])
        return {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self) -> Metrics:
        if self.val is None:
            return {}
        self.mdrnn.eval()
        rows = []
        for i, batch in enumerate(self.val):
            if self._legacy_dir is None and i >= self.config.data.val_batches:
                break
            rows.append({f"val_{k}": float(v) for k, v in self._losses(batch).items()})
        metrics = mean_metrics(rows)
        if metrics:
            self.scheduler.step(metrics["val_loss"])
            if self._legacy_dir is not None:
                self._record_val(metrics["val_loss"])
        return metrics

    def save_checkpoint(self, directory: Path) -> None:
        super().save_checkpoint(directory)
        if self._legacy_dir is not None:
            self._save_legacy(
                self._legacy_dir,
                self.mdrnn,
                self.update.optimizer,
                self.scheduler,
            )

    def world_model(self) -> Optional[Any]:
        from simulusion.world_model.vae_mdrnn import MDRNNWorldModel

        return MDRNNWorldModel(self.vae, self.mdrnn)

    def export(self, directory: Path) -> Optional[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "mdrnn.tar"
        torch.save({"state_dict": self.mdrnn.state_dict()}, path)
        return directory

    def state_dict(self) -> dict[str, Any]:
        return {
            "mdrnn": self.mdrnn.state_dict(),
            "update": self.update.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mdrnn.load_state_dict(state["mdrnn"])
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
        self,
        config: Any = None,
        *,
        env_factory: Optional[Any] = None,
        model: Optional[Any] = None,
        legacy_dir: Optional[str] = None,
    ) -> None:
        """Args:
        config: The task config.
        env_factory: Builds the environment instead of ``gymnasium.make``.
        model: The V + M world model itself, instead of ``world_model``.
        legacy_dir: Run as ``train_controller`` does, writing ``best.tar``
            (``{"epoch", "reward", "state_dict"}``) here whenever the best
            return improves, starting from it when it exists, and stopping
            once CMA-ES converges or the best return passes
            ``model.target_return``.
        """
        super().__init__(config)
        self._env_factory = env_factory
        self._model = model
        self._legacy_dir = Path(legacy_dir) if legacy_dir is not None else None

    def setup(self, trainer: Any) -> None:
        import cma

        from simulusion.models.controller import Controller
        from simulusion.world_model import load_world_model

        cfg = self.config
        if cfg.mode not in ("env", "dream"):
            raise ValueError("mode must be 'env' or 'dream'.")
        if self._model is not None:
            self.wm: Any = self._model
        elif not cfg.world_model:
            raise ValueError(
                "controller acts through a world model: set world_model=<dir>."
            )
        else:
            self.wm = load_world_model(
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
        if self._legacy_dir is not None and (self._legacy_dir / "best.tar").exists():
            state = torch.load(
                self._legacy_dir / "best.tar", map_location="cpu", weights_only=True
            )
            self.controller.load_state_dict(state["state_dict"])
            self.best_return = float(state["reward"])
            self.best_params = self._flat()
            logger.info("Previous best was %s...", self.best_return)
            self.es = cma.CMAEvolutionStrategy(
                self.best_params,
                cfg.sigma,
                {"popsize": cfg.model.pop_size, "seed": trainer.config.seed + 1},
            )
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

    def _episode(self, flat: np.ndarray) -> float:
        """One episode's return for controller parameters ``flat``."""
        self._load(flat)
        return controller_episode(
            self.wm,
            self.controller,
            self.env,
            time_limit=self.config.model.time_limit,
            dream=self.config.mode == "dream",
        )

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
            if self._legacy_dir is not None:
                self._save_best()
        self.generation += 1
        return {
            "return_mean": float(np.mean(returns)),
            "return_max": float(np.max(returns)),
            "best_return": self.best_return,
            "sigma": float(self.es.sigma),
        }

    def _save_best(self) -> None:
        assert self._legacy_dir is not None
        self._legacy_dir.mkdir(parents=True, exist_ok=True)
        current = self._flat()
        self._load(self.best_params)
        torch.save(
            {
                "epoch": self.generation,
                "reward": self.best_return,
                "state_dict": self.controller.state_dict(),
            },
            self._legacy_dir / "best.tar",
        )
        self._load(current)

    def should_stop(self) -> bool:
        if self._legacy_dir is None:
            return False
        return (
            bool(self.es.stop()) or self.best_return > self.config.model.target_return
        )

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


@torch.no_grad()
def controller_episode(
    wm: Any, controller: Any, env: Any, *, time_limit: int, dream: bool = False
) -> float:
    """One episode of World Models' C acting on V + M's ``[z, h]``; its return.

    Args:
        wm: A V + M world model (:class:`~simulusion.world_model.vae_mdrnn.
            MDRNNWorldModel`).
        controller: Maps ``[z, h]`` to an action.
        env: Gives the first frame, and in the real environment every reward.
        time_limit: Steps at most.
        dream: Act inside the world model, with M's reward and done heads,
            instead of in ``env``.
    """
    obs = _env_reset(env)
    frame = torch.as_tensor(np.asarray(obs["image"] if isinstance(obs, dict) else obs))
    state = wm.encode(frame.unsqueeze(0))
    total = 0.0
    for _ in range(time_limit):
        action = controller(torch.cat([state["z"], state["h"]], dim=-1))
        if dream:
            out = wm.step(state, action)
            state = out.state
            total += float(out.reward[0])
            if bool(out.terminated[0]):
                break
            continue
        # Real environment: M only tracks h from the actions taken.
        next_obs, reward, done = _env_step(env, action[0].cpu().numpy())
        total += float(reward)
        if done:
            break
        out = wm.step(state, action)
        frame = torch.as_tensor(
            np.asarray(next_obs["image"] if isinstance(next_obs, dict) else next_obs)
        )
        z = wm.encode(frame.unsqueeze(0))["z"]
        state = {"z": z, "h": out.state["h"], "c": out.state["c"]}
    return total


def _env_reset(env: Any) -> Any:
    """The first observation, from a Gymnasium ``(obs, info)`` or a bare obs."""
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _env_step(env: Any, action: Any) -> tuple[Any, float, bool]:
    """``(obs, reward, done)`` from a Gymnasium 5-tuple or an old-Gym 4-tuple.

    GymImageEnv, which the controller acts in by default, uses the old API.
    """
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, _ = out
        return obs, float(reward), bool(terminated or truncated)
    obs, reward, done, _ = out
    return obs, float(reward), bool(done)


__all__ = [
    "ControllerTask",
    "ControllerTaskConfig",
    "MDRNNTask",
    "MDRNNTaskConfig",
    "VAETask",
    "VAETaskConfig",
]
