from functools import partial
import os
import torch
import numpy as np

from simulusion.utils.utils import to_tensor_obs, preprocess_img
from simulusion.models.rssm import RecurrentStateSpaceModel
from simulusion.controller.rssm_policy import RSSMPolicy
from simulusion.controller.rollout_generator import RolloutGenerator
from simulusion.utils.utils import (
    TensorBoardMetrics,
    postprocess_img,
    TorchImageEnvWrapper,
)
from typing import Any
from simulusion.memory.planet_memory import Memory, Episode
from simulusion.export import ExportableAgentMixin


class Planet(ExportableAgentMixin):
    """
    High-level Planet wrapper.

    Usage example:
      from simulusion.models.planet import Planet
      p = Planet(env='CartPole-v1', bit_depth=5)
      p.train(epochs=50)
    """

    def __init__(
        self,
        env: Any,
        bit_depth: int = 5,
        device: torch.device | None = None,
        state_size: int = 200,
        latent_size: int = 30,
        embedding_size: int = 1024,
        memory_size: int = 100,
        policy_cfg: dict | None = None,
        headless: bool = False,
        max_episode_steps: int | None = None,
        action_repeats: int = 1,
        results_dir: str | None = None,
    ) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        if device:
            self.device = device
        else:
            from simulusion.utils.device import get_default_device

            self.device = get_default_device()
            if self.device.type == "cpu":
                print("WARNING: no GPU (CUDA or MPS) available, using CPU")
        self.bit_depth = bit_depth

        if isinstance(env, str):
            self.env = TorchImageEnvWrapper(env, bit_depth, None, action_repeats)
        elif hasattr(env, "action_size"):
            self.env = env
        else:
            self.env = self._wrap_raw_env(env, bit_depth, action_repeats)

        self.rssm = RecurrentStateSpaceModel(
            self.env.action_size, state_size, latent_size, embedding_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.rssm.parameters(), lr=3e-4, eps=1e-4, weight_decay=1e-5
        )

        policy_cfg = policy_cfg or {}
        self.policy = RSSMPolicy(
            model=self.rssm,
            planning_horizon=policy_cfg.get("planning_horizon", 20),
            num_candidates=policy_cfg.get("num_candidates", 1000),
            num_iterations=policy_cfg.get("num_iterations", 10),
            top_candidates=policy_cfg.get("top_candidates", 100),
            device=self.device,
        )

        env_max_steps = (
            max_episode_steps or getattr(self.env, "max_episode_steps", None) or 100
        )
        self.rollout_gen = RolloutGenerator(
            self.env,
            self.device,
            policy=self.policy,
            episode_gen=lambda: Episode(partial(postprocess_img, depth=self.bit_depth)),
            max_episode_steps=env_max_steps,
        )

        self.memory = Memory(memory_size)
        self.summary: Any = None
        self.results_dir = results_dir or "results/planet"

    def _wrap_raw_env(self, env: Any, bit_depth: int, action_repeats: int) -> Any:
        class SimpleEnvWrapper:
            def __init__(self, env: Any, bit_depth: int, action_repeats: int) -> None:
                self.env = env
                self.bit_depth = bit_depth
                self.action_repeats = action_repeats

            @property
            def action_size(self) -> int:
                if hasattr(self.env.action_space, "n"):
                    return 1
                else:
                    return self.env.action_space.shape[0]

            @property
            def observation_size(self) -> tuple:
                return (3, 64, 64)

            @property
            def max_episode_steps(self) -> int:
                if (
                    hasattr(self.env, "_max_episode_steps")
                    and self.env._max_episode_steps is not None
                ):
                    return self.env._max_episode_steps
                if hasattr(self.env, "spec") and hasattr(
                    self.env.spec, "max_episode_steps"
                ):
                    return self.env.spec.max_episode_steps
                return 1000

            def sample_random_action(self) -> torch.Tensor:
                return torch.tensor(self.env.action_space.sample(), dtype=torch.float32)

            def reset(self) -> torch.Tensor:
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                frame = self._get_frame(obs)
                if frame is None or frame.size == 0:
                    raise RuntimeError("Environment returned invalid frame on reset")
                x = to_tensor_obs(frame)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(
                        f"Invalid tensor before preprocessing: min={x.min()}, max={x.max()}"
                    )
                    raise ValueError("NaN/Inf in observation tensor")
                preprocess_img(x, self.bit_depth)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(
                        f"Invalid tensor after preprocessing: min={x.min()}, max={x.max()}"
                    )
                    raise ValueError("NaN/Inf after preprocessing")
                return x

            def step(self, action: Any) -> tuple:
                if isinstance(action, torch.Tensor):
                    action = action.detach().numpy()
                if hasattr(self.env.action_space, "n"):
                    action = int(np.clip(action, 0, self.env.action_space.n - 1))
                obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                frame = self._get_frame(obs)
                x = to_tensor_obs(frame)
                preprocess_img(x, self.bit_depth)
                return x, reward, done, info

            def _get_frame(self, obs: Any) -> np.ndarray:
                frame = self.env.render()
                if isinstance(frame, tuple):
                    frame = frame[0]
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    return frame
                if isinstance(obs, np.ndarray) and obs.ndim == 1:
                    vals = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
                    canvas: np.ndarray = np.zeros((64, 64, 3), dtype=np.uint8)
                    for i, v in enumerate(vals[:8]):
                        band = int(255 * v)
                        canvas[:, i * 8 : (i + 1) * 8, :] = band
                    return canvas
                return np.zeros((64, 64, 3), dtype=np.uint8)

            def __getattr__(self, name: str) -> Any:
                return getattr(self.env, name)

        return SimpleEnvWrapper(env, bit_depth, action_repeats)

    def warmup(self, n_episodes: int = 1, random_policy: bool = True) -> None:
        """Collect n_episodes of rollouts into memory (used as warmup)."""
        eps = self.rollout_gen.rollout_n(n=n_episodes, random_policy=random_policy)
        self.memory.append(eps)

    def train(
        self,
        epochs: int = 100,
        steps_per_epoch: int = 150,
        batch_size: int = 32,
        H: int = 50,
        beta: float = 1.0,
        save_every: int = 25,
        record_grads: bool = False,
        results_dir: str | None = None,
        scheduler_type: str = "step",
        scheduler_kwargs: dict | None = None,
    ) -> str:
        """Train for ``epochs`` epochs; returns the results directory.

        Runs on the shared :class:`~simulusion.training.Trainer` through the
        ``planet`` task, with this model, planner and replay. Each epoch is
        ``steps_per_epoch`` gradient steps on ``batch_size`` chunks of ``H``
        steps, then one exploration and one evaluation episode. As before,
        metrics go to the TensorBoard summary in ``results_dir``, every
        evaluation episode is saved as ``vid_<epoch>``, and the RSSM is
        written to ``ckpt_<epoch>.pth`` every ``save_every`` epochs.

        Args:
            scheduler_type (str): Type of scheduler to use ("step", "cosine", "exponential", "plateau", None)
            scheduler_kwargs (dict): Additional arguments for the scheduler
        """
        from simulusion.training.legacy import fit_legacy
        from simulusion.training.tasks.planet import (
            PlanetConfig,
            PlanetTask,
            PlanetTaskConfig,
        )

        # allow caller to override results dir for this training run
        if results_dir is not None:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary = TensorBoardMetrics(self.results_dir)
        config = PlanetConfig(
            bit_depth=self.bit_depth,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            chunk_length=H,
            beta=beta,
            record_grads=record_grads,
            scheduler_type=scheduler_type,
            scheduler_kwargs=dict(scheduler_kwargs or {}),
        )
        fit_legacy(
            # The replay is not checkpointed: this loop never resumed.
            PlanetTask(PlanetTaskConfig(model=config, save_replay=False), planet=self),
            self.results_dir,
            max_steps=epochs,
            device=self.device,
            checkpoint_every=save_every,
        )
        return self.results_dir


def make_planet_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str | None,
    scheduler_kwargs: dict | None,
    epochs: int,
) -> Any:
    """The per-epoch learning-rate scheduler ``Planet.train`` uses.

    ``scheduler_type`` is ``"step"`` (halve every 50 epochs by default),
    ``"cosine"``, ``"exponential"``, ``"plateau"`` or ``None`` / ``"none"``.
    """
    if scheduler_type is None or scheduler_type.lower() == "none":
        return None
    scheduler_kwargs = scheduler_kwargs or {}
    kind = scheduler_type.lower()
    if kind == "step":
        # Default: reduce LR by factor of 0.5 every 50 epochs
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_kwargs.get("step_size", 50),
            gamma=scheduler_kwargs.get("gamma", 0.5),
        )
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_kwargs.get("T_max", epochs),
            eta_min=scheduler_kwargs.get("eta_min", 1e-6),
        )
    if kind == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=scheduler_kwargs.get("gamma", 0.95)
        )
    if kind == "plateau":
        # Reduce on plateau (based on loss)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_kwargs.get("mode", "min"),
            factor=scheduler_kwargs.get("factor", 0.5),
            patience=scheduler_kwargs.get("patience", 10),
        )
    print(
        f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used."
    )
    return None
