import torch
from simulusion.utils.device import default_device_name
import numpy as np
from collections import defaultdict
import os
import random
from collections.abc import Callable, Sequence
from typing import Any, Optional

from types import ModuleType
from typing import Optional as _Optional

from simulusion.configs.iris_config import IRISConfig
from simulusion.experiments import (
    dump_config,
    load_experiment_config,
    update_config_object,
)
from simulusion.models.iris_agent import IRISAgent
from simulusion.memory.iris_memory import IRISReplayBuffer
from simulusion.envs.ale_atari_env import make_atari_env

# Optional OpenCV import at module scope (avoid function-local imports)
cv2: _Optional[ModuleType] = None
try:
    import cv2 as _cv2

    cv2 = _cv2
except Exception:
    cv2 = None


# Paper Appendix H: Freeway's reward is sparse (the agent only scores by fully
# crossing the road) and cars knock it back down, so a random policy will almost
# never see a non-zero reward inside the 100k budget. IRIS keeps its fixed
# epsilon-greedy parameter and instead lowers the collection sampling
# temperature from 1 to 0.01, which suppresses the random walks that would
# otherwise stop the agent ever reaching the far side.
FREEWAY_COLLECT_TEMPERATURE = 0.01


def default_collect_temperature(game: str, configured: float) -> float:
    """Collection sampling temperature, applying the paper's Freeway exception.

    Only overrides a temperature left at the default of 1.0, so an explicit
    setting in a config file or CLI override still wins.
    """
    if "freeway" in game.lower() and configured == 1.0:
        return FREEWAY_COLLECT_TEMPERATURE
    return configured


def _action_size(space: Any) -> int:
    """Number of action dimensions from a Gym/Gymnasium-style space.

    Avoids importing ``gym`` / ``gymnasium`` at module load so paper-alignment
    tests can import collection helpers on a base ``pip install simulusion``.
    """
    n = getattr(space, "n", None)
    if n is not None:
        return int(n)
    shape = getattr(space, "shape", None)
    if shape:
        return int(np.prod(tuple(shape)))
    raise TypeError(f"Unsupported action_space type: {type(space)}")


class IRISTrainer:
    """Training loop for IRIS on Atari 100k benchmark."""

    def __init__(
        self,
        game: str = "ALE/Pong-v5",
        device: str = "cuda",
        seed: int = 42,
        config: Optional[IRISConfig] = None,
        env: Optional[Any] = None,
    ) -> None:
        """Train IRIS on an environment.

        Args:
            game: Environment id, used to build an Atari environment when ``env``
                is not supplied, and as the label for checkpoint filenames.
            device: Torch device string.
            seed: Seed for random, numpy and torch.
            config: IRIS hyperparameters; defaults to the paper's values.
            env: An already-constructed Gymnasium-style environment to train on.
                Supply this to run IRIS outside Atari. It must expose the usual
                ``reset() -> (obs, info)`` / ``step(a) -> (obs, r, term, trunc,
                info)`` API, a discrete ``action_space``, and image observations
                that :meth:`preprocess_frame` can resize to the configured frame
                size. When omitted, an Atari environment is built from ``game``
                using the config's action-repeat and sticky-action settings.
        """
        self.game = game
        from simulusion.utils.device import resolve_device

        self.device = resolve_device(device)
        self.seed = seed

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Config
        self.config = config if config is not None else IRISConfig()

        self.collect_temperature = default_collect_temperature(
            game, self.config.collect_temperature
        )

        # Environment: use the caller's if given, otherwise build the Atari one
        # this trainer defaults to. Everything downstream only relies on the
        # Gymnasium step/reset API and a discrete action space, so IRIS can be
        # trained on any image-observation environment by passing `env`.
        self._env_factory: Optional[Callable[[], Any]] = None
        if env is not None:
            self.env = env
        else:

            def _build_env() -> Any:
                return make_atari_env(
                    game,
                    obs_type="rgb",
                    frameskip=self.config.action_repeat,
                    repeat_action_probability=self.config.repeat_action_probability,
                    max_episode_steps=self.config.max_episode_steps,
                )

            self._env_factory = _build_env
            self.env = _build_env()

        # Built on first evaluation; see _evaluation_env.
        self._eval_env: Optional[Any] = None

        # Discrete spaces expose ``n``; Box-like spaces expose ``shape``.
        # Duck-type both so importing this module does not require gym/gymnasium.
        self.action_size = _action_size(self.env.action_space)

        # Create replay buffer
        self.replay_buffer = IRISReplayBuffer(
            size=100000,  # 100k buffer
            obs_shape=(3, 64, 64),  # Resize frames to 64x64
            action_size=self.action_size,
            seq_len=self.config.transformer_timesteps,
            batch_size=self.config.transformer_batch_size,
        )

        # Create agent
        self.agent = IRISAgent(
            config=self.config,
            action_size=self.action_size,
            device=self.device,
        )

        # Metrics: map metric name -> series of numeric values
        self.metrics: defaultdict[str, list[float]] = defaultdict(list)

        # Persistent collection state. Episodes are longer than one epoch's step
        # budget, so the environment, the policy's recurrent state and the
        # running return all have to survive across collect_experience calls.
        self._collect_obs: Optional[np.ndarray] = None
        self._collect_hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._collect_return: float = 0.0
        self._last_episode_return: float = 0.0
        self.env_steps: int = 0

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame: resize to 64x64, return CHW uint8.

        Frames are kept as uint8 because that is the dtype of the replay buffer
        (``IRISReplayBuffer.observations``). Returning floats in [0, 1] here
        silently truncated every pixel to 0 on insertion, so the world model and
        policy trained exclusively on black images. Normalisation to [0, 1]
        happens at consumption time via :meth:`to_float_tensor`.
        """
        if cv2 is None:
            raise ImportError(
                "cv2 is required for frame preprocessing. Install opencv-python"
            )

        frame = np.asarray(frame)
        if frame.ndim == 2:  # grayscale (H, W)
            frame = frame[:, :, None]
        if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = frame.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

        size = (self.config.frame_width, self.config.frame_height)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        if frame.ndim == 2:  # cv2 drops a trailing single channel
            frame = frame[:, :, None]
        if frame.shape[2] == 1 and self.config.frame_channels == 3:
            frame = np.repeat(frame, 3, axis=2)

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        return frame.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    def to_float_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a uint8 CHW frame (or batch) to a float tensor in [0, 1]."""
        tensor = torch.as_tensor(np.ascontiguousarray(obs)).to(self.device)
        return tensor.float().div_(255.0)

    def collect_experience(
        self,
        num_steps: int,
        epsilon: float = 0.01,
    ) -> float:
        """Collect exactly ``num_steps`` environment steps (paper Algorithm 1).

        The step count is exact because the Atari 100k budget is defined in
        environment steps: ``collection_epochs * env_steps_per_epoch`` must equal
        100k. Episodes span epochs, so the environment, the policy's LSTM state
        and the running episode return all persist between calls and are reset
        only on a real episode boundary.

        Two details follow the paper rather than convenience:

        * Frames are passed through the discrete autoencoder before reaching the
          policy (A.1), because the policy is trained purely on reconstructions.
        * The LSTM state is threaded across steps (A.3); the policy is recurrent
          and a per-step reset would leave it unable to perceive motion.

        Args:
            num_steps: Exact number of environment steps to collect
            epsilon: Random action probability

        Returns:
            Mean return of episodes that finished during this call, or the most
            recently completed episode's return if none finished.
        """
        if self._collect_obs is None:
            raw, _ = self.env.reset()
            self._collect_obs = self.preprocess_frame(raw)
            self._collect_hidden = None
            self._collect_return = 0.0

        obs = self._collect_obs
        episode_returns: list[float] = []

        for _ in range(num_steps):
            frame_tensor = self.to_float_tensor(obs).unsqueeze(0)
            # Paper A.1: keep the policy's input distribution unchanged by
            # reconstructing real frames through the autoencoder.
            policy_input = self.agent.reconstruct(frame_tensor)

            act_out = self.agent.act(
                policy_input,
                epsilon=epsilon,
                temperature=self.collect_temperature,
                hidden=self._collect_hidden,
                return_hidden=True,
            )
            assert isinstance(act_out, tuple)
            action_tensor, self._collect_hidden = act_out
            action = int(action_tensor.item())

            next_raw, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            action_one_hot: np.ndarray = np.zeros(self.action_size, dtype=np.float32)
            action_one_hot[action] = 1.0

            # The termination head models d_t, the environment's episode
            # *termination* (paper 2). A time-limit truncation is not one: the
            # MDP would have continued, so labelling it terminal teaches the
            # world model to predict episode ends that the game never produces,
            # and zeroes the lambda-return's bootstrap at an arbitrary cut. The
            # episode is still reset below either way.
            self.replay_buffer.add(obs, action_one_hot, float(reward), bool(terminated))
            self.env_steps += 1
            self._collect_return += float(reward)

            if done:
                episode_returns.append(self._collect_return)
                self._last_episode_return = self._collect_return
                self._collect_return = 0.0
                # A new episode means a new hidden state; carrying it across the
                # boundary would leak the previous episode into the next one.
                self._collect_hidden = None
                raw, _ = self.env.reset()
                obs = self.preprocess_frame(raw)
            else:
                obs = self.preprocess_frame(next_raw)

        self._collect_obs = obs

        if episode_returns:
            return float(np.mean(episode_returns))
        return self._last_episode_return

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Phase 1: Collect experience. Collection stops after
        # ``collection_epochs`` so the run respects the Atari 100k budget
        # (paper Table 5: 500 collection epochs x 200 steps = 100k); the
        # remaining epochs keep training on the data already gathered.
        if epoch < self.config.collection_epochs:
            mean_return = self.collect_experience(
                num_steps=self.config.env_steps_per_epoch,
                epsilon=self.get_epsilon(epoch),
            )
            metrics["collection_return"] = mean_return
        metrics["env_steps"] = float(self.env_steps)

        # Only update components after warm-start periods
        if epoch >= self.config.start_autoencoder_after:
            # Phase 2: Update autoencoder
            ae_metrics: dict = {}
            for _ in range(self.config.training_steps_per_epoch):
                # Sample random frames
                indices = np.random.randint(
                    0, len(self.replay_buffer), size=self.config.autoencoder_batch_size
                )

                frames = self.to_float_tensor(self.replay_buffer.observations[indices])

                ae_metrics = self.agent.update_autoencoder(frames)

            metrics["recon_loss"] = ae_metrics.get("recon_loss", 0)
            metrics["vq_loss"] = ae_metrics.get("vq_loss", 0)
            metrics["perceptual_loss"] = ae_metrics.get("perceptual_loss", 0)
            # Perplexity is the effective codebook size. If it trends toward 1
            # the autoencoder has collapsed and everything downstream is junk.
            metrics["perplexity"] = ae_metrics.get("perplexity", 0)

        if (
            epoch >= self.config.start_transformer_after
            and len(self.replay_buffer) >= self.config.transformer_timesteps + 1
        ):
            # Phase 3: Update transformer
            tf_metrics: dict = {}
            for _ in range(self.config.transformer_steps_per_epoch):
                obs, acts, rews, terms = self.replay_buffer.sample_sequence()

                obs_tensor = self.to_float_tensor(obs)
                acts_tensor = torch.tensor(acts, dtype=torch.float32).to(self.device)
                rews_tensor = torch.tensor(rews, dtype=torch.float32).to(self.device)
                terms_tensor = torch.tensor(terms, dtype=torch.long).to(self.device)

                tf_metrics = self.agent.update_transformer(
                    obs_tensor, acts_tensor, rews_tensor, terms_tensor
                )

            metrics["token_loss"] = tf_metrics.get("token_loss", 0)
            metrics["reward_loss"] = tf_metrics.get("reward_loss", 0)

        if (
            epoch >= self.config.start_actor_critic_after
            and len(self.replay_buffer) >= 50
        ):
            # Phase 4: Update actor-critic in imagination
            ac_metrics: dict = {}
            for _ in range(self.config.actor_critic_steps_per_epoch):
                # Sample initial frames plus the frames preceding them, which
                # initialise the policy's LSTM state (paper A.3).
                start_obs, burn_in_obs = self.replay_buffer.sample_with_burn_in(
                    batch_size=self.config.actor_critic_batch_size,
                    burn_in=self.config.burn_in_length,
                )

                # The policy only ever sees reconstructions, so the real frames
                # that seed imagination must be reconstructed too.
                initial_frames = self.to_float_tensor(start_obs)
                burn_in_frames = self.agent.reconstruct(
                    self.to_float_tensor(burn_in_obs)
                )

                # Generate imagined trajectories
                imagined = self.agent.imagine_rollout(
                    initial_frame=initial_frames,
                    horizon=self.config.imagination_horizon,
                    burn_in_frames=burn_in_frames,
                )

                # Update policy
                ac_metrics = self.agent.update_actor_critic(imagined)

            metrics["actor_loss"] = ac_metrics.get("actor_loss", 0)
            metrics["value_loss"] = ac_metrics.get("value_loss", 0)
            metrics["entropy"] = ac_metrics.get("entropy", 0)
            metrics["value_mean"] = ac_metrics.get("value_mean", 0)
            metrics["return_mean"] = ac_metrics.get("return_mean", 0)

        self.agent.current_epoch = epoch
        self.agent.global_step += self.config.env_steps_per_epoch

        return metrics

    def get_epsilon(self, epoch: int) -> float:
        """Exploration epsilon for collection.

        Paper Table 5 uses a *fixed* epsilon-greedy parameter of 0.01 combined
        with sampling from the policy, rather than the decaying schedule most
        Atari 100k baselines use (Appendix H discusses the tradeoff). A decaying
        schedule starting near-random makes the early world model fit a data
        distribution the policy never revisits.
        """
        del epoch  # fixed schedule; kept for signature compatibility
        return self.config.collect_epsilon

    def _evaluation_env(self) -> tuple[Any, bool]:
        """Return ``(env, shares_collection_env)`` for evaluation.

        Evaluation plays whole episodes, while ``collect_experience`` keeps a
        partial one alive across calls (its last observation, the policy's
        recurrent state and the running return). Both used ``self.env``, so an
        evaluation reset the environment out from under that state: the next
        collection step fed the stale pre-eval observation to the policy, then
        stepped an environment sitting wherever the evaluation had left it, and
        wrote the resulting mismatched transition into the replay buffer.

        A second environment keeps the two apart. When the caller supplied the
        environment there is nothing to build one from, so evaluation borrows it
        and the collection state is dropped afterwards instead.
        """
        if self._env_factory is None:
            return self.env, True
        if self._eval_env is None:
            self._eval_env = self._env_factory()
        return self._eval_env, False

    def _reset_collection_state(self) -> None:
        """Drop the in-flight collection episode; the next call starts a new one."""
        self._collect_obs = None
        self._collect_hidden = None
        self._collect_return = 0.0

    def evaluate(self, num_episodes: int = 100, render: bool = False) -> dict | tuple:
        """Evaluate agent performance.

        Args:
            num_episodes: Number of evaluation episodes
            render: If True, also return video frames and per-step latent vectors

        Returns:
            If render is False (default): dict with evaluation metrics
            If render is True: tuple (episode_returns_array, videos_list, latents_array)
        """
        episode_returns = []
        videos: list[list[np.ndarray]] = []
        latents_all: list[np.ndarray] = []

        env, shares_collection_env = self._evaluation_env()

        for _ in range(num_episodes):
            raw_obs, _ = env.reset()
            obs = self.preprocess_frame(raw_obs)

            episode_return: float = 0.0
            done = False
            frames: list[np.ndarray] = []
            # Fresh recurrent state per episode, carried across steps within it.
            hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None

            while not done:
                # Prepare frame for policy (CHW, float32, 0-1) and reconstruct it,
                # matching the input distribution the policy was trained on.
                frame_tensor = self.to_float_tensor(obs).unsqueeze(0)
                policy_input = self.agent.reconstruct(frame_tensor)
                act_out = self.agent.act(
                    policy_input,
                    epsilon=0.0,
                    temperature=self.config.eval_temperature,
                    hidden=hidden,
                    return_hidden=True,
                )
                assert isinstance(act_out, tuple)
                action_tensor, hidden = act_out
                action = int(action_tensor.item())

                next_raw, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Only when the caller asked for them: a raw Atari frame is
                # ~100 KB, so at the default eval_episodes=100 and the 27000-step
                # episode cap, collecting them unconditionally could hold tens of
                # GB until the eval returned -- enough for the OS to kill the
                # process, with no Python error to show for it.
                if render:
                    try:
                        frames.append(np.asarray(next_raw))
                    except Exception:
                        # Fallback: convert processed obs back to HWC
                        proc = np.asarray(obs)
                        if proc.ndim == 3:
                            # CHW -> HWC
                            frames.append(proc.transpose(1, 2, 0))

                next_obs = self.preprocess_frame(next_raw)

                # Compute latent embedding via encoder (quantized embeddings).
                # Also render-only: it is a per-step forward pass whose result
                # is returned alongside the video and dropped otherwise.
                if render:
                    try:
                        proc_frame = next_obs if not done else obs
                        with torch.no_grad():
                            ft = self.to_float_tensor(proc_frame).unsqueeze(0)
                            # eval mode: the quantizer's dead-code revival must not
                            # fire while merely reading out a latent for logging.
                            was_training = self.agent.encoder.training
                            self.agent.encoder.eval()
                            try:
                                z_q, _, _ = self.agent.encoder(ft)
                            finally:
                                self.agent.encoder.train(was_training)
                            # z_q: (B, C, H', W') -> reduce spatial dims and take mean over channels
                            latent = z_q.mean(dim=(2, 3)).squeeze(0).cpu().numpy()
                            latents_all.append(latent.astype(np.float32))
                    except Exception:
                        # If encoder fails, skip latent for this step
                        pass

                episode_return += float(reward)
                obs = next_obs if not done else obs

            episode_returns.append(episode_return)
            if render:
                videos.append(frames)

        if shares_collection_env:
            # The evaluation left this environment mid-episode somewhere else;
            # the half-finished collection episode that referred to it is gone.
            self._reset_collection_state()

        if render:
            # Stack latents into (N, D) array if any
            if latents_all:
                latents_array = np.vstack(latents_all).astype(np.float32)
            else:
                latents_array = np.empty((0,), dtype=np.float32)
            return np.array(episode_returns), videos, latents_array

        # Non-render fallback: return simple metrics dict for compatibility
        return {
            "eval_mean_return": float(
                np.mean(episode_returns) if episode_returns else 0.0
            ),
            "eval_std_return": float(
                np.std(episode_returns) if episode_returns else 0.0
            ),
            "eval_max_return": float(
                np.max(episode_returns) if episode_returns else 0.0
            ),
            "eval_min_return": float(
                np.min(episode_returns) if episode_returns else 0.0
            ),
        }

    def train(
        self,
        total_epochs: Optional[int] = None,
        eval_interval: int = 50,
        save_dir: str = "checkpoints/iris",
    ) -> Any:
        """Train for ``total_epochs`` epochs (``config.total_epochs`` by default).

        Runs on the shared :class:`~simulusion.training.Trainer` through the
        ``iris`` task, with this trainer's environment, replay and agent.
        Every ``eval_interval`` epochs, once the policy trains, it evaluates
        (monitoring only); the best of those is written to
        ``<save_dir>/best_<game>.pt``, and early stopping on it applies when
        ``config.early_stopping`` is set. ``checkpoint_<epoch>.pt`` is written
        every ``config.checkpoint_interval`` epochs. The paper's benchmark
        number is the final evaluation over ``config.eval_episodes``
        (``self.final_eval_return``, model in ``final_<game>.pt``). The
        trainer adds resumable checkpoints under ``<save_dir>/checkpoints/``,
        ``run.json``, ``run_config.yaml`` and ``train_metrics.jsonl``.

        Returns:
            ``self.metrics``: every epoch's metrics, by name.
        """
        import shutil

        from simulusion.training.legacy import fit_legacy
        from simulusion.training.tasks.iris import IRISTask, IRISTaskConfig

        config = self.config
        epochs = int(total_epochs if total_epochs is not None else config.total_epochs)
        early = bool(getattr(config, "early_stopping", False))
        name = self.game.split("/")[-1]
        os.makedirs(save_dir, exist_ok=True)
        print(f"Starting training for {epochs} epochs on {self.game}")
        print(f"Action space: {self.action_size}")
        print(f"Device: {self.device}")

        task = IRISTask(
            # No replay in checkpoints: 100k frames per checkpoint, and this
            # loop never resumed.
            IRISTaskConfig(model=config, game=self.game, save_replay=False),
            iris=self,
            legacy_dir=save_dir,
        )
        result = fit_legacy(
            task,
            save_dir,
            max_steps=epochs,
            device=self.device,
            seed=self.seed,
            log_every=10,
            eval_every=eval_interval,
            checkpoint_every=config.checkpoint_interval,
            # Return, not loss: the autoencoder and transformer losses keep
            # falling well after the policy stops improving.
            best_metric="eval_mean_return",
            best_mode="max",
            early_stopping_metric="eval_mean_return" if early else None,
            early_stopping_mode="max",
            patience=config.patience,
            min_delta=config.min_delta,
        )
        if result.best_checkpoint is not None:
            shutil.copyfile(
                result.best_checkpoint / "task" / "agent.pt",
                os.path.join(save_dir, f"best_{name}.pt"),
            )
        # Paper 3.2: the benchmark number is the average over eval_episodes
        # episodes at the end of training -- not the best periodic evaluation,
        # which takes the maximum of a noisy quantity.
        self.final_eval_return = float(result.metrics.get("eval_mean_return", 0.0))
        save_path = os.path.join(save_dir, f"final_{name}.pt")
        self.agent.save(save_path)
        print(
            "\nTraining complete. Benchmark score (mean over "
            f"{config.eval_episodes} episodes at end of training): "
            f"{self.final_eval_return:.2f} +/- "
            f"{result.metrics.get('eval_std_return', 0.0):.2f}"
        )
        if result.best_value is not None:
            print(
                f"  (best evaluation during training was "
                f"{result.best_value:.2f}; monitoring only, not comparable to "
                f"the paper's Table 1)"
            )
        print(f"  Final model: {save_path}")
        return self.metrics


# Overrides that configure the training *run* rather than the model. They are
# not IRISConfig fields, so they must be split out before the config is composed
# -- `update_config_object` is strict and rejects any key it does not recognise.
RUNTIME_OVERRIDE_KEYS = ("game", "device", "seed", "epochs", "save_dir")


def _split_runtime_overrides(
    overrides: Sequence[str],
) -> tuple[dict[str, str], list[str]]:
    """Partition ``key=value`` overrides into runtime options and config fields."""
    runtime: dict[str, str] = {}
    config_overrides: list[str] = []
    for item in overrides:
        key, sep, value = item.partition("=")
        if sep and key.strip() in RUNTIME_OVERRIDE_KEYS:
            runtime[key.strip()] = value.strip()
        else:
            config_overrides.append(item)
    return runtime, config_overrides


def main(argv: list[str] | None = None) -> IRISConfig:
    """Run IRIS training with YAML config files and Hydra dot-list overrides."""
    from simulusion.experiments import parse_experiment_args

    args = parse_experiment_args(argv, description="Train IRIS on Atari")

    # Split first: passing `game=`/`device=`/`seed=` through to the strict config
    # loader previously raised ExperimentConfigError, making them unusable.
    runtime, config_overrides = _split_runtime_overrides(args.overrides)

    config = IRISConfig()
    values = load_experiment_config(config, args.config, config_overrides)
    config = update_config_object(config, values)

    game = runtime.get("game", config.env)
    device = runtime.get("device") or default_device_name()
    seed = int(runtime.get("seed", 42))
    total_epochs = int(runtime.get("epochs", config.total_epochs))
    save_dir = runtime.get("save_dir", "checkpoints/iris")

    if args.print_config:
        print(dump_config(values))
        return config

    trainer = IRISTrainer(
        game=game,
        device=device,
        seed=seed,
        config=config,
    )

    trainer.train(
        total_epochs=total_epochs,
        save_dir=save_dir,
    )
    return config


if __name__ == "__main__":
    main()
