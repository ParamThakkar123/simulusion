from typing import Any, cast
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import os
from pathlib import Path
from simulusion.models.model_io import (
    apply_config_overrides,
    coerce_config,
    module_summary,
    resolve_pretrained_file,
    save_config_next_to_checkpoint,
)
from simulusion.training.precision import GradientUpdate, MixedPrecision
from simulusion.configs.diamond_config import (
    DiamondConfig,
    HUMAN_SCORES,
    RANDOM_SCORES,
)
from simulusion.experiments import (
    dump_config,
    instantiate_dataclass,
    parse_experiment_args,
)
from simulusion.envs.diamond_atari import make_diamond_atari_env
from gymnasium.spaces import Discrete, Box
from simulusion.datasets.diamond_dataset import (
    ReplayBuffer,
    SequenceDataset,
)
from simulusion.models.diffusion.reward_termination import RewardTerminationLoss
from simulusion.models.diffusion.actor_critic import (
    ActorCriticNetwork,
    RLLoss,
)


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """uint8 HWC frame -> float32 HWC in [-1, 1].

    Must match ``datasets.diamond_dataset.to_model_domain``, which is what the
    replay-buffer path applies: the policy sees real frames through this
    function and imagined frames through the diffusion sampler, and the two have
    to land in the same domain.
    """
    return np.asarray(frame, dtype=np.float32) / 127.5 - 1.0


class DiamondAgent:
    """
    DIAMOND: DIffusion As a Model Of eNvironment Dreams

    RL agent trained entirely within a diffusion world model.
    """

    def __init__(self, config: DiamondConfig, *, env: Optional[Any] = None) -> None:
        """Args:
        config: DIAMOND's settings.
        env: An environment to train on instead of the Atari one built from
            ``config.game``. It must follow the DIAMOND wrapper's API:
            ``reset() -> (obs, info)`` and ``step(a) -> (obs, reward, done,
            info)`` with ``info["terminated"]`` / ``info["truncated"]``, and
            ``(obs_size, obs_size, 3)`` uint8 frames.
        """
        self.config = coerce_config(DiamondConfig, config)
        config = self.config
        from simulusion.utils.device import resolve_device

        self.device = resolve_device(config.device)

        self.env = (
            env
            if env is not None
            else make_diamond_atari_env(
                game=config.game,
                frameskip=config.frameskip,
                max_noop=config.max_noop,
                terminate_on_life_loss=config.terminate_on_life_loss,
                reward_clip=True,
                resize=(config.obs_size, config.obs_size),
                seed=config.seed,
            )
        )

        # action_space may be different Space types; prefer Discrete.n when available
        # Declare attribute type for static checkers
        self.action_dim: int = 0

        if isinstance(self.env.action_space, Discrete):
            self.action_dim = int(self.env.action_space.n)
        elif isinstance(self.env.action_space, Box):
            # continuous action space -> flatten dimensions
            shape = getattr(self.env.action_space, "shape", None)
            if shape is None:
                raise TypeError("Box action_space has no shape")
            # ensure shape is numeric sequence before taking product
            self.action_dim = int(np.prod(tuple(shape)))
        else:
            # fallback: try to read 'n' attribute, else raise informative error
            if hasattr(self.env.action_space, "n"):
                self.action_dim = int(getattr(self.env.action_space, "n"))
            else:
                raise TypeError(
                    f"Unsupported action_space type: {type(self.env.action_space)}"
                )

        self.use_amp = bool(
            getattr(config, "use_amp", True)
            and getattr(self.device, "type", str(self.device)) == "cuda"
        )
        # What `use_amp` selects: fp16 autocast with loss scaling on CUDA. The
        # shared Trainer replaces it with the run's precision (set_precision).
        self.precision = MixedPrecision.from_use_amp(self.use_amp, self.device)

        self._build_models()

        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            obs_shape=(config.obs_size, config.obs_size, 3),
            action_dim=1,
            device=config.device,
        )

        self.obs_history: List[np.ndarray] = []
        # keep a raw-uint8 history in parallel with the normalized history
        self.obs_history_raw: List[np.ndarray] = []
        self.action_history: List[int] = []

        self.total_steps = 0
        self.global_step = 0
        # last LSTM hidden states (saved for reproducible imagined rollouts)
        self.last_policy_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.last_reward_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # policy hidden state maintained across collection steps for temporal consistency
        self.collect_policy_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @classmethod
    def from_config(
        cls,
        config: DiamondConfig | dict | str | Path | None = None,
        **overrides: Any,
    ) -> "DiamondAgent":
        """Build a DIAMOND agent from a config object, dict, YAML file, or YAML string."""

        args = apply_config_overrides(coerce_config(DiamondConfig, config), overrides)
        return cls(args)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: DiamondConfig | dict | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        **overrides: Any,
    ) -> "DiamondAgent":
        """Load a DIAMOND checkpoint from a local path/directory or HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("checkpoint.pt", "model.pt", "diamond.pt", "pytorch_model.bin")
        )
        checkpoint_path = resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Could not find a DIAMOND checkpoint for "
                f"{pretrained_model_name_or_path!r}."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if config is None and isinstance(checkpoint.get("config"), dict):
            args = DiamondConfig.from_dict(checkpoint["config"])
        elif config is None:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "diamond_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = DiamondConfig.from_yaml(config_path)
        else:
            args = coerce_config(DiamondConfig, config)
        agent = cls(apply_config_overrides(args, overrides))
        agent.load_checkpoint(str(checkpoint_path))
        return agent

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            param.numel()
            for module in (self.diffusion_model, self.reward_model, self.actor_critic)
            for param in module.parameters()
            if not trainable_only or param.requires_grad
        )

    def summary(self) -> dict:
        return module_summary(
            {
                "diffusion_model": self.diffusion_model,
                "reward_model": self.reward_model,
                "actor_critic": self.actor_critic,
            }
        )

    def _build_models(self) -> None:
        """Initialize all DIAMOND models."""
        from simulusion.world_model.diamond import build_diamond_networks

        # The world model's networks are built in one place, shared with
        # DiamondWorldModel, so a checkpoint loads into either.
        self.diffusion_model, self.sampler, self.reward_model = build_diamond_networks(
            self.config, self.action_dim, self.device
        )
        self.edm_precond = self.sampler.edm_precond

        self.reward_loss_fn = RewardTerminationLoss()

        self.actor_critic = ActorCriticNetwork(
            obs_channels=3,
            action_dim=self.action_dim,
            channels=tuple(self.config.actor_channels),
            lstm_dim=self.config.actor_lstm_dim,
            res_blocks=self.config.actor_res_blocks,
            frame_size=self.config.obs_size,
        ).to(self.device)

        self.rl_loss_fn = RLLoss(
            discount_factor=self.config.discount_factor,
            lambda_returns=self.config.lambda_returns,
            entropy_weight=self.config.entropy_weight,
        )

        self.diffusion_opt = optim.AdamW(
            self.diffusion_model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay_diffusion,
        )

        self.reward_opt = optim.AdamW(
            self.reward_model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay_reward,
        )

        self.actor_opt = optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay_actor,
        )
        self._build_updates()

    def _build_updates(self, all_reduce: bool = False) -> None:
        # DIAMOND does not clip gradients (Appendix C lists no clipping).
        self.diffusion_update = GradientUpdate(
            self.diffusion_opt,
            self.diffusion_model.parameters(),
            self.precision,
            all_reduce=all_reduce,
        )
        self.reward_update = GradientUpdate(
            self.reward_opt,
            self.reward_model.parameters(),
            self.precision,
            all_reduce=all_reduce,
        )
        self.actor_update = GradientUpdate(
            self.actor_opt, self.actor_critic.parameters(), self.precision
        )

    def set_precision(
        self, precision: MixedPrecision, *, all_reduce: bool = False
    ) -> None:
        """Train at ``precision`` from now on (fresh loss scalers).

        ``all_reduce`` averages the world model's gradients across processes,
        for data-parallel training of the denoiser and reward model.
        """
        self.precision = precision
        self.use_amp = precision.enabled
        self._build_updates(all_reduce=all_reduce)

    def _move_batch_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: (
                value.to(self.device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for key, value in batch.items()
        }

    def _update_diffusion_model(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update diffusion world model."""
        self.diffusion_model.train()

        obs_seq = batch["obs_seq"].to(self.device)
        action_seq = batch["action_seq"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)

        B, T, C, H, W = obs_seq.shape

        # use only the last `num_conditioning_frames` for conditioning
        obs_history = obs_seq[:, -self.config.num_conditioning_frames :]
        target_obs = next_obs

        sigma = self.edm_precond.sample_noise_level(B, self.device)
        sigma = sigma.view(B, 1, 1, 1)

        noise = torch.randn_like(target_obs)
        noisy_target = target_obs + sigma * noise

        precond = self.edm_precond.get_preconditioners(sigma)
        model_input = precond["c_in"] * noisy_target

        # use c_noise (log-sigma transform) for time conditioning as in EDM
        t_cond = precond["c_noise"].squeeze(-1).squeeze(-1)

        # Debug asserts: ensure shapes are as expected
        with self.precision.autocast():
            model_output: Any = self.diffusion_model(
                x=model_input,
                t=t_cond,
                obs_history=obs_history,
                actions=action_seq[:, -self.config.num_conditioning_frames :],
            )

            target = (next_obs - precond["c_skip"] * noisy_target) / precond["c_out"]

            loss = F.mse_loss(model_output, target)

        self.diffusion_update(loss)

        return loss.detach().item()

    def _update_reward_model(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update reward/termination model."""
        self.reward_model.train()

        obs_seq = batch["obs_seq"].to(self.device)
        action_seq = batch["action_seq"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)

        B, T, C, H, W = obs_seq.shape

        with self.precision.autocast():
            reward_logits, term_logits, _ = self.reward_model(
                obs=obs_seq,
                actions=action_seq,
            )

            # Align target sequence lengths: reward_logits has same temporal length as obs_seq
            # We predict next-step rewards for all but the last conditioning frame.
            # Align rewards/dones length with reward_logits[:, :-1]
            T_logits = reward_logits.shape[1]
            target_len = max(0, T_logits - 1)
            rewards_target = rewards[:, :target_len]
            dones_target = dones[:, :target_len]

            total_loss, reward_loss, term_loss = self.reward_loss_fn(
                reward_logits=reward_logits[:, :-1],
                termination_logits=term_logits[:, :-1],
                rewards=rewards_target,
                terminated=dones_target,
            )

        self.reward_update(total_loss)

        return total_loss.detach()

    def _update_actor_critic(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update actor-critic using imagined rollouts.

        This replaces using real dataset trajectories for policy/value updates
        and follows the paper: compute policy and value losses on imagined
        trajectories produced by the diffusion world model.
        """
        self.actor_critic.train()

        obs_seq = batch["obs_seq"].to(self.device)
        action_seq = batch.get("action_seq", batch.get("actions"))
        if action_seq is not None:
            action_seq = action_seq.to(self.device)

        B, seq_T, C, H, W = obs_seq.shape

        # Determine burn-in / conditioning length and horizon
        burn_in = self.config.burn_in_length

        # safety checks
        assert seq_T >= burn_in, "Sequence shorter than burn-in"

        # initial conditioning sequences for imagination
        obs_history = obs_seq[:, :burn_in]
        if action_seq is None:
            action_history = torch.zeros(
                (B, burn_in), dtype=torch.long, device=self.device
            )
        else:
            action_history = action_seq[:, :burn_in]

        # Burn in both recurrent models over the conditioning window (paper
        # Algorithm 1: "Burn-in buffer with R_psi, pi_phi and V_phi to initialize
        # LSTM states"). The burn-in stops one frame short of the end: imagination
        # begins *at* x_{L-1}, so the policy must still consume that frame itself
        # and the reward model must still see the (x_{L-1}, a_{L-1}) pair. Feeding
        # them the whole window here would double-count it.
        #
        # The states are always recomputed from this batch's own burn-in frames.
        # Reusing `last_policy_hidden` / `last_reward_hidden` across updates --
        # as an earlier version did -- froze the state at whatever the first
        # batch produced and leaked one batch's recurrent context into every
        # later, unrelated batch. Those attributes remain for checkpointing.
        with torch.no_grad():
            reward_hidden = self.reward_model.init_hidden(B, self.device)
            policy_hidden_init = self.actor_critic.init_hidden(B, self.device)
            if burn_in > 1:
                _, _, reward_hidden = self.reward_model(
                    obs_history[:, :-1], action_history[:, :-1], reward_hidden
                )
                _, _, policy_hidden_init = self.actor_critic(
                    obs_history[:, :-1], policy_hidden_init
                )

        # store a CPU copy so checkpoints are device independent
        try:
            self.last_policy_hidden = (
                policy_hidden_init[0].detach().cpu(),
                policy_hidden_init[1].detach().cpu(),
            )
        except Exception:
            self.last_policy_hidden = None

        # Imagine a trajectory and obtain policy actions taken during imagination
        (
            obs_imag,
            rewards_imag,
            dones_imag,
            policy_actions_imag,
            reward_hidden,
        ) = self._imagine_trajectory(
            obs_history, action_history, reward_hidden, policy_hidden=policy_hidden_init
        )

        # store the final reward hidden state (CPU) for checkpointing / replay
        try:
            self.last_reward_hidden = (
                reward_hidden[0].detach().cpu(),
                reward_hidden[1].detach().cpu(),
            )
        except Exception:
            self.last_reward_hidden = None

        # Compute policy logits and values for imagined observations using the
        # same initial policy hidden state used during imagination so the
        # logits align with the sampled actions.
        with self.precision.autocast():
            # obs_imag is x_0..x_H, so this yields H+1 logits/values. The
            # trailing value is V(x_H), the lambda-return's bootstrap (paper
            # eq. 14); the trailing logits belong to an action that was never
            # taken and are dropped.
            policy_logits, values, _ = self.actor_critic(obs_imag, policy_hidden_init)

            # values: [B, H+1, 1] -> [B, H+1]
            values_full = values.squeeze(-1)

            # lambda-returns on imagined rewards
            lambda_returns = self.rl_loss_fn.compute_lambda_returns(
                rewards=rewards_imag,
                values=values_full,
                dones=dones_imag,
            )

            policy_loss = self.rl_loss_fn.policy_loss(
                policy_logits=policy_logits[:, :-1],
                actions=policy_actions_imag,
                lambda_returns=lambda_returns,
                values=values_full,
            )

            value_loss = self.rl_loss_fn.value_loss(
                values=values_full.unsqueeze(-1),
                lambda_returns=lambda_returns,
            )

            total_loss = policy_loss + value_loss

        self.actor_update(total_loss)

        return policy_loss.detach(), value_loss.detach()

    def _collect_experience(self, num_steps: int) -> List[float]:
        """Collect experience from the real environment."""
        rewards = []

        if len(self.obs_history) == 0:
            raw_obs, _ = self.env.reset()
            norm_obs = _normalize_frame(raw_obs)
            # maintain both normalized and raw histories
            self.obs_history = [norm_obs] * self.config.num_conditioning_frames
            self.obs_history_raw = [raw_obs] * self.config.num_conditioning_frames

        for _ in tqdm(range(num_steps), desc="Collecting experience", leave=False):
            # build tensor [1, L, C, H, W] with channels-first
            obs_np = np.stack(self.obs_history[-self.config.num_conditioning_frames :])
            # obs_np: [L, H, W, C] -> transpose to [L, C, H, W]
            obs_np = obs_np.transpose(0, 3, 1, 2)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

            # maintain LSTM hidden state across collection steps
            if self.collect_policy_hidden is None:
                self.collect_policy_hidden = self.actor_critic.init_hidden(
                    1, self.device
                )

            action, self.collect_policy_hidden = self.actor_critic.get_action(
                obs_tensor[:, -1],
                self.collect_policy_hidden,
                deterministic=False,
            )

            if random.random() < self.config.epsilon_greedy:
                action = self.env.action_space.sample()

            next_obs, reward, done, step_info = self.env.step(action)

            # env typically returns uint8 frames; keep raw and normalized
            next_obs_raw = next_obs
            next_obs = _normalize_frame(next_obs_raw)

            # R_psi's d_t head models episode *termination* (Algorithm 1), which
            # for DIAMOND includes life loss (Table 3) but not a time-limit
            # truncation: the MDP would have continued past a truncation, so
            # storing it as terminal teaches the model an ending the game never
            # produces. The episode is reset on `done` either way.
            terminated = bool(step_info.get("terminated", done))
            truncated = bool(step_info.get("truncated", False)) or (
                done and not terminated
            )

            # store raw uint8 frames in the replay buffer (avoid lossy casts)
            self.replay_buffer.add(
                obs=self.obs_history_raw[-1],
                action=action,
                reward=reward,
                done=terminated,
                next_obs=next_obs_raw,
                truncated=truncated,
            )

            rewards.append(reward)

            # update both normalized and raw histories
            self.obs_history.append(next_obs)
            self.obs_history_raw.append(next_obs_raw)
            self.action_history.append(action)
            # Only the last `num_conditioning_frames` are ever read. Without
            # trimming, these lists held every frame of the current episode.
            keep = self.config.num_conditioning_frames
            if len(self.obs_history) > keep:
                del self.obs_history[:-keep]
                del self.obs_history_raw[:-keep]
            if len(self.action_history) > keep:
                del self.action_history[:-keep]

            if done:
                raw_obs, _ = self.env.reset()
                norm_obs = _normalize_frame(raw_obs)
                self.obs_history = [norm_obs] * self.config.num_conditioning_frames
                self.obs_history_raw = [raw_obs] * self.config.num_conditioning_frames
                self.action_history = []
                self.collect_policy_hidden = None

        return rewards

    @torch.no_grad()
    def _imagine_trajectory(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        policy_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Imagine a trajectory using the diffusion world model.

        Args:
            obs_history: Initial observations [B, L, C, H, W]
            action_history: Initial actions [B, L]
            hidden_state: Initial LSTM hidden state

        Returns:
            obs_trajectory: [B, H+1, C, H, W] -- x_0 (the last conditioning
                frame) followed by the H imagined frames, so the trailing frame
                supplies the bootstrap value V(x_H).
            actions: [B, H] actions the policy took, a_i sampled at x_i
            rewards: [B, H] rewards r_i earned by a_i at x_i
            dones: [B, H]
            hidden_state: Updated hidden state
        """
        B = obs_history.shape[0]
        horizon = self.config.imagination_horizon

        # Paper Algorithm 1 (update_actor_critic) orders each imagined step as:
        #     a_i ~ pi(a_i | x_i);  r_i, d_i ~ R(x_i, a_i);  x_{i+1} ~ D(x_<=i, a_<=i)
        # i.e. the policy acts on the *current* observation, the reward model is
        # queried on that same (observation, action) pair -- which is exactly the
        # pairing it is trained on in `_update_reward_model` -- and only then is
        # the next frame sampled. Predicting the frame first and then feeding the
        # reward model (x_{i+1}, a_i) both skews it away from its training
        # distribution and shifts every reward one step relative to the action
        # that earned it, which corrupts the REINFORCE advantage.
        #
        # `obs_window` holds the L frames the diffusion model conditions on,
        # ending at the current observation x_i. `prev_actions` holds the L-1
        # actions that precede it; a_i is appended at sampling time.
        # `action_history[:, j]` is the action taken at `obs_history[:, j]`, so
        # those are a_0..a_{L-2}. The recorded a_{L-1} is dropped: the policy
        # picks the action at x_{L-1} itself. Slicing `[:, 1:]` instead paired
        # every conditioning frame with the action one step after it.
        obs_window = obs_history
        prev_actions = action_history[:, :-1]

        current_obs = obs_history[:, -1]

        # x_i for i = 0..H: one more frame than there are actions, so V(x_H) can
        # bootstrap the lambda-return.
        obs_trajectory = [current_obs]
        rewards_list = []
        dones_list = []
        policy_actions_list = []

        # initialize a policy hidden state for batched policy sampling during imagination
        # allow caller to provide an initial policy hidden state (primed by
        # burn-in sequence); otherwise initialize fresh hidden state
        if policy_hidden is None:
            policy_hidden = self.actor_critic.init_hidden(B, self.device)

        for _t in range(horizon):
            # a_i ~ pi(. | x_i). The policy LSTM state is carried across imagined
            # timesteps so the policy conditions on the whole imagined trajectory.
            policy_actions, policy_hidden = self.actor_critic.get_actions(
                current_obs, policy_hidden, deterministic=False
            )

            # r_i, d_i ~ R(x_i, a_i)
            reward, done, next_hidden_state = self.reward_model.predict(
                obs=current_obs,
                actions=policy_actions,
                hidden_state=hidden_state,
            )
            hidden_state = cast(Tuple[torch.Tensor, torch.Tensor], next_hidden_state)

            # x_{i+1} ~ D(x_{i-L+1..i}, a_{i-L+1..i})
            actions_window = torch.cat(
                [prev_actions, policy_actions.unsqueeze(-1)], dim=1
            )
            sampled = self.sampler.sample(
                model=self.diffusion_model,
                shape=(B, 3, self.config.obs_size, self.config.obs_size),
                device=self.device,
                obs_history=obs_window,
                actions=actions_window,
            )

            policy_actions_list.append(policy_actions)
            rewards_list.append(reward)
            dones_list.append(done)
            obs_trajectory.append(sampled)

            # Slide both conditioning windows forward by one step.
            obs_window = torch.cat([obs_window[:, 1:], sampled.unsqueeze(1)], dim=1)
            prev_actions = actions_window[:, 1:]
            current_obs = sampled

        return (
            torch.stack(obs_trajectory, dim=1),
            torch.stack(rewards_list, dim=1),
            torch.stack(dones_list, dim=1),
            torch.stack(policy_actions_list, dim=1),
            hidden_state,
        )

    def train(self) -> None:
        """Train for ``config.num_epochs`` epochs of Algorithm 1.

        Runs on the shared :class:`~simulusion.training.Trainer` through the
        ``diamond`` task, with this agent's environment, replay and networks.
        Evaluation runs every ``eval_interval`` epochs, early stopping on its
        reward when ``early_stopping`` is set, and ``checkpoint_<epoch>.pt``
        is written to ``checkpoint_dir`` every ``save_interval`` epochs and at
        the end. The trainer adds resumable checkpoints under
        ``<checkpoint_dir>/checkpoints/``, ``run.json``, ``run_config.yaml``
        and ``train_metrics.jsonl``.
        """
        from simulusion.training.legacy import fit_legacy
        from simulusion.training.tasks.diamond import DiamondTask, DiamondTaskConfig

        config = self.config
        print(f"Training DIAMOND on {config.game}")
        print(f"Device: {self.device}")
        print(f"Action space: {self.action_dim}")
        early = bool(getattr(config, "early_stopping", False))
        fit_legacy(
            DiamondTask(DiamondTaskConfig(model=config), agent=self),
            config.checkpoint_dir,
            max_steps=config.num_epochs,
            device=self.device,
            seed=config.seed,
            log_every=config.log_interval,
            eval_every=config.eval_interval,
            checkpoint_every=config.save_interval,
            # Return, not loss: the diffusion loss keeps falling long after
            # the policy stops improving, so a loss plateau never arrives.
            early_stopping_metric="eval_reward" if early else None,
            early_stopping_mode="max",
            patience=config.patience,
            min_delta=config.min_delta,
        )

    def train_epoch(self) -> Optional[Dict[str, float]]:
        """One epoch of Algorithm 1: collect, then train every component.

        Collects ``environment_steps_per_epoch`` steps, then runs
        ``training_steps_per_epoch`` updates of the diffusion model, the
        reward/termination model and the actor-critic. Returns the epoch's
        mean losses, or ``None`` while the replay holds too little to train.
        """
        collected_rewards = self._collect_experience(
            self.config.environment_steps_per_epoch
        )

        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None

        dataset = SequenceDataset(
            replay_buffer=self.replay_buffer,
            sequence_length=self.config.burn_in_length
            + self.config.imagination_horizon,
            burn_in=self.config.burn_in_length,
        )
        num_workers = getattr(self.config, "data_loader_num_workers", 4)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=getattr(self.config, "pin_memory", True)
            and self.device.type == "cuda",
            persistent_workers=getattr(self.config, "persistent_workers", True)
            and num_workers > 0,
        )

        diffusion_losses = []
        reward_losses = []
        policy_losses = []
        value_losses = []

        # iterate over the dataloader properly; avoid recreating iterator each step
        data_iter = iter(dataloader)
        for _ in tqdm(
            range(self.config.training_steps_per_epoch),
            desc="Training",
            leave=False,
        ):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = self._move_batch_to_device(batch)

            diffusion_loss = self._update_diffusion_model(batch)
            diffusion_losses.append(diffusion_loss)

            reward_loss = self._update_reward_model(batch)
            reward_losses.append(reward_loss)

            policy_loss, value_loss = self._update_actor_critic(batch)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        return {
            "diffusion_loss": float(torch.tensor(diffusion_losses).mean()),
            "reward_loss": float(torch.tensor(reward_losses).mean()),
            "policy_loss": float(torch.stack(policy_losses).mean()),
            "value_loss": float(torch.stack(value_losses).mean()),
            "collected_reward": float(np.mean(collected_rewards)),
        }

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 1) -> float:
        """Evaluate the agent."""
        self.actor_critic.eval()
        self.diffusion_model.eval()
        self.reward_model.eval()

        total_reward = 0.0

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            obs = _normalize_frame(obs)

            obs_history = [obs] * self.config.num_conditioning_frames
            # initialize separate hidden states for reward model and policy
            policy_hidden = self.actor_critic.init_hidden(1, self.device)

            done = False
            episode_reward = 0.0
            steps = 0

            # Atari's standard 27000-agent-step cap. Without it a policy that
            # has learnt to stall (common early in training, and the reason
            # Atari benchmarks impose the limit at all) makes evaluation hang.
            while not done and steps < self.config.max_episode_steps:
                obs_np = np.stack(obs_history[-self.config.num_conditioning_frames :])
                obs_np = obs_np.transpose(0, 3, 1, 2)
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

                # pass batched observation [1, C, H, W]
                action, next_policy_hidden = self.actor_critic.get_action(
                    obs_tensor[:, -1],
                    policy_hidden,
                    deterministic=True,
                )
                policy_hidden = cast(
                    Tuple[torch.Tensor, torch.Tensor], next_policy_hidden
                )

                next_raw, reward, done, _ = self.env.step(action)

                episode_reward += reward
                steps += 1

                obs_history.append(_normalize_frame(next_raw))
                # Only the last L frames are ever read; keeping the whole episode
                # grew this list to tens of thousands of 64x64x3 float arrays.
                if len(obs_history) > self.config.num_conditioning_frames:
                    del obs_history[: -self.config.num_conditioning_frames]

            total_reward += episode_reward

        return total_reward / num_episodes

    def _compute_human_normalized_score(self, score: float) -> float:
        """Compute human-normalized score."""
        game = self.config.game
        human = HUMAN_SCORES.get(game, 1.0)
        random = RANDOM_SCORES.get(game, 0.0)

        if human == random:
            return 0.0

        return (score - random) / (human - random)

    def save_checkpoint(self, path: Optional[Union[str, os.PathLike]] = None) -> None:
        """Save model checkpoint.

        Args:
            path: Optional path where to write the checkpoint. If `path` is None
                or a bare filename, the file is written into
                `config.checkpoint_dir/<filename>`. If `path` contains a
                directory component or is an absolute/relative path, it is used
                directly. When `path` is None the checkpoint is written to
                `config.checkpoint_dir/checkpoint.pt`.
        """
        # Determine output path. A bare filename or None lands in the
        # configured checkpoint directory (checkpoints/diamond by default).

        default_dir = Path(self.config.checkpoint_dir)
        if path is None:
            default_dir.mkdir(parents=True, exist_ok=True)
            out_path = default_dir / "checkpoint.pt"
        else:
            pathp = Path(path)
            if pathp.parent != Path(""):
                pathp.parent.mkdir(parents=True, exist_ok=True)
                out_path = pathp
            else:
                default_dir.mkdir(parents=True, exist_ok=True)
                out_path = default_dir / pathp
        # Trim replay buffer arrays to current size to avoid saving full-capacity
        # arrays which are wasteful for checkpoints and can blow up memory.
        # Trim and persist replay buffer arrays to separate numpy file(s) to
        # avoid saving large Python objects inside the torch checkpoint. This
        # reduces checkpoint size and avoids unsafe pickle/unpickle usage when
        # restoring Python lists/containers.
        rb_state_trim: dict[str, Any] | None = None
        replay_file: str | None = None
        obs_file: str | None = None
        try:
            rb_state = self.replay_buffer.state_dict()
            n = int(self.replay_buffer.size)
            if n > 0:
                rb_state_trim = {
                    "observations": rb_state["observations"][:n].copy(),
                    "next_observations": rb_state["next_observations"][:n].copy(),
                    "actions": rb_state["actions"][:n].copy(),
                    "rewards": rb_state["rewards"][:n].copy(),
                    "dones": rb_state["dones"][:n].copy(),
                    "truncations": rb_state["truncations"][:n].copy(),
                    "position": int(self.replay_buffer.position),
                    "size": int(n),
                    # The buffer's real capacity, not the trimmed length: it
                    # used to record `n`, so a resumed run continued with a
                    # replay only as large as it was at the checkpoint and
                    # overwrote its oldest data from then on.
                    "capacity": int(self.replay_buffer.capacity),
                }
            else:
                rb_state_trim = rb_state
        except Exception:
            rb_state_trim = None

        # Prepare checkpoint (model weights + metadata). We keep hidden states
        # in the torch checkpoint but save large numpy arrays to separate files
        # with a common basename derived from the output path.
        save_config_next_to_checkpoint(self.config, out_path)
        checkpoint: dict[str, Any] = {
            "config": self.config.to_dict(),
            "diffusion_model": self.diffusion_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "diffusion_opt": self.diffusion_opt.state_dict(),
            "reward_opt": self.reward_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            # optional LSTM hidden states saved for reproducible imagination
            "last_policy_hidden": self.last_policy_hidden,
            "last_reward_hidden": self.last_reward_hidden,
        }

        # If we have a trimmed replay buffer, write it to a compressed npz in
        # the same directory as the checkpoint. Also write obs_history_raw as
        # a stacked numpy file if present. Record filenames inside the torch
        # checkpoint for later restoration.
        base = os.path.splitext(str(out_path))[0]
        if rb_state_trim is not None:
            replay_file = base + "_replay.npz"
            # np.savez_compressed will store arrays with the provided keys
            try:
                np.savez_compressed(replay_file, **rb_state_trim)
                # Just the file name: it sits next to the checkpoint, and an
                # absolute path breaks as soon as the directory is moved.
                checkpoint["replay_buffer_file"] = os.path.basename(replay_file)
            except Exception:
                # If saving fails, do not embed large Python objects in the
                # torch checkpoint; simply omit the replay buffer files and
                # warn the caller.
                print("Warning: failed to write replay buffer file; skipping embedding")

        if self.obs_history_raw is not None and len(self.obs_history_raw) > 0:
            obs_file = base + "_obs.npy"
            try:
                # Stack into a single array (N, H, W, C)
                obs_arr = np.stack(self.obs_history_raw)
                np.save(obs_file, obs_arr)
                checkpoint["obs_history_file"] = os.path.basename(obs_file)
            except Exception:
                print("Warning: failed to write obs_history file; skipping embedding")

        torch.save(checkpoint, out_path)

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        """Load model checkpoint.

        Args:
            path: Optional path to checkpoint. If None,
                `config.checkpoint_dir/checkpoint.pt` is loaded. If a bare
                filename is provided, we try
                `config.checkpoint_dir/<filename>`; if a path with directory
                components is provided we use it directly.
        """
        # Resolve path similarly to save_checkpoint behaviour
        default_dir = str(self.config.checkpoint_dir)
        if path is None:
            fpath = os.path.join(default_dir, "checkpoint.pt")
        else:
            if os.path.exists(path):
                fpath = path
            else:
                alt = os.path.join(default_dir, path)
                if os.path.exists(alt):
                    fpath = alt
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {path} or {alt}")

        # The checkpoint stores tensor state_dicts and primitive metadata only;
        # larger replay/observation arrays are loaded from separate npz/npy
        # files below. Keep torch deserialization in weights-only mode to avoid
        # executing arbitrary pickle payloads from untrusted checkpoints.
        checkpoint = torch.load(fpath, map_location=self.device, weights_only=True)
        self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.diffusion_opt.load_state_dict(checkpoint["diffusion_opt"])
        self.reward_opt.load_state_dict(checkpoint["reward_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        # restore optional last hidden states (move to configured device)
        if (
            "last_policy_hidden" in checkpoint
            and checkpoint["last_policy_hidden"] is not None
        ):
            h, c = checkpoint["last_policy_hidden"]
            self.last_policy_hidden = (h.to(self.device), c.to(self.device))
        else:
            self.last_policy_hidden = None

        if (
            "last_reward_hidden" in checkpoint
            and checkpoint["last_reward_hidden"] is not None
        ):
            h, c = checkpoint["last_reward_hidden"]
            self.last_reward_hidden = (h.to(self.device), c.to(self.device))
        else:
            self.last_reward_hidden = None

        # restore replay buffer from separate npz file if provided; fall back
        # to embedded dict when necessary.
        if "replay_buffer_file" in checkpoint:
            try:
                replay_file = _beside(fpath, checkpoint["replay_buffer_file"])
                with np.load(replay_file, allow_pickle=False) as data:
                    rb_state = {k: data[k] for k in data.files}
                self.replay_buffer.load_state_dict(rb_state)
            except Exception:
                print(
                    "Warning: failed to load replay_buffer from file; trying embedded state"
                )
                try:
                    self.replay_buffer.load_state_dict(
                        checkpoint.get("replay_buffer", {})
                    )
                except Exception:
                    print("Warning: failed to load replay_buffer from checkpoint")
        elif "replay_buffer" in checkpoint and checkpoint["replay_buffer"] is not None:
            try:
                self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
            except Exception:
                print("Warning: failed to load replay_buffer from checkpoint")

        # restore obs_history_raw from separate .npy file if present
        if "obs_history_file" in checkpoint:
            try:
                obs_file = _beside(fpath, checkpoint["obs_history_file"])
                obs_arr = np.load(obs_file, allow_pickle=False)
                self.obs_history_raw = [o for o in obs_arr]
                self.obs_history = [_normalize_frame(o) for o in self.obs_history_raw]
            except Exception:
                print(
                    "Warning: failed to load obs_history from file; trying embedded state"
                )
                try:
                    self.obs_history_raw = checkpoint.get("obs_history_raw", [])
                    self.obs_history = [
                        _normalize_frame(o) for o in self.obs_history_raw
                    ]
                except Exception:
                    print("Warning: failed to load obs_history_raw from checkpoint")
        elif (
            "obs_history_raw" in checkpoint
            and checkpoint["obs_history_raw"] is not None
        ):
            try:
                self.obs_history_raw = checkpoint["obs_history_raw"]
                self.obs_history = [_normalize_frame(o) for o in self.obs_history_raw]
            except Exception:
                print("Warning: failed to load obs_history_raw from checkpoint")


def _beside(checkpoint_path: str, name: str) -> str:
    """A side file of a checkpoint, by name (or a legacy absolute path)."""
    local = os.path.join(os.path.dirname(checkpoint_path), os.path.basename(name))
    return local if os.path.exists(local) else name


def train_diamond(
    game: str | None = None,
    seed: int | None = None,
    preset: Optional[str] = None,
    device: str | None = None,
    config: DiamondConfig | None = None,
) -> None:
    """Train DIAMOND on a specific game or a composed experiment config."""
    if config is None:
        config = DiamondConfig()
        if game is not None:
            config.game = game
        if seed is not None:
            config.seed = seed
        if preset is not None:
            config.preset = preset
            config.__post_init__()
        if device is not None:
            config.device = device

    agent = DiamondAgent(config)
    agent.train()


def main(argv: list[str] | None = None) -> Any:
    """Compose DIAMOND config from YAML/dot-list overrides and launch training."""
    args = parse_experiment_args(argv, description="Train DIAMOND on Atari")
    config = instantiate_dataclass(DiamondConfig, args.config, args.overrides)
    if args.print_config:
        print(dump_config(config.__dict__))
        return config
    train_diamond(config=config)
    return config


if __name__ == "__main__":
    main()
