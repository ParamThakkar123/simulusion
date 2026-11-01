"""DreamerV3: a general world-model agent with fixed hyperparameters.

DreamerV3 learns a world model from replayed experience and improves its policy
purely on trajectories imagined by that model. Relative to DreamerV2 the changes
that matter are robustness techniques rather than architecture search:

============================  ====================================================
Technique                     Effect
============================  ====================================================
Categorical latents + unimix  Well-behaved KL terms, no deterministic collapse
Free bits on KL terms         A small representation loss without collapse
symexp two-hot reward/critic  Gradient scale decoupled from target scale
Percentile return norm        One entropy scale works for sparse and dense rewards
Critic EMA + replay loss      Stable bootstrapping, better values under hard rewards
LaProp + adaptive clipping    Loss-scale-independent optimization
============================  ====================================================

The class exposes the same surface as :class:`~simulusion.models.dreamer.Dreamer`
(``world_model_loss``, ``train_one_batch``, ``act_and_collect_data``,
``evaluate``, ``save``, ``restore_checkpoint``), so
:class:`~simulusion.models.dreamer.DreamerAgent`'s training loop drives it
without modification.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from simulusion.configs.dreamer_v3_config import DreamerV3Config
from simulusion.memory.dreamer_v3_memory import DreamerV3ReplayBuffer
from simulusion.models.categorical_rssm import CategoricalRSSM
from simulusion.models.dreamer import (
    DreamerAgent,
    _apply_config_overrides,
    _resolve_pretrained_file,
    _save_config_next_to_checkpoint,
    get_available_memory,
    make_env,
)
from simulusion.optim import LaProp, adaptive_grad_clip_
from simulusion.utils.dreamer_utils import FreezeParameters
from simulusion.utils.dreamer_v3_utils import (
    ReturnNormalizer,
    free_bits,
    lambda_return,
)
from simulusion.utils.logging_utils import assert_finite, get_package_logger
from simulusion.vision.dreamer_v3_nets import (
    DreamerV3Actor,
    DreamerV3Decoder,
    DreamerV3Encoder,
    DreamerV3Head,
    preprocess_image,
)

logger = get_package_logger(__name__)

__all__ = ["DreamerV3", "DreamerV3Agent", "coerce_dreamer_v3_config"]


def coerce_dreamer_v3_config(config: Any | None) -> DreamerV3Config:
    """Normalize config inputs to a :class:`DreamerV3Config` instance."""
    if config is None:
        return DreamerV3Config()
    if isinstance(config, DreamerV3Config):
        return config
    if isinstance(config, dict):
        return DreamerV3Config.from_dict(config)
    if isinstance(config, (str, Path)):
        return DreamerV3Config.from_yaml(config)
    raise TypeError(
        "config must be a DreamerV3Config, dict, YAML path/string, or None; "
        f"got {type(config).__name__}."
    )


class DreamerV3:
    """World model, actor, and critic trained together from replayed experience.

    Args:
        args: A :class:`DreamerV3Config` (or any object exposing the same
            attributes).
        obs_shape: Observation shape, ``(C, H, W)`` or ``(D,)``.
        action_size: Number of discrete actions, or continuous action dimension.
        device: Torch device to place the modules on.
        restore: Whether to restore from ``args.checkpoint_path`` after building.
    """

    def __init__(
        self,
        args: Any,
        obs_shape: Any,
        action_size: int,
        device: torch.device | str,
        restore: bool = False,
    ) -> None:
        self.args = args
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_size = int(action_size)
        self.device = torch.device(device)
        self.restore = bool(restore)
        self.restore_path = getattr(args, "checkpoint_path", "")
        self.is_image_obs = len(self.obs_shape) == 3
        self.discrete_actions = str(getattr(args, "actor_dist", "auto")) == "onehot"
        self.last_metrics: dict[str, float] = {}

        obs_dtype = np.uint8 if self.is_image_obs else np.float32
        bytes_per_obs = int(np.prod(self.obs_shape)) * np.dtype(obs_dtype).itemsize
        # actions + reward + is_terminal + is_first
        bytes_per_sample = bytes_per_obs + self.action_size * 4 + 12
        max_buffer_size = int(
            (get_available_memory() * 0.8) // max(1, bytes_per_sample)
        )
        buffer_size = min(int(args.buffer_size), max_buffer_size)
        if buffer_size < int(args.buffer_size):
            logger.warning(
                "Reducing replay capacity from %s to %s due to memory constraints.",
                args.buffer_size,
                buffer_size,
            )

        self.data_buffer = DreamerV3ReplayBuffer(
            size=buffer_size,
            obs_shape=self.obs_shape,
            action_size=self.action_size,
            seq_len=int(args.train_seq_len),
            batch_size=int(args.batch_size),
            obs_dtype=obs_dtype,
            online_fraction=float(getattr(args, "online_fraction", 0.5)),
        )

        self.use_amp = bool(
            getattr(args, "use_amp", False) and self.device.type == "cuda"
        )
        self.return_norm = ReturnNormalizer(
            decay=float(args.return_norm_decay),
            limit=float(args.return_norm_limit),
            low=float(args.return_norm_low),
            high=float(args.return_norm_high),
        )
        self._build_model(restore=self.restore)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_model(self, restore: bool = False) -> None:
        args = self.args
        hidden = args.resolved_hidden_size
        deter = args.resolved_recurrent_units
        classes = args.resolved_latent_classes
        depth = args.resolved_cnn_depth

        self.obs_encoder = DreamerV3Encoder(
            self.obs_shape,
            cnn_depth=depth,
            hidden_size=hidden,
            mlp_layers=int(args.mlp_layers),
            activation=str(args.activation),
        ).to(self.device)

        self.rssm = CategoricalRSSM(
            action_size=self.action_size,
            embed_size=self.obs_encoder.embed_size,
            latent_dim=int(args.latent_dim),
            latent_classes=classes,
            deter_size=deter,
            hidden_size=hidden,
            gru_blocks=int(args.gru_blocks),
            unimix=float(args.unimix),
            activation=str(args.activation),
        ).to(self.device)

        feature_size = self.rssm.feature_size
        self.obs_decoder = DreamerV3Decoder(
            feature_size,
            self.obs_shape,
            encoder=self.obs_encoder,
            cnn_depth=depth,
            hidden_size=hidden,
            mlp_layers=int(args.mlp_layers),
            activation=str(args.activation),
        ).to(self.device)

        head_kwargs: dict[str, Any] = dict(
            layers=int(args.mlp_layers),
            units=hidden,
            num_bins=int(args.num_buckets),
            symlog_range=float(args.symlog_range),
            activation=str(args.activation),
        )
        self.reward_model = DreamerV3Head(
            feature_size, dist="symexp_twohot", zero_init_output=True, **head_kwargs
        ).to(self.device)
        self.continue_model = DreamerV3Head(
            feature_size, dist="binary", zero_init_output=False, **head_kwargs
        ).to(self.device)
        self.value_model = DreamerV3Head(
            feature_size, dist="symexp_twohot", zero_init_output=True, **head_kwargs
        ).to(self.device)
        # Slow-moving copy of the critic used as a regularization target. It is
        # never optimized directly, so its gradients stay disabled.
        self.slow_value_model = copy.deepcopy(self.value_model).to(self.device)
        for param in self.slow_value_model.parameters():
            param.requires_grad_(False)

        self.actor = DreamerV3Actor(
            feature_size,
            self.action_size,
            discrete=self.discrete_actions,
            layers=int(args.mlp_layers),
            units=hidden,
            unimix=float(args.unimix),
            min_std=float(args.actor_min_std),
            max_std=float(args.actor_max_std),
            activation=str(args.activation),
        ).to(self.device)

        self.world_model_modules: list[nn.Module] = [
            self.rssm,
            self.obs_encoder,
            self.obs_decoder,
            self.reward_model,
            self.continue_model,
        ]
        self.actor_modules: list[nn.Module] = [self.actor]
        self.value_modules: list[nn.Module] = [self.value_model]

        self.world_model_params = [
            param
            for module in self.world_model_modules
            for param in module.parameters()
        ]

        # DreamerV3 uses one learning rate and one optimizer configuration for
        # the world model, the actor, and the critic alike.
        def make_optimizer(parameters: list[torch.nn.Parameter]) -> LaProp:
            return LaProp(
                parameters,
                lr=float(args.learning_rate),
                betas=(float(args.opt_beta1), float(args.opt_beta2)),
                eps=float(args.opt_eps),
                weight_decay=float(args.weight_decay),
            )

        self.world_model_opt = make_optimizer(self.world_model_params)
        self.actor_opt = make_optimizer(list(self.actor.parameters()))
        self.value_opt = make_optimizer(list(self.value_model.parameters()))

        self.world_model_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.actor_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.value_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self._updates = 0
        if restore and self.restore_path:
            self.restore_checkpoint(self.restore_path)

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        obs_shape: tuple[int, ...] | None = None,
        action_size: int | None = None,
        device: str | torch.device | None = None,
        restore: bool | None = None,
        **overrides: Any,
    ) -> "DreamerV3":
        """Build a DreamerV3 core model from a config object, dict, or YAML file.

        When ``obs_shape`` or ``action_size`` is omitted, a temporary environment
        is constructed from the config to infer them.
        """
        args = cast(
            DreamerV3Config,
            _apply_config_overrides(coerce_dreamer_v3_config(config), overrides),
        )
        args.resolve()
        if obs_shape is None or action_size is None:
            env = make_env(args)
            if obs_shape is None:
                obs_shape = tuple(env.observation_space["image"].shape)
            if action_size is None:
                action_size = int(env.action_space.shape[0])
            if str(args.actor_dist) == "auto":
                args.actor_dist = _resolve_actor_dist(env)
        if str(args.actor_dist) == "auto":
            args.actor_dist = "normal"

        if device is not None:
            torch_device = torch.device(device)
        elif torch.cuda.is_available() and not args.no_gpu:
            torch_device = torch.device("cuda")
        else:
            torch_device = torch.device("cpu")
        should_restore = args.restore if restore is None else restore
        return cls(args, obs_shape, action_size, torch_device, should_restore)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: Any = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **overrides: Any,
    ) -> "DreamerV3":
        """Load a DreamerV3 checkpoint from a local path/directory or the HF Hub."""
        candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("model.pt", "pytorch_model.bin", "checkpoint.pt", "ckpt.pt")
        )
        checkpoint_path = _resolve_pretrained_file(
            pretrained_model_name_or_path,
            candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Could not find a DreamerV3 checkpoint for "
                f"{pretrained_model_name_or_path!r}."
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or "cpu", weights_only=True
        )
        checkpoint_config = (
            checkpoint.get("config") if isinstance(checkpoint, dict) else None
        )
        if config is not None:
            args = coerce_dreamer_v3_config(config)
        elif checkpoint_config is not None:
            args = coerce_dreamer_v3_config(checkpoint_config)
        else:
            config_path = _resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "dreamer_v3_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = DreamerV3Config.from_yaml(config_path)
        args = cast(DreamerV3Config, _apply_config_overrides(args, overrides))

        obs_shape = (
            checkpoint.get("obs_shape") if isinstance(checkpoint, dict) else None
        )
        action_size = (
            checkpoint.get("action_size") if isinstance(checkpoint, dict) else None
        )
        model = cls.from_config(
            args,
            obs_shape=tuple(obs_shape) if obs_shape is not None else None,
            action_size=int(action_size) if action_size is not None else None,
            device=map_location,
            restore=False,
        )
        model.restore_checkpoint(checkpoint_path, map_location=map_location)
        return model

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _named_modules(self) -> dict[str, nn.Module]:
        return {
            "rssm": self.rssm,
            "obs_encoder": self.obs_encoder,
            "obs_decoder": self.obs_decoder,
            "reward_model": self.reward_model,
            "continue_model": self.continue_model,
            "value_model": self.value_model,
            "actor": self.actor,
        }

    def parameter_count(self, trainable_only: bool = False) -> int:
        """Total number of parameters across the DreamerV3 modules."""
        return sum(
            param.numel()
            for module in self._named_modules().values()
            for param in module.parameters()
            if not trainable_only or param.requires_grad
        )

    def summary(self) -> dict[str, Any]:
        """Compact per-module parameter-count summary."""
        modules = self._named_modules()
        module_params = {
            name: sum(param.numel() for param in module.parameters())
            for name, module in modules.items()
        }
        trainable = {
            name: sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            for name, module in modules.items()
        }
        return {
            "total_parameters": sum(module_params.values()),
            "trainable_parameters": sum(trainable.values()),
            "modules": module_params,
            "trainable_modules": trainable,
        }

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """Map raw observations into the model's input space."""
        if self.is_image_obs:
            return preprocess_image(obs)
        return obs.to(torch.float32)

    # ------------------------------------------------------------------
    # World model
    # ------------------------------------------------------------------

    @assert_finite
    def world_model_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        is_terminal: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the world model loss over a batch of replayed sequences.

        Args:
            obs: ``(T, B, *obs_shape)`` raw observations.
            actions: ``(T, B, action_size)`` actions taken at each observation.
            rewards: ``(T, B)`` rewards received for those actions.
            is_terminal: ``(T, B)`` true-termination flags.
            is_first: ``(T, B)`` episode-start flags.

        Returns:
            ``(loss, posterior)`` where ``posterior`` holds the ``(T, B, ...)``
            model states used for actor-critic learning.
        """
        args = self.args
        processed = self.preprocess(obs)
        embed = self.obs_encoder(processed)

        # The sequence model consumes a_{t-1}, so shift the action sequence and
        # pad the first step with zeros.
        prev_actions = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)

        init_state = self.rssm.init_state(obs.shape[1], self.device)
        posterior, prior = self.rssm.observe_rollout(
            embed, prev_actions, is_first.unsqueeze(-1), init_state
        )
        features = self.rssm.get_feat(posterior)

        reconstruction = self.obs_decoder(features)
        recon_loss = self.obs_decoder.loss(reconstruction, processed)

        reward_dist = self.reward_model(features)
        reward_loss = -reward_dist.log_prob(rewards)

        continue_dist = self.continue_model(features)
        continue_loss = -continue_dist.log_prob(1.0 - is_terminal)

        # Dynamics loss trains the prior towards the posterior; representation
        # loss trains the posterior towards the prior. Free bits switch each off
        # once it is already minimized below one nat.
        dyn_loss = free_bits(
            self.rssm.kl_divergence(posterior["logit"].detach(), prior["logit"]),
            args.free_nats,
        )
        rep_loss = free_bits(
            self.rssm.kl_divergence(posterior["logit"], prior["logit"].detach()),
            args.free_nats,
        )

        prediction_loss = recon_loss + reward_loss + continue_loss
        loss = (
            float(args.beta_pred) * prediction_loss
            + float(args.beta_dyn) * dyn_loss
            + float(args.beta_rep) * rep_loss
        ).mean()

        self.last_metrics.update(
            {
                "wm/recon_loss": float(recon_loss.mean().detach()),
                "wm/reward_loss": float(reward_loss.mean().detach()),
                "wm/continue_loss": float(continue_loss.mean().detach()),
                "wm/dyn_loss": float(dyn_loss.mean().detach()),
                "wm/rep_loss": float(rep_loss.mean().detach()),
            }
        )
        return loss, posterior

    # ------------------------------------------------------------------
    # Imagination and actor-critic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _imagine(self, posterior: dict) -> dict:
        """Roll the world model forward under the current policy.

        Everything here is computed without gradients: DreamerV3 uses the
        Reinforce estimator, so the actor learns through ``log_prob`` of detached
        actions rather than by backpropagating through the dynamics.

        Indexing follows the replay buffer's convention throughout: the reward
        and continuation heads are trained so that their prediction at a state
        describes *leaving* that state. So for imagined states ``s_0 .. s_H``,
        the reward for action ``a_t`` is ``rewards[t]`` and the episode survives
        that transition with probability ``continues[t]`` -- both indexed by the
        state acted from, not the state arrived at. Mixing the two conventions
        shifts credit by one step and quietly corrupts every value estimate.

        Returns:
            A dict with the imagined features, returns, values, and the
            discounted weights used to average the actor and critic objectives.
        """
        args = self.args
        horizon = int(args.imagine_horizon)
        discount = float(args.discount)

        start = self.rssm.detach_state(self.rssm.seq_to_batch(posterior))
        with FreezeParameters(self.world_model_modules + self.actor_modules):
            states, actions = self.rssm.imagine_rollout(
                lambda feat: self.actor(feat), start, horizon
            )
            features = self.rssm.get_feat(states)
            rewards = self.reward_model(features).mean
            continues = self.continue_model(features).probs
            values = self.value_model(features).mean

        discounts = discount * continues

        returns = lambda_return(
            rewards=rewards[:-1],
            values=values[1:],
            continues=discounts[:-1],
            bootstrap=values[-1],
            lambda_=float(args.td_lambda),
        )

        # Probability of reaching each imagined state and acting from it. The
        # first state is a real replayed state, so it is reached with certainty.
        weights = torch.cumprod(
            torch.cat(
                [torch.ones_like(discounts[:1]), discounts[: horizon - 1]], dim=0
            ),
            dim=0,
        )

        return {
            "features": features,
            "actions": actions,
            "returns": returns,
            "values": values,
            "weights": weights,
            "rewards": rewards,
        }

    @assert_finite
    def actor_loss(self, imagined: dict) -> torch.Tensor:
        """Reinforce objective with normalized returns and an entropy bonus."""
        args = self.args
        features = imagined["features"][:-1]
        actions = imagined["actions"]
        returns = imagined["returns"]
        values = imagined["values"][:-1]
        weights = imagined["weights"]

        denominator = self.return_norm.update(returns)
        advantage = ((returns - values) / denominator).detach()

        dist = self.actor.dist(features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        objective = log_prob * advantage + float(args.actor_entropy) * entropy
        loss = -(weights * objective).mean()

        self.last_metrics.update(
            {
                "actor/entropy": float(entropy.mean().detach()),
                "actor/advantage": float(advantage.mean()),
                "actor/return_scale": self.return_norm.scale,
                "actor/imag_reward": float(imagined["rewards"].mean()),
            }
        )
        return loss

    @assert_finite
    def value_loss(
        self,
        imagined: dict,
        replay: dict | None = None,
    ) -> torch.Tensor:
        """Two-hot critic loss on imagined returns, plus regularizers.

        The critic is regularized towards an exponential moving average of its
        own weights, and optionally also fit to lambda-returns computed over
        replayed rewards.
        """
        args = self.args
        features = imagined["features"][:-1].detach()
        targets = imagined["returns"].detach()
        weights = imagined["weights"]

        dist = self.value_model(features)
        loss = -dist.log_prob(targets)

        if float(args.critic_ema_regularizer) > 0.0:
            with torch.no_grad():
                slow_dist = self.slow_value_model(features)
            loss = loss + float(args.critic_ema_regularizer) * dist.cross_entropy_to(
                slow_dist
            )

        total = float(args.critic_loss_scale) * (weights * loss).mean()

        if replay is not None and float(args.critic_replay_loss_scale) > 0.0:
            replay_dist = self.value_model(replay["features"])
            replay_loss = -replay_dist.log_prob(replay["returns"])
            total = total + float(args.critic_replay_loss_scale) * replay_loss.mean()
            self.last_metrics["critic/replay_loss"] = float(replay_loss.mean().detach())

        self.last_metrics.update(
            {
                "critic/value": float(imagined["values"].mean()),
                "critic/target": float(targets.mean()),
            }
        )
        return total

    @torch.no_grad()
    def _replay_value_targets(
        self,
        posterior: dict,
        rewards: torch.Tensor,
        is_terminal: torch.Tensor,
        imagined: dict,
    ) -> dict:
        """Build lambda-return targets over the replayed rewards.

        The imagination returns at the start states act as on-policy value
        annotations for the replayed states; lambda-returns are then accumulated
        over the actual replayed rewards.

        Indexing matches :meth:`_imagine`: ``rewards[t]`` and ``is_terminal[t]``
        describe leaving replayed state ``t``, so they pair with the value
        annotation of state ``t + 1``.
        """
        seq_len, batch = rewards.shape
        features = self.rssm.get_feat(self.rssm.detach_state(posterior))
        annotations = imagined["returns"][0].reshape(seq_len, batch)
        discounts = float(self.args.discount) * (1.0 - is_terminal)

        returns = lambda_return(
            rewards=rewards[:-1],
            values=annotations[1:],
            continues=discounts[:-1],
            bootstrap=annotations[-1],
            lambda_=float(self.args.td_lambda),
        )
        return {"features": features[:-1].detach(), "returns": returns.detach()}

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _optimize(
        self,
        loss: torch.Tensor,
        optimizer: LaProp,
        scaler: Any,
        parameters: list[torch.nn.Parameter],
        name: str,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = adaptive_grad_clip_(
            parameters,
            clip=float(self.args.agc_clip),
            eps=float(self.args.agc_eps),
        )
        self.last_metrics[f"grad_norm/{name}"] = float(grad_norm)
        scaler.step(optimizer)
        scaler.update()

    def _update_slow_critic(self) -> None:
        decay = float(self.args.critic_ema_decay)
        with torch.no_grad():
            for slow, fast in zip(
                self.slow_value_model.parameters(), self.value_model.parameters()
            ):
                slow.mul_(decay).add_(fast.detach(), alpha=1.0 - decay)

    @property
    def metrics(self) -> dict[str, float]:
        """Diagnostics from the last update, under the name the trainer reads."""

        return self.last_metrics

    def train_one_batch(self) -> list[float]:
        """Run one world model, actor, and critic update.

        Returns:
            ``[model_loss, actor_loss, value_loss]`` as plain floats, matching
            the contract expected by the shared Dreamer training loop.
        """
        if not self.data_buffer.can_sample:
            return [0.0, 0.0, 0.0]

        obs, actions, rewards, is_terminal, is_first = self.data_buffer.sample()
        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminal_t = torch.as_tensor(
            is_terminal, dtype=torch.float32, device=self.device
        )
        first_t = torch.as_tensor(is_first, dtype=torch.float32, device=self.device)

        device_type = self.device.type
        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            model_loss, posterior = self.world_model_loss(
                obs_t, actions_t, rewards_t, terminal_t, first_t
            )
        self._optimize(
            model_loss,
            self.world_model_opt,
            self.world_model_scaler,
            self.world_model_params,
            "world_model",
        )

        imagined = self._imagine(posterior)

        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            actor_loss = self.actor_loss(imagined)
        self._optimize(
            actor_loss,
            self.actor_opt,
            self.actor_scaler,
            list(self.actor.parameters()),
            "actor",
        )

        replay_targets = None
        if float(self.args.critic_replay_loss_scale) > 0.0:
            replay_targets = self._replay_value_targets(
                posterior, rewards_t, terminal_t, imagined
            )
        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            value_loss = self.value_loss(imagined, replay_targets)
        self._optimize(
            value_loss,
            self.value_opt,
            self.value_scaler,
            list(self.value_model.parameters()),
            "critic",
        )

        self._update_slow_critic()
        self._updates += 1

        return [
            float(model_loss.detach()),
            float(actor_loss.detach()),
            float(value_loss.detach()),
        ]

    # ------------------------------------------------------------------
    # Environment interaction
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        array = obs["image"] if isinstance(obs, dict) else obs
        tensor = torch.as_tensor(np.asarray(array).copy(), device=self.device)
        return tensor.unsqueeze(0)

    def act_with_world_model(
        self,
        obs: Any,
        prev_state: dict,
        prev_action: torch.Tensor,
        explore: bool = False,
        is_first: bool = False,
    ) -> tuple[dict, torch.Tensor]:
        """Update the model state with a new observation and choose an action."""
        obs_t = self.preprocess(self._obs_to_tensor(obs))
        embed = self.obs_encoder(obs_t)
        first = torch.full(
            (1, 1), float(is_first), device=self.device, dtype=torch.float32
        )
        posterior, _ = self.rssm.observe_step(prev_state, prev_action, embed, first)
        features = self.rssm.get_feat(posterior)
        action = self.actor(features, deterministic=not explore)
        if explore:
            action = self.actor.add_exploration(
                action, float(getattr(self.args, "action_noise", 0.0))
            )
        return posterior, action

    def _initial_action(self) -> torch.Tensor:
        return torch.zeros(1, self.action_size, device=self.device)

    def act_and_collect_data(self, env: Any, collect_steps: int) -> np.ndarray:
        """Interact with ``env`` under the current policy, filling the buffer."""
        obs = env.reset()
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = self._initial_action()
        is_first = True
        episode_rewards = [0.0]

        for step in range(collect_steps):
            with torch.no_grad():
                posterior, action = self.act_with_world_model(
                    obs, prev_state, prev_action, explore=True, is_first=is_first
                )
            action_np = action[0].cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)
            executed = (
                info["action"]
                if isinstance(info, dict) and "action" in info
                else action_np
            )
            terminal = _is_true_termination(info, done)
            self.data_buffer.add(obs, executed, reward, done, is_terminal=terminal)
            episode_rewards[-1] += reward

            if done:
                obs = env.reset()
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = self._initial_action()
                is_first = True
                if step != collect_steps - 1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs
                prev_state = posterior
                prev_action = torch.as_tensor(
                    np.asarray(executed, dtype=np.float32), device=self.device
                ).reshape(1, self.action_size)
                is_first = False

        return np.array(episode_rewards)

    def collect_random_episodes(self, env: Any, seed_steps: int) -> np.ndarray:
        """Fill the buffer with uniformly random actions before training starts."""
        obs = env.reset()
        episode_rewards = [0.0]

        for step in range(seed_steps):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            executed = (
                info["action"]
                if isinstance(info, dict) and "action" in info
                else action
            )
            terminal = _is_true_termination(info, done)
            self.data_buffer.add(obs, executed, reward, done, is_terminal=terminal)
            episode_rewards[-1] += reward
            if done:
                obs = env.reset()
                if step != seed_steps - 1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs

        return np.array(episode_rewards)

    def evaluate(
        self, env: Any, eval_episodes: int, render: bool = False
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Run greedy evaluation episodes and optionally capture frames."""
        episode_rewards = np.zeros(eval_episodes)
        video_images: list[list[Any]] = [[] for _ in range(eval_episodes)]
        latents: list[Any] | None = [] if render else None

        for episode in range(eval_episodes):
            obs = env.reset()
            done = False
            prev_state = self.rssm.init_state(1, self.device)
            prev_action = self._initial_action()
            is_first = True

            while not done:
                with torch.no_grad():
                    posterior, action = self.act_with_world_model(
                        obs, prev_state, prev_action, explore=False, is_first=is_first
                    )
                action_np = action[0].cpu().numpy()
                next_obs, reward, done, info = env.step(action_np)
                executed = (
                    info["action"]
                    if isinstance(info, dict) and "action" in info
                    else action_np
                )
                episode_rewards[episode] += reward

                if render:
                    image = obs["image"] if isinstance(obs, dict) else obs
                    if np.asarray(image).ndim == 3:
                        video_images[episode].append(
                            np.asarray(image).transpose(1, 2, 0).copy()
                        )
                    if latents is not None:
                        latents.append(self.rssm.get_feat(posterior)[0].cpu().numpy())

                prev_state = posterior
                prev_action = torch.as_tensor(
                    np.asarray(executed, dtype=np.float32), device=self.device
                ).reshape(1, self.action_size)
                is_first = False
                obs = next_obs

        latents_arr = (
            np.array(latents) if latents is not None and len(latents) > 0 else None
        )
        max_videos = int(getattr(self.args, "max_videos_to_save", 2))
        return (
            episode_rewards,
            np.array(video_images[:max_videos], dtype=object),
            latents_arr,
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save(self, save_path: str) -> None:
        """Write model, optimizer, and normalizer state to ``save_path``."""
        _save_config_next_to_checkpoint(self.args, save_path)
        torch.save(
            {
                "config": self.args.to_dict(),
                "obs_shape": tuple(self.obs_shape),
                "action_size": int(self.action_size),
                "rssm": self.rssm.state_dict(),
                "obs_encoder": self.obs_encoder.state_dict(),
                "obs_decoder": self.obs_decoder.state_dict(),
                "reward_model": self.reward_model.state_dict(),
                "continue_model": self.continue_model.state_dict(),
                "value_model": self.value_model.state_dict(),
                "slow_value_model": self.slow_value_model.state_dict(),
                "actor": self.actor.state_dict(),
                "world_model_optimizer": self.world_model_opt.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(),
                "value_optimizer": self.value_opt.state_dict(),
                "return_norm": self.return_norm.state_dict(),
                "updates": self._updates,
            },
            save_path,
        )

    def restore_checkpoint(
        self, ckpt_path: str | Path, map_location: Any = None
    ) -> None:
        """Load model, optimizer, and normalizer state from a checkpoint."""
        checkpoint = torch.load(
            ckpt_path, map_location=map_location or self.device, weights_only=True
        )
        self.rssm.load_state_dict(checkpoint["rssm"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.obs_decoder.load_state_dict(checkpoint["obs_decoder"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.continue_model.load_state_dict(checkpoint["continue_model"])
        self.value_model.load_state_dict(checkpoint["value_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        if checkpoint.get("slow_value_model") is not None:
            self.slow_value_model.load_state_dict(checkpoint["slow_value_model"])
        else:
            self.slow_value_model.load_state_dict(self.value_model.state_dict())

        self.world_model_opt.load_state_dict(checkpoint["world_model_optimizer"])
        self.actor_opt.load_state_dict(checkpoint["actor_optimizer"])
        self.value_opt.load_state_dict(checkpoint["value_optimizer"])
        if checkpoint.get("return_norm") is not None:
            self.return_norm.load_state_dict(checkpoint["return_norm"])
        self._updates = int(checkpoint.get("updates", 0))


def _is_true_termination(info: Any, done: bool | float) -> float:
    """Distinguish a genuine terminal state from a time-limit truncation.

    The continue predictor should learn that the episode really ended, not that
    a wall-clock budget expired; bootstrapping is still correct after a
    truncation. Simulusion's ``TimeLimit`` wrapper sets ``info["time_limit"]``.
    """
    if not done:
        return 0.0
    if isinstance(info, dict):
        if info.get("time_limit") or info.get("TimeLimit.truncated"):
            return 0.0
        if "discount" in info:
            # dm_control style: discount == 1 at a truncation boundary.
            try:
                return 0.0 if float(info["discount"]) == 1.0 else 1.0
            except (TypeError, ValueError):
                pass
    return 1.0


def _resolve_actor_dist(env: Any) -> str:
    """Choose the policy distribution from an environment's action space."""
    action_space = env.action_space
    if hasattr(action_space, "n") and not hasattr(action_space, "low"):
        return "onehot"
    # The OneHotAction wrapper exposes a Box space but keeps a discrete
    # underlying space, which it advertises through `_env.action_space`.
    inner = getattr(env, "_env", None)
    if inner is not None and hasattr(getattr(inner, "action_space", None), "n"):
        return "onehot"
    return "normal"


class DreamerV3Agent(DreamerAgent):
    """High-level DreamerV3 agent: builds environments, trains, and evaluates.

    Usage::

        import simulusion

        agent = simulusion.create_model(
            "dreamer-v3", env="Pendulum-v1", env_backend="gym", total_steps=5_000
        )
        agent.train()
    """

    core_cls = DreamerV3  # type: ignore[assignment]
    _config_cls = DreamerV3Config

    def _build_core(
        self, obs_shape: Any, action_size: int, device: torch.device
    ) -> DreamerV3:
        args = cast(DreamerV3Config, self.args)
        if str(getattr(args, "actor_dist", "auto")) == "auto":
            args.actor_dist = _resolve_actor_dist(self.train_env)
        args.resolve()
        return DreamerV3(args, obs_shape, action_size, device, args.restore)
