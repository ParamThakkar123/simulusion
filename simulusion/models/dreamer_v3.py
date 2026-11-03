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
from collections.abc import Mapping
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
    DreamerV3Head,
    DreamerV3MultiDecoder,
    DreamerV3MultiEncoder,
    preprocess_image,
    split_obs_space,
)

logger = get_package_logger(__name__)

__all__ = [
    "DreamerV3",
    "DreamerV3Agent",
    "coerce_dreamer_v3_config",
    "select_obs_space",
]

# Keys tried, in order, when a single-input model is given an observation dict
# that does not use the key it was built with.
_SINGLE_INPUT_ALIASES = ("image", "obs", "state")


def select_obs_space(
    observation_space: Any, obs_keys: Any = None
) -> dict[str, tuple[int, ...]]:
    """Choose which observations DreamerV3 encodes, and their shapes.

    Args:
        observation_space: A Gym ``Dict`` space (or a single ``Box``, which is
            treated as ``"image"`` if rank 3 and ``"state"`` otherwise).
        obs_keys: Keys to encode, as a list or a comma-separated string. ``None``
            picks ``"image"`` if the environment has one, else ``"state"`` (or
            ``"obs"``), plus ``"instr"`` when present. That matches the paper's
            benchmarks: pixels for Visual Control, the state vector for Proprio
            Control, and image plus instruction for DMLab language levels.

    Returns:
        Observation key -> shape, in encoding order.

    Raises:
        ValueError: If a requested key is missing, or nothing usable is found.
    """
    space: Any = observation_space
    spaces: Any = getattr(space, "spaces", None)
    if spaces is None and isinstance(space, Mapping):
        spaces = space  # a plain dict of spaces
    if spaces is None:
        shape = tuple(int(d) for d in getattr(space, "shape"))
        spaces = {("image" if len(shape) == 3 else "state"): space}
    available = {
        key: tuple(int(d) for d in space.shape) for key, space in spaces.items()
    }

    if isinstance(obs_keys, str):
        obs_keys = [key.strip() for key in obs_keys.split(",") if key.strip()]
    if obs_keys:
        missing = [key for key in obs_keys if key not in available]
        if missing:
            raise ValueError(
                f"obs_keys {missing} are not observations of this environment; "
                f"it has {sorted(available)}."
            )
        keys = list(obs_keys)
    else:
        keys = [
            next(
                (key for key in ("image", "state", "obs") if key in available),
                "",
            )
        ]
        if not keys[0]:
            raise ValueError(
                "Could not pick an observation to encode from "
                f"{sorted(available)}; set obs_keys."
            )
        if "instr" in available and "instr" not in keys:
            keys.append("instr")
    selected = {key: available[key] for key in keys}
    split_obs_space(selected)  # validates ranks and image sizes
    return selected


def _as_obs_space(obs_shape: Any) -> dict[str, tuple[int, ...]]:
    """Normalize a single shape or a key -> shape mapping."""
    if isinstance(obs_shape, Mapping):
        return {str(k): tuple(int(d) for d in v) for k, v in obs_shape.items()}
    shape = tuple(int(dim) for dim in obs_shape)
    return {("image" if len(shape) == 3 else "state"): shape}


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
        obs_shape: Observation shape, ``(C, H, W)`` or ``(D,)``, or a mapping
            from observation key to shape to encode several inputs (see
            :func:`select_obs_space`).
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
        #: Encoded observations: key -> shape.
        self.obs_space = _as_obs_space(obs_shape)
        self._single_input = not isinstance(obs_shape, Mapping)
        self.image_keys, self.vector_keys = split_obs_space(self.obs_space)
        # The single input's shape, kept for code written against one input.
        self.obs_shape = (
            next(iter(self.obs_space.values())) if len(self.obs_space) == 1 else ()
        )
        self.action_size = int(action_size)
        self.device = torch.device(device)
        self.restore = bool(restore)
        self.restore_path = getattr(args, "checkpoint_path", "")
        self.is_image_obs = bool(self.image_keys)
        self.discrete_actions = str(getattr(args, "actor_dist", "auto")) == "onehot"
        self._metrics: dict[str, Any] = {}
        # Per-environment collection state, so successive collection calls
        # continue the same episode instead of resetting the environment.
        self._carry: dict[int, dict[str, Any]] = {}

        obs_dtype = np.uint8
        store_latents = bool(getattr(args, "replay_latents", True))
        deter_size = args.resolved_recurrent_units if store_latents else 0
        latent_dim = int(args.latent_dim) if store_latents else 0
        bytes_per_record = DreamerV3ReplayBuffer.bytes_per_record(
            self.obs_space, obs_dtype, self.action_size, deter_size, latent_dim
        )
        max_buffer_size = int(
            (get_available_memory() * 0.8) // max(1, bytes_per_record)
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
            obs_shape=self.obs_space,
            action_size=self.action_size,
            seq_len=int(args.train_seq_len),
            batch_size=int(args.batch_size),
            obs_dtype=obs_dtype,
            online_fraction=float(getattr(args, "online_fraction", 1.0)),
            deter_size=deter_size,
            latent_dim=latent_dim,
            latent_classes=args.resolved_latent_classes if store_latents else 0,
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
        self._observe_graph: Any = None
        self._imagine_graph: Any = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_model(self, restore: bool = False) -> None:
        args = self.args
        hidden = args.resolved_hidden_size
        deter = args.resolved_recurrent_units
        classes = args.resolved_latent_classes
        depth = args.resolved_cnn_depth

        self.obs_encoder = DreamerV3MultiEncoder(
            self.obs_space,
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
        self.obs_decoder = DreamerV3MultiDecoder(
            feature_size,
            encoder=self.obs_encoder,
            cnn_depth=depth,
            hidden_size=hidden,
            mlp_layers=int(args.mlp_layers),
            activation=str(args.activation),
        ).to(self.device)

        head_kwargs: dict[str, Any] = dict(
            units=hidden,
            num_bins=int(args.num_buckets),
            symlog_range=float(args.symlog_range),
            activation=str(args.activation),
        )
        # The reward and continue predictors are 1-layer MLPs, the critic a
        # 3-layer MLP. Reward and critic outputs start at zero so the agent does
        # not hallucinate rewards and values at initialization.
        self.reward_model = DreamerV3Head(
            feature_size,
            layers=int(getattr(args, "reward_layers", 1)),
            dist="symexp_twohot",
            zero_init_output=True,
            **head_kwargs,
        ).to(self.device)
        self.continue_model = DreamerV3Head(
            feature_size,
            layers=int(getattr(args, "cont_layers", 1)),
            dist="binary",
            zero_init_output=False,
            **head_kwargs,
        ).to(self.device)
        self.value_model = DreamerV3Head(
            feature_size,
            layers=int(args.mlp_layers),
            dist="symexp_twohot",
            zero_init_output=True,
            **head_kwargs,
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
        obs_shape: Any = None,
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
                obs_shape = select_obs_space(
                    env.observation_space, getattr(args, "obs_keys", None)
                )
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

        obs_shape: Any = None
        if isinstance(checkpoint, dict):
            if checkpoint.get("obs_space") is not None:
                obs_shape = {
                    str(k): tuple(v) for k, v in checkpoint["obs_space"].items()
                }
            elif checkpoint.get("obs_shape") is not None:
                obs_shape = tuple(checkpoint["obs_shape"])
        action_size = (
            checkpoint.get("action_size") if isinstance(checkpoint, dict) else None
        )
        model = cls.from_config(
            args,
            obs_shape=obs_shape,
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

    def _as_obs_dict(self, obs: Any) -> dict[str, Any]:
        """Observations keyed like :attr:`obs_space`.

        A single-input model also accepts a bare array, or a dict that stores
        its input under one of the usual names (``image``, ``obs``, ``state``).
        """
        if not isinstance(obs, Mapping):
            if len(self.obs_space) != 1:
                raise TypeError(
                    f"This model encodes {sorted(self.obs_space)}; pass a dict."
                )
            return {next(iter(self.obs_space)): obs}
        selected: dict[str, Any] = {}
        for key, shape in self.obs_space.items():
            if key in obs:
                selected[key] = obs[key]
                continue
            alias = None
            if self._single_input:
                alias = next(
                    (
                        name
                        for name in _SINGLE_INPUT_ALIASES
                        if name in obs and np.ndim(obs[name]) >= len(shape)
                    ),
                    None,
                )
            if alias is None:
                raise KeyError(f"Observation has no {key!r}; it has {sorted(obs)}.")
            selected[key] = obs[alias]
        return selected

    def preprocess(self, obs: Any) -> dict[str, torch.Tensor]:
        """Map raw observations into the model's input space.

        Images go to ``[0, 1]``; vectors are cast to float32 (the encoder applies
        symlog). Returns a dict keyed like :attr:`obs_space`.
        """
        processed: dict[str, torch.Tensor] = {}
        for key, value in self._as_obs_dict(obs).items():
            tensor = torch.as_tensor(value, device=self.device)
            if key in self.image_keys:
                processed[key] = preprocess_image(tensor)
            else:
                processed[key] = tensor.to(torch.float32)
        return processed

    # ------------------------------------------------------------------
    # World model
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        deter: torch.Tensor | None = None,
        stoch: torch.Tensor | None = None,
    ) -> dict:
        """Model state to start a replayed sequence from.

        Zeros, unless the stored state of the record before the sequence is
        given (``deter``: ``(B, deter_size)``, ``stoch``: ``(B, latent_dim,
        latent_classes)`` one-hot). Rows without a stored state are all zeros,
        which is the same as the zero initial state.
        """
        state = self.rssm.init_state(batch_size, self.device)
        if deter is not None and stoch is not None:
            state["deter"] = deter.to(self.device, torch.float32)
            state["stoch"] = stoch.to(self.device, torch.float32)
        return state

    @assert_finite
    def world_model_loss(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        rewards: torch.Tensor,
        is_terminal: torch.Tensor,
        is_first: torch.Tensor,
        init_state: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """World model loss (Eq. 2-3) over a batch of replayed sequences.

        The inputs follow the replay layout: entry ``t`` is observation ``x_t``
        together with the action ``a_{t-1}`` and reward ``r_t`` that led to it.
        So the sequence model advances on ``a_{t-1}``, and the reward and
        continue predictors at ``s_t = (h_t, z_t)`` are trained on ``r_t`` and
        ``c_t = 1 - is_terminal_t`` -- quantities ``s_t`` can actually predict,
        since ``h_t`` has seen ``a_{t-1}``.

        Args:
            obs: ``(T, B, *shape)`` raw observations ``x_t``: a dict keyed like
                :attr:`obs_space`, or a single array for a single-input model.
            prev_actions: ``(T, B, action_size)`` actions ``a_{t-1}``.
            rewards: ``(T, B)`` rewards ``r_t``.
            is_terminal: ``(T, B)`` whether ``x_t`` is a terminal state.
            is_first: ``(T, B)`` whether ``x_t`` starts an episode.
            init_state: Model state before ``x_0``; zeros when omitted.

        Returns:
            ``(loss, posterior)`` where ``posterior`` holds the ``(T, B, ...)``
            model states used for actor-critic learning.
        """
        args = self.args
        processed = self.preprocess(obs)
        embed = self.obs_encoder(processed)

        if init_state is None:
            init_state = self.rssm.init_state(rewards.shape[1], self.device)
        posterior, prior = self._observe_rollout(
            embed, prev_actions, is_first.unsqueeze(-1), init_state
        )
        features = self.rssm.get_feat(posterior)

        reconstruction = self.obs_decoder(features)
        recon_terms = self.obs_decoder.loss_terms(reconstruction, processed)
        recon_loss = sum(recon_terms.values(), torch.zeros_like(rewards))

        reward_loss = -self.reward_model(features).log_prob(rewards)
        continue_loss = -self.continue_model(features).log_prob(1.0 - is_terminal)

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

        self._metrics.update(
            {
                "wm/recon_loss": recon_loss.mean().detach(),
                "wm/reward_loss": reward_loss.mean().detach(),
                "wm/continue_loss": continue_loss.mean().detach(),
                "wm/dyn_loss": dyn_loss.mean().detach(),
                "wm/rep_loss": rep_loss.mean().detach(),
            }
        )
        if len(recon_terms) > 1:
            for key, term in recon_terms.items():
                self._metrics[f"wm/recon_loss/{key}"] = term.mean().detach()
        return loss, posterior

    # ------------------------------------------------------------------
    # Imagination and actor-critic
    # ------------------------------------------------------------------

    def _graphs_enabled(self) -> bool:
        return (
            bool(getattr(self.args, "cuda_graphs", True))
            and self.device.type == "cuda"
            and not self.use_amp
        )

    def _observe_rollout(
        self,
        embed: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        init_state: dict,
    ) -> tuple[dict, dict]:
        """``rssm.observe_rollout``, replayed as a CUDA graph when possible."""
        if self._graphs_enabled() and torch.is_grad_enabled() and embed.requires_grad:
            graph = self._observe_graph
            if graph is None or not graph.matches(embed, actions, is_first, init_state):
                from simulusion.models.dreamer_v3_graphs import GraphedObserveRollout

                graph = self._capture(
                    "observe_graph",
                    lambda: GraphedObserveRollout(
                        self.rssm, embed, actions, is_first, init_state
                    ),
                )
            if graph is not None:
                return graph(embed, actions, is_first, init_state)
        return self.rssm.observe_rollout(embed, actions, is_first, init_state)

    def _capture(self, name: str, build: Any) -> Any:
        """Build a CUDA graph, or turn graphs off (with a warning) if that fails."""
        try:
            graph = build()
        except RuntimeError as exc:
            logger.warning("CUDA graph capture failed; running eagerly: %s", exc)
            self.args.cuda_graphs = False
            torch.cuda.synchronize()
            return None
        setattr(self, f"_{name}", graph)
        return graph

    def _imagine_rollout(self, start: dict, horizon: int) -> tuple[dict, torch.Tensor]:
        """``rssm.imagine_rollout`` under the actor, as a CUDA graph when possible."""
        policy = self.actor.forward
        if self._graphs_enabled() and not torch.is_grad_enabled():
            graph = self._imagine_graph
            if graph is None or not graph.matches(start, horizon):
                from simulusion.models.dreamer_v3_graphs import GraphedImagination

                graph = self._capture(
                    "imagine_graph",
                    lambda: GraphedImagination(self.rssm, policy, start, horizon),
                )
            if graph is not None:
                return graph(start)
        return self.rssm.imagine_rollout(policy, start, horizon)

    @torch.no_grad()
    def _imagine(
        self, posterior: dict, is_terminal: torch.Tensor | None = None
    ) -> dict:
        """Roll the world model forward under the current policy.

        Everything here is computed without gradients: DreamerV3 uses the
        Reinforce estimator, so the actor learns through ``log_prob`` of detached
        actions rather than by backpropagating through the dynamics.

        Imagination starts from every replayed model state ``s_0`` and produces
        ``s_0 .. s_H`` with actions ``a_0 .. a_{H-1}``. As in the world model,
        the reward and continuation for taking ``a_t`` in ``s_t`` are predicted
        at the state it leads to, ``s_{t+1}``, so the lambda-return (Eq. 5) is

            R_t = r_{t+1} + gamma c_{t+1} ((1 - lambda) v_{t+1} + lambda R_{t+1}),
            R_H = v_H.

        Args:
            posterior: ``(T, B, ...)`` replayed model states.
            is_terminal: ``(T, B)`` terminal flags of those states. A terminal
                start state has nothing to act on, so it gets zero weight.

        Returns:
            A dict with the imagined features, actions, returns, values, the
            predicted rewards, and the weights used to average the actor and
            critic objectives.
        """
        args = self.args
        horizon = int(args.imagine_horizon)
        discount = float(args.discount)

        start = self.rssm.detach_state(self.rssm.seq_to_batch(posterior))
        with FreezeParameters(self.world_model_modules + self.actor_modules):
            states, actions = self._imagine_rollout(start, horizon)
            features = self.rssm.get_feat(states)
            rewards = self.reward_model(features[1:]).mean
            continues = self.continue_model(features[1:]).probs
            values = self.value_model(features).mean

        returns = lambda_return(
            rewards=rewards,
            values=values[1:],
            continues=discount * continues,
            bootstrap=values[-1],
            lambda_=float(args.td_lambda),
        )

        # Weight of acting from s_t: the discounted probability of reaching it,
        # gamma^t * c_0 * ... * c_t, where c_0 says whether the replayed start
        # state is alive and c_1.. are predicted.
        if is_terminal is None:
            start_continue = torch.ones_like(continues[0])
        else:
            start_continue = 1.0 - is_terminal.reshape(-1).to(continues.dtype)
        alive = torch.cat([start_continue[None], continues[: horizon - 1]], dim=0)
        weights = torch.cumprod(discount * alive, dim=0) / discount

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
        """Reinforce objective with normalized returns and an entropy bonus (Eq. 6)."""
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

        self._metrics.update(
            {
                "actor/entropy": entropy.mean().detach(),
                "actor/advantage": advantage.mean().detach(),
                "actor/return_scale": self.return_norm.scale_tensor(),
                "actor/imag_reward": imagined["rewards"].mean().detach(),
            }
        )
        return loss

    def _critic_terms(self, features: torch.Tensor, targets: torch.Tensor) -> Any:
        """Two-hot loss on ``targets`` plus the slow-critic regularizer."""
        dist = self.value_model(features)
        loss = -dist.log_prob(targets)
        scale = float(self.args.critic_ema_regularizer)
        if scale > 0.0:
            with torch.no_grad():
                slow_dist = self.slow_value_model(features)
            loss = loss + scale * dist.cross_entropy_to(slow_dist)
        return loss

    @assert_finite
    def value_loss(
        self,
        imagined: dict,
        replay: dict | None = None,
    ) -> torch.Tensor:
        """Critic loss on imagined returns, plus the replay loss.

        Both terms regress the critic's two-hot distribution onto lambda-returns
        and regularize it towards an exponential moving average of its own
        weights. The imagination term has scale ``critic_loss_scale`` (1), the
        replay term ``critic_replay_loss_scale`` (0.3).
        """
        args = self.args
        features = imagined["features"][:-1].detach()
        targets = imagined["returns"].detach()
        weights = imagined["weights"]

        loss = self._critic_terms(features, targets)
        total = float(args.critic_loss_scale) * (weights * loss).mean()

        if replay is not None and float(args.critic_replay_loss_scale) > 0.0:
            replay_loss = self._critic_terms(replay["features"], replay["returns"])
            replay_loss = (replay["weights"] * replay_loss).mean()
            total = total + float(args.critic_replay_loss_scale) * replay_loss
            self._metrics["critic/replay_loss"] = replay_loss.detach()

        self._metrics.update(
            {
                "critic/value": imagined["values"].mean().detach(),
                "critic/target": targets.mean().detach(),
            }
        )
        return total

    @torch.no_grad()
    def _replay_value_targets(
        self,
        posterior: dict,
        rewards: torch.Tensor,
        is_terminal: torch.Tensor,
        is_first: torch.Tensor,
        is_last: torch.Tensor,
        imagined: dict,
    ) -> dict:
        """Lambda-return targets over the replayed rewards.

        The imagination returns at the start states, ``A_t = R^lambda_0(s_t)``,
        serve as on-policy value annotations for the replayed states, and the
        returns are accumulated over the actual replayed rewards:

            R_t = r_{t+1} + gamma c_{t+1} ((1 - lambda) A_{t+1} + lambda R_{t+1}).

        Where an episode ends at ``t`` (``is_last``, or the next record starts a
        new episode) there is no next transition and ``R_t = A_t``. Terminal
        states are excluded from the loss. Only ``t < T - 1`` is trained, since
        the last step has no replayed transition to learn from.
        """
        seq_len, batch = rewards.shape
        features = self.rssm.get_feat(self.rssm.detach_state(posterior))
        annotations = imagined["returns"][0].reshape(seq_len, batch)
        gamma = float(self.args.discount)
        lam = float(self.args.td_lambda)
        alive = 1.0 - is_terminal

        returns = [annotations[-1]]
        for t in range(seq_len - 2, -1, -1):
            boundary = torch.maximum(is_last[t], is_first[t + 1])
            step = rewards[t + 1] + gamma * alive[t + 1] * (
                (1.0 - lam) * annotations[t + 1] + lam * returns[-1]
            )
            returns.append(boundary * annotations[t] + (1.0 - boundary) * step)
        stacked = torch.flip(torch.stack(returns, dim=0), dims=(0,))
        return {
            "features": features[:-1].detach(),
            "returns": stacked[:-1].detach(),
            "weights": alive[:-1],
        }

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
        self._metrics[f"grad_norm/{name}"] = grad_norm.detach()
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
    def last_metrics(self) -> dict[str, float]:
        """Diagnostics from the last update, as plain floats.

        Updates record them as GPU tensors; they are copied to the host here in
        a single transfer instead of one synchronizing ``float()`` per value.
        """
        pending = [k for k, v in self._metrics.items() if torch.is_tensor(v)]
        if pending:
            device = self._metrics[pending[0]].device
            values = torch.stack(
                [
                    self._metrics[k].detach().float().reshape(()).to(device)
                    for k in pending
                ]
            ).tolist()
            self._metrics.update(zip(pending, values))
        return self._metrics

    @last_metrics.setter
    def last_metrics(self, value: dict[str, Any]) -> None:
        self._metrics = dict(value)

    @property
    def metrics(self) -> dict[str, float]:
        """Diagnostics from the last update, under the name the trainer reads."""

        return self.last_metrics

    def _batch_to_tensors(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        def as_float(name: str) -> torch.Tensor:
            return torch.as_tensor(batch[name], dtype=torch.float32, device=self.device)

        return {
            "obs": self._as_obs_dict(batch["observations"]),
            "prev_actions": as_float("prev_actions"),
            "rewards": as_float("rewards"),
            "is_first": as_float("is_first"),
            "is_last": as_float("is_last"),
            "is_terminal": as_float("is_terminal"),
            "init_deter": as_float("init_deter"),
            "init_stoch": as_float("init_stoch"),
        }

    def train_one_batch(self) -> list[float]:
        """Run one world model, actor, and critic update.

        Returns:
            ``[model_loss, actor_loss, value_loss]`` as plain floats, matching
            the contract expected by the shared Dreamer training loop.
        """
        if not self.data_buffer.can_sample:
            return [0.0, 0.0, 0.0]

        batch = self.data_buffer.sample()
        data = self._batch_to_tensors(batch)
        batch_size = data["rewards"].shape[1]
        init_state = None
        if self.data_buffer.stores_latents:
            init_state = self.initial_state(
                batch_size, data["init_deter"], data["init_stoch"]
            )

        device_type = self.device.type
        with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
            model_loss, posterior = self.world_model_loss(
                data["obs"],
                data["prev_actions"],
                data["rewards"],
                data["is_terminal"],
                data["is_first"],
                init_state,
            )
        self._optimize(
            model_loss,
            self.world_model_opt,
            self.world_model_scaler,
            self.world_model_params,
            "world_model",
        )
        # Replay keeps the freshest model states, so the next time one of these
        # sequences' successors is sampled it starts from an up-to-date state.
        self.data_buffer.update_latents(
            batch["indices"], posterior["deter"], posterior["stoch"]
        )

        imagined = self._imagine(posterior, data["is_terminal"])

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
                posterior,
                data["rewards"],
                data["is_terminal"],
                data["is_first"],
                data["is_last"],
                imagined,
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

        self._metrics.update(
            {
                "model_loss": model_loss.detach(),
                "actor_loss": actor_loss.detach(),
                "value_loss": value_loss.detach(),
            }
        )
        metrics = self.last_metrics  # One host transfer for every value.
        return [metrics["model_loss"], metrics["actor_loss"], metrics["value_loss"]]

    # ------------------------------------------------------------------
    # Environment interaction
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: Any) -> dict[str, torch.Tensor]:
        """One environment observation as a batch of one, per encoded input."""
        return {
            key: torch.as_tensor(np.asarray(value).copy(), device=self.device)[None]
            for key, value in self._as_obs_dict(obs).items()
        }

    def observe(
        self,
        obs: Any,
        prev_state: dict,
        prev_action: torch.Tensor,
        is_first: bool = False,
    ) -> dict:
        """Update the model state with a new observation ``x_t``."""
        obs_t = self.preprocess(self._obs_to_tensor(obs))
        embed = self.obs_encoder(obs_t)
        first = torch.full(
            (1, 1), float(is_first), device=self.device, dtype=torch.float32
        )
        posterior, _ = self.rssm.observe_step(prev_state, prev_action, embed, first)
        return posterior

    def act_with_world_model(
        self,
        obs: Any,
        prev_state: dict,
        prev_action: torch.Tensor,
        explore: bool = False,
        is_first: bool = False,
    ) -> tuple[dict, torch.Tensor]:
        """Update the model state with a new observation and choose an action.

        While collecting, actions are sampled from the actor (``explore``); the
        paper uses no lookahead planning and no injected noise.
        """
        posterior = self.observe(obs, prev_state, prev_action, is_first)
        features = self.rssm.get_feat(posterior)
        action = self.actor(features, deterministic=not explore)
        if explore:
            action = self.actor.add_exploration(
                action, float(getattr(self.args, "action_noise", 0.0))
            )
        return posterior, action

    def _initial_action(self) -> torch.Tensor:
        return torch.zeros(1, self.action_size, device=self.device)

    @staticmethod
    def _latent(state: dict) -> dict:
        return {"deter": state["deter"][0], "stoch": state["stoch"][0]}

    def _start_episode(self, env: Any) -> dict[str, Any]:
        """Reset ``env`` and close whatever episode the buffer had open."""
        self.data_buffer.end_episode()
        return {
            "obs": env.reset(),
            "prev_action": np.zeros(self.action_size, dtype=np.float32),
            "reward": 0.0,
            "is_first": True,
            "state": None,
            "episode_return": 0.0,
        }

    def _resume(self, env: Any, need_state: bool) -> dict[str, Any]:
        """Continue the episode left open in ``env`` by the last collection call.

        An episode collected with random actions has no model state to continue
        from, so policy collection ends it (as a truncation) and resets.
        """
        carry = self._carry.get(id(env))
        if carry is None:
            return self._start_episode(env)
        if carry["is_first"]:
            return carry  # Freshly reset; nothing recorded for it yet.
        if not self.data_buffer.episode_open:
            # Something closed the episode since (evaluation, a replay restore).
            return self._start_episode(env)
        if need_state and carry["state"] is None:
            return self._start_episode(env)
        return carry

    def _collect(self, env: Any, steps: int, policy: bool) -> np.ndarray:
        """Step ``env`` ``steps`` times, writing one replay record per observation.

        Every observation is recorded with the action and reward that led to it
        and, when acting with the policy, the model state computed from it. The
        final observation of an episode gets its own record (``is_last``, and
        ``is_terminal`` if the environment terminated) so the continue
        predictor learns what terminal states look like.

        Returns:
            Returns of the episodes finished during this call, or the return so
            far of the open episode if none finished.
        """
        carry = self._resume(env, need_state=policy)
        finished: list[float] = []

        for _ in range(steps):
            obs = carry["obs"]
            latent = None
            if policy:
                prev_state = carry["state"] or self.rssm.init_state(1, self.device)
                prev_action = torch.as_tensor(
                    carry["prev_action"], dtype=torch.float32, device=self.device
                ).reshape(1, self.action_size)
                with torch.no_grad():
                    state, action_t = self.act_with_world_model(
                        obs, prev_state, prev_action, True, carry["is_first"]
                    )
                action = action_t[0].cpu().numpy()
                latent = self._latent(state)
            else:
                state = None
                action = env.action_space.sample()

            self.data_buffer.add(
                obs,
                carry["prev_action"],
                carry["reward"],
                is_first=carry["is_first"],
                latent=latent,
            )

            next_obs, reward, done, info = env.step(action)
            executed = np.asarray(
                info["action"]
                if isinstance(info, dict) and "action" in info
                else action,
                dtype=np.float32,
            ).reshape(self.action_size)
            episode_return = carry["episode_return"] + float(reward)

            if done:
                final_latent = None
                if policy and state is not None:
                    with torch.no_grad():
                        final_state = self.observe(
                            next_obs,
                            state,
                            torch.as_tensor(executed, device=self.device).reshape(
                                1, self.action_size
                            ),
                        )
                    final_latent = self._latent(final_state)
                self.data_buffer.add(
                    next_obs,
                    executed,
                    float(reward),
                    is_first=False,
                    is_last=True,
                    is_terminal=bool(_is_true_termination(info, done)),
                    latent=final_latent,
                )
                finished.append(episode_return)
                carry = self._start_episode(env)
            else:
                carry = {
                    "obs": next_obs,
                    "prev_action": executed,
                    "reward": float(reward),
                    "is_first": False,
                    "state": state,
                    "episode_return": episode_return,
                }

        self._carry[id(env)] = carry
        if not finished:
            finished.append(carry["episode_return"])
        return np.asarray(finished, dtype=np.float64)

    def act_and_collect_data(self, env: Any, collect_steps: int) -> np.ndarray:
        """Interact with ``env`` under the current policy, filling the buffer.

        Collection continues the episode the previous call left open, so the
        environment is only reset when an episode actually ends.
        """
        return self._collect(env, int(collect_steps), policy=True)

    def collect_random_episodes(self, env: Any, seed_steps: int) -> np.ndarray:
        """Fill the buffer with uniformly random actions before training starts."""
        return self._collect(env, int(seed_steps), policy=False)

    def evaluate(
        self, env: Any, eval_episodes: int, render: bool = False
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Run greedy evaluation episodes and optionally capture frames."""
        if id(env) in self._carry:
            # Evaluation resets the environment, so a collection episode open
            # in it cannot be continued afterwards.
            self._carry.pop(id(env))
            self.data_buffer.end_episode()
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
                    frame = _video_frame(env, obs)
                    if frame is not None:
                        video_images[episode].append(frame)
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
                "obs_space": {
                    key: list(shape) for key, shape in self.obs_space.items()
                },
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


def _video_frame(env: Any, obs: Any) -> np.ndarray | None:
    """An HWC frame for evaluation videos: the image observation, else a render."""
    image = obs.get("image") if isinstance(obs, Mapping) else obs
    if image is not None and np.ndim(image) == 3:
        return np.asarray(image).transpose(1, 2, 0).copy()
    render = getattr(env, "render", None)
    if render is None:
        return None
    try:
        frame = np.asarray(render())
    except Exception:
        return None
    return frame.copy() if frame.ndim == 3 else None


def _is_true_termination(info: Any, done: bool | float) -> float:
    """Distinguish a genuine terminal state from a time-limit truncation.

    The continue predictor should learn that the episode really ended, not that
    a wall-clock budget expired; bootstrapping is still correct after a
    truncation. Simulusion's ``TimeLimit`` wrapper sets ``info["time_limit"]``.
    """
    if not done:
        return 0.0
    if isinstance(info, dict):
        # The environment contract (`finalize_step_info`) reports both flags.
        if "terminated" in info:
            return 1.0 if bool(info["terminated"]) else 0.0
        if info.get("truncated"):
            return 0.0
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
    # Backends such as DMLab, BSuite and Minecraft expose a discrete choice as
    # a one-hot Box (marked `one_hot`); wrappers may replace that space with a
    # plain Box, so look through the whole wrapper chain.
    layer: Any = env
    for _ in range(32):
        if layer is None:
            break
        if getattr(getattr(layer, "action_space", None), "one_hot", False):
            return "onehot"
        layer = layer.__dict__.get("_env") if hasattr(layer, "__dict__") else None
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

    def _observation_shape(self, env: Any) -> Any:
        """Every observation DreamerV3 encodes (``obs_keys``), not only the image."""
        return select_obs_space(
            env.observation_space, getattr(self.args, "obs_keys", None)
        )

    def _build_core(
        self, obs_shape: Any, action_size: int, device: torch.device
    ) -> DreamerV3:
        args = cast(DreamerV3Config, self.args)
        if str(getattr(args, "actor_dist", "auto")) == "auto":
            args.actor_dist = _resolve_actor_dist(self.train_env)
        args.resolve()
        return DreamerV3(args, obs_shape, action_size, device, args.restore)
