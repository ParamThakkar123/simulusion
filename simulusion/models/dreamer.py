import os
import random
import sys
import time
import importlib.util
import numpy as np
import ctypes

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

from collections import OrderedDict
from pathlib import Path
from typing import Any

from simulusion.memory.dreamer_memory import ReplayBuffer
from simulusion.models.dreamer_rssm import RSSM
from simulusion.vision.dreamer_decoder import ConvDecoder, DenseDecoder, ActionDecoder
from simulusion.vision.dreamer_encoder import ConvEncoder
from simulusion.training.precision import GradientUpdate, MixedPrecision
from simulusion.utils.memory_utils import enable_performance_defaults
from simulusion.utils.dreamer_utils import (
    Logger,
    FreezeParameters,
    compute_return,
)
from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.export import ExportableAgentMixin
from simulusion.utils.logging_utils import (
    assert_finite,
    get_package_logger,
    setup_logging,
)

logger = get_package_logger(__name__)

# Default MuJoCo to headless EGL rendering on Linux only. Windows raises a
# RuntimeError on import with 'egl', and macOS has no EGL at all, so both keep
# MuJoCo's own default. An existing environment value is always respected.
if sys.platform.startswith("linux") and os.environ.get("MUJOCO_GL") is None:
    os.environ["MUJOCO_GL"] = "egl"


def get_available_memory() -> int:
    """Get available physical memory in bytes."""
    # ``sys.platform`` rather than ``os.name``: mypy narrows on the former, so
    # the Windows-only ctypes below is skipped when checking on Linux. With
    # ``os.name`` mypy analyses this branch everywhere and ``ctypes.WinDLL``
    # then needs a ``type: ignore`` that is itself unused on Windows.
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            raise OSError("Failed to get memory status")
        return memory_status.ullAvailPhys
    else:  # Linux/Mac
        try:
            import psutil

            return psutil.virtual_memory().available
        except ImportError:
            # Fallback: read /proc/meminfo on Linux
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            avail_kb = int(line.split()[1])
                            return avail_kb * 1024  # Convert KB to bytes
            except (FileNotFoundError, ValueError, IndexError):
                pass
            # Ultimate fallback: assume 8GB available
            return 8 * 1024 * 1024 * 1024


def _resolve_image_size(args: Any) -> tuple:
    size = getattr(args, "image_size", (64, 64))
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Invalid image_size={size}. Expected int or (H, W).")


def make_env(args: Any) -> Any:
    """Construct a Dreamer-compatible environment from `DreamerConfig` options.

    Supports DMC, DMLab, Gym/Gymnasium, MuJoCo, Gymnasium Robotics, Procgen,
    Brax, BSuite, and Unity ML-Agents backends
    and applies the standard wrapper stack: action repeat, action normalization,
    and time limit.
    """
    import simulusion.envs.wrappers as env_wrapper

    size = _resolve_image_size(args)
    backend = str(getattr(args, "env_backend", "dmc")).lower()

    env_instance = getattr(args, "env_instance", None)
    if env_instance is not None:
        from simulusion.envs.gym_env import GymImageEnv

        env: Any = GymImageEnv(
            env_instance,
            seed=args.seed,
            size=size,
            render_mode=getattr(args, "gym_render_mode", "rgb_array"),
        )
    elif backend == "dmc":
        from simulusion.envs.dmc import DeepMindControlEnv

        env = DeepMindControlEnv(args.env, args.seed, size=size)
    elif backend in {"dmlab", "deepmind_lab", "deepmindlab"}:
        from simulusion.envs.dmlab import DMLabEnv

        env = DMLabEnv(
            args.env,
            seed=args.seed,
            size=size,
            action_repeat=int(getattr(args, "dmlab_action_repeat", 4)),
            action_set=getattr(args, "dmlab_action_set", None),
            observations=getattr(args, "dmlab_observations", None),
            config=getattr(args, "dmlab_config", None),
            renderer=getattr(args, "dmlab_renderer", "hardware"),
        )
    elif backend in {"gym", "gymnasium", "generic"}:
        from simulusion.envs.gym_env import GymImageEnv

        env = GymImageEnv(
            args.env,
            seed=args.seed,
            size=size,
            render_mode=getattr(args, "gym_render_mode", "rgb_array"),
        )
    elif backend in {"mujoco", "mjcf", "native_mujoco"}:
        from simulusion.envs.mujoco_env import make_mujoco_env_from_config

        env = make_mujoco_env_from_config(args, size)
    elif backend in {"procgen", "coinrun"}:
        from simulusion.envs.procgen_env import ProcgenImageEnv

        env = ProcgenImageEnv(
            args.env,
            seed=args.seed,
            size=size,
            distribution_mode=getattr(args, "procgen_distribution_mode", "easy"),
            num_levels=int(getattr(args, "procgen_num_levels", 0)),
            start_level=getattr(args, "procgen_start_level", None),
            max_episode_steps=int(getattr(args, "time_limit", 1000)),
        )
    elif backend in {"robotics", "gymnasium_robotics"}:
        from simulusion.envs.robotics_env import make_robotics_env

        env = make_robotics_env(
            args.env,
            seed=args.seed,
            size=size,
            render_mode=getattr(args, "gym_render_mode", "rgb_array"),
        )
    elif backend in {"bsuite", "behavior_suite", "behaviour_suite"}:
        from simulusion.envs.bsuite_env import BSuiteImageEnv

        env = BSuiteImageEnv(
            args.env,
            seed=args.seed,
            size=size,
        )
    elif backend in {"brax", "jax_brax"}:
        from simulusion.envs.brax_env import BraxImageEnv

        env = BraxImageEnv(
            args.env,
            seed=args.seed,
            size=size,
            backend=getattr(args, "brax_backend", "generalized"),
            episode_length=int(getattr(args, "time_limit", 1000)),
            auto_reset=bool(getattr(args, "brax_auto_reset", False)),
            jit=bool(getattr(args, "brax_jit", True)),
            suppress_warp_warnings=bool(
                getattr(args, "brax_suppress_warp_warnings", True)
            ),
        )
    elif backend in {"unity", "unity_mlagents", "mlagents"}:
        from simulusion.envs.unity_env import UnityMLAgentsEnv

        unity_file_name = getattr(args, "unity_file_name", None)
        if not unity_file_name:
            raise ValueError(
                "unity_file_name must be provided when env_backend='unity_mlagents'."
            )
        env = UnityMLAgentsEnv(
            file_name=unity_file_name,
            behavior_name=getattr(args, "unity_behavior_name", None),
            seed=args.seed,
            size=size,
            worker_id=int(getattr(args, "unity_worker_id", 0)),
            base_port=int(getattr(args, "unity_base_port", 5005)),
            no_graphics=bool(getattr(args, "unity_no_graphics", True)),
            time_scale=float(getattr(args, "unity_time_scale", 20.0)),
            quality_level=int(getattr(args, "unity_quality_level", 1)),
            max_episode_steps=int(getattr(args, "time_limit", 1000)),
        )
    else:
        raise ValueError(
            f"Unknown env_backend='{backend}'. Use one of: dmc, dmlab, gym, mujoco, "
            "robotics, procgen, bsuite, brax, unity_mlagents."
        )

    env = env_wrapper.ActionRepeat(env, int(args.action_repeat))
    env = env_wrapper.NormalizeActions(env)
    frame_stack = max(1, int(getattr(args, "frame_stack", 1)))
    if frame_stack > 1:
        env = env_wrapper.FrameStack(env, frame_stack)
    repeat = max(1, int(args.action_repeat))
    duration = max(1, int(args.time_limit) // repeat)
    env = env_wrapper.TimeLimit(env, duration)
    return env


def _coerce_dreamer_config(config: Any | None) -> DreamerConfig:
    """Normalize Dreamer config inputs to a ``DreamerConfig`` instance."""

    if config is None:
        return DreamerConfig()
    if isinstance(config, DreamerConfig):
        return config
    if isinstance(config, dict):
        return DreamerConfig.from_dict(config)
    if isinstance(config, (str, Path)):
        return DreamerConfig.from_yaml(config)
    raise TypeError(
        "config must be a DreamerConfig, dict, YAML path/string, or None; "
        f"got {type(config).__name__}."
    )


def _apply_config_overrides(
    config: DreamerConfig, overrides: dict[str, Any]
) -> DreamerConfig:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Invalid DreamerConfig override: {key}")
        setattr(config, key, value)
    return config


def _default_device(config: DreamerConfig) -> torch.device:
    from simulusion.utils.device import get_default_device

    return get_default_device(allow_gpu=not config.no_gpu)


def _find_local_pretrained_file(path: Path, candidates: tuple[str, ...]) -> Path | None:
    if path.is_file():
        return path
    if path.is_dir():
        for name in candidates:
            candidate = path / name
            if candidate.exists():
                return candidate
        for pattern in ("*.pt", "*.pth", "*.bin"):
            matches = sorted(path.glob(pattern))
            if matches:
                return matches[0]
    return None


def _resolve_pretrained_file(
    pretrained_model_name_or_path: str | Path,
    candidates: tuple[str, ...],
    *,
    repo_type: str | None = None,
    revision: str | None = None,
) -> Path | None:
    local_path = Path(pretrained_model_name_or_path)
    local_file = _find_local_pretrained_file(local_path, candidates)
    if local_file is not None:
        return local_file

    if importlib.util.find_spec("huggingface_hub") is None:
        return None

    from huggingface_hub import hf_hub_download

    repo_id = str(pretrained_model_name_or_path)
    for filename in candidates:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
            )
        except Exception:
            continue
        return Path(downloaded)
    return None


def _save_config_next_to_checkpoint(
    config: DreamerConfig, checkpoint_path: str | Path
) -> None:
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(checkpoint.parent / "config.yaml")


def _true_termination(done: bool, info: Any) -> bool:
    """Whether a step ended the episode for real, not by a time limit.

    Wrappers such as :class:`simulusion.envs.wrappers.TimeLimit` report ``done``
    for truncations too, and describe which it was via ``terminated`` /
    ``truncated`` in ``info``. Without that information ``done`` is used.
    """
    if not done:
        return False
    if isinstance(info, dict):
        if "terminated" in info:
            return bool(info["terminated"])
        if info.get("truncated") or info.get("TimeLimit.truncated"):
            return False
    return True


def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    """Convert raw uint8 image observations to Dreamer float input space.

    Images are scaled from `[0, 255]` to roughly `[-0.5, 0.5]`, matching the
    normalization expected by Dreamer encoders.
    """
    obs = obs.to(torch.float32) / 255.0 - 0.5
    return obs


class Dreamer:
    """Core Dreamer training system combining world model, actor, and value nets.

    This class owns model construction, replay sampling, imagination rollouts,
    loss computation, optimization steps, evaluation loops, and checkpoint I/O.
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
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = torch.device(device)
        self.restore = bool(restore)
        self.restore_path = args.checkpoint_path

        if getattr(args, "perf_defaults", True):
            enable_performance_defaults(tf32=bool(getattr(args, "tf32", True)))

        # Calculate memory per sample
        obs_bytes = np.prod(obs_shape) * 1  # uint8
        action_bytes = action_size * 4  # float32
        reward_bytes = 4  # float32
        terminal_bytes = 4  # float32
        bytes_per_sample = obs_bytes + action_bytes + reward_bytes + terminal_bytes

        # Get available memory
        available_memory = get_available_memory()
        # Use 80% of available memory for buffer to leave margin for other processes
        max_buffer_size = int((available_memory * 0.8) // bytes_per_sample)
        adaptive_buffer_size = min(args.buffer_size, max_buffer_size)

        if adaptive_buffer_size < args.buffer_size:
            logger.warning(
                "Reducing buffer size from %s to %s due to memory constraints.",
                args.buffer_size,
                adaptive_buffer_size,
            )

        self.data_buffer = ReplayBuffer(
            adaptive_buffer_size,
            self.obs_shape,
            self.action_size,
            self.args.train_seq_len,
            self.args.batch_size,
        )

        self.use_amp = bool(
            getattr(args, "use_amp", True)
            and getattr(device, "type", str(device)) == "cuda"
        )
        # What `use_amp` selects: fp16 autocast with loss scaling on CUDA. The
        # shared Trainer replaces it with the run's precision (set_precision).
        self.precision = MixedPrecision.from_use_amp(self.use_amp, self.device)
        # `ConvDecoder` emits Independent(Normal(mean, 1), len(obs_shape)), whose
        # negative log-likelihood carries a constant term of
        # prod(obs_shape) * 0.5 * log(2*pi). For 3x64x64 that is ~11292 nats,
        # which swamps the few nats of reconstruction signal and makes the
        # summed `model_loss` look frozen. Subtract it when logging.
        self._obs_nll_constant = float(
            np.prod(self.obs_shape) * 0.5 * np.log(2.0 * np.pi)
        )
        # Latest per-component loss values, refreshed by `train_one_batch`.
        self.metrics: dict[str, float] = {}
        self._model_components: dict[str, float] = {}
        self._build_model(restore=self.restore)

    def _build_model(self, restore: bool = False) -> None:
        self.rssm = RSSM(
            action_size=self.action_size,
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            hidden_size=self.args.deter_size,
            obs_embed_size=self.args.obs_embed_size,
            activation=self.args.dense_activation_function,
        ).to(self.device)

        self.actor = ActionDecoder(
            action_size=self.action_size,
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            units=self.args.num_units,
            n_layers=4,
            activation=self.args.dense_activation_function,
        ).to(self.device)
        self.obs_encoder = ConvEncoder(
            input_shape=self.obs_shape,
            embed_size=self.args.obs_embed_size,
            activation=self.args.cnn_activation_function,
        ).to(self.device)
        self.obs_decoder = ConvDecoder(
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            output_shape=self.obs_shape,
            activation=self.args.cnn_activation_function,
        ).to(self.device)
        head_dist, head_kwargs = self._get_head_config()

        self.reward_model = DenseDecoder(
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            output_shape=(1,),
            n_layers=2,
            units=self.args.num_units,
            activation=self.args.dense_activation_function,
            dist=head_dist,
            **head_kwargs,
        ).to(self.device)
        self.value_model = DenseDecoder(
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            output_shape=(1,),
            n_layers=3,
            units=self.args.num_units,
            activation=self.args.dense_activation_function,
            dist=head_dist,
            **head_kwargs,
        ).to(self.device)
        if self.args.use_disc_model:
            self.discount_model = DenseDecoder(
                stoch_size=self.args.stoch_size,
                deter_size=self.args.deter_size,
                output_shape=(1,),
                n_layers=2,
                units=self.args.num_units,
                activation=self.args.dense_activation_function,
                dist="binary",
            ).to(self.device)

        if self.args.use_disc_model:
            self.world_model_params = (
                list(self.rssm.parameters())
                + list(self.obs_encoder.parameters())
                + list(self.obs_decoder.parameters())
                + list(self.reward_model.parameters())
                + list(self.discount_model.parameters())
            )
        else:
            self.world_model_params = (
                list(self.rssm.parameters())
                + list(self.obs_encoder.parameters())
                + list(self.obs_decoder.parameters())
                + list(self.reward_model.parameters())
            )

        self.world_model_opt = optim.Adam(
            self.world_model_params, self.args.model_learning_rate
        )
        self.value_opt = optim.Adam(
            self.value_model.parameters(), self.args.value_learning_rate
        )
        self.actor_opt = optim.Adam(
            self.actor.parameters(), self.args.actor_learning_rate
        )
        self._build_updates()

        self.world_model_modules: list[nn.Module]
        if self.args.use_disc_model:
            self.world_model_modules = [
                self.rssm,
                self.obs_encoder,
                self.obs_decoder,
                self.reward_model,
                self.discount_model,
            ]
        else:
            self.world_model_modules = [
                self.rssm,
                self.obs_encoder,
                self.obs_decoder,
                self.reward_model,
            ]
        self.value_modules: list[nn.Module] = [self.value_model]
        self.actor_modules: list[nn.Module] = [self.actor]

        if restore:
            self.restore_checkpoint(self.restore_path)

    @classmethod
    def from_config(
        cls,
        config: DreamerConfig | dict[str, Any] | str | Path | None = None,
        *,
        obs_shape: tuple[int, ...] | None = None,
        action_size: int | None = None,
        device: str | torch.device | None = None,
        restore: bool | None = None,
        **overrides: Any,
    ) -> "Dreamer":
        """Build a core Dreamer model from a config object, dict, or YAML file.

        ``obs_shape`` and ``action_size`` may be supplied directly. When either
        is omitted, this method constructs a temporary environment from the
        config to infer the model shapes.
        """

        args = _apply_config_overrides(_coerce_dreamer_config(config), overrides)
        if obs_shape is None or action_size is None:
            env = make_env(args)
            if obs_shape is None:
                obs_shape = tuple(env.observation_space["image"].shape)
            if action_size is None:
                action_size = int(env.action_space.shape[0])
        torch_device = (
            torch.device(device) if device is not None else _default_device(args)
        )
        should_restore = args.restore if restore is None else restore
        return cls(args, obs_shape, action_size, torch_device, should_restore)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: DreamerConfig | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **overrides: Any,
    ) -> "Dreamer":
        """Load a Dreamer checkpoint from a local path/directory or the HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("model.pt", "pytorch_model.bin", "checkpoint.pt", "ckpt.pt")
        )
        checkpoint_path = _resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find a Dreamer checkpoint for {pretrained_model_name_or_path!r}."
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or "cpu", weights_only=True
        )
        checkpoint_config = (
            checkpoint.get("config") if isinstance(checkpoint, dict) else None
        )
        if config is None and checkpoint_config is not None:
            args = _coerce_dreamer_config(checkpoint_config)
        elif config is not None:
            args = _coerce_dreamer_config(config)
        else:
            config_path = _resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "dreamer_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = DreamerConfig.from_yaml(config_path)
        args = _apply_config_overrides(args, overrides)

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

    def parameter_count(self, trainable_only: bool = False) -> int:
        """Return the total number of parameters owned by the Dreamer modules."""

        return sum(
            param.numel()
            for module in (
                self.world_model_modules + self.actor_modules + self.value_modules
            )
            for param in module.parameters()
            if not trainable_only or param.requires_grad
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact parameter-count summary for the Dreamer modules."""

        modules: dict[str, nn.Module] = {
            "rssm": self.rssm,
            "actor": self.actor,
            "reward_model": self.reward_model,
            "obs_encoder": self.obs_encoder,
            "obs_decoder": self.obs_decoder,
            "value_model": self.value_model,
        }
        if self.args.use_disc_model:
            modules["discount_model"] = self.discount_model
        module_params = {
            name: sum(param.numel() for param in module.parameters())
            for name, module in modules.items()
        }
        trainable_params = {
            name: sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            for name, module in modules.items()
        }
        return {
            "total_parameters": sum(module_params.values()),
            "trainable_parameters": sum(trainable_params.values()),
            "modules": module_params,
            "trainable_modules": trainable_params,
        }

    @assert_finite
    def _build_updates(self) -> None:
        clip = self.args.grad_clip_norm
        self.world_model_update = GradientUpdate(
            self.world_model_opt,
            self.world_model_params,
            self.precision,
            max_grad_norm=clip,
        )
        self.actor_update = GradientUpdate(
            self.actor_opt, self.actor.parameters(), self.precision, max_grad_norm=clip
        )
        self.value_update = GradientUpdate(
            self.value_opt,
            self.value_model.parameters(),
            self.precision,
            max_grad_norm=clip,
        )

    def set_precision(self, precision: MixedPrecision) -> None:
        """Train at ``precision`` from now on (fresh loss scalers)."""
        self.precision = precision
        self.use_amp = precision.enabled
        self._build_updates()

    def update_world_model(
        self,
        obs: torch.Tensor,
        acs: torch.Tensor,
        rews: torch.Tensor,
        nonterms: torch.Tensor,
        continues: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One world-model optimizer step; returns ``(loss, grad_norm)``.

        Inputs are time-major, as the replay buffer samples them: ``obs``
        ``(L, B, C, H, W)`` uint8, ``acs`` ``(L, B, A)``, ``rews``,
        ``nonterms`` and ``continues`` ``(L, B, 1)``, where row ``t`` holds
        the action taken at ``obs[t]`` and what followed it.
        """
        with self.precision.autocast():
            model_loss = self.world_model_loss(obs, acs, rews, nonterms, continues)
        return model_loss, self.world_model_update(model_loss)

    def world_model_loss(
        self,
        obs: torch.Tensor,
        acs: torch.Tensor,
        rews: torch.Tensor,
        nonterms: torch.Tensor,
        continues: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """World-model loss.

        ``nonterms`` masks recurrent state at every episode boundary.
        ``continues`` is the discount-head target, which is 0 only on true
        terminations; it defaults to ``nonterms`` when not supplied.
        """
        if continues is None:
            continues = nonterms
        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs[1:])
        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(
            obs_embed, acs[:-1], nonterms[:-1], init_state, self.args.train_seq_len - 1
        )
        features = torch.cat([self.posterior["stoch"], self.posterior["deter"]], dim=-1)
        rew_dist = self.reward_model(features)
        obs_dist = self.obs_decoder(features)
        disc_dist = None
        if self.args.use_disc_model:
            disc_dist = self.discount_model(features)

        prior_dist = self.rssm.get_dist(prior["mean"], prior["std"])
        post_dist = self.rssm.get_dist(self.posterior["mean"], self.posterior["std"])

        kl_loss = self._compute_kl_loss(prior, self.posterior, post_dist, prior_dist)
        target_rew = self._get_reward_target(rews)

        obs_loss = -torch.mean(obs_dist.log_prob(obs[1:]))
        rew_loss = -torch.mean(rew_dist.log_prob(target_rew))

        model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss
        if self.args.use_disc_model and disc_dist is not None:
            disc_loss = -torch.mean(disc_dist.log_prob(continues[:-1]))
            model_loss = model_loss + self.args.disc_loss_coeff * disc_loss
        else:
            disc_loss = None

        # The unclamped KL is the quantity that actually reveals whether the
        # free-nats floor is active; `kl_loss` is pinned to `free_nats` (and
        # carries no gradient) whenever the raw value sits below it.
        with torch.no_grad():
            kl_raw = torch.mean(distributions.kl.kl_divergence(post_dist, prior_dist))

        components = {
            "loss/obs_nll": obs_loss,
            "loss/obs_recon": obs_loss - self._obs_nll_constant,
            "loss/reward_nll": rew_loss,
            "loss/kl_raw": kl_raw,
            "loss/kl_used": kl_loss,
        }
        if disc_loss is not None:
            components["loss/discount_nll"] = disc_loss
        self._model_components = {
            name: float(value.detach()) for name, value in components.items()
        }
        self._model_components["loss/kl_at_free_nats_floor"] = float(
            kl_raw < self.args.free_nats
        )

        return model_loss

    @assert_finite
    def actor_loss(self) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.rssm.detach_state(self.rssm.seq_to_batch(self.posterior))

        with FreezeParameters(self.world_model_modules):
            imag_states = self.rssm.imagine_rollout(
                self.actor, posterior, self.args.imagine_horizon
            )

        self.imag_feat = torch.cat([imag_states["stoch"], imag_states["deter"]], dim=-1)

        with FreezeParameters(
            list(self.world_model_modules) + list(self.value_modules)
        ):
            imag_rew_dist = self.reward_model(self.imag_feat)
            imag_val_dist = self.value_model(self.imag_feat)

            imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.args.use_disc_model:
                imag_disc_dist = self.discount_model(self.imag_feat)
                # ``mean`` is a property on torch distributions. The head is
                # trained on continuation flags, so its mean is P(continue);
                # scale by gamma as in the Dreamer reference
                # (pcont_target = discount * data["discount"]).
                discounts = self.args.discount * imag_disc_dist.mean.detach()
            else:
                discounts = self.args.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(
            imag_rews[:-1],
            imag_vals[:-1],
            discounts[:-1],
            self.args.td_lambda,
            imag_vals[-1],
        )

        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()

        return self._compute_actor_loss(self.returns, self.discounts)

    @assert_finite
    def value_loss(self) -> torch.Tensor:
        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            value_targ = self.returns.detach()
        return self._compute_value_loss(value_feat, value_targ, self.discounts)

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        """Move a replay-buffer array to the training device.

        ``torch.from_numpy`` shares the buffer instead of copying it the way
        ``torch.tensor`` does, and pinning lets the copy overlap compute rather
        than blocking on it. Crucially the array's dtype is preserved: the
        observation buffer is ``uint8``, and widening it here would quadruple
        the bytes crossing the bus for a cast ``preprocess_obs`` already does on
        the device.
        """
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if getattr(self.device, "type", str(self.device)) == "cuda":
            return tensor.pin_memory().to(self.device, non_blocking=True)
        return tensor.to(self.device)

    def train_one_batch(self) -> list[float]:
        obs, acs, rews, terms, terminated = self.data_buffer.sample(
            include_terminated=True
        )
        # Stays uint8 across the bus; `preprocess_obs` casts and normalises on
        # the device, so the result is bit-identical to casting up front.
        obs_t = self._to_device(obs)

        if obs_t.shape[0] <= 1:
            return [0.0, 0.0, 0.0]

        acs_t = self._to_device(acs).to(torch.float32)
        rews_t = self._to_device(rews).to(torch.float32).unsqueeze(-1)
        reward_scale = float(getattr(self.args, "reward_scale", 1.0))
        if reward_scale != 1.0:
            # Scaling here keeps the reward head, imagined rewards, lambda
            # returns, and value targets in one consistent space. Rewards
            # logged from the environment stay in their native units.
            rews_t = rews_t * reward_scale
        # ``1.0 - terms`` runs on the device now; it used to allocate a second
        # host array before the transfer.
        nonterms = (1.0 - self._to_device(terms).to(torch.float32)).unsqueeze(-1)
        continues = (1.0 - self._to_device(terminated).to(torch.float32)).unsqueeze(-1)

        model_loss, model_grad_norm = self.update_world_model(
            obs_t, acs_t, rews_t, nonterms, continues
        )

        with self.precision.autocast():
            actor_loss = self.actor_loss()
        actor_grad_norm = self.actor_update(actor_loss)

        with self.precision.autocast():
            value_loss = self.value_loss()
        value_grad_norm = self.value_update(value_loss)

        clip = self.args.grad_clip_norm
        self.metrics = dict(self._model_components)
        for name, norm in (
            ("model", model_grad_norm),
            ("actor", actor_grad_norm),
            ("value", value_grad_norm),
        ):
            norm_value = float("nan") if norm is None else float(norm)
            self.metrics[f"grad_norm/{name}"] = norm_value
            # A norm that is non-finite means AMP overflowed and GradScaler
            # silently skipped the update; a norm above `clip` means the step
            # size was decided by the clip threshold rather than the loss.
            self.metrics[f"grad_norm/{name}_nonfinite"] = float(
                not np.isfinite(norm_value)
            )
            self.metrics[f"grad_norm/{name}_clipped"] = float(
                np.isfinite(norm_value) and norm_value > clip
            )

        return (
            torch.stack([model_loss.detach(), actor_loss.detach(), value_loss.detach()])
            .cpu()
            .tolist()
        )

    def act_with_world_model(
        self,
        obs: Any,
        prev_state: Any,
        prev_action: torch.Tensor,
        explore: bool = False,
    ) -> tuple:
        obs = obs["image"]
        obs = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        # DreamerRSSM.observe_step returns (posterior, prior). Acting on the
        # prior would discard the frame just encoded and roll the agent's belief
        # forward blind; the posterior is the state that has seen ``obs``.
        posterior, _ = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = torch.cat([posterior["stoch"], posterior["deter"]], dim=-1)
        action = self.actor(features, deter=not explore)
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)

        return posterior, action

    def act_and_collect_data(self, env: Any, collect_steps: int) -> np.ndarray:
        obs = env.reset()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.action_size).to(self.device)

        episode_rewards = [0.0]

        for i in range(collect_steps):
            with torch.no_grad():
                posterior, action = self.act_with_world_model(
                    obs, prev_state, prev_action, explore=True
                )
            action = action[0].cpu().numpy()
            next_obs, rew, done, info = env.step(action)
            executed_action = (
                info["action"]
                if isinstance(info, dict) and ("action" in info)
                else action
            )
            self.data_buffer.add(
                obs, executed_action, rew, done, _true_termination(done, info)
            )

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                if i != collect_steps - 1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs
                prev_state = posterior
                prev_action = (
                    torch.tensor(executed_action, dtype=torch.float32)
                    .to(self.device)
                    .unsqueeze(0)
                )

        return np.array(episode_rewards)

    def evaluate(self, env: Any, eval_episodes: int, render: bool = False) -> tuple:
        episode_rew = np.zeros((eval_episodes))

        video_images: list[list[Any]] = [[] for _ in range(eval_episodes)]
        latents: list[Any] | None = [] if render else None

        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            prev_state = self.rssm.init_state(1, self.device)
            prev_action = torch.zeros(1, self.action_size).to(self.device)

            while not done:
                with torch.no_grad():
                    posterior, action = self.act_with_world_model(
                        obs, prev_state, prev_action
                    )
                action = action[0].cpu().numpy()
                next_obs, rew, done, info = env.step(action)
                executed_action = (
                    info["action"]
                    if isinstance(info, dict) and ("action" in info)
                    else action
                )
                prev_state = posterior
                prev_action = (
                    torch.tensor(executed_action, dtype=torch.float32)
                    .to(self.device)
                    .unsqueeze(0)
                )

                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs["image"].transpose(1, 2, 0).copy())
                    if latents is not None:
                        latents.append(
                            torch.cat([posterior["stoch"], posterior["deter"]], dim=-1)
                            .cpu()
                            .numpy()
                        )
                obs = next_obs
        latents_arr: Any = None
        if latents is not None and len(latents) > 0:
            latents_arr = np.array(latents)
        return (
            episode_rew,
            np.array(video_images[: self.args.max_videos_to_save]),
            latents_arr,
        )

    def collect_random_episodes(self, env: Any, seed_steps: int) -> np.ndarray:
        obs = env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = env.action_space.sample()
            next_obs, rew, done, info = env.step(action)
            executed_action = (
                info["action"]
                if isinstance(info, dict) and ("action" in info)
                else action
            )

            self.data_buffer.add(
                obs, executed_action, rew, done, _true_termination(done, info)
            )
            seed_episode_rews[-1] += rew
            if done:
                obs = env.reset()
                if i != seed_steps - 1:
                    seed_episode_rews.append(0.0)
                done = False
            else:
                obs = next_obs

        return np.array(seed_episode_rews)

    def save(self, save_path: str) -> None:
        _save_config_next_to_checkpoint(self.args, save_path)
        torch.save(
            {
                "config": self.args.to_dict(),
                "obs_shape": tuple(self.obs_shape),
                "action_size": int(self.action_size),
                "rssm": self.rssm.state_dict(),
                "actor": self.actor.state_dict(),
                "value_model": self.value_model.state_dict(),
                "reward_model": self.reward_model.state_dict(),
                "obs_encoder": self.obs_encoder.state_dict(),
                "obs_decoder": self.obs_decoder.state_dict(),
                "discount_model": (
                    self.discount_model.state_dict()
                    if self.args.use_disc_model
                    else None
                ),
                "actor_optimizer": self.actor_opt.state_dict(),
                "value_optimizer": self.value_opt.state_dict(),
                "world_model_optimizer": self.world_model_opt.state_dict(),
            },
            save_path,
        )

    def restore_checkpoint(
        self, ckpt_path: str | Path, map_location: Any = None
    ) -> None:
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        self.rssm.load_state_dict(checkpoint["rssm"])
        self.actor.load_state_dict(checkpoint["actor"])
        # Checkpoints written before the critic was saved lack this key. They
        # still load, but the critic keeps its fresh initialisation, so its
        # optimizer state (built for other weights) is skipped below too.
        has_critic = checkpoint.get("value_model") is not None
        if has_critic:
            self.value_model.load_state_dict(checkpoint["value_model"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.obs_encoder.load_state_dict(checkpoint["obs_encoder"])
        self.obs_decoder.load_state_dict(checkpoint["obs_decoder"])
        if self.args.use_disc_model and (checkpoint["discount_model"] is not None):
            self.discount_model.load_state_dict(checkpoint["discount_model"])

        self.world_model_opt.load_state_dict(checkpoint["world_model_optimizer"])
        self.actor_opt.load_state_dict(checkpoint["actor_optimizer"])
        if has_critic:
            self.value_opt.load_state_dict(checkpoint["value_optimizer"])

    def _get_head_config(self) -> tuple[str, dict]:
        return "normal", {}

    def _compute_kl_loss(
        self,
        prior: dict,
        posterior: dict,
        post_dist: distributions.Distribution,
        prior_dist: distributions.Distribution,
    ) -> torch.Tensor:
        kl_loss = torch.mean(distributions.kl.kl_divergence(post_dist, prior_dist))
        kl_loss = torch.max(
            kl_loss, kl_loss.new_full(kl_loss.size(), self.args.free_nats)
        )
        return kl_loss

    def _get_reward_target(self, rews: torch.Tensor) -> torch.Tensor:
        return rews[:-1]

    def _compute_actor_loss(
        self, returns: torch.Tensor, discounts: torch.Tensor
    ) -> torch.Tensor:
        return -torch.mean(discounts * returns)

    def _compute_value_loss(
        self,
        value_feat: torch.Tensor,
        value_targ: torch.Tensor,
        discounts: torch.Tensor,
    ) -> torch.Tensor:
        value_dist = self.value_model(value_feat)
        log_prob = value_dist.log_prob(value_targ).unsqueeze(-1)
        return -torch.mean(discounts * log_prob)


class DreamerAgent(ExportableAgentMixin):
    """High-level user API for running Dreamer experiments end to end.

    It builds environments from config, initializes seeds and logging,
    instantiates `Dreamer`, and exposes simple `train()` / `evaluate()` methods.
    """

    #: The world model + actor-critic this agent trains. Versioned agents
    #: (``DreamerV1Agent``, ``DreamerV2Agent``) override it.
    core_cls: type[Dreamer] = Dreamer
    #: Run-folder name used when the config leaves ``algo`` at its default.
    algo_name: str | None = None

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        self.args = _coerce_dreamer_config(config)
        if self.algo_name and self.args.algo == DreamerConfig.algo:
            self.args.algo = self.algo_name

        self.last_latents_ref = kwargs.get("last_latents_ref", None)

        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            elif key == "logdir":
                # Accept either `logdir` (server/legacy) and mirror into
                # both `logdir` and `log_dir` so downstream code using either
                # naming convention picks up the value.
                setattr(self.args, "logdir", value)
                try:
                    setattr(self.args, "log_dir", value)
                except Exception:
                    pass
            elif key == "last_latents_ref":
                self.last_latents_ref = value
            elif key == "data_path":
                # Backwards-compatible alias for the new configurable data_dir.
                setattr(self.args, "data_dir", value)
            else:
                raise ValueError(f"Invalid argument: {key}")

        data_path = (
            getattr(self.args, "data_dir", None)
            or os.environ.get("SIMULUSION_DATA_DIR")
            or getattr(self.args, "log_dir", None)
            or "runs"
        )
        data_path = os.path.abspath(os.path.expanduser(data_path))

        if not (os.path.exists(data_path)):
            os.makedirs(data_path)

        # Allow caller to pass an absolute `logdir`. If provided and absolute,
        # use it verbatim. Otherwise keep the historical behavior of creating
        # a subdir under the package data path so examples/tests remain stable.
        if hasattr(self.args, "logdir") and self.args.logdir is not None:
            self.logdir = self.args.logdir
        else:
            self.logdir = (
                self.args.env
                + "_"
                + self.args.algo
                + "_"
                + self.args.exp_name
                + "_"
                + time.strftime("%d-%m-%Y-%H-%M-%S")
            )

        # If `self.logdir` is not an absolute path, place it under the
        # configurable data directory instead of the package source tree.
        if not os.path.isabs(self.logdir):
            self.logdir = os.path.join(data_path, self.logdir)
        if not (os.path.exists(self.logdir)):
            os.makedirs(self.logdir)
        self.args.to_yaml(os.path.join(self.logdir, "config.yaml"))

        setup_logging(
            "simulusion",
            getattr(self.args, "log_level", "INFO"),
            getattr(self.args, "log_file", None),
        )
        if getattr(self.args, "detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
            logger.warning("torch.autograd anomaly detection is enabled")

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        device = _default_device(self.args)
        if device.type == "cuda":
            torch.cuda.manual_seed(self.args.seed)
        elif device.type == "cpu":
            print("WARNING: no GPU (CUDA or MPS) available, using CPU")

        self.train_env = make_env(self.args)
        self.test_env = make_env(self.args)

        obs_shape = self.train_env.observation_space["image"].shape
        action_size = self.train_env.action_space.shape[0]
        self.dreamer = self.core_cls(
            self.args, obs_shape, action_size, device, self.args.restore
        )

        self.logger = Logger(
            self.logdir,
            enable_wandb=self.args.enable_wandb,
            wandb_project=self.args.wandb_project,
            wandb_entity=self.args.wandb_entity,
            video_format=self.args.video_format,
            video_fps=self.args.video_fps,
            enable_tensorboard=getattr(self.args, "enable_tensorboard", False),
            enable_console=getattr(self.args, "enable_console_metrics", True),
            enable_jsonl=getattr(self.args, "enable_jsonl", True),
            jsonl_filename=getattr(self.args, "jsonl_filename", "metrics.jsonl"),
        )

    @classmethod
    def from_config(
        cls,
        config: DreamerConfig | dict[str, Any] | str | Path | None = None,
        **overrides: Any,
    ) -> "DreamerAgent":
        """Build a high-level Dreamer agent from a config object, dict, or YAML file."""

        return cls(_apply_config_overrides(_coerce_dreamer_config(config), overrides))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: DreamerConfig | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **overrides: Any,
    ) -> "DreamerAgent":
        """Create a Dreamer agent and restore weights from a local path or HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("model.pt", "pytorch_model.bin", "checkpoint.pt", "ckpt.pt")
        )
        checkpoint_path = _resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find a Dreamer checkpoint for {pretrained_model_name_or_path!r}."
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or "cpu", weights_only=True
        )
        checkpoint_config = (
            checkpoint.get("config") if isinstance(checkpoint, dict) else None
        )
        if config is None and checkpoint_config is not None:
            args = _coerce_dreamer_config(checkpoint_config)
        elif config is not None:
            args = _coerce_dreamer_config(config)
        else:
            config_path = _resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "dreamer_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = DreamerConfig.from_yaml(config_path)
        args = _apply_config_overrides(args, overrides)
        args.restore = False
        agent = cls(args)
        agent.dreamer.restore_checkpoint(checkpoint_path, map_location=map_location)
        return agent

    def parameter_count(self, trainable_only: bool = False) -> int:
        """Return the total number of Dreamer parameters."""

        return self.dreamer.parameter_count(trainable_only=trainable_only)

    def summary(self) -> dict[str, Any]:
        """Return a compact parameter-count summary for the wrapped Dreamer model."""

        return self.dreamer.summary()

    def train(self, total_steps: int | None = None) -> None:
        """Train for ``total_steps`` environment steps (``args.total_steps`` by default).

        Runs on the shared :class:`~simulusion.training.Trainer` through the
        ``dreamer`` task, with this agent's environments, model and logger, so
        metrics and evaluation videos go where they always went. Evaluation
        runs every ``test_interval`` steps, early stopping on its average
        reward when ``early_stopping`` is set, and checkpoints land in
        ``<logdir>/ckpts/<step>_ckpt.pt`` every ``checkpoint_interval`` steps
        and at the end, as before. The trainer adds a resumable copy of each
        under ``<logdir>/checkpoints/``, plus ``run.json``,
        ``run_config.yaml`` and ``train_metrics.jsonl``.
        """
        from simulusion.training.legacy import fit_legacy
        from simulusion.training.tasks.dreamer import DreamerTask, DreamerTaskConfig

        args = self.args
        early = bool(getattr(args, "early_stopping", False))
        fit_legacy(
            # The replay buffer is not checkpointed here: this loop never
            # resumed, and the buffer can hold millions of frames.
            DreamerTask(DreamerTaskConfig(model=args, save_replay=False), agent=self),
            self.logdir,
            max_steps=int(total_steps if total_steps is not None else args.total_steps),
            device=self.dreamer.device,
            seed=args.seed,
            eval_every=args.test_interval,
            checkpoint_every=args.checkpoint_interval,
            early_stopping_metric="eval_avg_reward" if early else None,
            early_stopping_mode="max",
            patience=getattr(args, "patience", 10),
            min_delta=getattr(args, "min_delta", 1e-4),
            # The agent's own logger already prints to the console.
            console=False,
        )

    def evaluate(self) -> tuple:
        logs = OrderedDict()
        episode_rews, video_images, latents = self.dreamer.evaluate(
            self.test_env, self.args.test_episodes, render=True
        )
        if self.last_latents_ref is not None and latents is not None:
            self.last_latents_ref[0] = latents
        logs.update(
            {
                "test_avg_reward": np.mean(episode_rews),
                "test_max_reward": np.max(episode_rews),
                "test_min_reward": np.min(episode_rews),
                "test_std_reward": np.std(episode_rews),
            }
        )
        self.logger.dump_scalars_to_pickle(logs, 0, log_title="test_scalars.pkl")
        self.logger.log_videos(
            video_images, 0, max_videos_to_save=self.args.max_videos_to_save
        )
        self.logger.flush()
        return episode_rews, video_images, latents
