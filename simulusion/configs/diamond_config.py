from dataclasses import dataclass, field
from typing import List, Optional


from simulusion.configs.serialization import SerializableConfigMixin
from simulusion.benchmarks import (
    ATARI100K_GAMES as _GAMES,
    ATARI_HUMAN_SCORES,
    ATARI_RANDOM_SCORES,
    atari_env_id,
    atari_game,
)


def get_default_device() -> str:
    try:
        from simulusion.utils.device import default_device_name

        return default_device_name()
    except AttributeError:
        return "cpu"


@dataclass
class ModelPreset(SerializableConfigMixin):
    """Model architecture preset for different hardware tiers."""

    diffusion_channels: List[int]
    diffusion_res_blocks: int
    diffusion_cond_dim: int
    reward_channels: List[int]
    reward_lstm_dim: int
    actor_channels: List[int]
    actor_lstm_dim: int


MODEL_PRESETS = {
    "small": ModelPreset(
        diffusion_channels=[32, 32, 32, 32],
        diffusion_res_blocks=2,
        diffusion_cond_dim=128,
        reward_channels=[16, 16, 16, 16],
        reward_lstm_dim=256,
        actor_channels=[16, 16, 32, 32],
        actor_lstm_dim=256,
    ),
    "medium": ModelPreset(
        diffusion_channels=[64, 64, 64, 64],
        diffusion_res_blocks=2,
        diffusion_cond_dim=256,
        reward_channels=[32, 32, 32, 32],
        reward_lstm_dim=512,
        actor_channels=[32, 32, 64, 64],
        actor_lstm_dim=512,
    ),
    "large": ModelPreset(
        diffusion_channels=[128, 128, 128, 128],
        diffusion_res_blocks=3,
        diffusion_cond_dim=512,
        reward_channels=[64, 64, 64, 64],
        reward_lstm_dim=1024,
        actor_channels=[64, 64, 128, 128],
        actor_lstm_dim=1024,
    ),
}


@dataclass
class DiamondConfig(SerializableConfigMixin):
    # Preset selection (overrides manual model config if set)
    preset: Optional[str] = None  # "small", "medium", "large", or None

    def __post_init__(self) -> None:
        if self.preset and self.preset in MODEL_PRESETS:
            p = MODEL_PRESETS[self.preset]
            self.diffusion_channels = p.diffusion_channels
            self.diffusion_res_blocks = p.diffusion_res_blocks
            self.diffusion_cond_dim = p.diffusion_cond_dim
            self.reward_channels = list(
                p.reward_channels
            )  # convert tuple for dataclass
            self.reward_lstm_dim = p.reward_lstm_dim
            self.actor_channels = list(p.actor_channels)
            self.actor_lstm_dim = p.actor_lstm_dim

    # Environment
    game: str = "Breakout-v5"
    seed: int = 0
    obs_size: int = 64
    frameskip: int = 4
    max_noop: int = 30
    terminate_on_life_loss: bool = True
    reward_clip: List[int] = field(default_factory=lambda: [-1, 0, 1])
    # Standard Atari agent-step cap (108k frames / frameskip 4).
    max_episode_steps: int = 27000

    # Frame stacking (observation conditioning)
    num_conditioning_frames: int = 4

    # Diffusion model (Dθ) - used if preset is None
    diffusion_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    diffusion_res_blocks: int = 2
    diffusion_cond_dim: int = 256

    # EDM hyperparameters
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: int = 7
    p_mean: float = -0.4
    p_std: float = 1.2

    # Diffusion sampling
    sampling_method: str = "euler"
    num_sampling_steps: int = 3

    # Reward/Termination model (Rψ) - used if preset is None
    reward_channels: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    reward_res_blocks: int = 2
    reward_cond_dim: int = 128
    reward_lstm_dim: int = 512
    burn_in_length: int = 4

    # RL Agent (actor-critic) - used if preset is None
    actor_channels: List[int] = field(default_factory=lambda: [32, 32, 64, 64])
    actor_res_blocks: int = 1
    actor_lstm_dim: int = 512

    # Training
    num_epochs: int = 1000
    training_steps_per_epoch: int = 400
    batch_size: int = 32
    environment_steps_per_epoch: int = 100
    epsilon_greedy: float = 0.01
    data_loader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # RL hyperparameters
    imagination_horizon: int = 15
    discount_factor: float = 0.985
    entropy_weight: float = 0.001
    lambda_returns: float = 0.95

    # Optimization
    learning_rate: float = 1e-4
    adam_epsilon: float = 1e-8
    weight_decay_diffusion: float = 1e-2
    weight_decay_reward: float = 1e-2
    weight_decay_actor: float = 0.0
    use_amp: bool = True

    # Device
    device: str = field(default_factory=get_default_device)

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100

    # Where save_checkpoint writes when it is handed a bare filename such as
    # "checkpoint_3.pt". Sweeps that redirect their output (see
    # scripts/run_all_models.sh --ckpt-root) need this; without it the trainer
    # always wrote under ./checkpoints/diamond relative to the working
    # directory, and the caller had no way to find what it had just written.
    checkpoint_dir: str = "checkpoints/diamond"

    # Stop once evaluation return stops improving, rather than running the full
    # epoch budget. Off by default: the published configuration is a fixed-length
    # run, and stopping early would change what the numbers mean. When on, the
    # epoch count becomes a ceiling. Return, not loss, is the signal here -- the
    # world-model loss keeps falling long after the agent stops getting better.
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4


# Atari 100k games and reference scores, from the shared benchmark tables
# (simulusion.benchmarks), keyed the way DIAMOND names games.
ATARI_100K_GAMES = [atari_env_id(g).removeprefix("ALE/") for g in _GAMES]
HUMAN_SCORES = {g: ATARI_HUMAN_SCORES[atari_game(g)] for g in ATARI_100K_GAMES}
RANDOM_SCORES = {g: ATARI_RANDOM_SCORES[atari_game(g)] for g in ATARI_100K_GAMES}
