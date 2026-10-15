from dataclasses import dataclass, field
from typing import List, Optional


from simulusion.configs.serialization import SerializableConfigMixin


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


# Atari 100k benchmark games
ATARI_100K_GAMES = [
    "Alien-v5",
    "Amidar-v5",
    "Assault-v5",
    "Asterix-v5",
    "BankHeist-v5",
    "BattleZone-v5",
    "Boxing-v5",
    "Breakout-v5",
    "ChopperCommand-v5",
    "CrazyClimber-v5",
    "DemonAttack-v5",
    "Freeway-v5",
    "Frostbite-v5",
    "Gopher-v5",
    "Hero-v5",
    "Jamesbond-v5",
    "Kangaroo-v5",
    "Krull-v5",
    "KungFuMaster-v5",
    "MsPacman-v5",
    "Pong-v5",
    "PrivateEye-v5",
    "Qbert-v5",
    "RoadRunner-v5",
    "Seaquest-v5",
    "UpNDown-v5",
]


# Human normalized scores (for evaluation)
HUMAN_SCORES = {
    "Alien-v5": 7127.7,
    "Amidar-v5": 1719.5,
    "Assault-v5": 742.0,
    "Asterix-v5": 8503.3,
    "BankHeist-v5": 753.1,
    "BattleZone-v5": 37187.5,
    "Boxing-v5": 12.1,
    "Breakout-v5": 30.5,
    "ChopperCommand-v5": 7387.8,
    "CrazyClimber-v5": 35829.4,
    "DemonAttack-v5": 1971.0,
    "Freeway-v5": 29.6,
    "Frostbite-v5": 4334.7,
    "Gopher-v5": 2412.5,
    "Hero-v5": 30826.4,
    "Jamesbond-v5": 302.8,
    "Kangaroo-v5": 3035.0,
    "Krull-v5": 2665.5,
    "KungFuMaster-v5": 22736.3,
    "MsPacman-v5": 6951.6,
    "Pong-v5": 14.6,
    "PrivateEye-v5": 69571.3,
    "Qbert-v5": 13455.0,
    "RoadRunner-v5": 7845.0,
    "Seaquest-v5": 42054.7,
    "UpNDown-v5": 11693.2,
}

RANDOM_SCORES = {
    "Alien-v5": 227.8,
    "Amidar-v5": 5.8,
    "Assault-v5": 222.4,
    "Asterix-v5": 210.0,
    "BankHeist-v5": 14.2,
    "BattleZone-v5": 2360.0,
    "Boxing-v5": 0.1,
    "Breakout-v5": 1.7,
    "ChopperCommand-v5": 811.0,
    "CrazyClimber-v5": 10780.5,
    "DemonAttack-v5": 152.1,
    "Freeway-v5": 0.0,
    "Frostbite-v5": 65.2,
    "Gopher-v5": 257.6,
    "Hero-v5": 1027.0,
    "Jamesbond-v5": 29.0,
    "Kangaroo-v5": 52.0,
    "Krull-v5": 1598.0,
    "KungFuMaster-v5": 258.5,
    "MsPacman-v5": 307.3,
    "Pong-v5": -20.7,
    "PrivateEye-v5": 24.9,
    "Qbert-v5": 163.9,
    "RoadRunner-v5": 11.5,
    "Seaquest-v5": 68.4,
    "UpNDown-v5": 533.4,
}
