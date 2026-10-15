from dataclasses import dataclass

from simulusion.configs.serialization import SerializableConfigMixin


@dataclass
class IRISConfig(SerializableConfigMixin):
    """Configuration for IRIS (Imagination with auto-Regression over an Inner Speech)

    Based on paper: "Transformers are Sample-Efficient World Models"
    Implements discrete autoencoder + autoregressive Transformer for sample-efficient RL.
    """

    # === Discrete Autoencoder (VQVAE) ===
    frame_height: int = 64
    frame_width: int = 64
    frame_channels: int = 3

    vocab_size: int = 512
    tokens_per_frame: int = 16
    token_embedding_dim: int = 512

    encoder_channels: int = 64
    encoder_layers: int = 4
    encoder_residual_blocks: int = 2

    # Table 2 gives 64 channels in convolutions and notes the encoder's
    # hyperparameters apply to the decoder as well.
    decoder_depth: int = 64

    # Paper A.1 weights the L1, commitment and perceptual terms equally.
    reconstruction_weight: float = 1.0
    commitment_weight: float = 1.0
    perceptual_weight: float = 1.0
    # Number of VGG16 conv blocks compared by the perceptual loss. LPIPS uses 5.
    perceptual_blocks: int = 5
    # Optional path to LPIPS's learned per-channel linear weights. Without them
    # the loss uses uniform channel weights (see vision/perceptual_loss.py).
    perceptual_linear_weights: str = ""
    # "ema" uses an EMA-updated codebook; "gradient" reproduces the paper's
    # objective, where the codebook is trained by the codebook loss term.
    quantizer: str = "gradient"

    # === Transformer (World Model) ===
    transformer_timesteps: int = 20
    transformer_embed_dim: int = 256
    transformer_layers: int = 10
    transformer_heads: int = 4
    transformer_dropout: float = 0.1

    # Reward handling. Atari returns unbounded integer rewards (thousands in
    # Krull or UpNDown), which would dominate the value scale and the lambda
    # return. The standard Atari convention -- and what IRIS uses -- is to take
    # the sign, making the target categorical over {-1, 0, +1}. Paper 2.2 allows
    # "a mean-squared error loss or a cross-entropy loss for the reward
    # predictor, depending on the reward function"; the categorical target calls
    # for cross-entropy. Set reward_transform="none" + reward_loss="mse" for
    # environments with meaningful continuous rewards.
    reward_transform: str = "sign"
    reward_loss: str = "cross_entropy"

    # === Actor-Critic === (paper Appendix B, Table 6)
    imagination_horizon: int = 20
    # Paper A.3: "we burn-in the 20 previous frames to initialize the hidden
    # state" before starting imagination.
    burn_in_length: int = 20
    discount: float = 0.995
    td_lambda: float = 0.95
    entropy_coef: float = 0.001

    actor_hidden_size: int = 512
    actor_layers: int = 1

    # NOTE: there is no separate value network. Paper A.3: "the weights of the
    # actor and critic are shared except for the last layer", so the critic is a
    # linear head on the shared CNN+LSTM trunk. Any value_hidden_size /
    # value_layers setting would be meaningless and is therefore not offered.

    # === Training ===
    total_epochs: int = 600
    collection_epochs: int = 500
    env_steps_per_epoch: int = 200
    training_steps_per_epoch: int = 200

    # Per-component gradient steps per epoch. The paper (Table 5) takes
    # ``training_steps_per_epoch`` steps for every component. Even with KV
    # caching, one actor-critic step means H sequential frame generations of K
    # tokens each, so it is far more expensive than an autoencoder step; these
    # are separately tunable to keep epoch time manageable. Raise them toward
    # ``training_steps_per_epoch`` for a faithful run if compute allows.
    transformer_steps_per_epoch: int = 200
    actor_critic_steps_per_epoch: int = 200

    model_learning_rate: float = 1e-4
    actor_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    grad_clip_norm: float = 10.0
    use_amp: bool = True
    gradient_checkpointing: bool = True

    collect_epsilon: float = 0.01
    eval_temperature: float = 0.5
    # Sampling temperature used when acting in the real environment. Paper
    # Appendix H lowers this to 0.01 for Freeway; see FREEWAY_COLLECT_TEMPERATURE
    # in simulusion.training.train_iris.
    collect_temperature: float = 1.0

    start_autoencoder_after: int = 5
    start_transformer_after: int = 25
    start_actor_critic_after: int = 50

    autoencoder_batch_size: int = 256
    transformer_batch_size: int = 64
    actor_critic_batch_size: int = 64

    # === Atari 100k Benchmark ===
    atari_100k: bool = True
    max_env_steps: int = 100000

    # === Environment ===
    env_backend: str = "gym"
    env: str = "ALE/Pong-v5"
    action_repeat: int = 4
    # The Atari 100k protocol (and the IRIS paper) evaluates without sticky
    # actions. make_atari_env defaults to 0.25, which makes the task materially
    # harder and is not comparable to the published numbers.
    repeat_action_probability: float = 0.0
    # Per-episode agent-step cap, applied by the environment's TimeLimit. 27000
    # is the standard Atari limit and what the paper's numbers assume; it is a
    # field rather than a literal so short smoke runs can bound collection and
    # evaluation, which otherwise play out full episodes.
    max_episode_steps: int = 27000

    # === Logging ===
    log_interval: int = 1000
    eval_episodes: int = 100
    checkpoint_interval: int = 50

    # Stop once evaluation return stops improving, rather than running the full
    # epoch budget. Off by default: the published configuration is a fixed-length
    # run, and stopping early would change what the numbers mean. When on, the
    # epoch count becomes a ceiling. Return, not loss, is the signal here -- the
    # world-model loss keeps falling long after the agent stops getting better.
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4

    def get_frame_shape(self) -> tuple:
        return (self.frame_channels, self.frame_height, self.frame_width)

    def get_autoencoder_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "tokens_per_frame": self.tokens_per_frame,
            "embedding_dim": self.token_embedding_dim,
            "encoder_channels": self.encoder_channels,
            "encoder_layers": self.encoder_layers,
            "encoder_residual_blocks": self.encoder_residual_blocks,
            "decoder_depth": self.decoder_depth,
            "frame_shape": self.get_frame_shape(),
            "reconstruction_weight": self.reconstruction_weight,
            "commitment_weight": self.commitment_weight,
            "perceptual_weight": self.perceptual_weight,
        }

    def get_transformer_config(self) -> dict:
        return {
            "timesteps": self.transformer_timesteps,
            "embed_dim": self.transformer_embed_dim,
            "layers": self.transformer_layers,
            "heads": self.transformer_heads,
            "dropout": self.transformer_dropout,
            "vocab_size": self.vocab_size,
            "tokens_per_frame": self.tokens_per_frame,
            "action_size": None,
        }

    def get_rl_config(self) -> dict:
        return {
            "imagination_horizon": self.imagination_horizon,
            "discount": self.discount,
            "td_lambda": self.td_lambda,
            "entropy_coef": self.entropy_coef,
            "actor_hidden_size": self.actor_hidden_size,
            "actor_layers": self.actor_layers,
            "burn_in_length": self.burn_in_length,
            "frame_shape": self.get_frame_shape(),
        }
