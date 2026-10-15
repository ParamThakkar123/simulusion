from dataclasses import dataclass
from typing import Literal

from simulusion.configs.serialization import SerializableConfigMixin


@dataclass
class GenieConfig(SerializableConfigMixin):
    """Configuration for Genie model."""

    num_frames: int = 8
    image_size: int = 32
    in_channels: int = 3

    tokenizer_vocab_size: int = 1024
    tokenizer_embedding_dim: int = 32
    tokenizer_encoder_dim: int = 256
    tokenizer_decoder_dim: int = 512
    tokenizer_encoder_depth: int = 4
    tokenizer_decoder_depth: int = 8
    # 16 is what Genie has always built; these fields used to be ignored, and
    # their former default of 8 never reached the model.
    tokenizer_num_heads: int = 16

    action_vocab_size: int = 8
    action_embedding_dim: int = 32
    action_encoder_dim: int = 256
    # Genie's own default, kept explicit so the field exists rather than being
    # decided inside the constructor where no config could reach it.
    action_decoder_dim: int = 1024
    action_encoder_depth: int = 4
    action_num_heads: int = 16
    action_pooling: Literal["mean", "windowed_attention"] = "mean"
    window_attention_heads: int = 1

    dynamics_dim: int = 512
    dynamics_depth: int = 8
    dynamics_num_heads: int = 8

    batch_size: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 5000
    max_steps: int = 125000

    mask_prob_min: float = 0.5
    mask_prob_max: float = 1.0

    sample_temperature: float = 2.0
    maskgit_steps: int = 25

    # Off by default to match DreamerConfig: enabling autocast changes a run's
    # numerics, so it is an explicit opt-in rather than a silent default.
    # bfloat16 is preferred where the device supports it, and needs no scaler.
    use_amp: bool = False

    # Stop once held-out reconstruction loss stops improving, rather than at a
    # fixed max_steps. Off by default so existing runs keep their exact length;
    # max_steps then bounds the run instead of defining it.
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4
    # Fraction of clips held out to measure that loss on.
    val_split: float = 0.1


@dataclass
class GenieSmallConfig(SerializableConfigMixin):
    """Small configuration for development/testing."""

    num_frames: int = 16
    image_size: int = 64
    in_channels: int = 3

    tokenizer_vocab_size: int = 1024
    tokenizer_embedding_dim: int = 32
    tokenizer_encoder_dim: int = 256
    tokenizer_decoder_dim: int = 512
    tokenizer_encoder_depth: int = 4
    tokenizer_decoder_depth: int = 8
    # 16 is what Genie has always built; these fields used to be ignored, and
    # their former default of 8 never reached the model.
    tokenizer_num_heads: int = 16

    action_vocab_size: int = 8
    action_embedding_dim: int = 32
    action_encoder_dim: int = 512
    action_decoder_dim: int = 1024
    action_encoder_depth: int = 8
    action_num_heads: int = 16
    action_pooling: Literal["mean", "windowed_attention"] = "mean"
    window_attention_heads: int = 1

    dynamics_dim: int = 512
    dynamics_depth: int = 8
    dynamics_num_heads: int = 8

    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 50000

    mask_prob_min: float = 0.5
    mask_prob_max: float = 1.0

    sample_temperature: float = 2.0
    maskgit_steps: int = 25

    # Off by default to match DreamerConfig: enabling autocast changes a run's
    # numerics, so it is an explicit opt-in rather than a silent default.
    # bfloat16 is preferred where the device supports it, and needs no scaler.
    use_amp: bool = False

    # Stop once held-out reconstruction loss stops improving, rather than at a
    # fixed max_steps. Off by default so existing runs keep their exact length;
    # max_steps then bounds the run instead of defining it.
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4
    # Fraction of clips held out to measure that loss on.
    val_split: float = 0.1


@dataclass
class STTransformerConfig(SerializableConfigMixin):
    """Configuration for Spatiotemporal Transformer."""

    num_frames: int = 16
    num_patches_per_frame: int = 256
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass
class VideoTokenizerConfig(SerializableConfigMixin):
    """Configuration for Video Tokenizer."""

    num_frames: int = 16
    image_size: int = 64
    in_channels: int = 3
    encoder_dim: int = 512
    decoder_dim: int = 1024
    encoder_depth: int = 12
    decoder_depth: int = 20
    num_heads: int = 16
    patch_size: int = 4
    vocab_size: int = 1024
    embedding_dim: int = 32
    use_ema: bool = False
    ema_decay: float = 0.99
    commitment_weight: float = 0.25


@dataclass
class LatentActionModelConfig(SerializableConfigMixin):
    """Configuration for Latent Action Model."""

    num_frames: int = 16
    image_size: int = 64
    in_channels: int = 3
    encoder_dim: int = 1024
    encoder_depth: int = 20
    num_heads: int = 16
    patch_size: int = 16
    vocab_size: int = 8
    embedding_dim: int = 32
    commitment_weight: float = 1.0
    action_pooling: Literal["mean", "windowed_attention"] = "mean"
    window_attention_heads: int = 1


@dataclass
class DynamicsModelConfig(SerializableConfigMixin):
    """Configuration for Dynamics Model."""

    num_frames: int = 16
    image_size: int = 64
    vocab_size: int = 1024
    embedding_dim: int = 32
    action_vocab_size: int = 8
    dim: int = 5120
    depth: int = 48
    # 5120 / 40 = 128 per head; the former 36 did not divide 5120.
    num_heads: int = 40
    patch_size: int = 4
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
