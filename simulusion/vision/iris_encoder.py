import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from simulusion.vision.vq_layer import VectorQuantizer, VectorQuantizerEMA
from simulusion.export import ExportableAgentMixin

# Paper Table 2: "Self-attention layers at resolution 8 / 16", for both the
# encoder and the decoder.
ATTENTION_RESOLUTIONS = (8, 16)


class IRISEncoder(ExportableAgentMixin, nn.Module):
    """CNN Encoder for IRIS discrete autoencoder.

    Encodes image observations into latent features, which are then quantized
    into discrete tokens using the VectorQuantizer.

    Architecture:
        - 4 convolutional layers with residual blocks
        - Self-attention at 8x8 and 16x16 resolutions
        - Vector quantization to produce discrete tokens
    """

    def __init__(
        self,
        vocab_size: int = 512,
        tokens_per_frame: int = 16,
        embedding_dim: int = 512,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        num_residual_blocks: int = 2,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
        commitment_weight: float = 1.0,
        quantizer: str = "gradient",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame
        self.embedding_dim = embedding_dim

        # CNN encoder body. Table 2 lists 4 layers with 2 residual blocks *per
        # layer*, so each downsampling convolution is followed by its own stack
        # of residual blocks rather than a single stack at the end.
        self.conv_blocks = nn.ModuleList()
        self.layer_residuals = nn.ModuleList()
        in_ch = in_channels

        # Each layer halves the resolution. With the paper's 4 layers a 64x64
        # frame reaches 4x4 = 16 tokens (Table 3).
        #
        # The width stays constant at `base_channels`. Table 2 lists a single
        # "Channels in convolutions: 64" for the whole encoder, and the VQGAN
        # config IRIS inherits from uses an all-ones channel multiplier. Doubling
        # per layer (64/128/256/512, as this did) inflates the autoencoder far
        # past the paper's 30M-parameter total and changes what the token
        # bottleneck is actually compressing.
        channels = [base_channels] * num_layers
        spatial = frame_shape[1] // (2**num_layers)
        if spatial * spatial != tokens_per_frame:
            raise ValueError(
                f"{num_layers} layers on a {frame_shape[1]}x{frame_shape[2]} frame "
                f"gives {spatial}x{spatial}={spatial * spatial} tokens, but "
                f"tokens_per_frame is {tokens_per_frame}. These must agree."
            )
        self.expected_spatial_size = spatial

        for out_ch in channels:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.ReLU(),
                )
            )
            self.layer_residuals.append(
                nn.Sequential(
                    *[ResidualBlock(out_ch) for _ in range(num_residual_blocks)]
                )
            )
            in_ch = out_ch

        # Table 2: "Self-attention layers at resolution 8 / 16". Derive which
        # layer emits each resolution rather than hardcoding indices, so this
        # stays correct if num_layers or the frame size changes.
        self.attention_at_layer: dict[int, str] = {}
        self.attentions = nn.ModuleDict()
        for layer_idx, out_ch in enumerate(channels):
            resolution = frame_shape[1] // (2 ** (layer_idx + 1))
            if resolution in ATTENTION_RESOLUTIONS:
                name = f"attn_{resolution}"
                self.attentions[name] = SelfAttentionBlock(out_ch)
                self.attention_at_layer[layer_idx] = name

        # Project to embedding dimension
        self.projection = nn.Conv2d(channels[-1], embedding_dim, 1)

        # Vector quantizer. "gradient" reproduces the paper's objective (A.1),
        # where the codebook is trained by the codebook-loss term; "ema" uses
        # exponential moving averages instead, which is often steadier but is
        # not what the paper specifies.
        self.quantizer: VectorQuantizer | VectorQuantizerEMA
        if quantizer == "ema":
            self.quantizer = VectorQuantizerEMA(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                commitment_weight=commitment_weight,
            )
        elif quantizer == "gradient":
            self.quantizer = VectorQuantizer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                commitment_weight=commitment_weight,
            )
        else:
            raise ValueError(
                f"Unknown quantizer {quantizer!r}; expected 'gradient' or 'ema'."
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode images to discrete tokens.

        Args:
            x: Input images (B, C, H, W) - should be 64x64

        Returns:
            z_q: Quantized tokens (B, C, H', W')
            indices: Token indices (B, H', W')
            vq_loss: Dictionary with VQ loss components
        """
        # CNN encoding: each conv block halves the resolution, is followed by its
        # own residual stack (Table 2: 2 residual blocks per layer), and by
        # self-attention at the 8x8 and 16x16 resolutions.
        h = x
        for layer_idx, (conv, residuals) in enumerate(
            zip(self.conv_blocks, self.layer_residuals)
        ):
            h = residuals(conv(h))
            name = self.attention_at_layer.get(layer_idx)
            if name is not None:
                h = self.attentions[name](h)

        # Project to embedding dimension
        h = self.projection(h)

        # Quantize
        z_q, indices, vq_loss = self.quantizer(h)

        return z_q, indices, vq_loss

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Encode directly to token indices (for world model)."""
        with torch.no_grad():
            _, indices, _ = self.forward(x)
        return indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to embeddings (for decoder)."""
        return self.quantizer.decode_indices(indices)


class ResidualBlock(nn.Module):
    """Residual block for encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention block for encoder.

    Applies spatial self-attention to capture long-range dependencies.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Compute Q, K, V
        q = self.query(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = self.key(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        v = self.value(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        # Apply fused scaled dot-product attention over spatial tokens.
        out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        ).squeeze(1)  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        # Residual connection with learned weight
        return x + self.gamma * out
