import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from simulusion.vision.iris_encoder import (
    ATTENTION_RESOLUTIONS,
    IRISEncoder,
    SelfAttentionBlock,
)
from simulusion.export import ExportableAgentMixin


class IRISDecoder(ExportableAgentMixin, nn.Module):
    """CNN Decoder for IRIS discrete autoencoder.

    Decodes discrete tokens back into image observations.
    Uses transposed convolutions to upsample from 4x4 to 64x64.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 512,
        base_channels: int = 64,
        out_channels: int = 3,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
        num_residual_blocks: int = 2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.frame_shape = frame_shape
        self.out_channels = out_channels

        # Input projection: step the token embedding (d = 512, Table 3) down to
        # the convolutional width straight away. Table 2 gives a single "Channels
        # in convolutions: 64" for both halves of the autoencoder, so the
        # bottleneck stack below runs at 64 -- not at the embedding dimension,
        # which put ~5M parameters into two residual blocks alone.
        self.input_proj = nn.Conv2d(embedding_dim, base_channels, 1)

        # Residual blocks before upsampling
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )

        # Upsampling blocks (4 -> 8 -> 16 -> 32 -> 64), each followed by its own
        # residual stack. Table 2 gives the encoder's "2 residual blocks per
        # layer" and states the same hyperparameters apply to the decoder.
        self.upsample_blocks = nn.ModuleList()
        self.layer_residuals = nn.ModuleList()

        # Constant width throughout, mirroring the encoder.
        upsample_specs = [
            # (in, mid, out) for 4->8, 8->16, 16->32, 32->64
            (base_channels, base_channels, base_channels),
            (base_channels, base_channels, base_channels),
            (base_channels, base_channels, base_channels),
            (base_channels, base_channels, base_channels),
        ]
        for in_ch, mid_ch, out_ch in upsample_specs:
            self.upsample_blocks.append(UpsampleBlock(in_ch, mid_ch, out_ch))
            self.layer_residuals.append(
                nn.Sequential(
                    *[ResidualBlock(out_ch) for _ in range(num_residual_blocks)]
                )
            )

        # Table 2: "Self-attention layers at resolution 8 / 16", and the encoder's
        # hyperparameters apply to the decoder too. Resolutions are derived from
        # the upsampling schedule rather than hardcoded to stage indices.
        start_resolution = frame_shape[1] // (2 ** len(upsample_specs))
        self.attention_at_stage: dict[int, str] = {}
        self.attentions = nn.ModuleDict()
        for stage_idx, (_in_ch, _mid_ch, out_ch) in enumerate(upsample_specs):
            resolution = start_resolution * (2 ** (stage_idx + 1))
            if resolution in ATTENTION_RESOLUTIONS:
                name = f"attn_{resolution}"
                self.attentions[name] = SelfAttentionBlock(out_ch)
                self.attention_at_stage[stage_idx] = name

        # Final output projection
        self.output_proj = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode tokens to images.

        Args:
            z: Token embeddings (B, C, H, W) - e.g., (B, 512, 4, 4)

        Returns:
            reconstructed: Reconstructed images (B, C, H, W) - e.g., (B, 3, 64, 64)
        """
        # Project input
        h = self.input_proj(z)

        # Residual blocks
        h = self.residual_blocks(h)

        # Upsampling, each stage followed by its residual stack, with
        # self-attention at the 8x8 and 16x16 resolutions (Table 2).
        for stage_idx, (upsample_block, residuals) in enumerate(
            zip(self.upsample_blocks, self.layer_residuals)
        ):
            h = residuals(upsample_block(h))
            name = self.attention_at_stage.get(stage_idx)
            if name is not None:
                h = self.attentions[name](h)

        # Final output
        h = self.output_proj(h)

        # Ensure output matches frame shape
        _, _, out_h, out_w = h.shape
        target_h, target_w = self.frame_shape[1], self.frame_shape[2]

        if out_h != target_h or out_w != target_w:
            h = F.interpolate(
                h, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        return h

    def decode_from_embeddings(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Decode flattened token embeddings to images.

        Args:
            z_flat: Flattened tokens (B, H*W, C) or (B, C, H, W)

        Returns:
            Reconstructed images
        """
        if z_flat.dim() == 3:  # (B, H*W, C)
            B, HW, C = z_flat.shape
            H = W = int(HW**0.5)
            z = z_flat.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            z = z_flat

        return self.forward(z)

    def decode_from_indices(
        self, indices: torch.Tensor, codebook: nn.Embedding
    ) -> torch.Tensor:
        """Decode discrete token indices into images.

        The codebook must be passed in explicitly -- it is the quantizer's table
        (``IRISEncoder.quantizer.codebook``), the only one the commitment and
        reconstruction losses train. This decoder previously owned a private
        ``index_to_embedding`` table that no objective ever touched, so decoding
        through it returned noise that looked plausible enough to go unnoticed.

        Args:
            indices: Tensor of shape (B, H, W) or (B, H*W) containing integer
                token indices in the range [0, vocab_size).
            codebook: The encoder's quantizer codebook.

        Returns:
            Reconstructed images (B, C, H, W)
        """
        if indices.dim() == 3:
            B, H, W = indices.shape
            flat = indices.reshape(B, -1)
        elif indices.dim() == 2:
            B, HW = indices.shape
            H = W = int(HW**0.5)
            flat = indices
        else:
            raise ValueError("indices must be shape (B, H, W) or (B, H*W)")

        # (B, HW, C)
        emb = F.embedding(flat, codebook.weight)
        # convert to (B, C, H, W)
        emb = emb.permute(0, 2, 1).reshape(B, self.embedding_dim, H, W)
        return self.forward(emb)


class UpsampleBlock(nn.Module):
    """Upsampling block with optional residual connection."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        )

        # Skip connection projection if needed
        # Use the broad Module type because we may assign either Identity or Conv2d here.
        self.skip: nn.Module = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.skip(x) + self.block(x)


class ResidualBlock(nn.Module):
    """Residual block for decoder."""

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


class DiscreteAutoencoder(nn.Module):
    """Complete Discrete Autoencoder combining encoder and decoder.

    Used for training the VQVAE component of IRIS.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        tokens_per_frame: int = 16,
        embedding_dim: int = 512,
        base_channels: int = 64,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__()

        self.encoder = IRISEncoder(
            vocab_size=vocab_size,
            tokens_per_frame=tokens_per_frame,
            embedding_dim=embedding_dim,
            in_channels=frame_shape[0],
            base_channels=base_channels,
            frame_shape=frame_shape,
        )

        self.decoder = IRISDecoder(
            embedding_dim=embedding_dim,
            base_channels=32,  # decoder uses smaller channels
            out_channels=frame_shape[0],
            frame_shape=frame_shape,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Full encode-decode forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            reconstruction: Reconstructed images
            indices: Token indices (B, H', W')
            loss_dict: Dictionary with loss components
        """
        z_q, indices, vq_loss = self.encoder(x)

        # Decode with gradients flowing through reconstruction path
        reconstruction_st = self.decoder(z_q)

        # Compute reconstruction loss
        recon_loss = F.l1_loss(reconstruction_st, x)

        # Combine losses
        loss = recon_loss + vq_loss["vq_loss"]

        loss_dict = {
            "reconstruction": recon_loss,
            "vq": vq_loss["vq_loss"],
            "perplexity": vq_loss["perplexity"],
            "total": loss,
        }

        return reconstruction_st, indices, loss_dict

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to token indices."""
        return self.encoder.encode_to_indices(x)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to images."""
        embeddings = self.encoder.decode_from_indices(indices)
        return self.decoder(embeddings)
