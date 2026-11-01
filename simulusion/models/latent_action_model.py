import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Literal

from simulusion.vision.vq_layer import VectorQuantizer
from simulusion.blocks.st_transformer import STTransformer
from simulusion.export import ExportableAgentMixin


class LatentActionModel(ExportableAgentMixin, nn.Module):
    """Latent Action Model (LAM) for unsupervised action learning.

    Learns discrete latent actions from unlabeled video frames using a VQ-VAE
    based objective. The model infers latent actions between frames that encode
    the most meaningful changes for future frame prediction.

    Based on Genie paper - learns actions without action labels from Internet videos.

    Components:
    - Encoder: Takes all previous frames x1:t and next frame x_t+1 → outputs latent actions
    - Decoder: Takes previous frames x1:t-1 and latent actions a1:t-1 → predicts next frame x_t
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        encoder_dim: int = 256,
        decoder_dim: int = 512,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
        num_heads: int = 8,
        patch_size: int = 16,
        vocab_size: int = 8,
        embedding_dim: int = 32,
        commitment_weight: float = 1.0,
        action_pooling: Literal["mean", "windowed_attention"] = "mean",
        window_attention_heads: int = 1,
    ):
        super().__init__()
        if action_pooling not in {"mean", "windowed_attention"}:
            raise ValueError(
                "action_pooling must be either 'mean' or 'windowed_attention'"
            )
        if action_pooling == "windowed_attention":
            if window_attention_heads < 1:
                raise ValueError("window_attention_heads must be at least 1")
            if embedding_dim % window_attention_heads != 0:
                raise ValueError(
                    "embedding_dim must be divisible by window_attention_heads"
                )
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.action_pooling = action_pooling

        num_patches = (image_size // patch_size) ** 2

        # ===== ENCODER =====
        # Takes x1:t and x_t+1, outputs latent actions at each timestep
        # Uses ST-Transformer as per Genie paper
        # Input is T+1 frames (T previous + 1 next), output is T latent actions
        self.patch_embed = nn.Conv2d(
            in_channels,
            encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Encoder processes T frames (x1:t + x_{t+1} combined), need T positions
        # Support up to max_seq_len frames (should be set to max expected frames)
        self.max_seq_len = max(num_frames + 1, 32)  # At least 32 for flexibility
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len * num_patches, encoder_dim)
        )
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)

        # ST-Transformer for encoder (per paper - uses spatiotemporal attention)
        self.encoder = STTransformer(
            num_frames=self.max_seq_len,
            num_patches_per_frame=num_patches,
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        self.to_vq_embedding = nn.Linear(encoder_dim, embedding_dim)
        self.window_attention = (
            nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=window_attention_heads,
                batch_first=True,
            )
            if action_pooling == "windowed_attention"
            else None
        )

        self.vq = VectorQuantizer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            commitment_weight=commitment_weight,
        )

        # ===== DECODER =====
        # Takes x1:t-1 and a1:t-1, predicts x_t
        # Uses masked frames (all but first) to force action usage
        # Uses ST-Transformer as per Genie paper
        self.decoder_patch_embed = nn.Conv2d(
            in_channels,
            decoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Decoder processes T frames (x1:t), outputs T predictions
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len * num_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder uses action embeddings as additive conditioning
        self.decoder_action_proj = nn.Linear(embedding_dim, decoder_dim)

        # ST-Transformer for decoder (per paper - uses spatiotemporal attention)
        self.decoder = STTransformer(
            num_frames=self.max_seq_len,
            num_patches_per_frame=num_patches,
            dim=decoder_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        # Output projection to reconstruct pixels
        self.decoder_out = nn.Linear(decoder_dim, in_channels * patch_size * patch_size)
        nn.init.xavier_uniform_(self.decoder_out.weight)
        nn.init.zeros_(self.decoder_out.bias)

    def _pool_windowed_attention(
        self, x_t_embed: torch.Tensor, x_next_embed: torch.Tensor
    ) -> torch.Tensor:
        """Apply length-2 temporal attention, then mean-pool the attended tokens.

        Args:
            x_t_embed: Current timestep embeddings with shape (B, N, embedding_dim).
            x_next_embed: Next timestep embeddings with shape (B, N, embedding_dim).

        Returns:
            Pooled action embeddings with shape (B, embedding_dim).
        """
        if self.window_attention is None:
            raise RuntimeError("window attention module is not initialized")

        B, N, E = x_t_embed.shape
        windows = torch.stack([x_t_embed, x_next_embed], dim=2)
        windows = windows.reshape(B * N, 2, E)
        attended_windows, _ = self.window_attention(
            windows,
            windows,
            windows,
            need_weights=False,
        )
        attended_windows = attended_windows.reshape(B, N, 2, E)
        return attended_windows.mean(dim=(1, 2))

    def encode(
        self, x_prev: torch.Tensor, x_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode frames to latent actions.

        Args:
            x_prev: Previous frames (B, C, T, H, W)
            x_next: Next frame (B, C, H, W)

        Returns:
            latent_actions: Discrete latent action indices (B, T)
            z_q: Quantized embeddings (B, T, embedding_dim)
        """
        latent_actions, z_q, _ = self._encode(x_prev, x_next)
        return latent_actions, z_q

    def _encode(
        self, x_prev: torch.Tensor, x_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode frames to latent actions, also returning the VQ loss.

        Same as :meth:`encode`, but additionally returns the mean VQ loss
        (codebook + commitment) measured between the *encoder outputs* and the
        codebook. This is the objective that trains the action codebook; it must
        be taken from the quantization of the encoder output, not from a second
        quantization of the already-quantized embeddings.

        Returns:
            latent_actions: Discrete latent action indices (B, T)
            z_q: Quantized embeddings (B, T, embedding_dim)
            vq_loss: Scalar mean VQ loss across timesteps
        """
        B, C, T, H, W = x_prev.shape

        # Concatenate all frames (x1:t and x_{t+1}) as per paper
        x_all = torch.cat([x_prev, x_next.unsqueeze(2)], dim=2)  # (B, C, T+1, H, W)

        # Patch embed
        x = x_all.permute(0, 2, 1, 3, 4).reshape(B * (T + 1), C, H, W)
        x = self.patch_embed(x)
        _, C_enc, H_enc, W_enc = x.shape
        num_patches = H_enc * W_enc

        # Reshape for ST-Transformer: (B, T+1, N, C) then flatten to (B, (T+1)*N, C)
        x = x.reshape(B, (T + 1), num_patches, C_enc)

        # Add positional embeddings
        x = x.reshape(B, (T + 1) * num_patches, C_enc)
        x = x + self.encoder_pos_embed[:, : (T + 1) * num_patches, :]

        # ST-Transformer forward (expects B, T*N, C format)
        x = self.encoder(x)  # (B, (T+1)*N, C)

        # Reshape back to (B, T+1, N, C) and keep the LAST T positions.
        #
        # Paper 2.1: the encoder sees "all previous frames x_1:t as well as the
        # next frame x_t+1" and emits the action between them. Temporal attention
        # is causal, so output position i has seen frames 1..i+1 (0-indexed
        # 0..i). The action between frame t and t+1 therefore has to be read at
        # position t+1, not position t -- taking the first T positions yields
        # actions that never observed the frame they are supposed to describe,
        # making them independent of the transition entirely.
        x = x.reshape(B, T + 1, num_patches, C_enc)
        x = x[:, 1:, :, :]  # (B, T, N, C)

        # Quantize each timestep
        z_all = []
        indices_all = []
        vq_loss_terms = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, C)

            x_t_embed = self.to_vq_embedding(x_t)  # (B, N, embedding_dim)

            if self.action_pooling == "windowed_attention":
                next_t = min(t + 1, T - 1)
                x_next_t = x[:, next_t, :, :]  # (B, N, C)
                x_next_t_embed = self.to_vq_embedding(x_next_t)  # (B, N, embedding_dim)
                x_t_pooled = self._pool_windowed_attention(
                    x_t_embed, x_next_t_embed
                )  # (B, embedding_dim)
            else:
                # Pool across spatial dimension
                x_t_pooled = x_t_embed.mean(dim=1)  # (B, embedding_dim)

            z_q_t, indices_t, vq_info_t = self.vq(x_t_pooled)
            z_all.append(z_q_t)
            indices_all.append(indices_t)
            vq_loss_terms.append(vq_info_t["vq_loss"])

        z_q = torch.stack(z_all, dim=1)  # (B, T, embedding_dim)
        indices = torch.stack(indices_all, dim=1)  # (B, T, 1)

        latent_actions = indices.squeeze(-1)  # (B, T)

        # Mean VQ loss over timesteps, measured on the encoder outputs (the term
        # that actually trains the codebook + commitment).
        vq_loss = torch.stack(vq_loss_terms).mean()

        return latent_actions, z_q, vq_loss

    def decode(
        self,
        x_prev: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent actions to predict next frame.

        Args:
            x_prev: Previous frames (B, C, T, H, W) - will mask all but first
            z_q: Quantized action embeddings (B, T-1, embedding_dim)

        Returns:
            predicted_next_frame: (B, C, H, W)
        """
        B, C, T, H, W = x_prev.shape

        # Mask all frames except the first - forces decoder to use actions
        x_masked = x_prev.clone()
        for t in range(1, T):
            x_masked[:, :, t, :, :] = 0

        # Patch embed
        x = x_masked.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.decoder_patch_embed(x)
        _, C_dec, H_dec, W_dec = x.shape
        num_patches = H_dec * W_dec

        # Reshape for ST-Transformer: (B, T, N, C) then flatten to (B, T*N, C)
        x = x.reshape(B, T, num_patches, C_dec)

        # Project action embeddings to decoder dimension
        action_cond = self.decoder_action_proj(z_q)  # (B, T-1, decoder_dim)

        # Add action embeddings to frames (align T-1 actions with T-1 future frames)
        # Frames 1 to T-1 get action conditioning, frame 0 is the prompt
        action_expanded = torch.zeros(B, T, num_patches, C_dec, device=x.device)
        action_expanded[:, 1:, :, :] = action_cond.unsqueeze(2)

        # Combine frame embeddings with action conditioning
        x = x + action_expanded

        # Add positional embeddings and flatten for ST-Transformer
        x = x.reshape(B, T * num_patches, C_dec)
        x = x + self.decoder_pos_embed[:, : T * num_patches, :]

        # ST-Transformer forward
        x = self.decoder(x)  # (B, T*N, C)

        # Reshape back to (B, T, N, C)
        x = x.reshape(B, T, num_patches, C_dec)

        # Use only last frame (predicted next frame)
        x = x[:, -1, :, :]  # (B, N, C)

        # Project to pixel space
        x = self.decoder_out(x)  # (B, N, C*patch_size*patch_size)

        # Reshape to image
        x = x.reshape(B, H_dec, W_dec, C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, H_dec, patch_size, W_dec, patch_size)
        x = x.reshape(B, C, H_dec * self.patch_size, W_dec * self.patch_size)

        # Resize to original image size if needed
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return torch.sigmoid(x)

    def forward(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode to get actions, decode to reconstruct.

        Args:
            x_prev: Previous frames (B, C, T, H, W)
            x_next: Next frame (B, C, H, W)

        Returns:
            Dictionary with losses and outputs
        """
        # Encode to get latent actions and the VQ (codebook + commitment) loss
        # measured on the encoder outputs.
        latent_actions, z_q, vq_loss = self._encode(x_prev, x_next)

        # Use actions from timestep 1 to T-1 (T-1 actions for T-1 to predict T)
        # latent_actions has shape (B, T) from encoding T frames
        # We need T-1 actions to predict T-1 future frames
        num_actions = z_q.shape[1] - 1  # T-1
        z_q_for_decode = z_q[:, :num_actions, :] if num_actions > 0 else z_q

        # Decode to reconstruct next frame
        # decode expects x_prev with T frames and z_q with T-1 actions
        reconstructed = self.decode(x_prev, z_q_for_decode)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x_next)

        return {
            "latent_actions": latent_actions,
            "z_q": z_q,
            "reconstructed": reconstructed,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }


def create_latent_action_model(
    num_frames: int = 16,
    image_size: int = 64,
    in_channels: int = 3,
    encoder_dim: int = 256,
    decoder_dim: int = 512,
    encoder_depth: int = 4,
    decoder_depth: int = 4,
    num_heads: int = 8,
    patch_size: int = 16,
    vocab_size: int = 8,
    embedding_dim: int = 32,
    action_pooling: Literal["mean", "windowed_attention"] = "mean",
    window_attention_heads: int = 1,
) -> LatentActionModel:
    """Factory function to create a Latent Action Model."""
    return LatentActionModel(
        num_frames=num_frames,
        image_size=image_size,
        in_channels=in_channels,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        patch_size=patch_size,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        action_pooling=action_pooling,
        window_attention_heads=window_attention_heads,
    )
