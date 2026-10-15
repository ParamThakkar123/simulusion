import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from simulusion.vision.vq_layer import VectorQuantizer, VectorQuantizerEMA
from simulusion.blocks.st_transformer import STTransformer
from simulusion.export import ExportableAgentMixin


class VideoTokenizer(ExportableAgentMixin, nn.Module):
    """Video Tokenizer using VQ-VAE with Spatiotemporal Transformer.

    This is a core component of Genie (Google DeepMind, 2024), used to compress
    raw video frames into discrete latent tokens that can be processed by
    downstream models like the LatentActionModel and DynamicsModel.

    The tokenizer uses Vector Quantized Variational Autoencoder (VQ-VAE) objective
    to learn a discrete codebook of video representations. Unlike standard VQ-VAE,
    this uses a Spatiotemporal (ST) Transformer in both encoder and decoder to
    better capture temporal dynamics in videos.

    **Architecture**

    1. Patch Embedding: Convert (B, C, T, H, W) video to patch tokens
    2. Encoder ST-Transformer: Process spatial-temporal patches
    3. Vector Quantization: Discretize continuous embeddings to codebook entries
    4. Decoder ST-Transformer: Reconstruct video from quantized tokens
    5. Patch Unembedding: Convert tokens back to video frames

    **Key Features**

    - Causal processing: Each frame's encoding only uses previous frames
    - Discrete tokens: Enables autoregressive prediction with latent actions
    - Memory efficient: Uses ST-Transformer instead of full ViT to reduce complexity

    **Usage with Genie**::

        tokenizer = VideoTokenizer(
            num_frames=16,
            image_size=64,
            patch_size=4,
            vocab_size=1024,
            embedding_dim=32
        )
        reconstructed, indices, loss_dict = tokenizer(video_frames)

        # For discrete token input to dynamics model:
        token_embeddings = tokenizer.decode_indices(indices)

    The tokenizer is trained with VQ-VAE objective:
    - Reconstruction loss: MSE between input and reconstructed video
    - VQ loss: Commit to codebook embeddings
    - Commitment loss: Penalizes encoder outputs drifting from codebook

    Reference:
        Genie: Generative Interactive Environments
        Bruce et al., Google DeepMind, 2024 - https://arxiv.org/abs/2402.15391
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        encoder_dim: int = 512,
        decoder_dim: int = 1024,
        encoder_depth: int = 12,
        decoder_depth: int = 20,
        num_heads: int = 16,
        patch_size: int = 4,
        vocab_size: int = 1024,
        embedding_dim: int = 32,
        commitment_weight: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        num_patches = (image_size // patch_size) ** 2

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * num_patches, encoder_dim)
        )
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)

        self.encoder = STTransformer(
            num_frames=num_frames,
            num_patches_per_frame=num_patches,
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        self.to_vq_embedding = nn.Linear(encoder_dim, embedding_dim)

        # vq can be either the EMA variant or the plain VectorQuantizer
        self.vq: Union[VectorQuantizer, VectorQuantizerEMA]

        if use_ema:
            self.vq = VectorQuantizerEMA(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                commitment_weight=commitment_weight,
                ema_decay=ema_decay,
            )
        else:
            self.vq = VectorQuantizer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                commitment_weight=commitment_weight,
            )

        self.from_vq_embedding = nn.Linear(embedding_dim, decoder_dim)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * num_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder = STTransformer(
            num_frames=num_frames,
            num_patches_per_frame=num_patches,
            dim=decoder_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        self.decoder_proj = nn.ConvTranspose2d(
            decoder_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode video to discrete tokens.

        Args:
            x: Video tensor (B, C, T, H, W)

        Returns:
            z_q: Quantized embeddings (B, T, H', W', embedding_dim)
            indices: Token indices (B, T, H', W')
            vq_loss: Dictionary with VQ loss components
        """
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        x = self.patch_embed(x)
        _, C_enc, H_enc, W_enc = x.shape

        x = x.reshape(B, T, C_enc, H_enc, W_enc)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, T * H_enc * W_enc, C_enc)

        seq_len = T * H_enc * W_enc
        pos_embed = self.encoder_pos_embed[:, :seq_len, :]
        x = x + pos_embed

        x = self.encoder(x)

        x = x.reshape(B, T, H_enc, W_enc, self.encoder_dim)

        z_all = []
        indices_all = []
        vq_loss_all: dict[str, list[torch.Tensor]] = {}

        for t in range(T):
            x_t = x[:, t, :, :, :]
            x_t = x_t.permute(0, 3, 1, 2).reshape(B, self.encoder_dim, H_enc * W_enc)
            x_t = x_t.permute(0, 2, 1)

            x_t_embed = self.to_vq_embedding(x_t)
            x_t_embed = x_t_embed.reshape(B, H_enc, W_enc, self.embedding_dim)
            x_t_embed = x_t_embed.permute(0, 3, 1, 2)

            z_q_t, indices_t, vq_loss_t = self.vq(x_t_embed)
            # The quantizer returns its input's channel-first (B, D, H', W')
            # layout. Reshaping it to (B, H', W', D) scrambled embeddings across
            # positions, so the decoder learned a layout that `decode_indices`
            # (a plain codebook lookup) never produces, and every frame
            # generated from tokens decoded to noise.
            z_all.append(z_q_t.permute(0, 2, 3, 1))
            indices_all.append(indices_t.reshape(B, H_enc, W_enc))
            for k, v in vq_loss_t.items():
                if k not in vq_loss_all:
                    vq_loss_all[k] = []
                vq_loss_all[k].append(v)

        z_q = torch.stack(z_all, dim=1)
        indices = torch.stack(indices_all, dim=1)

        vq_loss = {}
        for k, v in vq_loss_all.items():
            stacked = torch.stack(v)
            if k == "perplexity":
                vq_loss[k] = stacked.mean()
            else:
                vq_loss[k] = stacked.mean()

        return z_q, indices, vq_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to embeddings for video frames.

        Args:
            indices: Token indices (B, T, H', W') or (B, T, N) where N = H' x W'

        Returns:
            z_q: Quantized embeddings (B, T, H', W', embedding_dim)
        """
        if indices.dim() == 4:
            B, T, H_idx, W_idx = indices.shape
            indices_reshaped = indices
        elif indices.dim() == 3:
            B, T, N = indices.shape
            num_patches = int(N**0.5)
            if num_patches**2 == N:
                H_idx = W_idx = num_patches
                indices_reshaped = indices.reshape(B, T, H_idx, W_idx)
            else:
                indices_reshaped = indices
                H_idx = W_idx = 1
        else:
            raise ValueError(f"Expected 3D or 4D indices, got {indices.dim()}D")

        z_q = F.embedding(indices_reshaped, self.vq.codebook.weight)

        z_q = z_q.reshape(B, T, H_idx, W_idx, self.embedding_dim)

        return z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens to video frames.

        Args:
            z_q: Quantized embeddings (B, T, H', W', embedding_dim)

        Returns:
            Reconstructed video (B, C, T, H, W)
        """
        B, T, H_dec, W_dec, _ = z_q.shape

        x = z_q.reshape(B, T * H_dec * W_dec, -1)

        x = self.from_vq_embedding(x)

        seq_len = T * H_dec * W_dec
        pos_embed = self.decoder_pos_embed[:, :seq_len, :]
        x = x + pos_embed

        x = self.decoder(x)

        x = x.reshape(B, T, H_dec, W_dec, self.decoder_dim)
        # (B, T, H', W', D) -> (B, T, D, H', W') -> (B*T, D, H', W'). Permuting to
        # (B, D, T, H', W') first and then flattening B*T interleaved channels of
        # different frames, so every decoded frame mixed features from the whole
        # clip: the decoder stopped being causal and the tokenizer learned to
        # spread each frame's content across time, which made tokens flicker
        # between frames and left the dynamics model nothing stable to predict.
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, self.decoder_dim, H_dec, W_dec)

        x = self.decoder_proj(x)

        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Full forward pass with VQ-VAE objective.

        Args:
            x: Video tensor (B, C, T, H, W)

        Returns:
            reconstructed: Reconstructed video (B, C, T, H, W)
            indices: Token indices (B, T, H', W')
            loss_dict: Dictionary containing loss components
        """
        z_q, indices, vq_loss = self.encode(x)

        reconstructed = self.decode(z_q)

        recon_loss = F.mse_loss(reconstructed, x)

        loss_dict = {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss["vq_loss"],
            "perplexity": vq_loss["perplexity"],
        }

        return reconstructed, indices, loss_dict


def create_video_tokenizer(
    num_frames: int = 16,
    image_size: int = 64,
    in_channels: int = 3,
    encoder_dim: int = 512,
    decoder_dim: int = 1024,
    encoder_depth: int = 12,
    decoder_depth: int = 20,
    num_heads: int = 16,
    patch_size: int = 4,
    vocab_size: int = 1024,
    embedding_dim: int = 32,
    use_ema: bool = False,
) -> VideoTokenizer:
    """Factory function to create a Video Tokenizer."""
    return VideoTokenizer(
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
        use_ema=use_ema,
    )
