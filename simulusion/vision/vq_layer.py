from torch import nn
import torch
import torch.nn.functional as F
from simulusion.export import ExportableAgentMixin


# A code that is never selected decays as ``usage *= ema_decay`` each step. With
# the default decay of 0.99, this threshold corresponds to a code that has won
# fewer than one assignment per ~100 steps -- i.e. genuinely dead, rather than
# merely rare. Raising it toward 1.0 restarts codes that are still in use.
DEFAULT_DEAD_CODE_THRESHOLD = 0.01


@torch.no_grad()
def restart_dead_codes(
    codebook: nn.Embedding,
    usage: torch.Tensor,
    z_flat: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-seed codebook entries that have fallen out of use.

    Nearest-neighbour quantizers are prone to codebook collapse: a code that
    stops winning any assignment receives no further update and can never come
    back, so the effective vocabulary shrinks (visible as the perplexity metric
    dropping toward 1). The standard remedy is to periodically reset unused
    entries onto randomly drawn encoder outputs, which puts them back in a
    region of space where they can win assignments again.

    Args:
        codebook: The embedding table to modify in place.
        usage: (vocab_size,) EMA of how often each code was selected. Under an
            EMA with decay d, this converges to the code's mean assignments per
            step, so the threshold is interpretable in those units.
        z_flat: (N, C) encoder outputs from the current batch, used as the pool
            of candidate re-seed locations.
        threshold: Codes with usage below this are considered dead. Values near
            1.0 are far too aggressive for a full codebook -- with V codes and
            roughly V assignments per step the mean usage is ~1, so half the
            book would be restarted every step. See
            ``DEFAULT_DEAD_CODE_THRESHOLD``.

    Returns:
        ``(num_restarted, restarted_mask)`` -- a scalar tensor counting the codes
        that were re-seeded, and the boolean mask identifying them. Callers that
        keep their own accumulators for the codebook (the EMA quantizer's
        ``ema_embed_avg``) need the mask: they cannot recover it afterwards by
        comparing usage against the threshold, because the restart deliberately
        lifts usage above it.
    """
    dead = usage < threshold
    num_dead = int(dead.sum().item())
    if num_dead == 0:
        return torch.zeros((), device=codebook.weight.device), dead

    n_samples = z_flat.shape[0]
    # Sample with replacement so this works even when dead codes outnumber the
    # encoder outputs available in the batch.
    pick = torch.randint(0, n_samples, (num_dead,), device=z_flat.device)
    codebook.weight.data[dead] = z_flat[pick].to(codebook.weight.dtype)
    # Give restarted codes a fresh lease so they are not culled again next step.
    # It has to be *above* the threshold, not equal to it: usage decays before
    # the next check, so seeding at exactly `threshold` puts every restarted code
    # back under it immediately and any code that misses a single batch is
    # re-seeded on every subsequent step -- churning the codebook forever instead
    # of rescuing it once. One order of magnitude of headroom gives a code that
    # wins nothing roughly log(10)/log(1/decay) steps (~230 at decay 0.99) to
    # start winning assignments before it is considered dead again.
    usage[dead] = threshold * 10.0

    return torch.tensor(float(num_dead), device=codebook.weight.device), dead


class VectorQuantizer(ExportableAgentMixin, nn.Module):
    """Vector Quantizer for discrete autoencoder.

    Implements the VQ-VAE quantization from:
    "Neural Discrete Representation Learning" (Van Den Oord et al., 2017)

    Uses exponential moving averages for codebook updates and straight-through
    estimator for gradient flow.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 512,
        commitment_weight: float = 1.0,
        restart_dead_codes_after: float = DEFAULT_DEAD_CODE_THRESHOLD,
        usage_decay: float = 0.99,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.restart_dead_codes_after = restart_dead_codes_after
        self.usage_decay = usage_decay

        # Codebook: learnable embeddings
        self.codebook = nn.Embedding(vocab_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        # EMA of per-code selection counts, used only to detect dead codes.
        # Initialised to zeros, matching VectorQuantizerEMA's ema_cluster_size:
        # starting at ones would keep every unused code above the threshold for
        # ~log(threshold)/log(decay) steps (about 460 at the defaults), so a
        # codebook that collapses early would not be rescued until long after
        # the damage was done. Starting at zero also makes the first few steps
        # act as a data-dependent codebook initialisation, seeding entries from
        # real encoder outputs instead of the uniform prior above.
        self.register_buffer("code_usage", torch.zeros(vocab_size))
        self.code_usage: torch.Tensor = self.code_usage

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize the input latents.

        Args:
            z: Input tensor of shape (B, C, H, W) or (B, C)

        Returns:
            z_q: Quantized tensor (same shape as input)
            indices: Token indices for each position (B, H, W) or (B,)
            loss: Dictionary containing VQ loss components
        """
        # Reshape for quantization

        z_flat: torch.Tensor
        B: int
        C: int
        H: int
        W: int

        if z.dim() == 4:  # (B, C, H, W)
            B, C, H, W = z.shape
            # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
            z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif z.dim() == 2:  # (B, C)
            B = z.shape[0]
            C = z.shape[1]
            H = 1
            W = 1
            z_flat = z.unsqueeze(1)  # (B, 1, C)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {z.dim()}D")

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z·e
        z_flat = z_flat.float()
        codebook: torch.Tensor = self.codebook.weight.float()

        d: torch.Tensor = (
            torch.sum(z_flat**2, dim=-1, keepdim=True)
            + torch.sum(codebook**2, dim=-1)
            - 2 * torch.matmul(z_flat, codebook.t())
        )  # (B, H*W, vocab_size)

        # Find nearest codebook entries (indices)
        indices: torch.Tensor = torch.argmin(d, dim=-1)  # (B, H*W) or (B, 1)

        # Get the quantized values
        z_q: torch.Tensor = F.embedding(indices, codebook)  # (B, H*W, C)

        # VQ-VAE loss with a gradient-trained codebook:
        #   codebook loss: pull codebook vectors toward encoder outputs
        #   commitment loss (weighted by beta): pull encoder outputs toward codebook
        codebook_loss: torch.Tensor = F.mse_loss(z_q, z_flat.detach())
        commitment_loss: torch.Tensor = F.mse_loss(z_q.detach(), z_flat)
        vq_loss: torch.Tensor = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator: forward uses z_q, backward copies gradients
        # to the encoder as if z_q == z (argmin is non-differentiable).
        z_q = z_flat + (z_q - z_flat).detach()

        # Perplexity: measure of how many codebook entries are used
        encodings: torch.Tensor = F.one_hot(
            indices.reshape(-1), self.vocab_size
        ).float()
        avg_probs: torch.Tensor = torch.mean(encodings, dim=0)
        perplexity: torch.Tensor = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        # Track usage and re-seed collapsed codes (see restart_dead_codes).
        # Usage tracking is independent of revival: it is a useful diagnostic on
        # its own, and gating both on the same flag would leave the buffer empty
        # whenever revival is switched off.
        num_restarted = torch.zeros((), device=z_flat.device)
        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0)
                self.code_usage.mul_(self.usage_decay).add_(
                    counts, alpha=1 - self.usage_decay
                )
                if self.restart_dead_codes_after > 0:
                    num_restarted, _ = restart_dead_codes(
                        self.codebook,
                        self.code_usage,
                        z_flat.reshape(-1, C).detach(),
                        self.restart_dead_codes_after,
                    )

        # Reshape back to original spatial dimensions
        if z.dim() == 4:
            z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            z_q = z_q.squeeze(1)

        indices_reshaped: torch.Tensor = (
            indices.reshape(B, H, W) if z.dim() == 4 else indices.squeeze(-1)
        )

        loss: dict[str, torch.Tensor] = {
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "dead_codes_restarted": num_restarted,
        }

        return z_q, indices_reshaped, loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices back to embeddings.

        Args:
            indices: Token indices (B, H, W) or (B,)

        Returns:
            Embeddings (B, C, H, W) or (B, C)
        """
        if indices.dim() == 3:  # (B, H, W)
            B, H, W = indices.shape
            # Flatten first: F.embedding on a 3D index tensor yields (B, H, W, C),
            # which a 3-argument permute cannot handle.
            indices_flat = indices.reshape(B, -1)  # (B, H*W)
            z_q = F.embedding(indices_flat, self.codebook.weight)  # (B, H*W, C)
            z_q = z_q.permute(0, 2, 1).reshape(B, -1, H, W)
        else:
            z_q = F.embedding(indices, self.codebook.weight)

        return z_q


class VectorQuantizerEMA(ExportableAgentMixin, nn.Module):
    """Vector Quantizer with Exponential Moving Average updates.

    Uses EMA updates for the codebook instead of gradient-based updates,
    which leads to more stable training.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 512,
        commitment_weight: float = 1.0,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_dead_codes_after: float = DEFAULT_DEAD_CODE_THRESHOLD,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.restart_dead_codes_after = restart_dead_codes_after

        # Codebook
        self.codebook = nn.Embedding(vocab_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        # EMA tracking
        # Annotate buffers so the type checker knows these are tensors
        self.register_buffer("ema_cluster_size", torch.zeros(vocab_size))
        self.ema_cluster_size: torch.Tensor = self.ema_cluster_size

        self.register_buffer("ema_embed_avg", self.codebook.weight.data.clone())
        self.ema_embed_avg: torch.Tensor = self.ema_embed_avg

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize with EMA updates."""
        # Flatten spatial dims
        B, C, H, W = z.shape
        z_flat: torch.Tensor = z.permute(0, 2, 3, 1).reshape(B, H * W, C).float()

        # Compute distances
        codebook: torch.Tensor = self.codebook.weight.float()
        d: torch.Tensor = (
            torch.sum(z_flat**2, dim=-1, keepdim=True)
            + torch.sum(codebook**2, dim=-1)
            - 2 * torch.matmul(z_flat, codebook.t())
        )

        indices: torch.Tensor = torch.argmin(d, dim=-1)

        # Quantize: nearest codebook lookup (non-differentiable in indices)
        z_q: torch.Tensor = F.embedding(indices, codebook)  # (B, H*W, C)

        num_restarted = torch.zeros((), device=z.device)

        # EMA update (only during training). The codebook is updated by
        # exponential moving averages of the assigned encoder outputs (Van Den
        # Oord et al., 2017), not by gradient descent.
        if self.training:
            with torch.no_grad():
                V = self.vocab_size
                flat_enc: torch.Tensor = F.one_hot(
                    indices.reshape(-1), V
                ).float()  # (N, V)
                z_in: torch.Tensor = z_flat.reshape(-1, C)  # (N, C)

                cluster_counts = flat_enc.sum(dim=0)  # (V,)
                dw = flat_enc.t() @ z_in  # (V, C) sum of z assigned to each code

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    cluster_counts, alpha=1 - self.ema_decay
                )
                self.ema_embed_avg.mul_(self.ema_decay).add_(
                    dw, alpha=1 - self.ema_decay
                )

                # Laplace smoothing to avoid dividing by zero for unused codes.
                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.epsilon) / (n + V * self.epsilon) * n
                )  # (V,)
                self.codebook.weight.data.copy_(
                    self.ema_embed_avg / smoothed.unsqueeze(1)
                )

                # Re-seed codes that have stopped winning assignments. Without
                # this, ema_embed_avg for an unused code decays toward zero and
                # the code is permanently lost (perplexity collapses to 1).
                if self.restart_dead_codes_after > 0:
                    restarted, revived = restart_dead_codes(
                        self.codebook,
                        self.ema_cluster_size,
                        z_in.detach(),
                        self.restart_dead_codes_after,
                    )
                    if restarted > 0:
                        # Keep the EMA accumulator consistent with the new
                        # vectors, otherwise the next update would immediately
                        # pull them back toward the stale average. Use the mask
                        # the restart reports rather than re-deriving it from the
                        # threshold: the restart lifts usage *above* the
                        # threshold on purpose, so a threshold comparison here
                        # would select nothing and silently undo every restart.
                        self.ema_embed_avg[revived] = self.codebook.weight.data[
                            revived
                        ] * self.ema_cluster_size[revived].unsqueeze(1)
                    num_restarted = restarted

        # Reshape back to (B, C, H, W)
        z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Commitment loss: pull encoder outputs toward the codebook (the codebook
        # itself is moved by EMA, so there is no gradient-based codebook loss).
        commitment_loss: torch.Tensor = F.mse_loss(z_q.detach(), z)

        # Straight-through estimator: forward returns z_q, but gradients flow to
        # the encoder as if z_q == z. Without this the encoder receives no
        # reconstruction gradient (argmin is non-differentiable).
        z_q = z + (z_q - z).detach()

        # Perplexity
        encodings_eval: torch.Tensor = F.one_hot(indices, self.vocab_size).float()
        avg_probs: torch.Tensor = torch.mean(
            encodings_eval.reshape(-1, self.vocab_size), dim=0
        )
        perplexity: torch.Tensor = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        loss: dict[str, torch.Tensor] = {
            "vq_loss": commitment_loss * self.commitment_weight,
            "perplexity": perplexity,
            "dead_codes_restarted": num_restarted,
        }

        return z_q, indices.reshape(B, H, W), loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to embeddings.

        Args:
            indices: Token indices (B, H, W) or (B,)

        Returns:
            Embeddings (B, C, H, W) or (B, C)
        """
        if indices.dim() == 3:  # (B, H, W)
            B, H, W = indices.shape
            indices_flat = indices.reshape(B, -1)  # (B, H*W)
            z_q = F.embedding(indices_flat, self.codebook.weight)  # (B, H*W, C)
            z_q = z_q.permute(0, 2, 1).reshape(B, -1, H, W)
        else:
            z_q = F.embedding(indices, self.codebook.weight)

        return z_q
