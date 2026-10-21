import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any
from einops import rearrange
from simulusion.configs.dit_config import DiTConfig as Config
from simulusion.models.model_io import (
    apply_config_overrides,
    coerce_config,
    module_summary,
    resolve_pretrained_file,
    save_config_next_to_checkpoint,
)
from simulusion.blocks.mhsa import MultiHeadSelfAttention
from simulusion.datasets.cifar10 import make_cifar10
from simulusion.datasets.imagenet1k import make_imagenet1k, make_imagefolder
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from simulusion.transforms.image import make_transforms
from torchvision.utils import save_image
import os
from pathlib import Path
from simulusion.export import ExportableAgentMixin


def sinusoidal_time_embedding(
    timesteps: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for diffusion conditioning.

    Math:
        embedding[t] = [sin(t / P^(2i/d)), cos(t / P^(2i/d))] for i in [0, d/2)

    Note the *division* by increasing powers of ``max_period``: frequencies decay
    from 1 down to ``1/max_period``. Building them the other way round (ascending
    to ``max_period``) makes the sine argument reach ~1e7 radians at t=999, which
    aliases so badly that adjacent timesteps receive near-orthogonal embeddings --
    the model then has to memorise every noise level instead of interpolating
    between them.

    Args:
        timesteps: Tensor of timesteps, shape (B,) or (B, 1)
        dim: Embedding dimension
        max_period: Longest sinusoid period; 10000 follows DDPM/ADM.

    Returns:
        Tensor of shape (B, dim) with sinusoidal embeddings.
    """
    timesteps = timesteps.reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sine-cosine positional embeddings (ViT/DiT convention).

    The paper applies "standard ViT frequency-based positional embeddings (the
    sine-cosine version) to all input tokens" after patchify. These are constant,
    not learned, which is what lets a trained model be evaluated at a different
    token count without the embeddings being meaningless.

    Args:
        embed_dim: Token dimension; must be divisible by 4.
        grid_size: Tokens per side, i.e. ``input_size // patch_size``.

    Returns:
        (grid_size**2, embed_dim) positional embeddings.
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be divisible by 4 for 2D sin-cos embeddings, got {embed_dim}."
        )

    # Half the channels encode the row index, half the column index.
    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(coords, coords, indexing="ij")

    emb_h = sinusoidal_time_embedding(grid_h.reshape(-1), embed_dim // 2)
    emb_w = sinusoidal_time_embedding(grid_w.reshape(-1), embed_dim // 2)
    return torch.cat([emb_h, emb_w], dim=1)


class TimestepEmbedder(nn.Module):
    """Embed diffusion timesteps into the transformer's conditioning space.

    Paper A: "a 256-dimensional frequency embedding followed by a two-layer MLP
    with dimensionality equal to the transformer's hidden size and SiLU
    activations".
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = sinusoidal_time_embedding(t, self.frequency_embedding_size)
        return self.fc2(self.act(self.fc1(freqs.to(self.fc1.weight.dtype))))


class LabelEmbedder(nn.Module):
    """Embed class labels, with dropout to a learned null token for guidance.

    Classifier-free guidance (paper 3.1) needs the model to also score the
    unconditional distribution. That is obtained by randomly replacing the label
    with a learned "null" embedding during training, so the embedding table holds
    ``num_classes + 1`` entries and index ``num_classes`` is the null token.

    Args:
        num_classes: Number of real classes.
        hidden_size: Conditioning dimension.
        dropout_prob: Probability of dropping the label during training. 0
            disables guidance support entirely.
    """

    def __init__(
        self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + int(use_cfg_embedding), hidden_size
        )
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Replace a random subset of labels with the null class."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive-layer-norm scale and shift to a token sequence."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(ExportableAgentMixin, nn.Module):
    """Patchify an image into a sequence of learnable patch tokens.

    Used in Vision Transformers (ViT) and DiT to convert 2D images into
    sequences of token embeddings that can be processed by transformers.

    Process:
        1. Conv2d with kernel_size=stride=patch_size extracts non-overlapping patches
        2. Each patch is projected to embed_dim via linear layer (Conv2d)
        3. Learnable positional embeddings are added for spatial information

    Input: (B, C, H, W) images
    Output: (B, N, embed_dim) where N = (H/patch_size) * (W/patch_size)

    Args:
        img_size: Image size (assumes square), e.g., 32 for CIFAR
        patch_size: Size of each patch (typically 4, 8, or 16)
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Output dimension for each patch token

    Usage with DiT:
        patch_embed = PatchEmbed(img_size=32, patch_size=4, in_channels=3, embed_dim=256)
        tokens = patch_embed(images)  # (B, 64, 256) for 32x32 image with patch_size=4
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        learnable_pos: bool = False,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size {img_size} must be divisible by patch_size {patch_size}."
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        # ViT-style init: treat the patch projection as the linear layer it is.
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.shape[0], -1))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # Paper 3.2 uses fixed sine-cosine embeddings. `learnable_pos` keeps the
        # earlier behaviour available but is not the paper's configuration.
        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size).unsqueeze(0)
        if learnable_pos:
            self.pos = nn.Parameter(pos.clone())
        else:
            self.register_buffer("pos", pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.pos
        return x


class PatchUnEmbed(ExportableAgentMixin, nn.Module):
    """Reconstruct image-like tensors from patch-token sequences.

    The inverse of `PatchEmbed`, this module reshapes token sequences into
    grids and uses transposed convolution to decode spatial outputs.
    """

    def __init__(
        self, img_size: int, patch_size: int, embed_dim: int, out_channels: int
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = w = self.img_size // self.patch_size
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning (paper 3.2).

    The conditioning vector -- the *sum* of the timestep and class embeddings --
    is mapped by a single SiLU + Linear to six vectors per block: shift, scale
    and gate for each of the attention and MLP sub-layers. That is the ``6x
    hidden`` output the paper specifies for adaLN-Zero (vanilla adaLN uses ``4x``
    because it has no gates).

    The "-Zero" is the important half. The modulation layer is zero-initialised,
    so at step 0 every gate is 0 and the whole block is the identity function.
    Figure 5 shows this matters a lot: adaLN-Zero reaches roughly half the FID of
    in-context conditioning at 400K steps, and clearly beats vanilla adaLN, which
    is identical apart from the gates and their initialisation.

    Normalisation is ``LayerNorm`` without affine parameters -- the scale and
    shift come from the conditioning instead, so learned per-channel affines
    would be redundant.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        drop: float,
        t_dim: int | None = None,
    ) -> None:
        super(TransformerBlock, self).__init__()
        cond_dim = d_model if t_dim is None else t_dim

        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            # Paper A: "GELU nonlinearities (approximated with tanh)".
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(drop),
        )

        modulation_proj = nn.Linear(cond_dim, 6 * d_model, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), modulation_proj)
        # Identity initialisation: zero weights and biases make every gate zero.
        nn.init.zeros_(modulation_proj.weight)
        nn.init.zeros_(modulation_proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, D) tokens; c (B, cond_dim) conditioning."""
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.ff(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Adaptive-layer-norm + linear decode to per-patch outputs (paper 3.2).

    "We apply the final layer norm (adaptive if using adaLN) and linearly decode
    each token into a p x p x 2C tensor". Both the modulation and the linear
    decode are zero-initialised, so the model starts by predicting zero noise
    rather than an arbitrary field.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        modulation_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), modulation_proj)
        nn.init.zeros_(modulation_proj.weight)
        nn.init.zeros_(modulation_proj.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(ExportableAgentMixin, nn.Module):
    """Diffusion Transformer model for image denoising and generation.

    The module maps noisy images and timesteps to predicted noise residuals
    and also provides a classmethod training entrypoint for common datasets.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        depth: int,
        heads: int,
        drop: float = 0.0,
        t_dim: int = 256,
        num_classes: int = 0,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Build a Diffusion Transformer.

        Args:
            img_size: Spatial size of the input. For latent diffusion this is the
                latent resolution (32 for 256x256 images through an f8 VAE), not
                the pixel resolution.
            patch_size: Patch side length p. Paper explores 2, 4 and 8; smaller p
                quadruples the token count and therefore the Gflops, which
                Figure 8 shows is what actually drives FID.
            in_channels: Channels of the input (4 for an f8 VAE latent).
            d_model: Transformer hidden size.
            depth: Number of DiT blocks.
            heads: Attention heads.
            drop: Dropout inside the MLP.
            t_dim: Width of the raw frequency embedding before the timestep MLP.
            num_classes: Number of classes for class-conditional generation. 0
                builds an unconditional model.
            class_dropout_prob: Label dropout probability enabling classifier-free
                guidance. Ignored when ``num_classes`` is 0.
            learn_sigma: Predict the diagonal covariance alongside the noise, so
                the output has ``2 * in_channels`` channels (paper 3.1, following
                Nichol & Dhariwal). Set False for an epsilon-only model.
            mlp_ratio: MLP expansion factor.
        """
        super(DiT, self).__init__()
        self.t_dim = t_dim
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.config = Config(
            IMG_SIZE=img_size,
            CHANNELS=in_channels,
            PATCH=patch_size,
            WIDTH=d_model,
            DEPTH=depth,
            HEADS=heads,
            DROP=drop,
            NUM_CLASSES=num_classes,
            CLASS_DROPOUT_PROB=class_dropout_prob,
            LEARN_SIGMA=learn_sigma,
        )

        self.t_embedder = TimestepEmbedder(d_model, frequency_embedding_size=t_dim)
        self.y_embedder = (
            LabelEmbedder(num_classes, d_model, class_dropout_prob)
            if num_classes > 0
            else None
        )

        self.patchify = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, heads, mlp_ratio=mlp_ratio, drop=drop, t_dim=d_model
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(d_model, patch_size, self.out_channels)

        self._init_transformer_weights()

    def _init_transformer_weights(self) -> None:
        """ViT-style init for the transformer body (paper 4: 'standard weight
        initialization techniques from ViT'). The adaLN and final layers are
        zero-initialised in their own constructors."""
        for module in self.transformer_blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Re-zero the modulation layers that the sweep above just overwrote.
        for block in self.transformer_blocks:
            assert isinstance(block, TransformerBlock)
            modulation_proj = block.adaLN_modulation[1]
            assert isinstance(modulation_proj, nn.Linear)
            nn.init.zeros_(modulation_proj.weight)
            nn.init.zeros_(modulation_proj.bias)

    def unpatchify_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange decoded tokens (B, T, p*p*C_out) back to (B, C_out, H, W)."""
        h = w = self.patchify.grid_size
        return rearrange(
            x,
            "b (h w) (p q c) -> b c (h p) (w q)",
            h=h,
            w=w,
            p=self.patch_size,
            q=self.patch_size,
            c=self.out_channels,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict noise (and covariance when ``learn_sigma``).

        Args:
            x: Noised input (B, C, H, W).
            t: Diffusion timesteps (B,).
            y: Class labels (B,). Required when the model is class-conditional.

        Returns:
            (B, out_channels, H, W); when ``learn_sigma`` the first ``C``
            channels are the predicted noise and the rest the covariance.
        """
        c = self.t_embedder(t)
        if self.y_embedder is not None:
            if y is None:
                raise ValueError(
                    "This DiT is class-conditional (num_classes="
                    f"{self.num_classes}); pass `y`."
                )
            # Conditioning is the *sum* of timestep and class embeddings (3.2).
            c = c + self.y_embedder(y, self.training)
        elif y is not None:
            raise ValueError(
                "This DiT is unconditional (num_classes=0) but `y` was provided."
            )

        x = self.patchify(x)
        for block in self.transformer_blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify_tokens(x)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        guided_channels: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance (paper 3.1).

        Computes ``eps = eps(x, null) + s * (eps(x, y) - eps(x, null))`` by
        running the conditional and unconditional branches as one batch.

        Args:
            x: Noised input (B, C, H, W).
            t: Timesteps (B,).
            y: Class labels (B,).
            cfg_scale: Guidance scale s; 1.0 recovers standard sampling.
            guided_channels: Apply guidance to only the first N channels. The
                paper's appendix guides 3 of the 4 latent channels; None guides
                all of them, for which the equivalent scale is roughly
                ``1 + 0.75 * (s - 1)``.

        Returns:
            Guided model output, same shape as a plain forward pass.
        """
        if self.y_embedder is None:
            raise ValueError("Classifier-free guidance needs a class-conditional DiT.")

        half = x[: len(x) // 2] if x.shape[0] % 2 == 0 else x
        combined = torch.cat([half, half], dim=0)
        null = torch.full_like(y[: half.shape[0]], self.num_classes)
        labels = torch.cat([y[: half.shape[0]], null], dim=0)

        model_out = self.forward(combined, torch.cat([t[: half.shape[0]]] * 2), labels)

        n_guided = guided_channels if guided_channels is not None else self.in_channels
        eps, rest = model_out[:, :n_guided], model_out[:, n_guided:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([guided_eps, guided_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @classmethod
    def from_config(
        cls,
        config: Config | dict[str, Any] | str | Path | None = None,
        **overrides: Any,
    ) -> "DiT":
        """Build DiT from a config object, dict, YAML file, or YAML string."""

        args = apply_config_overrides(coerce_config(Config, config), overrides)
        return cls(
            img_size=args.IMG_SIZE,
            patch_size=args.PATCH,
            in_channels=args.CHANNELS,
            d_model=args.WIDTH,
            depth=args.DEPTH,
            heads=args.HEADS,
            drop=args.DROP,
            num_classes=args.NUM_CLASSES,
            class_dropout_prob=args.CLASS_DROPOUT_PROB,
            learn_sigma=args.LEARN_SIGMA,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: Config | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **overrides: Any,
    ) -> "DiT":
        """Load DiT weights from a local path/directory or HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("dit_model.pth", "model.pt", "pytorch_model.bin", "checkpoint.pt")
        )
        checkpoint_path = resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find a DiT checkpoint for {pretrained_model_name_or_path!r}."
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or "cpu", weights_only=True
        )
        checkpoint_config = (
            checkpoint.get("config") if isinstance(checkpoint, dict) else None
        )
        if config is None and isinstance(checkpoint_config, dict):
            args = Config.from_dict(checkpoint_config)
        elif config is None:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "dit_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = Config.from_yaml(config_path)
        else:
            args = coerce_config(Config, config)
        model = cls.from_config(apply_config_overrides(args, overrides))
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save DiT weights and config in a from_pretrained-compatible format."""

        checkpoint_path = Path(path)
        if checkpoint_path.suffix == "":
            checkpoint_path = checkpoint_path / "dit_model.pth"
        save_config_next_to_checkpoint(self.config, checkpoint_path)
        torch.save(
            {"config": self.config.to_dict(), "model_state_dict": self.state_dict()},
            checkpoint_path,
        )

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            param.numel()
            for param in self.parameters()
            if not trainable_only or param.requires_grad
        )

    def summary(self) -> dict[str, Any]:
        modules: dict[str, nn.Module] = {
            "t_embedder": self.t_embedder,
            "patchify": self.patchify,
            "transformer_blocks": self.transformer_blocks,
            "final_layer": self.final_layer,
        }
        if self.y_embedder is not None:
            modules["y_embedder"] = self.y_embedder
        return module_summary(modules)

    def train(self, mode: bool = True) -> "DiT":
        """Set training mode -- the standard :meth:`torch.nn.Module.train`.

        This used to be shadowed by the training-loop classmethod, which made
        ``model.eval()`` raise ``TypeError: missing 1 required positional
        argument: 'dataset'`` and forced callers to reach for
        ``nn.Module.train(model, False)``. The training loop now lives in
        :meth:`fit`.
        """
        if not isinstance(mode, bool):
            raise TypeError(
                "DiT.train(mode: bool) toggles train/eval mode. The training "
                "loop is now DiT.fit(epochs=..., dataset=...), so that "
                "model.eval() works as it does for every other nn.Module."
            )
        return super().train(mode)

    @classmethod
    def fit(
        cls,
        epochs: int,
        dataset: Any,
        batch_size: int = 256,
        lr: float = 1e-4,
        img_size: int = 32,
        channels: int = 3,
        patch: int = 4,
        width: int = 384,
        depth: int = 12,
        heads: int = 6,
        drop: float = 0.0,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        ema: bool = True,
        ema_decay: float = 0.9999,
        num_classes: int = 0,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
        workdir: str = "./dit_demo",
        root_path: str = "./data",
        image_folder: str | None = None,
        crop_size: int | None = None,
        num_workers: int = 4,
        download: bool = True,
        copy_data: bool = False,
        subset_file: str | None = None,
        val_split: float | None = None,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_every: int = 0,
    ) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("WARNING: CUDA not available, using CPU")

        train_loader, val_loader = dit_loaders(
            dataset,
            batch_size=batch_size,
            img_size=img_size,
            root_path=root_path,
            image_folder=image_folder,
            crop_size=crop_size,
            num_workers=num_workers,
            download=download,
            copy_data=copy_data,
            subset_file=subset_file,
            val_split=val_split,
            with_val=early_stopping,
        )

        # The loop itself runs on the shared Trainer through the `dit` task:
        # L_simple, AdamW at a constant `lr` with no weight decay (paper 4),
        # gradient clipping at 1 and the EMA. Early stopping scores held-out
        # loss once per epoch and keeps the best epoch's weights.
        from simulusion.training.legacy import fit_legacy
        from simulusion.training.tasks.dit import DiTTask, DiTTaskConfig

        config = Config(
            DATASET=dataset,
            BATCH=batch_size,
            EPOCHS=epochs,
            LR=lr,
            WEIGHT_DECAY=0.0,
            IMG_SIZE=img_size,
            CHANNELS=channels,
            PATCH=patch,
            WIDTH=width,
            DEPTH=depth,
            HEADS=heads,
            DROP=drop,
            BETA_START=beta_start,
            BETA_END=beta_end,
            TIMESTEPS=timesteps,
            EMA=ema,
            EMA_DECAY=ema_decay,
            NUM_CLASSES=num_classes,
            CLASS_DROPOUT_PROB=class_dropout_prob,
            LEARN_SIGMA=learn_sigma,
            WORKDIR=workdir,
            ROOT_PATH=root_path,
        )
        task = DiTTask(
            DiTTaskConfig(model=config, image_folder=image_folder),
            loaders=(train_loader, val_loader),
            legacy_workdir=workdir,
        )
        per_epoch = max(1, len(train_loader))
        result = fit_legacy(
            task,
            workdir,
            max_steps=epochs * per_epoch,
            device=device,
            log_every=100,
            eval_every=per_epoch if early_stopping else 0,
            checkpoint_every=checkpoint_every * per_epoch,
            best_metric="val_loss" if early_stopping else None,
            best_mode="min",
            early_stopping_metric="val_loss" if early_stopping else None,
            early_stopping_mode="min",
            patience=patience,
            min_delta=min_delta,
        )
        model = task.model
        ema_model = task.ema
        ddpm = task.ddpm
        best_state = None
        best_val = float("inf")
        if result.best_checkpoint is not None:
            # Keep the best weights, not the last: on a plateau the run ends
            # `patience` epochs past the best model.
            saved = torch.load(
                result.best_checkpoint / "task" / "task.pt",
                map_location="cpu",
                weights_only=True,
            )
            best_state = saved["ema"] if ema_model is not None else saved["model"]
            if result.best_value is not None:
                best_val = float(result.best_value)
        print("Training Complete.")

        os.makedirs(workdir, exist_ok=True)

        model_to_save = ema_model if ema_model is not None else model
        if best_state is not None:
            model_to_save.load_state_dict(best_state)
            print(f"Restored the best held-out checkpoint (val_loss {best_val:.4f}).")
        checkpoint_path = Path(workdir) / "dit_model.pth"
        train_config = Config(
            DATASET=dataset,
            BATCH=batch_size,
            EPOCHS=epochs,
            LR=lr,
            IMG_SIZE=img_size,
            CHANNELS=channels,
            PATCH=patch,
            WIDTH=width,
            DEPTH=depth,
            HEADS=heads,
            DROP=drop,
            BETA_START=beta_start,
            BETA_END=beta_end,
            TIMESTEPS=timesteps,
            EMA=ema,
            EMA_DECAY=ema_decay,
            # Without these a class-conditional checkpoint reloads as an
            # unconditional model and fails on the missing y_embedder weights.
            NUM_CLASSES=num_classes,
            CLASS_DROPOUT_PROB=class_dropout_prob,
            LEARN_SIGMA=learn_sigma,
            WORKDIR=workdir,
            ROOT_PATH=root_path,
        )
        save_config_next_to_checkpoint(train_config, checkpoint_path)
        torch.save(model_to_save.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Generate new Images
        model_to_sample = ema_model if ema_model is not None else model
        model_to_sample.eval()
        with torch.no_grad():
            sample_labels = (
                torch.arange(16, device=device) % num_classes
                if num_classes > 0
                else None
            )
            samples = ddpm.sample(
                model_to_sample,
                n=16,
                img_size=img_size,
                channels=channels,
                y=sample_labels,
            )
            os.makedirs(workdir, exist_ok=True)
            save_image((samples + 1) / 2, f"{workdir}/generated_samples.png", nrow=4)
            print(f"Generated samples saved to {workdir}/generated_samples.png")


def dit_loaders(
    dataset: str,
    *,
    batch_size: int,
    img_size: int,
    root_path: str = "./data",
    image_folder: str | None = None,
    crop_size: int | None = None,
    num_workers: int = 4,
    download: bool = True,
    copy_data: bool = False,
    subset_file: str | None = None,
    val_split: float | None = None,
    with_val: bool = False,
) -> tuple[Any, Any]:
    """DiT's ``(train_loader, val_loader)``; ``val_loader`` is ``None`` without ``with_val``.

    ``dataset`` is ``"cifar10"`` (downloads into ``root_path``), ``"imagenet"``
    or ``"imagefolder"``. Images are scaled to ``[-1, 1]`` for CIFAR-10.
    """
    if dataset.lower() == "cifar10":
        # Scale to [-1, 1], the range the DDPM forward process, the sampler's
        # final clamp and every `(x + 1) / 2` display path assume. Plain
        # ToTensor() trained on [0, 1] and sampled washed-out images.
        transform = Compose(
            [
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        # Default the crop to the model's own input size. These are two
        # independent knobs, and a mismatch does not fail until the first
        # forward pass, as a bare shape error naming neither of them.
        resolved_crop = crop_size if crop_size else img_size
        transform = make_transforms(
            crop_size=resolved_crop,
            crop_scale=(0.3, 1.0),
            color_jitter=0.5,
            horizontal_flip=True,
            color_distortion=True,
            gaussian_blur=True,
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    # Held-out loader, built only when early stopping needs one. Training
    # loss cannot drive it: it keeps falling well past the point where
    # samples stop improving, so a plateau on it never arrives.
    val_loader: torch.utils.data.DataLoader | None = None

    if dataset.lower() == "cifar10":
        _, train_loader, _ = make_cifar10(
            transform=transform,
            batch_size=batch_size,
            collator=None,
            pin_mem=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            root_path=root_path,
            drop_last=True,
            train=True,
            download=download,
        )
        if with_val:
            # CIFAR-10 ships a test split, so nothing has to come out of
            # the training set.
            _, val_loader, _ = make_cifar10(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=num_workers,
                world_size=1,
                rank=0,
                root_path=root_path,
                drop_last=False,
                train=False,
                download=download,
            )
    elif dataset.lower() == "imagenet":
        _, train_loader, _ = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=None,
            pin_mem=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            training=True,
            copy_data=copy_data,
            drop_last=True,
            subset_file=subset_file,
        )
        if with_val:
            # ImageNet's own validation split; `training=False` selects it.
            _, val_loader, _ = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=num_workers,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                training=False,
                copy_data=copy_data,
                drop_last=False,
                subset_file=subset_file,
            )
    elif dataset.lower() == "imagefolder":
        _, train_loader, _ = make_imagefolder(
            transform=transform,
            batch_size=batch_size,
            collator=None,
            pin_mem=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            drop_last=True,
            val_split=val_split,
            split="train",
        )
        if with_val:
            if not val_split:
                raise ValueError(
                    "early_stopping on an imagefolder needs val_split > 0 "
                    "so there is held-out data to measure; pass e.g. "
                    "val_split=0.05"
                )
            _, val_loader, _ = make_imagefolder(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=num_workers,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                drop_last=False,
                val_split=val_split,
                split="val",
            )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported: cifar10, imagenet, imagefolder"
        )
    return train_loader, val_loader


def dit_ema_update(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    """``ema = decay * ema + (1 - decay) * model``, parameter by parameter."""
    with torch.no_grad():
        # Fused multi-tensor EMA: identical arithmetic to the per-tensor
        # mul_/add_ pair, but one kernel launch for the whole model.
        ema_params: list[torch.Tensor] = list(ema_model.parameters())
        src_params: list[torch.Tensor] = list(model.parameters())
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, src_params, alpha=1 - decay)


def dit_l_simple(
    model: nn.Module,
    ddpm: Any,
    images: torch.Tensor,
    labels: torch.Tensor | None,
    timesteps: int,
) -> torch.Tensor:
    """DiT's training objective, L_simple, on one batch.

    With ``learn_sigma`` the model emits 2C channels: noise then the covariance
    parameterisation. Paper 3.1 trains eps with L_simple and Sigma with the
    full variational bound; only the L_simple half is applied here, so the
    covariance head is carried but not learned. Set ``learn_sigma=False`` for
    a strictly epsilon-only model.
    """
    inner: Any = getattr(model, "module", model)  # through a DDP wrapper
    b = images.size(0)
    t = torch.randint(0, timesteps, (b,), device=images.device).long()
    noise = torch.randn_like(images)
    x_t = ddpm.q_sample(images, t, noise)
    y = labels if inner.y_embedder is not None else None
    pred = model(x_t, t, y)
    if inner.learn_sigma:
        pred = pred[:, : inner.in_channels]
    return F.mse_loss(pred, noise)


def create_dit(config: Any = None, **overrides: Any) -> DiT:
    """Create a :class:`DiT` from a ``DiTConfig`` or keyword overrides.

    The public factory API works with config objects, while ``DiT`` itself has a
    compact constructor. This adapter keeps the lower-level model constructor
    unchanged and maps the public config fields onto the expected arguments.
    """

    if config is None:
        config = Config()

    config_fields = set(getattr(config, "__dataclass_fields__", {}))
    config_overrides = {
        key: value for key, value in overrides.items() if key in config_fields
    }
    constructor_overrides = {
        key: value for key, value in overrides.items() if key not in config_fields
    }
    if config_overrides:
        from dataclasses import replace

        config = replace(config, **config_overrides)

    kwargs = {
        "img_size": config.IMG_SIZE,
        "patch_size": config.PATCH,
        "in_channels": config.CHANNELS,
        "d_model": config.WIDTH,
        "depth": config.DEPTH,
        "heads": config.HEADS,
        "drop": config.DROP,
        "num_classes": config.NUM_CLASSES,
        "class_dropout_prob": config.CLASS_DROPOUT_PROB,
        "learn_sigma": config.LEARN_SIGMA,
    }
    supported = set(kwargs) | {"t_dim", "mlp_ratio"}
    invalid = sorted(set(constructor_overrides) - supported)
    if invalid:
        raise ValueError(f"Unsupported DiT argument(s): {', '.join(invalid)}")
    kwargs.update(constructor_overrides)
    return DiT(**kwargs)
