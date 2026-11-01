import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple
import math


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization that conditions on actions and diffusion time."""

    def __init__(self, num_groups: int, num_channels: int, cond_dim: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.linear = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            cond: Conditioning tensor [B, cond_dim]
        """
        x = self.norm(x)
        scale, bias = self.linear(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + bias


class ResBlock(nn.Module):
    """Residual block with adaptive group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(32, in_channels, cond_dim)
        self.norm2 = AdaptiveGroupNorm(32, out_channels, cond_dim)
        self.dropout = nn.Dropout(dropout)

        # skip connection may be either a Conv2d or an Identity - annotate as
        # generic nn.Module to satisfy static type checkers.
        self.skip: nn.Module
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, cond))
        h = self.conv1(h)
        h = self.act(self.norm2(h, cond))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.channels = channels
        self.norm = AdaptiveGroupNorm(32, channels, cond_dim)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x, cond)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)

        h = torch.matmul(attn, v)
        h = self.proj(h)

        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class TimestepEmbedding(nn.Module):
    """Sinusoidal noise-level embedding followed by an MLP.

    The input is EDM's ``c_noise = log(sigma) / 4`` (Appendix C, eq. 11), which
    ranges over several units either side of zero as sigma sweeps the schedule. A
    plain ``Linear(1, ...)`` on that scalar -- the previous implementation,
    despite the docstring -- gives the network a single monotone feature to
    condition on, so nearby noise levels are nearly indistinguishable after the
    first layer. The sinusoidal expansion used by every U-Net in this lineage
    spreads the scalar across `freq_dim` frequencies first, which is what makes
    the noise level a usable conditioning signal.
    """

    def __init__(self, dim: int, freq_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        if freq_dim % 2 != 0:
            raise ValueError(f"freq_dim must be even, got {freq_dim}.")
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        exponents = torch.arange(half, device=t.device, dtype=torch.float32) / half
        freqs = torch.exp(-math.log(self.max_period) * exponents)
        args = t.float().view(-1, 1) * freqs.view(1, -1)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        first = self.mlp[0]
        assert isinstance(first, nn.Linear)
        emb = self._sinusoidal(t).to(dtype=first.weight.dtype)
        return self.mlp(emb)


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        attention: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(in_channels, out_channels, cond_dim))
            in_channels = out_channels

        # attn can be AttentionBlock or None
        self.attn: Optional[AttentionBlock]
        if attention:
            self.attn = AttentionBlock(out_channels, cond_dim)
        else:
            self.attn = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, cond)
        if self.attn is not None:
            x = self.attn(x, cond)
        return x


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder with skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        attention: bool = False,
        skip_channels: Optional[int] = None,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        # The first res block consumes the upsampled features concatenated with
        # the encoder skip at this resolution. Those two widths only coincide
        # when every channel multiplier is 1 (DIAMOND's default, Table 2), so the
        # skip width is taken explicitly; assuming `in_channels * 2` made any
        # non-uniform `channel_multipliers` fail to build.
        skip = in_channels if skip_channels is None else skip_channels
        self.res_blocks.append(ResBlock(in_channels + skip, out_channels, cond_dim))
        in_ch = out_channels
        for _ in range(num_res_blocks - 1):
            self.res_blocks.append(ResBlock(in_ch, out_channels, cond_dim))
            in_ch = out_channels

        self.attn: Optional[AttentionBlock]
        if attention:
            self.attn = AttentionBlock(out_channels, cond_dim)
        else:
            self.attn = None

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        for res_block in self.res_blocks:
            x = res_block(x, cond)
        if self.attn is not None:
            x = self.attn(x, cond)
        return x


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for EDM diffusion world model.
    Uses frame stacking for observation conditioning and adaptive group norm for action conditioning.
    """

    def __init__(
        self,
        obs_channels: int = 3,
        num_conditioning_frames: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 1, 1, 1),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        action_dim: int = 18,
    ):
        super().__init__()
        self.num_conditioning_frames = num_conditioning_frames
        self.obs_channels = obs_channels

        self.input_conv = nn.Conv2d(
            obs_channels * (num_conditioning_frames + 1), base_channels, 3, padding=1
        )

        self.time_embed = TimestepEmbedding(cond_dim)

        self.action_embed = nn.Embedding(action_dim, cond_dim)
        self.action_proj = nn.Linear(cond_dim * num_conditioning_frames, cond_dim)

        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        # Width of the skip emitted at each resolution, consumed in reverse by
        # the decoder half.
        skip_widths: list[int] = []
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    cond_dim,
                    num_res_blocks,
                    attention=False,
                )
            )
            skip_widths.append(out_ch)
            in_ch = out_ch

        self.middle_block = nn.ModuleList(
            [
                ResBlock(in_ch, in_ch, cond_dim),
                AttentionBlock(in_ch, cond_dim),
                ResBlock(in_ch, in_ch, cond_dim),
            ]
        )

        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    cond_dim,
                    num_res_blocks,
                    attention=False,
                    skip_channels=skip_widths[-(i + 1)],
                )
            )
            in_ch = out_ch

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, obs_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obs_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            x: Noised observation at timestep t [B, C, H, W]
            t: Diffusion timestep [B]
            obs_history: Past observations for conditioning [B, L, C, H, W]
            actions: Past actions [B, L]

        Returns:
            Predicted clean observation [B, C, H, W]
        """
        B = x.shape[0]

        # Normalize obs_history to [B, L, C, H, W]
        if obs_history.ndim == 5:
            # possible layouts: [B, L, C, H, W] or [B, L, H, W, C]
            if obs_history.shape[2] == self.obs_channels:
                bh = obs_history
            elif obs_history.shape[-1] == self.obs_channels:
                bh = obs_history.permute(0, 1, 4, 2, 3).contiguous()
            else:
                # fallback: try to reshape if already flattened
                bh = obs_history.contiguous()
        else:
            bh = obs_history.contiguous()

        # bh is [B, L, C, H, W] -> reshape to [B, L*C, H, W]
        if bh.ndim == 5:
            L = bh.shape[1]
            C = bh.shape[2]
            H = bh.shape[3]
            W = bh.shape[4]
            bh_flat = bh.reshape(B, L * C, H, W)
        else:
            bh_flat = bh.reshape(B, -1, bh.shape[-2], bh.shape[-1])

        # ensure x is [B, C, H, W]
        if (
            x.ndim == 4
            and x.shape[1] != self.obs_channels
            and x.shape[-1] == self.obs_channels
        ):
            # maybe x is [B, H, W, C]
            x = x.permute(0, 3, 1, 2).contiguous()

        # concat: channels should equal obs_channels * (L + 1)
        x = torch.cat([x, bh_flat], dim=1)

        # final sanity check
        expected_in = self.input_conv.in_channels
        if x.shape[1] != expected_in:
            raise RuntimeError(
                f"DiffusionUNet input channel mismatch: got {x.shape[1]}, expected {expected_in}"
            )

        h = self.input_conv(x)

        t_emb = self.time_embed(t)
        action_emb = self.action_embed(actions.long())
        B_a, L_a, D_a = action_emb.shape
        action_emb = self.action_proj(action_emb.reshape(B_a, L_a * D_a))
        cond = t_emb + action_emb

        skip_connections = []
        for down_block in self.down_blocks:
            h = down_block(h, cond)
            skip_connections.append(h)
            h = F.avg_pool2d(h, 2)

        for block in self.middle_block:
            if isinstance(block, AttentionBlock):
                h = block(h, cond)
            else:
                h = block(h, cond)

        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = up_block(h, cond, skip)

        return self.output_conv(h)


class EDMPreconditioner:
    """EDM preconditioner following Karras et al. (2022)."""

    def __init__(
        self,
        sigma_data: float = 0.5,
        p_mean: float = -0.4,
        p_std: float = 1.2,
    ):
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

    def get_preconditioners(self, sigma: torch.Tensor) -> dict:
        """
        Compute EDM preconditioners for given noise levels.

        Returns:
            Dictionary with c_skip, c_out, c_in, c_noise
        """
        c_skip = (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = torch.log(sigma) / 4.0

        return {
            "c_skip": c_skip,
            "c_out": c_out,
            "c_in": c_in,
            "c_noise": c_noise,
        }

    def sample_noise_level(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise level from log-normal distribution."""
        log_sigma = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        sigma = torch.exp(log_sigma)
        return sigma

    def denoise(
        self, model: nn.Module, x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Apply EDM denoising with preconditioners.

        Args:
            model: Diffusion model
            x: Noised input [B, C, H, W]
            sigma: Noise level [B]
            **kwargs: Additional conditioning (obs_history, actions)

        Returns:
            Denoised prediction [B, C, H, W]
        """
        sigma = sigma.view(-1, 1, 1, 1)
        precond = self.get_preconditioners(sigma)

        model_input = precond["c_in"] * x
        # use EDM's c_noise (log-sigma transform) as the network timestep
        t_cond = precond["c_noise"].squeeze(-1).squeeze(-1)
        model_output = model(model_input, t_cond, **kwargs)

        denoised = precond["c_skip"] * x + precond["c_out"] * model_output
        return denoised


class EulerSampler:
    """Euler method sampler for reverse diffusion."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: int = 7,
        num_steps: int = 3,
        edm_precond: Optional[EDMPreconditioner] = None,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps

        if num_steps > 1:
            step_indices = torch.arange(num_steps)
            t_steps = (
                sigma_max ** (1 / rho)
                + step_indices
                / (num_steps - 1)
                * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho
        else:
            t_steps = torch.tensor([sigma_max])
        # `t_steps` is already produced in descending order (sigma_max -> sigma_min),
        # which is the correct EDM reverse-diffusion sampling direction: we start
        # from a pure-noise sample at sigma_max and denoise down to sigma_min.
        # `t_next` appends a trailing 0 so the final Euler step lands on the clean
        # image. (A previous version flipped `t_steps`, which initialized the sample
        # at sigma_min and ran the schedule backwards, adding noise instead of
        # removing it.)
        self.t_steps = t_steps
        self.t_next = torch.cat([self.t_steps[1:], torch.tensor([0.0])])
        # attach EDM preconditioner instance (use provided or default)
        self.edm_precond = (
            edm_precond if edm_precond is not None else EDMPreconditioner()
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        obs_history: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples using Euler method.

        Args:
            model: Diffusion model
            shape: Output shape [B, C, H, W]
            device: Device to run on
            obs_history: Conditioning observations [B, L, C, H, W]
            actions: Conditioning actions [B, L]

        Returns:
            Generated samples [B, C, H, W]
        """
        B = shape[0]
        # Ensure t_steps and t_next are on the same device and dtype as model inputs.
        # self.t_steps and self.t_next are created on CPU in __init__, so move/cast here.
        t_steps = self.t_steps.to(device=device, dtype=torch.get_default_dtype())
        t_next = self.t_next.to(device=device, dtype=torch.get_default_dtype())

        x = torch.randn(shape, device=device) * t_steps[0]

        # Use the preconditioner instance attached to the sampler (injected
        # at construction time) to ensure a single canonical implementation
        # of the EDM preconditioning is used across training and sampling.
        edm_precond = self.edm_precond

        for i in range(self.num_steps):
            t_cur = t_steps[i].expand(B).to(device)
            t_nxt = t_next[i].expand(B).to(device)

            sigma_cur = t_cur.view(-1, 1, 1, 1)

            # Apply EDM preconditioning: model was trained on inputs scaled by
            # c_in and expected to output the network prediction which is then
            # combined with c_skip and c_out to form the denoised image. Use
            # the same transforms at sampling time.
            # Use the canonical EDM preconditioner denoise helper so all
            # preconditioning (c_in/c_out/c_skip) and time-conditioning
            # (c_noise) logic is centralized in one place.
            denoised = edm_precond.denoise(
                model,
                x,
                t_cur,
                obs_history=obs_history,
                actions=actions,
            )

            # Euler step for the probability-flow ODE (deterministic sampler)
            d_cur = (x - denoised) / sigma_cur
            x = x + (t_nxt.view(-1) - t_cur.view(-1)).view(-1, 1, 1, 1) * d_cur

        # Observations live in [-1, 1] (see datasets.diamond_dataset.to_model_domain),
        # which is the domain sigma_data = 0.5 is calibrated for. Clamping to
        # [0, 1] here would erase the whole negative half of every frame.
        return x.clamp(-1.0, 1.0)
