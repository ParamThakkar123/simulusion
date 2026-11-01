from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from simulusion.export import ExportableAgentMixin


class DDPM(ExportableAgentMixin, nn.Module):
    """Utility module implementing forward and reverse DDPM diffusion steps.

    Precomputes diffusion schedule terms and exposes helpers for noising
    training inputs (`q_sample`) and iterative denoising sampling (`sample`).
    """

    def __init__(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.timesteps = timesteps

        # Every schedule term is a *buffer*: it has to follow the module across
        # devices and dtypes, and be restored from a checkpoint. Assigning
        # ``self.betas = ...`` first and calling ``register_buffer("betas", ...)``
        # afterwards raises KeyError("attribute 'betas' already exists"), because
        # nn.Module stores a bare tensor as a plain instance attribute and then
        # refuses to shadow it with a buffer of the same name.
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        # Re-declare for static checkers: register_buffer is untyped.
        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alphas_cumprod: torch.Tensor
        self.alphas_cumprod_prev: torch.Tensor
        self.sqrt_alphas_cumprod: torch.Tensor
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor
        self.posterior_variance: torch.Tensor

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x_start + s2 * noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Predict noise. Models that also learn the covariance (e.g. DiT with
        # learn_sigma) emit 2C channels -- noise first, then the covariance
        # parameterisation -- so keep only the noise half here. `y` carries
        # class labels for class-conditional models.
        eps = model(x_t, t) if y is None else model(x_t, t, y)
        if eps.shape[1] == 2 * x_t.shape[1]:
            eps = eps[:, : x_t.shape[1]]
        # Compute x0_hat
        a_t = self.alphas[t].view(-1, 1, 1, 1)
        ac_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_hat = (x_t - sqrt_one_minus_ac * eps) / torch.sqrt(ac_t)

        # Compute Posterior mean
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        ac_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        coef1 = torch.sqrt(ac_prev) * beta_t / (1.0 - ac_t)
        coef2 = torch.sqrt(a_t) * (1.0 - ac_prev) / (1.0 - ac_t)
        mean = coef1 * x0_hat + coef2 * x_t

        # Add noise except for t == 0
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        n: int,
        img_size: int,
        channels: int,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.randn(n, channels, img_size, img_size).to(
            next(model.parameters()).device
        )
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, dtype=torch.long).to(x.device)
            x = self.p_sample(model, x, t, y)
        return x.clamp(-1.0, 1.0)
