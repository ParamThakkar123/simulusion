import torch
import torch.nn as nn
import torch.nn.functional as F
from simulusion.vision.planet_encoder import CNNEncoder
from simulusion.vision.planet_decoder import CNNDecoder
from simulusion.export import ExportableAgentMixin


class RecurrentStateSpaceModel(ExportableAgentMixin, nn.Module):
    """
    A Recurrent State Space Model (RSSM) for modeling latent dynamics in sequential data.
    """

    def __init__(
        self,
        action_size: int,
        state_size: int = 200,
        latent_size: int = 30,
        hidden_size: int = 200,
        embed_size: int = 1024,
        activation_function: str = "relu",
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.act_fn = getattr(F, activation_function)
        self.encoder = CNNEncoder(embed_size)
        self.decoder = CNNDecoder(state_size, latent_size, embed_size)
        self.grucell = nn.GRUCell(state_size, state_size)
        self.lat_act_layer = nn.Linear(latent_size + action_size, state_size)
        self.fc_prior_1 = nn.Linear(state_size, hidden_size)
        self.fc_prior_m = nn.Linear(hidden_size, latent_size)
        self.fc_prior_s = nn.Linear(hidden_size, latent_size)
        self.fc_posterior_1 = nn.Linear(state_size + embed_size, hidden_size)
        self.fc_posterior_m = nn.Linear(hidden_size, latent_size)
        self.fc_posterior_s = nn.Linear(hidden_size, latent_size)
        self.fc_reward_1 = nn.Linear(state_size + latent_size, hidden_size)
        self.fc_reward_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_reward_3 = nn.Linear(hidden_size, 1)

    def get_init_state(
        self,
        enc: torch.Tensor,
        h_t: torch.Tensor | None = None,
        s_t: torch.Tensor | None = None,
        a_t: torch.Tensor | None = None,
        mean: bool | None = None,
        *,
        sample: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next deterministic state and the posterior latent.

        By default the latent is the posterior *mean*. Pass ``sample=True`` to
        draw it from the posterior instead. ``mean`` is the old, inverted name
        for that flag (``mean=True`` used to *sample*); it is still accepted
        with its old meaning so existing callers keep their behaviour.
        """
        if sample is None:
            sample = bool(mean)
        N, dev = enc.size(0), enc.device
        h_t = torch.zeros(N, self.state_size).to(dev) if h_t is None else h_t
        s_t = torch.zeros(N, self.latent_size).to(dev) if s_t is None else s_t
        a_t = torch.zeros(N, self.action_size).to(dev) if a_t is None else a_t
        h_tp1 = self.deterministic_state_fwd(h_t, s_t, a_t)
        # Condition the posterior on the updated deterministic state, exactly as
        # training does; using h_t paired s_tp1 with a stale hidden state.
        if sample:
            s_tp1 = self.state_posterior(h_tp1, enc, sample=True)
        else:
            s_tp1, _ = self.state_posterior(h_tp1, enc)
        return h_tp1, s_tp1  # type: ignore[return-value]

    def deterministic_state_fwd(
        self, h_t: torch.Tensor, s_t: torch.Tensor, a_t: torch.Tensor
    ) -> torch.Tensor:
        """Deterministic transition update.

        Ensures a_t is 2D and matches batch dimension of h_t before concatenation.
        Accepts a_t shaped [B, action_size], [action_size] (expanded to [B, action_size]),
        or [B]/scalar (reshaped appropriately).
        """
        # ensure torch tensor
        if not isinstance(a_t, torch.Tensor):
            a_t = torch.tensor(a_t, dtype=h_t.dtype, device=h_t.device)

        # Fix dims: make a_t [B, action_size]
        if a_t.dim() == 0:
            a_t = a_t.view(1, 1)
        if a_t.dim() == 1:
            # If length equals batch size -> treat as per-batch scalar action -> make [B,1]
            if a_t.numel() == h_t.size(0):
                a_t = a_t.view(-1, 1).to(h_t.device).to(h_t.dtype)
            else:
                # treat as single action vector -> expand to batch
                a_t = (
                    a_t.view(1, -1).expand(h_t.size(0), -1).to(h_t.device).to(h_t.dtype)
                )
        elif a_t.dim() == 2 and a_t.size(0) != h_t.size(0):
            # if batch dim mismatches but first dim is 1, expand
            if a_t.size(0) == 1:
                a_t = a_t.expand(h_t.size(0), -1)
            else:
                raise ValueError(
                    f"Action batch size {a_t.size(0)} != state batch size {h_t.size(0)}"
                )

        # concatenate stochastic state and action, then pass through latent->state
        inp = torch.cat([s_t, a_t], dim=1)
        h = self.act_fn(self.lat_act_layer(inp))
        h = self.grucell(h, h_t)
        return h

    def state_prior(
        self, h_t: torch.Tensor, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Returns the prior distribution over the latent state given the deterministic state"""
        z = self.act_fn(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        s = F.softplus(self.fc_prior_s(z)) + 0.1
        s = torch.clamp(s, min=1e-6, max=10.0)
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def state_posterior(
        self, h_t: torch.Tensor, e_t: torch.Tensor, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Returns the state prior given the deterministic state and obs"""
        z = torch.cat([h_t, e_t], dim=1)
        z = self.act_fn(self.fc_posterior_1(z))
        m = self.fc_posterior_m(z)
        s = F.softplus(self.fc_posterior_s(z)) + 0.1
        s = torch.clamp(s, min=1e-6, max=10.0)
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def pred_reward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        r = self.act_fn(self.fc_reward_1(torch.cat([h_t, s_t], dim=-1)))
        r = self.act_fn(self.fc_reward_2(r))
        r = self.fc_reward_3(r)
        return r.squeeze()

    def rollout_prior(
        self, act: torch.Tensor, h_t: torch.Tensor, s_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states, latents = [], []
        for a_t in torch.unbind(act, dim=0):
            h_t = self.deterministic_state_fwd(h_t, s_t, a_t)
            s_t = self.state_prior(h_t, sample=True)  # type: ignore[assignment]
            states.append(h_t)
            latents.append(s_t)
        return torch.stack(states), torch.stack(latents)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> list[list]:
        """Forward through the RSSM for a batch of sequences.

        Args:
            x: Tensor [B, T+1, C, H, W]  (observations including initial frame)
            u: Tensor [B, T, action_size] (actions for T steps)

        Returns:
            states: list[T] of tensors [B, state_size]
            priors: list[T] of tuples (mean, std) each [B, latent_size]
            posteriors: list[T] of tuples (mean, std) each [B, latent_size]
        """
        B = x.size(0)
        T = u.size(1)
        device = x.device

        # encode all frames at once: [B*(T+1), C, H, W] -> [B, T+1, embed_size]
        x_flat = x.view(B * (T + 1), *x.shape[2:])
        e_flat = self.encoder(x_flat)
        e = e_flat.view(B, T + 1, -1)

        h = torch.zeros(B, self.state_size, device=device)
        s = torch.zeros(B, self.latent_size, device=device)

        states, priors, posteriors = [], [], []
        for t in range(T):
            a_t = u[:, t]
            h = self.deterministic_state_fwd(h, s, a_t)
            prior = self.state_prior(h)  # (m, s)
            posterior = self.state_posterior(h, e[:, t + 1])  # (m, s)

            states.append(h)
            priors.append(prior)
            posteriors.append(posterior)

            s = posterior[0]

        return [states, priors, posteriors]
