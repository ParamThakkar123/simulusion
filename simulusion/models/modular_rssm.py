"""Modular RSSM with swappable encoder/decoder/backbone components.

This module provides a flexible architecture for world model research,
allowing researchers to easily swap different encoder, decoder, and backbone
implementations for ablations and experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from simulusion.export import ExportableAgentMixin

_str_to_activation = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
    "gelu": nn.GELU(),
}


class EncoderBase(nn.Module, ABC):
    """Abstract base class for observation encoders."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to embeddings."""
        pass

    # Annotate expected attribute so type checkers know subclasses expose it.
    embed_size: int

    def get_embed_size(self) -> int:
        """Return the embedding size. Override in subclasses."""
        raise NotImplementedError


class DecoderBase(nn.Module, ABC):
    """Abstract base class for observation decoders."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> Any:
        """Decode latent features to observation distributions."""
        pass


class BackboneBase(nn.Module, ABC):
    """Abstract base class for recurrent dynamics backbones."""

    @abstractmethod
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None,
        nonterm: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Process one step of dynamics. Returns (prior, posterior)."""
        pass

    @abstractmethod
    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Initialize hidden state."""
        pass

    # Expose expected size attributes to help mypy understand types on
    # instances of concrete backbones.
    stoch_size: int
    deter_size: int


class ConvEncoder(EncoderBase):
    """Convolutional encoder from Dreamer (image observations)."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        embed_size: int,
        activation: str = "elu",
        depth: int = 32,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.embed_size = embed_size
        self.act_fn = _str_to_activation[activation]
        self.depth = depth
        self.kernels = [4, 4, 4, 4]

        # Use a generic module list since we append diverse nn.Module objects
        layers: list[nn.Module] = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = input_shape[0] if i == 0 else self.depth * (2 ** (i - 1))
            out_ch = self.depth * (2**i)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2))
            layers.append(self.act_fn)

        self.conv_block = nn.Sequential(*layers)
        self.fc = (
            nn.Linear(1024, self.embed_size)
            if self.embed_size != 1024
            else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        reshaped = obs.reshape(-1, *self.input_shape)
        embed = self.conv_block(reshaped)
        embed = torch.reshape(embed, (*obs.shape[:-3], -1))
        embed = self.fc(embed)
        return embed


class MLPEncoder(EncoderBase):
    """MLP encoder for state-based observations."""

    def __init__(
        self,
        input_dim: int,
        embed_size: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "elu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_size = embed_size
        self.act_fn = _str_to_activation[activation]

        # Use a generic module list since we append diverse nn.Module objects
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden_size), self.act_fn])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, embed_size))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class ViTEncoder(EncoderBase):
    """Vision Transformer encoder for image observations."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        embed_size: int,
        patch_size: int = 8,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.embed_size = embed_size
        self.patch_size = patch_size

        c, h, w = input_shape
        self.num_patches = (h // patch_size) * (w // patch_size)

        self.patch_embed = nn.Conv2d(
            c, embed_size, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_size))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, mlp_ratio, activation)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        x = self.patch_embed(obs)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        return self.norm(x[:, 0])


class TransformerBlock(nn.Module):
    """Transformer block for ViT encoder."""

    def __init__(
        self, embed_size: int, num_heads: int, mlp_ratio: float, activation: str
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, int(embed_size * mlp_ratio)),
            _str_to_activation[activation],
            nn.Linear(int(embed_size * mlp_ratio), embed_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ConvDecoder(DecoderBase):
    """Convolutional decoder for image observations."""

    def __init__(
        self,
        stoch_size: int,
        deter_size: int,
        output_shape: Tuple[int, int, int],
        activation: str = "elu",
        depth: int = 32,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.depth = depth
        self.kernels = [5, 5, 6, 6]
        self.act_fn = _str_to_activation[activation]

        self.dense = nn.Linear(stoch_size + deter_size, 32 * depth)

        layers: list[nn.Module] = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = 32 * depth if i == 0 else depth * (2 ** (len(self.kernels) - 1 - i))
            out_ch = (
                output_shape[0]
                if i == len(self.kernels) - 1
                else depth * (2 ** (len(self.kernels) - 2 - i))
            )
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2))
            if i != len(self.kernels) - 1:
                layers.append(self.act_fn)

        self.convtranspose = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Any:
        out_batch_shape = features.shape[:-1]
        out = self.dense(features)
        out = torch.reshape(out, [-1, 32 * self.depth, 1, 1])
        out = self.convtranspose(out)
        mean = torch.reshape(out, (*out_batch_shape, *self.output_shape))
        return distributions.independent.Independent(
            distributions.Normal(mean, 1), len(self.output_shape)
        )


class MLPDecoder(DecoderBase):
    """MLP decoder for state-based observations."""

    def __init__(
        self,
        stoch_size: int,
        deter_size: int,
        output_dim: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "elu",
        dist: str = "normal",
    ):
        super().__init__()
        self.input_size = stoch_size + deter_size
        self.output_dim = output_dim
        self.act_fn = _str_to_activation[activation]
        self.dist = dist

        layers = []
        in_dim = self.input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden_size), self.act_fn])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Any:
        out = self.model(features)
        if self.dist == "normal":
            return distributions.independent.Independent(
                distributions.Normal(out, 1), 1
            )
        return out


class GRUBackbone(BackboneBase):
    """GRU-based recurrent dynamics backbone (standard RSSM)."""

    def __init__(
        self,
        action_size: int,
        stoch_size: int,
        deter_size: int,
        hidden_size: int,
        embed_size: int,
        activation: str = "elu",
    ):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.act_fn = _str_to_activation[activation]

        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_state_action = nn.Linear(self.stoch_size + action_size, self.deter_size)
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior = nn.Linear(self.hidden_size, 2 * self.stoch_size)
        self.fc_embed_posterior = nn.Linear(
            self.embed_size + self.deter_size, self.hidden_size
        )
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2 * self.stoch_size)

    @property
    def embedding_size(self) -> int:
        return self.embed_size

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "mean": torch.zeros(batch_size, self.stoch_size).to(device),
            "std": torch.zeros(batch_size, self.stoch_size).to(device),
            "stoch": torch.zeros(batch_size, self.stoch_size).to(device),
            "deter": torch.zeros(batch_size, self.deter_size).to(device),
        }

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None,
        nonterm: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prior = self._imagine_step(state, action, nonterm)

        if obs_embed is not None:
            posterior_embed = self.act_fn(
                self.fc_embed_posterior(torch.cat([obs_embed, prior["deter"]], dim=-1))
            )
            posterior = self.fc_state_posterior(posterior_embed)
            mean, std = torch.chunk(posterior, 2, dim=-1)
            std = F.softplus(std) + 0.1
            sample = mean + torch.randn_like(mean) * std
            posterior = {
                "mean": mean,
                "std": std,
                "stoch": sample,
                "deter": prior["deter"],
            }
        else:
            posterior = prior

        return prior, posterior

    def _imagine_step(
        self, state: Dict[str, torch.Tensor], action: torch.Tensor, nonterm: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        state_action = self.act_fn(
            self.fc_state_action(torch.cat([state["stoch"] * nonterm, action], dim=-1))
        )
        deter = self.rnn(state_action, state["deter"] * nonterm)
        prior_embed = self.act_fn(self.fc_embed_prior(deter))
        mean, std = torch.chunk(self.fc_state_prior(prior_embed), 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std
        return {"mean": mean, "std": std, "stoch": sample, "deter": deter}


class LSTMBackbone(BackboneBase):
    """LSTM-based recurrent dynamics backbone."""

    def __init__(
        self,
        action_size: int,
        stoch_size: int,
        deter_size: int,
        hidden_size: int,
        embed_size: int,
        activation: str = "elu",
    ):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.act_fn = _str_to_activation[activation]

        self.rnn = nn.LSTMCell(self.deter_size, self.deter_size)
        self.fc_state_action = nn.Linear(self.stoch_size + action_size, self.deter_size)
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior = nn.Linear(self.hidden_size, 2 * self.stoch_size)
        self.fc_embed_posterior = nn.Linear(
            self.embed_size + self.deter_size, self.hidden_size
        )
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2 * self.stoch_size)

    @property
    def embedding_size(self) -> int:
        return self.embed_size

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "mean": torch.zeros(batch_size, self.stoch_size).to(device),
            "std": torch.zeros(batch_size, self.stoch_size).to(device),
            "stoch": torch.zeros(batch_size, self.stoch_size).to(device),
            "deter": torch.zeros(batch_size, self.deter_size).to(device),
            "cell": torch.zeros(batch_size, self.deter_size).to(device),
        }

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None,
        nonterm: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prior = self._imagine_step(state, action, nonterm)

        if obs_embed is not None:
            posterior_embed = self.act_fn(
                self.fc_embed_posterior(torch.cat([obs_embed, prior["deter"]], dim=-1))
            )
            posterior = self.fc_state_posterior(posterior_embed)
            mean, std = torch.chunk(posterior, 2, dim=-1)
            std = F.softplus(std) + 0.1
            sample = mean + torch.randn_like(mean) * std
            posterior = {
                "mean": mean,
                "std": std,
                "stoch": sample,
                "deter": prior["deter"],
                "cell": prior.get(
                    "cell", state.get("cell", torch.zeros_like(prior["deter"]))
                ),
            }
        else:
            posterior = prior

        return prior, posterior

    def _imagine_step(
        self, state: Dict[str, torch.Tensor], action: torch.Tensor, nonterm: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        state_action = self.act_fn(
            self.fc_state_action(torch.cat([state["stoch"] * nonterm, action], dim=-1))
        )
        h, c = self.rnn(
            state_action,
            (
                state["deter"] * nonterm,
                state.get("cell", torch.zeros_like(state["deter"])) * nonterm,
            ),
        )
        prior_embed = self.act_fn(self.fc_embed_prior(h))
        mean, std = torch.chunk(self.fc_state_prior(prior_embed), 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std
        return {"mean": mean, "std": std, "stoch": sample, "deter": h, "cell": c}


class TransformerBackbone(BackboneBase):
    """Transformer-based dynamics backbone for long-range dependencies."""

    def __init__(
        self,
        action_size: int,
        stoch_size: int,
        deter_size: int,
        embed_size: int,
        num_heads: int = 4,
        num_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.embed_size = embed_size
        self.act_fn = _str_to_activation[activation]

        self.action_embed = nn.Linear(action_size, embed_size)
        self.stoch_embed = nn.Linear(stoch_size, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=deter_size,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_embed_prior = nn.Linear(embed_size, deter_size)
        self.fc_state_prior = nn.Linear(deter_size, 2 * stoch_size)
        self.fc_embed_posterior = nn.Linear(embed_size + deter_size, deter_size)
        self.fc_state_posterior = nn.Linear(deter_size, 2 * stoch_size)

    @property
    def embedding_size(self) -> int:
        return self.embed_size

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "mean": torch.zeros(batch_size, self.stoch_size).to(device),
            "std": torch.zeros(batch_size, self.stoch_size).to(device),
            "stoch": torch.zeros(batch_size, self.stoch_size).to(device),
            "deter": torch.zeros(batch_size, self.deter_size).to(device),
        }

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None,
        nonterm: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prior = self._imagine_step(state, action, nonterm)

        if obs_embed is not None:
            # Handle both 2D (batch, embed) and 3D (seq, batch, embed) embeddings
            if obs_embed.dim() == 3:
                # Sequence data: need to handle matching dimensions
                seq_len = obs_embed.shape[0]
                prior_deter_expanded = (
                    prior["deter"].unsqueeze(0).expand(seq_len, -1, -1)
                )
            else:
                # Single step: 2D embeddings
                prior_deter_expanded = prior["deter"]

            posterior_embed = self.act_fn(
                self.fc_embed_posterior(
                    torch.cat([obs_embed, prior_deter_expanded], dim=-1)
                )
            )
            posterior = self.fc_state_posterior(posterior_embed)
            mean, std = torch.chunk(posterior, 2, dim=-1)
            std = F.softplus(std) + 0.1
            sample = mean + torch.randn_like(mean) * std
            posterior = {
                "mean": mean,
                "std": std,
                "stoch": sample,
                "deter": prior["deter"],
            }
        else:
            posterior = prior

        return prior, posterior

    def _imagine_step(
        self, state: Dict[str, torch.Tensor], action: torch.Tensor, nonterm: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        action_emb = self.action_embed(action)
        stoch_emb = self.stoch_embed(state["stoch"] * nonterm)
        x = action_emb + stoch_emb
        x = x.unsqueeze(0) if x.dim() == 2 else x
        x = self.transformer(x)
        h = x.squeeze(0) if x.shape[0] == 1 else x

        prior_embed = self.act_fn(self.fc_embed_prior(h))
        mean, std = torch.chunk(self.fc_state_prior(prior_embed), 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std
        return {"mean": mean, "std": std, "stoch": sample, "deter": prior_embed}


class ModularRSSM(ExportableAgentMixin, nn.Module):
    """Modular RSSM with swappable encoder, decoder, and backbone.

    This class allows researchers to easily experiment with different:
    - Encoders: Conv, MLP, ViT
    - Decoders: Conv, MLP
    - Backbones: GRU, LSTM, Transformer

    Example:
        >>> encoder = ConvEncoder((3, 64, 64), embed_size=1024)
        >>> decoder = ConvDecoder(32, 200, (3, 64, 64))
        >>> backbone = GRUBackbone(action_size=6, stoch_size=32, deter_size=200, hidden_size=200, embed_size=1024)
        >>> rssm = ModularRSSM(encoder, decoder, backbone)
    """

    def __init__(
        self,
        encoder: EncoderBase,
        decoder: DecoderBase,
        backbone: BackboneBase,
        reward_decoder: Optional[DecoderBase] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.backbone = backbone
        self.reward_decoder = reward_decoder

    @property
    def stoch_size(self) -> int:
        return self.backbone.stoch_size

    @property
    def deter_size(self) -> int:
        return self.backbone.deter_size

    @property
    def embed_size(self) -> int:
        # Some backbones may expose embedding_size as an int attribute or
        # property. Cast to int to make the return type explicit for
        # type-checkers.
        return int(self.backbone.embedding_size)  # type: ignore[arg-type]

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return self.backbone.init_state(batch_size, device)

    def get_dist(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> distributions.Distribution:
        distribution = distributions.Normal(mean, std)
        return distributions.independent.Independent(distribution, 1)

    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        obs: torch.Tensor,
        nonterm: Any = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode ``obs`` and advance one step.

        Returns:
            ``(prior, posterior)``. Note the order: the core Dreamer
            :meth:`simulusion.models.dreamer_rssm.RSSM.observe_step` returns
            ``(posterior, prior)``. Both states have identical keys, so mixing
            them up raises no error; the posterior is the one that has seen
            ``obs`` and is what should be acted on and carried forward.
        """
        obs_embed = self.encoder(obs)
        prior, posterior = self.backbone.forward(
            prev_state, prev_action, obs_embed, nonterm
        )
        return prior, posterior

    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        nonterm: Any = 1.0,
    ) -> Dict[str, torch.Tensor]:
        prior, _ = self.backbone.forward(prev_state, prev_action, None, nonterm)
        return prior

    def observe_rollout(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        nonterms: torch.Tensor,
        prev_state: Dict[str, torch.Tensor],
        horizon: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        priors = []
        posteriors = []

        for t in range(horizon):
            nonterm_t = nonterms[t]
            if nonterm_t.dim() > 1:
                nonterm_t = nonterm_t.squeeze(-1)
            elif nonterm_t.dim() == 1:
                nonterm_t = nonterm_t.unsqueeze(-1)
            prev_action = actions[t] * nonterm_t
            prior_state, posterior_state = self.observe_step(
                prev_state, prev_action, obs[t], 1.0
            )
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        return self._stack_states(priors), self._stack_states(posteriors)

    def imagine_rollout(
        self,
        actor: nn.Module,
        prev_state: Dict[str, torch.Tensor],
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        rssm_state = prev_state
        next_states = []

        for _ in range(horizon):
            features = torch.cat(
                [rssm_state["stoch"], rssm_state["deter"]], dim=-1
            ).detach()
            action = actor(features)
            rssm_state = self.imagine_step(rssm_state, action)
            next_states.append(rssm_state)

        return self._stack_states(next_states)

    def decode_observation(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def decode_reward(self, features: torch.Tensor) -> torch.Tensor:
        if self.reward_decoder is None:
            raise ValueError("Reward decoder not provided")
        return self.reward_decoder(features)

    def _stack_states(
        self, states: List[Dict[str, torch.Tensor]], dim: int = 0
    ) -> Dict[str, torch.Tensor]:
        return {
            "mean": torch.stack([s["mean"] for s in states], dim=dim),
            "std": torch.stack([s["std"] for s in states], dim=dim),
            "stoch": torch.stack([s["stoch"] for s in states], dim=dim),
            "deter": torch.stack([s["deter"] for s in states], dim=dim),
        }

    def detach_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach() for k, v in state.items()}

    def seq_to_batch(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:]) if v.dim() > 2 else v
            for k, v in state.items()
        }


def create_modular_rssm(
    encoder_type: str = "conv",
    decoder_type: str = "conv",
    backbone_type: str = "gru",
    obs_shape: Union[Tuple[int, int, int], Tuple[int]] = (3, 64, 64),
    action_size: int = 6,
    stoch_size: int = 32,
    deter_size: int = 200,
    embed_size: int = 1024,
    hidden_size: int = 200,
    activation: str = "elu",
    **kwargs: Any,
) -> ModularRSSM:
    """Factory function to create a modular RSSM with specified components.

    Args:
        encoder_type: Type of encoder ("conv", "mlp", "vit")
        decoder_type: Type of decoder ("conv", "mlp")
        backbone_type: Type of backbone ("gru", "lstm", "transformer")
        obs_shape: Shape of observations (C, H, W) for images or (D,) for state
        action_size: Action space dimension
        stoch_size: Stochastic latent dimension
        deter_size: Deterministic hidden dimension
        embed_size: Encoder embedding dimension
        hidden_size: Hidden layer dimension
        activation: Activation function name

    Returns:
        Configured ModularRSSM instance
    """
    image_shape: Tuple[int, int, int] = (3, 64, 64)
    if len(obs_shape) == 3:
        image_shape = (int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2]))

    # Declare with base types so mypy understands the variable may hold any
    # concrete implementation chosen below.
    encoder: EncoderBase
    if encoder_type == "conv":
        encoder = ConvEncoder(image_shape, embed_size, activation)
    elif encoder_type == "mlp":
        encoder = MLPEncoder(obs_shape[0], embed_size)
    elif encoder_type == "vit":
        encoder = ViTEncoder(image_shape, embed_size)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    decoder: DecoderBase
    if decoder_type == "conv":
        decoder = ConvDecoder(stoch_size, deter_size, image_shape, activation)
    elif decoder_type == "mlp":
        decoder = MLPDecoder(
            stoch_size, deter_size, obs_shape[0], activation=activation
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    backbone: BackboneBase
    if backbone_type == "gru":
        backbone = GRUBackbone(
            action_size, stoch_size, deter_size, hidden_size, embed_size, activation
        )
    elif backbone_type == "lstm":
        backbone = LSTMBackbone(
            action_size, stoch_size, deter_size, hidden_size, embed_size, activation
        )
    elif backbone_type == "transformer":
        num_heads = kwargs.get("num_heads", 4)
        num_layers = kwargs.get("num_layers", 2)
        backbone_activation = "gelu"  # PyTorch Transformer only supports relu/gelu
        backbone = TransformerBackbone(
            action_size,
            stoch_size,
            deter_size,
            embed_size,
            num_heads,
            num_layers,
            backbone_activation,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    reward_decoder = MLPDecoder(
        stoch_size, deter_size, 1, activation=activation, dist="normal"
    )

    return ModularRSSM(encoder, decoder, backbone, reward_decoder)
