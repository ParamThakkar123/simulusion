"""LeWorldModel (LeWM): a JEPA world model trained end to end from pixels.

Maes, Le Lidec, Scieur, LeCun and Balestriero, "LeWorldModel: Stable
End-to-End Joint-Embedding Predictive Architecture from Pixels", 2026
(https://arxiv.org/abs/2603.19312).

A ViT encoder maps each frame to an embedding, and a causal transformer
predicts the next frame's embedding from the history and the actions. The two
train jointly on two terms (Eq. 3)::

    loss = MSE(predicted next embedding, encoded next embedding)
           + lambda * SIGReg(embeddings)

SIGReg (:class:`SIGReg`) keeps the embeddings from collapsing by pushing their
distribution towards an isotropic Gaussian: it projects them onto random
directions and penalises each projection's Epps-Pulley distance from a
standard normal. There is no stop-gradient, EMA target, reconstruction or
reward. To act, a planner optimizes actions so the predicted embedding reaches
a goal frame's embedding (:class:`~simulusion.agents.GoalCEMPlanner`).

Every default follows the paper and the authors' code (MIT,
https://github.com/lucas-maes/le-wm): see :class:`~simulusion.configs.
lewm_config.LeWMConfig`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from simulusion.export import ExportableAgentMixin

from simulusion.vision.lewm_encoder import LeWMEncoder
from simulusion.blocks.causal_attention import CausalAttention

#: ImageNet statistics; the encoder sees frames normalized with them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ SIGReg


def epps_pulley(
    projections: torch.Tensor, knots: int = 17, t_max: float = 3.0
) -> torch.Tensor:
    """Epps-Pulley statistic of each column of ``projections`` against N(0, 1).

    The weighted squared distance between the empirical characteristic
    function of the samples and the standard normal's, ``exp(-t^2 / 2)``
    (Appendix A, Eq. EP), with weight ``w(t) = exp(-t^2 / 2)``. The integral
    over the real line is twice the one over ``[0, t_max]`` (the integrand is
    even), taken with the trapezoid rule on ``knots`` points; the result is
    scaled by the sample count, as a test statistic is.

    Args:
        projections: ``(..., N, M)``: ``N`` samples of ``M`` univariate
            projections; leading dimensions are kept.
        knots: Quadrature points.
        t_max: Upper end of the integration range.

    Returns:
        ``(..., M)``, one statistic per projection.
    """
    t = torch.linspace(0, t_max, knots, device=projections.device)
    dt = t_max / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=projections.device)
    weights[[0, -1]] = dt
    target = torch.exp(-t.square() / 2.0)
    x_t = projections.float().unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - target).square() + x_t.sin().mean(-3).square()
    return (err @ (weights * target)) * projections.size(-2)


class SIGReg(ExportableAgentMixin, nn.Module):
    """Sketched Isotropic Gaussian Regularizer (Eq. 2 and Appendix A).

    Projects embeddings onto ``num_proj`` random unit directions, drawn afresh
    on every call, and averages the :func:`epps_pulley` statistic of each
    projection. By the Cramer-Wold theorem the embeddings match ``N(0, I)``
    when every one-dimensional projection matches ``N(0, 1)``.

    Args:
        knots: Quadrature points of the Epps-Pulley integral (17).
        num_proj: Random projections per call (1024).
        t_max: Upper end of the integration range (3).
    """

    t: torch.Tensor
    phi: torch.Tensor
    weights: torch.Tensor

    def __init__(self, knots: int = 17, num_proj: int = 1024, t_max: float = 3.0):
        super().__init__()
        self.knots = int(knots)
        self.num_proj = int(num_proj)
        self.t_max = float(t_max)
        t = torch.linspace(0, self.t_max, self.knots)
        dt = self.t_max / (self.knots - 1)
        weights = torch.full((self.knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """``embeddings``: ``(T, B, D)``, one SIGReg per time step, averaged.

        A 2-D ``(B, D)`` input is one time step.
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        directions = torch.randn(
            embeddings.size(-1), self.num_proj, device=embeddings.device
        )
        directions = directions / directions.norm(p=2, dim=0)
        x_t = (embeddings.float() @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * embeddings.size(-2)
        return statistic.mean()


# --------------------------------------------------------------- predictor


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLNBlock(ExportableAgentMixin, nn.Module):
    """Causal transformer block conditioned on actions with AdaLN-zero.

    The action embedding yields a shift, a scale and a gate for each of the
    attention and MLP branches; the projection producing them starts at zero,
    so action conditioning enters training progressively (Sec. 3.1).
    """

    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = CausalAttention(dim, heads, dim_head, dropout)
        self.mlp = _FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        modulation = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), modulation)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(
            cond
        ).chunk(6, dim=-1)
        x = x + gate_a * self.attn(_modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x


class ARPredictor(ExportableAgentMixin, nn.Module):
    """Autoregressive latent predictor: embeddings and actions -> next embeddings.

    A causal transformer over the history of frame embeddings, with learned
    position embeddings and every block conditioned on that step's action
    embedding (:class:`AdaLNBlock`). Output ``t`` predicts the embedding of
    frame ``t + 1`` from frames ``<= t``.

    Args:
        num_frames: Longest history (position embeddings).
        dim: Width of embeddings, actions and the transformer.
        depth: Blocks.
        heads: Attention heads.
        dim_head: Width per head.
        mlp_dim: Hidden width of each block's MLP.
        dropout: Dropout inside the blocks.
        emb_dropout: Dropout on the input embeddings.
    """

    def __init__(
        self,
        *,
        num_frames: int = 3,
        dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [AdaLNBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``(B, T, D)`` embeddings and ``(B, T, D)`` action embeddings -> ``(B, T, D)``."""
        T = embeddings.size(1)
        if T > self.num_frames:
            raise ValueError(
                f"The predictor attends to at most {self.num_frames} frames, got {T}."
            )
        x = self.dropout(embeddings + self.pos_embedding[:, :T])
        for layer in self.layers:
            x = layer(x, actions)
        return self.norm(x)


class ActionEmbedder(ExportableAgentMixin, nn.Module):
    """Embed each step's (frame-skipped) action block.

    A 1x1 convolution to ``smoothed_dim`` channels, then a two-layer SiLU MLP
    to ``dim``. With a frame skip of ``k``, the input of each step is its
    ``k`` actions concatenated.
    """

    def __init__(
        self, input_dim: int, dim: int = 192, smoothed_dim: int = 10, mlp_scale: int = 4
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * dim, dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """``(B, T, input_dim)`` -> ``(B, T, dim)``."""
        x = self.patch_embed(actions.float().transpose(1, 2)).transpose(1, 2)
        return self.embed(x)


class BNProjector(ExportableAgentMixin, nn.Module):
    """Linear, BatchNorm, GELU, Linear: the projection after encoder and predictor.

    The encoder's last LayerNorm keeps SIGReg from shaping its output
    directly; projecting through a BatchNorm MLP fixes that (Sec. 3.1).
    """

    def __init__(self, dim: int = 192, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return self.net(x.reshape(-1, shape[-1])).reshape(shape)


# ------------------------------------------------------------------- model


class LeWM(ExportableAgentMixin, nn.Module):
    """LeWorldModel: encoder, projector, action embedder, predictor, SIGReg.

    Frames are float in ``[0, 1]``; they are resized to ``config.img_size``
    and normalized with ImageNet statistics here. Actions are raw environment
    actions, frame-skipped into blocks of ``config.frameskip`` (so one step's
    action has ``frameskip * action_dim`` values) and z-scored with
    :attr:`action_mean` / :attr:`action_std`, which training fills from the
    dataset.

    Args:
        config: A :class:`~simulusion.configs.lewm_config.LeWMConfig`.
        action_dim: Dimensions of one raw environment action.
    """

    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    def __init__(self, config: Any = None, *, action_dim: Optional[int] = None) -> None:
        super().__init__()
        from simulusion.configs.lewm_config import LeWMConfig

        cfg = config if config is not None else LeWMConfig()
        if action_dim is not None:
            cfg.action_dim = int(action_dim)
        if not cfg.action_dim:
            raise ValueError("LeWM needs action_dim, the size of one raw action.")
        self.config = cfg
        block = cfg.frameskip * cfg.action_dim
        self.encoder = LeWMEncoder(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_channels,
            embed_dim=cfg.encoder_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.encoder_heads,
        )
        self.projector = BNProjector(cfg.encoder_dim, cfg.projector_hidden_dim)
        if cfg.embed_dim != cfg.encoder_dim:
            raise ValueError(
                "embed_dim must equal encoder_dim: the projector keeps the width."
            )
        self.action_encoder = ActionEmbedder(
            block, cfg.embed_dim, cfg.action_smoothed_dim, cfg.action_mlp_scale
        )
        self.predictor = ARPredictor(
            num_frames=cfg.history_size,
            dim=cfg.embed_dim,
            depth=cfg.predictor_depth,
            heads=cfg.predictor_heads,
            dim_head=cfg.predictor_dim_head,
            mlp_dim=cfg.predictor_mlp_dim,
            dropout=cfg.predictor_dropout,
            emb_dropout=cfg.predictor_emb_dropout,
        )
        self.pred_proj = BNProjector(cfg.embed_dim, cfg.projector_hidden_dim)
        self.sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_num_proj, cfg.sigreg_t_max)
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("action_mean", torch.zeros(block))
        self.register_buffer("action_std", torch.ones(block))

    @property
    def action_block_dim(self) -> int:
        """Values in one step's action: ``frameskip * action_dim``."""
        return self.config.frameskip * self.config.action_dim

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the z-score statistics of one raw action, ``(action_dim,)`` each.

        They are tiled over the frame-skip block.
        """
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        k = self.config.frameskip
        self.action_mean.copy_(mean.repeat(k))
        self.action_std.copy_(std.clamp_min(1e-6).repeat(k))

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw action blocks -> z-scored; NaN (episode padding) becomes 0."""
        z = (actions.float() - self.action_mean) / self.action_std
        return torch.nan_to_num(z, nan=0.0)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.float() * self.action_std + self.action_mean

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W)`` frames in ``[0, 1]`` -> resized and normalized."""
        size = self.config.img_size
        frames = frames.float()
        if tuple(frames.shape[-2:]) != (size, size):
            frames = F.interpolate(
                frames,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (frames - self.pixel_mean) / self.pixel_std

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """``(B, T, C, H, W)`` frames in ``[0, 1]`` -> ``(B, T, D)`` embeddings."""
        B, T = frames.shape[:2]
        cls = self.encoder(self.preprocess(frames.flatten(0, 1)))
        return self.projector(cls).reshape(B, T, -1)

    def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw ``(B, T, frameskip * action_dim)`` action blocks -> ``(B, T, D)``."""
        return self.action_encoder(self.normalize_actions(actions))

    def predict(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``(B, T, D)`` embeddings and raw action blocks -> next embeddings ``(B, T, D)``."""
        out = self.predictor(embeddings, self.embed_actions(actions))
        return self.pred_proj(out)

    def loss(
        self, frames: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The LeWM objective on frame-skipped windows (Algorithm 1).

        Args:
            frames: ``(B, history + num_preds, C, H, W)`` in ``[0, 1]``.
            actions: ``(B, >= history, frameskip * action_dim)`` raw action
                blocks; ``actions[:, t]`` is taken at frame ``t``.

        Returns:
            ``loss``, ``pred_loss`` and ``sigreg_loss``.
        """
        cfg = self.config
        H, n = cfg.history_size, cfg.num_preds
        embeddings = self.encode(frames)
        predicted = self.predict(embeddings[:, :H], actions[:, :H])
        pred_loss = (predicted - embeddings[:, n : n + H]).pow(2).mean()
        sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
        return {
            "loss": pred_loss + cfg.sigreg_weight * sigreg_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
        }

    @torch.no_grad()
    def rollout(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ahead autoregressively, as the planner does.

        Args:
            embeddings: ``(B, T, D)`` context embeddings.
            actions: ``(B, T, A)`` raw action blocks taken at the context
                frames (the last one leads to the first prediction).
            future_actions: ``(B, H, A)``: the actions after that.

        Returns:
            ``(B, T + H + 1, D)``: the context, then one predicted embedding per
            action in ``actions[:, -1]`` and ``future_actions``.
        """
        K = self.config.history_size
        emb, act = embeddings, actions
        steps = future_actions.shape[1]
        for t in range(steps + 1):
            nxt = self.predict(emb[:, -K:], act[:, -K:])[:, -1:]
            emb = torch.cat([emb, nxt], dim=1)
            if t < steps:
                act = torch.cat([act, future_actions[:, t : t + 1]], dim=1)
        return emb


def temporal_straightness(embeddings: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity of consecutive latent velocities (Eq. 9).

    ``embeddings``: ``(B, T, D)`` with ``T >= 3``. Near 1, trajectories are
    nearly straight lines in latent space.
    """
    v = embeddings[:, 1:] - embeddings[:, :-1]
    return F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1).mean()


def surprise(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Per-step prediction error, the violation-of-expectation signal (Sec. 5.2).

    ``(B, T, D)`` predicted and observed embeddings -> ``(B, T)`` mean squared
    error.
    """
    return (predicted - observed).pow(2).mean(-1)


def lr_lambda(warmup_steps: int, max_steps: int) -> Any:
    """Linear warmup from 0, then cosine annealing to 0, as a ``LambdaLR`` factor."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        span = max(1, max_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


__all__ = [
    "ARPredictor",
    "ActionEmbedder",
    "AdaLNBlock",
    "BNProjector",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LeWM",
    "LeWMEncoder",
    "SIGReg",
    "epps_pulley",
    "lr_lambda",
    "surprise",
    "temporal_straightness",
]
