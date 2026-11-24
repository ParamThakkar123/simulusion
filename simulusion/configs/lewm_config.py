"""LeWorldModel's settings: the paper's and the authors' code by default.

Sources: Maes et al., 2026 (https://arxiv.org/abs/2603.19312), Sec. 3,
Appendices A, B and D; and the official code (https://github.com/lucas-maes/
le-wm, ``config/train/lewm.yaml``, ``config/train/model/lewm.yaml``,
``config/eval/solver/cem.yaml``). Where the two disagree the code is followed
and the field says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulusion.configs.serialization import SerializableConfigMixin


@dataclass
class LeWMConfig(SerializableConfigMixin):
    """Architecture, objective, optimization and planning of LeWM.

    Attributes:
        img_size: Frames are resized to this square size (224).
        in_channels: Frame channels.
        patch_size: ViT patch size (14).
        encoder_dim: ViT width (192, ViT-tiny).
        encoder_depth: ViT layers (12).
        encoder_heads: ViT heads (3).
        embed_dim: Width of the latent embedding (192, the [CLS] width).
        projector_hidden_dim: Hidden width of the BatchNorm projectors (2048).
        action_dim: Size of one raw environment action; set from the data.
        frameskip: Raw steps per model step; each step's action is the
            ``frameskip`` raw actions concatenated (5, App. D).
        action_smoothed_dim: Channels of the action embedder's 1x1 conv (10).
        action_mlp_scale: Hidden width of the action MLP, in units of
            ``embed_dim`` (4).
        history_size: Frames of context the predictor attends to (3; 1 for
            TwoRoom, App. D).
        num_preds: Steps ahead the training target lies (1).
        predictor_depth: Predictor blocks (6).
        predictor_heads: Predictor heads (16).
        predictor_dim_head: Width per predictor head (64).
        predictor_mlp_dim: Predictor MLP width (2048).
        predictor_dropout: Dropout inside the predictor (0.1, Table 9).
        predictor_emb_dropout: Dropout on the predictor's inputs (0).
        sigreg_weight: lambda in Eq. 3: 0.1, the paper's stated default
            (Sec. 3.1, Algorithm 1). Fig. 16 peaks near 0.09, the value the
            authors' code ships; anything in [0.01, 0.2] stays above 80% on
            Push-T.
        sigreg_num_proj: Random projections M (1024).
        sigreg_knots: Quadrature points of the Epps-Pulley integral (17).
        sigreg_t_max: Integration range ``[0, t_max]`` (3, as the code).
        lr: AdamW learning rate (5e-5).
        weight_decay: AdamW weight decay (1e-3).
        warmup_fraction: Linear warmup from 0 over this fraction of the
            optimizer steps, then cosine decay to 0 (0.01).
        grad_clip: Gradient-norm clip (1.0).
        batch_size: Windows per batch (128).
        epochs: Passes over the dataset (10 in the paper's experiments, App. E).
        action_stats_episodes: Episodes read to estimate the action z-score
            statistics; ``0`` reads all of them.
        plan_horizon: Planning horizon in model steps (5, i.e. 25 raw steps).
        plan_samples: CEM candidates per iteration (300).
        plan_iterations: CEM iterations (30 for PushT, 10 elsewhere).
        plan_elites: CEM elites (30).
        plan_init_std: Initial CEM standard deviation, in z-scored action
            units (1).
    """

    img_size: int = 224
    in_channels: int = 3
    patch_size: int = 14
    encoder_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    embed_dim: int = 192
    projector_hidden_dim: int = 2048

    action_dim: int = 0
    frameskip: int = 5
    action_smoothed_dim: int = 10
    action_mlp_scale: int = 4

    history_size: int = 3
    num_preds: int = 1
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0

    sigreg_weight: float = 0.1
    sigreg_num_proj: int = 1024
    sigreg_knots: int = 17
    sigreg_t_max: float = 3.0

    lr: float = 5e-5
    weight_decay: float = 1e-3
    warmup_fraction: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 128
    epochs: int = 10
    action_stats_episodes: int = 1000

    plan_horizon: int = 5
    plan_samples: int = 300
    plan_iterations: int = 30
    plan_elites: int = 30
    plan_init_std: float = 1.0

    @property
    def window_frames(self) -> int:
        """Raw frames one training window spans: model steps x frame skip."""
        return (self.history_size + self.num_preds - 1) * self.frameskip + 1


__all__ = ["LeWMConfig"]
