"""Configuration for DreamerV3.

:class:`DreamerV3Config` extends :class:`~simulusion.configs.dreamer_config.DreamerConfig`
so every environment backend option carries over unchanged, then overrides the
optimization defaults and adds the fields introduced by DreamerV3. The defaults
reproduce Table 4 of the paper and are intended to be used unmodified across
domains -- that fixed-hyperparameter property is the central claim of the work.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from dataclasses import dataclass

from simulusion.configs.dreamer_config import DreamerConfig

__all__ = ["DreamerV3Config", "MODEL_SIZES"]


# Table 3: the model dimension `d` determines every other width. Recurrent units
# are `8d` (split into 8 blocks of size `d`), while convolution channels and
# latent classes are `d / 16`. Layer and latent counts stay fixed across sizes.
MODEL_SIZES: dict[str, dict[str, int]] = {
    "12m": {
        "hidden_size": 256,
        "recurrent_units": 1024,
        "cnn_depth": 16,
        "latent_classes": 16,
    },
    "25m": {
        "hidden_size": 384,
        "recurrent_units": 3072,
        "cnn_depth": 24,
        "latent_classes": 24,
    },
    "50m": {
        "hidden_size": 512,
        "recurrent_units": 4096,
        "cnn_depth": 32,
        "latent_classes": 32,
    },
    "100m": {
        "hidden_size": 768,
        "recurrent_units": 6144,
        "cnn_depth": 48,
        "latent_classes": 48,
    },
    "200m": {
        "hidden_size": 1024,
        "recurrent_units": 8192,
        "cnn_depth": 64,
        "latent_classes": 64,
    },
    "400m": {
        "hidden_size": 1536,
        "recurrent_units": 12288,
        "cnn_depth": 96,
        "latent_classes": 96,
    },
}


@dataclass
class DreamerV3Config(DreamerConfig):
    """Hyperparameters for DreamerV3 training and evaluation.

    Width fields left as ``None`` are filled in from ``model_size`` when the
    config is constructed, so ``DreamerV3Config(model_size="200m")`` is all that
    is needed to scale the agent up. Setting a width explicitly overrides the
    preset for that field only.
    """

    algo: str = "DreamerV3"

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    # One of MODEL_SIZES. The paper reports monotonically better performance and
    # lower data requirements as this grows, with all other values held fixed.
    model_size: str = "12m"
    hidden_size: int | None = None
    recurrent_units: int | None = None
    cnn_depth: int | None = None
    latent_classes: int | None = None
    latent_dim: int = 32
    gru_blocks: int = 8
    # Encoder/decoder (vector inputs), actor and critic are 3-layer MLPs; the
    # reward and continue predictors are 1-layer MLPs (paper, Methods).
    mlp_layers: int = 3
    reward_layers: int = 1
    cont_layers: int = 1
    # Observations to encode and reconstruct, e.g. ["image", "instr"] or
    # "state". Images are stacked into one CNN, vectors into one symlog MLP.
    # None picks "image" if the environment has one, else "state", plus
    # "instr" when present.
    obs_keys: list[str] | None = None
    activation: str = "silu"
    # Uniform mixture for world model latents and the discrete policy, which
    # keeps categorical distributions from becoming deterministic.
    unimix: float = 0.01
    # "auto" resolves to "onehot" or "normal" from the environment action space.
    actor_dist: str = "auto"
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0

    # ------------------------------------------------------------------
    # World model losses
    # ------------------------------------------------------------------
    beta_pred: float = 1.0
    beta_dyn: float = 1.0
    beta_rep: float = 0.1
    # Free bits: KL terms are clipped below this many nats so that, once they are
    # minimized well, learning focuses on the prediction loss instead.
    free_nats: float = 1.0

    # ------------------------------------------------------------------
    # Actor critic
    # ------------------------------------------------------------------
    discount: float = 0.997
    td_lambda: float = 0.95
    imagine_horizon: int = 15
    actor_entropy: float = 3e-4
    critic_loss_scale: float = 1.0
    # Applying the critic loss to replayed trajectories as well improves value
    # prediction where rewards are hard to predict from imagination alone.
    critic_replay_loss_scale: float = 0.3
    critic_ema_decay: float = 0.98
    critic_ema_regularizer: float = 1.0
    # Percentile return normalization. Returns are divided by an EMA of the
    # 5th-to-95th percentile range, but only when that range exceeds the limit,
    # so small returns under sparse rewards are not amplified.
    return_norm_decay: float = 0.99
    return_norm_limit: float = 1.0
    return_norm_low: float = 5.0
    return_norm_high: float = 95.0

    # ------------------------------------------------------------------
    # Prediction heads
    # ------------------------------------------------------------------
    num_buckets: int = 255
    # Bin locations are symexp(linspace(-symlog_range, +symlog_range, buckets)).
    symlog_range: float = 20.0

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    # DreamerV3 uses one learning rate for all three optimizers.
    learning_rate: float = 4e-5
    agc_clip: float = 0.3
    agc_eps: float = 1e-3
    opt_eps: float = 1e-20
    opt_beta1: float = 0.9
    opt_beta2: float = 0.99
    weight_decay: float = 0.0

    batch_size: int = 16
    train_seq_len: int = 64
    # Time steps trained on per time step collected, counted without action
    # repeat. `update_steps` is derived from this unless auto_update_steps=False.
    # The paper picks it per benchmark to fit the step budget (Table 2: 32 for
    # Atari/DMLab/Minecraft, 64 ProcGen, 128 Atari100k, 512 for the control
    # suites, 1024 BSuite); 32 is the cheapest of those.
    replay_ratio: int = 32
    auto_update_steps: bool = True
    # Each minibatch is filled first from trajectories not trained on yet (the
    # online queue), then topped up uniformly. 1.0 lets the queue fill the batch.
    online_fraction: float = 1.0
    # Table 4. The agent lowers it to fit the available memory.
    buffer_size: int = 5_000_000
    # Store each record's model state in replay, start replayed sequences from
    # it, and write fresh states back after training (paper, Methods).
    replay_latents: bool = True

    use_amp: bool = False
    grad_clip_norm: float = 0.0  # Superseded by adaptive gradient clipping.
    seed_steps: int = 5000
    collect_steps: int = 1000
    action_repeat: int = 2
    action_noise: float = 0.0  # Exploration comes from the entropy regularizer.

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        self.resolve()

    def resolve(self) -> "DreamerV3Config":
        """Fill in width fields from ``model_size`` and derive ``update_steps``."""
        key = str(self.model_size).strip().lower()
        if key not in MODEL_SIZES:
            options = ", ".join(sorted(MODEL_SIZES, key=lambda name: int(name[:-1])))
            raise ValueError(
                f"Unknown model_size {self.model_size!r}. Options: {options}"
            )

        preset = MODEL_SIZES[key]
        for field_name, value in preset.items():
            if getattr(self, field_name) is None:
                setattr(self, field_name, value)

        if self.recurrent_units is not None and self.recurrent_units % self.gru_blocks:
            raise ValueError(
                f"recurrent_units={self.recurrent_units} must be divisible by "
                f"gru_blocks={self.gru_blocks}."
            )

        if self.auto_update_steps:
            # `collect_steps` counts environment frames; the replay ratio counts
            # policy steps, so divide out the action repeat. With a ratio of 32,
            # action repeat 4 and 16x64 batches this is one gradient step per
            # 128 frames, the paper's worked example.
            steps_per_batch = max(1, self.batch_size * self.train_seq_len)
            policy_steps = self.collect_steps / max(1, int(self.action_repeat))
            self.update_steps = max(
                1, round(policy_steps * self.replay_ratio / steps_per_batch)
            )
        return self

    # ------------------------------------------------------------------
    # Convenience accessors, so downstream code never sees `None`.
    # ------------------------------------------------------------------

    @property
    def resolved_hidden_size(self) -> int:
        return int(self.hidden_size or MODEL_SIZES[self.model_size]["hidden_size"])

    @property
    def resolved_recurrent_units(self) -> int:
        return int(
            self.recurrent_units or MODEL_SIZES[self.model_size]["recurrent_units"]
        )

    @property
    def resolved_cnn_depth(self) -> int:
        return int(self.cnn_depth or MODEL_SIZES[self.model_size]["cnn_depth"])

    @property
    def resolved_latent_classes(self) -> int:
        return int(
            self.latent_classes or MODEL_SIZES[self.model_size]["latent_classes"]
        )
