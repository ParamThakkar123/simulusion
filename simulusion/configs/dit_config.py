from dataclasses import dataclass, replace
from typing import Any

from simulusion.configs.serialization import SerializableConfigMixin

_SNAKE_TO_UPPER = {
    "dataset": "DATASET",
    "batch": "BATCH",
    "epochs": "EPOCHS",
    "lr": "LR",
    "img_size": "IMG_SIZE",
    "channels": "CHANNELS",
    "patch_size": "PATCH",
    "width": "WIDTH",
    "depth": "DEPTH",
    "heads": "HEADS",
    "drop": "DROP",
    "beta_start": "BETA_START",
    "beta_end": "BETA_END",
    "timesteps": "TIMESTEPS",
    "ema": "EMA",
    "ema_decay": "EMA_DECAY",
    "workdir": "WORKDIR",
    "root_path": "ROOT_PATH",
    "early_stopping": "EARLY_STOPPING",
    "patience": "PATIENCE",
    "min_delta": "MIN_DELTA",
    "val_split": "VAL_SPLIT",
    "crop_size": "CROP_SIZE",
    "num_workers": "NUM_WORKERS",
    "num_classes": "NUM_CLASSES",
    "class_dropout_prob": "CLASS_DROPOUT_PROB",
    "learn_sigma": "LEARN_SIGMA",
    "weight_decay": "WEIGHT_DECAY",
}

# Table 1 of the paper: the four transformer configs, which jointly scale depth,
# width and head count following ViT. Combined with a patch size (2, 4 or 8) they
# name a model, e.g. DiT-XL/2.
DIT_PRESETS: dict[str, dict[str, int]] = {
    "DiT-S": {"DEPTH": 12, "WIDTH": 384, "HEADS": 6},
    "DiT-B": {"DEPTH": 12, "WIDTH": 768, "HEADS": 12},
    "DiT-L": {"DEPTH": 24, "WIDTH": 1024, "HEADS": 16},
    "DiT-XL": {"DEPTH": 28, "WIDTH": 1152, "HEADS": 16},
}


@dataclass
class DiTConfig(SerializableConfigMixin):
    """Default configuration values for Diffusion Transformer (DiT) training.

    The fields define dataset selection, model architecture, diffusion schedule,
    optimization hyperparameters, and output paths used by the built-in
    training entrypoints.

    Field names use UPPER_CASE for backward compatibility with the original DiT
    codebase. Snake-case aliases are accepted via ``__getattr__`` and
    ``get_dit_config()``.
    """

    DATASET: str = "CIFAR10"
    BATCH: int = 256  # paper 4
    EPOCHS: int = 3
    LR: float = 1e-4  # paper 4: constant 1e-4, no warmup
    WEIGHT_DECAY: float = 0.0  # paper 4: "no weight decay"
    IMG_SIZE: int = 32
    CHANNELS: int = 3
    PATCH: int = 4
    # DiT-S from Table 1. Use DIT_PRESETS / dit_preset_config for the others.
    WIDTH: int = 384
    DEPTH: int = 12
    HEADS: int = 6
    DROP: float = 0.0  # paper 4 found regularization unnecessary
    # Class-conditional generation and classifier-free guidance (paper 3.1).
    # 0 classes builds an unconditional model.
    NUM_CLASSES: int = 0
    CLASS_DROPOUT_PROB: float = 0.1
    # Predict the diagonal covariance alongside the noise (paper 3.1).
    LEARN_SIGMA: bool = True
    BETA_START: float = 1e-4
    BETA_END: float = 0.02
    TIMESTEPS: int = 1000
    EMA: bool = True
    EMA_DECAY: float = 0.9999  # paper 4
    WORKDIR: str = "./dit_demo"
    ROOT_PATH: str = "./data"
    # Epochs between intermediate checkpoints. 0 writes nothing until training
    # finishes, which is the original behaviour and loses everything if the run
    # is cut short.
    CHECKPOINT_EVERY: int = 0
    # Stop once held-out loss stops improving, instead of at a fixed EPOCHS.
    # Off by default so existing runs keep their exact length; EPOCHS then acts
    # as the ceiling rather than the target.
    EARLY_STOPPING: bool = False
    PATIENCE: int = 10
    MIN_DELTA: float = 1e-4
    # Held-out fraction for imagefolder/imagenet. CIFAR-10 has its own test
    # split, which is used instead.
    VAL_SPLIT: float = 0.05
    # Transform resolution for imagenet/imagefolder. None follows IMG_SIZE, so
    # the data and the model agree by default. Unused for CIFAR-10.
    CROP_SIZE: Any = None
    NUM_WORKERS: int = 4

    def __getattr__(self, name: str) -> Any:
        upper = _SNAKE_TO_UPPER.get(name)
        if upper is not None:
            return getattr(self, upper)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        upper = _SNAKE_TO_UPPER.get(name, name)
        super().__setattr__(upper, value)


def canonical_dit_key(name: str) -> str:
    """Map a snake_case alias to its UPPER_CASE field name.

    ``DiTConfig`` keeps the original DiT codebase's UPPER_CASE field names, so
    dot-list overrides composed by the training entrypoint have to be
    translated before they reach the strict config loader. Names that are
    already canonical (or unknown) are returned unchanged.
    """
    return _SNAKE_TO_UPPER.get(name, name)


def dit_preset_config(name: str, patch_size: int, **overrides: Any) -> DiTConfig:
    """Build a config for a named Table 1 model, e.g. ``dit_preset_config("DiT-XL", 2)``.

    Args:
        name: One of ``DiT-S``, ``DiT-B``, ``DiT-L``, ``DiT-XL`` (case-insensitive,
            and the ``DiT-`` prefix is optional).
        patch_size: Latent patch size; the paper explores 2, 4 and 8.
        **overrides: Further config fields, UPPER_CASE or snake_case.

    Returns:
        A :class:`DiTConfig` for that model. Defaults target latent diffusion of
        256x256 ImageNet: a 32x32x4 latent with 1000 classes.
    """
    key = str(name).strip()
    if not key.lower().startswith("dit-"):
        key = f"DiT-{key}"
    canonical = {k.lower(): k for k in DIT_PRESETS}.get(key.lower())
    if canonical is None:
        raise ValueError(
            f"Unknown DiT preset {name!r}. Valid names: {', '.join(DIT_PRESETS)}."
        )
    if patch_size not in (2, 4, 8):
        raise ValueError(
            f"patch_size must be 2, 4 or 8 (the paper's design space), got {patch_size}."
        )

    fields: dict[str, Any] = {
        **DIT_PRESETS[canonical],
        "PATCH": patch_size,
        # Latent-space defaults: 256x256 ImageNet through an f8 VAE.
        "IMG_SIZE": 32,
        "CHANNELS": 4,
        "NUM_CLASSES": 1000,
    }
    for key_, value in overrides.items():
        fields[_SNAKE_TO_UPPER.get(key_, key_)] = value
    return replace(DiTConfig(), **fields)


def list_dit_presets() -> list[str]:
    """Return the Table 1 model names."""
    return list(DIT_PRESETS)


def get_dit_config(**overrides: Any) -> DiTConfig:
    """
    Returns a DiTConfig instance with default values overridden by the provided keyword arguments.

    Both UPPER_CASE and snake_case override keys are accepted.

    Example usage:
        cfg = get_dit_config(BATCH=64, EPOCHS=10, LR=1e-3)
        cfg = get_dit_config(batch=64, epochs=10, lr=1e-3)
    """
    translated = {}
    for key, value in overrides.items():
        translated[_SNAKE_TO_UPPER.get(key, key)] = value
    return replace(DiTConfig(), **translated)
