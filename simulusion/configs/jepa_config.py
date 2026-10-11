import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

from simulusion.configs.serialization import SerializableConfigMixin, make_yaml_safe


@dataclass
class JEPAConfig(SerializableConfigMixin):
    """
    Minimal configuration container for JEPA training.
    Converts to the nested dict expected by `train_jepa.main`.

    Defaults reproduce I-JEPA as specified in the paper (Assran et al., CVPR
    2023, Sec. 3 and Appendix A): no hand-crafted view augmentations, 4 target
    blocks of scale (0.15, 0.2), a single context block of scale (0.85, 1.0),
    and the paper's AdamW/EMA schedules at the reference batch size of 2048.
    See ``configs/experiments/jepa_small_gpu.yaml`` for a single-GPU preset.
    """

    # meta
    use_bfloat16: bool = False
    model_name: str = "vit_base"
    load_checkpoint: bool = False
    read_checkpoint: str | None = None
    copy_data: bool = False
    # ``None`` selects the paper's predictor depth for ``model_name``
    # (6 for ViT-B, 12 for ViT-L/H, 16 for ViT-G); set an int to override.
    pred_depth: int | None = None
    pred_emb_dim: int = 384
    # "l2" is the paper's loss (Sec. 3), mean-reduced; "l2_sum" is its literal
    # per-block sum and "smooth_l1" matches the reference implementation. See
    # ``train_jepa.build_loss_fn``.
    loss_type: str = "l2"

    # data
    dataset: str = "imagenet"
    val_split: float | None = None
    # The paper's central claim is that no hand-crafted view augmentations are
    # needed; only the random resized crop of the reference code is kept.
    use_gaussian_blur: bool = False
    use_horizontal_flip: bool = False
    use_color_distortion: bool = False
    color_jitter_strength: float = 0.0
    batch_size: int = 2048
    pin_mem: bool = True
    num_workers: int = 8
    root_path: str = os.environ.get("IMAGENET_ROOT", "/data/imagenet")
    image_folder: str = "train"
    crop_size: int = 224
    crop_scale: Tuple[float, float] = (0.3, 1.0)
    download: bool = False

    # mask
    allow_overlap: bool = False
    patch_size: int = 16
    num_enc_masks: int = 1
    min_keep: int = 10
    enc_mask_scale: Tuple[float, float] = (0.85, 1.0)
    num_pred_masks: int = 4
    pred_mask_scale: Tuple[float, float] = (0.15, 0.2)
    aspect_ratio: Tuple[float, float] = (0.75, 1.5)

    # optimization
    ema: Tuple[float, float] = (0.996, 1.0)
    ipe_scale: float = 1.0
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    epochs: int = 300
    warmup: int = 15
    start_lr: float = 1e-4
    lr: float = 1e-3
    final_lr: float = 1e-6
    # ``lr``/``start_lr`` are the paper's values at a batch size of 2048; they
    # are scaled linearly with the effective batch size. Set to ``None`` to use
    # the configured learning rates verbatim.
    lr_reference_batch_size: int | None = 2048
    # Stop once held-out loss stops improving, rather than running all `epochs`,
    # which then acts as a ceiling. Off by default so existing runs keep their
    # exact length. Needs `val_split` (imagefolder) or CIFAR-10's test split.
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4

    # logging
    folder: str = "results/jepa"
    write_tag: str = "jepa_run"
    enable_wandb: bool = False
    wandb_project: str = "simulusion"
    wandb_entity: str = ""
    enable_sweep: bool = False
    sweep_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return make_yaml_safe(
            {
                "meta": {
                    "use_bfloat16": self.use_bfloat16,
                    "model_name": self.model_name,
                    "load_checkpoint": self.load_checkpoint,
                    "read_checkpoint": self.read_checkpoint,
                    "copy_data": self.copy_data,
                    "pred_depth": self.pred_depth,
                    "pred_emb_dim": self.pred_emb_dim,
                    "loss_type": self.loss_type,
                },
                "data": {
                    "dataset": self.dataset,
                    "val_split": self.val_split,
                    "use_gaussian_blur": self.use_gaussian_blur,
                    "use_horizontal_flip": self.use_horizontal_flip,
                    "use_color_distortion": self.use_color_distortion,
                    "color_jitter_strength": self.color_jitter_strength,
                    "batch_size": self.batch_size,
                    "pin_mem": self.pin_mem,
                    "num_workers": self.num_workers,
                    "root_path": self.root_path,
                    "image_folder": self.image_folder,
                    "crop_size": self.crop_size,
                    "crop_scale": self.crop_scale,
                    "download": self.download,
                },
                "mask": {
                    "allow_overlap": self.allow_overlap,
                    "patch_size": self.patch_size,
                    "num_enc_masks": self.num_enc_masks,
                    "min_keep": self.min_keep,
                    "enc_mask_scale": self.enc_mask_scale,
                    "num_pred_masks": self.num_pred_masks,
                    "pred_mask_scale": self.pred_mask_scale,
                    "aspect_ratio": self.aspect_ratio,
                },
                "optimization": {
                    "ema": self.ema,
                    "ipe_scale": self.ipe_scale,
                    "weight_decay": self.weight_decay,
                    "final_weight_decay": self.final_weight_decay,
                    "epochs": self.epochs,
                    "warmup": self.warmup,
                    "start_lr": self.start_lr,
                    "lr": self.lr,
                    "final_lr": self.final_lr,
                    "lr_reference_batch_size": self.lr_reference_batch_size,
                    "early_stopping": self.early_stopping,
                    "patience": self.patience,
                    "min_delta": self.min_delta,
                },
                "logging": {
                    "folder": self.folder,
                    "write_tag": self.write_tag,
                    "enable_wandb": self.enable_wandb,
                    "wandb_project": self.wandb_project,
                    "wandb_entity": self.wandb_entity,
                    "enable_sweep": self.enable_sweep,
                    "sweep_config": self.sweep_config,
                },
            }
        )

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "JEPAConfig":
        """Load flat field values or the nested trainer dictionary."""

        config = cls()
        if any(
            key in values for key in ("meta", "data", "mask", "optimization", "logging")
        ):
            values = {
                key: value
                for section in values.values()
                if isinstance(section, dict)
                for key, value in section.items()
            }
        for key, value in values.items():
            if not hasattr(config, key):
                raise ValueError(f"Invalid JEPAConfig field: {key}")
            current = getattr(config, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(config, key, value)
        return config

    def to_train_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the nested dictionary expected by train_jepa."""

        return self.to_dict()

    def to_nested_dict(self) -> Dict[str, Dict[str, Any]]:
        """Backward-compatible alias for the nested JEPA trainer dictionary."""

        return self.to_train_dict()
