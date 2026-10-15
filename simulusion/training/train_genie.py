import argparse
import os

import torch
from simulusion.utils.device import default_device_name
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional, Dict, Tuple, Literal
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from simulusion.configs.serialization import SerializableConfigMixin
from simulusion.models.genie import Genie
from simulusion.models.model_io import save_config_next_to_checkpoint
from simulusion.utils.memory_utils import enable_performance_defaults
from simulusion.utils.train_utils import EarlyStopping


@dataclass
class GenieConfig(SerializableConfigMixin):
    """Configuration for Genie training."""

    num_frames: int = 16
    image_size: int = 64
    in_channels: int = 3

    tokenizer_vocab_size: int = 1024
    tokenizer_embedding_dim: int = 32
    tokenizer_encoder_dim: int = 512
    tokenizer_decoder_dim: int = 1024
    tokenizer_encoder_depth: int = 12
    tokenizer_decoder_depth: int = 20

    action_vocab_size: int = 8
    action_embedding_dim: int = 32
    action_encoder_dim: int = 1024
    action_decoder_dim: int = 1024
    action_encoder_depth: int = 20
    action_pooling: Literal["mean", "windowed_attention"] = "mean"
    window_attention_heads: int = 1

    dynamics_dim: int = 512
    dynamics_depth: int = 8
    dynamics_num_heads: int = 8

    batch_size: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 5000
    max_steps: int = 125000

    mask_prob_min: float = 0.5
    mask_prob_max: float = 1.0

    sample_temperature: float = 2.0
    maskgit_steps: int = 25

    # Stop once held-out reconstruction loss stops improving, rather than at a
    # fixed max_steps, which then bounds the run instead of defining it. Off by
    # default so existing runs keep their exact length. Mirrors the same fields
    # on simulusion.configs.genie_config.GenieConfig/GenieSmallConfig, either of
    # which may be handed to the trainer instead of this one.
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4
    val_split: float = 0.1


class VideoDataset(Dataset):
    """Video clips for Genie training, returned as ``(C, T, H, W)`` float tensors.

    Each entry in ``video_paths`` may be:

    * a ``.npy`` / ``.npz`` array of shape ``(T, H, W, C)`` or ``(C, T, H, W)``
    * a ``.pt`` / ``.pth`` tensor of the same layouts
    * a video file (``.mp4``, ``.avi``, ``.mkv``, ``.webm``, ``.mov``) loaded
      with OpenCV when ``opencv-python`` is installed (the ``viz`` extra)

    Frames are uniformly sampled to ``num_frames`` and resized to ``image_size``.
    """

    def __init__(
        self, video_paths: list, num_frames: int = 16, image_size: int = 64
    ) -> None:
        if not video_paths:
            raise ValueError("video_paths must contain at least one clip")
        self.video_paths = [Path(p) for p in video_paths]
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.video_paths[idx]
        frames = self._load_thwc(path)
        frames = self._sample_frames(frames)
        frames = self._resize_frames(frames)
        # (T, H, W, C) -> (C, T, H, W) as GenieTrainer.train_step expects.
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2)
        return tensor.float()

    def _load_thwc(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path)
            return self._to_thwc(array)
        if suffix == ".npz":
            payload = np.load(path)
            key = "arr_0" if "arr_0" in payload.files else payload.files[0]
            return self._to_thwc(payload[key])
        if suffix in {".pt", ".pth"}:
            array = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(array, torch.Tensor):
                array = array.detach().cpu().numpy()
            return self._to_thwc(np.asarray(array))
        return self._load_video_file(path)

    def _to_thwc(self, array: np.ndarray) -> np.ndarray:
        if array.ndim != 4:
            raise ValueError(f"Expected a 4-D clip, got shape {array.shape}")
        array = np.asarray(array)
        # Channel-first (C, T, H, W) vs channel-last (T, H, W, C).
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 3, 0))
        elif array.shape[-1] not in (1, 3):
            raise ValueError(
                f"Could not infer channel axis for clip of shape {array.shape}"
            )
        if array.dtype == np.uint8:
            return array.astype(np.float32) / 255.0
        array = array.astype(np.float32)
        if array.max() > 1.0:
            array = array / 255.0
        return array

    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        total = int(frames.shape[0])
        if total == 0:
            raise ValueError("clip contains no frames")
        if total == self.num_frames:
            return frames
        indices = np.linspace(0, total - 1, self.num_frames).astype(int)
        return frames[indices]

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        _, h, w, _c = frames.shape
        if h == self.image_size and w == self.image_size:
            return frames
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Resizing Genie clips requires Pillow, which Simulusion already "
                "pulls in via torchvision."
            ) from exc
        resized = []
        for frame in frames:
            image = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
            image = image.resize((self.image_size, self.image_size))
            resized.append(np.asarray(image, dtype=np.float32) / 255.0)
        return np.stack(resized, axis=0)

    def _load_video_file(self, path: Path) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                f"Loading {path.name} needs OpenCV. Install it with "
                "`pip install simulusion[viz]`, or pass .npy/.pt clips instead."
            ) from exc
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Could not open video: {path}")
        frames_list: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames_list:
            raise ValueError(f"No frames decoded from {path}")
        return np.stack(frames_list, axis=0).astype(np.float32) / 255.0


class GenieTrainer:
    """Trainer for Genie model."""

    def __init__(
        self,
        model: nn.Module,
        config: GenieConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config

        if device is None:
            self.device = torch.device(default_device_name())
        else:
            self.device = device

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = self._create_scheduler()

        if getattr(config, "perf_defaults", True):
            enable_performance_defaults(tf32=bool(getattr(config, "tf32", True)))

        self.use_amp = bool(getattr(config, "use_amp", False))
        self._amp_device = getattr(self.device, "type", str(self.device))
        # bf16 carries fp32's dynamic range, so it needs no loss scaling; fp16
        # does. Only build a scaler for the case that actually requires one.
        self._amp_dtype = (
            torch.bfloat16
            if self._amp_device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            self._amp_device,
            enabled=self.use_amp and self._amp_dtype is torch.float16,
        )

        self.global_step = 0

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler with warmup and cosine decay."""
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                # A plain float: an np.float64 lands in the scheduler's
                # `_last_lr`, and checkpoints holding it then refuse to load
                # under `torch.load(weights_only=True)`.
                return float(0.5 * (1.0 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor | float | None]:
        """Single training step.

        Args:
            batch: (B, C, T, H, W) video batch

        Returns:
            Dictionary of losses
        """
        self.model.train()
        batch = batch.to(self.device)

        B, C, T, H, W = batch.shape
        mask_prob = (
            torch.rand(1).item()
            * (self.config.mask_prob_max - self.config.mask_prob_min)
            + self.config.mask_prob_min
        )

        with torch.amp.autocast(
            device_type=self._amp_device,
            dtype=self._amp_dtype,
            enabled=self.use_amp,
        ):
            outputs = self.model(batch, mask_prob=mask_prob)

        recon_loss = outputs.get("recon_loss", 0.0)
        vq_loss = outputs.get("vq_loss", 0.0)

        dynamics_loss = outputs.get(
            "dynamics_loss", torch.tensor(0.0, device=self.device)
        )

        z_q_for_dynamics = outputs.get("z_q_for_dynamics", None)
        z_q_for_dynamics_mean = (
            z_q_for_dynamics.mean().item()
            if isinstance(z_q_for_dynamics, torch.Tensor)
            else None
        )

        total_loss = outputs["total_loss"]

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        # Gradients must be unscaled before clipping, or the norm is measured on
        # scaled values and the clip threshold means nothing.
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.global_step += 1

        # Normalize all returned metrics to torch.Tensor for consistent typing
        def as_tensor(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            try:
                return torch.tensor(float(x))
            except Exception:
                return torch.tensor(float("nan"))

        return {
            "total_loss": as_tensor(total_loss),
            "recon_loss": as_tensor(recon_loss),
            "vq_loss": as_tensor(vq_loss),
            "dynamics_loss": as_tensor(dynamics_loss),
            "learning_rate": torch.tensor(float(self.scheduler.get_last_lr()[0])),
            "z_q_for_dynamics_mean": as_tensor(z_q_for_dynamics_mean),
        }

    def validate(self, val_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            val_batch: (B, C, T, H, W) validation video batch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(val_batch, mask_prob=0.0)
            recon_loss = outputs["tokenizer_loss"].get("recon_loss", 0.0)
            return {
                "val_recon_loss": recon_loss.detach().cpu()
                if isinstance(recon_loss, torch.Tensor)
                else torch.tensor(float(recon_loss)),
            }

    def validate_epoch(self, val_dataloader: DataLoader) -> float:
        """Mean held-out reconstruction loss over the whole validation loader.

        ``validate`` scores a single batch, which is far too noisy to drive a
        plateau test -- the batch-to-batch spread swamps the epoch-to-epoch
        trend, so early stopping on it would fire on noise. Averaging over the
        loader gives a comparable number per validation.
        """
        total, count = 0.0, 0
        for val_batch in val_dataloader:
            val_batch = val_batch.to(self.device)
            metrics = self.validate(val_batch)
            batch_size = val_batch.size(0)
            total += float(metrics["val_recon_loss"]) * batch_size
            count += batch_size
        return total / max(count, 1)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_steps: Optional[int] = None,
        log_interval: int = 100,
        val_interval: int = 1000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 0,
    ) -> None:
        """Full training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_steps: Number of training steps (uses config.max_steps if None)
            log_interval: Logging frequency
            val_interval: Validation frequency
            checkpoint_dir: Where periodic checkpoints go. Without it nothing is
                written until the caller saves, so a run killed by a timeout, an
                OOM or Ctrl+C leaves nothing behind.
            checkpoint_interval: Steps between checkpoints. 0 disables them,
                which is the previous behaviour.
        """
        if num_steps is None:
            num_steps = self.config.max_steps

        periodic = checkpoint_dir is not None and checkpoint_interval > 0
        if periodic:
            assert checkpoint_dir is not None  # narrowed for the type checker
            os.makedirs(checkpoint_dir, exist_ok=True)

        stopper = None
        best_val = float("inf")
        if getattr(self.config, "early_stopping", False):
            if val_dataloader is None:
                raise ValueError(
                    "early_stopping needs a val_dataloader; build one with "
                    "create_tinyworlds_dataloader(val_split=..., split='val')"
                )
            stopper = EarlyStopping(
                mode="min",
                patience=self.config.patience,
                threshold=self.config.min_delta,
            )

        train_iter = iter(train_dataloader)

        while self.global_step < num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            batch = batch.to(self.device)

            losses = self.train_step(batch)

            if self.global_step % log_interval == 0:
                print(
                    f"Step {self.global_step}/{num_steps} | "
                    f"Loss: {losses['total_loss']:.4f} | "
                    f"Recon: {losses['recon_loss']:.4f} | "
                    f"VQ: {losses['vq_loss']:.4f} | "
                    f"Dynamics: {losses['dynamics_loss']:.4f} | "
                    f"LR: {losses['learning_rate']:.6f}"
                )

            if periodic and self.global_step % checkpoint_interval == 0:
                self.save_checkpoint(
                    os.path.join(
                        str(checkpoint_dir), f"genie_step_{self.global_step}.pt"
                    )
                )

            if val_dataloader is not None and self.global_step % val_interval == 0:
                val_loss = self.validate_epoch(val_dataloader)
                message = f"Validation: val_recon_loss={val_loss:.4f}"
                if stopper is not None:
                    stopper.step(val_loss)
                    if val_loss < best_val:
                        best_val = val_loss
                        message += " (best)"
                    print(message)
                    if stopper.stop:
                        print(
                            f"Early stopping at step {self.global_step}: held-out "
                            f"reconstruction loss has not improved by "
                            f"{self.config.min_delta} for {self.config.patience} "
                            f"validations."
                        )
                        break
                else:
                    print(message)

        if periodic:
            self.save_checkpoint(
                os.path.join(str(checkpoint_dir), f"genie_step_{self.global_step}.pt")
            )

        print("Training complete!")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        save_config_next_to_checkpoint(self.config, path)
        torch.save(
            {
                "config": self.config.to_dict(),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]


def create_genie_trainer(
    config: Optional[GenieConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[GenieTrainer, nn.Module]:
    """Factory function to create Genie trainer and model."""
    if config is None:
        config = GenieConfig()

    # tokenizer_encoder_dim, tokenizer_decoder_dim and action_encoder_dim used
    # to be left off this call, so Genie fell back to its own much wider
    # defaults (512/1024/1024) and those three config fields did nothing at all
    # -- a config that said 256 built a 512-wide tokenizer, and the checkpoint
    # then recorded the width it did not have.
    model = Genie(
        num_frames=config.num_frames,
        image_size=config.image_size,
        in_channels=config.in_channels,
        tokenizer_vocab_size=config.tokenizer_vocab_size,
        tokenizer_embedding_dim=config.tokenizer_embedding_dim,
        tokenizer_encoder_dim=config.tokenizer_encoder_dim,
        tokenizer_decoder_dim=config.tokenizer_decoder_dim,
        action_vocab_size=config.action_vocab_size,
        action_embedding_dim=config.action_embedding_dim,
        action_encoder_dim=config.action_encoder_dim,
        action_decoder_dim=config.action_decoder_dim,
        dynamics_dim=config.dynamics_dim,
        dynamics_depth=config.dynamics_depth,
        dynamics_num_heads=config.dynamics_num_heads,
        encoder_depth=config.tokenizer_encoder_depth,
        decoder_depth=config.tokenizer_decoder_depth,
        latent_action_depth=config.action_encoder_depth,
        action_pooling=config.action_pooling,
        window_attention_heads=config.window_attention_heads,
        # Only simulusion.configs' Genie configs carry head counts; without passing
        # them a `tokenizer_num_heads` override was recorded in config.yaml but
        # the model was always built with 16 heads.
        tokenizer_num_heads=getattr(config, "tokenizer_num_heads", 16),
        action_num_heads=getattr(config, "action_num_heads", 16),
    )

    trainer = GenieTrainer(model, config, device)

    return trainer, model


def main(argv: Optional[list[str]] = None) -> None:
    """Console entrypoint for Genie trainer setup.

    ``VideoDataset`` loads ``.npy`` / ``.pt`` clips, or video files when OpenCV
    is installed. For the TinyWorlds HDF5 path, use
    ``scripts/train_genie_tinyworlds.py``.
    """
    parser = argparse.ArgumentParser(description="Prepare Genie training")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the default Genie max training steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct the trainer and print its resolved configuration without training.",
    )
    args = parser.parse_args(argv)

    config = GenieConfig()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    device = torch.device(args.device) if args.device else None
    trainer, _ = create_genie_trainer(config=config, device=device)

    if args.dry_run:
        print(
            "Created GenieTrainer "
            f"on {trainer.device} with max_steps={trainer.config.max_steps}"
        )
        return

    parser.error(
        "Genie requires a concrete video dataset; use --dry-run to validate "
        "trainer construction or scripts/train_genie_tinyworlds.py for an "
        "end-to-end example."
    )


if __name__ == "__main__":
    main()
