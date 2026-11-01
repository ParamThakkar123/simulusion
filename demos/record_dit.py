#!/usr/bin/env python3
"""Sample images from a trained DiT checkpoint, headlessly.

``DiT.fit`` already writes a ``generated_samples.png`` when it finishes, but
that is the only way to see anything out of the model — there is no sampling
entrypoint, so a saved checkpoint cannot be turned back into images without
writing this loop by hand. This script is that entrypoint.

Two artifacts, because they answer different questions:

- ``dit_samples.png`` — a grid of finished samples. "Is the model any good?"
- ``dit_denoising.mp4`` — the reverse diffusion trajectory. "What is it doing?"
  This is the more demo-legible one: pure noise resolving into structure over
  the sampled timesteps.

Usage:
    python demos/record_dit.py -c dit_demo/dit_model.pth
    python demos/record_dit.py -c dit_demo/dit_model.pth --samples 64 --ddim-steps 100
    python demos/record_dit.py --random-init --ddim-steps 25   # pipeline check only
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from simulusion.configs.dit_config import DiTConfig
from simulusion.models.diffusion.DDPM import DDPM
from simulusion.models.diffusion.DiT import DiT
from simulusion.utils.utils import StreamingVideoWriter


def infer_architecture(state_dict: dict[str, Any]) -> dict[str, int]:
    """Recover DiT hyperparameters from tensor shapes.

    ``DiT.fit`` saves a bare ``state_dict`` and writes the config to a
    separate ``config.yaml``. When that YAML is missing (or the checkpoint was
    moved away from it) the architecture is still fully determined by the
    weights, so read it off them rather than guessing from current defaults.
    """
    shape: dict[str, int] = {}

    proj = state_dict.get("patchify.proj.weight")
    if proj is not None:
        # Conv2d(in_channels, d_model, kernel=patch, stride=patch).
        shape["WIDTH"] = int(proj.shape[0])
        shape["CHANNELS"] = int(proj.shape[1])
        shape["PATCH"] = int(proj.shape[2])

    pos = state_dict.get("pos_embed")
    if pos is not None and "PATCH" in shape:
        # (1, num_patches, d_model) over a square grid.
        side = int(round(math.sqrt(int(pos.shape[1]))))
        shape["IMG_SIZE"] = side * shape["PATCH"]

    # Count the blocks themselves rather than any one weight inside them: this
    # DiT keeps separate W_q/W_k/W_v projections, so the fused ".attn.qkv"
    # tensor this used to look for never exists and depth silently came back 0,
    # leaving DiTConfig's default of 12. Anything but a 12-block checkpoint then
    # failed to load.
    blocks = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"transformer_blocks\.(\d+)\.", key)) is not None
    }
    if blocks:
        shape["DEPTH"] = max(blocks) + 1

    return shape


def set_eval(model: DiT) -> DiT:
    """Put a DiT in eval mode.

    ``model.eval()`` works normally now that the training loop lives in
    ``DiT.fit`` instead of shadowing ``nn.Module.train``.
    """
    model.eval()
    return model


def build_model(args: argparse.Namespace) -> tuple[DiT, DiTConfig]:
    """Return an eval-mode DiT plus the config it was actually built with."""
    if args.random_init:
        config = DiTConfig(IMG_SIZE=args.img_size, DEPTH=4, WIDTH=192, HEADS=6)
        print("--random-init: sampling from an UNTRAINED model (noise in, noise out).")
        return set_eval(DiT.from_config(config)), config

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )

    arch = infer_architecture(state_dict)
    config = DiTConfig(**arch)  # type: ignore[arg-type]
    # `heads` is the one field the weights do not pin down: qkv is (3*d, d)
    # regardless of how many heads that d is split across. Keep the default
    # unless it cannot divide the width.
    if config.WIDTH % config.HEADS != 0:
        config.HEADS = max(1, config.WIDTH // 64)
    print(
        f"Architecture from checkpoint: img={config.IMG_SIZE} patch={config.PATCH} "
        f"width={config.WIDTH} depth={config.DEPTH} heads={config.HEADS} "
        f"channels={config.CHANNELS}"
    )

    model = DiT.from_config(config)
    model.load_state_dict(state_dict)
    return set_eval(model), config


def to_uint8_grid(batch: torch.Tensor, nrow: int) -> np.ndarray:
    """Tile a (B, C, H, W) batch in [-1, 1] into a single uint8 HxWxC image."""
    imgs = ((batch.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).cpu().numpy()
    n, c, h, w = imgs.shape
    ncol = int(math.ceil(n / nrow))
    canvas = np.zeros((ncol * h, nrow * w, c), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, col = divmod(i, nrow)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = img.transpose(1, 2, 0)
    return canvas if c == 3 else np.repeat(canvas, 3, axis=2)


@torch.no_grad()
def sample_with_trajectory(
    ddpm: DDPM,
    model: DiT,
    n: int,
    img_size: int,
    channels: int,
    device: torch.device,
    stride: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the reverse chain, keeping every ``stride``-th intermediate state.

    ``DDPM.sample`` returns only the final image, so the trajectory has to be
    collected here. Snapshots are the *current* ``x_t``, which is what makes the
    video read as denoising rather than as a sequence of finished guesses.
    """
    x = torch.randn(n, channels, img_size, img_size, device=device)
    trajectory = [x.clone()]
    for i in reversed(range(ddpm.timesteps)):
        t = torch.full((n,), i, dtype=torch.long, device=device)
        x = ddpm.p_sample(model, x, t)
        if i % stride == 0:
            trajectory.append(x.clone())
    return x.clamp(-1.0, 1.0), trajectory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample images from a DiT checkpoint, headlessly"
    )
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip the checkpoint and sample from an untrained model. Verifies the "
        "sampling pipeline; the images are meaningless.",
    )
    parser.add_argument("--samples", "-n", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4, help="Images per grid row.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Diffusion schedule length. Defaults to the config value (1000). "
        "Must match training, so only set it if the checkpoint used a different "
        "schedule.",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=None,
        help="Keep only this many trajectory snapshots for the video. Does NOT "
        "shorten the reverse chain — DDPM.p_sample is a full-chain sampler.",
    )
    parser.add_argument("--img-size", type=int, default=32, help="--random-init only.")
    parser.add_argument("--out-dir", default="demos/out")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--scale", type=int, default=4, help="Nearest-neighbour zoom.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.checkpoint and not args.random_init:
        parser.error("pass --checkpoint/-c, or --random-init to check the pipeline")

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, config = build_model(args)
    model.to(device)
    print(f"Parameters: {model.parameter_count() / 1e6:.2f}M on {device}")

    timesteps = args.timesteps or config.TIMESTEPS
    ddpm = DDPM(
        timesteps=timesteps,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
    ).to(device)

    if device.type == "cpu":
        print(
            f"Sampling {args.samples} images through {timesteps} reverse steps on CPU. "
            "This is slow; use --samples 4 or a GPU."
        )

    stride = max(1, timesteps // (args.ddim_steps or 60))
    samples, trajectory = sample_with_trajectory(
        ddpm, model, args.samples, config.IMG_SIZE, config.CHANNELS, device, stride
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = to_uint8_grid(samples, args.nrow)
    grid_path = out_dir / "dit_samples.png"
    _write_png(grid, grid_path, args.scale)
    print(
        f"Wrote {grid_path}  ({args.samples} samples, {grid.shape[1]}x{grid.shape[0]})"
    )

    video_path = out_dir / "dit_denoising.mp4"
    writer = StreamingVideoWriter(str(video_path), fps=args.fps)
    for step in trajectory:
        frame = to_uint8_grid(step, args.nrow)
        writer.write_frame(_zoom(frame, args.scale))
    # Hold the finished grid so the clip does not end on a flash.
    for _ in range(args.fps):
        writer.write_frame(_zoom(grid, args.scale))
    writer.close()
    print(f"Wrote {video_path}  ({len(trajectory) + args.fps} frames)")
    return 0


def _zoom(image: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale. 32x32 grids are unreadable at native size."""
    if scale <= 1:
        return image
    return image.repeat(scale, axis=0).repeat(scale, axis=1)


def _write_png(image: np.ndarray, path: Path, scale: int) -> None:
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(_zoom(image, scale), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    raise SystemExit(main())
