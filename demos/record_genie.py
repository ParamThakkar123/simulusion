#!/usr/bin/env python3
"""Generate frames from a trained Genie checkpoint, headlessly.

Genie generates video frames from a single prompt frame using a learned
dynamics model and latent action model. This script:

1. Loads a trained Genie checkpoint (TinyWorlds or paper-scale).
2. Takes a prompt frame: --prompt-image, else a frame from the TinyWorlds
   dataset if it is already on disk, else random noise.
3. Generates ``num_frames`` future frames using the dynamics model.
4. Writes:
   - ``genie_frames.mp4`` — the generated video.
   - ``genie_grid.png`` — a grid of prompt + generated frames.

Usage:
    python demos/record_genie.py -c checkpoints/genie_sonic_final.pt
    python demos/record_genie.py -c ckpt.pt --num-frames 32 --prompt-index 1
    python demos/record_genie.py --random-init --num-frames 8   # pipeline check
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from simulusion.models.genie import Genie, create_genie_small
from simulusion.utils.utils import StreamingVideoWriter


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) float tensor to uint8 HxWxC numpy array."""
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round()
    else:
        arr = np.clip(arr, 0, 255).round()
    return arr.astype(np.uint8)


def to_uint8_grid(frames: torch.Tensor, nrow: int) -> np.ndarray:
    """Tile (B, C, H, W) frames in [-1, 1] / [0, 1] into a single uint8 image."""
    imgs = frames.detach().cpu()
    if imgs.min() >= -0.01:
        imgs = (imgs.clamp(0, 1) * 255).round().to(torch.uint8)
    else:
        imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    imgs = imgs.numpy()
    n, c, h, w = imgs.shape
    ncol = int(math.ceil(n / nrow))
    canvas = np.zeros((ncol * h, nrow * w, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, nrow)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = imgs[i].transpose(
            1, 2, 0
        )
    return canvas


# GenieConfig field -> Genie constructor argument, where the two disagree.
_CONFIG_TO_KWARG = {
    "tokenizer_encoder_depth": "encoder_depth",
    "tokenizer_decoder_depth": "decoder_depth",
    "action_encoder_depth": "latent_action_depth",
}
# Constructor arguments that come straight off the saved config.
_DIRECT_KWARGS = (
    "num_frames",
    "image_size",
    "in_channels",
    "tokenizer_vocab_size",
    "tokenizer_embedding_dim",
    "tokenizer_encoder_dim",
    "tokenizer_decoder_dim",
    "action_vocab_size",
    "action_embedding_dim",
    "action_encoder_dim",
    "action_decoder_dim",
    "dynamics_dim",
    "dynamics_depth",
    "dynamics_num_heads",
    "action_pooling",
    "window_attention_heads",
)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Create a Genie model, rebuilt to match the checkpoint's architecture.

    The trainer stores the ``GenieConfig`` it ran with alongside the weights, so
    the model is reconstructed from that rather than from ``create_genie_small``
    plus the CLI defaults. Loading is strict: a mismatched checkpoint used to be
    absorbed by ``strict=False`` and the demo would generate from random weights
    while reporting success, which is worse than an error.
    """
    if args.random_init:
        print("--random-init: generating from an UNTRAINED model (noise).")
        return create_genie_small(
            num_frames=args.num_frames, image_size=args.image_size
        ).eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config")

    if isinstance(config, dict):
        kwargs = {key: config[key] for key in _DIRECT_KWARGS if key in config}
        for field, kwarg in _CONFIG_TO_KWARG.items():
            if field in config:
                kwargs[kwarg] = config[field]
        # num_frames sizes the positional embeddings, so it has to stay at the
        # checkpoint's value; the clip length is a generate() argument instead.
        args.image_size = kwargs.get("image_size", args.image_size)
        trained_frames = kwargs.get("num_frames", args.num_frames)
        if args.num_frames > trained_frames:
            print(
                f"Clip length {args.num_frames} exceeds the {trained_frames} "
                "frames this checkpoint was trained for; generating "
                f"{trained_frames}."
            )
            args.num_frames = trained_frames
        print(
            "Architecture from checkpoint: "
            f"image_size={kwargs.get('image_size')} "
            f"dynamics_dim={kwargs.get('dynamics_dim')} "
            f"dynamics_depth={kwargs.get('dynamics_depth')}"
        )
        model = Genie(**kwargs)
    else:
        print(
            "Checkpoint carries no config; falling back to create_genie_small "
            "defaults. Pass --image-size if this is not what it was trained at."
        )
        model = create_genie_small(
            num_frames=args.num_frames, image_size=args.image_size
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"Checkpoint does not match the model: {len(missing)} missing and "
            f"{len(unexpected)} unexpected tensors "
            f"(first missing: {missing[:3]}). Generating from this would show "
            "untrained weights, so nothing was written."
        )
    print(f"Loaded checkpoint (step {checkpoint.get('global_step', '?')})")
    return model.eval()


def make_prompt(
    args: argparse.Namespace,
    image_size: int,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the (1, C, H, W) frame Genie continues from, in [0, 1].

    A real frame from the training distribution, when one can be reached: the
    generated clip is a continuation of the prompt, so prompting on noise makes
    even a well-trained model produce noise and the demo shows nothing. Random
    noise stays as the last resort, and says so.
    """
    if args.prompt_image:
        from PIL import Image
        from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

        if not Path(args.prompt_image).exists():
            raise SystemExit(f"--prompt-image {args.prompt_image} does not exist")
        transform = Compose([Resize(image_size), CenterCrop(image_size), ToTensor()])
        image = Image.open(args.prompt_image).convert("RGB")
        return transform(image).unsqueeze(0).to(device)

    if not args.no_dataset_prompt:
        try:
            from simulusion.datasets import create_tinyworlds_dataloader

            dataset, _ = create_tinyworlds_dataloader(
                dataset_name=args.prompt_dataset,
                num_frames=max(2, args.prompt_index + 1),
                image_size=image_size,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                data_file=args.prompt_file,
                # Never pull the dataset down just to draw a picture.
                download=False,
            )
            # (C, T, H, W) in [0, 1].
            clip = dataset[0]
            frame = clip[:, min(args.prompt_index, clip.shape[1] - 1)]
            print(f"Prompt: {args.prompt_dataset} clip 0 frame {args.prompt_index}")
            return frame.unsqueeze(0).to(device)
        except Exception as exc:  # dataset absent, or not downloaded
            print(f"Could not read a dataset prompt ({exc}); using noise instead.")

    print(
        "Prompting on random noise. The clip below shows that generation runs, "
        "not what the model has learnt -- pass --prompt-image or make the "
        "TinyWorlds dataset available."
    )
    return torch.rand(1, channels, image_size, image_size, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate video frames from a Genie checkpoint"
    )
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip checkpoint; generate from an untrained model.",
    )
    parser.add_argument(
        "--num-frames", "-n", type=int, default=16, help="Frames to generate."
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--nrow", type=int, default=8, help="Grid columns.")
    parser.add_argument("--out-dir", default="demos/out")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prompt-image", default=None, help="Image file to continue from."
    )
    parser.add_argument(
        "--prompt-dataset",
        default="SONIC",
        help="TinyWorlds dataset to draw the prompt frame from.",
    )
    parser.add_argument(
        "--prompt-file", default=None, help="Local TinyWorlds HDF5 file."
    )
    parser.add_argument(
        "--prompt-index", type=int, default=0, help="Frame within the prompt clip."
    )
    parser.add_argument(
        "--no-dataset-prompt",
        action="store_true",
        help="Skip the dataset lookup and prompt on random noise.",
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.random_init:
        parser.error("pass --checkpoint/-c, or --random-init to check the pipeline")

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model(args)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")

    prompt = make_prompt(args, args.image_size, args.channels, device)
    print(f"Prompt shape: {prompt.shape}")

    with torch.no_grad():
        generated = model.generate(
            prompt, num_frames=args.num_frames, actions=None, use_maskgit=False
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # generate() returns (B, C, T, H, W), so dropping the batch leaves
    # (C, T, H, W). to_uint8_grid tiles over the leading axis and expects
    # (N, C, H, W), so time has to move in front of channels first -- otherwise
    # it reads T as the channel count and builds a T-channel canvas that
    # cv2.cvtColor rejects.
    # Prompt first, then the continuation, which is what makes the grid
    # readable: the question is whether the generated frames still look like
    # the frame they started from.
    tiles = torch.cat(
        [prompt.cpu(), generated.squeeze(0).permute(1, 0, 2, 3).cpu()], dim=0
    )
    grid = to_uint8_grid(tiles, args.nrow)
    grid_path = out_dir / "genie_grid.png"
    _write_png(grid, grid_path)
    print(f"Wrote {grid_path}  (tile 1 is the prompt)")

    video_path = out_dir / "genie_frames.mp4"
    writer = StreamingVideoWriter(str(video_path), fps=args.fps)
    frames = generated.squeeze(0)
    for t in range(frames.shape[1]):
        frame = frames[:, t, :, :]
        img = tensor_to_uint8_img(frame)
        writer.write_frame(np.tile(img, (2, 2, 1)) if args.image_size < 128 else img)
    for _ in range(args.fps):
        writer.write_frame(
            np.tile(tensor_to_uint8_img(frames[:, -1, :, :]), (2, 2, 1))
            if args.image_size < 128
            else tensor_to_uint8_img(frames[:, -1, :, :])
        )
    writer.close()
    # args.fps frames of the last image are appended so the clip does not snap
    # back instantly on loop; count them.
    print(f"Wrote {video_path}  ({args.num_frames} generated + {args.fps} hold frames)")

    return 0


def _write_png(image: np.ndarray, path: Path) -> None:
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    raise SystemExit(main())
