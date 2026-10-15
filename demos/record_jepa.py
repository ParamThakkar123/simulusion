#!/usr/bin/env python3
"""Visualise I-JEPA mask prediction from a trained checkpoint, headlessly.

I-JEPA (Image-based Joint Embedding Predictive Architecture) learns to predict
the representations of masked image regions from visible context regions, using
a ViT encoder + predictor.

This script:
1. Loads a JEPA encoder + predictor checkpoint.
2. Applies context and target masks to a sample image.
3. Runs the encoder on the visible patches and the predictor on the masked ones.
4. Saves:
   - ``jepa_masked_input.png`` — image with masked regions blanked out.
   - ``jepa_prediction_similarity.png`` — similarity heatmap between the
     predicted and actual representations of the target patches.

Usage:
    python demos/record_jepa.py -c results/jepa/jepa_run-latest.pth.tar
    python demos/record_jepa.py -c ckpt.pth.tar --image-size 224 --device cpu
    python demos/record_jepa.py --random-init                       # pipeline check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from simulusion.helpers.jepa_helper import init_model
from simulusion.masks.multiblock import MaskCollator as MBMaskCollator
from simulusion.utils.utils import apply_masks
from simulusion.utils.jepa_utils import repeat_interleave_batch


# Backbone width uniquely identifies the ViT size (simulusion.models.vit).
_WIDTH_TO_MODEL = {
    192: "vit_tiny",
    384: "vit_small",
    768: "vit_base",
    1024: "vit_large",
    1280: "vit_huge",
    1408: "vit_giant",
}


def infer_architecture(checkpoint: dict) -> dict:
    """Recover the encoder/predictor geometry from a JEPA checkpoint.

    ``init_model`` builds whatever the CLI defaults say (vit_base/16 at 224),
    so a checkpoint trained at any other size failed to load with a wall of
    shape mismatches. The weights pin every one of these values, so read them
    off instead of trusting the defaults.
    """
    encoder = checkpoint["encoder"]
    predictor = checkpoint.get("predictor", {})
    arch: dict = {}

    proj = encoder.get("patch_embed.proj.weight")
    if proj is not None:
        # Conv2d(in_chans, embed_dim, kernel=patch, stride=patch).
        width = int(proj.shape[0])
        arch["patch_size"] = int(proj.shape[2])
        arch["model_name"] = _WIDTH_TO_MODEL.get(width)
        arch["embed_dim"] = width

    pos = encoder.get("pos_embed")
    if pos is not None and "patch_size" in arch:
        # (1, num_patches, embed_dim) over a square grid.
        side = round(float(pos.shape[1]) ** 0.5)
        arch["crop_size"] = side * arch["patch_size"]

    pred_embed = predictor.get("predictor_embed.weight")
    if pred_embed is not None:
        arch["pred_emb_dim"] = int(pred_embed.shape[0])

    depth = {
        int(key.split(".")[1])
        for key in predictor
        if key.startswith("predictor_blocks.") and key.split(".")[1].isdigit()
    }
    if depth:
        arch["pred_depth"] = max(depth) + 1

    return arch


def load_image(
    path: str | None, crop_size: int, dataset_root: str | None = None
) -> torch.Tensor:
    """Pick the demo image and return it as (1, 3, H, W).

    In order: an explicit ``--image``; a sample from a CIFAR-10 copy already on
    disk; failing both, random noise. The last case is a real fallback, not a
    default worth having -- a cosine-similarity heatmap over noise looks like a
    working demo and tells you nothing, so it announces itself.
    """
    from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

    transform = Compose([Resize(crop_size), CenterCrop(crop_size), ToTensor()])

    if path:
        # A mistyped path used to fall through to noise and still "succeed".
        if not Path(path).exists():
            raise SystemExit(f"--image {path} does not exist")
        from PIL import Image

        return transform(Image.open(path).convert("RGB")).unsqueeze(0)

    if dataset_root and Path(dataset_root, "cifar-10-batches-py").exists():
        from torchvision.datasets import CIFAR10

        dataset = CIFAR10(root=dataset_root, train=False, download=False)
        image, _ = dataset[0]
        print(f"Using a CIFAR-10 test image from {dataset_root}")
        return transform(image.convert("RGB")).unsqueeze(0)

    print(
        "No --image and no CIFAR-10 under --dataset-root: running on random "
        "noise. The heatmap below is a pipeline check, not a result."
    )
    return torch.rand(1, 3, crop_size, crop_size)


def patch_grid(img_size: int, patch_size: int) -> tuple[int, int]:
    """Return (num_patches_h, num_patches_w)."""
    return img_size // patch_size, img_size // patch_size


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualise I-JEPA mask prediction from a trained checkpoint"
    )
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip checkpoint; run an untrained model pipeline check.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an input image. Without it, a CIFAR-10 test image from "
        "--dataset-root, or random noise if that is not on disk either.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Where to look for CIFAR-10 when --image is omitted. Never "
        "downloads; the training runs populate this.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--model",
        default="vit_base",
        choices=["vit_tiny", "vit_small", "vit_base", "vit_large"],
    )
    # None -> the paper's predictor depth for the chosen backbone, which is
    # what a checkpoint trained with the defaults will contain.
    parser.add_argument("--pred-depth", type=int, default=None)
    parser.add_argument("--pred-emb-dim", type=int, default=384)
    parser.add_argument("--out-dir", default="demos/out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.checkpoint and not args.random_init:
        parser.error("pass --checkpoint/-c, or --random-init to check the pipeline")

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_name = args.model
    patch_size = args.patch_size
    image_size = args.image_size
    pred_depth = args.pred_depth
    pred_emb_dim = args.pred_emb_dim
    checkpoint = None

    if not args.random_init:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        arch = infer_architecture(checkpoint)
        if arch.get("model_name") is None:
            raise SystemExit(
                f"Unrecognised encoder width {arch.get('embed_dim')} in "
                f"{args.checkpoint}; pass --model explicitly."
            )
        # The weights decide, not the CLI defaults. Anything the caller passed
        # that disagrees would only produce a shape mismatch.
        model_name = arch["model_name"]
        patch_size = arch["patch_size"]
        image_size = arch.get("crop_size", image_size)
        pred_depth = arch.get("pred_depth", pred_depth)
        pred_emb_dim = arch.get("pred_emb_dim", pred_emb_dim)
        print(
            f"Architecture from checkpoint: {model_name} patch={patch_size} "
            f"img={image_size} pred_depth={pred_depth} pred_emb_dim={pred_emb_dim}"
        )

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        model_name=model_name,
        crop_size=image_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
    )

    if checkpoint is not None:
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        epoch = checkpoint.get("epoch", "?")
        print(f"Loaded checkpoint epoch {epoch}")
    else:
        print("--random-init: using untrained weights (pipeline check only).")

    encoder.eval()
    predictor.eval()

    img = load_image(args.image, image_size, args.dataset_root).to(device)
    n_h, n_w = patch_grid(image_size, patch_size)

    mask_collator = MBMaskCollator(
        input_size=image_size,
        patch_size=patch_size,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 0.95),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=1,
        allow_overlap=False,
        min_keep=4,
    )

    # Build a dummy batch to get masks
    dummy = [img]
    _, masks_enc, masks_pred = mask_collator(dummy)
    masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
    masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

    h = encoder(img)
    h = F.layer_norm(h, (h.size(-1),))
    B = len(h)

    h_target = apply_masks(h, masks_pred)
    h_target = repeat_interleave_batch(h_target, B, repeat=len(masks_enc))

    z = encoder(img, masks_enc)
    z = predictor(z, masks_enc, masks_pred)

    similarity = F.cosine_similarity(z, h_target, dim=-1)
    mean_sim = similarity.mean().item()
    print(f"Mean target-prediction cosine similarity: {mean_sim:.4f}")

    # Build a similarity heatmap that can be reshaped back into the patch grid
    pred_patches = (
        masks_pred[0][0]
        if len(masks_pred[0]) > 0
        else torch.arange(n_h * n_w, device=device)
    )
    sim_per_patch = torch.zeros(n_h * n_w, device=device)
    for i, pid in enumerate(pred_patches[: similarity.shape[0]]):
        sim_per_patch[int(pid.item())] = similarity[0, i].item()
    heatmap = sim_per_patch.reshape(n_h, n_w).cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Visualise: the masked input image
    _plot_masked_input(img, masks_enc, masks_pred, patch_size, out_dir)

    # Visualise: similarity heatmap over the predicted patches
    _plot_heatmap(heatmap, out_dir / "jepa_prediction_similarity.png")
    print(
        f"Wrote {out_dir / 'jepa_prediction_similarity.png'}  (mean sim={mean_sim:.3f})"
    )
    print(f"Wrote {out_dir / 'jepa_masked_input.png'}")

    return 0


def _plot_masked_input(
    img: torch.Tensor,
    masks_enc: list[torch.Tensor],
    masks_pred: list[torch.Tensor],
    patch_size: int,
    out_dir: Path,
) -> None:
    """Draw the input image with context patches visible and target patches dimmed."""
    import cv2

    B, C, H, W = img.shape
    img_np = img[0].cpu().numpy().transpose(1, 2, 0)
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    n_h, n_w = H // patch_size, W // patch_size

    mask = np.zeros((H, W), dtype=np.uint8)
    dark = np.ones_like(img_np, dtype=np.uint8) * 64

    enc = masks_enc[0] if len(masks_enc) > 0 else torch.empty(0, dtype=torch.long)
    visible = set(int(p) for p in enc[0].cpu().tolist())

    for idx in range(n_h * n_w):
        r, c = divmod(idx, n_w)
        y1, y2 = r * patch_size, (r + 1) * patch_size
        x1, x2 = c * patch_size, (c + 1) * patch_size
        if idx not in visible:
            mask[y1:y2, x1:x2] = 255
            img_np[y1:y2, x1:x2] = dark[y1:y2, x1:x2]

    out_path = out_dir / "jepa_masked_input.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


def _plot_heatmap(heatmap: np.ndarray, path: Path) -> None:
    """Write a colourised heatmap of patch-wise prediction similarity."""
    import cv2

    h_map = (np.clip(heatmap, -1, 1) + 1) / 2 * 255
    h_map = h_map.astype(np.uint8)
    h_map = cv2.resize(h_map, (256, 256), interpolation=cv2.INTER_NEAREST)
    colour = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), colour)


if __name__ == "__main__":
    raise SystemExit(main())
