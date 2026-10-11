#!/usr/bin/env python3
"""Run inference with a trained Genie checkpoint on TinyWorlds data.

Loads a TinyWorlds sample (or a single image) and generates frames with the
trained Genie model. Saves generated frames as PNGs in save_dir.

Example:
  python scripts/infer_genie_tinyworlds.py \
    checkpoint=checkpoints/genie_sonic_final.pt \
    data_file=/home/azureuser/.cache/tinyworlds/sonic_frames.h5 \
    dataset=SONIC num_frames=16 save_dir=out/genie_infer
"""

import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from simulusion.training.train_genie import create_genie_trainer
from simulusion.datasets import create_tinyworlds_dataloader


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) float tensor to uint8 HxWxC numpy array."""
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round()
    else:
        arr = np.clip(arr, 0, 255).round()
    return arr.astype(np.uint8)


def main():
    cli_cfg = OmegaConf.from_cli()
    checkpoint = cli_cfg.checkpoint
    dataset = cli_cfg.get("dataset", "SONIC")
    data_file = cli_cfg.get("data_file", None)
    num_frames = int(cli_cfg.get("num_frames", 16))
    image_size = int(cli_cfg.get("image_size", 64))
    batch_size = int(cli_cfg.get("batch_size", 2))
    device_str = cli_cfg.get("device", None)
    sample_index = int(cli_cfg.get("sample_index", 0))
    save_dir = cli_cfg.get("save_dir", "out/genie_infer")
    use_maskgit = bool(cli_cfg.get("use_maskgit", False))

    if not checkpoint:
        raise SystemExit("checkpoint=... is required")

    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Using device: {device}")

    trainer, model = create_genie_trainer()
    model.to(device)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        trainer.load_checkpoint(str(ckpt_path))
        print(f"Loaded full trainer checkpoint from {ckpt_path}")
    except Exception:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            print(f"Loaded model_state_dict from checkpoint {ckpt_path}")
        else:
            model.load_state_dict(state)
            print(f"Loaded state_dict from checkpoint {ckpt_path}")

    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    dataset_obj, loader = create_tinyworlds_dataloader(
        dataset_name=dataset,
        num_frames=num_frames,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        download=False,
        data_file=data_file,
    )

    if sample_index >= len(dataset_obj):
        raise IndexError(
            f"sample_index {sample_index} >= dataset size {len(dataset_obj)}"
        )

    sample = dataset_obj[sample_index]
    sample = sample.unsqueeze(0).to(device)

    prompt_frame = (
        sample[:, :, 0, :, :].squeeze(2) if sample.dim() == 5 else sample[:, :, 0, :, :]
    )

    print(f"Prompt frame shape: {prompt_frame.shape}")

    with torch.no_grad():
        generated = model.generate(
            prompt_frame,
            num_frames=num_frames,
            actions=None,
            use_maskgit=use_maskgit,
        )

    generated = generated.cpu()
    B = generated.shape[0]

    for b in range(B):
        gen = generated[b]
        T = gen.shape[1]
        for t in range(T):
            frame = gen[:, t, :, :]
            img = tensor_to_uint8_img(frame)
            out_path = Path(save_dir) / f"sample{sample_index}_b{b}_frame{t:03d}.png"
            Image.fromarray(img).save(out_path)

    print(f"Saved generated frames to {save_dir}")


if __name__ == "__main__":
    main()
