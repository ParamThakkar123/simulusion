"""DiT: a class-conditional Diffusion Transformer trained on CIFAR-10.

    python demos/dit_demo.py train  --epochs 40
    python demos/dit_demo.py record --run demos/runs/dit

``record`` writes into ``<run>/media``:

* ``samples.png`` -- one row per CIFAR-10 class, sampled with classifier-free
  guidance.
* ``denoising.mp4`` -- the same grid being denoised from pure noise, one frame
  every few of the 1000 DDPM steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import grid, hstack, label, upscale, vstack, write_mp4, write_png  # noqa: E402

CLASSES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def train(args: argparse.Namespace) -> None:
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "simulusion.training.train_dit",
        "DATASET=cifar10",
        f"ROOT_PATH={args.data}",
        f"WORKDIR={run.as_posix()}",
        f"EPOCHS={args.epochs}",
        f"BATCH={args.batch_size}",
        # DiT-S/4 at 32x32
        "PATCH=4",
        "WIDTH=384",
        "DEPTH=12",
        "HEADS=6",
        "NUM_CLASSES=10",
        "EMA=true",
        # The default 0.9999 averages over ~10k steps, longer than this run.
        "EMA_DECAY=0.999",
        "NUM_WORKERS=4",
        "CHECKPOINT_EVERY=10",
    ]
    print(" ".join(cmd), flush=True)
    with (run / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=_HERE.parent
        )


def record(args: argparse.Namespace) -> None:
    import torch

    from simulusion.configs.dit_config import DiTConfig
    from simulusion.models.diffusion.DDPM import DDPM
    from simulusion.models.diffusion.DiT import DiT

    run = Path(args.run)
    out = run / "media"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT.from_pretrained(run, map_location=device).to(device).eval()
    cfg = model.config
    trained = DiTConfig.from_yaml(
        run / "config.yaml"
    )  # the noise schedule it was trained with
    ddpm = DDPM(trained.TIMESTEPS, trained.BETA_START, trained.BETA_END).to(device)

    n = args.per_class
    y = torch.arange(10, device=device).repeat_interleave(n)

    class Guided(torch.nn.Module):
        """Classifier-free guidance as a plain eps-model for DDPM.p_sample."""

        def forward(self, x, t):
            # forward_with_cfg runs the conditional and null-label halves as one batch.
            both = model.forward_with_cfg(
                torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, y]), args.cfg_scale
            )
            return both[: len(x)]

    guided = Guided()

    torch.manual_seed(args.seed)
    x = torch.randn(len(y), cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)
    snapshots = []
    with torch.no_grad():
        for i in reversed(range(ddpm.timesteps)):
            t = torch.full((len(y),), i, dtype=torch.long, device=device)
            x = ddpm.p_sample(guided, x, t)
            if i % args.snapshot_every == 0 or i < 20 and i % 4 == 0:
                snapshots.append(((x.clamp(-1, 1) + 1) / 2).cpu())

    def sheet(batch, step_label: str | None = None):
        images = [upscale(img.permute(1, 2, 0).numpy(), 2) for img in batch]
        rows = []
        for c in range(10):
            row = grid(images[c * n : (c + 1) * n], cols=n, gap=2)
            rows.append(
                hstack([label(np.zeros((row.shape[0], 70, 3)), CLASSES[c]), row], gap=0)
            )
        body = vstack(rows, gap=2)
        return label(body, step_label, accent=True) if step_label else body

    final = snapshots[-1]
    print("wrote", write_png(sheet(final), out / "samples.png"))
    frames = [sheet(s, "denoising") for s in snapshots]
    frames += [sheet(final, "sample")] * 12  # hold the final grid
    print("wrote", write_mp4(frames, out / "denoising.mp4", fps=args.fps))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--data", default="data")
    t.add_argument("--run", default="demos/runs/dit")

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/dit")
    r.add_argument("--per-class", type=int, default=8)
    r.add_argument("--cfg-scale", type=float, default=4.0)
    r.add_argument("--snapshot-every", type=int, default=25)
    r.add_argument("--fps", type=int, default=12)
    r.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
