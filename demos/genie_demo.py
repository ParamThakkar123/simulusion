"""Genie: learn a playable world model from unlabelled gameplay video.

    python demos/genie_demo.py train  --steps 6000
    python demos/genie_demo.py record --run demos/runs/genie

Genie is trained on raw video with no actions. A latent action model discovers
a small vocabulary of actions (8 by default) from how consecutive frames
differ, and a dynamics model learns to generate the next frame given one.

``record`` writes into ``<run>/media``:

* ``tokens.mp4`` -- a real clip beside its reconstruction from video tokens.
* ``replay.mp4`` -- a real clip beside Genie's generation from only its first
  frame, driven by the latent actions inferred from the real clip.
* ``actions.mp4`` -- one prompt frame, generated forward under each of the
  latent actions: the "controls" Genie discovered on its own.
* ``curve.png`` -- reconstruction and dynamics loss over training.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import grid, hstack, label, plot_curves, upscale, write_mp4  # noqa: E402

# A ~25M-parameter Genie that trains on a 4 GB GPU. The default GenieSmallConfig
# (~466M parameters) needs ~7.5 GB for AdamW state alone.
ARCH = dict(
    tokenizer_encoder_dim=128,
    tokenizer_decoder_dim=256,
    tokenizer_encoder_depth=4,
    tokenizer_decoder_depth=4,
    tokenizer_num_heads=8,
    tokenizer_vocab_size=512,
    action_encoder_dim=128,
    action_decoder_dim=128,
    action_encoder_depth=2,
    action_num_heads=8,
    dynamics_dim=256,
    dynamics_depth=6,
    dynamics_num_heads=8,
)


def train(args: argparse.Namespace) -> None:
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/train_genie_tinyworlds.py",
        f"dataset={args.dataset}",
        f"num_frames={args.num_frames}",
        "image_size=64",
        f"batch_size={args.batch_size}",
        "num_workers=0",
        f"max_steps={args.steps}",
        "warmup_steps=500",
        "learning_rate=3e-4",
        # Batch 4 x 16 frames needs ~5 GB and, on Windows, silently spills into
        # system memory at ~7 s/step. Batch 3 with AMP fits in ~3.2 GB.
        "use_amp=true",
        "log_interval=50",
        "checkpoint_interval=2000",
        f"checkpoint_dir={run.as_posix()}",
        *(f"{k}={v}" for k, v in ARCH.items()),
    ]
    print(" ".join(cmd), flush=True)
    with (run / "train.log").open("w", encoding="utf-8") as log:
        # The trainer is a plain script, so make the repo's `simulusion` importable.
        paths = [str(_HERE.parent), os.environ.get("PYTHONPATH", "")]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
        subprocess.run(
            cmd,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=_HERE.parent,
            env=env,
        )


def _clip_frames(video) -> list[np.ndarray]:
    """(C, T, H, W) in [0, 1] -> list of HWC frames."""
    return [
        video[:, t].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        for t in range(video.shape[1])
    ]


def record(args: argparse.Namespace) -> None:
    import torch

    from simulusion.datasets.tinyworlds import TinyWorldsDataset
    from simulusion.models.genie import Genie

    run = Path(args.run)
    out = run / "media"
    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint
        else run / f"genie_{args.dataset.lower()}_final.pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Genie.from_pretrained(ckpt, map_location=device).to(device).eval()
    model.sampler.temperature = args.temperature
    T = model.num_frames
    print(f"Loaded {ckpt}")

    data = TinyWorldsDataset(args.dataset, num_frames=T, image_size=64)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    with torch.no_grad():
        # 1. Tokenizer reconstructions and action-conditioned replays of real clips.
        tokens, replay = [], []
        for idx in rng.integers(0, len(data), size=args.clips):
            clip = data[int(idx)].unsqueeze(0).to(device)  # (1, C, T, H, W)
            recon = model.video_tokenizer(clip)[0].clamp(0, 1)
            actions = model.infer_actions(clip)
            generated = model.generate(
                clip[:, :, 0],
                num_frames=T,
                actions=actions,
                use_maskgit=True,
                use_cache=True,
            )
            for real_f, recon_f, gen_f in zip(
                _clip_frames(clip[0]),
                _clip_frames(recon[0]),
                _clip_frames(generated[0]),
            ):
                tokens.append(
                    hstack(
                        [
                            label(upscale(real_f, 4), "Real"),
                            label(upscale(recon_f, 4), "Tokenized", accent=True),
                        ]
                    )
                )
                replay.append(
                    hstack(
                        [
                            label(upscale(real_f, 4), "Real"),
                            label(upscale(gen_f, 4), "Genie", accent=True),
                        ]
                    )
                )
        print("wrote", write_mp4(tokens, out / "tokens.mp4", fps=args.fps))
        print("wrote", write_mp4(replay, out / "replay.mp4", fps=args.fps))

        # 2. One prompt frame, every latent action held for the whole clip.
        prompt = data[int(rng.integers(0, len(data)))][:, 0].unsqueeze(0).to(device)
        n_actions = model.latent_action_model.vocab_size
        prompts = prompt.expand(n_actions, -1, -1, -1)
        acts = torch.arange(n_actions, device=device)[:, None].expand(n_actions, T - 1)
        videos = model.generate(
            prompts, num_frames=T, actions=acts, use_maskgit=True, use_cache=True
        )
        frames = []
        for t in range(T):
            tiles = [
                label(
                    upscale(
                        videos[a, :, t].permute(1, 2, 0).clamp(0, 1).cpu().numpy(), 3
                    ),
                    f"action {a}",
                )
                for a in range(n_actions)
            ]
            frames.append(grid(tiles, cols=4))
        print("wrote", write_mp4(frames, out / "actions.mp4", fps=args.fps // 2 or 1))

    log = (run / "train.log").read_text(encoding="utf-8", errors="replace")
    rows = re.findall(
        r"Step (\d+)/\d+ \| Loss: [\d.]+ \| Recon: ([\d.]+) .*?Dynamics: ([\d.]+)", log
    )
    if rows:
        steps = [int(r[0]) for r in rows]
        plot_curves(
            {
                "reconstruction": (steps, [float(r[1]) for r in rows]),
                "dynamics": (steps, [float(r[2]) for r in rows]),
            },
            out / "curve.png",
            title=f"Genie on TinyWorlds {args.dataset.title()}",
            xlabel="gradient steps",
            ylabel="loss",
        )
        print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--dataset", default="SONIC")
    t.add_argument("--num-frames", type=int, default=16)
    t.add_argument("--batch-size", type=int, default=3)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--run", default="demos/runs/genie")

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/genie")
    r.add_argument("--dataset", default="SONIC")
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--clips", type=int, default=4)
    r.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="token sampling temperature; each frame is one independent pass",
    )
    r.add_argument("--fps", type=int, default=8)
    r.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
