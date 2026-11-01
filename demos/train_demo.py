#!/usr/bin/env python3
"""Launch a demo-oriented training run for a Simulusion algorithm.

The stock defaults are tuned for full research runs: DreamerConfig ships
``checkpoint_interval=10000``, ``test_interval=10000`` and ``log_video_freq=-1``,
so a short run finishes without ever writing a checkpoint or a video. This
launcher overrides those intervals so that every run leaves behind the
artifacts a demo needs — checkpoints to replay from, videos to show, and a
metrics file to plot.

It composes the same ``key=value`` overrides the training entrypoints already
accept and prints the underlying command before running, so anything here can
be copied and run directly.

Usage:
    python demos/train_demo.py --algo dreamer
    python demos/train_demo.py --algo dreamer --env walker-walk --env-backend dmc
    python demos/train_demo.py --algo diamond --preset small --steps 200
    python demos/train_demo.py --algo iris --dry-run
    python demos/train_demo.py --algo dreamer -- total_steps=1000000 seed=3
"""

from __future__ import annotations

import argparse
import subprocess
import sys

TRAINING_MODULES = {
    "dreamer": "simulusion.training.train_dreamer",
    "diamond": "simulusion.training.train_diamond",
    "iris": "simulusion.training.train_iris",
    "genie": "scripts/train_genie_tinyworlds.py",
    "ijepa": "simulusion.training.train_jepa",
}

# Demo defaults per algorithm. `steps` means total env steps for Dreamer and
# epochs for the epoch-driven DIAMOND/IRIS trainers, which is why each profile
# builds its own overrides instead of sharing one budget knob.
DEFAULT_STEPS = {
    "dreamer": 100_000,
    "diamond": 500,
    "iris": 100,
    "genie": 1000,
    "ijepa": 5,
}
DEFAULT_ENVS = {
    "dreamer": "Pendulum-v1",
    "diamond": "Breakout-v5",
    "iris": "ALE/Pong-v5",
    "genie": "SONIC",
    "ijepa": "cifar10",
}


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to cuda when a GPU is visible, else cpu."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def genie_overrides(args: argparse.Namespace) -> list[str]:
    """Build Genie overrides (OmegaConf key=value for scripts/train_genie_tinyworlds.py)."""
    steps = args.steps
    overrides = [
        f"dataset={args.env}",
        f"max_steps={steps}",
        f"device={resolve_device(args.device)}",
        "batch_size=2",
        "checkpoint_dir=checkpoints/genie_demo",
    ]
    return overrides


def ijepa_overrides(args: argparse.Namespace) -> list[str]:
    """Build I-JEPA overrides (key=value dot-list, fed via Hydra-like syntax)."""
    overrides = [
        f"data.batch_size={args.batch_size or 16}",
        "data.num_workers=0",
        f"data.dataset={args.env}",
        "data.download=True",
        f"optimization.epochs={args.steps}",
        "optimization.warmup=0",
        "meta.use_bfloat16=False",
        "logging.folder=results/jepa_demo",
    ]
    return overrides


def dreamer_overrides(args: argparse.Namespace) -> list[str]:
    """Build Dreamer overrides that emit checkpoints, videos and eval metrics."""
    steps = args.steps
    # Five checkpoints/videos across the run reads well in a demo without
    # drowning the run in eval episodes.
    interval = max(1000, steps // 5)
    overrides = [
        f"env={args.env}",
        f"total_steps={steps}",
        f"seed={args.seed}",
        f"checkpoint_interval={interval}",
        f"test_interval={interval}",
        f"log_video_freq={interval}",
        # seed_steps defaults to 5000; a run shorter than that would only ever
        # collect random data and never train.
        f"seed_steps={min(1000, max(200, steps // 10))}",
        "test_episodes=3",
        "video_format=mp4",
        f"enable_tensorboard={args.tensorboard}",
        f"exp_name={args.name}",
    ]
    if args.env_backend:
        overrides.append(f"env_backend={args.env_backend}")
    if resolve_device(args.device) == "cpu":
        overrides.append("no_gpu=True")
    return overrides


def diamond_overrides(args: argparse.Namespace) -> list[str]:
    """Build DIAMOND overrides sized for a single consumer GPU."""
    epochs = args.steps
    interval = max(1, epochs // 5)
    overrides = [
        f"game={args.env}",
        f"num_epochs={epochs}",
        f"seed={args.seed}",
        f"save_interval={interval}",
        f"eval_interval={interval}",
        "log_interval=1",
        f"device={resolve_device(args.device)}",
    ]
    if args.preset:
        overrides.append(f"preset={args.preset}")
    if args.batch_size:
        overrides.append(f"batch_size={args.batch_size}")
    return overrides


def iris_overrides(args: argparse.Namespace) -> list[str]:
    """Build IRIS overrides.

    ``game``/``device``/``seed`` are runtime options rather than IRISConfig
    fields; ``train_iris.main`` splits them out before composing the config.
    """
    return [
        f"env={args.env}",
        f"game={args.env}",
        f"total_epochs={args.steps}",
        f"seed={args.seed}",
        f"device={resolve_device(args.device)}",
    ]


OVERRIDE_BUILDERS = {
    "dreamer": dreamer_overrides,
    "diamond": diamond_overrides,
    "iris": iris_overrides,
    "genie": genie_overrides,
    "ijepa": ijepa_overrides,
}


def build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    """Compose the full training command."""
    entry = TRAINING_MODULES[args.algo]
    overrides = OVERRIDE_BUILDERS[args.algo](args)
    if entry.endswith(".py"):
        cmd = [sys.executable, entry]
    else:
        cmd = [sys.executable, "-m", entry]
    # User overrides go last so they win over the demo defaults.
    return [*cmd, *overrides, *extra]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a Simulusion training job with demo-friendly artifact intervals",
        epilog="Anything after `--` is forwarded verbatim as key=value overrides.",
    )
    parser.add_argument(
        "--algo",
        "-a",
        required=True,
        choices=sorted(TRAINING_MODULES),
        help="Algorithm to train. DiT has no entrypoint; use record_dit.py instead.",
    )
    parser.add_argument(
        "--env",
        "-e",
        default=None,
        help="Environment/game id. Defaults per algorithm: "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_ENVS.items()),
    )
    parser.add_argument(
        "--env-backend",
        default=None,
        help="Dreamer only: gym, dmc, mujoco, brax, ... (dmc needs the [dmc] extra).",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=None,
        help="Env steps for dreamer, epochs for diamond/iris. Defaults: "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_STEPS.items()),
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["small", "medium", "large"],
        help="DIAMOND model preset. Use small for GPUs with <8 GB of VRAM.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="DIAMOND batch size override."
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--name", default="demo", help="Dreamer experiment name.")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Dreamer only: enable TensorBoard logging (needs the [ml] extra).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    args, extra = parser.parse_known_args()

    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.env is None:
        args.env = DEFAULT_ENVS[args.algo]
    if args.steps is None:
        args.steps = DEFAULT_STEPS[args.algo]
    return args, extra


def artifact_hint(algo: str) -> str:
    """Tell the user where to look for the checkpoint after training."""
    if algo == "dreamer":
        return "Checkpoints: runs/<env>_<algo>_<name>_<timestamp>/ckpts/<step>_ckpt.pt"
    if algo == "diamond":
        return "Checkpoints: checkpoints/diamond/checkpoint_<epoch>.pt"
    if algo == "iris":
        return (
            "Checkpoints: checkpoints/iris/checkpoint_<epoch>.pt\n"
            "Note: train_iris hardcodes device=cuda and seed=42; --device/--seed are ignored."
        )
    if algo == "genie":
        return "Checkpoints: checkpoints/genie_demo/genie_<dataset>_final.pt"
    if algo == "ijepa":
        return "Checkpoints: results/jepa_demo/jepa_run-latest.pth.tar"
    return ""


def main() -> int:
    args, extra = parse_args()
    cmd = build_command(args, extra)

    print(f"Device: {resolve_device(args.device)}")
    print(f"Running: {' '.join(cmd)}")
    print(artifact_hint(args.algo))
    if args.dry_run:
        return 0

    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
