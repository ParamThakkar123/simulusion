"""Train and record every demo in sequence, then rebuild the docs gallery.

    python demos/run_all.py                 # everything not recorded yet
    python demos/run_all.py dreamer genie   # just these
    python demos/run_all.py --record-only   # re-record from existing runs

Each demo runs alone: several of them hold gigabytes of replay data in RAM and
most of a 4 GB GPU, so running two at once fails in confusing ways. Logs go to
``demos/runs/<demo>.log``. On the reference machine (RTX 3050 Laptop, 4 GB)
the full sequence takes about thirteen hours.

Safe to stop and rerun: finished demos are skipped, and the Dreamer demos
continue from their newest checkpoint. Any other demo that was interrupted
starts its training again from scratch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# name: (script, train args, record args, file that marks the demo as done)
DEMOS = {
    "dreamer": ("dreamer_demo.py", ["--steps", "60000", "--resume"], [], "dream.mp4"),
    "dreamer_v2": (
        "dreamer_demo.py",
        [
            "--algo",
            "dreamer-v2",
            "--steps",
            "60000",
            "--run",
            "demos/runs/dreamer_v2",
            "--resume",
            # V2's two-hot heads raise its peak; at batch 50 PyTorch holds
            # ~3.7 GB of a 4 GB card and MuJoCo's OpenGL renderer is killed.
            "--batch-size",
            "32",
        ],
        ["--run", "demos/runs/dreamer_v2"],
        "dream.mp4",
    ),
    "planet": ("planet_demo.py", ["--minutes", "75"], [], "dream.mp4"),
    "modular_rssm": (
        "modular_rssm_demo.py",
        ["--minutes-per-backbone", "25"],
        [],
        "dream.mp4",
    ),
    "diamond": ("diamond_demo.py", ["--epochs", "21"], [], "dream.mp4"),
    # The gallery's Boxing run: 300 epochs, several days on a 4 GB laptop GPU.
    # --until-epoch makes --resume stop at the same epoch after any restart.
    "iris_boxing": (
        "iris_demo.py",
        [
            *("--game", "ALE/Boxing-v5", "--run", "demos/runs/iris_boxing"),
            *("--until-epoch", "300", "--minutes", "100000", "--resume"),
            *("--autoencoder-steps", "200", "--transformer-steps", "200"),
            *("--actor-critic-steps", "50", "--actor-critic-batch", "64"),
        ],
        ["--run", "demos/runs/iris_boxing", "--policy-steps", "2000"],
        "dream.mp4",
    ),
    "genie": ("genie_demo.py", ["--steps", "6000"], [], "replay.mp4"),
    "dit": ("dit_demo.py", ["--epochs", "40", "--batch-size", "64"], [], "samples.png"),
    "jepa": ("jepa_demo.py", ["--epochs", "30"], [], "neighbours.png"),
}


def run(cmd: list[str], log: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    started = time.time()
    with log.open("a", encoding="utf-8") as f:
        subprocess.run(
            cmd, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=HERE.parent
        )
    print(f"  done in {(time.time() - started) / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("demos", nargs="*", help=f"any of: {', '.join(DEMOS)}")
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="redo demos that are already recorded"
    )
    args = parser.parse_args()

    unknown = sorted(set(args.demos) - set(DEMOS))
    if unknown:
        parser.error(f"unknown demo(s) {unknown}; choose from {list(DEMOS)}")
    RUNS.mkdir(parents=True, exist_ok=True)
    for name in args.demos or list(DEMOS):
        script, train_args, record_args, marker = DEMOS[name]
        if (RUNS / name / "media" / marker).exists() and not args.force:
            print(f"{name}: already recorded, skipping")
            continue
        log = RUNS / f"{name}.log"
        script_path = str(HERE / script)
        if not args.record_only:
            run([sys.executable, script_path, "train", *train_args], log)
        run([sys.executable, script_path, "record", *record_args], log)

    run([sys.executable, str(HERE / "build_gallery.py")], RUNS / "gallery.log")


if __name__ == "__main__":
    main()
