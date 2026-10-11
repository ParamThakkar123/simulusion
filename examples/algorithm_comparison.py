#!/usr/bin/env python3
"""Swap the algorithm, keep the code: one API across every world model.

Simulusion's differentiator is that Dreamer, PlaNet, IRIS, DIAMOND, JEPA, and
Genie are all reachable through the same factory. This script trains several of
them on the same cheap task and plots the result side by side::

    python examples/algorithm_comparison.py
    python examples/algorithm_comparison.py --algos dreamer-v1 dreamer-v2 --steps 20000

The step budget below is small enough to run on a laptop CPU. These are
illustrative runs, **not** published results - see `docs/source/benchmarks.md`
for the real benchmark harness.

## A caveat worth stating plainly

Construction is unified today; *training* is not. `create_model()` and
`create_config()` accept every registered name, but the returned agents do not
yet share one `train()` signature - `Planet.train()` takes epochs, and
`IRISAgent` inherits `nn.Module.train()`, which toggles train/eval mode rather
than training anything. This script therefore trains the algorithms that expose
a step-budget `train()` and reports the rest as "construction only" instead of
pretending they ran.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import simulusion

# Algorithms with a `train(total_steps)` entry point, cheap enough to be
# runnable by a reader on CPU.
DEFAULT_ALGOS = ["dreamer-v1", "dreamer-v2"]


def trains_by_step_budget(agent: Any) -> bool:
    """Does this agent expose a `train(total_steps=...)` loop?

    `nn.Module.train(mode=True)` is a mode toggle, not a training loop, so a
    `train` attribute alone is not enough to go on.
    """

    train = getattr(agent, "train", None)
    if train is None:
        return False
    try:
        params = inspect.signature(train).parameters
    except (TypeError, ValueError):
        return False
    return "total_steps" in params


def run_algorithm(algo: str, env: str, backend: str, steps: int, seed: int) -> dict:
    """Build one agent through the shared factory and train it if it can be."""

    print(f"\n=== {algo} ===")
    # The same two calls for every algorithm - this is the pitch.
    agent = simulusion.create_model(
        algo, env=env, env_backend=backend, seed=seed, no_gpu=False
    )

    result: dict[str, Any] = {"algo": algo, "env": env, "steps": steps}

    if not trains_by_step_budget(agent):
        print(f"  {algo}: constructed; no step-budget train() - skipping training")
        result["status"] = "construction-only"
        return result

    agent.train(total_steps=steps)
    episode_rewards, _, _ = agent.evaluate()

    result["status"] = "trained"
    result["mean_return"] = float(sum(episode_rewards) / len(episode_rewards))
    result["returns"] = [float(r) for r in episode_rewards]
    print(f"  {algo}: mean return {result['mean_return']:.1f} after {steps} steps")
    return result


def plot(results: list[dict], out_path: Path) -> Path | None:
    trained = [r for r in results if r.get("status") == "trained"]
    if not trained:
        print("Nothing trained, so nothing to plot.")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [r["algo"] for r in trained],
        [r["mean_return"] for r in trained],
        color="#4C78A8",
    )
    ax.set_ylabel("Mean evaluation return")
    ax.set_title(
        f"{trained[0]['env']} - {trained[0]['steps']:,} steps "
        "(illustrative, not published results)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--env-backend", default="gym")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/algo_comparison"))
    parser.add_argument(
        "--list",
        action="store_true",
        help="List every registered algorithm and exit",
    )
    args = parser.parse_args()

    if args.list:
        for name in simulusion.list_models():
            print(f"{name:16} {simulusion.get_model_spec(name).description}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_algorithm(algo, args.env, args.env_backend, args.steps, args.seed)
        for algo in args.algos
    ]

    results_path = args.out_dir / "algorithm_comparison.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {results_path}")
    plot(results, args.out_dir / "algorithm_comparison.png")


if __name__ == "__main__":
    main()
