"""Score DreamerV3's BSuite runs with BSuite's own analysis.

Each run directory written by ``dreamer_v3_demo.py train --benchmark bsuite``
holds a ``bsuite_csv`` log. This reads them with ``bsuite.logging.csv_load``
and scores each with that experiment's ``analysis.score`` (the numbers BSuite
reports and the DreamerV3 paper's Table L.1 averages over every setting).

    python demos/bsuite_scores.py demos/runs/dreamer_v3_bsuite_*
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

#: DreamerV3's per-environment score averaged over all settings (arXiv v1,
#: Table L.1); ours runs one setting, so this is context, not a target.
PAPER = {"bandit": 0.859, "umbrella_length": 0.783, "mnist": 0.762, "memory_len": 0.478}


def score_run(run: Path) -> dict:
    from bsuite.logging import csv_load

    df, _ = csv_load.load_bsuite(str(run / "bsuite_csv"))
    rows = {}
    for bsuite_id, group in df.groupby("bsuite_id"):
        env = str(bsuite_id).split("/")[0]
        analysis = importlib.import_module(f"bsuite.experiments.{env}.analysis")
        sweep = importlib.import_module(f"bsuite.experiments.{env}.sweep")
        episodes = int(group["episode"].max())
        rows[str(bsuite_id)] = {
            "score": float(analysis.score(group)),
            "episodes": episodes,
            "budget": int(sweep.NUM_EPISODES),
            "finished": episodes >= int(sweep.NUM_EPISODES),
            "paper_mean_over_settings": PAPER.get(env),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    results = {}
    for run in args.runs:
        if (run / "bsuite_csv").is_dir():
            results[run.name] = score_run(run)
    for name, rows in results.items():
        for bsuite_id, row in rows.items():
            print(
                f"{name:40s} {bsuite_id:20s} score {row['score']:.3f} "
                f"({row['episodes']}/{row['budget']} episodes; "
                f"paper mean over settings {row['paper_mean_over_settings']})"
            )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
