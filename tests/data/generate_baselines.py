#!/usr/bin/env python3
"""Regenerate golden regression baselines.

Usage:
    python tests/data/generate_baselines.py

This re-computes the baseline values for all registered regression tests
and writes them to regression_baselines.json.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASELINE_FILE = _HERE / "regression_baselines.json"

# Import the metric generators from the test module, which needs the repository
# root on sys.path when this file runs as a script.
sys.path.insert(0, str(_HERE.parent.parent))
from tests.test_determinism_regressions import (  # noqa: E402
    _dreamer_rssm_regression_metrics,
    _ppo_regression_metrics,
)

_GENERATORS: dict[str, Callable[[], dict]] = {
    "ppo": _ppo_regression_metrics,
    "dreamer_rssm": _dreamer_rssm_regression_metrics,
}


def main() -> None:
    with open(_BASELINE_FILE) as f:
        current = json.load(f)

    for name, gen_fn in _GENERATORS.items():
        metrics = gen_fn()
        tolerances = current.get(name, {}).get("_tolerances", {})
        entry = {k: v for k, v in metrics.items()}
        tolerances_new = {}
        for k in metrics:
            if k in tolerances:
                tolerances_new[k] = tolerances[k]
            else:
                tolerances_new[k] = 1e-6
        entry["_tolerances"] = tolerances_new
        current[name] = entry
        print(f"Updated {name}: { {k: round(v, 10) for k, v in metrics.items()} }")

    with open(_BASELINE_FILE, "w") as f:
        json.dump(current, f, indent=2)
        f.write("\n")
    print(f"Wrote {_BASELINE_FILE}")


if __name__ == "__main__":
    main()
