Benchmarking utilities for Simulusion
================================

This folder contains a small benchmarking harness to run standardized evaluations
for different world-model agents (DIAMOND, IRIS, DreamerV1/V2) and export
results in formats suitable for research reports (JSON/CSV/Markdown).

Quickstart
----------

1. Run from the CLI (example):

   ```bash
   # run IRIS on Pong for 3 episodes using seed 0
   python -m simulusion.benchmarks.cli --agent iris --game ALE/Pong-v5 --seeds 1 --episodes 3
   
   # run DIAMOND on Breakout with two seeds
   python -m simulusion.benchmarks.cli --agent diamond --game Breakout-v5 --seeds 0,1 --episodes 5
   
   # run ALL agents on the same game for comparison
   python -m simulusion.benchmarks.cli --all-agents --game ALE/Pong-v5 --seeds 2 --episodes 5
   ```

2. Outputs are written to `results/bench` by default (change with `--out_dir`).

Design Notes
------------

- Adapters normalize each agent's `evaluate()` output into a dict with
  `episode_returns: List[float]`. Optional keys include `videos` and `latents`.
- `runner.py` orchestrates per-seed evaluations and exports JSON/CSV/Markdown.
- `metrics.py` contains basic aggregate computations (mean, median, IQM) and a
  simple bootstrap CI helper.

Extending
---------

- Add a new adapter in `adapters.py` implementing `load_checkpoint()` and
  `evaluate(num_episodes, render)` that returns standardized outputs.
- Register the adapter in `cli.py` to expose it via the command-line.

CI / Tests
----------

- Add smoke tests under `simulusion/benchmarks/tests/` to validate adapters
  without running full training (the repo contains a `mocking_classes.py` file
  you can use to create lightweight fake agents/environments for CI).

License
-------

Same license as the project: see top-level `LICENSE`.
