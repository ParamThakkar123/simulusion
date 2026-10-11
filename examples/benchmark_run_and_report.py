"""Run a small benchmark and print IQM + bootstrap CI.

This example demonstrates the programmatic API. It does a light-weight run
and prints the computed IQM and 95% bootstrap CI for the per-seed means.

Usage:
    python examples/benchmark_run_and_report.py
"""

from simulusion.benchmarks.runner import BenchmarkRunner
from simulusion.benchmarks import adapters, metrics


def main():
    out_dir = "results/bench_example"

    # Use IRIS adapter as a lightweight example (change to diamond/dreamer as needed)
    runner = BenchmarkRunner(adapter_cls=adapters.IRISAdapter, out_dir=out_dir)

    # Small run: 2 seeds, 2 episodes each (adjust for real experiments)
    res = runner.run(env_spec={"game": "ALE/Pong-v5"}, seeds=[0, 1], num_episodes=2)

    per_seed = res.get("aggregate", {}).get("per_seed_means", [])
    iqm = metrics.iqm_of_array(per_seed)
    lower, upper = metrics.bootstrap_iqm_ci(per_seed, num_samples=2000, alpha=0.05)

    print("\nBenchmark summary")
    print("-----------------")
    print(f"Per-seed means: {per_seed}")
    print(f"IQM: {iqm:.3f}")
    print(f"95% bootstrap CI for IQM: [{lower:.3f}, {upper:.3f}]")

    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
