"""Example script to benchmark all world models on the same environment.

Usage:
    python examples/benchmark_all_agents.py game=ALE/Pong-v5 seeds=2 episodes=3
"""

import os
import sys
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulusion.benchmarks.runner import MultiAgentBenchmarkRunner
from simulusion.benchmarks import adapters


def main():
    cli_cfg = OmegaConf.from_cli()

    game = cli_cfg.game
    seeds = int(cli_cfg.get("seeds", 2))
    episodes = int(cli_cfg.get("episodes", 3))
    out_dir = cli_cfg.get("out_dir", "results/bench_all")
    device = cli_cfg.get("device", "cuda")

    all_adapters = [
        adapters.DiamondAdapter,
        adapters.IRISAdapter,
        adapters.DreamerV1Adapter,
        adapters.DreamerV2Adapter,
    ]

    print(f"Benchmarking all agents on {game}")
    print(f"Agents: {[cls.__name__ for cls in all_adapters]}")
    print(f"Seeds: {seeds}, Episodes per seed: {episodes}")
    print(f"Output directory: {out_dir}")

    runner = MultiAgentBenchmarkRunner(adapter_classes=all_adapters, out_dir=out_dir)

    results = runner.run_all(
        env_spec={"game": game},
        seeds=list(range(seeds)),
        num_episodes=episodes,
        extra_kwargs={"device": device},
    )

    print("\nBenchmark completed!")
    print(f"Results saved to: {out_dir}")
    print("\nSummary of results:")
    for agent_name, result in results.items():
        aggregate = result.get("aggregate", {})
        iqm = aggregate.get("iqm", 0.0)
        num_seeds = aggregate.get("num_seeds", 0)
        print(f"{agent_name}: IQM = {iqm:.3f} (over {num_seeds} seeds)")


if __name__ == "__main__":
    main()
