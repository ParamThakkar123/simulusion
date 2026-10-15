from __future__ import annotations

from typing import TYPE_CHECKING, List
from simulusion.utils.device import default_device_name

from simulusion.benchmarks.runner import BenchmarkRunner, MultiAgentBenchmarkRunner
from simulusion.benchmarks import adapters

if TYPE_CHECKING:
    from omegaconf import DictConfig

AGENTS = {
    "diamond": adapters.DiamondAdapter,
    "iris": adapters.IRISAdapter,
    "dreamerv1": adapters.DreamerV1Adapter,
    "dreamerv2": adapters.DreamerV2Adapter,
}


def parse_seeds(s: str) -> List[int]:
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if s.isdigit():
        return list(range(int(s)))
    return [int(s)]


def main(cfg: "DictConfig") -> None:
    """Run benchmark for an agent on an environment."""

    if not cfg.get("all_agents", False) and not cfg.get("agent"):
        raise SystemExit("Either agent=... or all_agents=true must be specified")

    if cfg.get("all_agents", False) and cfg.get("agent"):
        raise SystemExit("agent and all_agents=true cannot be used together")

    if cfg.get("agent") and cfg.get("checkpoint") is None:
        raise SystemExit(
            "checkpoint=... is required when using agent=.... Only trained models should be benchmarked."
        )

    game = cfg.game
    env_backend = cfg.get("env_backend", None)
    device = cfg.get("device", default_device_name())
    seeds = parse_seeds(str(cfg.get("seeds", "1")))
    episodes = int(cfg.get("episodes", 5))
    checkpoint = cfg.get("checkpoint", None)
    out_dir = cfg.get("out_dir", "results/bench")
    preset = cfg.get("preset", None)
    train_epochs = cfg.get("train_epochs", None)

    if cfg.get("all_agents", False):
        all_adapters = list(AGENTS.values())
        multi_runner = MultiAgentBenchmarkRunner(
            adapter_classes=all_adapters, out_dir=out_dir
        )

        multi_runner.run_all(
            env_spec={
                "game": game,
                **({"env_backend": env_backend.lower()} if env_backend else {}),
            },
            seeds=seeds,
            num_episodes=episodes,
            checkpoints=None,
            extra_kwargs={"device": device, "preset": preset},
            train_epochs=train_epochs,
        )

        print("Multi-agent benchmark finished. Results written to:", out_dir)
    else:
        adapter_cls = AGENTS[str(cfg.agent)]

        runner = BenchmarkRunner(adapter_cls=adapter_cls, out_dir=out_dir)

        env_spec = {"game": game}
        if env_backend:
            env_spec["env_backend"] = env_backend.lower()

        runner.run(
            env_spec=env_spec,
            seeds=seeds,
            num_episodes=episodes,
            checkpoint=checkpoint,
            extra_kwargs={"device": device, "preset": preset},
        )

        print("Benchmark finished. Results written to:", out_dir)


def run() -> None:
    """Hydra entry point. Hydra and OmegaConf are imported only here so that
    ``simulusion benchmark`` (which just needs :data:`AGENTS`) works without them.
    """
    import hydra

    hydra.main(version_base=None, config_path=None, config_name=None)(main)()


if __name__ == "__main__":
    run()
