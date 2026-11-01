"""Dreamer training entrypoint for the Simulusion CLI.

Usage:
    simulusion train dreamer env_backend=dmc env=walker-walk total_steps=5_000_000
    python -m simulusion.training.train_dreamer env_backend=gym env=Pendulum-v1
"""

from typing import Any
from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.experiments import (
    dump_config,
    instantiate_dataclass,
    parse_experiment_args,
)
from simulusion.models.dreamer import DreamerAgent


def train_dreamer(config: DreamerConfig | None = None, **kwargs: Any) -> DreamerAgent:
    if config is None:
        config = DreamerConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    agent = DreamerAgent(config)
    agent.train()
    return agent


def main(argv: list[str] | None = None) -> DreamerConfig:
    args = parse_experiment_args(argv, description="Train Dreamer")
    config = instantiate_dataclass(DreamerConfig, args.config, args.overrides)
    if args.print_config:
        print(dump_config(config.__dict__))
        return config
    train_dreamer(config=config)
    return config


if __name__ == "__main__":
    main()
