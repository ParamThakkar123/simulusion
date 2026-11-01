"""World Models on CarRacing: :mod:`~simulusion.training.train_world_model`
with ``--env CarRacing-v3`` unless another ``--env`` is given.

Usage:
    python -m simulusion.training.train_world_model_carracing --stage vae
    python -m simulusion.training.train_world_model_carracing --test

Every other flag is ``train_world_model``'s.
"""

import sys
from typing import Optional, Sequence

from simulusion.training.train_world_model import (
    generate_rollouts,
    run_training_pipeline,
    test_trained_model,
)
from simulusion.training.train_world_model import main as _main

ENV_NAME = "CarRacing-v3"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--env" or a.startswith("--env=") for a in args):
        args = ["--env", ENV_NAME, *args]
    _main(args)


__all__ = [
    "ENV_NAME",
    "generate_rollouts",
    "main",
    "run_training_pipeline",
    "test_trained_model",
]


if __name__ == "__main__":
    main()
