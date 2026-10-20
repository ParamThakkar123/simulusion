"""Training for every Simulusion family.

:class:`Trainer` runs any :class:`TrainingTask` (one or more per family, see
:mod:`simulusion.training.tasks`) from a :class:`TrainerConfig`;
:func:`run` does it from a YAML run config. The per-family trainers below
(``IRISTrainer``, ``train_diamond``, ...) are the older entrypoints, kept for
existing scripts.
"""

from typing import Any


_CORE = {
    "Trainer": "simulusion.training.trainer",
    "TrainerConfig": "simulusion.training.trainer",
    "TrainResult": "simulusion.training.trainer",
    "TrainingTask": "simulusion.training.task",
    "register_task": "simulusion.training.task",
    "get_task": "simulusion.training.task",
    "list_tasks": "simulusion.training.task",
    "MixedPrecision": "simulusion.training.precision",
    "GradientUpdate": "simulusion.training.precision",
    "run": "simulusion.training.run",
    "resolve_run_config": "simulusion.training.run",
}


def __getattr__(name: str) -> Any:
    if name in _CORE:
        import importlib

        return getattr(importlib.import_module(_CORE[name]), name)
    if name == "train_convae":
        from simulusion.training.train_convvae import train_convae

        return train_convae
    if name == "train_mdn_rnn":
        from simulusion.training.train_mdn_rnn import train_mdn_rnn

        return train_mdn_rnn
    if name == "train_controller":
        from simulusion.training.train_controller import train_controller

        return train_controller
    if name == "DiamondAgent":
        from simulusion.training.train_diamond import DiamondAgent

        return DiamondAgent
    if name == "train_diamond":
        from simulusion.training.train_diamond import train_diamond

        return train_diamond
    if name == "train_dit":
        from simulusion.training.train_dit import train_dit

        return train_dit
    if name == "train_dreamer":
        from simulusion.training.train_dreamer import train_dreamer

        return train_dreamer
    if name == "GenieTrainer":
        from simulusion.training.train_genie import GenieTrainer

        return GenieTrainer
    if name == "create_genie_trainer":
        from simulusion.training.train_genie import create_genie_trainer

        return create_genie_trainer
    if name == "IRISTrainer":
        from simulusion.training.train_iris import IRISTrainer

        return IRISTrainer
    if name == "train_planet":
        from simulusion.training.train_planet import train as train_planet

        return train_planet
    if name == "train_rssm":
        from simulusion.training.train_rssm import train_rssm

        return train_rssm
    if name == "run_atari_100k":
        from simulusion.training.run_atari_100k import run_atari_100k

        return run_atari_100k
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_CORE,
    "DiamondAgent",
    "GenieTrainer",
    "IRISTrainer",
    "create_genie_trainer",
    "run_atari_100k",
    "train_convae",
    "train_controller",
    "train_diamond",
    "train_dit",
    "train_dreamer",
    "train_mdn_rnn",
    "train_planet",
    "train_rssm",
]
