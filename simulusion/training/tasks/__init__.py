"""The built-in training tasks, one or more per family.

Registered lazily: listing them imports nothing; asking for one imports its
family. ``simulusion train --help`` lists them.
"""

from typing import Any

from simulusion.training.task import register_task

#: name -> "module:Class"
BUILTIN_TASKS = {
    "genie": "simulusion.training.tasks.genie:GenieTask",
    "dreamer": "simulusion.training.tasks.dreamer:DreamerTask",
    "dreamer-v1": "simulusion.training.tasks.dreamer:DreamerV1Task",
    "dreamer-v2": "simulusion.training.tasks.dreamer:DreamerV2Task",
    "dreamer-wm": "simulusion.training.tasks.dreamer:DreamerWorldModelTask",
    "planet": "simulusion.training.tasks.planet:PlanetTask",
    "iris": "simulusion.training.tasks.iris:IRISTask",
    "iris-wm": "simulusion.training.tasks.iris:IRISWorldModelTask",
    "diamond": "simulusion.training.tasks.diamond:DiamondTask",
    "diamond-wm": "simulusion.training.tasks.diamond:DiamondWorldModelTask",
    "dit": "simulusion.training.tasks.dit:DiTTask",
    "jepa": "simulusion.training.tasks.jepa:JEPATask",
    "vae": "simulusion.training.tasks.world_models:VAETask",
    "mdrnn": "simulusion.training.tasks.world_models:MDRNNTask",
    "controller": "simulusion.training.tasks.world_models:ControllerTask",
}

for _name, _target in BUILTIN_TASKS.items():
    register_task(_name, target=_target)

_EXPORTS = {
    "ControllerTask": "simulusion.training.tasks.world_models",
    "DiTTask": "simulusion.training.tasks.dit",
    "JEPATask": "simulusion.training.tasks.jepa",
    "MDRNNTask": "simulusion.training.tasks.world_models",
    "VAETask": "simulusion.training.tasks.world_models",
    "DiamondTask": "simulusion.training.tasks.diamond",
    "DiamondTaskConfig": "simulusion.training.tasks.diamond",
    "DiamondWorldModelTask": "simulusion.training.tasks.diamond",
    "DiamondWorldModelTaskConfig": "simulusion.training.tasks.diamond",
    "DreamerTask": "simulusion.training.tasks.dreamer",
    "DreamerTaskConfig": "simulusion.training.tasks.dreamer",
    "DreamerWorldModelTask": "simulusion.training.tasks.dreamer",
    "DreamerWorldModelTaskConfig": "simulusion.training.tasks.dreamer",
    "IRISTask": "simulusion.training.tasks.iris",
    "IRISTaskConfig": "simulusion.training.tasks.iris",
    "IRISWorldModelTask": "simulusion.training.tasks.iris",
    "IRISWorldModelTaskConfig": "simulusion.training.tasks.iris",
    "PlanetTask": "simulusion.training.tasks.planet",
    "PlanetTaskConfig": "simulusion.training.tasks.planet",
    "GenieTask": "simulusion.training.tasks.genie",
    "GenieTaskConfig": "simulusion.training.tasks.genie",
    "SequenceDataConfig": "simulusion.training.tasks.common",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted([*_EXPORTS, "BUILTIN_TASKS"])
