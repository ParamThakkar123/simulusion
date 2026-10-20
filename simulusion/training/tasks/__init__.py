"""The built-in training tasks, one or more per family.

Registered lazily: listing them imports nothing; asking for one imports its
family. ``simulusion train --help`` lists them.
"""

from typing import Any

from simulusion.training.task import register_task

#: name -> "module:Class"
BUILTIN_TASKS = {
    "genie": "simulusion.training.tasks.genie:GenieTask",
}

for _name, _target in BUILTIN_TASKS.items():
    register_task(_name, target=_target)

_EXPORTS = {
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
