"""Nested dataclass configs to and from plain dicts.

:mod:`simulusion.configs.serialization` handles one flat config. A training
run nests them (the trainer's settings, the model's, the data's), so these
helpers walk dataclass-typed fields recursively. Unknown keys are errors
that name their full dotted path, so a typo in a YAML file fails the run
before it starts instead of being silently ignored.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, Mapping, TypeVar, Union

T = TypeVar("T")


class ConfigError(ValueError):
    """A config mapping does not fit its dataclass."""


def to_plain(value: Any) -> Any:
    """``value`` as YAML-safe builtins, recursing into dataclasses.

    Dataclass fields marked ``metadata={"serialize": False}`` (live objects
    such as a prebuilt environment) are left out.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_plain(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.metadata.get("serialize", True)
        }
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def from_plain(cls: type[T], values: Mapping[str, Any] | None, path: str = "") -> T:
    """Build dataclass ``cls`` from a mapping, recursing into nested dataclasses.

    Missing keys keep the field's default. Lists become tuples where the field
    is a tuple. ``path`` prefixes error messages (``"model."``).
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass.")
    values = dict(values or {})
    hints = _type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        names = ", ".join(f"{path}{key}" for key in unknown)
        raise ConfigError(f"Unknown config field(s) for {cls.__name__}: {names}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        kwargs[name] = _coerce(hints.get(name, Any), value, f"{path}{name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Could not build {cls.__name__} at '{path or '.'}': {exc}")


def _coerce(hint: Any, value: Any, path: str) -> Any:
    target = _dataclass_in(hint)
    if target is not None and isinstance(value, Mapping):
        return from_plain(target, value, f"{path}.")
    if isinstance(value, list) and _is_tuple(hint):
        return tuple(value)
    return value


def _dataclass_in(hint: Any) -> type | None:
    """The dataclass ``hint`` names, looking inside ``Optional[...]``."""
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return hint
    origin = typing.get_origin(hint)
    if origin is Union or origin is types.UnionType:
        found = [a for a in typing.get_args(hint) if _dataclass_in(a) is not None]
        if len(found) == 1:
            return _dataclass_in(found[0])
    return None


def _is_tuple(hint: Any) -> bool:
    if hint is tuple or typing.get_origin(hint) is tuple:
        return True
    origin = typing.get_origin(hint)
    if origin is Union or origin is types.UnionType:
        return any(_is_tuple(a) for a in typing.get_args(hint))
    return False


def _type_hints(cls: type) -> dict[str, Any]:
    try:
        return typing.get_type_hints(cls)
    except Exception:
        # Forward references to names the module never imports at runtime;
        # fall back to whatever is already resolved.
        return {f.name: f.type for f in dataclasses.fields(cls)}


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into a copy of ``base``; mappings merge key-wise."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), Mapping) and isinstance(value, Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


__all__ = ["ConfigError", "from_plain", "merge", "to_plain"]
