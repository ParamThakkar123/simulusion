"""Experiment configuration helpers backed by OmegaConf and Hydra.

The functions in this module compose library defaults, YAML files, and Hydra
``key=value`` dot-list overrides into plain Python containers or existing config
objects.  OmegaConf and Hydra are used throughout for parsing and composition.
If OmegaConf is not installed the same public API falls back to the PyYAML
parser that Simulusion already depends on, so importing training modules stays
cheap.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

_omegaconf_module = (
    importlib.import_module("omegaconf")
    if importlib.util.find_spec("omegaconf") is not None
    else None
)
OmegaConf: Any | None = (
    getattr(_omegaconf_module, "OmegaConf") if _omegaconf_module is not None else None
)

T = TypeVar("T")


class ExperimentConfigError(ValueError):
    """Raised when an experiment config or override cannot be composed."""


@dataclasses.dataclass(frozen=True)
class ExperimentArgs:
    """Parsed experiment CLI arguments shared by training entrypoints."""

    config: str | None
    overrides: tuple[str, ...]
    print_config: bool = False


def public_config_dict(config: Any) -> dict[str, Any]:
    """Convert dataclasses, mappings, and simple config objects to dictionaries."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(cast(Any, config))
    if isinstance(config, Mapping):
        return {str(key): _to_plain(value) for key, value in config.items()}
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return _to_plain(config.to_dict())

    values: dict[str, Any] = {}
    for key in dir(config):
        if key.startswith("_"):
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        values[key] = _to_plain(value)
    return values


def load_experiment_config(
    defaults: Any,
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Compose defaults with an optional YAML file and dot-list overrides.

    Args:
        defaults: Dataclass, mapping, or simple object used as base values.
        config_path: Optional YAML config file. Values in this file override
            defaults.
        overrides: Hydra/OmegaConf-style ``key=value`` overrides. Nested keys
            use dot notation, e.g. ``optimization.lr=3e-4``.

    Returns:
        A plain Python ``dict`` safe to pass to trainers, loggers, or tests.
    """
    base = public_config_dict(defaults)
    if OmegaConf is not None:
        cfg = OmegaConf.create(base)
        if config_path is not None:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(str(config_path)))
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        return _to_plain(OmegaConf.to_container(cfg, resolve=True))

    composed = dict(base)
    if config_path is not None:
        file_values = _load_yaml(config_path)
        _deep_merge(composed, file_values)
    if overrides:
        _deep_merge(composed, dotlist_to_dict(overrides))
    return composed


def instantiate_dataclass(
    config_cls: type[T],
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> T:
    """Instantiate a dataclass config after YAML and dot-list composition."""
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError("instantiate_dataclass expects a dataclass type")
    values = load_experiment_config(config_cls(), config_path, overrides)
    field_names = {field.name for field in dataclasses.fields(config_cls)}
    unknown = sorted(set(values).difference(field_names))
    if unknown:
        joined = ", ".join(unknown)
        raise ExperimentConfigError(f"Unknown {config_cls.__name__} field(s): {joined}")
    return config_cls(**values)


def update_config_object(
    config: T, values: Mapping[str, Any], *, strict: bool = True
) -> T:
    """Update an existing config object's public attributes from a mapping."""
    for key, value in values.items():
        if strict and not hasattr(config, key):
            raise ExperimentConfigError(
                f"Unknown {config.__class__.__name__} field: {key}"
            )
        setattr(config, key, value)
    return config


def parse_experiment_args(
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> ExperimentArgs:
    """Parse shared ``--config``/``--print-config`` plus Hydra dot-list overrides.

    Uses OmegaConf-style ``key=value`` syntax for overrides (no ``--`` prefix).
    """
    raw = list(argv) if argv is not None else sys.argv[1:]

    config_path: str | None = None
    print_config = False
    overrides: list[str] = []

    i = 0
    while i < len(raw):
        arg = raw[i]
        if arg in ("--config", "-c") and i + 1 < len(raw):
            config_path = raw[i + 1]
            i += 2
        elif arg == "--print-config":
            print_config = True
            i += 1
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
        else:
            overrides.append(arg)
            i += 1

    return ExperimentArgs(
        config=config_path,
        overrides=tuple(overrides),
        print_config=print_config,
    )


def dump_config(config: Mapping[str, Any]) -> str:
    """Serialize a config mapping as stable YAML for logs or --print-config."""
    if OmegaConf is not None:
        return OmegaConf.to_yaml(OmegaConf.create(dict(config)), resolve=True)
    return yaml.safe_dump(dict(config), sort_keys=False)


def dotlist_to_dict(overrides: Sequence[str]) -> dict[str, Any]:
    """Parse ``key=value`` dot-list overrides without importing OmegaConf."""
    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ExperimentConfigError(
                f"Invalid override '{item}'. Expected key=value syntax."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ExperimentConfigError(f"Invalid override '{item}'. Key is empty.")
        value = _parse_override_value(raw_value)
        _assign_nested(result, key.split("."), value)
    return result


def _parse_override_value(raw_value: str) -> Any:
    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ExperimentConfigError(f"Config file must contain a mapping: {path}")
    return _to_plain(loaded)


def _assign_nested(
    target: MutableMapping[str, Any], keys: list[str], value: Any
) -> None:
    current: MutableMapping[str, Any] = target
    for key in keys[:-1]:
        next_value = current.setdefault(key, {})
        if not isinstance(next_value, MutableMapping):
            raise ExperimentConfigError(f"Cannot override nested key below '{key}'")
        current = next_value
    current[keys[-1]] = value


def _deep_merge(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = _to_plain(value)


def _to_plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(cast(Any, value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value
