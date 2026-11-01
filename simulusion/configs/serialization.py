"""Shared serialization helpers for Simulusion config objects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

ConfigT = TypeVar("ConfigT", bound="SerializableConfigMixin")


def make_yaml_safe(value: Any) -> Any:
    """Convert common Python containers into values accepted by safe YAML."""

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return make_yaml_safe(asdict(value))
    if isinstance(value, tuple):
        return [make_yaml_safe(item) for item in value]
    if isinstance(value, list):
        return [make_yaml_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_yaml_safe(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Config field value {value!r} of type {type(value).__name__} "
        "is not YAML-serializable."
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    """Return a YAML/JSON-serializable dictionary of config fields."""

    if is_dataclass(config) and not isinstance(config, type):
        # Fields marked ``metadata={"serialize": False}`` hold runtime objects
        # (for example a live environment) that have no YAML form. They are
        # skipped here rather than via ``asdict``, which would deep-copy them.
        raw = {
            field.name: getattr(config, field.name)
            for field in fields(config)
            if field.metadata.get("serialize", True)
        }
    else:
        raw = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_") and not callable(value)
        }
    return {key: make_yaml_safe(deepcopy(value)) for key, value in raw.items()}


def update_config_from_dict(config: Any, values: dict[str, Any]) -> Any:
    """Apply flat field values to an existing config instance with validation."""

    valid = _config_field_names(config)
    for key, value in values.items():
        if key not in valid:
            raise ValueError(f"Invalid {type(config).__name__} field: {key}")
        current = getattr(config, key, None)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(config, key, value)
    return config


def _config_field_names(config_or_cls: Any) -> set[str]:
    if is_dataclass(config_or_cls):
        return {field.name for field in fields(config_or_cls)}
    return {
        key
        for key, value in vars(config_or_cls).items()
        if not key.startswith("_") and not callable(value)
    }


def read_yaml(path_or_yaml: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from a file path or a YAML string."""

    if isinstance(path_or_yaml, Path):
        yaml_text = path_or_yaml.read_text(encoding="utf-8")
    elif "\n" not in path_or_yaml and len(path_or_yaml) < 4096:
        path_candidate = Path(path_or_yaml)
        yaml_text = (
            path_candidate.read_text(encoding="utf-8")
            if path_candidate.exists()
            else path_or_yaml
        )
    else:
        yaml_text = path_or_yaml
    values = yaml.safe_load(yaml_text) or {}
    if not isinstance(values, dict):
        raise ValueError("Config YAML must contain a mapping.")
    return values


class SerializableConfigMixin:
    """Mixin adding dict/YAML serialization to config containers."""

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @classmethod
    def from_dict(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
        if not isinstance(values, dict):
            raise ValueError(f"{cls.__name__}.from_dict expects a mapping.")
        if is_dataclass(cls):
            valid = _config_field_names(cls)
            invalid = sorted(set(values) - valid)
            if invalid:
                msg = (
                    f"Invalid {cls.__name__} field"
                    f"{'s' if len(invalid) > 1 else ''}: {', '.join(invalid)}"
                )
                raise ValueError(msg)
            config = cls()
            return update_config_from_dict(config, values)
        config = cls()
        return update_config_from_dict(config, values)

    def to_yaml(self, path: str | Path | None = None) -> str:
        yaml_text = yaml.safe_dump(self.to_dict(), sort_keys=True)
        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(yaml_text, encoding="utf-8")
        return yaml_text

    @classmethod
    def from_yaml(cls: type[ConfigT], path_or_yaml: str | Path) -> ConfigT:
        return cls.from_dict(read_yaml(path_or_yaml))

    def save_yaml(self, path: str | Path) -> None:
        self.to_yaml(path)
