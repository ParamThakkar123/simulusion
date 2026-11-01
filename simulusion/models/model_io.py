"""Shared model UX helpers for config-based construction and checkpoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch


class ConfigProtocol(Protocol):
    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Any: ...

    @classmethod
    def from_yaml(cls, path_or_yaml: str | Path) -> Any: ...


ConfigT = TypeVar("ConfigT", bound=ConfigProtocol)


def apply_config_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Invalid {type(config).__name__} override: {key}")
        setattr(config, key, value)
    return config


def coerce_config(config_cls: type[ConfigT], config: Any | None) -> ConfigT:
    if config is None:
        return config_cls()
    if isinstance(config, config_cls):
        return config
    if isinstance(config, dict):
        return config_cls.from_dict(config)
    if isinstance(config, (str, Path)):
        return config_cls.from_yaml(config)
    raise TypeError(
        f"config must be a {config_cls.__name__}, dict, YAML path/string, "
        f"or None; got {type(config).__name__}."
    )


def save_config_next_to_checkpoint(config: Any, checkpoint_path: str | Path) -> None:
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(checkpoint.parent / "config.yaml")


def find_local_pretrained_file(path: Path, candidates: tuple[str, ...]) -> Path | None:
    if path.is_file():
        return path
    if path.is_dir():
        for name in candidates:
            candidate = path / name
            if candidate.exists():
                return candidate
        for pattern in ("*.pt", "*.pth", "*.bin"):
            matches = sorted(path.glob(pattern))
            if matches:
                return matches[0]
    return None


def resolve_pretrained_file(
    pretrained_model_name_or_path: str | Path,
    candidates: tuple[str, ...],
    *,
    repo_type: str | None = None,
    revision: str | None = None,
) -> Path | None:
    local_path = Path(pretrained_model_name_or_path)
    local_file = find_local_pretrained_file(local_path, candidates)
    if local_file is not None:
        return local_file

    if importlib.util.find_spec("huggingface_hub") is None:
        return None

    from huggingface_hub import hf_hub_download

    repo_id = str(pretrained_model_name_or_path)
    for filename in candidates:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
            )
        except Exception:
            continue
        return Path(downloaded)
    return None


def parameter_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        param.numel()
        for param in module.parameters()
        if not trainable_only or param.requires_grad
    )


def module_summary(modules: dict[str, torch.nn.Module]) -> dict[str, Any]:
    module_params = {
        name: parameter_count(module, trainable_only=False)
        for name, module in modules.items()
    }
    trainable_params = {
        name: parameter_count(module, trainable_only=True)
        for name, module in modules.items()
    }
    return {
        "total_parameters": sum(module_params.values()),
        "trainable_parameters": sum(trainable_params.values()),
        "modules": module_params,
        "trainable_modules": trainable_params,
    }
