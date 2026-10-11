from __future__ import annotations

from typing import Callable

from simulusion.api import EnvBackendSpec, ModelSpec

_WORLD_MODEL_REGISTRY: dict[str, ModelSpec] = {}
_WORLD_MODEL_ALIASES: dict[str, str] = {}
_ENV_BACKEND_REGISTRY: dict[str, EnvBackendSpec] = {}
_ENV_BACKEND_ALIASES: dict[str, str] = {}


def register_world_model(
    name: str,
    *,
    import_path: str,
    config_path: str | None = None,
    description: str = "",
    aliases: tuple[str, ...] = (),
    override: bool = False,
) -> Callable[[type], type] | None:
    """Register a world model architecture in the Simulusion plugin registry.

    Can be used as a decorator or called directly::

        @register_world_model("my-model", import_path="my_pkg.model:MyModel")
        class MyModel:
            ...

        # or without a class:
        register_world_model("other-model", import_path="other:factory")

    Once registered, the model is available via ``create_model("my-model")``
    and appears in ``list_models()``.
    """
    canonical = name.strip().lower().replace("_", "-")

    if canonical in _WORLD_MODEL_REGISTRY and not override:
        raise ValueError(
            f"Model {canonical!r} is already registered. "
            "Use override=True to replace it."
        )

    spec = ModelSpec(
        name=canonical,
        import_path=import_path,
        config_path=config_path,
        description=description,
        aliases=aliases,
    )
    _WORLD_MODEL_REGISTRY[canonical] = spec
    for alias in aliases:
        _WORLD_MODEL_ALIASES[alias.strip().lower().replace("_", "-")] = canonical

    def _noop(cls: type) -> type:
        return cls

    return _noop


def deregister_world_model(name: str) -> None:
    """Remove a previously registered model from the plugin registry."""
    canonical = name.strip().lower().replace("_", "-")
    spec = _WORLD_MODEL_REGISTRY.pop(canonical, None)
    if spec is None:
        raise KeyError(f"No registered model named {canonical!r}")
    for alias in spec.aliases:
        _WORLD_MODEL_ALIASES.pop(alias.strip().lower().replace("_", "-"), None)


def get_registered_model_spec(name: str) -> ModelSpec | None:
    """Look up a registered (non-built-in) model spec by name or alias."""
    canonical = name.strip().lower().replace("_", "-")
    if canonical in _WORLD_MODEL_REGISTRY:
        return _WORLD_MODEL_REGISTRY[canonical]
    resolved = _WORLD_MODEL_ALIASES.get(canonical)
    if resolved is not None:
        return _WORLD_MODEL_REGISTRY.get(resolved)
    return None


def list_registered_models() -> list[str]:
    """Return names of all externally registered world models."""
    return sorted(_WORLD_MODEL_REGISTRY)


def register_env_backend(
    name: str,
    *,
    factory_path: str,
    description: str = "",
    aliases: tuple[str, ...] = (),
    override: bool = False,
) -> None:
    """Register a custom environment backend."""
    canonical = name.strip().lower().replace("_", "-")

    if canonical in _ENV_BACKEND_REGISTRY and not override:
        raise ValueError(
            f"Env backend {canonical!r} is already registered. "
            "Use override=True to replace it."
        )

    spec = EnvBackendSpec(
        name=canonical,
        factory_path=factory_path,
        description=description,
        aliases=aliases,
    )
    _ENV_BACKEND_REGISTRY[canonical] = spec
    for alias in aliases:
        _ENV_BACKEND_ALIASES[alias.strip().lower().replace("_", "-")] = canonical


def deregister_env_backend(name: str) -> None:
    """Remove a previously registered environment backend."""
    canonical = name.strip().lower().replace("_", "-")
    spec = _ENV_BACKEND_REGISTRY.pop(canonical, None)
    if spec is None:
        raise KeyError(f"No registered env backend named {canonical!r}")
    for alias in spec.aliases:
        _ENV_BACKEND_ALIASES.pop(alias.strip().lower().replace("_", "-"), None)


def get_registered_env_backend_spec(name: str) -> EnvBackendSpec | None:
    """Look up a registered env backend spec by name or alias."""
    canonical = name.strip().lower().replace("_", "-")
    if canonical in _ENV_BACKEND_REGISTRY:
        return _ENV_BACKEND_REGISTRY[canonical]
    resolved = _ENV_BACKEND_ALIASES.get(canonical)
    if resolved is not None:
        return _ENV_BACKEND_REGISTRY.get(resolved)
    return None


def list_registered_env_backends() -> list[str]:
    """Return names of all externally registered env backends."""
    return sorted(_ENV_BACKEND_REGISTRY)


__all__ = [
    "register_world_model",
    "deregister_world_model",
    "get_registered_model_spec",
    "list_registered_models",
    "register_env_backend",
    "deregister_env_backend",
    "get_registered_env_backend_spec",
    "list_registered_env_backends",
]
