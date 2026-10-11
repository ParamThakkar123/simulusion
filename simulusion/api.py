"""User-facing convenience APIs for Simulusion.

The lower-level modules remain available for research workflows, but this module
collects the common discovery and construction paths behind small, predictable
factory functions.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from importlib import import_module
from inspect import signature
from typing import Any, Callable, NamedTuple, cast


class ModelSpec(NamedTuple):
    """Metadata describing a model available through :func:`create_model`."""

    name: str
    import_path: str
    config_path: str | None = None
    description: str = ""
    aliases: tuple[str, ...] = ()


class EnvBackendSpec(NamedTuple):
    """Metadata describing an environment backend available through ``make_env``."""

    name: str
    factory_path: str
    description: str = ""
    aliases: tuple[str, ...] = ()


MODEL_SPECS: dict[str, ModelSpec] = {
    "dreamer": ModelSpec(
        name="dreamer",
        import_path="simulusion.models.dreamer:DreamerAgent",
        config_path="simulusion.configs.dreamer_config:DreamerConfig",
        description="High-level Dreamer agent (alias for dreamer-v1; prefer explicit versioned names).",
        aliases=("dreamer_agent",),
    ),
    "dreamer-v1": ModelSpec(
        name="dreamer-v1",
        import_path="simulusion.models.dreamer_v1:DreamerV1Agent",
        config_path="simulusion.configs.dreamer_config:DreamerConfig",
        description="DreamerV1 world model (normal Gaussian heads, standard KL).",
        aliases=("dreamerv1",),
    ),
    "dreamer-v2": ModelSpec(
        name="dreamer-v2",
        import_path="simulusion.models.dreamer_v2:DreamerV2Agent",
        config_path="simulusion.configs.dreamer_config:DreamerConfig",
        description="DreamerV2 world model (symlog two-hot heads, balanced KL).",
        aliases=("dreamerv2",),
    ),
    "planet": ModelSpec(
        name="planet",
        import_path="simulusion.models.planet:Planet",
        description="PlaNet agent and planner for image-based control.",
        aliases=("pla_net",),
    ),
    "jepa": ModelSpec(
        name="jepa",
        import_path="simulusion.models.jepa_agent:JEPAAgent",
        config_path="simulusion.configs.jepa_config:JEPAConfig",
        description="JEPA self-supervised visual representation trainer.",
        aliases=("ijepa", "i-jepa"),
    ),
    "iris": ModelSpec(
        name="iris",
        import_path="simulusion.models.iris_agent:IRISAgent",
        config_path="simulusion.configs.iris_config:IRISConfig",
        description="IRIS world model and actor-critic module.",
        aliases=("iris_agent",),
    ),
    "genie": ModelSpec(
        name="genie",
        import_path="simulusion.models.genie:create_genie",
        config_path="simulusion.configs.genie_config:GenieConfig",
        description="Genie generative interactive environment model.",
        aliases=("genie_base",),
    ),
    "genie-small": ModelSpec(
        name="genie-small",
        import_path="simulusion.models.genie:create_genie_small",
        config_path="simulusion.configs.genie_config:GenieSmallConfig",
        description="Smaller Genie variant for development and tests.",
        aliases=("genie_small",),
    ),
    "genie-large": ModelSpec(
        name="genie-large",
        import_path="simulusion.models.genie:create_genie_large",
        description="Large Genie variant.",
        aliases=("genie_large",),
    ),
    "diamond": ModelSpec(
        name="diamond",
        import_path="simulusion.training.train_diamond:DiamondAgent",
        config_path="simulusion.configs.diamond_config:DiamondConfig",
        description="DIAMOND diffusion world model agent for Atari-style control.",
        aliases=("diamond_agent",),
    ),
    "dit": ModelSpec(
        name="dit",
        import_path="simulusion.models.diffusion.DiT:create_dit",
        config_path="simulusion.configs.dit_config:DiTConfig",
        description="Diffusion Transformer (DiT) image denoising model.",
        aliases=("diffusion-transformer", "diffusion_transformer"),
    ),
    "modular-rssm": ModelSpec(
        name="modular-rssm",
        import_path="simulusion.models.modular_rssm:create_modular_rssm",
        description="Factory for a modular recurrent state-space model.",
        aliases=("modular_rssm", "rssm"),
    ),
}

ENV_BACKEND_SPECS: dict[str, EnvBackendSpec] = {
    "auto": EnvBackendSpec(
        name="auto",
        factory_path="simulusion.envs:make_env",
        description="Try Simulusion env factories and fall back to Gym.",
        aliases=("default",),
    ),
    "dmc": EnvBackendSpec(
        name="dmc",
        factory_path="simulusion.envs:make_dmc_env",
        description="DeepMind Control Suite tasks (e.g. walker-walk); Dreamer's default.",
        aliases=("dm_control", "deepmind_control"),
    ),
    "gym": EnvBackendSpec(
        name="gym",
        factory_path="simulusion.envs:make_gym_env",
        description="Gym/Gymnasium image environments.",
        aliases=("gymnasium",),
    ),
    "atari": EnvBackendSpec(
        name="atari",
        factory_path="simulusion.envs:make_atari_env",
        description="Atari environments through ALE.",
        aliases=("ale",),
    ),
    "mujoco": EnvBackendSpec(
        name="mujoco",
        factory_path="simulusion.envs:make_mujoco_env",
        description="MuJoCo physics environments.",
        aliases=("mjcf", "native_mujoco"),
    ),
    "robotics": EnvBackendSpec(
        name="robotics",
        factory_path="simulusion.envs:make_robotics_env",
        description="Gymnasium Robotics environments.",
        aliases=("gymnasium_robotics",),
    ),
    "procgen": EnvBackendSpec(
        name="procgen",
        factory_path="simulusion.envs:make_procgen_env",
        description="Procedurally generated benchmark environments.",
        aliases=("coinrun",),
    ),
    "dmlab": EnvBackendSpec(
        name="dmlab",
        factory_path="simulusion.envs:make_dmlab_env",
        description="DeepMind Lab 3D navigation and puzzle tasks.",
        aliases=("deepmind_lab", "deepmindlab"),
    ),
    "brax": EnvBackendSpec(
        name="brax",
        factory_path="simulusion.envs:make_brax_env",
        description="JAX/Brax continuous-control environments.",
        aliases=(),
    ),
    "bsuite": EnvBackendSpec(
        name="bsuite",
        factory_path="simulusion.envs:make_bsuite_env",
        description="Behaviour Suite for Reinforcement Learning (BSuite) environments.",
        aliases=("behavior_suite", "behaviour_suite"),
    ),
    "unity": EnvBackendSpec(
        name="unity",
        factory_path="simulusion.envs:make_unity_mlagents_env",
        description="Unity ML-Agents executables.",
        aliases=("mlagents", "unity_mlagents"),
    ),
    "world-model": EnvBackendSpec(
        name="world-model",
        factory_path="simulusion.envs:make_world_model_env",
        description="Gymnasium-compatible rollouts inside a trained world model.",
        aliases=("world_model", "model", "wm"),
    ),
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _alias_map(
    specs: dict[str, ModelSpec] | dict[str, EnvBackendSpec],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, spec in specs.items():
        aliases[_normalize(canonical)] = canonical
        for alias in spec.aliases:
            aliases[_normalize(alias)] = canonical
    return aliases


def _resolve_model_name(name: str) -> str:
    aliases = _alias_map(MODEL_SPECS)
    try:
        return aliases[_normalize(name)]
    except KeyError:
        pass
    from simulusion.registry import get_registered_model_spec

    registered = get_registered_model_spec(name)
    if registered is not None:
        return registered.name
    available = ", ".join(list_models())
    raise ValueError(f"Unknown model {name!r}. Available models: {available}") from None


def _resolve_backend_name(name: str) -> str:
    aliases = _alias_map(ENV_BACKEND_SPECS)
    try:
        return aliases[_normalize(name)]
    except KeyError:
        pass
    from simulusion.registry import get_registered_env_backend_spec

    registered = get_registered_env_backend_spec(name)
    if registered is not None:
        return registered.name
    available = ", ".join(list_env_backends())
    raise ValueError(
        f"Unknown environment backend {name!r}. Available: {available}"
    ) from None


def _load_object(import_path: str) -> Any:
    module_name, attr = import_path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, attr)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(cast(Any, config))
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and not callable(value)
    }


def _config_fields(config: Any) -> set[str]:
    if config is None:
        return set()
    if isinstance(config, dict):
        return set(config)
    if is_dataclass(config):
        return set(config.__dataclass_fields__)
    return {key for key in vars(config) if not key.startswith("_")}


def _split_config_and_constructor_overrides(
    config: Any, overrides: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_keys = _config_fields(config)
    config_overrides = {
        key: value for key, value in overrides.items() if key in config_keys
    }
    constructor_overrides = {
        key: value for key, value in overrides.items() if key not in config_keys
    }
    return config_overrides, constructor_overrides


def _apply_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return config
    if isinstance(config, dict):
        updated = dict(config)
        updated.update(overrides)
        return updated
    if is_dataclass(config):
        valid = set(config.__dataclass_fields__)
        invalid = sorted(set(overrides) - valid)
        if invalid:
            raise ValueError(f"Invalid config override(s): {', '.join(invalid)}")
        return replace(cast(Any, config), **overrides)
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Invalid config override: {key}")
        setattr(config, key, value)
    return config


def _call_with_supported_kwargs(
    factory: Callable[..., Any], kwargs: dict[str, Any]
) -> Any:
    params = signature(factory).parameters
    accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
    if accepts_kwargs:
        return factory(**kwargs)

    supported = {name for name in params if name != "self"}
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(set(kwargs) - supported)
    if ignored:
        ignored_list = ", ".join(ignored)
        raise ValueError(
            f"Unsupported argument(s) for {factory.__name__}: {ignored_list}"
        )
    return factory(**filtered)


def list_models() -> list[str]:
    """Return canonical model names accepted by :func:`create_model`."""

    from simulusion.registry import list_registered_models

    return sorted(set(MODEL_SPECS) | set(list_registered_models()))


def get_model_spec(name: str) -> ModelSpec:
    """Return metadata for a model name or alias."""

    resolved = _resolve_model_name(name)
    if resolved in MODEL_SPECS:
        return MODEL_SPECS[resolved]
    from simulusion.registry import get_registered_model_spec

    spec = get_registered_model_spec(resolved)
    if spec is not None:
        return spec
    raise ValueError(f"Model {name!r} resolved to {resolved!r} but no spec found.")


def list_env_backends() -> list[str]:
    """Return canonical backend names accepted by :func:`make_env`."""

    from simulusion.registry import list_registered_env_backends

    return sorted(set(ENV_BACKEND_SPECS) | set(list_registered_env_backends()))


def get_env_backend_spec(name: str) -> EnvBackendSpec:
    """Return metadata for an environment backend name or alias."""

    canonical = _resolve_backend_name(name)
    if canonical in ENV_BACKEND_SPECS:
        return ENV_BACKEND_SPECS[canonical]
    from simulusion.registry import get_registered_env_backend_spec

    registered = get_registered_env_backend_spec(canonical)
    assert registered is not None
    return registered


def create_config(model: str, **overrides: Any) -> Any:
    """Create the default config object for ``model`` and apply overrides.

    Examples:
        >>> cfg = create_config("dreamer", env="walker-walk", seed=7)
        >>> cfg.env
        'walker-walk'
    """

    spec = get_model_spec(model)
    if spec.config_path is None:
        if overrides:
            raise ValueError(f"Model {spec.name!r} does not define a config object")
        return None
    config_cls = _load_object(spec.config_path)
    config = config_cls()
    return _apply_overrides(config, overrides)


def create_model(model: str, config: Any | None = None, **overrides: Any) -> Any:
    """Instantiate a model or agent from a simple string name.

    ``config`` is optional for models that define a config class. Keyword
    overrides are applied to the config when possible, otherwise they are passed
    directly to the underlying constructor/factory.

    Examples:
        >>> agent = create_model(
        ...     "dreamer", env="Pendulum-v1", env_backend="gym", total_steps=1000
        ... )
        >>> genie = create_model("genie-small", image_size=32)
    """

    spec = get_model_spec(model)
    factory = _load_object(spec.import_path)

    if spec.config_path is not None:
        if config is None:
            config = create_config(spec.name)
        config_overrides, constructor_overrides = (
            _split_config_and_constructor_overrides(config, overrides)
        )
        config = _apply_overrides(config, config_overrides)
        if spec.name in {"genie", "genie-small", "genie-large"}:
            # Build Genie directly from the full config mapping: filtering the
            # config through the factory's signature dropped the tokenizer and
            # latent-action widths/depths, which then fell back to defaults.
            from simulusion.models.genie import Genie, genie_kwargs_from_config

            kwargs = genie_kwargs_from_config(config)
            kwargs.update(constructor_overrides)
            return _call_with_supported_kwargs(Genie, kwargs)
        return factory(config, **constructor_overrides)

    kwargs = _config_to_dict(config)
    kwargs.update(overrides)
    return _call_with_supported_kwargs(factory, kwargs)


def make_env(env_id: str, backend: str = "auto", **kwargs: Any) -> Any:
    """Create an environment with a consistent Simulusion entry point.

    Args:
        env_id: Environment id, XML path, Unity executable path, or backend-specific id.
        backend: One of :func:`list_env_backends`; ``"auto"`` tries Simulusion's
            compatibility helper.
        **kwargs: Backend-specific options.
    """

    spec = get_env_backend_spec(backend)
    factory = _load_object(spec.factory_path)
    if spec.name == "auto":
        kwargs.setdefault("backend", backend)
    return factory(env_id, **kwargs)


def list_envs(model: str | None = None) -> list[str] | dict[str, list[str]]:
    """List known environment ids, optionally filtered by model family."""

    from simulusion.catalog import ENVIRONMENTS_BY_MODEL

    if model is None:
        return {key: list(value) for key, value in ENVIRONMENTS_BY_MODEL.items()}
    canonical = _normalize(model).replace("-", "")
    try:
        return list(ENVIRONMENTS_BY_MODEL[canonical])
    except KeyError as exc:
        available = ", ".join(sorted(ENVIRONMENTS_BY_MODEL))
        raise ValueError(
            f"Unknown model environment catalog {model!r}: {available}"
        ) from exc


__all__ = [
    "EnvBackendSpec",
    "ModelSpec",
    "MODEL_SPECS",
    "ENV_BACKEND_SPECS",
    "create_config",
    "create_model",
    "get_env_backend_spec",
    "get_model_spec",
    "list_env_backends",
    "list_envs",
    "list_models",
    "make_env",
]
