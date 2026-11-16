# Plugin Registry

Simulusion's plugin registry lets you register custom world models and environment
backends so they are discoverable through the standard factory API alongside
built-in models.

```{contents} Contents
```

## Why use the registry?

- **Discovery**: `create_model("your-model")` and `list_models()` work for
  registered models — no code change needed in experiment scripts.
- **Interop**: Registered models appear in `get_model_spec()` and support the
  same config override flow as Dreamer, JEPA, IRIS, etc.
- **Aliases**: Attach short or alternative names to your model.

## Registering a world model

Use `register_world_model` as a decorator on your model class:

```python
from simulusion import register_world_model


@register_world_model(
    "my-agent",
    import_path="my_package.models:MyAgent",
    config_path="my_package.configs:MyConfig",
    description="My custom world model agent",
    aliases=("my_agent", "custom"),
)
class MyAgent:
    def __init__(self, config):
        self.config = config
```

After registration:

```python
import simulusion

# Standard factory API works:
cfg = simulusion.create_config("my-agent", learning_rate=1e-4)
agent = simulusion.create_model("my-agent", cfg)

# Discovery:
print(simulusion.list_models())  # includes "my-agent"
spec = simulusion.get_model_spec("my-agent")
print(spec.description)  # "My custom world model agent"
```

### Without a config class

If your model does not need a config object, omit `config_path`:

```python
@register_world_model("simple-model", import_path="my_package.models:create_simple")
class SimpleModel: ...
```

`create_config("simple-model")` returns `None`; keyword overrides are forwarded
directly to the factory:

```python
model = simulusion.create_model("simple-model", hidden_size=128)
```

### Without a class (direct registration)

Passing a Python import-path string skips the decorator pattern entirely:

```python
from simulusion import register_world_model

register_world_model(
    "my-agent",
    import_path="my_package.models:create_my_agent",
    config_path="my_package.configs:MyAgentConfig",
)
```

This is useful when the model is built by a factory function instead of a class
constructor.

### Aliases

Aliases let users refer to your model by multiple names:

```python
@register_world_model(
    "my-agent",
    import_path="...",
    aliases=("my_agent", "custom", "ma"),
)
class MyAgent: ...
```

All of the following resolve to the same model:

```python
simulusion.create_model("my-agent")
simulusion.create_model("my_agent")
simulusion.create_model("custom")
simulusion.create_model("ma")
```

### Overriding an existing model

Pass `override=True` to replace a previously registered model (including
built-in models — use with caution):

```python
@register_world_model(
    "dreamer",
    import_path="my_package.models:MyDreamer",
    override=True,
)
class MyDreamer: ...
```

This shadows the built-in `dreamer` spec. All existing scripts calling
`create_model("dreamer")` will now use your replacement.

## Environment backends

The same registry pattern works for environment backends:

```python
from simulusion import register_env_backend

register_env_backend(
    "custom-env",
    factory_path="my_package.envs:make_custom_env",
    description="My custom environment backend",
    aliases=("ce",),
)
```

Registered backends appear in `list_env_backends()` and work with
`make_env(backend="custom-env")`.

## Discovering registered models

```python
import simulusion

# All models (built-in + registered):
all_models = simulusion.list_models()

# Only externally registered models:
ext_models = simulusion.list_registered_models()

# Spec lookup:
spec = simulusion.get_model_spec("my-agent")
print(spec.name, spec.import_path, spec.config_path, spec.description)

# Remove a registration:
simulusion.deregister_world_model("my-agent")
```

## Deprecating a model

Use the `deprecated_class` decorator to mark an older model as deprecated and
point users to its replacement:

```python
from simulusion import deprecated_class


@deprecated_class(version="0.6.0", alternative="MyNewAgent")
@register_world_model("old-agent", import_path="my_package.models:OldAgent")
class OldAgent: ...
```

Instantiating `OldAgent` (or using `create_model("old-agent")`) emits a
`DeprecationWarning`:

```
DeprecationWarning: 'OldAgent' is deprecated since v0.6.0 — Use MyNewAgent instead
```

Lower-level `deprecated_function` and generic `deprecated` decorators are also
available from `simulusion`:

```python
from simulusion import deprecated, deprecated_function
```

## Complete example

```python
"""my_package/models.py"""

import simulusion
from simulusion import register_world_model, deprecated_class


class MyConfig:
    def __init__(self):
        self.learning_rate = 1e-4
        self.hidden_size = 256


@register_world_model(
    "research-model",
    import_path="my_package.models:ResearchModel",
    config_path="my_package.models:MyConfig",
    description="Research world model with config overrides.",
    aliases=("research", "rm"),
)
class ResearchModel:
    def __init__(self, config):
        self.config = config

    def forward(self, obs): ...
```

Usage in any experiment script:

```python
import simulusion

model = simulusion.create_model(
    "research-model",
    learning_rate=3e-4,  # overrides the default in MyConfig
)
```

## See Also

- {doc}`public_api` — factory helpers (`create_model`, `create_config`, ...)
- {doc}`api_reference` — full list of built-in models and their specs
- {doc}`configs_reference` — configuration classes and serialization
