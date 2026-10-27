# Exporting Models for Deployment

Simulusion provides a unified export system that converts trained models and agents
into deployable formats — `torch.export` programs, AOTInductor packages, ONNX,
TensorRT and (legacy) TorchScript — without requiring each model to implement
its own export logic. For making inference fast, choosing what to export for
a recurrent world model, and packaging it with its config, see
{doc}`deployment_guide`.

```{contents} Contents
:depth: 3
```

## Overview

The export system is built around three levels of API:

| Level | Function / Method | When to use |
|---|---|---|
| **Module method** | `module.export(path, format, ...)` | Every `nn.Module` class exported from the top-level `simulusion` namespace (`Genie`, `VisionTransformer`, `RSSM`, ...) |
| **Agent method** | `agent.export(path, format, ...)` | High-level agents (`DreamerAgent`, `JEPAAgent`, `IRISAgent`) that inherit `ExportableAgentMixin` |
| **Standalone** | `export_any(obj, path, ...)` / `export_model(module, path, ...)` | When you need explicit control over which submodule is exported or want to bypass automatic target resolution |

### Supported formats

| Format | Extension | Use case |
|---|---|---|
| `"exported_program"` (aliases: `"export"`, `"pt2"`, `"ep"`) | `.pt2` | **Recommended.** `torch.export` graph; loadable without the model's source, and the input to AOTInductor, ExecuTorch and TensorRT |
| `"aoti"` (aliases: `"aotinductor"`) | `.pt2` | Ahead-of-time compiled package, runnable from Python or C++ (LibTorch) without the model's Python code; needs a C++ toolchain at export time |
| `"onnx"` | `.onnx` | Cross-platform inference, mobile, edge devices, TensorRT conversion |
| `"tensorrt"` (aliases: `"trt"`) | `.ep` | NVIDIA GPU-optimized inference (requires `torch_tensorrt`). Compiles through `ir="dynamo"` by default; pass `ir="ts"` for the legacy TorchScript path |
| `"torchscript"` (aliases: `"jit"`, `"ts"`, `"pt"`, `"script"`) | `.pt` | Legacy. TorchScript is in maintenance mode upstream; prefer `"exported_program"` or `"aoti"` |

`torch.export` options pass straight through, for example
`dynamic_shapes={"x": {0: torch.export.Dim("batch")}}` to keep the batch size
symbolic.

### Loading and verifying artifacts

```python
from simulusion import export_model, load_exported, verify_export

path = export_model(module, "step.pt2", format="exported_program", example_inputs=x)
max_err = verify_export(module, path, x)  # raises AssertionError on mismatch
runner = load_exported(path)  # format inferred from the suffix
runner = load_exported("step_aoti.pt2", format="aoti")
```

`load_exported` returns a plain callable for every format. ONNX artifacts run
through ONNX Runtime when it is installed.

## Quick start

### Exporting any `nn.Module`

`simulusion.export_model(module, path, ...)` exports any `torch.nn.Module`,
including your own. Simulusion does not add methods to `torch.nn.Module`; its
public model classes get `.export()` from `ExportableAgentMixin`, which you can
also inherit in your own classes (see [Custom agents](#custom-agents)).

```python
import torch
from simulusion import export_model


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)

    def forward(self, x):
        return self.linear(x)


model = MyModel()
model.eval()

# torch.export program (recommended)
export_model(
    model, "model.pt2", format="exported_program", example_inputs=torch.zeros(1, 64)
)

# ONNX (requires example_inputs)
export_model(model, "model.onnx", format="onnx", example_inputs=torch.zeros(1, 64))
```

Simulusion's own models (for example `simulusion.create_model("genie-small")`) can
call the same thing as a method: `genie.export("genie.pt2", format="exported_program", example_inputs=video)`.

### Exporting a trained agent

High-level agents support the same `.export()` method directly:

```python
import simulusion

agent = simulusion.create_model("dreamer", env="walker-walk", total_steps=1000)
# ... train the agent ...

# Export the policy to ONNX
agent.export("policy.onnx", format="onnx")

# Export the RSSM world model to TorchScript
agent.export("rssm.pt", format="torchscript", target="rssm")

# Export the observation encoder to ONNX
agent.export("encoder.onnx", format="onnx", target="obs_encoder")
```

The system automatically resolves which submodule to export and infers the
correct example inputs for each agent type.

## Target resolution

When you call `.export()`, the system needs to decide which `nn.Module` to
serialize. It uses a priority-based resolution strategy:

### 1. Agent-specific defaults

Each agent type has a preferred default target:

| Agent | Default target | Example inputs |
|---|---|---|
| `DreamerAgent` | Policy head (actor) | `[batch, stoch_size + deter_size]` |
| `IRISAgent` | Actor-critic head | `[batch, 1, channels, h, w]` |
| `JEPAAgent` | Vision Transformer encoder | `[batch, 3, crop_size, crop_size]` |

### 2. Explicit `target=` parameter

Override the default by naming a specific submodule:

```python
# Export by attribute path
agent.export("reward.pt", format="torchscript", target="dreamer.reward_model")

# If the attribute name is unique, the short name works
agent.export("value.pt", format="torchscript", target="value_model")
```

If the short name matches multiple modules, the system raises an error and
lists the available fully qualified paths.

### 3. Single-module fallback

If the object is itself an `nn.Module`, it is exported directly. If it
contains exactly one `nn.Module` attribute, that attribute is exported.
If it contains multiple modules, the system picks the first match from this
priority list: `actor`, `policy`, `actor_critic`, `rssm`, `model`,
`world_model`, `encoder`.

## Format-specific details

### ONNX

```python
agent.export(
    "policy.onnx",
    format="onnx",
    input_names=["latent"],
    output_names=["action"],
    dynamic_axes={"latent": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17,  # default
)
```

See the [PyTorch ONNX export docs](https://pytorch.org/docs/stable/onnx.html)
for all supported keyword arguments.

### TorchScript

Two modes, controlled by whether `example_inputs` is provided:

| Mode | When to use | Limitation |
|---|---|---|
| **Tracing** (with `example_inputs`) | Fast export of a fixed forward graph | May not handle dynamic control flow |
| **Scripting** (without `example_inputs`) | Full module graph with control flow | May fail on unsupported Python constructs |

```python
# Trace (requires example_inputs)
agent.export("traced.pt", format="torchscript", example_inputs=torch.zeros(1, 230))

# Script (no example_inputs needed)
agent.export("scripted.pt", format="torchscript")
```

### TensorRT

Requires the optional `torch_tensorrt` package:

```sh
pip install torch_tensorrt
```

```python
agent.export(
    "policy.trt",
    format="tensorrt",
    example_inputs=torch.zeros(1, 230, device="cuda"),
    enabled_precisions={torch.float16},  # FP16 inference
)
```

## Custom agents

Inherit `ExportableAgentMixin` to give a custom agent (or your own
`nn.Module`) the `.export()` method:

```python
from simulusion import ExportableAgentMixin


class MyAgent(ExportableAgentMixin):
    def __init__(self):
        self.policy = torch.nn.Linear(64, 5)
        self.encoder = torch.nn.Linear(1024, 64)
```

The mixin will automatically discover `self.policy` and prefer it as the
default target. Pass `target="encoder"` to export a different submodule.

### Custom example input inference

If the auto-inferred example inputs are wrong for your agent, pass them
explicitly:

```python
agent.export(
    "policy.onnx",
    format="onnx",
    example_inputs=torch.zeros(1, 128),  # your custom shape
)
```

Or add inference support by implementing a matching pattern in
`_infer_example_inputs` in `simulusion/export.py`.

## Low-level API

For scripting or batch export, use the standalone functions directly:

```python
from simulusion import export_any, export_model

# export_any resolves the target module from any object
export_any(agent, "policy.onnx", format="onnx")

# export_model exports a raw nn.Module
export_model(agent.policy, "policy.pt", format="torchscript")
```

## Common pitfalls

1. **Missing `.eval()`**: Export always sets the module to eval mode before
   tracing and restores the original mode afterwards. Call `.eval()` manually
   if you are inspecting the exported graph afterwards.

2. **Dynamic control flow with ONNX**: ONNX requires tracing. If your module
   has `if` statements or loops that depend on tensor values, use TorchScript
   instead.

3. **CUDA tensors for TensorRT**: TensorRT export requires example inputs on
   the same device as the module. Pass CUDA tensors as example inputs.

4. **Multiple matches for short names**: If you see `"matched multiple
   modules"`, use the fully qualified path, e.g. `target="dreamer.actor"`
   instead of `target="actor"`.

## See Also

- {doc}`inference_guide` — running exported models in production
- {doc}`public_api` — `export_any` and `export_model` factory functions
