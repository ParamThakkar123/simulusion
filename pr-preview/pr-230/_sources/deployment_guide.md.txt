# Efficient Inference and Deployment

This guide covers running trained world models fast, measuring what each
optimization costs, and shipping them as self-contained artifacts. Everything
here lives in `simulusion.inference` and `simulusion.export`, and every optimization
is **opt-in**: with default settings, results are identical to calling the
model directly.

```{contents} Contents
:depth: 2
```

## Why world models need a different recipe

Most inference guides optimize a single large forward pass. World-model
inference is a **loop of small, stateful steps**: an RSSM transition per
timestep, one token at a time in IRIS, one frame at a time in Genie, several
denoising passes per frame in DIAMOND. That has three consequences, and the
tools in this guide follow from them:

1. **Each step is launch-bound, not compute-bound.** A Dreamer step runs a few
   hundred tiny kernels. On a GPU the time goes to Python and kernel-launch
   overhead, so `torch.compile` with CUDA graphs is usually the biggest single
   win: it replays the whole step as one launch.
2. **State must be explicit.** Batching environments, compiling and exporting
   all need the step to be a pure function of tensors, with sampling noise as
   an input. The {ref}`stepper interface <steppers>` gives every model that
   shape.
3. **Errors compound.** Reduced precision or quantization that looks harmless
   on one step can drift over a 50-step imagined rollout, because the state is
   fed back into itself. Check every optimization with
   {ref}`rollout drift <drift>`, not only single-step error.

## Quick reference

| Goal | Tool |
|---|---|
| Faster in-process inference | `optimize_for_inference(module, precision="auto", compile=True)` |
| One precision policy | `inference_context(device, "bf16")`, `resolve_precision` |
| Uniform step loop | `make_stepper(agent)` → `init_state` / `observe` / `act` / `imagine` |
| Export a pure step | `DreamerStepper(agent).step_module()` |
| Measure latency | `benchmark_step(fn, *args)` → p50/p90/p99, steps/s, peak memory |
| Measure fidelity | `rollout_drift(reference, candidate, state, inputs)` |
| Reduce weight memory | `quantize_weights(module)` (int8 weight-only) |
| Genie frame generation | `genie.generate(..., use_cache=True)` |
| Portable artifact | `export_model(module, path, format="exported_program")` |
| Python-free / C++ | `format="aoti"` (AOTInductor) |
| Ship everything together | `save_bundle(dir, module, example_inputs=...)` |
| Inspect / time a bundle | `simulusion deploy inspect DIR`, `simulusion deploy bench DIR` |

## In-process optimization

`optimize_for_inference` wraps any module with eval mode, no gradients, a
precision policy and optionally `torch.compile`:

```python
import torch
from simulusion.inference import optimize_for_inference

policy = optimize_for_inference(
    agent.dreamer.actor,
    precision="auto",  # bf16 on Ampere+ GPUs, fp16 on older GPUs, fp32 on CPU
    compile=True,  # torch.compile, mode="reduce-overhead" (CUDA graphs)
)
action = policy(features, deter=True)
```

| Option | Default | Effect |
|---|---|---|
| `precision` | `"fp32"` | `"bf16"`, `"fp16"` or `"auto"`. Runs under autocast, so softmax, norms and reductions stay in fp32. |
| `compile` | `False` | Wraps the forward in `torch.compile`. If the backend is missing (for example no Triton on a Windows CUDA build), it warns once and runs eagerly. |
| `compile_mode` | `"reduce-overhead"` | Uses CUDA graphs, which suits small fixed-shape steps. Use `"max-autotune"` for large batched steps. |
| `channels_last` | `False` | NHWC layout for conv-heavy encoders and decoders. |
| `cast_weights` | `False` | Stores weights in the reduced dtype too, halving their memory. |
| `output_dtype` | `torch.float32` | Casts outputs back so the wrapper is a drop-in replacement. `None` keeps them as computed. |
| `clone_outputs` | auto | CUDA-graph replays reuse output buffers. The wrapper clones outputs when compiling with CUDA graphs, so a value kept across steps is not overwritten. |

The wrapped module keeps its identity and `state_dict` keys. The lower-level
`simulusion.maybe_compile(fn_or_module, enabled=True)` does the same for a single
function or module. It swaps the module's `forward` rather than returning an
`OptimizedModule`, so checkpoints keep loading into uncompiled models.

:::{note}
`torch.compile` compiles lazily, so the first call is slow: seconds to minutes
depending on the model. It also recompiles when an input shape changes. Keep
batch sizes fixed in a serving loop, and warm up before measuring.
:::

### Precision

`simulusion.inference.inference_context(device, precision)` is the single
precision policy used by every tool in this guide. It enters
`torch.inference_mode()` plus autocast. At fp32 it enters no autocast region at
all, so it is bit-identical to plain `inference_mode`.

- **bf16** has the same exponent range as fp32 and needs no loss scaling. It is
  the recommended reduced precision on GPUs that support it.
- **fp16** is supported on CUDA only.
- On CPU, `"auto"` stays at fp32, because reduced precision is usually slower
  without AMX/AVX512-BF16.

(steppers)=
## A uniform step interface

Each model family names its steps differently. A *stepper* gives them one
shape, with state as an explicit `dict[str, Tensor]` whose batch is on dim 0:

```python
import torch
from simulusion.inference import make_stepper

stepper = make_stepper(agent)  # DreamerStepper or IRISStepper
state = stepper.init_state(batch_size=8)  # 8 environments, one batched step
prev_action = None

with torch.inference_mode():
    while running:
        state = stepper.observe(state, obs, prev_action)  # filter the real frame
        prev_action = stepper.act(state)  # explore=False by default
        obs = envs.step(prev_action)

        if stepper.supports_imagination:  # plan or dream
            step = stepper.imagine(
                state, prev_action
            )  # .state, .reward, .continue_prob
```

| Stepper | `observe` input | `imagine` | Notes |
|---|---|---|---|
| `DreamerStepper` | raw `[0, 255]` images `(B, C, H, W)` | yes | `action_mode="mode"` (default) matches `Dreamer.act_with_world_model`. `"mean"` is deterministic and cheap. |
| `IRISStepper` | frames in `[0, 1]` `(B, C, H, W)` | no | Threads the policy LSTM state. For imagination, use `IRISAgent.imagine_rollout`, which manages the KV cache and cache rebuilding from the paper. |

To support another model, implement the `WorldModelStepper` protocol
(`init_state`, `observe`, `act`, and optionally `imagine`/`decode`).

### Deterministic Dreamer actions

`ActionDecoder.forward(deter=True)` estimates the policy's mode by drawing 100
samples and keeping the most likely one. That makes it random and about 100×
the cost of a single sample. For deployment, `ActionDecoder.mean_action(features)`
returns `tanh(mean)` in one deterministic pass. Select it with
`DreamerStepper(agent, action_mode="mean")`. It is not bit-identical to the
Monte Carlo mode, so evaluation returns can differ slightly. Measure before
switching a benchmark over.

### Explicit noise

The RSSM draws its stochastic state with `randn_like`. For reproducible or
exported steps, pass the noise in instead:

```python
state = stepper.observe(state, obs, prev_action, noise=(prior_noise, posterior_noise))
step = stepper.imagine(state, action, noise=prior_noise)
```

`RSSM.observe_step` and `RSSM.imagine_step` accept the same `noise` argument.
Omitting it keeps the previous behavior.

### An exportable step

`DreamerStepper(agent).step_module()` returns a `DreamerStepModule`: one
observe+act step as a pure `nn.Module`.

```python
module = stepper.step_module()
deter, stoch, action = module(
    deter, stoch, prev_action, obs, prior_noise, posterior_noise
)
inputs = module.example_inputs(batch_size=1)  # zero tensors of the right shapes
```

Because the noise is an input, the graph contains no random number generator.
The eager and exported versions therefore give the same results, and the host
program controls seeding. Pass zeros for the noise to act on the posterior
mean.

## Measuring speed

```python
from simulusion.inference import benchmark_step

report = benchmark_step(module, *inputs, warmup=20, iterations=200, batch_size=1)
print(report.summary())
# step: p50 0.412 ms, p99 0.530 ms, 2398.1 steps/s, 2398.1 items/s (batch 1), peak 41.2 MB
```

Each iteration is timed separately, with a CUDA synchronization, so tail latency
(p99) is visible. Tail latency is what matters when acting in a real-time
environment. The warm-up iterations absorb compilation, CUDA-graph capture and
cuDNN autotuning. To benchmark a stateful loop, pass a closure that advances
its own state.

(drift)=
## Measuring fidelity: rollout drift

```python
from simulusion.inference import rollout_drift


def step(fn):
    def run(state, obs, prior_noise, post_noise):
        deter, stoch, action = state
        return fn(deter, stoch, action, obs, prior_noise, post_noise)

    return run


report = rollout_drift(step(eager_module), step(candidate), init_state, per_step_inputs)
print(report.summary())  # drift over 50 steps: final max|err| 3.1e-06, ...
assert report.within(atol=1e-3)
```

Both rollouts run closed-loop from the same initial state, and each feeds back
its *own* output. The report therefore shows compounding error, per step, as
both max absolute error and relative L2 error. Pass randomness through the
step inputs. Compiled and exported graphs don't consume the global RNG the way
eager code does, so seeding alone won't make the two rollouts comparable.

Typical use: compare `fp32` eager against `bf16`, a quantized model, or an
exported artifact before switching to it.

## Reducing model cost

These change numerics, so measure them with rollout drift.

### Weight-only int8 quantization

```python
from simulusion.inference import quantize_weights
from simulusion.inference.quantize import weight_memory_bytes

before = weight_memory_bytes(module)
quantized = quantize_weights(module)  # names of the replaced layers
print(before / weight_memory_bytes(module))
```

This stores `nn.Linear` weights as int8 with one scale per output channel, and
dequantizes them on the fly. Activations stay in floating point, so no
calibration data is needed. Under `torch.compile` the dequantization fuses into
the matmul. The native implementation needs no extra dependency and exports
with `torch.export`. `backend="torchao"` uses torchao's kernels when that
package is installed.

Left in floating point by default, and why:

- **VQ codebooks and quantizers** (IRIS, Genie tokenizers): tokens are chosen
  by nearest-neighbor distance, so small weight errors change which token is
  picked.
- **RSSM stochastic heads** (`fc_state_prior`, `fc_state_posterior`): their
  outputs are sampled from and fed back every step.
- **Layers with `in_features < 64`**: little memory to save, and the most
  relative error.

Use `skip=` and `min_in_features=` to change this.

### Fewer sampling steps

- **DIAMOND**: `DiamondConfig.num_sampling_steps` (default 3) sets the
  denoising steps per frame. Cost is linear in it.
- **Genie**: the `MaskGITSampler` settings control per-frame refinement.

### Genie: temporal KV cache

The ST-transformer's spatial attention and MLP act within a frame, and its
temporal attention is causal. Earlier frames therefore never depend on later
ones, and generating the next frame only needs the cached temporal keys and
values of the previous frames:

```python
video = genie.generate(prompt_frame, num_frames=16, actions=actions, use_cache=True)
tokens = genie.dynamics_model.autoregressive_sample(prompt, actions, 16, use_cache=True)
```

Without the cache, frame *t* re-runs the network over all *t* previous frames,
which is O(T²) over a rollout. With it, each frame costs one frame's compute,
which is O(T). The logits match the uncached path up to float rounding. The
lower-level API (`DynamicsModel.init_cache` / `forward_cached`, and
`STTransformer.forward(x, cache=..., commit=...)`) supports uncommitted
forwards for evaluating a candidate frame repeatedly. It is off by default.

IRIS already decodes with a preallocated KV cache (`IRISTransformer.init_cache`,
`generate_frame_cached`), and every attention block in Simulusion uses
`scaled_dot_product_attention`, so FlashAttention and memory-efficient kernels
are selected automatically on supported GPUs.

## Exporting

```python
from simulusion import export_model, load_exported, verify_export

path = export_model(
    module, "step.pt2", format="exported_program", example_inputs=inputs
)
verify_export(module, path, inputs)  # raises if outputs differ; returns max |err|
runner = load_exported(path)  # callable, no model source needed
```

| Format | Output | Use it for |
|---|---|---|
| `"exported_program"` (aliases `"export"`, `"pt2"`) | `torch.export` graph, `.pt2` | **Recommended.** Loadable without the model's code. Input to AOTInductor, ExecuTorch and TensorRT. |
| `"aoti"` | AOTInductor package, `.pt2` | Ahead-of-time compiled. Runs from Python or C++ (`torch::inductor::AOTIModelPackageLoader`) without the model's Python code. Needs a C++ toolchain at export time. |
| `"onnx"` | `.onnx` | ONNX Runtime, TensorRT, and edge runtimes. |
| `"tensorrt"` | Torch-TensorRT module | NVIDIA GPUs. Now compiles through the maintained `ir="dynamo"` frontend. Pass `ir="ts"` for the old behavior. |
| `"torchscript"` | `.pt` | Legacy. TorchScript is in maintenance mode upstream. |

Pass `dynamic_shapes=` (the `torch.export` form) to keep dimensions such as
batch symbolic. Export the **step**, not the rollout: the loop stays in host
code, which keeps the graph small and the state explicit.

See the {doc}`export_guide` for target resolution on agents
(`agent.export(..., target="obs_encoder")`).

:::{note}
Simulusion's public models have an `export()` method. For any other module, use
`simulusion.export_model(module, ...)`.
:::

## Deployment bundles

A checkpoint alone isn't enough to deploy: the server also needs the config,
the input shapes and dtypes, and ideally a compiled artifact. A bundle is a
directory holding all of that:

```python
from simulusion.inference import save_bundle, load_bundle

save_bundle(
    "dreamer_walker_bundle",
    stepper.step_module(),
    example_inputs=module.example_inputs(batch_size=1),
    formats=("exported_program", "onnx"),
    config=agent.args,
    metadata={"env": "walker-walk"},
)

bundle = load_bundle("dreamer_walker_bundle")
step = bundle.load_artifact("exported_program")  # deployment path
bundle.load_weights(eager_module)  # or restore eager weights
```

```text
dreamer_walker_bundle/
    manifest.json          versions, input/output spec, artifact index, verification
    weights.safetensors    (weights.pt if safetensors is not installed)
    config.json
    model.pt2
    model.onnx
```

Each artifact is checked against the eager module when it is written, and the
maximum error is recorded under `verification` in the manifest.

From the command line:

```bash
simulusion deploy inspect dreamer_walker_bundle
simulusion deploy bench dreamer_walker_bundle --device cuda --iterations 500 --json
```

`deploy bench` builds zero inputs from the recorded input spec and times the
artifact with `benchmark_step`.

## Recommended recipes

**Acting in a real environment (latency).** Batch size 1, fixed shapes.
Use a stepper with `action_mode="mean"`, wrap the step with
`optimize_for_inference(..., compile=True)` for CUDA graphs, and track p99
latency. For example, a Dreamer observe+act step (default sizes) on an RTX 3050
Laptop GPU went from 6.5 ms (p50, eager) to 0.9 ms when compiled. Don't expect
bf16 to help here: a batch-1 step is launch-bound, so autocast only adds cast
kernels (the same step was *slower* in eager bf16, 7.9 ms). Reduced precision
pays off with larger batches or large models.

**Imagination during training (throughput).** Batch as many rollouts as memory
allows, use `compile_mode="max-autotune"` and bf16, and track `items_per_sec`.

**Interactive generation (Genie/DIAMOND play).** Genie: `use_cache=True`.
DIAMOND: reduce `num_sampling_steps`, compile the denoiser, and check visual
quality with the {doc}`evaluation_guide` metrics (PSNR, LPIPS, FVD).

**Serving without the training code.** Export a step module to
`exported_program`, or to `aoti` for C++. Ship it with `save_bundle` and verify
it with `simulusion deploy bench` on the target machine.

## Platform notes

- **Windows**: `torch.compile` on CUDA needs Triton, which is not installed by
  default. Simulusion warns and falls back to eager execution. AOTInductor needs
  MSVC. For production builds, use Linux.
- **Optional packages**: `onnx`/`onnxscript` (ONNX export), `onnxruntime` (running
  ONNX artifacts), `torch-tensorrt`, `torchao` and `safetensors`. None are
  required by the core inference API.
