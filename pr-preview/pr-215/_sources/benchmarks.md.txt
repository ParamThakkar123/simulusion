# Benchmarking World Models

Simulusion includes a lightweight benchmark harness for running standardized
evaluations of trained world-model agents and exporting results that can be used
in experiment logs, reports, and papers.

## Quick Overview

- Preferred CLI entrypoint: `simulusion benchmark`.
- Benchmark adapters live in the Simulusion source tree under `simulusion/benchmarks/`.

## Supported adapters

The benchmark CLI currently registers these adapters out of the box:

- `diamond` - DIAMOND diffusion world-model agent
- `iris` - IRIS transformer-based agent
- `dreamerv1` / `dreamerv2` - Dreamer family

Benchmarks are intended for trained models. For single-agent runs, pass a
checkpoint with `--checkpoint`. For multi-agent runs, pass one or more
`--checkpoint-map AGENT=PATH` values, or use `--train-epochs` when you
intentionally want the CLI to train before evaluating.

## Simulusion CLI examples

Run IRIS on Pong for 3 episodes using seed 0 and write the standard result files
to `results/bench`:

```bash
simulusion benchmark \
  --agent iris \
  --game ALE/Pong-v5 \
  --checkpoint checkpoints/iris/pong.pt \
  --seeds 1 \
  --episodes 3
```

Run DIAMOND on Breakout with two explicit seeds and 5 episodes per seed:

```bash
simulusion benchmark \
  --agent diamond \
  --game Breakout-v5 \
  --checkpoint checkpoints/diamond/breakout.pt \
  --seeds 0,1 \
  --episodes 5 \
  --out-dir results/diamond_breakout
```

Run DreamerV2 on a Gym environment:

```bash
simulusion benchmark \
  --agent dreamerv2 \
  --game Pong-v5 \
  --checkpoint checkpoints/dreamerv2/pong.pt \
  --seeds 1 \
  --episodes 10 \
  --device cpu
```


Run DreamerV1 on a DeepMind BSuite diagnostic task:

```bash
simulusion benchmark \
  --agent dreamerv1 \
  --game catch/0 \
  --env-backend bsuite \
  --checkpoint checkpoints/dreamerv1/bsuite_catch.pt \
  --seeds 0,1 \
  --episodes 10 \
  --device cpu
```

The BSuite backend is optional. Install it with `pip install simulusion[bsuite]` or
`pip install bsuite` before running BSuite tasks. Simulusion wraps BSuite's compact
`dm_env` observations as synthetic RGB images so the existing pixel-based
Dreamer benchmark path can evaluate trained world-model agents.

Run all registered adapters on the same game with per-agent checkpoints:

```bash
simulusion benchmark \
  --all-agents \
  --game ALE/Pong-v5 \
  --checkpoint-map iris=checkpoints/iris/pong.pt \
  --checkpoint-map diamond=checkpoints/diamond/pong.pt \
  --checkpoint-map dreamerv1=checkpoints/dreamerv1/pong.pt \
  --checkpoint-map dreamerv2=checkpoints/dreamerv2/pong.pt \
  --seeds 0,1 \
  --episodes 5 \
  --out-dir results/pong_comparison
```

## CLI options

Common `simulusion benchmark` options:

- `--agent AGENT` / `-a AGENT`: run one adapter (`iris`, `diamond`, `dreamerv1`, or `dreamerv2`).
- `--all-agents`: run every registered adapter on the same environment.
- `--game GAME` / `-g GAME`: environment id, such as `ALE/Pong-v5` or a BSuite id like `catch/0`.
- `--checkpoint PATH` / `-c PATH`: checkpoint path for single-agent benchmarks.
- `--checkpoint-map AGENT=PATH`: repeatable per-agent checkpoint mapping for `--all-agents`.
- `--env-backend BACKEND`: optional backend hint forwarded to adapters. Use `bsuite` for DeepMind BSuite ids such as `catch/0`.
- `--seeds SEEDS`: either `N` for seeds `0..N-1`, or a comma-separated list such as `0,1,2`.
- `--episodes N` / `-n N`: number of evaluation episodes per seed.
- `--out-dir DIR`: output directory for report artifacts. The legacy alias `--out_dir` is also accepted.
- `--device DEVICE`: device forwarded to adapters. Defaults to CUDA when available, otherwise CPU.
- `--preset PRESET`: optional adapter/model preset.
- `--train-epochs N`: for `--all-agents`, train first when checkpoint maps are not supplied.

You can also run `simulusion benchmark --help` to see the installed CLI help.

## Python usage

For benchmark runs, prefer the main Simulusion CLI so commands are consistent with
the rest of the package:

```bash
simulusion benchmark --agent iris --game ALE/Pong-v5 --checkpoint checkpoints/iris/pong.pt
```

The command writes benchmark JSON reports under the configured output
directory. Load those reports with standard Python tools when you need custom
analysis:

```py
import json

res = json.load(open("results/bench/benchmark_results.json"))
per_seed = res["aggregate"]["per_seed_means"]
print(per_seed)
```

## Running the Atari 100k benchmark

To run the full Atari 100k benchmark on all configured games with the
centralized benchmark module:

```bash
python -m simulusion.benchmarks.atari_100k --benchmark
```

This runs the Atari 100k evaluator from `simulusion/benchmarks`, computes
human-normalized scores, and reports aggregate metrics across games and seeds.

## Outputs

The runner saves these files into the selected `out_dir` (default
`results/bench`):

- `benchmark_results.json` - raw structured results.
- `benchmark_results.csv` - per-seed rows.
- `benchmark_results.md` - human-readable markdown table.
- `benchmark_results.tex` - LaTeX table ready for papers.

Multi-agent runs also write combined reports such as
`combined_benchmark_results.json` and `combined_benchmark_results.csv` in the
root output directory, with per-agent details under subdirectories.

## Computing IQM and bootstrap CIs

The runner stores per-seed means in the JSON under
`aggregate.per_seed_means`. Use your preferred statistics package to compute
IQM and confidence intervals from that array.

## Extending the harness

- Create an adapter in `simulusion/benchmarks/adapters.py` that implements:
  - `load_checkpoint(path: str)`
  - `evaluate(num_episodes: int, render: bool = False)` returning `{"episode_returns": list[float]}`
- Register your adapter in `simulusion/benchmarks/cli.py` to expose it through `simulusion benchmark`.

## Compute benchmarks (no checkpoints required)

`simulusion benchmark` measures *returns*, so it needs trained agents and a
Gymnasium install. To measure *cost* - parameters, latency, throughput and
peak memory - use the auto-run sweep instead:

```bash
# every core model, tiny scale, on the best available device
bash scripts/benchmark_models.sh

# a realistic training step on the GPU, including the full IRIS/Genie stacks
bash scripts/benchmark_models.sh --preset small --all --device cuda

# one family at the sizes its paper uses
bash scripts/benchmark_models.sh --family iris --preset paper

# inference-only, half precision
bash scripts/benchmark_models.sh --no-backward --dtype bf16

# see what would run
bash scripts/benchmark_models.sh --list
```

The shell driver is the entrypoint: it installs uv if it is missing, installs
the project and its locked dependencies with `uv sync --inexact`, runs the
sweep with `uv run`, and prints the results table. Nothing needs to be set up
first - no virtualenv, no activation step. `--inexact` means the sync never
uninstalls packages the lock does not mention, so it is safe to point at an
environment you already use.

`scripts/benchmark_models.py` does the measuring and can be run directly
(`python scripts/benchmark_models.py ...`) when you manage the environment
yourself. Every model is built from synthetic tensors, so the sweep needs no
checkpoints, datasets or environments. Each case is isolated: a model that
fails to build or run is reported as `failed` and the sweep continues.

One caveat if you installed torch from a CUDA index (as `pyproject.toml`
describes): `uv sync` normally leaves it alone, but if uv has to rebuild
`.venv` from scratch it installs the plain wheel the lock pins, which on
Windows is CPU-only. Pass `--no-sync` to keep such an environment untouched.

Wrapper-only flags:

- `--no-sync` skips the install step and uses the current environment.
- `--extra NAME` (repeatable) installs an optional dependency group, e.g.
  `--extra viz --extra ml`.
- `--python VERSION` picks the interpreter for the environment.
- `--uv-help` prints the driver's own help.

Everything else is forwarded to the Python sweep:

- `--preset tiny|small|paper` sets batch, sequence length and the width/depth
  overrides; `paper` uses each model's library defaults.
- `--batch-size`, `--seq-len` and `--image-size` override individual preset
  fields; `--warmup` and `--iters` control the timing loop.
- `--models a,b` or `--family dreamer,iris` narrows the sweep; `--all` adds the
  heavy tier (`iris-world-model`, `genie-small`, `jepa-vit-small`).
- Reports land in `results/model_benchmarks/` as `model_benchmarks.{json,csv,md}`
  alongside the run metadata (device, dtype, torch version, thread count).

`make bench` and `make bench-all` wrap the two common invocations; pass extra
flags through `BENCH_ARGS`.

## Inference videos and interactive play

The same shell driver can run a **trained** model instead of the synthetic
compute sweep. `--infer` writes a video of the model playing (or generating).
`--play` opens a window so you can play **with** the policy (keys override it)
or **against** it (`--versus`: you drive, the policy's chosen action is shown).

```bash
# video of DIAMOND playing Breakout, plus a dream clip
bash scripts/benchmark_models.sh --infer --model diamond \
    -c checkpoints/diamond/checkpoint_0.pt --game Breakout-v5

# IRIS on Pong, headless
bash scripts/benchmark_models.sh --infer --model iris \
    -c checkpoints/iris/checkpoint_0.pt --game ALE/Pong-v5

# Dreamer in the real env (no OpenCV window)
bash scripts/benchmark_models.sh --infer --model dreamer \
    -c runs/.../ckpts/10000_ckpt.pt --game Pendulum-v1

# interactive: you and the policy share the controls
bash scripts/benchmark_models.sh --play --model diamond -c ckpt.pt

# interactive: you drive, the policy is the on-screen opponent
bash scripts/benchmark_models.sh --play --model diamond -c ckpt.pt --versus

# Genie latent-action play (or --random-init to check the pipeline)
bash scripts/benchmark_models.sh --play --model genie -c checkpoints/genie.pt
bash scripts/benchmark_models.sh --infer --model dit --random-init
bash scripts/benchmark_models.sh --infer --list
```

`--play` needs a display. `--infer` is the headless path. Videos default to
`results/model_inference/`. DiT and I-JEPA only support `--infer` (sample /
mask visualisations), not a game loop.

`make bench-infer` and `make bench-play` wrap those two modes.

To add a model to the sweep, append a `BenchCase` to `CASES` in
`scripts/benchmark_models.py` with a builder that returns the module plus a
function producing its synthetic inputs.

## Tests and CI

- Place smoke tests under `simulusion/benchmarks/tests/` so CI can run them quickly.
- The repo contains a `mocking_classes.py` helper for building fake agents and environments for fast unit tests.

## Where to start

- Run the examples in `examples/benchmark_iris.py` or use `simulusion benchmark` directly.
- If you need help wiring specific agent configs, use `--device`, `--preset`, and checkpoint options, or call the runner programmatically and pass `extra_kwargs`.
