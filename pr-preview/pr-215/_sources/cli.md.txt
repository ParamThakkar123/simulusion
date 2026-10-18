Simulusion CLI
===========

The project exposes a small command-line interface for common developer tasks:

- Run the CLI with: `python -m simulusion.cli <command>`; after installing the
  package an installed entrypoint is available as `simulusion <command>`. Tests
  and plugin integrations may invoke the top-level Click app (`simulusion.cli.app`)
  or the console-script callable (`simulusion.cli.run`).
- The CLI uses Click directly and lazy imports to keep startup fast; some
  commands require optional dependencies (listed below).

Commands
--------

- `version` - Print the installed `simulusion` package version (or "unknown" if
  the package cannot be imported).

- `envs list` - List built-in environment backends and example environment ids.
  This reads the environment catalog from `simulusion.catalog` if available.

- `datasets list [PATH]` - List dataset entries under `PATH`. If `PATH` is not
  provided the command uses `SIMULUSION_HOME` or defaults to `~/.simulusion`.

- `datasets convert <src> [--dest-format video] [--out-dir DIR]` - Convert a
  simple dataset file into another format. The initial implementation supports
  converting HDF5 (`.h5`) or NumPy (`.npz`/`.npy`) datasets into MP4 video
  files (one file per episode) when `--dest-format video` is used. Output files
  are written to the specified `--out-dir` or `./converted_datasets` by
  default.

- `collect --env <ENV_ID> [--steps N] [--out PATH] [--random-policy]` - Run a
  (random) policy for a number of environment steps and save interactions to
  a compressed `.npz` with keys `observations`, `actions`, `rewards`,
  `dones`.

- `train <model> [extra args...] [--inproc]` - Launch an existing training
  entrypoint. The CLI maps simple model names to modules in
  `simulusion.training` (e.g. `diamond`, `iris`, `planet`, `jepa`, `rssm`,
  `genie`). By default `train` spawns a subprocess running
  `python -m simulusion.training.<name>` and forwards any extra args. Use
  `--inproc` to attempt running the training entrypoint in-process (calls the
  module's `main()` if available). DIAMOND, IRIS, and JEPA accept
  `--config PATH`, `--print-config`, and OmegaConf/Hydra-style dot-list
  overrides such as `total_epochs=100` or `optimization.lr=3e-4`.

- `eval --model <NAME> --checkpoint <PATH> [options]` - Evaluate a trained world
  model. Generative models (`diamond`) are scored with FID, FVD, LPIPS, and PSNR,
  comparing real trajectories (collected from the environment) against generated
  ones. `jepa` has no rollout to score, so it runs the paper's frozen-encoder
  linear probe instead and takes `--root-path` rather than `--game`; options that
  belong to the other model are rejected rather than ignored. See
  {doc}`evaluation_guide` for details and interpretation.

  Key options:
  - `--model`, `-m` — model type (currently `diamond`)
  - `--checkpoint`, `-c` — path to checkpoint
  - `--game`, `-g` — environment name
  - `--num-videos` — number of trajectories (default 256)
  - `--metrics` — comma-separated metrics, e.g. `fid,fvd,lpips,psnr`
  - `--record PATH` — save real and generated videos
  - `--output`, `-o` — save results JSON

- `play --model <NAME> --checkpoint <PATH> [options]` - Interactively play
  inside a trained world model. Two modes toggled by `TAB`: **REAL** (env
  stepping) and **DREAM** (model imagination). Press arrow keys / WASD to
  override the agent's actions.

  Key options:
  - `--model`, `-m` — model type (currently `diamond`)
  - `--checkpoint`, `-c` — path to checkpoint
  - `--game`, `-g` — environment name
  - `--deterministic` / `--stochastic` — action selection (default deterministic)
  - `--record PATH` — save gameplay video
  - `--record-fps` — video framerate (default 20)

- `models list` - Print the known training entrypoints and (when available)
  exported model names from `simulusion.models`.

Environment / optional dependencies
----------------------------------

- SIMULUSION_HOME - Directory used by `datasets list` when no path is provided.

- The following commands require optional packages which may not be installed
  in all environments:
  - `collect`: requires `gym`/`gymnasium` and `numpy`.
  - `datasets convert`: requires `h5py`, `numpy` and video helpers used by the
    repository.

Notes and examples
------------------

- Example: show version

```bash
simulusion version
```

- Example: list environments

```bash
simulusion envs list
```

- Example: list datasets in default location

```bash
simulusion datasets list
```

- Example: convert a local HDF5 dataset to MP4 files

```bash
simulusion datasets convert data/my_dataset.h5 --out-dir /tmp/videos
```

- Example: collect 1000 steps from Pong and save as `pong.npz`

```bash
simulusion collect --env ALE/Pong-v5 --steps 1000 --out pong.npz
```

- Example: run IRIS training with a library YAML config and a dot-list override

```bash
simulusion train iris --config simulusion/configs/experiments/iris.yaml total_epochs=100
```

- Example: inspect a composed JEPA config without starting training

```bash
simulusion train jepa --config simulusion/configs/experiments/jepa.yaml optimization.epochs=50 --print-config
```

- Example: launch a DIAMOND preset from the unified training CLI

```bash
simulusion train diamond --config simulusion/configs/experiments/diamond.yaml preset=small seed=3
```

- Example: evaluate a DIAMOND checkpoint

```bash
simulusion eval --model diamond --checkpoint checkpoints/diamond/checkpoint.pt --game Breakout-v5
```

- Example: linear-probe an I-JEPA checkpoint

```bash
simulusion eval --model jepa --checkpoint results/jepa/jepa_run-latest.pth.tar \
    --root-path /data/imagenet --model-name vit_base --output probe.json
```

- Example: interactively play inside a DIAMOND world model

```bash
simulusion play --model diamond --checkpoint checkpoints/diamond/checkpoint.pt --game Breakout-v5 --record gameplay.mp4
```

Maintaining this page
---------------------

If you add or rename CLI commands in `simulusion.cli`, update this page with the
new usage, examples, and any additional optional dependencies.

