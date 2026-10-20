# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- `simulusion train MODEL` runs registered tasks on the shared Trainer.
  Commands written for the older per-family entrypoints (a `--config` without
  `task:`, or flat overrides such as `total_epochs=100`) still reach them, as
  does `--legacy`.

### Fixed
- A resumed DIAMOND run kept a replay buffer only as large as it was at the
  checkpoint: `save_checkpoint` recorded the trimmed length as the capacity.
  Its replay and observation side files are now found next to the
  checkpoint even after the directory moves.
- The World Models CMA-ES controller search minimised the episode return
  instead of maximising it, and its rollouts called `Controller` with two
  arguments where `forward` takes one, so every rollout raised.
- `GenieWorldModel.encode` without actions failed for Genie's default mean-
  pooled latent actions, which infer one action per clip.

### Renamed
- **Synora is now Simulusion.** The distribution is `simulusion` on PyPI
  (`pip install simulusion`, extras as `simulusion[gym]`), the package is
  `import simulusion`, the CLI commands are `simulusion` / `simulusion-train`,
  and environment variables use the `SIMULUSION_` prefix. Nothing was published
  under the Synora name, so there is no `synora` import alias. Checkpoints load
  unchanged: they are state dicts read with `weights_only=True`.

### Added
- **One trainer for every family** (`simulusion.training.Trainer`), built on
  Hugging Face Accelerate (now a core dependency). Any training task runs to a
  step budget with one precision setting (`fp32` / `bf16` / `fp16` / `auto`),
  data parallelism (`accelerate launch` / `torchrun`), gradient accumulation,
  logging to `metrics.jsonl` / TensorBoard / W&B, periodic evaluation, early
  stopping, atomic checkpoints with resume (bit-exact in one process; replay
  buffers included for online agents) and export of the trained world model.
  Tasks: `dreamer`, `dreamer-v1`, `dreamer-v2`, `planet`, `iris`, `diamond`
  (agents, collecting online), `dreamer-wm`, `iris-wm`, `diamond-wm`, `genie`,
  `vae`, `mdrnn` (world models from data), `controller` (World Models' C by
  CMA-ES, in the real environment or inside the dream), `dit` and `jepa`.
  Write your own with `TrainingTask` and `register_task`. See the new "One
  Trainer for Every Model" guide.
- **Declarative run configs.** One YAML names a task and fills `trainer:` and
  the task's sections; `key=value` overrides reach any nested field and
  unknown keys fail with their dotted path. Every run writes its resolved
  `config.yaml` (pass it back with `--config` to repeat the run) and
  `run.json` (git commit and dirty flag, package versions, GPUs, command
  line). Shipped configs for every task are in `simulusion/configs/runs/`.
  `simulusion train` runs them; `simulusion.run_training()` from Python.
- **World models train without their agent.** The `-wm` tasks fit Dreamer's,
  IRIS's or DIAMOND's world model to recorded episodes with no environment
  and no policy, and `simulusion.agents.PlanningAgent` / `CEMPlanner` act
  through any world model with a reward head by planning in it.
- **One episode format for video, RL and robot data** (`simulusion.data`): an
  `.npz` per episode with frames, optional actions / rewards / episode ends
  between them, extra cameras, proprioception and a language instruction;
  `EpisodeFolder` / `EpisodeWriter`, and `SequenceDataset` windows that never
  cross an episode boundary. Readers for LeRobot datasets (v2.x and v3.0,
  local or from the Hub; `pip install simulusion[lerobot]`) and RLDS / Open
  X-Embodiment (with TensorFlow Datasets), plus `simulusion datasets import`
  and `simulusion datasets info`.
- `MixedPrecision` and `GradientUpdate` replace the hand-written GradScaler
  blocks in Dreamer, IRIS, DIAMOND, Genie, PlaNet, ConvVAE and MDN-RNN.
  `use_amp` still selects what it did, so results are unchanged.
- **One `WorldModel` protocol for every family** (`simulusion.world_model`):
  `encode` / `step` / `rollout` / `decode` / `save`, with the same frame,
  action and state conventions everywhere. `as_world_model(model)` wraps a
  trained Dreamer (V1/V2), PlaNet, IRIS, DIAMOND, Genie or (ConvVAE, MDN-RNN)
  model; `load_world_model(path, family)` reads each family's own training
  checkpoints, and any world model saves to one directory format.
  `WorldModelSpec` says what a model reads and predicts (action space, context
  length, reward and termination heads). Steps go through each family's own
  code: IRIS matches `imagine_rollout`, Dreamer its stepper, PlaNet its
  planner. A DIAMOND world model loads without the Atari emulator or replay
  buffer. JEPA and DiT have no adapter, since neither is action-conditioned.
  See the new "One interface for every world model" guide.
- A demo per model in `demos/` (Dreamer, DreamerV2, PlaNet, ModularRSSM,
  DIAMOND, IRIS, Genie, DiT, I-JEPA). Each trains from scratch on a 4 GB laptop
  GPU in one to two hours and records video of what it learned;
  `demos/run_all.py` reproduces all of them.
- A documentation gallery with the recorded clips, plus a "See it in action"
  section on every algorithm page and in the README.
- IRIS imagination replays each single-position Transformer step as a CUDA
  graph on GPU (`GraphedDecodeStep`). Outputs are identical; on an RTX 3050 a
  20-step imagination rollout drops from 11.4 s to 3.2 s, because the eager
  path spent most of its time launching ~55k small kernels. Disable with
  `agent.use_cuda_graphs = False`.
- `DreamerV1Agent` and `DreamerV2Agent`, the agents behind
  `create_model("dreamer-v1")` / `create_model("dreamer-v2")`.

### Fixed
- `create_model("dreamer-v1")` and `create_model("dreamer-v2")` raised a
  `TypeError`; they now build agents that train their own version.
- DreamerV2 could not train: its two-hot heads exposed `mean` as a method with
  an extra trailing axis, and its actor loss detached the lambda-returns, so the
  actor received no gradient. The actor also maximised `symlog` of the returns
  rather than the returns themselves (DreamerV2, Sec. 2.4); on DMC cartpole
  swing-up that pinned every action at -1 and stalled the return at ~75.
- DIAMOND's imagination paired each conditioning frame with the action taken one
  step later, so actor-critic training rolled the diffusion model forward on
  shifted actions.
- The TinyWorlds loader read each game's flat `(N, H, W, 3)` frame stream as
  grayscale clips, so every Genie sample was one scrambled frame; clip files
  also had channels interleaved across frames.
- Genie's video tokenizer reshaped the quantizer's channel-first output instead
  of permuting it, scrambling embeddings across positions. The decoder learned
  that layout, so every frame Genie generated from tokens decoded to noise.
  Genie models trained before this fix need retraining.
- Genie checkpoints saved after warmup failed to load with `weights_only=True`,
  and the trainer ignored `tokenizer_num_heads` / `action_num_heads`.
- DiT trained on CIFAR-10 in `[0, 1]` instead of `[-1, 1]`, a class-conditional
  run crashed at its final sampling step, its `config.yaml` omitted the class
  settings needed to reload it, and periodic checkpoints were one epoch late.

## [1.0.0] — 2026-10-04

First stable release, and the first under the name **Synora** (formerly
TorchWM). The public API surface documented in `docs/source/public_api.md` is
now covered by semantic versioning: breaking changes to it require a major
version bump and a deprecation cycle.

Because 0.5.0 below was never tagged or published, upgrading from TorchWM 0.4.2
— the last `torchwm` release on PyPI — also brings in every 0.5.0 change.

### Renamed
- **TorchWM is now Synora.** The distribution is `synora` on PyPI, the package
  is `import synora`, the CLI commands are `synora` / `synora-train`, and
  environment variables use the `SYNORA_` prefix. There is no `torchwm`
  import alias. Checkpoints load unchanged: they are saved as state dicts
  and read with `weights_only=True`, so they hold no module paths. Deployment
  bundle manifests record `synora_version` instead of `torchwm_version`.

### Known limitations
- Shared step-budget `train()` covers the Dreamer family; other models use
  their own trainers or `synora train`
- On CPython 3.13 the `dmc` extra installs everything except `dm_control`
  itself; run `python -m synora.install_dmc` to add it

### Added
- `synora.inference`, an efficient-inference and deployment toolkit (see the
  new *Efficient Inference and Deployment* guide):
  - `optimize_for_inference` / `InferenceModel`: eval + no-grad + one precision
    policy (`fp32`/`bf16`/`fp16`/`auto`) + optional `torch.compile` with CUDA
    graphs, channels-last and weight casting. Defaults are numerically
    identical to eager.
  - A stateful step interface (`make_stepper`, `DreamerStepper`,
    `IRISStepper`) and `DreamerStepModule`, a pure observe+act step with
    sampling noise as an input, ready for `torch.export`.
  - `benchmark_step` (p50/p90/p99 latency, throughput, peak memory) and
    `rollout_drift` (closed-loop divergence of an optimized step from a
    reference).
  - `quantize_weights`: weight-only int8 for `nn.Linear`, skipping VQ
    codebooks, RSSM stochastic heads and small layers by default; optional
    torchao backend.
  - `save_bundle` / `load_bundle`: deployment bundles with weights, config,
    exported artifacts, a manifest and per-artifact verification.
- `synora deploy inspect` and `synora deploy bench` CLI commands.
- Export formats `exported_program` (`torch.export`, `.pt2`) and `aoti`
  (AOTInductor), plus `load_exported` and `verify_export`.
- Temporal KV cache for the ST-transformer (`STKVCache`,
  `DynamicsModel.init_cache` / `forward_cached`). Genie generation takes
  `use_cache=True` to generate each frame in O(1) frames of compute instead of
  re-running the whole prefix. Off by default.
- `RSSM.observe_step` / `imagine_step` accept explicit `noise`, and
  `ActionDecoder.mean_action` gives a deterministic single-pass action.
- `python -m synora.install_dmc` (also `make install-dmc`) installs the
  DeepMind Control backend on CPython 3.13, where dm-control's `labmaze`
  dependency has no wheel and builds with Bazel. It uses the pure-Python
  `labmaze-new` and installs dm-control with `--no-deps`. `--check` verifies an
  environment, `--dry-run` prints the commands. It keeps the installed mujoco and
  matches dm-control to it, because mujoco 3.13 removed an enum mujoco-mjx still
  uses and upgrading would break the `brax` extra; `--upgrade-mujoco` opts out.
  `[tool.uv] constraint-dependencies` caps `mujoco<3.13` for the same reason
- CUDA, then Apple MPS, then CPU device selection
  (`synora.utils.device`)
- `worldmodels` extra (albumentations, cma); `hydra-core`/`omegaconf` in `ml`;
  `h5py` in `viz`; `ruff` in `dev`
- Throughput instrumentation: `synora.ThroughputMeter`, `synora.measure_steps`
  and `synora.tensor_nbytes` report steps/sec, ms/step and bytes shipped to the
  device per step, synchronising on CUDA so the timer measures execution rather
  than queueing
- Performance helpers: `synora.enable_performance_defaults` (cuDNN autotuning
  and TF32, no-ops without CUDA), `synora.maybe_compile` (opt-in
  `torch.compile` with an eager fallback) and `synora.to_channels_last`
- `GenieConfig.use_amp` / `GenieSmallConfig.use_amp` — autocast for Genie
  training, preferring bfloat16 where supported so no gradient scaler is needed
- Genie `VideoDataset` loads `.npy` / `.npz` / `.pt` clips, or video files when
  OpenCV (`synora[viz]`) is installed
- `DreamerConfig.perf_defaults` and `DreamerConfig.tf32`
- `RSSMPolicy(..., compile_rollout=True)` compiles the CEM candidate-rollout
  step, which runs `num_iterations * planning_horizon` tiny kernels per env step
- I-JEPA linear evaluation (`synora.training.eval_jepa`, exported as
  `synora.jepa_linear_probe` / `synora.load_jepa_encoder`), implementing the
  paper's Appendix A.2 protocol: frozen target-encoder, average-pooled patch
  tokens, LARS-trained linear head, and the published sweep over learning rate,
  weight decay, batch-norm head, and last-layer vs last-four-layer features
- `VisionTransformer.get_intermediate_layers()`, needed for the last-four-layer
  probe representation
- `synora/configs/experiments/jepa_small_gpu.yaml` — single-GPU preset
  that reduces only batch size, backbone, and epochs, leaving every method
  parameter at the paper's value
- `tests/models/test_jepa_paper_alignment.py`, pinning the I-JEPA masking,
  architecture, loss, and schedule details against the paper
- `synora eval --model jepa` runs the linear probe from the CLI, alongside the
  existing `--model diamond` FID/FVD/LPIPS path. The two evaluations share only
  `--checkpoint`, `--batch-size`, `--device` and `--output`; passing an option
  that belongs to the other model is an error rather than a silent no-op

### Changed
- TensorRT export compiles through Torch-TensorRT's `ir="dynamo"` frontend by
  default (was the legacy `ir="ts"`); pass `ir="ts"` to keep the old path.
- **Breaking: one package.** The implementation moved from `world_models/` into
  `synora/`, and the alias layer that made `synora.<name>` resolve to
  `world_models.<name>` is gone. `synora` is now the only import path —
  `import world_models` raises `ModuleNotFoundError`. Replace
  `from world_models.x import y` with `from synora.x import y`; the public
  `synora` surface is unchanged
- **I-JEPA defaults now reproduce the paper.** The shipped configuration
  previously combined the two worst settings in the paper's own ablations:
  `enc_mask_scale` was `(0.15, 0.2)` where the context block calls for
  `(0.85, 1.0)` (Table 9), and `num_pred_masks` was `1` where the paper uses `4`
  (Table 10: 9.0 vs 54.2 low-shot top-1). View augmentations were also enabled
  by default, contradicting the paper's central claim; `use_gaussian_blur`,
  `use_horizontal_flip` and `use_color_distortion` now default to `False`.
  `min_keep` is `10`, `crop_scale` is `(0.3, 1.0)`, and the optimizer follows
  Appendix A: batch 2048, warmup 1e-4 -> 1e-3 over 15 epochs, cosine to 1e-6
- I-JEPA learning rates are quoted at the paper's batch size of 2048 and scaled
  linearly to the effective batch size; set `lr_reference_batch_size = None` to
  opt out
- I-JEPA trains with the paper's L2 loss (`loss_type="l2"`) instead of the
  reference implementation's Smooth-L1, which remains available as
  `loss_type="smooth_l1"` alongside the literal per-block sum `"l2_sum"`
- `JEPAConfig.pred_depth` defaults to `None`, which selects the paper's
  predictor depth for the configured backbone (6 for ViT-B, 12 for ViT-L/H, 16
  for ViT-G) instead of silently building a 6-layer predictor for every model
- The multi-block mask sampler draws block scale and aspect ratio from
  independent uniforms, as Sec. 3 specifies, rather than sharing one draw
- No `GradScaler` is created for bfloat16 training, which does not need loss
  scaling; `torch.autocast` replaces the deprecated `torch.cuda.amp` entry points
- Dreamer keeps replay observations in `uint8` across the host-to-device copy.
  The buffer already stores `uint8` and `preprocess_obs` already casts on the
  device, so the old `torch.tensor(obs, dtype=torch.float32)` widened the batch
  on the host and moved four times the bytes for a cast that happened anyway —
  123MB per step at stock config, now 30.7MB. Transfers use `torch.from_numpy`
  (no extra host copy) and pinned, non-blocking staging on CUDA
- The ViT/I-JEPA backbone uses `scaled_dot_product_attention` instead of an
  explicit `softmax(q @ k^T * scale) @ v`, so the (B, heads, N, N) score matrix
  is never materialised. Every other attention block in the package already did.
  A custom `qk_scale` is passed through unchanged; outputs match the explicit
  form to 1.7e-16 in float64
- Target-network EMA updates (I-JEPA, DiT) use `torch._foreach_*`: one fused
  multi-tensor op per step rather than two kernels per parameter tensor, and
  `alpha=` applies the `(1 - m)` scale inside the add instead of allocating a
  scaled copy of every source tensor. Results differ by at most one ulp, in the
  fused form's favour — it rounds once where the explicit form rounded twice
- Dataloaders pin host memory when CUDA is present and keep workers alive across
  epochs. `tinyworlds` hardcoded `pin_memory=False`, and the ImageNet loaders
  respawned every worker at each epoch boundary
- FID/LPIPS/FVD share one frozen backbone per device instead of rebuilding (and
  re-downloading) Inception or VGG for every metric instance
- The PlaNet CEM planner runs under `torch.inference_mode` rather than
  `no_grad`, and no longer clones state before broadcasting it over candidates
- The Dreamer replay buffer's sequence sampler tests the episode-boundary
  rejection arithmetically instead of materialising the index window for every
  rejected draw. The RNG is consumed identically, so a given seed still yields
  exactly the same sequences
- uv no longer resolves the whole project through a CUDA-specific PyTorch index.
  `[tool.uv]` set `https://download.pytorch.org/whl/cu121` as the *default*
  index with `index-strategy = "unsafe-best-match"`, which routed every package
  through it and pinned contributors to one CUDA build regardless of hardware —
  contradicting the README. Resolution now comes from PyPI; add the index for
  your platform explicitly if you need a specific wheel set

### Removed
These were deprecated, or would have been frozen into the 1.x API, and go
before 1.0 instead of being carried through it:
- The `.export()` method TorchWM installed on every `torch.nn.Module` at import,
  and the `TORCHWM_NO_GLOBAL_EXPORT` switch for it. Synora no longer modifies
  `torch.nn.Module`: every `nn.Module` class in the top-level `synora`
  namespace gets `.export()` from `ExportableAgentMixin` (a test enforces
  this), and any other module goes through `synora.export_model`.
  `install_export_method` is gone with it.
- `torchwm.utils.jit_utils` (TorchScript helpers); use `synora.maybe_compile`
  for speed or `synora.export_model(..., format="exported_program")` for
  deployment.
- The `dreamer-v3` / `dreamerv3` registry names and the `DreamerV3` export.
  They built the same `DreamerAgent` as `dreamer` - there is no DreamerV3
  implementation - so the name promised an algorithm the library does not have.
- The `procgen` extra, which could never install on Python >= 3.11
- `dm-control` from the `dmc` extra on CPython 3.13 only, so
  `pip install synora[dmc]` no longer fails there. New `dmc-uv` extra keeps it
  unconditional for uv, which drops the `labmaze` pin through
  `[tool.uv] override-dependencies`
- `tools` is no longer installed as a top-level package
- `nginx.conf`, which proxied a frontend that no longer exists
- **Breaking:** the `synora.inference` operator
  package (`get_operator`, `OperatorABC`, `TensorSpec`, `DreamerOperator`,
  `JEPAOperator`, `IrisOperator`, `PlaNetOperator`). The operators only resized
  and normalized tensors, and `JEPAOperator` masked uniformly at random, which
  is not I-JEPA's masking at all. Preprocess inputs directly, or use
  `synora.transforms.image.make_transforms` and
  `synora.masks.MultiblockMaskCollator`
- **Breaking:** the unused `operator_state_dim` / `operator_action_dim` fields
  on `DiamondConfig`
- **Breaking:** the `minerl` and `minedojo` extras, and the `selenium` extra.
  Neither Minecraft extra could ever install — MineRL 1.x has no Python 3.11+
  release and MineDojo pins `gym==0.21.0`, whose sdist no longer builds — and
  between them they made `uv lock` unresolvable for the whole project.
  `synora.envs.minecraft_env` is unchanged; `docs/source/iris.md` documents the
  manual Python 3.10 install

### Fixed
- `pip install synora[gym]` crashed the README quick start on gymnasium 1.3:
  the `gym` extra required `pygame` directly while `gymnasium[box2d]` 1.3
  depends on `pygame-ce`, and the two install over each other, so rendering
  any classic-control environment died with an access violation. The extra
  no longer requires pygame itself; gymnasium brings the renderer it needs.
- FID and FVD could stall for minutes: `scipy.linalg.sqrtm` ran a recursive
  Schur decomposition on the product of two 2048x2048 covariances, which is
  rank-deficient whenever there are fewer samples than feature dimensions.
  The Fréchet distance now takes only the trace it needs, from the
  eigenvalues of a symmetric PSD matrix (`numpy.linalg.eigh`), which is exact
  and no longer needs SciPy.
- Constructing a `DreamerAgent` called `setup_logging("synora")`, which turned
  off propagation on the package logger, so every `synora.*` record stopped
  reaching the application's own logging handlers (and pytest's `caplog`).
  `setup_logging` now leaves propagation on and only adds its console handler
  when the root logger has none, so messages still print exactly once.
- MP4 videos (`StreamingVideoWriter`, `save_video`, `combine_videos`, Dreamer
  rollout videos and every `demos/record_*.py`) were encoded with OpenCV's
  `mp4v` fourcc (MPEG-4 Part 2), which no browser plays. They now go through
  the new `synora.utils.utils.Mp4Writer`, which writes H.264 via
  `imageio-ffmpeg` (added to the `viz` and `worldmodels` extras), falls back
  to OpenCV `avc1`, and only then to `mp4v` with a warning.
- `maybe_compile` only guarded the `torch.compile` wrap, but compilation is
  lazy, so a missing backend (e.g. no Triton on Windows CUDA builds) raised on
  the first call instead of falling back. The first call is now guarded too,
  and compiling an `nn.Module` keeps its `state_dict` keys.
- IRIS evaluation collected every raw frame and a per-step latent even with
  `render=False`, the default during training, and then discarded them. At the
  default `eval_episodes=100` and the 27000-step episode cap that is tens of GB
  held until the evaluation returns, so the OS could kill the process outright
  (in a notebook, the kernel just restarts). Both are now render-only
- IRIS evaluation reset the same environment `collect_experience` keeps a
  partial episode on, so the next collection step paired the stale pre-eval
  observation with the post-eval environment and wrote a transition that never
  happened into the replay buffer. Evaluation now runs on its own environment;
  when the caller supplied the environment, the in-flight collection episode is
  dropped instead
- Genie training on TinyWorlds could kill the process with no traceback (a
  notebook kernel just restarts) whenever `num_workers > 0`, the default:
  `TinyWorldsDataset` kept one HDF5 handle open from `__init__`, and forked
  DataLoader workers all read through that inherited handle, which HDF5 does
  not support. Each process now opens its own handle on first read. The same
  handle made spawned workers (Windows, macOS) fail with
  `TypeError: h5py objects cannot be pickled`
- System-metrics logging crashed training on GPU machines: `collect_system_stats`
  guarded `torch.cuda.utilization` with `hasattr`, which is always true, while
  calling it raises `ModuleNotFoundError` without NVML (`nvidia-ml-py`). The
  counter is now skipped with a single warning, and `nvidia-ml-py` is part of the
  `ml` extra
- Dreamer acted on the RSSM prior instead of the posterior, so the current
  observation was encoded and then ignored during collection and evaluation
  (also in `play dreamer` and `scripts/benchmark_infer.py`)
- Dreamer checkpoints omitted the critic (`value_model`); older checkpoints
  still load, keeping a fresh critic and skipping its optimizer state
- `Dreamer.evaluate(render=True)` raised `KeyError: 0`
- `use_disc_model=True` raised `TypeError` in the actor loss; the discount head
  output is now also scaled by `discount`, as in the reference implementation
- Time-limit truncations were stored as terminal transitions; the replay buffer
  now keeps episode boundaries and true terminations separately
- `DreamerConfig.env_instance` made startup crash while writing `config.yaml`
- `Dreamer(restore=...)` ignored its argument
- `DreamerAgent.train()` now always writes a final checkpoint
- PlaNet sampled latents with uniform instead of Gaussian noise; `rollout_prior`
  returned tuples; `get_init_state` conditioned the posterior on the stale
  deterministic state
- Genie's default 5120-wide dynamics model used 36 heads, which do not divide
  the width (now 40); attention layers validate `dim % num_heads`
- `create_model("genie"/"genie-small")` silently dropped tokenizer and
  latent-action widths/depths from the config; `tokenizer_num_heads` and
  `action_num_heads` are now honoured (default 16, which is what was always
  built)
- DIAMOND collection kept every frame of the current episode in memory
- `synora train --inproc` re-ran training without arguments on `TypeError`
  and in a subprocess on any other error
- `synora eval` / `synora play` imported modules that are not in the wheel;
  they now live in `synora.inference`, and the metrics package moved from the
  top-level `evals` to `synora.evals`
- `register_env_backend` backends were never used by `make_env`; a `dmc`
  backend was added
- Every `torch.load` in scripts and demos now uses `weights_only=True`
- IRIS play in `scripts/benchmark_infer.py` reset the policy LSTM every step
- MP4 video logging wrapped uint8 frames around; GIFs are written with Pillow
  instead of the undeclared moviepy
- `MUJOCO_GL=egl` is only defaulted on Linux (macOS has no EGL)
- A broken W&B install no longer breaks importing JEPA training
- The multi-block mask sampler raised no error when `min_keep` exceeded the
  patches a block can hold; it now fails with an explanatory message instead of
  looping forever
- The test suite can run to completion in a single process again. Every
  `nn.Module` holds reference cycles, so a model built in a test is reclaimable
  only by CPython's cyclic collector — which triggers on allocation *counts*,
  not bytes, so a handful of very large tensors never trips it. The suite
  retained ~4.4GB (1.6GB from `tests/models/test_genie.py`, 1.4GB from
  `tests/evals/test_evals.py`) and died partway through with either
  `RuntimeError: can't start new thread` or a Windows access violation. A new
  `tests/conftest.py` collects after each test; the suite now finishes, at about
  15% more wall-clock

## [0.5.0] — 2026-07-27 (never tagged or published)

### Added
- Real PEP 561 typing stub: `torchwm/__init__.pyi` now declares every public
  export instead of falling back to `Any`, generated from the export map by
  `python -m tools.gen_type_stub` and verified in CI. `torchwm` ships a
  `py.typed` marker so the re-exports stay typed
- `torchwm play -m dreamer` — interactive REAL/DREAM playback for Dreamer
  checkpoints alongside DIAMOND (`scripts/play_dreamer.py`)
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
- CI job running the full test suite with every optional backend installed —
  the configuration a contributor gets from `CONTRIBUTING.md`, previously
  untested
- Tests asserting every `__all__` entry resolves, that the stub matches the
  export map, and that the README algorithms table matches the model registry
- Complete `torchwm` public import surface: every implementation submodule is now
  reachable through the friendly namespace (`from torchwm.models import Dreamer`,
  `import torchwm.envs`, ...), not just top-level factory helpers
- `dmc` optional-dependency extra (`pip install torchwm[dmc]`) that installs
  `dm-control` for the default DeepMind Control backend
- Actionable error from the DMC backend that names the missing `dm_control`
  dependency and points to `torchwm[dmc]` or the gym backend

### Changed
- README "Supported Algorithms" table now lists all 13 registered models
  (previously 5), keyed by the name `create_model()` accepts
- `viz` extra installs only what it uses (opencv, umap-learn, scikit-learn,
  plotly); the unused FastAPI/uvicorn/starlette/python-multipart entries and the
  duplicated docs dependencies are gone
- CI runs the test matrix on `push` to `main` only, not on every push to a PR
  branch, which ran the whole matrix twice per commit
- Documentation, examples, and scripts now import through the `torchwm` public
  namespace
- Quick-start examples (README, docs landing page, getting-started guide) now use
  the base-installable `Pendulum-v1` gym backend so they run on
  `pip install torchwm[gym]` out of the box; DMC usage is documented separately

### Fixed
- `torchwm.DreamerRSSM` and `torchwm.MujocoEnv` raised `AttributeError` on
  access: the export map pointed `DreamerRSSM` at a module that names the class
  `RSSM`, and `MujocoEnv` had no implementation behind it (it now resolves to
  `MuJoCoImageEnv`)
- README advertised a FastAPI visualization feature that does not exist
- `create_model("dreamer", env="walker-walk")` no longer raises a bare
  `ModuleNotFoundError`; the DMC dependency is installable and the error is clear
- Broken landing-page example that called `train(env_name=..., total_steps=...)`
  (the `train` method only accepts `total_steps`)
- Env-adapter tests patched non-existent module attributes
  (`torchwm.models.dreamer.env_wrapper.*`) and the wrong Gymnasium registry
  reference, causing 10 spurious failures
- Removed stray build artifacts (`nul`, empty `testsdata/` directories) and added
  `.gitignore` guards

## [0.4.2] — 2026-06-20

### Added
- Dreamer integration test for Pendulum-v1 wired into CI
- DIAMOND world model documentation
- `from_pretrained` and `from_config` class methods for Dreamer, IRIS, JEPA, Genie agents
- Gymnasium wrapper for world model environment
- ONNX export support for agents
- `py.typed` marker for PEP 561 type declarations
- PSNR evaluation metric

### Changed
- Restructured dreamer docs with separate V1/V2 theory and examples
- Stripped base deps to minimum, moved extras to optional groups
- Centralized version in `_version.py` as single source of truth
- Improved block exports and testing utilities

### Fixed
- Arbitrary code execution risk in pickle.load (hardened checkpoint/replay deserialization)
- Zero-element tensor reshape crash in ConvEncoder
- Empty sequence edge case in Dreamer training
- 25 GitHub Dependabot vulnerabilities (upgraded 8 packages)
- Dockerfile referencing removed `torchwm_ui` folder
- Replaced debug print calls with proper logging
- MyPy type errors across 40+ files
- Missing imports in `train_jepa.py` (mp, F, DistributedDataParallel)
- CI workflow and docs dependency config
- DreamerConfig documentation field sync

## [0.4.1] — 2026-06-01

### Added
- Modular RSSM with swappable LSTM/Transformer/MLP backbones
- Genie model support (video tokenizer, latent action model, dynamics model)
- Brax environment backend
- BSuite environment backend
- DMLab environment backend
- Procgen environment backend
- Robotics environment backend (gymnasium-robotics)
- Unity ML-Agents environment backend
- Sphinx documentation with auto-deploy to GitHub Pages
- Benchmark runners and reporting utilities
- CLI tools (`torchwm`, `torchwm-train`)

### Changed
- Migrated configs to dataclass style for consistency
- Improved lazy import architecture for faster CLI startup

### Fixed
- Cross-platform memory detection (Windows ctypes + Linux /proc/meminfo + psutil fallback)
- Environment wrapper stack consistency across backends

## [0.4.0] — 2026-05-15

### Added
- Initial public release
- Dreamer (V1/V2) agent implementation
- PlaNet agent implementation
- JEPA self-supervised learning agent
- IRIS sample-efficient RL agent
- DiT (Diffusion Transformer) support
- DIAMOND diffusion world model for Atari
- Core environment backends (DMC, Gym, Atari, MuJoCo)
- Replay buffers (Dreamer, IRIS, PlaNet)
- VQ-VAE and ConvVAE vision components
- HuggingFace Hub checkpoint loading
- TensorBoard and WandB logging integration
