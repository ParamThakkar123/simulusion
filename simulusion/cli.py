"""Simulusion command-line interface.

The installed ``simulusion`` console script and ``python -m simulusion.cli`` both enter through this module. ``tools.cli`` remains as a backward-compatible wrapper.  The CLI intentionally keeps imports lightweight at
startup and only imports optional training, benchmark, dataset, or environment
packages inside the commands that need them.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("simulusion.cli")

# Keep this mapping cheap to import so ``simulusion models list`` and validation do
# not pull PyTorch or environment packages into every CLI process.
TRAINING_MODULES = {
    "diamond": "simulusion.training.train_diamond",
    "dit": "simulusion.training.train_dit",
    "dreamer": "simulusion.training.train_dreamer",
    "genie": "simulusion.training.train_genie",
    "iris": "simulusion.training.train_iris",
    "jepa": "simulusion.training.train_jepa",
    "planet": "simulusion.training.train_planet",
    "rssm": "simulusion.training.train_rssm",
    "world-model": "simulusion.training.train_world_model",
}

EVAL_MODULES = {
    "diamond": "simulusion.inference.eval_diamond",
    "jepa": "simulusion.training.eval_jepa",
}

# ``eval`` covers two different evaluations, so most of its options apply to
# only one model. Anything a model does not read is rejected rather than
# silently ignored.
EVAL_MODEL_OPTIONS = {
    "diamond": {"game", "num_videos", "trajectory_length", "metrics", "record", "seed"},
    "jepa": {
        "root_path",
        "dataset",
        "model_name",
        "patch_size",
        "crop_size",
        "weights",
        "train_folder",
        "val_folder",
        "epochs",
        "head_batch_size",
        "num_workers",
        "download",
    },
}

PLAY_MODULES = {
    "diamond": "simulusion.inference.play_diamond",
    "dreamer": "simulusion.inference.play_dreamer",
}

# `--game` means an Atari ROM for DIAMOND and a control task for Dreamer, so
# the default has to follow the model rather than the flag.
PLAY_DEFAULT_GAMES = {
    "diamond": "Breakout-v5",
    "dreamer": "walker-walk",
}

BENCHMARK_AGENT_NAMES = ("diamond", "iris", "dreamerv1", "dreamerv2")
BENCHMARK_ENV_BACKENDS = (
    "gym",
    "gymnasium",
    "dmc",
    "mujoco",
    "robotics",
    "bsuite",
    "brax",
    "unity_mlagents",
)


def _option_given(name: str) -> bool:
    """True when the user typed this option, rather than taking its default."""

    source = click.get_current_context().get_parameter_source(name)
    return source is not None and source.name == "COMMANDLINE"


def _reject_foreign_options(model: str) -> None:
    """Fail on options that belong to a different ``eval`` model."""

    mine = EVAL_MODEL_OPTIONS.get(model, set())
    foreign = set().union(*EVAL_MODEL_OPTIONS.values()) - mine
    passed = sorted(name for name in foreign if _option_given(name))
    if passed:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in passed)
        raise click.UsageError(f"{flags} cannot be used with --model {model}")


def _echo_error(message: str) -> None:
    """Print user-facing errors to stdout for stable CLI/test output."""
    click.echo(message)


def _parse_benchmark_seeds(value: str) -> list[int]:
    """Parse ``--seeds`` values as either N, a comma list, or one integer."""
    value = value.strip()
    if not value:
        raise ValueError("--seeds cannot be empty")
    if "," in value:
        seeds = [item.strip() for item in value.split(",") if item.strip()]
        if not seeds:
            raise ValueError("--seeds must contain at least one integer")
        return [int(item) for item in seeds]
    if value.isdigit():
        return list(range(int(value)))
    return [int(value)]


def _parse_checkpoint_map(
    values: tuple[str, ...] | list[str] | None,
) -> dict[str, str]:
    """Parse repeated ``--checkpoint-map AGENT=PATH`` options."""
    checkpoints: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(
                f"Invalid checkpoint map '{value}'. Expected AGENT=PATH, for example iris=checkpoints/iris.pt."
            )
        agent, path = value.split("=", 1)
        agent = agent.strip().lower()
        path = path.strip()
        if not agent or not path:
            raise ValueError(
                f"Invalid checkpoint map '{value}'. Agent and path must both be non-empty."
            )
        checkpoints[agent] = path
    return checkpoints


def _ensure_gym() -> Any:
    """Lazy-import gym, falling back to gymnasium."""
    try:
        return importlib.import_module("gym")
    except Exception:
        return importlib.import_module("gymnasium")


def _ensure_numpy() -> Any | None:
    """Lazy-import numpy and return ``None`` when it is unavailable."""
    try:
        return importlib.import_module("numpy")
    except Exception:
        return None


def _load_catalog() -> dict[str, Any]:
    """Lazy-load the environment catalog used by ``simulusion envs list``."""
    try:
        from simulusion.catalog import ENV_BACKENDS
    except Exception:
        logger.debug("Could not import simulusion.catalog", exc_info=True)
        return {}
    return ENV_BACKENDS


def _load_benchmark_runtime() -> tuple[Any, Any, Any, Any]:
    """Lazy-load benchmark classes after lightweight CLI validation passes."""
    from simulusion.benchmarks.cli import AGENTS
    from simulusion.benchmarks.runner import (
        BenchmarkRunner,
        MultiAgentBenchmarkRunner,
    )
    import torch

    return AGENTS, BenchmarkRunner, MultiAgentBenchmarkRunner, torch


@click.group(
    name="simulusion",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def app(ctx: click.Context) -> None:
    """Simulusion command-line tool."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@app.command("version")
def version() -> None:
    """Show the installed Simulusion package version."""
    try:
        import simulusion as pkg
    except Exception:
        click.echo("simulusion (unknown version)")
        return
    click.echo(getattr(pkg, "__version__", "simulusion (unknown version)"))


@app.group("envs")
def envs_app() -> None:
    """Environment discovery commands."""


@envs_app.command("list")
def envs_list() -> None:
    """List built-in environment backends and example environment ids."""
    catalog = _load_catalog()
    if not catalog:
        click.echo(
            "No environment catalog available (simulusion.catalog failed to import)."
        )
        return

    for backend, info in catalog.items():
        click.echo(f"{backend}: {info.get('label', '')}")
        items = info.get("environments", [])
        for env in items[:20]:
            click.echo(f"  - {env}")
        if len(items) > 20:
            click.echo(f"  ... and {len(items) - 20} more")


@app.group("datasets")
def datasets_app() -> None:
    """Dataset inspection and conversion commands."""


@datasets_app.command("list")
@click.argument("path", required=False, type=click.Path(path_type=Path))
def datasets_list(path: Path | None = None) -> None:
    """List dataset entries under PATH, SIMULUSION_HOME, or ~/.simulusion."""
    home = Path(os.environ.get("SIMULUSION_HOME", Path.home() / ".simulusion"))
    root = path or home
    try:
        has_entries = any(root.iterdir()) if root.exists() else False
    except PermissionError:
        has_entries = False

    if not root.exists() or not has_entries:
        click.echo(f"No datasets found at {root}")
        return

    click.echo(f"Datasets under: {root}")
    for entry in sorted(root.iterdir()):
        click.echo(f"- {entry.name}")


@datasets_app.command("convert")
@click.argument("src", type=click.Path(path_type=Path))
@click.option(
    "--dest-format",
    default="video",
    show_default=True,
    help="Destination format. Only 'video' is currently supported.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    help="Output directory for converted data.",
)
def datasets_convert(src: Path, dest_format: str, out_dir: Path | None) -> None:
    """Convert a simple HDF5/NumPy dataset into MP4 videos."""
    if dest_format != "video":
        _echo_error("Only 'video' dest_format is supported.")
        raise click.exceptions.Exit(1)

    srcp = src
    if not srcp.exists():
        _echo_error(f"Source not found: {srcp}")
        raise click.exceptions.Exit(1)

    numpy = _ensure_numpy()
    if numpy is None:
        _echo_error(
            "Install numpy and the dataset/video optional dependencies to use conversion."
        )
        raise click.exceptions.Exit(1)

    from simulusion.datasets.video_datasets import HDF5Dataset, NumPyDataset
    from simulusion.utils.utils import save_video

    out = out_dir or (Path.cwd() / "converted_datasets")
    out.mkdir(parents=True, exist_ok=True)

    dataset: HDF5Dataset | NumPyDataset
    if srcp.suffix in {".h5", ".hdf5"}:
        dataset = HDF5Dataset(str(srcp), num_frames=16, image_size=64)
    elif srcp.suffix in {".npz", ".npy"}:
        dataset = NumPyDataset(str(srcp), num_frames=16, image_size=64)
    else:
        _echo_error(
            "Unsupported source format for conversion. Provide .h5, .hdf5, .npz, or .npy."
        )
        raise click.exceptions.Exit(1)

    click.echo(f"Converting {len(dataset)} items from {srcp} -> {out} ...")
    for index in range(len(dataset)):
        item = dataset[index]
        arr = item.numpy() if hasattr(item, "numpy") else item
        if getattr(arr, "ndim", None) == 4 and arr.shape[1] in (1, 3, 4):
            arr = numpy.moveaxis(arr, 1, -1)
        if getattr(arr, "dtype", None) is not None and str(arr.dtype).endswith("uint8"):
            arr = arr.astype("float32") / 255.0  # type: ignore[union-attr]
        elif hasattr(arr, "max") and float(arr.max()) > 1.0:
            arr = arr.astype("float32") / 255.0  # type: ignore[union-attr]
        elif hasattr(arr, "astype"):
            arr = arr.astype("float32")

        out_name = out / f"ep_{index:06d}"
        save_video(arr, str(out_name.parent), out_name.name)
        click.echo(f"Wrote: {out_name}.mp4")


@app.group("models")
def models_app() -> None:
    """Model discovery commands."""


@models_app.command("list")
def models_list() -> None:
    """List supported training entrypoints and exported model names."""
    click.echo("Training entrypoints:")
    for name, module in sorted(TRAINING_MODULES.items()):
        click.echo(f"- {name}: {module}")

    try:
        import simulusion.models as wm_models
    except Exception:
        return

    exported = getattr(wm_models, "__all__", None)
    if exported:
        click.echo("\nExported model names:")
        for name in sorted(exported):
            click.echo(f"- {name}")


@app.command("benchmark")
@click.option(
    "--agent",
    "agent",
    "-a",
    help="Agent adapter to evaluate (for example: iris, diamond, dreamerv1, dreamerv2).",
)
@click.option(
    "--all-agents",
    is_flag=True,
    help="Run all registered benchmark adapters on the same environment.",
)
@click.option(
    "--game",
    "game",
    "-g",
    required=True,
    help="Environment id to benchmark, such as ALE/Pong-v5 or the BSuite id catch/0.",
)
@click.option(
    "--env-backend",
    type=click.Choice(BENCHMARK_ENV_BACKENDS, case_sensitive=False),
    help="Environment backend for Dreamer-compatible adapters, for example gym or bsuite.",
)
@click.option(
    "--seeds",
    default="1",
    show_default=True,
    help="Either N (creates seeds 0..N-1) or a comma-separated seed list.",
)
@click.option(
    "--episodes",
    "episodes",
    "-n",
    default=5,
    show_default=True,
    type=int,
    help="Evaluation episodes to run per seed.",
)
@click.option(
    "--checkpoint",
    "checkpoint",
    "-c",
    type=click.Path(path_type=Path),
    help="Checkpoint path for a single-agent benchmark.",
)
@click.option(
    "--checkpoint-map",
    "checkpoint_map",
    multiple=True,
    help="For --all-agents, repeat as AGENT=PATH (for example --checkpoint-map iris=ckpt.pt).",
)
@click.option(
    "--out-dir",
    "out_dir",
    default=Path("results/bench"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory where benchmark reports are written.",
)
@click.option(
    "--out_dir",
    "out_dir_alias",
    type=click.Path(path_type=Path),
    help="Alias for --out-dir.",
)
@click.option(
    "--device",
    help="Device passed to adapters. Defaults to CUDA when available, otherwise CPU.",
)
@click.option(
    "--preset",
    help="Optional adapter/model preset to forward to benchmark adapters.",
)
@click.option(
    "--train-epochs",
    type=int,
    help="For --all-agents only: train agents first for this many epochs if no checkpoints are supplied.",
)
def benchmark(
    agent: str | None,
    all_agents: bool,
    game: str,
    env_backend: str | None,
    seeds: str,
    episodes: int,
    checkpoint: Path | None,
    checkpoint_map: tuple[str, ...],
    out_dir: Path,
    out_dir_alias: Path | None,
    device: str | None,
    preset: str | None,
    train_epochs: int | None,
) -> None:
    """Run Simulusion benchmark evaluations and write report artifacts."""
    try:
        if not all_agents and not agent:
            _echo_error("Either --agent or --all-agents must be specified")
            raise click.exceptions.Exit(1)
        if all_agents and agent:
            _echo_error("--agent and --all-agents cannot be used together")
            raise click.exceptions.Exit(1)

        if agent:
            agent = agent.strip().lower()
            if agent not in BENCHMARK_AGENT_NAMES:
                _echo_error(
                    f"Unknown benchmark agent '{agent}'. Known: {', '.join(sorted(BENCHMARK_AGENT_NAMES))}"
                )
                raise click.exceptions.Exit(1)
            if checkpoint is None:
                _echo_error(
                    "--checkpoint is required when using --agent. Only trained models should be benchmarked."
                )
                raise click.exceptions.Exit(1)

        seed_values = _parse_benchmark_seeds(seeds)
        out_dir = out_dir_alias or out_dir
        checkpoints: dict[str, str] = {}
        if all_agents:
            checkpoints = _parse_checkpoint_map(checkpoint_map)
            unknown = sorted(set(checkpoints).difference(BENCHMARK_AGENT_NAMES))
            if unknown:
                _echo_error(
                    f"Unknown benchmark checkpoint agent(s): {', '.join(unknown)}. Known: {', '.join(sorted(BENCHMARK_AGENT_NAMES))}"
                )
                raise click.exceptions.Exit(1)
            if not checkpoints and train_epochs is None:
                _echo_error(
                    "Provide --checkpoint-map AGENT=PATH at least once, or provide --train-epochs for --all-agents."
                )
                raise click.exceptions.Exit(1)

        agents, runner_cls, multi_runner_cls, torch = _load_benchmark_runtime()
        if not device:
            from simulusion.utils.device import default_device_name

            device = default_device_name()
        extra_kwargs = {"device": device, "preset": preset}
        env_spec = {"game": game}
        if env_backend:
            env_spec["env_backend"] = env_backend.lower()

        if all_agents:
            runner = multi_runner_cls(
                adapters=list(agents.values()), out_dir=str(out_dir)
            )
            runner.run_all(
                env_spec=env_spec,
                seeds=seed_values,
                num_episodes=episodes,
                checkpoints=checkpoints or None,
                extra_kwargs=extra_kwargs,
                train_epochs=train_epochs,
            )
            click.echo(f"Multi-agent benchmark finished. Results written to: {out_dir}")
            return

        assert agent is not None
        assert checkpoint is not None
        runner = runner_cls(adapter_cls=agents[agent], out_dir=str(out_dir))
        runner.run(
            env_spec=env_spec,
            seeds=seed_values,
            num_episodes=episodes,
            checkpoint=str(checkpoint),
            extra_kwargs=extra_kwargs,
        )
        click.echo(f"Benchmark finished. Results written to: {out_dir}")
    except ValueError as exc:
        _echo_error(str(exc))
        raise click.exceptions.Exit(1) from exc
    except KeyboardInterrupt:
        _echo_error("Benchmark interrupted by user")
        raise click.exceptions.Exit(1)


@app.group("deploy")
def deploy_app() -> None:
    """Inspect and benchmark deployment bundles (see simulusion.inference)."""


def _open_bundle(path: Path) -> Any:
    from simulusion.inference.bundle import load_bundle

    try:
        return load_bundle(path)
    except (FileNotFoundError, ValueError) as exc:
        _echo_error(str(exc))
        raise click.exceptions.Exit(1)


@deploy_app.command("inspect")
@click.argument("bundle", type=click.Path(path_type=Path))
def deploy_inspect(bundle: Path) -> None:
    """Print a bundle's manifest."""
    loaded = _open_bundle(bundle)
    click.echo(json.dumps(loaded.manifest, indent=2))


def _zeros_from_spec(spec: Any, device: str) -> Any:
    import torch

    if isinstance(spec, dict) and "shape" in spec and "dtype" in spec:
        return torch.zeros(
            spec["shape"], dtype=getattr(torch, spec["dtype"]), device=device
        )
    if isinstance(spec, list):
        return tuple(_zeros_from_spec(s, device) for s in spec)
    raise click.ClickException(f"Cannot build an input from spec {spec!r}.")


@deploy_app.command("bench")
@click.argument("bundle", type=click.Path(path_type=Path))
@click.option("--format", "fmt", default=None, help="Artifact format (default: first).")
@click.option("--device", default="cpu", show_default=True)
@click.option("--iterations", default=100, show_default=True, type=int)
@click.option("--warmup", default=10, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
def deploy_bench(
    bundle: Path,
    fmt: str | None,
    device: str,
    iterations: int,
    warmup: int,
    as_json: bool,
) -> None:
    """Time a bundle's exported artifact on zero inputs of its recorded shapes."""
    from simulusion.inference.benchmark import benchmark_step

    loaded = _open_bundle(bundle)
    spec = loaded.manifest.get("inputs")
    if spec is None:
        raise click.ClickException("The bundle records no input spec.")
    inputs = _zeros_from_spec(spec, device)
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    try:
        runner = loaded.load_artifact(fmt, device=device)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc))
    first = args[0] if args else None
    batch = int(first.shape[0]) if first is not None and first.dim() else 1
    report = benchmark_step(
        runner,
        *args,
        warmup=warmup,
        iterations=iterations,
        batch_size=batch,
        device=device,
        name=f"{bundle.name}:{fmt or loaded.formats[0]}",
    )
    click.echo(json.dumps(report.as_dict(), indent=2) if as_json else report.summary())


@app.command("collect")
@click.option("--env", "env", required=True, help="Environment id (Gym/Atari).")
@click.option(
    "--steps",
    default=1000,
    show_default=True,
    type=int,
    help="Number of environment steps to collect.",
)
@click.option(
    "--out",
    default=Path("./collected.npz"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output file path (.npz).",
)
@click.option(
    "--random-policy/--no-random-policy",
    default=True,
    show_default=True,
    help="Use random actions.",
)
def collect(env: str, steps: int, out: Path, random_policy: bool) -> None:
    """Collect environment interactions and save observations/actions/rewards/dones."""
    numpy = _ensure_numpy()
    if numpy is None:
        _echo_error("Please install numpy to use collect")
        raise click.exceptions.Exit(1)
    try:
        gym = _ensure_gym()
    except Exception:
        _echo_error("Please install gym or gymnasium to use collect")
        raise click.exceptions.Exit(1)

    click.echo(f"Creating environment: {env}")
    try:
        try:
            from simulusion.envs import make_env

            env_obj = make_env(env)
        except Exception:
            env_obj = gym.make(env)
    except Exception as exc:
        logger.debug("Failed to create env %s: %s", env, exc, exc_info=True)
        _echo_error(f"Failed to create environment: {exc}")
        raise click.exceptions.Exit(1) from exc

    observations: list[Any] = []
    actions: list[Any] = []
    rewards: list[float] = []
    dones: list[bool] = []

    obs = env_obj.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for _ in range(steps):
        action = (
            env_obj.action_space.sample()
            if random_policy
            else numpy.zeros_like(env_obj.action_space.sample())
        )
        result = env_obj.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            next_obs, reward, terminated, truncated, _info = result
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, _info = result
            done = bool(done)
        observations.append(numpy.asarray(obs))
        actions.append(numpy.asarray(action))
        rewards.append(float(reward))
        dones.append(done)
        if done:
            obs = env_obj.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
        else:
            obs = next_obs

    click.echo(f"Collected {len(observations)} steps; saving to {out}")
    numpy.savez_compressed(
        str(out),
        observations=numpy.stack(observations),
        actions=numpy.stack(actions),
        rewards=numpy.array(rewards),
        dones=numpy.array(dones),
    )
    click.echo("Saved.")


@app.command("eval")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(list(EVAL_MODULES), case_sensitive=False),
    help="Model type to evaluate",
)
@click.option(
    "--checkpoint", "-c", required=True, help="Path to model checkpoint (.pt file)"
)
@click.option("--game", "-g", default="Breakout-v5", help="Environment/game name")
@click.option(
    "--num-videos", default=256, type=int, help="Number of trajectories to compare"
)
@click.option("--trajectory-length", default=20, type=int, help="Frames per trajectory")
@click.option("--batch-size", default=16, type=int, help="Batch size for generation")
@click.option(
    "--device", default=None, help="Device to run on (default: cuda if available)"
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to save results JSON",
)
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option(
    "--metrics",
    default="fid,fvd,lpips",
    help="Comma-separated metrics to compute (fid,fvd,lpips,psnr)",
)
@click.option(
    "--record",
    default=None,
    type=click.Path(path_type=Path),
    help="Save real + generated videos (e.g. eval.mp4)",
)
@click.option(
    "--root-path",
    default=None,
    help="[jepa] Dataset root holding the probe's train/ and val/ splits",
)
@click.option(
    "--dataset", default="imagenet", help="[jepa] imagenet | cifar10 | imagefolder"
)
@click.option(
    "--model-name",
    default="vit_base",
    help="[jepa] Backbone the checkpoint was trained with",
)
@click.option("--patch-size", default=16, type=int, help="[jepa] ViT patch size")
@click.option(
    "--crop-size", default=224, type=int, help="[jepa] Probe input resolution"
)
@click.option(
    "--weights",
    default="target_encoder",
    type=click.Choice(["target_encoder", "encoder"]),
    help="[jepa] Which encoder to freeze; the paper evaluates the EMA target-encoder",
)
@click.option("--train-folder", default="train", help="[jepa] Train split folder name")
@click.option("--val-folder", default="val", help="[jepa] Val split folder name")
@click.option("--epochs", default=50, type=int, help="[jepa] Linear-head epochs")
@click.option(
    "--head-batch-size", default=16384, type=int, help="[jepa] Linear-head batch size"
)
@click.option("--num-workers", default=8, type=int, help="[jepa] Dataloader workers")
@click.option("--download", is_flag=True, help="[jepa] Download the dataset if missing")
def eval_command(
    model: str,
    checkpoint: str,
    game: str,
    num_videos: int,
    trajectory_length: int,
    batch_size: int,
    device: str | None,
    output: Path | None,
    seed: int,
    metrics: str,
    record: Path | None,
    root_path: str | None,
    dataset: str,
    model_name: str,
    patch_size: int,
    crop_size: int,
    weights: str,
    train_folder: str,
    val_folder: str,
    epochs: int,
    head_batch_size: int,
    num_workers: int,
    download: bool,
) -> None:
    """Evaluate a trained world model.

    Generative world models (DIAMOND) are scored with FID/FVD/LPIPS against
    real trajectories. JEPA has no rollout to score, so it is evaluated with
    the paper's frozen-encoder linear probe instead; that path needs
    ``--root-path`` rather than ``--game``.
    """
    module_path = EVAL_MODULES[model.lower()]
    mod = importlib.import_module(module_path)

    # Each model uses a disjoint slice of the options above; refuse the ones
    # that would otherwise be silently dropped.
    _reject_foreign_options(model.lower())

    if model.lower() == "jepa":
        if not root_path:
            raise click.UsageError("--root-path is required when evaluating jepa")
        # --batch-size defaults to a DIAMOND generation batch; feature
        # extraction through a frozen encoder can take a much larger one.
        if not _option_given("batch_size"):
            batch_size = 256
        results = mod.jepa_linear_probe(
            checkpoint=checkpoint,
            root_path=root_path,
            dataset=dataset,
            model_name=model_name,
            patch_size=patch_size,
            crop_size=crop_size,
            weights=weights,
            train_folder=train_folder,
            val_folder=val_folder,
            download=download,
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            head_batch_size=head_batch_size,
            device=device,
        )
        click.echo(f"Best linear-probe top-1: {results['top1']:.2f}%")
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            click.echo(f"Wrote {output}")
        return

    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    mod.run_eval(
        checkpoint=checkpoint,
        game=game,
        num_videos=num_videos,
        trajectory_length=trajectory_length,
        batch_size=batch_size,
        device=device,
        output=str(output) if output else None,
        seed=seed,
        metrics=metrics_list,
        record=str(record) if record else None,
    )


PLAY_MODEL_NAMES = tuple(sorted(PLAY_MODULES))


@app.command("play")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(PLAY_MODEL_NAMES, case_sensitive=False),
    help="Model type to play",
)
@click.option(
    "--checkpoint", "-c", required=True, help="Path to model checkpoint (.pt file)"
)
@click.option(
    "--game",
    "-g",
    default=None,
    help="Environment/game name (default: Breakout-v5 for diamond, walker-walk for dreamer)",
)
@click.option(
    "--device", default=None, help="Device to run on (default: cuda if available)"
)
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option(
    "--deterministic/--stochastic",
    default=True,
    help="Use deterministic policy actions",
)
@click.option(
    "--record",
    default=None,
    type=click.Path(path_type=Path),
    help="Save gameplay video (e.g. gameplay.mp4)",
)
@click.option("--record-fps", default=20, type=int, help="FPS for recorded video")
def play_command(
    model: str,
    checkpoint: str,
    game: str | None,
    device: str | None,
    seed: int,
    deterministic: bool,
    record: Path | None,
    record_fps: int,
) -> None:
    """Interactively play inside a trained world model (real env + dream mode)."""
    model_key = model.lower()
    module_path = PLAY_MODULES[model_key]
    mod = importlib.import_module(module_path)
    mod.run_play(
        checkpoint=checkpoint,
        game=game or PLAY_DEFAULT_GAMES[model_key],
        device=device,
        seed=seed,
        deterministic=deterministic,
        record=str(record) if record else None,
        record_fps=record_fps,
    )


@app.command(
    "train",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("model", required=False)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--legacy",
    is_flag=True,
    help="Run the family's own pre-Trainer entrypoint instead of the shared Trainer.",
)
@click.option(
    "--inproc",
    is_flag=True,
    help="With --legacy: run in-process instead of spawning a subprocess.",
)
def train(
    model: str | None, extra_args: tuple[str, ...], legacy: bool, inproc: bool
) -> None:
    """Train a model on the shared Trainer from a run config and overrides.

    MODEL is a training task (see `simulusion train --help-tasks` or the
    list below); it may be omitted when the config names its task.

    Examples:
        simulusion train genie data.path=data/clips trainer.max_steps=2000
        simulusion train --config simulusion/configs/runs/dreamer.yaml trainer.precision=bf16
        simulusion train --config runs/genie_20261020/config.yaml   # repeat a run
        simulusion train iris --legacy --config simulusion/configs/experiments/iris.yaml
    """
    from simulusion.training.task import list_tasks

    args = list(extra_args)
    if model is not None and (model.startswith("-") or "=" in model):
        args.insert(0, model)
        model = None
    if "--help-tasks" in args:
        click.echo("Training tasks: " + ", ".join(list_tasks()))
        return
    key = model.strip().lower() if model else None
    tasks = list_tasks()

    if key is not None and (legacy or key not in tasks):
        if key not in TRAINING_MODULES:
            _echo_error(
                f"Unknown model '{model}'. Training tasks: {', '.join(tasks)}. "
                f"Legacy entrypoints (--legacy): {', '.join(TRAINING_MODULES)}"
            )
            raise click.exceptions.Exit(1)
        _run_legacy_training(TRAINING_MODULES[key], args, inproc)
        return

    from simulusion.configs.structured import ConfigError
    from simulusion.training import run as run_module

    try:
        run_module.main(([key] if key else []) + args)
    except ConfigError as exc:
        _echo_error(str(exc))
        raise click.exceptions.Exit(1)
    except KeyboardInterrupt:
        _echo_error("Training interrupted by user")
        raise click.exceptions.Exit(1)


def _run_legacy_training(module: str, extra_args: list[str], inproc: bool) -> None:
    """Launch a family's own training entrypoint (the pre-Trainer path)."""
    if inproc:
        # Failures inside training propagate as a non-zero exit. The previous
        # version retried `main()` without the user's arguments on any
        # TypeError and re-ran the whole job in a subprocess on any other
        # exception, so a crash silently started a second, different run.
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            _echo_error(f"Could not import {module}: {exc}")
            raise click.exceptions.Exit(1)
        main_fn = getattr(mod, "main", None)
        if callable(main_fn):
            click.echo(f"Running in-process: {module}.main()")
            try:
                _call_training_main(main_fn, module, list(extra_args))
            except KeyboardInterrupt:
                _echo_error("Training interrupted by user")
                raise click.exceptions.Exit(1)
            return
        click.echo(
            f"Module {module} has no callable main(); falling back to subprocess"
        )

    cmd = [sys.executable, "-m", module, *extra_args]
    click.echo(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, check=False, shell=False
        )  # shell=False prevents injection
    except KeyboardInterrupt:
        _echo_error("Training interrupted by user")
        raise click.exceptions.Exit(1)
    raise click.exceptions.Exit(proc.returncode)


def _call_training_main(main_fn: Any, module: str, argv: list[str]) -> Any:
    """Call a training ``main`` with ``argv``, whichever way it accepts it.

    Entry points either take ``argv`` as their first positional parameter or
    read ``sys.argv`` themselves via argparse. The choice is made from the
    signature up front, never by catching errors raised during training.
    """
    import inspect

    positional = [
        param
        for param in inspect.signature(main_fn).parameters.values()
        if param.kind
        in (
            param.POSITIONAL_ONLY,
            param.POSITIONAL_OR_KEYWORD,
            param.VAR_POSITIONAL,
        )
    ]
    if positional:
        return main_fn(argv)
    saved_argv = sys.argv
    sys.argv = [module, *argv]
    try:
        return main_fn()
    finally:
        sys.argv = saved_argv


def main(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible callable for imports that use ``simulusion.cli:main``."""
    return app.main(*args, **kwargs)


def run() -> None:
    """Console-script entrypoint used by the installed ``simulusion`` command."""
    app(prog_name="simulusion")


def run_train() -> None:
    """Console-script entrypoint used by the installed ``simulusion-train`` command.

    This is equivalent to ``simulusion train [MODEL] [ARGS...]``: registered
    training tasks run on the shared Trainer, and ``--legacy`` reaches the
    entrypoints in ``TRAINING_MODULES``.
    """
    train.main(prog_name="simulusion-train")


if __name__ == "__main__":
    run()
