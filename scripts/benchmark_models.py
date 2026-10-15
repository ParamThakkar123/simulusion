"""Auto-run compute benchmark across every model implemented in Simulusion.

This is a *systems* benchmark: it builds each architecture, feeds it synthetic
tensors, and measures parameter count, forward latency, forward+backward
latency, throughput and peak memory. It needs no checkpoints, no datasets and
no environments, so it runs anywhere ``torch`` imports.

For *return*-based benchmarking of trained agents (IQM over seeds on Atari),
use ``python -m simulusion.benchmarks.cli`` instead -- that harness needs trained
checkpoints and a Gymnasium install.

The usual entrypoint is ``scripts/benchmark_models.sh``, which installs
everything with uv and runs this module through ``uv run``. Call this file
directly when you manage the environment yourself.

Usage::

    # the core tier at tiny scale, on the best available device
    python scripts/benchmark_models.py

    # a realistic scale on the GPU, including the expensive full-stack models
    python scripts/benchmark_models.py --preset small --all --device cuda

    # one family at the sizes the papers use
    python scripts/benchmark_models.py --family iris --preset paper

    # show what would run
    python scripts/benchmark_models.py --list

Results are printed as a table and written to ``--out-dir`` as JSON, CSV and
Markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Sequence

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Scale presets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """A scale at which every case is built and fed.

    ``width``/``depth``/``heads`` of ``None`` mean "use the library default",
    which for most models is the size its paper reports.
    """

    name: str
    batch: int
    seq: int
    image_size: int
    width: int | None
    depth: int | None
    heads: int | None


PRESETS: dict[str, Preset] = {
    # Fast enough to run the whole core tier on a laptop CPU.
    "tiny": Preset("tiny", batch=2, seq=4, image_size=64, width=128, depth=2, heads=4),
    # A realistic single-GPU training step.
    "small": Preset(
        "small", batch=8, seq=8, image_size=64, width=256, depth=4, heads=8
    ),
    # Library defaults, i.e. the published architectures. Large.
    "paper": Preset(
        "paper", batch=4, seq=16, image_size=64, width=None, depth=None, heads=None
    ),
}


# --------------------------------------------------------------------------
# Case plumbing
# --------------------------------------------------------------------------

InputFn = Callable[
    [torch.device, torch.dtype], "tuple[tuple[Any, ...], dict[str, Any]]"
]
CallFn = Callable[[nn.Module, "tuple[Any, ...]", "dict[str, Any]"], Any]


@dataclass
class Built:
    """A constructed model plus everything needed to call it once."""

    model: nn.Module
    make_inputs: InputFn
    call: CallFn | None = None

    def invoke(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if self.call is not None:
            return self.call(self.model, args, kwargs)
        return self.model(*args, **kwargs)


@dataclass(frozen=True)
class BenchCase:
    """One benchmarkable architecture."""

    name: str
    family: str
    build: Callable[[Preset], Built]
    tier: str = "core"
    note: str = ""


def _default(value: int | None, fallback: int) -> int:
    return fallback if value is None else value


# --------------------------------------------------------------------------
# Dreamer / PlaNet / World Models family
# --------------------------------------------------------------------------


def _build_dreamer_rssm(p: Preset) -> Built:
    from simulusion.models.dreamer_rssm import RSSM

    action_size, embed = 6, 1024
    size = _default(p.width, 200)
    model = RSSM(
        action_size=action_size,
        stoch_size=32,
        deter_size=size,
        hidden_size=size,
        obs_embed_size=embed,
        activation="elu",
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        # forward() flattens x[:, t] to (B, obs_embed_size), so feed embeddings.
        x = torch.randn(p.batch, p.seq + 1, embed, device=device, dtype=dtype)
        u = torch.randn(p.batch, p.seq, action_size, device=device, dtype=dtype)
        return (x, u), {}

    return Built(model, make_inputs)


def _build_planet_rssm(p: Preset) -> Built:
    from simulusion.models.rssm import RecurrentStateSpaceModel

    action_size = 6
    size = _default(p.width, 200)
    model = RecurrentStateSpaceModel(
        action_size=action_size,
        state_size=size,
        latent_size=30,
        hidden_size=size,
        embed_size=1024,
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        # The PlaNet CNN encoder is hard-wired to 64x64 frames.
        x = torch.randn(p.batch, p.seq + 1, 3, 64, 64, device=device, dtype=dtype)
        u = torch.randn(p.batch, p.seq, action_size, device=device, dtype=dtype)
        return (x, u), {}

    return Built(model, make_inputs)


def _call_observe_rollout(
    model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """ModularRSSM exposes its sequence pass as observe_rollout, not forward."""
    return model.observe_rollout(*args, **kwargs)


def _modular_rssm(backbone_type: str) -> Callable[[Preset], Built]:
    def build(p: Preset) -> Built:
        from simulusion.models.modular_rssm import create_modular_rssm

        action_size = 6
        size = _default(p.width, 200)
        model = create_modular_rssm(
            encoder_type="conv",
            decoder_type="conv",
            backbone_type=backbone_type,
            obs_shape=(3, 64, 64),
            action_size=action_size,
            stoch_size=32,
            deter_size=size,
            hidden_size=size,
            embed_size=1024,
        )

        def make_inputs(device: torch.device, dtype: torch.dtype):
            # Rollout tensors are time-major: (T, B, ...).
            obs = torch.randn(p.seq, p.batch, 3, 64, 64, device=device, dtype=dtype)
            actions = torch.randn(
                p.seq, p.batch, action_size, device=device, dtype=dtype
            )
            # ModularRSSM.observe_rollout expects per-step nonterms of (B,).
            nonterms = torch.ones(p.seq, p.batch, device=device, dtype=dtype)
            state = model.init_state(p.batch, device)
            return (obs, actions, nonterms, state, p.seq), {}

        return Built(model, make_inputs, call=_call_observe_rollout)

    return build


def _build_mdrnn(p: Preset) -> Built:
    from simulusion.models.mdrnn import MDRNN

    latents, actions = 32, 3
    model = MDRNN(
        latents=latents,
        actions=actions,
        hiddens=_default(p.width, 256),
        gaussians=5,
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        a = torch.randn(p.seq, p.batch, actions, device=device, dtype=dtype)
        z = torch.randn(p.seq, p.batch, latents, device=device, dtype=dtype)
        return (a, z), {}

    return Built(model, make_inputs)


def _build_convvae(p: Preset) -> Built:
    from simulusion.vision.VAE.ConvVAE import ConvVAE

    model = ConvVAE(img_channels=3, latent_size=32)

    def make_inputs(device: torch.device, dtype: torch.dtype):
        # The World Models ConvVAE is hard-wired to 64x64 frames.
        return (torch.randn(p.batch, 3, 64, 64, device=device, dtype=dtype),), {}

    return Built(model, make_inputs)


# --------------------------------------------------------------------------
# IRIS family
# --------------------------------------------------------------------------

_IRIS_VOCAB = 512
_IRIS_TOKENS_PER_FRAME = 16
_IRIS_ACTIONS = 18


def _iris_embedding_dim(p: Preset) -> int:
    return _default(p.width, 512)


def _build_iris_encoder(p: Preset) -> Built:
    from simulusion.vision.iris_encoder import IRISEncoder

    model = IRISEncoder(
        vocab_size=_IRIS_VOCAB,
        tokens_per_frame=_IRIS_TOKENS_PER_FRAME,
        embedding_dim=_iris_embedding_dim(p),
        base_channels=64,
        frame_shape=(3, 64, 64),
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        return (torch.randn(p.batch, 3, 64, 64, device=device, dtype=dtype),), {}

    return Built(model, make_inputs)


def _build_iris_decoder(p: Preset) -> Built:
    from simulusion.vision.iris_decoder import IRISDecoder

    embedding_dim = _iris_embedding_dim(p)
    model = IRISDecoder(
        vocab_size=_IRIS_VOCAB,
        embedding_dim=embedding_dim,
        base_channels=64,
        frame_shape=(3, 64, 64),
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        # 64x64 frames tokenize to a 4x4 grid (16 tokens per frame).
        z = torch.randn(p.batch, embedding_dim, 4, 4, device=device, dtype=dtype)
        return (z,), {}

    return Built(model, make_inputs)


def _build_iris_transformer(p: Preset) -> Built:
    from simulusion.models.iris_transformer import IRISTransformer

    model = IRISTransformer(
        vocab_size=_IRIS_VOCAB,
        tokens_per_frame=_IRIS_TOKENS_PER_FRAME,
        action_size=_IRIS_ACTIONS,
        embed_dim=_default(p.width, 256),
        num_layers=_default(p.depth, 10),
        num_heads=_default(p.heads, 4),
    )

    def make_inputs(device: torch.device, _dtype: torch.dtype):
        tokens = torch.randint(
            0,
            _IRIS_VOCAB,
            (p.batch, p.seq + 1, _IRIS_TOKENS_PER_FRAME),
            device=device,
        )
        actions = torch.randint(0, _IRIS_ACTIONS, (p.batch, p.seq), device=device)
        return (tokens, actions), {}

    return Built(model, make_inputs)


def _build_iris_world_model(p: Preset) -> Built:
    from simulusion.models.iris_transformer import IRISTransformer, IRISWorldModel
    from simulusion.vision.iris_decoder import IRISDecoder
    from simulusion.vision.iris_encoder import IRISEncoder

    embedding_dim = _iris_embedding_dim(p)
    model = IRISWorldModel(
        encoder=IRISEncoder(
            vocab_size=_IRIS_VOCAB,
            tokens_per_frame=_IRIS_TOKENS_PER_FRAME,
            embedding_dim=embedding_dim,
            base_channels=64,
            frame_shape=(3, 64, 64),
        ),
        decoder=IRISDecoder(
            vocab_size=_IRIS_VOCAB,
            embedding_dim=embedding_dim,
            base_channels=64,
            frame_shape=(3, 64, 64),
        ),
        transformer=IRISTransformer(
            vocab_size=_IRIS_VOCAB,
            tokens_per_frame=_IRIS_TOKENS_PER_FRAME,
            action_size=_IRIS_ACTIONS,
            embed_dim=_default(p.width, 256),
            num_layers=_default(p.depth, 10),
            num_heads=_default(p.heads, 4),
        ),
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        obs = torch.randn(p.batch, p.seq + 1, 3, 64, 64, device=device, dtype=dtype)
        actions = torch.randint(0, _IRIS_ACTIONS, (p.batch, p.seq), device=device)
        return (obs, actions), {}

    return Built(model, make_inputs)


# --------------------------------------------------------------------------
# DIAMOND family
# --------------------------------------------------------------------------

_DIAMOND_ACTIONS = 18
_DIAMOND_COND_FRAMES = 4


def _build_diamond_unet(p: Preset) -> Built:
    from simulusion.models.diffusion.diamond_diffusion import DiffusionUNet

    model = DiffusionUNet(
        obs_channels=3,
        num_conditioning_frames=_DIAMOND_COND_FRAMES,
        base_channels=_default(p.width, 64),
        channel_multipliers=(1, 1, 1, 1),
        num_res_blocks=_default(p.depth, 2),
        action_dim=_DIAMOND_ACTIONS,
    )
    size = p.image_size

    def make_inputs(device: torch.device, dtype: torch.dtype):
        x = torch.randn(p.batch, 3, size, size, device=device, dtype=dtype)
        t = torch.rand(p.batch, device=device, dtype=dtype)
        history = torch.randn(
            p.batch, _DIAMOND_COND_FRAMES, 3, size, size, device=device, dtype=dtype
        )
        actions = torch.randint(
            0, _DIAMOND_ACTIONS, (p.batch, _DIAMOND_COND_FRAMES), device=device
        )
        return (x, t, history, actions), {}

    return Built(model, make_inputs)


def _build_diamond_reward_termination(p: Preset) -> Built:
    from simulusion.models.diffusion.reward_termination import RewardTerminationModel

    channels = _default(p.width, 32)
    model = RewardTerminationModel(
        obs_channels=3,
        action_dim=_DIAMOND_ACTIONS,
        channels=(channels,) * 4,
        lstm_dim=512,
        res_blocks=_default(p.depth, 2),
        frame_size=p.image_size,
    )
    size = p.image_size

    def make_inputs(device: torch.device, dtype: torch.dtype):
        obs = torch.randn(p.batch, p.seq, 3, size, size, device=device, dtype=dtype)
        actions = torch.randint(0, _DIAMOND_ACTIONS, (p.batch, p.seq), device=device)
        return (obs, actions), {}

    return Built(model, make_inputs)


# --------------------------------------------------------------------------
# DiT
# --------------------------------------------------------------------------


def _build_dit(p: Preset) -> Built:
    from simulusion.models.diffusion.DiT import DiT

    size = p.image_size
    model = DiT(
        img_size=size,
        patch_size=8,
        in_channels=3,
        d_model=_default(p.width, 384),
        depth=_default(p.depth, 12),
        heads=_default(p.heads, 6),
        num_classes=0,
        learn_sigma=True,
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        x = torch.randn(p.batch, 3, size, size, device=device, dtype=dtype)
        t = torch.randint(0, 1000, (p.batch,), device=device)
        return (x, t), {}

    return Built(model, make_inputs)


# --------------------------------------------------------------------------
# Genie family
# --------------------------------------------------------------------------

_GENIE_VOCAB = 1024
_GENIE_ACTION_VOCAB = 8


def _build_genie_tokenizer(p: Preset) -> Built:
    from simulusion.vision.video_tokenizer import create_video_tokenizer

    size = p.image_size
    model = create_video_tokenizer(
        num_frames=p.seq,
        image_size=size,
        encoder_dim=_default(p.width, 512),
        decoder_dim=_default(p.width, 512) * 2,
        encoder_depth=_default(p.depth, 12),
        decoder_depth=_default(p.depth, 20),
        num_heads=_default(p.heads, 16),
        patch_size=8,
        vocab_size=_GENIE_VOCAB,
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        video = torch.randn(p.batch, 3, p.seq, size, size, device=device, dtype=dtype)
        return (video,), {}

    return Built(model, make_inputs)


def _build_genie_latent_action(p: Preset) -> Built:
    from simulusion.models.latent_action_model import create_latent_action_model

    size = p.image_size
    model = create_latent_action_model(
        num_frames=p.seq,
        image_size=size,
        encoder_dim=_default(p.width, 256),
        decoder_dim=_default(p.width, 512),
        encoder_depth=_default(p.depth, 4),
        decoder_depth=_default(p.depth, 4),
        num_heads=_default(p.heads, 8),
        patch_size=16,
        vocab_size=_GENIE_ACTION_VOCAB,
    )

    def make_inputs(device: torch.device, dtype: torch.dtype):
        prev = torch.randn(p.batch, 3, p.seq, size, size, device=device, dtype=dtype)
        nxt = torch.randn(p.batch, 3, size, size, device=device, dtype=dtype)
        return (prev, nxt), {}

    return Built(model, make_inputs)


def _build_genie_dynamics(p: Preset) -> Built:
    from simulusion.models.dynamics_model import create_dynamics_model

    size, patch = p.image_size, 8
    tokens_per_frame = (size // patch) ** 2
    model = create_dynamics_model(
        num_frames=p.seq,
        image_size=size,
        vocab_size=_GENIE_VOCAB,
        action_vocab_size=_GENIE_ACTION_VOCAB,
        dim=_default(p.width, 512),
        depth=_default(p.depth, 8),
        num_heads=_default(p.heads, 8),
        patch_size=patch,
    )

    def make_inputs(device: torch.device, _dtype: torch.dtype):
        tokens = torch.randint(
            0, _GENIE_VOCAB, (p.batch, p.seq, tokens_per_frame), device=device
        )
        actions = torch.randint(0, _GENIE_ACTION_VOCAB, (p.batch, p.seq), device=device)
        return (tokens, actions), {}

    return Built(model, make_inputs)


def _build_genie_small(p: Preset) -> Built:
    from simulusion.models.genie import create_genie_small

    size = p.image_size
    model = create_genie_small(num_frames=p.seq, image_size=size)

    def make_inputs(device: torch.device, dtype: torch.dtype):
        video = torch.randn(p.batch, 3, p.seq, size, size, device=device, dtype=dtype)
        return (video,), {}

    return Built(model, make_inputs)


# --------------------------------------------------------------------------
# JEPA family
# --------------------------------------------------------------------------


def _jepa_vit(factory_name: str) -> Callable[[Preset], Built]:
    def build(p: Preset) -> Built:
        from simulusion.models import vit as vit_module

        size = p.image_size
        model = getattr(vit_module, factory_name)(patch_size=8, img_size=[size])

        def make_inputs(device: torch.device, dtype: torch.dtype):
            frames = torch.randn(p.batch, 3, size, size, device=device, dtype=dtype)
            return (frames,), {}

        return Built(model, make_inputs)

    return build


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CASES: tuple[BenchCase, ...] = (
    BenchCase(
        "dreamer-rssm",
        "dreamer",
        _build_dreamer_rssm,
        note="RSSM rollout over pre-embedded observations",
    ),
    BenchCase(
        "planet-rssm",
        "planet",
        _build_planet_rssm,
        note="PlaNet RSSM including its CNN encoder (64x64 frames)",
    ),
    BenchCase(
        "modular-rssm-gru", "dreamer", _modular_rssm("gru"), note="observe_rollout"
    ),
    BenchCase(
        "modular-rssm-lstm", "dreamer", _modular_rssm("lstm"), note="observe_rollout"
    ),
    BenchCase(
        "modular-rssm-transformer",
        "dreamer",
        _modular_rssm("transformer"),
        note="observe_rollout",
    ),
    BenchCase("mdrnn", "world-models", _build_mdrnn, note="MDN-RNN sequence model"),
    BenchCase("convvae", "world-models", _build_convvae, note="World Models ConvVAE"),
    BenchCase("iris-encoder", "iris", _build_iris_encoder, note="VQ tokenizer encoder"),
    BenchCase("iris-decoder", "iris", _build_iris_decoder, note="VQ tokenizer decoder"),
    BenchCase(
        "iris-transformer", "iris", _build_iris_transformer, note="GPT world model"
    ),
    BenchCase(
        "iris-world-model",
        "iris",
        _build_iris_world_model,
        tier="heavy",
        note="encoder + transformer + per-step decode",
    ),
    BenchCase(
        "diamond-unet", "diamond", _build_diamond_unet, note="EDM denoising U-Net"
    ),
    BenchCase(
        "diamond-reward-termination",
        "diamond",
        _build_diamond_reward_termination,
        note="CNN + LSTM reward/end head",
    ),
    BenchCase("dit", "dit", _build_dit, note="Diffusion Transformer"),
    BenchCase(
        "genie-video-tokenizer", "genie", _build_genie_tokenizer, note="ST-ViViT VQ"
    ),
    BenchCase("genie-latent-action", "genie", _build_genie_latent_action, note="LAM"),
    BenchCase(
        "genie-dynamics", "genie", _build_genie_dynamics, note="MaskGIT ST-transformer"
    ),
    BenchCase(
        "genie-small",
        "genie",
        _build_genie_small,
        tier="heavy",
        note="full stack: tokenizer + LAM + dynamics",
    ),
    BenchCase("jepa-vit-tiny", "jepa", _jepa_vit("vit_tiny"), note="I-JEPA encoder"),
    BenchCase(
        "jepa-vit-small",
        "jepa",
        _jepa_vit("vit_small"),
        tier="heavy",
        note="I-JEPA encoder",
    ),
)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    family: str
    tier: str
    status: str
    params: int | None = None
    trainable_params: int | None = None
    fwd_ms_mean: float | None = None
    fwd_ms_median: float | None = None
    fwd_ms_p90: float | None = None
    fwd_ms_std: float | None = None
    bwd_ms_mean: float | None = None
    step_ms_mean: float | None = None
    samples_per_s: float | None = None
    peak_mem_mb: float | None = None
    batch: int | None = None
    iters: int | None = None
    note: str = ""
    error: str = ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, dtype: torch.dtype) -> ContextManager[Any]:
    if dtype is torch.float32:
        return nullcontext()
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _backward_target(output: Any) -> torch.Tensor | None:
    """Reduce an arbitrary model output to one scalar we can call backward on."""
    parts: list[torch.Tensor] = []

    def walk(obj: Any) -> None:
        if torch.is_tensor(obj):
            if obj.is_floating_point() and obj.requires_grad:
                parts.append(obj.float().mean())
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)

    walk(output)
    if not parts:
        return None
    return torch.stack(parts).sum()


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def run_case(
    case: BenchCase,
    preset: Preset,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    do_backward: bool,
    seed: int,
) -> Result:
    """Build one model, time it, and tear it down again."""
    result = Result(
        name=case.name,
        family=case.family,
        tier=case.tier,
        status="ok",
        batch=preset.batch,
        iters=iters,
        note=case.note,
    )

    torch.manual_seed(seed)
    built = case.build(preset)
    model = built.model.to(device)
    model.train(do_backward)

    result.params = sum(param.numel() for param in model.parameters())
    result.trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    # Inputs stay float32; reduced precision is applied through autocast so
    # models holding float32 buffers internally still run.
    args, kwargs = built.make_inputs(device, torch.float32)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    grad_ctx: ContextManager[Any] = (
        # no_grad rather than inference_mode: several models re-enable grad
        # internally (Genie's tokenizer phase), which inference tensors forbid.
        nullcontext() if do_backward else torch.no_grad()
    )

    with grad_ctx:
        for _ in range(warmup):
            with _autocast(device, dtype):
                output = built.invoke(args, kwargs)
            if do_backward:
                loss = _backward_target(output)
                if loss is None:
                    # Nothing differentiable came back (e.g. a pure sampler);
                    # fall back to timing the forward pass only.
                    do_backward = False
                    result.note = "; ".join(
                        part for part in (result.note, "forward-only output") if part
                    )
                    model.eval()
                    break
                model.zero_grad(set_to_none=True)
                loss.backward()
        _sync(device)

    fwd_times: list[float] = []
    bwd_times: list[float] = []

    with grad_ctx:
        for _ in range(iters):
            start = time.perf_counter()
            with _autocast(device, dtype):
                output = built.invoke(args, kwargs)
            _sync(device)
            mid = time.perf_counter()
            fwd_times.append((mid - start) * 1000.0)

            if do_backward:
                loss = _backward_target(output)
                if loss is None:
                    do_backward = False
                else:
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    _sync(device)
                    bwd_times.append((time.perf_counter() - mid) * 1000.0)
            del output

    result.fwd_ms_mean = statistics.fmean(fwd_times)
    result.fwd_ms_median = statistics.median(fwd_times)
    result.fwd_ms_p90 = _percentile(fwd_times, 0.9)
    result.fwd_ms_std = statistics.pstdev(fwd_times) if len(fwd_times) > 1 else 0.0
    result.samples_per_s = preset.batch / (result.fwd_ms_mean / 1000.0)
    if bwd_times:
        result.bwd_ms_mean = statistics.fmean(bwd_times)
        result.step_ms_mean = result.fwd_ms_mean + result.bwd_ms_mean

    if device.type == "cuda":
        result.peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    del model, built, args, kwargs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("name", "model"),
    ("family", "family"),
    ("params", "params"),
    ("fwd_ms_mean", "fwd ms"),
    ("bwd_ms_mean", "bwd ms"),
    ("step_ms_mean", "step ms"),
    ("samples_per_s", "samples/s"),
    ("peak_mem_mb", "peak MB"),
    ("status", "status"),
)


def _human_params(count: int | None) -> str:
    if count is None:
        return "-"
    if count >= 1_000_000_000:
        return f"{count / 1e9:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.1f}K"
    return str(count)


def _format(field_name: str, value: Any) -> str:
    if value is None:
        return "-"
    if field_name == "params":
        return _human_params(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_table(results: Iterable[Result]) -> None:
    rows = [[_format(key, getattr(r, key)) for key, _ in _COLUMNS] for r in results]
    headers = [header for _, header in _COLUMNS]
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows])
        for i in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    print(line(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def write_reports(
    results: list[Result], meta: dict[str, Any], out_dir: Path
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": [asdict(r) for r in results]}

    json_path = out_dir / "model_benchmarks.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "model_benchmarks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    md_path = out_dir / "model_benchmarks.md"
    lines = ["# Simulusion model benchmarks", ""]
    lines += [f"- **{key}**: {value}" for key, value in meta.items()]
    lines += ["", "| " + " | ".join(header for _, header in _COLUMNS) + " |"]
    lines.append("| " + " | ".join("---" for _ in _COLUMNS) + " |")
    for result in results:
        cells = [_format(key, getattr(result, key)) for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    failures = [r for r in results if r.status != "ok"]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{r.name}`: {r.error}" for r in failures]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "markdown": md_path}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_cases(args: argparse.Namespace) -> list[BenchCase]:
    cases = list(CASES)
    if args.models:
        wanted = {
            name.strip().lower() for name in args.models.split(",") if name.strip()
        }
        unknown = wanted - {case.name for case in cases}
        if unknown:
            raise SystemExit(
                f"Unknown model(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(case.name for case in cases)}"
            )
        return [case for case in cases if case.name in wanted]
    if args.family:
        families = {f.strip().lower() for f in args.family.split(",") if f.strip()}
        unknown = families - {case.family for case in cases}
        if unknown:
            raise SystemExit(
                f"Unknown family(ies): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted({case.family for case in cases}))}"
            )
        cases = [case for case in cases if case.family in families]
    if not args.all:
        cases = [case for case in cases if case.tier == "core"]
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        default="tiny",
        choices=sorted(PRESETS),
        help="Scale to build and feed every model at (default: tiny).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to run (overrides --family).",
    )
    parser.add_argument("--family", default="", help="Comma-separated families to run.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include the heavy tier (full IRIS/Genie stacks, larger ViTs).",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps."
    )
    parser.add_argument("--dtype", default="fp32", choices=sorted(_DTYPES))
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override the preset batch size."
    )
    parser.add_argument(
        "--seq-len", type=int, default=None, help="Override the preset sequence length."
    )
    parser.add_argument(
        "--image-size", type=int, default=None, help="Override the preset image size."
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Untimed iterations (default: 2)."
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="Timed iterations (default: 10)."
    )
    parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Measure inference only; skip the backward pass.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "model_benchmarks"),
        help="Where the JSON/CSV/Markdown reports are written.",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Print only; write no files."
    )
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument(
        "--fail-fast", action="store_true", help="Abort on the first failing model."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cases = select_cases(args)
    if args.list:
        for case in cases:
            print(f"{case.name:28s} {case.family:14s} {case.tier:6s} {case.note}")
        return 0
    if not cases:
        raise SystemExit("No models selected.")

    preset = PRESETS[args.preset]
    overrides: dict[str, Any] = {}
    if args.batch_size is not None:
        overrides["batch"] = args.batch_size
    if args.seq_len is not None:
        overrides["seq"] = args.seq_len
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if overrides:
        preset = Preset(**{**asdict(preset), **overrides})

    device = resolve_device(args.device)
    dtype = _DTYPES[args.dtype]
    do_backward = not args.no_backward

    meta: dict[str, Any] = {
        "preset": preset.name,
        "batch": preset.batch,
        "seq": preset.seq,
        "image_size": preset.image_size,
        "device": str(device),
        "dtype": args.dtype,
        "backward": do_backward,
        "warmup": args.warmup,
        "iters": args.iters,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(device)

    print("Simulusion model benchmark")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    print()

    results: list[Result] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name} ...", end=" ", flush=True)
        started = time.perf_counter()
        try:
            result = run_case(
                case,
                preset,
                device,
                dtype,
                warmup=args.warmup,
                iters=args.iters,
                do_backward=do_backward,
                seed=args.seed,
            )
            print(f"ok ({time.perf_counter() - started:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - one bad model must not stop the sweep
            if args.fail_fast:
                raise
            message = f"{type(exc).__name__}: {exc}".strip().splitlines()[0]
            result = Result(
                name=case.name,
                family=case.family,
                tier=case.tier,
                status="failed",
                batch=preset.batch,
                note=case.note,
                error=message[:400],
            )
            print("FAILED")
            traceback.print_exc(limit=3, file=sys.stderr)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(result)

    print()
    print_table(results)

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print()
        print(f"{len(failed)} model(s) failed:")
        for result in failed:
            print(f"  {result.name}: {result.error}")

    if not args.no_report:
        paths = write_reports(results, meta, Path(args.out_dir))
        print()
        for label, path in paths.items():
            print(f"{label}: {path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
