"""Export utilities for production deployment.

The public entry points are :func:`export_model` / :func:`export_any` and the
``obj.export(path, format=...)`` method. Supported formats:

* ``"exported_program"`` - ``torch.export`` graph saved as ``.pt2``. The
  recommended format: loadable without the model's source code, and the input
  to AOTInductor, ExecuTorch and TensorRT.
* ``"aoti"`` - AOTInductor package (``.pt2``): an ahead-of-time compiled shared
  library runnable from Python or C++ without the model's Python code.
* ``"onnx"`` - for ONNX Runtime / TensorRT / edge runtimes.
* ``"tensorrt"`` - Torch-TensorRT, compiled through the dynamo IR by default.
* ``"torchscript"`` - legacy; TorchScript is in maintenance mode upstream.

:func:`load_exported` loads any of these back and :func:`verify_export` checks
an artifact against its eager module.

Simulusion's public models and agents get ``.export()`` from
:class:`ExportableAgentMixin`. Any other ``nn.Module`` goes through
:func:`export_model`; Simulusion does not modify ``torch.nn.Module``.
"""

from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn as nn

ExportFormat = Literal["onnx", "torchscript", "tensorrt", "exported_program", "aoti"]

_FORMAT_ALIASES = {
    "exported-program": "exported_program",
    "exportedprogram": "exported_program",
    "export": "exported_program",
    "torch.export": "exported_program",
    "ep": "exported_program",
    "pt2": "exported_program",
    "aoti": "aoti",
    "aotinductor": "aoti",
    "aot-inductor": "aoti",
    "onnx": "onnx",
    "torchscript": "torchscript",
    "torch-script": "torchscript",
    "script": "torchscript",
    "jit": "torchscript",
    "ts": "torchscript",
    "pt": "torchscript",
    "tensorrt": "tensorrt",
    "tensor-rt": "tensorrt",
    "trt": "tensorrt",
}

_PREFERRED_TARGET_SUFFIXES = (
    "actor",
    "policy",
    "actor_critic",
    "rssm",
    "model",
    "world_model",
    "encoder",
)


def _normalize_format(format: str) -> ExportFormat:
    try:
        return _FORMAT_ALIASES[format.strip().lower().replace("_", "-")]  # type: ignore[return-value]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_FORMAT_ALIASES.values())))
        raise ValueError(
            f"Unsupported export format {format!r}. Use one of: {supported}."
        ) from exc


def _as_path(path: str | Path) -> Path:
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    return export_path


def _inputs_to_args(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _resolve_attr_path(obj: Any, target: str) -> Any:
    current = obj
    for part in target.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _discover_modules(obj: Any) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    seen: set[int] = set()

    def visit(value: Any, prefix: str) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, nn.Module):
            modules[prefix] = value
            for name, child in value.named_children():
                visit(child, f"{prefix}.{name}" if prefix else name)
            return
        if isinstance(value, dict):
            for name, child in value.items():
                if isinstance(name, str):
                    visit(child, f"{prefix}.{name}" if prefix else name)
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                visit(child, f"{prefix}.{idx}" if prefix else str(idx))
            return
        for name, child in vars(value).items() if hasattr(value, "__dict__") else []:
            if name.startswith("_"):
                continue
            if isinstance(child, (nn.Module, dict, list, tuple)) or hasattr(
                child, "__dict__"
            ):
                visit(child, f"{prefix}.{name}" if prefix else name)

    visit(obj, "")
    return {name: module for name, module in modules.items() if name}


class DreamerPolicyExport(nn.Module):
    """Traceable Dreamer policy head used by the generic export resolver."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor(features, deter=True)


class IRISActorCriticExport(nn.Module):
    """Traceable IRIS policy/value head used by the generic export resolver."""

    def __init__(self, agent: Any):
        super().__init__()
        self.agent = agent

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_logits, values, _ = self.agent.forward_actor_critic(frames)
        return action_logits, values


def _dreamer_default(obj: Any, target: str | None) -> nn.Module | None:
    dreamer = getattr(obj, "dreamer", None)
    if dreamer is None:
        return None
    if target in {None, "actor", "dreamer.actor"} and hasattr(dreamer, "actor"):
        return DreamerPolicyExport(dreamer.actor).to(dreamer.device)
    return None


def _iris_default(obj: Any, target: str | None) -> nn.Module | None:
    if target in {None, "actor_critic"} and hasattr(obj, "forward_actor_critic"):
        return IRISActorCriticExport(obj).to(obj.device)
    return None


def _jepa_default(obj: Any, target: str | None) -> nn.Module | None:
    if type(obj).__name__ != "JEPAAgent" or target not in {None, "encoder"}:
        return None
    encoder = getattr(obj, "encoder", None)
    if encoder is not None:
        return encoder
    cfg = obj.cfg
    vit = import_module("simulusion.models.vit")
    factory = getattr(vit, cfg.model_name)
    encoder = factory(img_size=[cfg.crop_size], patch_size=cfg.patch_size)
    setattr(obj, "encoder", encoder)
    return encoder


def _resolve_export_module(obj: Any, target: str | None = None) -> nn.Module:
    for adapter in (_dreamer_default, _iris_default, _jepa_default):
        module = adapter(obj, target)
        if module is not None:
            return module

    if target is not None:
        try:
            module = _resolve_attr_path(obj, target)
        except (AttributeError, KeyError, IndexError):
            modules = _discover_modules(obj)
            matches = [
                module
                for name, module in modules.items()
                if name == target or name.split(".")[-1] == target
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                available = ", ".join(
                    sorted(
                        name
                        for name in modules
                        if name == target or name.split(".")[-1] == target
                    )
                )
                raise ValueError(
                    f"Export target {target!r} matched multiple modules: {available}. "
                    "Use a fully qualified target path."
                )
            raise
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Export target {target!r} resolved to {type(module).__name__}, not torch.nn.Module."
            )
        return module

    if isinstance(obj, nn.Module):
        return obj

    modules = _discover_modules(obj)
    if not modules:
        raise TypeError(
            f"{type(obj).__name__} does not contain a torch.nn.Module to export. "
            "Attach a module attribute or pass target='path.to.module'."
        )
    if len(modules) == 1:
        return next(iter(modules.values()))
    for suffix in _PREFERRED_TARGET_SUFFIXES:
        for name, module in modules.items():
            if name.split(".")[-1] == suffix:
                return module
    available = ", ".join(sorted(modules))
    raise ValueError(
        f"{type(obj).__name__} contains multiple exportable modules. "
        f"Pass target=<name>; available targets: {available}."
    )


def _infer_example_inputs(
    obj: Any, module: nn.Module, target: str | None
) -> Any | None:
    if hasattr(obj, "dreamer") and target in {None, "actor", "dreamer.actor"}:
        args = obj.args
        return torch.zeros(
            1, args.stoch_size + args.deter_size, device=obj.dreamer.device
        )
    if hasattr(obj, "dreamer") and target in {"obs_encoder", "dreamer.obs_encoder"}:
        return torch.zeros(1, *obj.dreamer.obs_shape, device=obj.dreamer.device)
    if hasattr(obj, "dreamer") and target in {
        "reward_model",
        "value_model",
        "discount_model",
    }:
        args = obj.args
        return torch.zeros(
            1, args.stoch_size + args.deter_size, device=obj.dreamer.device
        )
    if hasattr(obj, "forward_actor_critic") and target in {None, "actor_critic"}:
        frame_shape = obj.config.get_frame_shape()
        return torch.zeros(1, 1, *frame_shape, device=obj.device)
    if type(obj).__name__ == "JEPAAgent" and target in {None, "encoder"}:
        device = next(module.parameters(), torch.empty(0)).device
        return torch.zeros(1, 3, obj.cfg.crop_size, obj.cfg.crop_size, device=device)
    if hasattr(obj, "num_frames") and hasattr(obj, "image_size"):
        device = next(module.parameters(), torch.empty(0)).device
        return torch.zeros(
            1, 3, obj.num_frames, obj.image_size, obj.image_size, device=device
        )
    if (
        hasattr(obj, "env")
        and hasattr(obj, "device")
        and module is getattr(obj, "rssm", None)
    ):
        obs = torch.zeros(1, 2, *obj.env.observation_size, device=obj.device)
        actions = torch.zeros(1, 1, obj.env.action_size, device=obj.device)
        return obs, actions
    return None


class ExportableAgentMixin:
    """Adds ``.export()`` to an agent or ``nn.Module``.

    Delegates to :func:`export_any`, which resolves the deployable module (the
    object itself, or ``target`` / a preferred submodule such as ``policy`` for
    agent wrappers) and infers example inputs where it can.
    """

    def export(
        self,
        path: str | Path,
        format: str = "onnx",
        *,
        example_inputs: Any | None = None,
        target: str | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 17,
        **kwargs: Any,
    ) -> Path:
        """Export this agent or one of its contained modules for deployment."""

        return export_any(
            self,
            path,
            format=format,
            example_inputs=example_inputs,
            target=target,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            **kwargs,
        )


def export_any(
    obj: Any,
    path: str | Path,
    format: str = "onnx",
    *,
    example_inputs: Any | None = None,
    target: str | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
    **kwargs: Any,
) -> Path:
    """Export any Simulusion model/agent or a target module contained by it."""

    module = _resolve_export_module(obj, target)
    if example_inputs is None:
        example_inputs = _infer_example_inputs(obj, module, target)
    return export_model(
        module,
        path,
        format=format,
        example_inputs=example_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        **kwargs,
    )


def export_model(
    module: nn.Module,
    path: str | Path,
    format: str = "onnx",
    *,
    example_inputs: Any | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
    **kwargs: Any,
) -> Path:
    """Export a ``torch.nn.Module`` to ONNX, TorchScript, or TensorRT."""

    export_format = _normalize_format(format)
    export_path = _as_path(path)
    was_training = module.training
    module.eval()

    if export_format in {"exported_program", "aoti"}:
        # Not under inference_mode: torch.export traces with its own fake
        # tensors, and inference tensors would leak into the graph.
        try:
            with torch.no_grad():
                _export_pt2(module, export_path, export_format, example_inputs, kwargs)
        finally:
            module.train(was_training)
        return export_path

    try:
        with torch.inference_mode():
            if export_format == "onnx":
                if example_inputs is None:
                    raise ValueError("example_inputs is required for ONNX export.")
                torch.onnx.export(
                    module,
                    _inputs_to_args(example_inputs),
                    str(export_path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=kwargs.pop("do_constant_folding", True),
                    **kwargs,
                )
            elif export_format == "torchscript":
                if example_inputs is None:
                    exported = torch.jit.script(module, **kwargs)
                else:
                    exported = torch.jit.trace(  # type: ignore[no-untyped-call]
                        module,
                        _inputs_to_args(example_inputs),
                        strict=kwargs.pop("strict", False),
                        **kwargs,
                    )
                exported.save(str(export_path))
            elif export_format == "tensorrt":
                if example_inputs is None:
                    raise ValueError("example_inputs is required for TensorRT export.")
                if util.find_spec("torch_tensorrt") is None:
                    raise RuntimeError(
                        "TensorRT export requires the optional torch-tensorrt package. "
                        "Install torch-tensorrt in your deployment environment or export ONNX first."
                    )
                torch_tensorrt = import_module("torch_tensorrt")
                trt_inputs = kwargs.pop("inputs", list(_inputs_to_args(example_inputs)))
                enabled_precisions = kwargs.pop("enabled_precisions", {torch.float32})
                # "dynamo" is Torch-TensorRT's maintained frontend; "ts" (the
                # previous default) goes through TorchScript and is legacy.
                ir = kwargs.pop("ir", "dynamo")
                compiled = torch_tensorrt.compile(
                    module,
                    ir=ir,
                    inputs=trt_inputs,
                    enabled_precisions=enabled_precisions,
                    **kwargs,
                )
                if ir == "ts":
                    torch.jit.save(compiled, str(export_path))
                else:
                    torch_tensorrt.save(
                        compiled,
                        str(export_path),
                        inputs=list(_inputs_to_args(example_inputs)),
                    )
            else:  # pragma: no cover - guarded by _normalize_format.
                raise AssertionError(f"Unhandled export format: {export_format}")
    finally:
        module.train(was_training)

    return export_path


def _export_pt2(
    module: nn.Module,
    export_path: Path,
    export_format: str,
    example_inputs: Any,
    kwargs: dict[str, Any],
) -> None:
    if example_inputs is None:
        raise ValueError(f"example_inputs is required for {export_format} export.")
    # ONNX-only options have no meaning for torch.export; drop them so the same
    # call site can switch formats.
    kwargs.pop("do_constant_folding", None)
    exported = torch.export.export(
        module,
        _inputs_to_args(example_inputs),
        dynamic_shapes=kwargs.pop("dynamic_shapes", None),
        strict=kwargs.pop("strict", False),
    )
    if export_format == "exported_program":
        torch.export.save(exported, str(export_path))
        return
    try:
        from torch._inductor import aoti_compile_and_package
    except ImportError as exc:  # pragma: no cover - torch < 2.6
        raise RuntimeError("AOTInductor export requires torch >= 2.6.") from exc
    aoti_compile_and_package(
        exported,
        package_path=str(export_path),
        inductor_configs=kwargs.pop("inductor_configs", None),
    )


def _format_from_suffix(path: Path) -> ExportFormat:
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".pt2":
        return "exported_program"
    return "torchscript"


def load_exported(
    path: str | Path,
    format: str | None = None,
    *,
    device: torch.device | str | None = None,
) -> Callable[..., Any]:
    """Load an exported artifact back as a callable.

    Args:
        path: Artifact written by :func:`export_model`.
        format: Its export format. Inferred from the suffix when omitted, with
            ``.pt2`` read as ``exported_program``; pass ``"aoti"`` for
            AOTInductor packages, which share the suffix.
        device: Device to move an ExportedProgram or TorchScript module to.

    ONNX artifacts run through ONNX Runtime (optional dependency); the returned
    callable takes and returns tensors like the other formats.
    """
    path = Path(path)
    export_format = _normalize_format(format) if format else _format_from_suffix(path)
    if export_format == "exported_program":
        module = torch.export.load(str(path)).module()
        return module.to(device) if device is not None else module
    if export_format == "aoti":
        from torch._inductor import aoti_load_package

        runner: Callable[..., Any] = aoti_load_package(str(path))
        return runner
    if export_format == "onnx":
        return _onnxruntime_callable(path)
    if export_format == "tensorrt":
        if util.find_spec("torch_tensorrt") is None:
            raise RuntimeError("Loading TensorRT artifacts requires torch-tensorrt.")
        import_module("torch_tensorrt")
        try:
            return torch.export.load(str(path)).module()
        except Exception:
            pass  # a legacy ir="ts" artifact: fall through to TorchScript
    scripted: Callable[..., Any] = torch.jit.load(str(path), map_location=device)  # type: ignore[no-untyped-call]
    return scripted


def _onnxruntime_callable(path: Path) -> Callable[..., Any]:
    if util.find_spec("onnxruntime") is None:
        raise RuntimeError(
            "Running ONNX artifacts requires the optional onnxruntime package."
        )
    ort = import_module("onnxruntime")
    session = ort.InferenceSession(str(path), providers=ort.get_available_providers())
    input_names = [i.name for i in session.get_inputs()]

    def run(*args: torch.Tensor) -> Any:
        feeds = {
            name: arg.detach().cpu().numpy() for name, arg in zip(input_names, args)
        }
        outputs = [torch.from_numpy(o) for o in session.run(None, feeds)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    return run


def verify_export(
    module: nn.Module,
    exported: str | Path | Callable[..., Any],
    example_inputs: Any,
    *,
    format: str | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> float:
    """Check that an exported artifact reproduces ``module`` on ``example_inputs``.

    ``exported`` is an artifact path (loaded with :func:`load_exported`) or an
    already-loaded callable. Returns the largest absolute difference over all
    tensor outputs, and raises ``AssertionError`` if any output falls outside
    ``atol``/``rtol``. For recurrent models also check closed-loop drift with
    :func:`simulusion.inference.rollout_drift`: one matching step does not rule out
    error that compounds over a rollout.
    """
    from torch.utils._pytree import tree_flatten

    runner = (
        load_exported(exported, format)
        if isinstance(exported, (str, Path))
        else exported
    )
    args = _inputs_to_args(example_inputs)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            expected = [
                t for t in tree_flatten(module(*args))[0] if isinstance(t, torch.Tensor)
            ]
            actual = [
                t for t in tree_flatten(runner(*args))[0] if isinstance(t, torch.Tensor)
            ]
    finally:
        module.train(was_training)
    if len(expected) != len(actual):
        raise AssertionError(
            f"Exported artifact returned {len(actual)} tensors; eager returned "
            f"{len(expected)}."
        )
    worst = 0.0
    for want, got in zip(expected, actual):
        got = got.to(want.device, want.dtype)
        if want.numel():
            worst = max(worst, float((want - got).abs().max()))
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    return worst


__all__ = [
    "DreamerPolicyExport",
    "ExportFormat",
    "ExportableAgentMixin",
    "IRISActorCriticExport",
    "export_any",
    "export_model",
    "load_exported",
    "verify_export",
]
