"""Self-describing deployment bundles.

A checkpoint alone is not deployable: a server also needs the config the model
was built with, the input shapes and dtypes it expects, and ideally a compiled
artifact it can run without the training code. A bundle is a directory holding
all of that, plus a ``manifest.json`` describing it::

    policy_bundle/
        manifest.json          # versions, input/output spec, artifact index
        weights.safetensors    # or weights.pt without safetensors
        config.json            # optional: the model config
        model.pt2              # optional: exported artifacts, one per format
        model.onnx

``load_bundle`` reads it back; ``Bundle.load_artifact()`` returns a callable
for the deployment format and ``Bundle.load_weights(module)`` restores eager
weights for the Python path.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn

MANIFEST = "manifest.json"
BUNDLE_VERSION = 1

_SUFFIXES = {
    "exported_program": ".pt2",
    "aoti": ".aoti.pt2",
    "onnx": ".onnx",
    "torchscript": ".ts.pt",
    "tensorrt": ".trt.ep",
}


def _tensor_spec(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype).split(".")[-1]}
    if isinstance(value, (list, tuple)):
        return [_tensor_spec(v) for v in value]
    if isinstance(value, dict):
        return {k: _tensor_spec(v) for k, v in value.items()}
    return repr(value)


def _config_dict(config: Any) -> Optional[dict[str, Any]]:
    if config is None:
        return None
    if isinstance(config, dict):
        return config
    from simulusion.configs.serialization import config_to_dict

    return config_to_dict(config)


def _save_weights(module: nn.Module, directory: Path) -> str:
    # Tied / shared tensors are cloned so safetensors accepts them; the file
    # stays loadable with ``load_state_dict`` either way.
    state = {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}
    if util.find_spec("safetensors") is not None:
        from safetensors.torch import save_file

        seen: dict[int, str] = {}
        for key, tensor in list(state.items()):
            ptr = tensor.untyped_storage().data_ptr()
            if ptr in seen:
                state[key] = tensor.clone()
            seen[ptr] = key
        save_file(state, str(directory / "weights.safetensors"))
        return "weights.safetensors"
    torch.save(state, directory / "weights.pt")
    return "weights.pt"


@dataclass
class Bundle:
    """A loaded deployment bundle."""

    path: Path
    manifest: dict[str, Any]
    config: Optional[dict[str, Any]] = None
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def formats(self) -> list[str]:
        return list(self.artifacts)

    def state_dict(self, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        weights = self.path / self.manifest["weights"]
        if weights.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(weights), device=str(map_location))
        state: dict[str, Any] = torch.load(
            weights, map_location=map_location, weights_only=True
        )
        return state

    def load_weights(self, module: nn.Module, strict: bool = True) -> nn.Module:
        """Load the bundled weights into an eagerly constructed ``module``."""
        device = next(module.parameters(), torch.empty(0)).device
        module.load_state_dict(self.state_dict(device), strict=strict)
        return module

    def load_artifact(
        self,
        format: Optional[str] = None,
        *,
        device: torch.device | str | None = None,
    ) -> Callable[..., Any]:
        """Load an exported artifact as a callable (first available by default)."""
        from simulusion.export import _normalize_format, load_exported

        if not self.artifacts:
            raise FileNotFoundError(f"Bundle {self.path} contains no artifacts.")
        name = _normalize_format(format) if format else next(iter(self.artifacts))
        if name not in self.artifacts:
            raise KeyError(
                f"Bundle has no {name!r} artifact; available: {self.formats}."
            )
        return load_exported(self.path / self.artifacts[name], name, device=device)


def save_bundle(
    directory: str | Path,
    module: nn.Module,
    *,
    example_inputs: Any = None,
    formats: Iterable[str] = ("exported_program",),
    config: Any = None,
    name: str = "model",
    metadata: Optional[dict[str, Any]] = None,
    verify: bool = True,
    export_kwargs: Optional[dict[str, Any]] = None,
) -> Bundle:
    """Write ``module`` as a deployment bundle.

    Args:
        directory: Output directory (created if missing).
        module: Module to bundle - typically a pure step module such as
            :class:`~simulusion.inference.steppers.DreamerStepModule`.
        example_inputs: Inputs used to export and to record the input spec.
            Required when ``formats`` is non-empty.
        formats: Export formats to include (see :mod:`simulusion.export`). Pass
            ``()`` for a weights-and-config bundle.
        config: Model config (dataclass, object or dict), stored as JSON.
        name: Base file name of the artifacts.
        metadata: Extra JSON-serialisable fields for the manifest.
        verify: Check each artifact against the eager module with
            :func:`~simulusion.export.verify_export` and record the max error.
        export_kwargs: Extra keyword arguments per format, e.g.
            ``{"exported_program": {"dynamic_shapes": ...}}``.
    """
    from simulusion import __version__
    from simulusion.export import _normalize_format, export_model, verify_export

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    formats = [_normalize_format(f) for f in formats]
    if formats and example_inputs is None:
        raise ValueError("example_inputs is required to export artifacts.")
    export_kwargs = export_kwargs or {}

    manifest: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulusion_version": __version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "module_class": f"{type(module).__module__}.{type(module).__qualname__}",
        "weights": _save_weights(module, directory),
        "inputs": _tensor_spec(example_inputs) if example_inputs is not None else None,
        "artifacts": {},
        "verification": {},
        "metadata": metadata or {},
    }

    if example_inputs is not None:
        was_training = module.training
        module.eval()
        with torch.no_grad():
            outputs = module(
                *(
                    example_inputs
                    if isinstance(example_inputs, tuple)
                    else (example_inputs,)
                )
            )
        module.train(was_training)
        manifest["outputs"] = _tensor_spec(outputs)

    config_dict = _config_dict(config)
    if config_dict is not None:
        (directory / "config.json").write_text(
            json.dumps(config_dict, indent=2, default=str), encoding="utf-8"
        )
        manifest["config"] = "config.json"

    for fmt in formats:
        filename = f"{name}{_SUFFIXES[fmt]}"
        export_model(
            module,
            directory / filename,
            format=fmt,
            example_inputs=example_inputs,
            **export_kwargs.get(fmt, {}),
        )
        manifest["artifacts"][fmt] = filename
        if verify:
            manifest["verification"][fmt] = {
                "max_abs_error": verify_export(
                    module, directory / filename, example_inputs, format=fmt
                )
            }

    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return load_bundle(directory)


def load_bundle(directory: str | Path) -> Bundle:
    """Read a bundle written by :func:`save_bundle`."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{directory} is not a Simulusion bundle (no {MANIFEST})."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_version", 0) > BUNDLE_VERSION:
        raise ValueError(
            f"Bundle version {manifest['bundle_version']} is newer than this "
            f"Simulusion supports ({BUNDLE_VERSION}); upgrade simulusion."
        )
    config = None
    if manifest.get("config"):
        config = json.loads(
            (directory / manifest["config"]).read_text(encoding="utf-8")
        )
    return Bundle(
        path=directory,
        manifest=manifest,
        config=config,
        artifacts=dict(manifest.get("artifacts", {})),
    )


__all__ = ["Bundle", "load_bundle", "save_bundle"]
