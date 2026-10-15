"""Logging, metrics, and numerical-safety helpers for simulusion."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import json
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Generator, Optional

import torch

_PACKAGE_LOGGER_NAME = "simulusion"


def get_package_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``simulusion`` package namespace."""
    if not name:
        return logging.getLogger(_PACKAGE_LOGGER_NAME)
    if name == _PACKAGE_LOGGER_NAME or name.startswith(f"{_PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def setup_logging(
    name: str = _PACKAGE_LOGGER_NAME,
    level: str | int = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Set up structured package logging with optional file output.

    Args:
        name: Logger name to configure. Defaults to the package logger.
        level: Logging level name or numeric value.
        log_file: Optional file path for a file handler.
        fmt: ``logging.Formatter`` format string.
    """
    logger = get_package_logger(name)
    resolved_level = (
        logging.getLevelName(level.upper()) if isinstance(level, str) else level
    )
    if isinstance(resolved_level, str):
        raise ValueError(f"Unknown logging level: {level}")

    logger.setLevel(resolved_level)
    # Keep propagating: an application's own logging configuration (and
    # pytest's caplog) listens on the root logger, and turning propagation off
    # here hid every ``simulusion.*`` record from it as soon as an agent was
    # constructed. A console handler is only attached when the root logger has
    # none; otherwise each message would print twice.
    logger.propagate = True

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt)
    if not logging.getLogger().handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        directory = os.path.dirname(log_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _to_scalar(value: Any) -> Any:
    """Convert tensors/numpy scalars to JSON-serializable scalar values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_summary_writer() -> Any | None:
    """Return a TensorBoard SummaryWriter class when an implementation exists."""
    if importlib.util.find_spec("torch.utils.tensorboard") is not None:
        tensorboard_module = importlib.import_module("torch.utils.tensorboard")
        return tensorboard_module.SummaryWriter
    if importlib.util.find_spec("tensorboardX") is not None:
        tensorboardx_module = importlib.import_module("tensorboardX")
        return tensorboardx_module.SummaryWriter
    return None


def _prepare_tensorboard_video(video: Any) -> torch.Tensor:
    """Convert a video to TensorBoard's ``(N, T, C, H, W)`` layout."""
    video_tensor = video if isinstance(video, torch.Tensor) else torch.as_tensor(video)
    video_tensor = video_tensor.detach().cpu()
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)
    if video_tensor.ndim != 5:
        raise ValueError(
            "TensorBoard videos must have shape (T,H,W,C), (T,C,H,W), "
            "(N,T,H,W,C), or (N,T,C,H,W)."
        )
    if video_tensor.shape[-1] in (1, 3, 4):
        video_tensor = video_tensor.permute(0, 1, 4, 2, 3)
    if video_tensor.dtype == torch.uint8:
        video_tensor = video_tensor.to(torch.float32) / 255.0
    else:
        video_tensor = video_tensor.to(torch.float32)
    return torch.clamp(video_tensor, 0.0, 1.0)


class MetricsLogger:
    """Fan-out metric logger for console, JSONL, TensorBoard, and W&B.

    JSONL output is enabled by default because it is dependency-free and easy to
    reload for offline plots. TensorBoard and W&B are optional and activated only
    when requested and available/configured.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        logger: logging.Logger | None = None,
        enable_console: bool = True,
        enable_jsonl: bool = True,
        jsonl_filename: str = "metrics.jsonl",
        enable_tensorboard: bool = False,
        enable_wandb: bool = False,
        wandb_project: str = "simulusion",
        wandb_entity: str = "",
        run_name: str | None = None,
    ) -> None:
        self.log_dir = log_dir
        self.logger = logger or get_package_logger("metrics")
        self.enable_console = enable_console
        self.enable_jsonl = enable_jsonl
        self.jsonl_path = os.path.join(log_dir, jsonl_filename)
        self._jsonl_file = None
        self._tb_writer = None
        self._wandb_run = None

        os.makedirs(log_dir, exist_ok=True)
        if self.enable_jsonl:
            self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        if enable_tensorboard:
            summary_writer = _load_summary_writer()
            if summary_writer is None:
                self.logger.warning("TensorBoard logging requested but unavailable")
            else:
                self._tb_writer = summary_writer(log_dir=log_dir)

        if enable_wandb:
            if importlib.util.find_spec("wandb") is None:
                raise ImportError("wandb is not installed")
            wandb = importlib.import_module("wandb")
            self._wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity or None,
                dir=log_dir,
                name=run_name or os.path.basename(log_dir),
            )

    def log(
        self, metrics: Mapping[str, Any], step: int, prefix: str | None = None
    ) -> dict[str, Any]:
        """Log scalar metrics to every enabled sink."""
        normalized = {}
        for key, value in metrics.items():
            metric_key = f"{prefix}/{key}" if prefix else str(key)
            normalized[metric_key] = _to_scalar(value)

        if self.enable_console and normalized:
            formatted = ", ".join(f"{key}={value}" for key, value in normalized.items())
            self.logger.info("step=%s %s", step, formatted)

        if self._jsonl_file is not None:
            self._jsonl_file.write(
                json.dumps(
                    {"time": time.time(), "step": int(step), **normalized},
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

        if self._tb_writer is not None:
            for key, value in normalized.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, step)

        if self._wandb_run is not None:
            self._wandb_run.log(normalized, step=step)

        return normalized

    def log_video(self, name: str, video: Any, step: int, fps: int = 20) -> None:
        """Log a video to TensorBoard and W&B when enabled."""
        if self._tb_writer is not None:
            self._tb_writer.add_video(
                name, _prepare_tensorboard_video(video), global_step=step, fps=fps
            )
        if self._wandb_run is not None:
            wandb = importlib.import_module("wandb")
            self._wandb_run.log({name: wandb.Video(video, fps=fps)}, step=step)

    def flush(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def close(self) -> None:
        self.flush()
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None


logger = logging.getLogger(__name__)

# GPU utilization needs NVML. Warn once rather than on every logging interval.
_NVML_WARNED = False


def _mark_nvml_warned() -> None:
    global _NVML_WARNED
    _NVML_WARNED = True


def collect_system_stats(device: torch.device | str | None = None) -> dict[str, float]:
    """Collect CPU/GPU memory and CUDA utilization counters when available."""
    stats: dict[str, float] = {}

    if importlib.util.find_spec("psutil") is not None:
        psutil = importlib.import_module("psutil")
        vm = psutil.virtual_memory()
        stats.update(
            {
                "system/cpu_percent": float(psutil.cpu_percent(interval=None)),
                "system/ram_used_mb": float(vm.used / (1024**2)),
                "system/ram_available_mb": float(vm.available / (1024**2)),
                "system/ram_percent": float(vm.percent),
            }
        )

    torch_device = torch.device(device) if device is not None else None
    if torch.cuda.is_available() and (
        torch_device is None or torch_device.type == "cuda"
    ):
        cuda_index = (
            torch_device.index
            if torch_device and torch_device.index is not None
            else torch.cuda.current_device()
        )
        stats.update(
            {
                "system/gpu_memory_allocated_mb": float(
                    torch.cuda.memory_allocated(cuda_index) / (1024**2)
                ),
                "system/gpu_memory_reserved_mb": float(
                    torch.cuda.memory_reserved(cuda_index) / (1024**2)
                ),
                "system/gpu_max_memory_allocated_mb": float(
                    torch.cuda.max_memory_allocated(cuda_index) / (1024**2)
                ),
            }
        )
        # `torch.cuda.utilization` always exists, so `hasattr` guarded nothing:
        # calling it needs NVML (`nvidia-ml-py`), and without that torch raises
        # ModuleNotFoundError from inside. That killed training runs on GPU
        # machines whenever `log_system_stats_freq` was set. Metrics are
        # optional, so a missing counter is skipped, never fatal.
        try:
            stats["system/gpu_utilization_percent"] = float(
                torch.cuda.utilization(cuda_index)
            )
        except Exception as exc:  # NVML missing, or a driver that cannot report
            if not _NVML_WARNED:
                logger.warning(
                    "GPU utilization is unavailable (%s); logging the other "
                    "system metrics without it. Install `nvidia-ml-py` (or "
                    "`simulusion[ml]`) to record it.",
                    exc,
                )
                _mark_nvml_warned()

    return stats


def _iter_tensors(value: Any) -> Generator[torch.Tensor, None, None]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def assert_finite_values(value: Any, name: str = "value") -> Any:
    """Raise ``FloatingPointError`` if any tensor contains NaN or Inf."""
    for tensor in _iter_tensors(value):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite tensor detected in {name}")
    return value


def assert_finite(fn: Any) -> Any:
    """Decorator that validates tensor outputs from loss functions are finite."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return assert_finite_values(result, getattr(fn, "__qualname__", fn.__name__))

    return wrapper
