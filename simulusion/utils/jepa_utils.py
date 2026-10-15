import math
import torch
from logging import getLogger
import os
import torch.distributed as dist

from types import ModuleType
from typing import Any, Optional, Tuple

# Optional WandB import; keep typed for static checkers.
wandb: Optional[ModuleType] = None
try:
    import wandb as _wandb

    wandb = _wandb
except ImportError:
    wandb = None

logger = getLogger()


def _no_grad_trunc_normal_(
    tensor: torch.Tensor, mean: float, std: float, a: float, b: float
) -> torch.Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l_ = norm_cdf((a - mean) / std)
        u_ = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l_ - 1, 2 * u_ - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2,
    b: float = 2.0,
) -> torch.Tensor:
    """Initialize a tensor in-place from a truncated normal distribution.

    Values are sampled from `N(mean, std)` and clipped to `[a, b]`.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int) -> torch.Tensor:
    """Repeat each batch chunk multiple times while preserving chunk ordering.

    Used in JEPA masking code to align context and target token batches.
    """
    N = len(x) // B
    x = torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )
    return x


class WarmupCosineSchedule(object):
    """Learning-rate schedule with linear warmup followed by cosine decay.

    Updates optimizer parameter-group LRs on each call to `step()`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        T_max: int,
        last_epoch: int = -1,
        final_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self) -> float:
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr


class CosineWDSchedule(object):
    """Cosine scheduler for optimizer weight decay values.

    Skips parameter groups flagged with `WD_exclude` to keep bias/norm decay at zero.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        ref_wd: float,
        T_max: int,
        final_wd: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self) -> float:
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd

        return new_wd


def gpu_timer(closure: Any, log_timings: bool = True) -> Tuple[Any, float]:
    """Measure CUDA execution time for a closure and return `(result, elapsed_ms)`.

    Falls back to `-1` elapsed time when CUDA timing is unavailable.
    """
    log_timings = log_timings and torch.cuda.is_available()
    elapsed_time = -1.0

    if log_timings:
        start: Any = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end: Any = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):
    """Lightweight CSV logger with per-column printf-style formatting and WandB support."""

    def __init__(
        self,
        fname: str,
        enable_wandb: bool = False,
        wandb_project: str = "simulusion",
        wandb_entity: str = "",
        *argv: Any,
    ) -> None:
        self.fname = fname
        self.types = []
        self.names = []
        self.enable_wandb = enable_wandb
        self._wandb_run = None

        if self.enable_wandb:
            if wandb is None:
                raise ImportError("wandb is not installed")
            self._wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                dir=os.path.dirname(fname),
                name=os.path.basename(fname).replace(".csv", ""),
            )

        with open(self.fname, "+a") as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                self.names.append(v[1])
                if i < len(argv):
                    print(v[1], end=",", file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, step: int, *argv: Any) -> None:
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = "," if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)

        # Log to WandB
        if self.enable_wandb and self._wandb_run:
            log_dict = {}
            for name, value in zip(self.names, argv):
                log_dict[name] = value
            self._wandb_run.log(log_dict, step=step)


class AverageMeter(object):
    """Track running statistics (`val`, `avg`, `min`, `max`, `sum`, `count`) for metrics."""

    def __init__(self) -> None:
        self.first_layer: float | None = None
        self.last_layer: float | None = None
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params: Any) -> AverageMeter:
    """Aggregate gradient norm statistics over model parameters for logging.

    Also exposes first/last qkv-layer gradient norms when available.
    """
    stats = AverageMeter()
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith(".bias") or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if "qkv" in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm

    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = 0.0
        stats.last_layer = 0.0

    return stats


def init_distributed(
    port: int = 40112, rank_and_world_size: tuple = (None, None)
) -> Tuple[int, int]:
    """Initialize torch distributed process groups when environment supports it.

    Returns `(world_size, rank)` and gracefully falls back to single-process mode.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=rank
        )
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"distributed training not available {e}")

    return world_size, rank


class AllGather(torch.autograd.Function):
    """Autograd-aware all-gather operation across distributed workers.

    Forward concatenates worker tensors; backward reduces and slices gradients.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, dim=0)
        return x

    @staticmethod
    def backward(ctx: Any, grads: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):
    """Autograd function that sums tensors across distributed workers in forward pass."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx: Any, grads: torch.Tensor) -> torch.Tensor:
        return grads


class AllReduce(torch.autograd.Function):
    """Autograd function that all-reduces and averages tensors across workers.

    Used to synchronize scalar losses for consistent distributed logging/training.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx: Any, grads: torch.Tensor) -> torch.Tensor:
        return grads
