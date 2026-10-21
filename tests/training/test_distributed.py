"""Two CPU processes (gloo): every rank must end with the same weights.

Covers both ways gradients are synchronised: DDP-wrapped modules (Genie) and
GradientUpdate(all_reduce=True) for models driven outside forward()
(Dreamer's world model).

The ranks are spawned from a separate Python process (this file run as a
script), not from pytest's own: on Windows, a process that has already run
many torch tests can crash natively while it waits on spawned children, and
that would take the whole test session down with it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.training.tasks.test_genie_task import TINY as GENIE_TINY

# End-to-end training runs: each builds a model, trains it for a few steps,
# checkpoints, resumes and exports. On a CPU runner that takes longer than the
# suite-wide --timeout=30 allows unit tests.
pytestmark = pytest.mark.timeout(180)

REPO = str(Path(__file__).resolve().parents[2])


#: Committable memory the helper and its two ranks need (each loads torch).
_NEEDED_COMMIT_GB = 8.0


def _available_commit_gb() -> float:
    """Windows' remaining commit charge, in GB (``inf`` elsewhere).

    Windows backs every allocation with commit charge, and a CUDA build of
    torch commits several GB per process even with no GPU in use. When the
    charge runs out, native allocations fail and processes die with access
    violations instead of a MemoryError, so the test skips rather than run.
    """
    if sys.platform != "win32":
        return float("inf")
    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPageFile / 2**30


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _worker(
    rank: int, world: int, port: int, task: str, overrides: list, out: str
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_WORLD_SIZE=str(world),
        USE_LIBUV="0",
        CUDA_VISIBLE_DEVICES="-1",
    )
    sys.path.insert(0, REPO)
    from simulusion.training.run import resolve_run_config
    from simulusion.training.trainer import Trainer

    cls, tcfg, cfg, resolved = resolve_run_config(overrides=overrides, task=task)
    trainer = Trainer(tcfg, run_config=resolved)
    instance = cls(cfg)
    trainer.fit(instance)
    torch.save(
        instance.state_dict() if task == "genie" else _dreamer_weights(instance),
        f"{out}/rank{rank}.pt",
    )
    dist.barrier()
    # Tear the group down explicitly: leaving it to interpreter exit can
    # crash gloo on Windows.
    dist.destroy_process_group()


def _dreamer_weights(task) -> dict:
    return {
        f"{i}.{k}": v
        for i, m in enumerate(task.dreamer.world_model_modules)
        for k, v in m.state_dict().items()
    }


def _flatten(state: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in state.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        elif isinstance(value, torch.Tensor):
            out[f"{prefix}{key}"] = value
    return out


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="needs gloo"
)
@pytest.mark.parametrize("task", ["genie", "dreamer-wm"])
def test_two_processes_stay_in_sync(tmp_path: Path, task: str) -> None:
    available = _available_commit_gb()
    if available < _NEEDED_COMMIT_GB:
        pytest.skip(
            f"{available:.1f} GB of commit charge left; two training processes "
            f"need about {_NEEDED_COMMIT_GB:.0f} GB"
        )
    from tests.training.tasks.conftest import write_episodes

    actions = None if task == "genie" else ("continuous", 2)
    size = 16 if task == "genie" else 64
    data = write_episodes(
        tmp_path / "eps", count=6, length=8, size=size, actions=actions
    )
    overrides = [
        "trainer.device=cpu",
        f"trainer.out_dir={tmp_path.as_posix()}/runs",
        "trainer.run_name=ddp",
        "trainer.console=false",
        "trainer.max_steps=3",
        "trainer.seed=0",
        f"data.path={data.as_posix()}",
        "data.batch_size=2",
    ]
    if task == "genie":
        overrides += GENIE_TINY + ["trainer.find_unused_parameters=true"]
    else:
        overrides += [
            "model.train_seq_len=4",
            "model.obs_embed_size=32",
            "model.num_units=16",
            "model.deter_size=16",
            "model.stoch_size=4",
            "model.use_amp=false",
        ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": REPO}
    proc = subprocess.run(
        [sys.executable, __file__, task, str(tmp_path), json.dumps(overrides)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    a = _flatten(torch.load(tmp_path / "rank0.pt", weights_only=True))
    b = _flatten(torch.load(tmp_path / "rank1.pt", weights_only=True))
    weights = [k for k in a if "update" not in k and torch.is_floating_point(a[k])]
    assert weights
    for key in weights:
        assert torch.equal(a[key], b[key]), key
    # Ranks saw different data (sharded loaders, rank-offset seeds), so the
    # agreement above is synchronisation, not coincidence: training moved them.
    assert (tmp_path / "runs" / "ddp" / "world_model" / "world_model.json").exists()


def _spawn(task: str, out: str, overrides: list) -> None:
    mp.spawn(_worker, args=(2, _free_port(), task, overrides, out), nprocs=2, join=True)


if __name__ == "__main__":
    _spawn(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
