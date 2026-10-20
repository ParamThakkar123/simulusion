"""Where a run came from: the code, the packages and the machine.

Written to ``run.json`` next to every run's ``config.yaml``. The config says
what was asked for; this says what it ran on, so a result can be traced back
to a commit and reproduced.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

import torch

_PACKAGES = ("simulusion", "torch", "numpy", "accelerate", "gymnasium")


def collect_provenance() -> dict[str, Any]:
    """Git state, package versions, hardware and the command line."""
    info: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]] if sys.argv else [],
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cuda": torch.version.cuda,
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }
    git = git_state(Path(__file__).resolve().parent)
    if git:
        info["git"] = git
    return info


def git_state(path: Path) -> Optional[dict[str, Any]]:
    """Commit, branch and dirty flag of the checkout holding ``path``, if any."""

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if not commit:
        return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


__all__ = ["collect_provenance", "git_state"]
