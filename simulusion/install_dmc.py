"""Install the DeepMind Control backend, including on CPython 3.13.

``pip install simulusion[dmc]`` covers Python 3.12 and below. On 3.13 it installs
``labmaze-new`` but leaves ``dm_control`` out, because
dm-control pins the ``labmaze`` distribution whose newest release (1.0.6)
publishes wheels only through CPython 3.12. Resolving that pin on 3.13 builds
labmaze from source with Bazel, which fails the whole install.

This module finishes the job: it installs ``labmaze-new`` (same package, built
with setuptools instead of Bazel), then ``dm-control`` with ``--no-deps`` so pip
never looks at that pin, then dm-control's remaining dependencies explicitly.

``labmaze-new`` 0.0.3 compiles labmaze's native extension, which provides
``labmaze.RandomMaze`` for Memory Maze and ``dm_control.locomotion`` mazes.
Without a C++17 compiler it installs as pure Python, which is still enough for
``dm_control.suite``, the only part Simulusion's dmc backend uses.

Usage::

    python -m simulusion.install_dmc                # install and verify
    python -m simulusion.install_dmc --memory-maze  # also install Memory Maze
    python -m simulusion.install_dmc --check        # verify only, install nothing
    python -m simulusion.install_dmc --dry-run      # print the commands

``--memory-maze`` adds the ``memory-maze`` package, again with ``--no-deps``:
its only requirement is dm-control, whose ``labmaze`` pin is the problem above.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys

# dm-control's own requirements, minus `labmaze` (supplied by labmaze-new on
# 3.13) and minus numpy, which Simulusion already requires through torch.
DM_CONTROL_DEPS: tuple[str, ...] = (
    "absl-py>=0.7.0",
    "dm-env",
    "dm-tree!=0.1.2",
    "glfw",
    "lxml",
    "protobuf>=3.19.4",
    "pyopengl>=3.1.4",
    "pyparsing>=3.0.0",
    "requests",
    "scipy",
    "setuptools!=50.0.0",
    "tqdm",
    "gymnasium>=1.2.2",
)

LABMAZE = "labmaze-new>=0.0.3"

# dm-control raises its `mujoco` floor with nearly every release, and upgrading
# mujoco to satisfy the newest one breaks other backends in the same
# environment: mujoco 3.13 removed `mjtEnableBit.mjENBL_MULTICCD`, which
# mujoco-mjx (and therefore brax) still uses. So the installed mujoco is left
# alone and the newest dm-control that works with it is chosen instead. Pass
# ``--upgrade-mujoco`` to do the opposite.
#
# Each entry is (minimum mujoco version, dm-control release requiring it),
# newest first.
DM_CONTROL_BY_MUJOCO: tuple[tuple[tuple[int, int, int], str], ...] = (
    ((3, 13, 0), "1.0.46"),
    ((3, 12, 0), "1.0.45"),
    ((3, 11, 0), "1.0.44"),
    ((3, 10, 0), "1.0.43"),
    ((3, 8, 1), "1.0.41"),
    ((3, 8, 0), "1.0.40"),
    ((3, 7, 0), "1.0.39"),
    ((3, 6, 0), "1.0.38"),
    ((3, 5, 0), "1.0.37"),
    ((3, 4, 0), "1.0.36"),
    ((3, 3, 6), "1.0.34"),
    ((3, 3, 3), "1.0.31"),
    ((3, 3, 2), "1.0.30"),
    ((3, 2, 7), "1.0.28"),
)

# Memory Maze's only dependency is dm-control, so it too installs with
# --no-deps. It needs labmaze's compiled RandomMaze, which labmaze-new 0.0.3
# builds when a C++17 compiler is available.
MEMORY_MAZE = "memory-maze>=1.0.3"

# Used when mujoco is absent, or with --upgrade-mujoco.
NEWEST_DM_CONTROL = "dm-control>=1.0.46"
NEWEST_MUJOCO = "mujoco>=3.13.0"


def installed_mujoco_version() -> tuple[int, ...] | None:
    """The installed mujoco version, or None when it is not installed."""
    try:
        from importlib.metadata import version

        raw = version("mujoco")
    except Exception:
        return None
    parts: list[int] = []
    for chunk in raw.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def dm_control_requirement(
    mujoco_version: tuple[int, ...] | None, upgrade_mujoco: bool = False
) -> str:
    """Pick the dm-control requirement to install.

    With no mujoco installed, or with ``upgrade_mujoco``, this is the newest
    release. Otherwise it is the newest release whose mujoco floor the installed
    version already satisfies, so mujoco is never upgraded out from under
    mujoco-mjx / brax.
    """
    if upgrade_mujoco or mujoco_version is None:
        return NEWEST_DM_CONTROL
    for floor, release in DM_CONTROL_BY_MUJOCO:
        if mujoco_version >= floor:
            return f"dm-control=={release}"
    oldest_floor, oldest = DM_CONTROL_BY_MUJOCO[-1]
    floor_text = ".".join(str(part) for part in oldest_floor)
    raise SystemExit(
        f"mujoco {'.'.join(str(p) for p in mujoco_version)} is older than "
        f"{floor_text}, the floor of the oldest dm-control release this "
        f"installer knows ({oldest}). Upgrade mujoco, or pass --upgrade-mujoco "
        "to install the newest mujoco and dm-control together."
    )


def _pip_command() -> list[str]:
    """Return the installer command, preferring uv when it runs this env."""
    if _running_under_uv():
        return ["uv", "pip", "install"]
    return [sys.executable, "-m", "pip", "install"]


def _running_under_uv() -> bool:
    try:
        import shutil

        if shutil.which("uv") is None:
            return False
        # A uv-managed venv records itself in pyvenv.cfg.
        from pathlib import Path

        cfg = Path(sys.prefix) / "pyvenv.cfg"
        return cfg.exists() and "uv" in cfg.read_text(encoding="utf-8").lower()
    except Exception:
        return False


def install_steps(
    upgrade_mujoco: bool = False, memory_maze: bool = False
) -> list[list[str]]:
    """The installer commands, in order."""
    pip = _pip_command()
    mujoco_version = installed_mujoco_version()
    dm_control = dm_control_requirement(mujoco_version, upgrade_mujoco)
    deps: list[str] = list(DM_CONTROL_DEPS)
    if mujoco_version is None or upgrade_mujoco:
        deps.append(NEWEST_MUJOCO)
    steps = [
        [*pip, LABMAZE],
        [*pip, "--no-deps", dm_control],
        [*pip, *deps],
    ]
    if memory_maze:
        steps.append([*pip, "--no-deps", MEMORY_MAZE])
    return steps


def check() -> tuple[bool, str]:
    """Report whether the DeepMind Control backend is importable."""
    try:
        importlib.import_module("labmaze")
    except Exception as exc:  # pragma: no cover - depends on the environment
        return False, f"labmaze is not importable: {exc}"
    try:
        suite = importlib.import_module("dm_control.suite")
    except Exception as exc:
        return False, f"dm_control.suite is not importable: {exc}"
    return True, f"{len(suite.ALL_TASKS)} DeepMind Control tasks available"


def check_memory_maze() -> tuple[bool, str]:
    """Report whether Memory Maze can generate mazes."""
    try:
        labmaze = importlib.import_module("labmaze")
    except Exception as exc:  # pragma: no cover - depends on the environment
        return False, f"labmaze is not importable: {exc}"
    if getattr(labmaze, "RandomMaze", None) is None:
        return False, (
            "labmaze has no compiled RandomMaze, which Memory Maze needs. "
            "Reinstall labmaze-new>=0.0.3 with a C++17 compiler available: "
            "pip install --force-reinstall --no-cache-dir labmaze-new"
        )
    try:
        from simulusion.envs.memory_maze_env import _require_memory_maze_tasks

        _require_memory_maze_tasks()
    except Exception as exc:
        return False, f"memory_maze is not importable: {exc}"
    return True, "Memory Maze available"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the DeepMind Control backend for Simulusion"
    )
    parser.add_argument(
        "--check", action="store_true", help="only verify the current environment"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands without running them"
    )
    parser.add_argument(
        "--upgrade-mujoco",
        action="store_true",
        help=(
            "install the newest mujoco and dm-control together. This can break "
            "mujoco-mjx / brax in the same environment, so by default the "
            "installed mujoco is kept and dm-control is matched to it."
        ),
    )
    parser.add_argument(
        "--memory-maze",
        action="store_true",
        help="also install Memory Maze (memory-maze, with --no-deps)",
    )
    args = parser.parse_args(argv)

    if args.check:
        ok, message = check()
        print(message)
        if ok and args.memory_maze:
            ok, message = check_memory_maze()
            print(message)
        return 0 if ok else 1

    mujoco_version = installed_mujoco_version()
    if mujoco_version is not None and not args.upgrade_mujoco:
        print(
            "keeping mujoco "
            + ".".join(str(part) for part in mujoco_version)
            + "; matching dm-control to it (--upgrade-mujoco to do the opposite)"
        )

    for command in install_steps(
        upgrade_mujoco=args.upgrade_mujoco, memory_maze=args.memory_maze
    ):
        printable = " ".join(command)
        if args.dry_run:
            print(printable)
            continue
        print(f"$ {printable}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"failed: {printable}", file=sys.stderr)
            return result.returncode

    if args.dry_run:
        return 0

    ok, message = check()
    print(message)
    if not ok:
        print(
            "DeepMind Control is still not importable. See "
            "docs/source/environments/dmc.md for the Python 3.12 fallback.",
            file=sys.stderr,
        )
    elif args.memory_maze:
        ok, message = check_memory_maze()
        print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
