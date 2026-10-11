"""The shipped stub must cover the whole public surface, and stay in sync.

``simulusion`` ships ``py.typed``, so type checkers trust ``simulusion/__init__.pyi``
completely.  A stale stub is therefore worse than no stub: it silently hands
users ``Any`` for symbols that do exist, or reports missing ones that do not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_PATH = REPO_ROOT / "simulusion" / "__init__.pyi"


def _stub_tree() -> ast.Module:
    return ast.parse(STUB_PATH.read_text(encoding="utf-8"))


def _stub_names() -> set[str]:
    """Names the stub binds - aliased imports plus annotated variables."""

    names: set[str] = set()
    for node in _stub_tree().body:
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_stub_declares_every_public_export():
    import simulusion

    missing = sorted(set(simulusion.__all__) - _stub_names())
    assert not missing, f"symbols missing from {STUB_PATH.name}: {missing}"


def test_stub_is_regenerated_from_the_current_export_map():
    pytest.importorskip("torch")
    from tools.gen_type_stub import MissingOptionalBackends, main

    try:
        assert main(["--check"]) == 0
    except MissingOptionalBackends as exc:
        # A base install cannot resolve the env-backend exports, so the
        # regenerated stub would not be comparable. The full-extras CI job
        # installs everything and does check this for real.
        pytest.skip(f"optional backends missing: {exc}")


def test_stub_re_exports_resolve_to_the_runtime_objects():
    # Every ``from X import Y as Z`` in the stub must name the module that
    # really defines the object ``simulusion.Z`` returns, or editors send users to
    # a symbol that is not there.
    import importlib

    import simulusion

    mismatched = []
    checked = 0
    for node in _stub_tree().body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            exported = alias.asname or alias.name
            if exported not in simulusion.__all__:
                continue
            try:
                source = importlib.import_module(node.module)
                runtime = getattr(simulusion, exported)
            except ModuleNotFoundError:
                # Backend for an extra that is not installed here - the stub
                # entry cannot be wrong about an object nothing can import.
                continue
            checked += 1
            if getattr(source, alias.name, None) is not runtime:
                mismatched.append(f"{node.module}:{alias.name} as {exported}")

    assert not mismatched, f"stub re-exports point at the wrong object: {mismatched}"
    assert checked, "no stub re-export could be checked - the sweep did nothing"


def test_package_ships_py_typed():
    # ``simulusion`` ships ``py.typed`` so checkers trust ``__init__.pyi``; without
    # the marker they treat every import from the package as untyped.
    assert (REPO_ROOT / "simulusion" / "py.typed").exists()
