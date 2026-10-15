"""The demo recorders must at least import.

These scripts live outside the package and nothing else imports them, so a
symbol that moves inside ``simulusion`` breaks them silently: the failure only
shows up when someone runs a demo, which is exactly when it is least welcome.
``demos/record_diamond.py`` imported ``make_agent`` from ``scripts.play_diamond``
for a while after that function had moved to ``simulusion.inference.play_diamond``,
leaving the script a dead entrypoint.
"""

import importlib.util
from pathlib import Path

import pytest

DEMOS = sorted((Path(__file__).resolve().parents[1] / "demos").glob("record_*.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_demo_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", DEMOS, ids=lambda p: p.stem)
def test_demo_imports_and_exposes_main(path: Path):
    pytest.importorskip("cv2", reason="the recorders write video with OpenCV")

    module = _load(path)

    assert callable(getattr(module, "main", None)), f"{path.name} has no main()"


def test_demos_were_found():
    assert DEMOS, "no demos/record_*.py found; the glob above is wrong"
