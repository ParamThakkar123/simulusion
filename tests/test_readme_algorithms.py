"""The README algorithms table is the first thing a visitor reads.

It drifted from the registry once already (5 rows against 13 registered
models), so pin it: every registry name gets a row, and every row names a real
registry entry.
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

# Rows look like: | `dreamer-v2` | **DreamerV2** | ... | ... |
_ROW = re.compile(r"^\|\s*`([a-z0-9-]+)`\s*\|", re.MULTILINE)


def _table_names() -> set[str]:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Supported Algorithms")
    end = text.index("## ", start + 1)
    return set(_ROW.findall(text[start:end]))


def test_readme_table_lists_every_registered_model():
    import simulusion

    registered = set(simulusion.list_models())
    listed = _table_names()

    assert not registered - listed, (
        f"models missing from the README table: {sorted(registered - listed)}"
    )
    assert not listed - registered, (
        f"README table lists unregistered models: {sorted(listed - registered)}"
    )
