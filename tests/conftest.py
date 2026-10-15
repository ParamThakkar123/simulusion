"""Shared pytest configuration for the Simulusion suite."""

import gc
import sys

import pytest


@pytest.fixture(autouse=True)
def _reclaim_cyclic_garbage():
    """Force a cyclic collection after every test.

    Every ``nn.Module`` holds reference cycles - a module references its
    parameters and each parameter's ``grad_fn`` graph can reference back - so a
    model built inside a test is *not* freed when the last local name goes out
    of scope. Only CPython's cyclic collector can reclaim it, and that collector
    is triggered by allocation *counts*, not bytes: a handful of very large
    tensors barely moves the counter, so nothing runs it.

    Left alone the suite retains roughly 4.4GB by the end - ~1.6GB from
    tests/models/test_genie.py and ~1.4GB from tests/evals/test_evals.py alone -
    and dies partway through with either ``RuntimeError: can't start new
    thread`` or a Windows access violation, whichever limit it hits first. Both
    are the same out-of-memory condition wearing different hats.
    """
    yield
    gc.collect()
    # Only if torch is already imported - importing it here would defeat the
    # torch-free collection path that several tests rely on.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
