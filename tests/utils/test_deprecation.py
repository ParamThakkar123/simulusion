"""The deprecation decorators warn once per use, at the caller, by name."""

from __future__ import annotations

import warnings

import pytest

from simulusion.utils.deprecation import (
    deprecated,
    deprecated_class,
    deprecated_function,
)


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, caught


def test_a_deprecated_function_warns_with_its_name_at_the_caller():
    @deprecated("0.5.0", reason="Use new_func instead")
    def old_func(x):
        return x + 1

    result, caught = _caught(lambda: old_func(1))
    assert result == 2
    (warning,) = caught
    assert issubclass(warning.category, DeprecationWarning)
    assert str(warning.message) == (
        "'test_a_deprecated_function_warns_with_its_name_at_the_caller."
        "<locals>.old_func' is deprecated since v0.5.0 — Use new_func instead"
    )
    assert warning.filename == __file__
    assert old_func.__name__ == "old_func"


def test_a_deprecated_class_still_constructs():
    # The wrapper once called the patched __init__, recursing forever.
    @deprecated_class("0.5.0", alternative="NewThing")
    class OldThing:
        def __init__(self, value):
            self.value = value

    thing, caught = _caught(lambda: OldThing(3))
    assert thing.value == 3
    (warning,) = caught
    assert "OldThing' is deprecated since v0.5.0 — Use NewThing instead" in str(
        warning.message
    )
    assert warning.filename == __file__


def test_a_message_without_a_reason_has_no_dangling_dash():
    @deprecated_function("1.0")
    def gone():
        return None

    _, caught = _caught(gone)
    assert str(caught[0].message).endswith("is deprecated since v1.0")


def test_warnings_can_be_made_errors():
    @deprecated("0.5.0")
    def old():
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DeprecationWarning):
            old()
