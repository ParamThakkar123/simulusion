from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    version: str,
    reason: str = "",
    stacklevel: int = 3,
    category: type[Warning] = DeprecationWarning,
) -> Callable[[F], F]:
    """Mark a function, method, or class as deprecated.

    Usage::

        @deprecated(version="0.5.0", reason="Use new_func instead")
        def old_func():
            ...

    The warning is emitted on every call for functions/methods, or on
    instantiation for classes.
    """

    def decorator(obj: F) -> F:
        if inspect.isclass(obj):

            @functools.wraps(obj.__init__)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(
                    _format_message(obj.__name__, version, reason, "class"),
                    category,
                    stacklevel=stacklevel,
                )
                obj.__init__(self, *args, **kwargs)

            obj.__init__ = wrapped_init
            return cast(F, obj)

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target = type(obj).__name__ if inspect.isfunction(obj) else obj.__name__
            warnings.warn(
                _format_message(target, version, reason, "function"),
                category,
                stacklevel=stacklevel,
            )
            return obj(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def deprecated_class(
    version: str,
    reason: str = "",
    alternative: str | None = None,
) -> Callable[[type], type]:
    """Shortcut decorator for deprecating a class with an alternative."""
    msg = reason or (f"Use {alternative} instead" if alternative else "")
    return deprecated(version=version, reason=msg)


def deprecated_function(
    version: str,
    reason: str = "",
    alternative: str | None = None,
) -> Callable[[F], F]:
    """Shortcut decorator for deprecating a function with an alternative."""
    msg = reason or (f"Use {alternative} instead" if alternative else "")
    return deprecated(version=version, reason=msg)


def _format_message(
    name: str,
    version: str,
    reason: str,
    kind: str,
) -> str:
    parts = [f"{name!r} is deprecated since v{version}"]
    if reason:
        parts.append(f" — {reason}")
    return " ".join(parts)


__all__ = [
    "deprecated",
    "deprecated_class",
    "deprecated_function",
]
