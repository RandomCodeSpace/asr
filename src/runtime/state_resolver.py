"""Resolve ``RuntimeConfig.state_class`` (a dotted path) to a class object.

The orchestrator calls :func:`resolve_state_class` once at construction
time and threads the resulting class through the storage layer. Doing the
import here (rather than relying on type-var introspection at runtime)
sidesteps PEP 484 generic erasure: ``Orchestrator[IncidentState]`` is
compiled away by the time we need a callable class.

Errors on:

- A dotted path that does not parse (no ``.`` separator).
- A module that fails to import.
- A module that imports but lacks the named attribute.
- An attribute that is not a subclass of :class:`runtime.state.Session`.
"""
from __future__ import annotations

import importlib
from typing import Type

from runtime.state import Session


def resolve_state_class(dotted_path: str | None) -> Type[Session]:
    """Resolve ``dotted_path`` to a concrete ``Session`` subclass.

    ``None`` and ``""`` are treated as "use the framework default
    (``runtime.state.Session``)". Any other input must be a fully
    qualified dotted import path (``pkg.module.ClassName``) pointing at a
    class that ``issubclass(Session)``.

    Raises:
        ValueError: if ``dotted_path`` is not a dotted path.
        ImportError: if the target module cannot be imported.
        AttributeError: if the module does not define the attribute.
        TypeError: if the resolved attribute is not a Session subclass.
    """
    if not dotted_path:
        return Session

    if "." not in dotted_path:
        raise ValueError(
            f"state_class must be a dotted path 'pkg.module.ClassName'; "
            f"got {dotted_path!r}"
        )

    module_path, _, attr_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"cannot import state_class module {module_path!r} "
            f"(from {dotted_path!r}): {exc}"
        ) from exc

    if not hasattr(module, attr_name):
        raise AttributeError(
            f"module {module_path!r} has no attribute {attr_name!r} "
            f"(state_class={dotted_path!r})"
        )

    cls = getattr(module, attr_name)
    if not isinstance(cls, type) or not issubclass(cls, Session):
        raise TypeError(
            f"state_class {dotted_path!r} must be a Session subclass; "
            f"got {cls!r}"
        )
    return cls
