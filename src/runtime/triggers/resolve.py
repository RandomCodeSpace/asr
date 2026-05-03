"""Resolve dotted paths to live Python objects at registry init time.

Used to bind ``payload_schema`` (a Pydantic ``BaseModel`` subclass) and
``transform`` (a callable) declared in YAML. Resolution happens once,
during ``TriggerRegistry.create`` — never per-request — so a typo
fails at startup, not at first webhook delivery.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Type

from pydantic import BaseModel


def _resolve_dotted(path: str) -> Any:
    """Import a module and return the named attribute.

    Accepts both ``a.b.c`` (last segment is the attribute) and
    ``a.b:c`` (colon-delimited per entry-point convention).
    """
    if ":" in path:
        module_path, attr = path.split(":", 1)
    else:
        module_path, _, attr = path.rpartition(".")
        if not module_path:
            raise ImportError(f"dotted path missing module: {path!r}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"cannot import module for {path!r}: {e}") from e
    if not hasattr(module, attr):
        raise ImportError(
            f"module {module_path!r} has no attribute {attr!r} "
            f"(resolving {path!r})"
        )
    return getattr(module, attr)


def resolve_payload_schema(path: str) -> Type[BaseModel]:
    """Resolve a dotted path to a Pydantic ``BaseModel`` subclass.

    Raises ``TypeError`` if the resolved object isn't a ``BaseModel``
    subclass — keeps the failure on the operator console at startup.
    """
    obj = _resolve_dotted(path)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise TypeError(
            f"payload_schema {path!r} did not resolve to a Pydantic "
            f"BaseModel subclass; got {obj!r}"
        )
    return obj


def resolve_transform(path: str) -> Callable[..., dict]:
    """Resolve a dotted path to a callable.

    The callable is expected to return a ``dict`` of keyword arguments
    suitable for ``Orchestrator.start_session(**kwargs)``. The framework
    does not enforce a stricter signature — apps own the contract with
    their own transform.
    """
    obj = _resolve_dotted(path)
    if not callable(obj):
        raise TypeError(
            f"transform {path!r} did not resolve to a callable; got {obj!r}"
        )
    return obj
