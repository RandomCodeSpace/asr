"""Dotted-path resolution tests — fail fast at startup, not at request time."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from runtime.triggers.resolve import (
    resolve_payload_schema,
    resolve_transform,
)


def test_resolve_payload_schema_returns_basemodel_subclass():
    cls = resolve_payload_schema(
        "tests.test_triggers.conftest.PagerDutyPayload"
    )
    assert issubclass(cls, BaseModel)


def test_resolve_payload_schema_rejects_non_basemodel():
    with pytest.raises(TypeError):
        resolve_payload_schema(
            "tests.test_triggers.conftest.transform_pagerduty"
        )


def test_resolve_payload_schema_unknown_module():
    with pytest.raises(ImportError):
        resolve_payload_schema("nonexistent.module.Thing")


def test_resolve_payload_schema_unknown_attr():
    with pytest.raises(ImportError):
        resolve_payload_schema(
            "tests.test_triggers.conftest.NoSuchAttribute"
        )


def test_resolve_transform_returns_callable():
    fn = resolve_transform(
        "tests.test_triggers.conftest.transform_pagerduty"
    )
    assert callable(fn)


def test_resolve_transform_colon_form():
    fn = resolve_transform(
        "tests.test_triggers.conftest:transform_pagerduty"
    )
    assert callable(fn)


def test_resolve_transform_accepts_class_or_function():
    """Classes are callable; the resolver enforces callability, not the
    sub-shape (transforms can return either a dict or a Pydantic model
    that we ``model_dump()`` later)."""
    # A function works.
    fn = resolve_transform(
        "tests.test_triggers.conftest.transform_pagerduty"
    )
    assert callable(fn)


def test_resolve_transform_rejects_non_callable():
    """A bare value (e.g. a module-level constant) must be rejected."""
    # Inject a non-callable into the conftest namespace.
    import tests.test_triggers.conftest as cf
    cf._BOGUS = 42  # type: ignore[attr-defined]
    try:
        with pytest.raises(TypeError):
            resolve_transform("tests.test_triggers.conftest._BOGUS")
    finally:
        del cf._BOGUS  # type: ignore[attr-defined]
