"""Configurable session-id prefix.

Pins the FrameworkAppConfig.session_id_prefix knob and its threading
through SessionStore -> Session.id_format. Apps pick their own session
id namespace via plain config (``INC`` for incident management,
``REVIEW`` for code review, ``HR`` for HR cases, ...); the framework
default ``SES`` keeps the surface domain-neutral.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from runtime.config import FrameworkAppConfig
from runtime.state import Session


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def test_default_prefix_is_ses():
    """``Session.id_format(seq=1)`` returns the generic ``SES-`` shape."""
    sid = Session.id_format(seq=1)
    assert sid == f"SES-{_today()}-001", sid


def test_custom_prefix_threads_through_session_store(tmp_path):
    """A SessionStore built with ``id_prefix="HR"`` mints ``HR-...`` ids.

    Verifies the integration boundary: the prefix supplied at store
    construction wins over the framework default and shows up in the
    rendered id without each app having to subclass ``Session``.
    """
    from runtime.config import MetadataConfig
    from runtime.storage.engine import build_engine
    from runtime.storage.models import Base
    from runtime.storage.session_store import SessionStore

    engine = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/state.db"))
    Base.metadata.create_all(engine)

    store = SessionStore(engine=engine, id_prefix="HR")
    inc = store.create(
        query="leave balance question", environment="prod",
    )
    assert inc.id.startswith("HR-"), inc.id
    assert re.match(r"^HR-\d{8}-\d{3}$", inc.id), inc.id


def test_invalid_prefix_rejected():
    """Whitespace and symbols raise a Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        FrameworkAppConfig(session_id_prefix="not valid!")


@pytest.mark.parametrize("bad", ["", " ", "has space", "INC!", "x" * 17, "under_score"])
def test_invalid_prefix_variants_rejected(bad: str):
    """Empty, whitespace, symbols, underscores, and >16 chars are all rejected."""
    with pytest.raises(ValidationError):
        FrameworkAppConfig(session_id_prefix=bad)


@pytest.mark.parametrize("good", ["INC", "REVIEW", "HR", "SES", "A1", "MY-APP", "x" * 16])
def test_valid_prefix_variants_accepted(good: str):
    """Alphanumerics + hyphens up to 16 chars all validate cleanly."""
    cfg = FrameworkAppConfig(session_id_prefix=good)
    assert cfg.session_id_prefix == good
