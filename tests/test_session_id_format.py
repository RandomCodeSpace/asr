"""P8-C: ``Session.id_format()`` classmethod hook.

Each app's state subclass mints its own session id format so the
framework's ``SessionStore._next_id`` is no longer hard-coded to the
incident ``INC-YYYYMMDD-NNN`` shape.

Contract:

* ``Session.id_format(seq)`` is the framework default — keeps the
  legacy ``INC-YYYYMMDD-NNN`` shape so apps that have **not**
  overridden it (and any existing on-disk rows) round-trip cleanly.
* ``IncidentState.id_format(seq)`` keeps the same legacy ``INC-`` shape
  explicitly.
* ``CodeReviewState.id_format(seq)`` returns the code-review shape
  ``CR-YYYYMMDD-NNN`` so the two apps cannot collide on id space.
"""
from __future__ import annotations

import re

from runtime.state import Session


_INC_RE = re.compile(r"^INC-\d{8}-\d{3}$")
_CR_RE = re.compile(r"^CR-\d{8}-\d{3}$")


def test_default_session_id_format_returns_inc_shape():
    """Framework default keeps the legacy INC- shape for back-compat."""
    sid = Session.id_format(seq=1)
    assert _INC_RE.match(sid), sid


def test_default_id_format_uses_supplied_sequence():
    sid_a = Session.id_format(seq=1)
    sid_b = Session.id_format(seq=42)
    assert sid_a.endswith("-001")
    assert sid_b.endswith("-042")


def test_incident_state_id_format_matches_legacy_regex():
    from examples.incident_management.state import IncidentState, _INC_ID_RE

    sid = IncidentState.id_format(seq=7)
    assert _INC_ID_RE.match(sid), sid


def test_code_review_state_id_format_distinct_from_incident():
    from examples.code_review.state import CodeReviewState

    sid = CodeReviewState.id_format(seq=3)
    assert _CR_RE.match(sid), sid
    assert not _INC_RE.match(sid)


def test_session_store_uses_state_class_id_format(tmp_path):
    """``SessionStore._next_id`` routes through ``state_cls.id_format``."""
    from examples.code_review.state import CodeReviewState
    from runtime.config import MetadataConfig
    from runtime.storage.engine import build_engine
    from runtime.storage.models import Base
    from runtime.storage.session_store import SessionStore

    engine = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/state.db"))
    Base.metadata.create_all(engine)

    store = SessionStore[CodeReviewState](engine=engine, state_cls=CodeReviewState)
    # _next_id is the seam; we just want to confirm it returns a CR- id
    # without going through ``create()`` (which the code-review state
    # cannot honour today because the row schema needs PR fields).
    from sqlalchemy.orm import Session as SqlSession

    with SqlSession(engine) as session:
        sid = store._next_id(session)
    assert _CR_RE.match(sid), sid
