"""``Session.id_format()`` classmethod hook.

Pins that the framework default mints the legacy ``INC-YYYYMMDD-NNN``
shape so apps that have **not** overridden ``id_format`` (and any
existing on-disk rows) round-trip cleanly.

The per-app overrides on ``IncidentState`` / ``CodeReviewState`` were
removed with the move to ``Session.extra_fields``; per-app id namespaces
are now the responsibility of an app-supplied ``Session`` subclass — or
deferred entirely until a second app actually ships.
"""
from __future__ import annotations

import re

from runtime.state import Session


_INC_RE = re.compile(r"^INC-\d{8}-\d{3}$")


def test_default_session_id_format_returns_inc_shape():
    """Framework default keeps the legacy INC- shape for back-compat."""
    sid = Session.id_format(seq=1)
    assert _INC_RE.match(sid), sid


def test_default_id_format_uses_supplied_sequence():
    sid_a = Session.id_format(seq=1)
    sid_b = Session.id_format(seq=42)
    assert sid_a.endswith("-001")
    assert sid_b.endswith("-042")


def test_session_store_uses_state_class_id_format(tmp_path):
    """``SessionStore._next_id`` routes through ``state_cls.id_format``.

    Verified against a tiny ``Session`` subclass that overrides
    ``id_format`` with a non-INC prefix; this is the contract apps
    rely on if they need a per-app id namespace.
    """
    from runtime.config import MetadataConfig
    from runtime.storage.engine import build_engine
    from runtime.storage.models import Base
    from runtime.storage.session_store import SessionStore

    class _CRSession(Session):
        @classmethod
        def id_format(cls, *, seq: int) -> str:
            from datetime import datetime, timezone

            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            return f"CR-{today}-{seq:03d}"

    _cr_re = re.compile(r"^CR-\d{8}-\d{3}$")

    engine = build_engine(MetadataConfig(url=f"sqlite:///{tmp_path}/state.db"))
    Base.metadata.create_all(engine)

    store = SessionStore[_CRSession](engine=engine, state_cls=_CRSession)
    from sqlalchemy.orm import Session as SqlSession

    with SqlSession(engine) as session:
        sid = store._next_id(session)
    assert _cr_re.match(sid), sid
