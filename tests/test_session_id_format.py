"""``Session.id_format()`` classmethod hook.

Pins that the framework default mints a generic ``SES-YYYYMMDD-NNN``
shape (``FrameworkAppConfig.session_id_prefix`` defaults to ``"SES"``).
Apps configure their own prefix via ``session_id_prefix`` in YAML; the
classmethod itself is still overridable on a ``Session`` subclass for
fully bespoke id shapes.
"""
from __future__ import annotations

import re

from runtime.state import Session


_SES_RE = re.compile(r"^SES-\d{8}-\d{3}$")


def test_default_session_id_format_returns_generic_shape():
    """Framework default mints SES-YYYYMMDD-NNN — no domain leak."""
    sid = Session.id_format(seq=1)
    assert _SES_RE.match(sid), sid


def test_default_id_format_uses_supplied_sequence():
    sid_a = Session.id_format(seq=1)
    sid_b = Session.id_format(seq=42)
    assert sid_a.endswith("-001")
    assert sid_b.endswith("-042")


def test_id_format_accepts_custom_prefix():
    """Caller-supplied prefix flows through to the rendered id."""
    sid = Session.id_format(seq=7, prefix="HR")
    assert re.match(r"^HR-\d{8}-007$", sid), sid


def test_session_store_uses_state_class_id_format(tmp_path):
    """``SessionStore._next_id`` routes through ``state_cls.id_format``.

    Verified against a tiny ``Session`` subclass that overrides
    ``id_format`` with a non-default prefix; this is the contract apps
    rely on if they need a per-app id namespace shaped differently from
    the framework default.
    """
    from runtime.config import MetadataConfig
    from runtime.storage.engine import build_engine
    from runtime.storage.models import Base
    from runtime.storage.session_store import SessionStore

    class _CRSession(Session):
        @classmethod
        def id_format(cls, *, seq: int, prefix: str = "SES") -> str:
            # Subclass ignores ``prefix`` — bespoke shape wins.
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
