"""Cross-session SSE payload enrichment."""
from __future__ import annotations

import json
from types import SimpleNamespace

from sqlalchemy import create_engine

from runtime.api_recent_events import _event_payload
from runtime.storage.event_log import EventLog
from runtime.storage.models import Base
from runtime.storage.session_store import SessionStore


def test_recent_events_session_created_payload_matches_session_summary(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    store = SessionStore(engine=engine)
    log = EventLog(engine=engine)
    inc = store.create(query="alpha", environment="dev")
    log.record(inc.id, "session.created")
    ev = list(log.iter_recent())[0]

    payload = {
        "seq": ev.seq,
        "kind": ev.kind,
        "session_id": ev.session_id,
        "payload": _event_payload(SimpleNamespace(store=store), ev),
        "ts": ev.ts,
    }
    frame = f"data: {json.dumps(payload)}\n\n"
    decoded = json.loads(frame.split("data: ", 1)[1])

    assert decoded["kind"] == "session.created"
    assert decoded["payload"]["id"] == inc.id
    assert {
        "id",
        "status",
        "created_at",
        "updated_at",
    } <= decoded["payload"].keys()
