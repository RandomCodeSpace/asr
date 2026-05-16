"""GET /api/v1/sessions/{id}/full returns the bootstrap bundle.

Single round-trip on session open: replaces multiple GETs.
Returns: session, agents_run, tool_calls, events, agent_definitions, vm_seq.
"""
from fastapi.testclient import TestClient
from runtime.api import build_app
from tests.test_api_v1_url_move import _cfg


def test_full_returns_404_for_unknown_session(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/v1/sessions/SES-20990101-999/full")
        assert r.status_code == 404
        body = r.json()
        assert body["error"]["code"] == "not_found"


def test_full_returns_complete_bundle_for_existing_session(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        # Create a session first
        r = client.post("/api/v1/sessions", json={
            "query": "test bootstrap", "environment": "dev",
            "submitter": {"id": "u1", "team": "platform"},
        })
        assert r.status_code == 201
        sid = r.json()["session_id"]

        # Bootstrap fetch
        r = client.get(f"/api/v1/sessions/{sid}/full")
        assert r.status_code == 200
        body = r.json()
        # All five keys present
        assert set(body.keys()) >= {
            "session", "agents_run", "tool_calls", "events", "agent_definitions", "vm_seq"
        }
        assert body["session"]["id"] == sid
        assert isinstance(body["agents_run"], list)
        assert isinstance(body["tool_calls"], list)
        assert isinstance(body["events"], list)
        assert isinstance(body["agent_definitions"], dict)
        assert isinstance(body["vm_seq"], int)


def test_full_agent_definitions_includes_skill_metadata(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/v1/sessions", json={
            "query": "x", "environment": "dev",
            "submitter": {"id": "u1", "team": "p"},
        })
        sid = r.json()["session_id"]

        r = client.get(f"/api/v1/sessions/{sid}/full")
        defs = r.json()["agent_definitions"]
        # Each agent has at least: name, kind, model, tools, routes
        for name, d in defs.items():
            assert "name" in d
            assert "kind" in d
            assert "model" in d
            assert "tools" in d  # list of tool ref strings
            assert "routes" in d  # dict of signal -> next agent


def test_full_vm_seq_matches_event_log_length(tmp_path):
    app = build_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/v1/sessions", json={
            "query": "x", "environment": "dev",
            "submitter": {"id": "u1", "team": "p"},
        })
        sid = r.json()["session_id"]

        r = client.get(f"/api/v1/sessions/{sid}/full")
        body = r.json()
        # vm_seq is the max event seq the bundle includes; if no events, both 0
        assert body["vm_seq"] >= 0
        if body["events"]:
            assert body["vm_seq"] == max(ev["seq"] for ev in body["events"])
        else:
            assert body["vm_seq"] == 0
