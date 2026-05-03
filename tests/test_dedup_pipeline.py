"""Unit tests for the dedup pipeline (framework-level).

Stage 1 wraps ``HistoryStore.find_similar`` and applies the top-K +
threshold cap; Stage 2 calls a stubbed LLM and parses Pydantic-typed
structured output. Tests assert short-circuit semantics, threshold
edges, and graceful handling of malformed LLM output.

The pipeline is generic over state — these tests use a tiny test-only
``BaseModel`` subclass to prove R4 (no import of IncidentState).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from pydantic import BaseModel

from runtime.dedup import (
    DedupConfig,
    DedupDecision,
    DedupPipeline,
    DedupResult,
    DedupScope,
    _parse_decision,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _SimpleSession(BaseModel):
    """Bare-bones state used by these tests. Deliberately NOT IncidentState."""
    id: str
    environment: str = ""
    text: str = ""


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _ScriptedLLM:
    """Sequence-driven stub: returns the next scripted response on each call."""

    def __init__(self, scripted: list[Any]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    async def ainvoke(self, _messages):  # noqa: ANN001 — match LangChain shape
        self.calls += 1
        if not self._scripted:
            raise RuntimeError("LLM stub exhausted")
        nxt = self._scripted.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        if isinstance(nxt, dict):
            return _StubMessage(json.dumps(nxt))
        return _StubMessage(str(nxt))


class _FakeHistoryStore:
    """Returns whatever ``find_similar`` is configured to return.

    Captures the kwargs of the most recent call so tests can assert
    that Stage 1 forwarded ``filter_kwargs`` / ``threshold`` correctly.
    """

    def __init__(self, results: list[tuple[Any, float]]) -> None:
        self.results = list(results)
        self.last_kwargs: dict[str, Any] = {}

    def find_similar(self, **kwargs):
        self.last_kwargs = kwargs
        return list(self.results)


def _extract_text(s: _SimpleSession) -> str:
    return s.text


def _no_llm():  # pragma: no cover — should never be called in stage-1-only tests
    raise AssertionError("LLM factory must not be invoked")


# ---------------------------------------------------------------------------
# DedupConfig validation
# ---------------------------------------------------------------------------


def test_dedup_config_defaults_off():
    cfg = DedupConfig()
    assert cfg.enabled is False
    assert cfg.stage1_threshold == 0.6
    assert cfg.stage2_model == "cheap"
    assert cfg.run_at == "post_intake"


def test_dedup_config_rejects_top_k_inversion():
    with pytest.raises(ValueError, match="stage2_top_k"):
        DedupConfig(stage1_top_k=2, stage2_top_k=5)


def test_dedup_config_threshold_bounds():
    with pytest.raises(ValueError):
        DedupConfig(stage2_min_confidence=1.5)
    with pytest.raises(ValueError):
        DedupConfig(stage1_threshold=-0.1)


def test_dedup_config_invalid_run_at_rejected():
    # ``post_intake`` is the only legal value at P7.
    with pytest.raises(ValueError):
        DedupConfig(run_at="passive")  # type: ignore[arg-type]


def test_dedup_config_assert_model_exists():
    from runtime.config import LLMConfig
    cfg = DedupConfig(enabled=True, stage2_model="not-a-model")
    with pytest.raises(ValueError, match="dedup.stage2_model"):
        cfg.assert_model_exists(LLMConfig.stub())


# ---------------------------------------------------------------------------
# Decision parser
# ---------------------------------------------------------------------------


def test_parse_decision_accepts_clean_json():
    out = _parse_decision('{"is_duplicate": true, "confidence": 0.9, '
                          '"rationale": "same root cause"}')
    assert isinstance(out, DedupDecision)
    assert out.is_duplicate is True
    assert out.confidence == 0.9


def test_parse_decision_strips_code_fence():
    raw = '```json\n{"is_duplicate": false, "confidence": 0.1, "rationale": "n/a"}\n```'
    out = _parse_decision(raw)
    assert out is not None
    assert out.is_duplicate is False


def test_parse_decision_returns_none_on_malformed():
    assert _parse_decision("not json") is None
    assert _parse_decision("") is None
    assert _parse_decision('{"is_duplicate": "yes"}') is None  # wrong type


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_disabled_returns_no_match_without_calling_history():
    cfg = DedupConfig(enabled=False)
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=_no_llm,
    )
    history = _FakeHistoryStore(results=[])
    result = await pipeline.run(
        session=_SimpleSession(id="A", environment="prod", text="x"),
        history_store=history,
    )
    assert result == DedupResult(matched=False)
    # Disabled => Stage 1 never runs.
    assert history.last_kwargs == {}


@pytest.mark.asyncio
async def test_stage1_drops_self_id():
    cfg = DedupConfig(enabled=True)
    sess = _SimpleSession(id="A", environment="prod", text="payments timeout")
    history = _FakeHistoryStore(results=[(sess, 0.95)])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=_no_llm,
    )
    result = await pipeline.run(session=sess, history_store=history)
    # The only candidate was itself — Stage 2 must not run.
    assert result.matched is False


@pytest.mark.asyncio
async def test_stage1_no_candidates_skips_stage2():
    cfg = DedupConfig(enabled=True)
    history = _FakeHistoryStore(results=[])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=_no_llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="A", environment="prod", text="x"),
        history_store=history,
    )
    assert result.matched is False
    # Confirms threshold/filter forwarded.
    assert history.last_kwargs["threshold"] == cfg.stage1_threshold
    assert history.last_kwargs["filter_kwargs"] == {"environment": "prod"}
    assert history.last_kwargs["status_filter"] == "resolved"
    assert history.last_kwargs["limit"] == cfg.stage1_top_k


@pytest.mark.asyncio
async def test_stage1_threshold_inclusive():
    """Hits at exactly ``stage1_threshold`` are kept (>=, not >)."""
    cfg = DedupConfig(enabled=True, stage1_threshold=0.6)
    prior = _SimpleSession(id="X", environment="prod", text="payments")
    history = _FakeHistoryStore(results=[(prior, 0.6)])
    llm = _ScriptedLLM([
        {"is_duplicate": True, "confidence": 0.9, "rationale": "match"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="A", environment="prod", text="payments"),
        history_store=history,
    )
    assert result.matched is True
    assert result.parent_session_id == "X"


@pytest.mark.asyncio
async def test_stage1_drops_below_threshold_locally():
    """Defensive: HistoryStore screens by threshold, but pipeline double-checks."""
    cfg = DedupConfig(enabled=True, stage1_threshold=0.7)
    prior = _SimpleSession(id="X", environment="prod", text="payments")
    # HistoryStore returns one below the configured threshold (e.g. a
    # subclass that ignores the kwarg) — the pipeline must still drop it.
    history = _FakeHistoryStore(results=[(prior, 0.5)])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=_no_llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="A", environment="prod", text="payments"),
        history_store=history,
    )
    assert result.matched is False


@pytest.mark.asyncio
async def test_stage1_handles_history_store_error():
    cfg = DedupConfig(enabled=True)

    class _BoomStore:
        def find_similar(self, **kwargs):
            raise RuntimeError("db down")

    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=_no_llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="A", environment="prod", text="x"),
        history_store=_BoomStore(),  # type: ignore[arg-type]
    )
    # Hard requirement: dedup never crashes the intake pipeline.
    assert result.matched is False


@pytest.mark.asyncio
async def test_stage1_scope_disables_environment_filter():
    cfg = DedupConfig(enabled=True, scope=DedupScope(same_environment=False))
    history = _FakeHistoryStore(results=[])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=_no_llm,
    )
    await pipeline.run(
        session=_SimpleSession(id="A", environment="prod", text="x"),
        history_store=history,
    )
    assert history.last_kwargs["filter_kwargs"] is None


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage2_confirms_high_confidence():
    cfg = DedupConfig(enabled=True, stage2_min_confidence=0.7)
    prior = _SimpleSession(id="P1", environment="prod", text="prior text")
    history = _FakeHistoryStore(results=[(prior, 0.95)])
    llm = _ScriptedLLM([
        {"is_duplicate": True, "confidence": 0.9, "rationale": "same fault"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new text"),
        history_store=history,
    )
    assert result.matched is True
    assert result.parent_session_id == "P1"
    assert result.candidate_id == "P1"
    assert result.decision.confidence == 0.9
    assert result.stage1_score == 0.95
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_stage2_rejects_low_confidence_match():
    cfg = DedupConfig(enabled=True, stage2_min_confidence=0.7)
    prior = _SimpleSession(id="P1", environment="prod", text="prior")
    history = _FakeHistoryStore(results=[(prior, 0.95)])
    llm = _ScriptedLLM([
        {"is_duplicate": True, "confidence": 0.5, "rationale": "maybe"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new"),
        history_store=history,
    )
    assert result.matched is False


@pytest.mark.asyncio
async def test_stage2_rejects_is_duplicate_false_even_at_high_confidence():
    cfg = DedupConfig(enabled=True)
    prior = _SimpleSession(id="P1", environment="prod", text="prior")
    history = _FakeHistoryStore(results=[(prior, 0.99)])
    llm = _ScriptedLLM([
        {"is_duplicate": False, "confidence": 0.99, "rationale": "different"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new"),
        history_store=history,
    )
    assert result.matched is False


@pytest.mark.asyncio
async def test_stage2_treats_malformed_json_as_not_duplicate():
    cfg = DedupConfig(enabled=True)
    prior = _SimpleSession(id="P1", environment="prod", text="prior")
    history = _FakeHistoryStore(results=[(prior, 0.95)])
    llm = _ScriptedLLM(["not json at all"])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new"),
        history_store=history,
    )
    assert result.matched is False
    # Single attempt — no retry.
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_stage2_caps_at_top_k_even_if_stage1_returns_more():
    cfg = DedupConfig(enabled=True, stage1_top_k=5, stage2_top_k=2)
    history = _FakeHistoryStore(results=[
        (_SimpleSession(id=f"P{i}", environment="prod", text=f"p{i}"), 0.9)
        for i in range(5)
    ])
    # All replies decline so Stage 2 visits up to stage2_top_k candidates.
    llm = _ScriptedLLM([
        {"is_duplicate": False, "confidence": 0.99, "rationale": "diff"},
        {"is_duplicate": False, "confidence": 0.99, "rationale": "diff"},
        {"is_duplicate": False, "confidence": 0.99, "rationale": "diff"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new"),
        history_store=history,
    )
    assert llm.calls == 2  # capped at stage2_top_k


@pytest.mark.asyncio
async def test_stage2_short_circuits_on_first_confirm():
    cfg = DedupConfig(enabled=True, stage2_top_k=3)
    history = _FakeHistoryStore(results=[
        (_SimpleSession(id="P1", environment="prod", text="x"), 0.95),
        (_SimpleSession(id="P2", environment="prod", text="y"), 0.90),
        (_SimpleSession(id="P3", environment="prod", text="z"), 0.85),
    ])
    llm = _ScriptedLLM([
        {"is_duplicate": True, "confidence": 0.9, "rationale": "match"},
        {"is_duplicate": True, "confidence": 0.9, "rationale": "match"},
        {"is_duplicate": True, "confidence": 0.9, "rationale": "match"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new"),
        history_store=history,
    )
    assert result.matched is True
    assert result.parent_session_id == "P1"
    assert llm.calls == 1  # short-circuit on first confirm


@pytest.mark.asyncio
async def test_stage2_llm_call_failure_continues_to_next_candidate():
    cfg = DedupConfig(enabled=True, stage2_top_k=3)
    history = _FakeHistoryStore(results=[
        (_SimpleSession(id="P1", environment="prod", text="x"), 0.95),
        (_SimpleSession(id="P2", environment="prod", text="y"), 0.85),
    ])
    llm = _ScriptedLLM([
        RuntimeError("transient"),
        {"is_duplicate": True, "confidence": 0.9, "rationale": "ok"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new"),
        history_store=history,
    )
    assert result.matched is True
    assert result.parent_session_id == "P2"


# ---------------------------------------------------------------------------
# Decoupling enforcement (R4)
# ---------------------------------------------------------------------------


def test_dedup_module_does_not_import_incident_state():
    """``runtime.dedup`` must not depend on the example-app state class."""
    import runtime.dedup as mod
    src = (mod.__file__ or "")
    if not src:  # pragma: no cover — defensive
        pytest.skip("module has no source path")
    text = open(src, encoding="utf-8").read()
    assert "examples.incident_management" not in text, (
        "P7 R4: runtime.dedup must not import example-app modules"
    )
    assert "IncidentState" not in text, (
        "P7 R4: runtime.dedup must not reference IncidentState"
    )


# ---------------------------------------------------------------------------
# Stage 2 parse-failure observability (parse_failures counter + warning log)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_failure_logs_warning(caplog):
    """Garbage Stage 2 output emits a structured ``dedup_parse_failure`` warning."""
    cfg = DedupConfig(enabled=True, stage2_model="cheap")
    prior = _SimpleSession(id="P1", environment="prod", text="prior")
    history = _FakeHistoryStore(results=[(prior, 0.95)])
    llm = _ScriptedLLM(["totally not json {{{"])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    with caplog.at_level(logging.WARNING, logger="runtime.dedup"):
        await pipeline.run(
            session=_SimpleSession(id="N1", environment="prod", text="new"),
            history_store=history,
        )

    parse_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "dedup_parse_failure"
    ]
    assert parse_records, (
        "expected at least one structured 'dedup_parse_failure' warning"
    )
    rec = parse_records[0]
    assert rec.levelno == logging.WARNING
    assert rec.event == "dedup_parse_failure"
    assert rec.error_type  # non-empty exception class name
    assert rec.error_msg
    assert rec.model == "cheap"
    # Excerpt is bounded and reflects the bad payload.
    assert isinstance(rec.raw_output_excerpt, str)
    assert len(rec.raw_output_excerpt) <= 200


@pytest.mark.asyncio
async def test_parse_failure_increments_counter():
    """Stage-2 garbage feeds the ``parse_failures`` counter and yields no match."""
    cfg = DedupConfig(enabled=True)
    prior = _SimpleSession(id="P1", environment="prod", text="prior")
    history = _FakeHistoryStore(results=[(prior, 0.95)])
    llm = _ScriptedLLM(["not json"])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    result = await pipeline.run(
        session=_SimpleSession(id="N1", environment="prod", text="new"),
        history_store=history,
    )
    assert result.matched is False
    assert result.parse_failures == 1


@pytest.mark.asyncio
async def test_clean_parse_does_not_log_or_increment(caplog):
    """Happy path: well-formed JSON yields no warning and no counter bump."""
    cfg = DedupConfig(enabled=True, stage2_min_confidence=0.7)
    prior = _SimpleSession(id="P1", environment="prod", text="prior")
    history = _FakeHistoryStore(results=[(prior, 0.95)])
    llm = _ScriptedLLM([
        {"is_duplicate": True, "confidence": 0.9, "rationale": "match"},
    ])
    pipeline = DedupPipeline(
        config=cfg, text_extractor=_extract_text, model_factory=lambda: llm,
    )
    with caplog.at_level(logging.WARNING, logger="runtime.dedup"):
        result = await pipeline.run(
            session=_SimpleSession(id="N1", environment="prod", text="new"),
            history_store=history,
        )

    assert result.matched is True
    assert result.parse_failures == 0
    parse_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "dedup_parse_failure"
    ]
    assert parse_records == []
