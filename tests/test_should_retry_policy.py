"""Phase 12 (FOC-05) -- pure should_retry policy matrix.

Mirrors test_should_gate_policy.py's structure (Phase 11). All 5
RetryDecision.reason values are exercised; precedence and boundary
conditions are pinned.
"""
from __future__ import annotations

import pydantic
from pydantic import BaseModel, Field

from runtime.agents.turn_output import EnvelopeMissingError
from runtime.config import OrchestratorConfig, RetryPolicy
from runtime.policy import RetryDecision, should_retry


def _cfg(
    *,
    max_retries: int = 2,
    retry_on_transient: bool = True,
    retry_low_confidence_threshold: float = 0.4,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            retry_on_transient=retry_on_transient,
            retry_low_confidence_threshold=retry_low_confidence_threshold,
        ),
    )


# ---- auto_retry path -----------------------------------------------

def test_should_retry_returns_auto_retry_for_transient_error_under_cap():
    cfg = _cfg()
    d = should_retry(retry_count=0,
                     error=TimeoutError("net blip"),
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=True, reason="auto_retry")


def test_should_retry_returns_auto_retry_for_oserror_under_cap():
    cfg = _cfg()
    d = should_retry(retry_count=1,
                     error=OSError("conn refused"),
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=True, reason="auto_retry")


# ---- max_retries_exceeded path -------------------------------------

def test_should_retry_max_retries_exceeded_at_cap():
    cfg = _cfg(max_retries=2)
    d = should_retry(retry_count=2,
                     error=TimeoutError(),
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="max_retries_exceeded")


def test_should_retry_max_retries_exceeded_above_cap():
    cfg = _cfg(max_retries=2)
    d = should_retry(retry_count=5,
                     error=TimeoutError(),
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="max_retries_exceeded")


def test_should_retry_max_retries_zero_caps_immediately():
    cfg = _cfg(max_retries=0)
    d = should_retry(retry_count=0,
                     error=TimeoutError(),
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="max_retries_exceeded")


# ---- permanent_error path ------------------------------------------

def test_should_retry_permanent_error_pydantic_validation():
    # Build a real ValidationError instance.
    class _M(BaseModel):
        x: int = Field(ge=0)
    err: pydantic.ValidationError | None = None
    try:
        _M(x=-1)
    except pydantic.ValidationError as e:
        err = e
    assert err is not None
    cfg = _cfg()
    d = should_retry(retry_count=0, error=err,
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="permanent_error")


def test_should_retry_permanent_error_envelope_missing():
    cfg = _cfg()
    d = should_retry(
        retry_count=0,
        error=EnvelopeMissingError(agent="intake", field="confidence"),
        confidence=0.9, cfg=cfg,
    )
    assert d == RetryDecision(retry=False, reason="permanent_error")


# ---- low_confidence_no_retry path ----------------------------------

def test_should_retry_low_confidence_no_retry_with_non_transient_error():
    cfg = _cfg(retry_low_confidence_threshold=0.4)
    d = should_retry(retry_count=0,
                     error=RuntimeError("misc opaque"),
                     confidence=0.2, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="low_confidence_no_retry")


def test_should_retry_low_confidence_does_not_block_transient_retry():
    cfg = _cfg(retry_low_confidence_threshold=0.4)
    d = should_retry(retry_count=0,
                     error=TimeoutError("net blip"),
                     confidence=0.2, cfg=cfg)
    # transient takes precedence over low confidence: low_confidence gate
    # only fires for NON-transient errors. Transient classification wins.
    assert d == RetryDecision(retry=True, reason="auto_retry")


def test_should_retry_low_confidence_boundary_inclusive():
    # Strict-less-than means confidence==threshold does NOT trigger
    # low_confidence_no_retry; falls through to permanent_error
    # fail-closed default.
    cfg = _cfg(retry_low_confidence_threshold=0.4)
    d = should_retry(retry_count=0,
                     error=RuntimeError("opaque"),
                     confidence=0.4, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="permanent_error")


# ---- transient_disabled path ---------------------------------------

def test_should_retry_transient_disabled():
    cfg = _cfg(retry_on_transient=False)
    d = should_retry(retry_count=0,
                     error=TimeoutError("net blip"),
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="transient_disabled")


# ---- fail-closed default -------------------------------------------

def test_should_retry_unknown_error_falls_through_to_permanent():
    cfg = _cfg()
    d = should_retry(retry_count=0,
                     error=RuntimeError("opaque -- not in either list"),
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="permanent_error")


def test_should_retry_none_error_treated_as_permanent():
    cfg = _cfg()
    d = should_retry(retry_count=0, error=None,
                     confidence=0.9, cfg=cfg)
    assert d == RetryDecision(retry=False, reason="permanent_error")


# ---- purity --------------------------------------------------------

def test_should_retry_is_pure_no_io():
    cfg = _cfg()
    decisions = [
        should_retry(retry_count=0,
                     error=TimeoutError(),
                     confidence=0.9, cfg=cfg)
        for _ in range(5)
    ]
    assert all(d == decisions[0] for d in decisions)
    assert decisions[0] == RetryDecision(retry=True, reason="auto_retry")
