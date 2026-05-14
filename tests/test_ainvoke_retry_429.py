"""Pin the two backoff regimes in ``runtime.graph._ainvoke_with_retry``.

* 5xx / streaming hiccups retry on a short-window backoff (``base_delay``).
* 429 rate-limit responses retry on a longer-window backoff
  (``rate_limit_base_delay``) — free / shared upstream tiers (e.g.
  OpenRouter ``…:free`` models) throttle on windows that need
  30-60s to clear; the 5xx default exhausts retries before the
  window opens again.

A single non-transient error (e.g. 401 / 422) propagates immediately
without retry so quota / schema / auth issues fail fast.
"""
from __future__ import annotations

import pytest

from runtime.graph import _ainvoke_with_retry


class _RecordingExecutor:
    """A fake agent executor whose ``ainvoke`` raises a configurable
    sequence of exceptions; the final entry can be a return value."""

    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.calls = 0

    async def ainvoke(self, _input, **_kwargs):
        self.calls += 1
        item = self._sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.mark.asyncio
async def test_retries_on_5xx_and_returns_eventually(monkeypatch):
    """5xx burst clears within the short-window backoff."""
    sleeps: list[float] = []

    async def _fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("runtime.graph.asyncio.sleep", _fake_sleep)

    exec_ = _RecordingExecutor([
        RuntimeError("Internal server error 500 from upstream"),
        {"messages": []},
    ])
    result = await _ainvoke_with_retry(exec_, {"messages": []})
    assert result == {"messages": []}
    assert exec_.calls == 2
    assert sleeps == [1.5]  # base_delay * (attempt+1) on the first retry


@pytest.mark.asyncio
async def test_retries_on_429_with_longer_backoff(monkeypatch):
    """429 retries fire on the rate_limit_base_delay (7.5s+) window
    instead of the 1.5s default — free upstream tiers need 30-60s
    to clear and the 5xx default would exhaust 3 attempts in 9s.
    """
    sleeps: list[float] = []

    async def _fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("runtime.graph.asyncio.sleep", _fake_sleep)

    exec_ = _RecordingExecutor([
        RuntimeError("Error code: 429 - rate limit exceeded"),
        RuntimeError("Error code: 429 - still rate-limited"),
        {"messages": []},
    ])
    result = await _ainvoke_with_retry(exec_, {"messages": []})
    assert result == {"messages": []}
    assert exec_.calls == 3
    # 7.5s on attempt 1, 15s on attempt 2 — confirms the rate-limit
    # branch fired with the longer backoff per attempt.
    assert sleeps == [7.5, 15.0]


@pytest.mark.asyncio
async def test_429_phrasings_all_match(monkeypatch):
    """Markers cover the variants real providers emit. Each phrase
    on its own should hit the 429 retry branch (assertable via the
    longer backoff)."""
    monkeypatch.setattr(
        "runtime.graph.asyncio.sleep",
        lambda _s: _noop(),  # type: ignore[arg-type]
    )

    async def _noop():
        return None

    for phrase in (
        "RateLimitError: too many requests",
        "Provider returned 429",
        "Status code: 429",
        "rate limited upstream",
        "rate-limited",
    ):
        exec_ = _RecordingExecutor([
            RuntimeError(phrase),
            {"messages": []},
        ])
        result = await _ainvoke_with_retry(exec_, {"messages": []})
        assert result == {"messages": []}, f"failed to retry on {phrase!r}"
        assert exec_.calls == 2, f"unexpected call count for {phrase!r}"


@pytest.mark.asyncio
async def test_non_transient_error_propagates_without_retry(monkeypatch):
    """4xx (other than 429) and validation errors must fail fast —
    retrying a 401/422 wastes time and masks the real problem."""
    sleeps: list[float] = []

    async def _fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("runtime.graph.asyncio.sleep", _fake_sleep)

    exec_ = _RecordingExecutor([
        RuntimeError("Error code: 401 - unauthorized"),
    ])
    with pytest.raises(RuntimeError, match="401"):
        await _ainvoke_with_retry(exec_, {"messages": []})
    assert exec_.calls == 1
    assert sleeps == []  # no retry → no sleep


@pytest.mark.asyncio
async def test_429_exhausts_max_attempts_then_raises(monkeypatch):
    """If the rate-limit window doesn't clear within max_attempts
    retries, the last 429 propagates — bounded so we don't loop
    forever on a real quota exhaustion."""
    sleeps: list[float] = []

    async def _fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("runtime.graph.asyncio.sleep", _fake_sleep)

    exec_ = _RecordingExecutor([
        RuntimeError("Error code: 429 - rate limit"),
        RuntimeError("Error code: 429 - rate limit"),
        RuntimeError("Error code: 429 - rate limit"),
    ])
    with pytest.raises(RuntimeError, match="429"):
        await _ainvoke_with_retry(exec_, {"messages": []})
    assert exec_.calls == 3
    # Only TWO sleeps: backoff happens BEFORE retries 2 and 3, and
    # the loop refuses to sleep before raising on the final attempt.
    assert sleeps == [7.5, 15.0]
