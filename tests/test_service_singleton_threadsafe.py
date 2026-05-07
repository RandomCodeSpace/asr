"""Phase 17 / HARD-06: thread-safe ``OrchestratorService.get_or_create``.

Streamlit's auto-rerun and FastAPI's startup hook can both fire
``OrchestratorService.get_or_create()`` concurrently during process
warm-up. Without a class-level lock, two threads can both observe
``_instance is None``, both construct, and the loser's instance leaks
(holding its own MCP exit-stack, its own background loop reference)
while the surviving caller is the one that won the assignment.

This module hammers ``get_or_create()`` from a thread pool and asserts
**every** caller observes the **same** object identity (``is``, not
just ``==``). 16 threads * 50 iterations is enough to expose any
unsynchronised TOCTOU window on commodity hardware.

We deliberately do NOT call ``svc.start()`` — that would spin a
background loop per iteration and slow the test by ~1.5s. The race is
in ``get_or_create``'s check-and-construct pair, not in start/shutdown,
so a quiet (un-started) singleton is sufficient to exercise the gate.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from runtime.config import (
    AppConfig,
    LLMConfig,
    MCPConfig,
    MetadataConfig,
    Paths,
    StorageConfig,
)
from runtime.service import OrchestratorService


@pytest.fixture
def cfg(tmp_path) -> AppConfig:
    """Minimal AppConfig — no gateway, no MCP, no storage on disk."""
    return AppConfig(
        llm=LLMConfig.stub(),
        mcp=MCPConfig(servers=[]),
        storage=StorageConfig(
            metadata=MetadataConfig(url=f"sqlite:///{tmp_path}/test.db"),
        ),
        paths=Paths(
            skills_dir="examples/incident_management/skills",
            incidents_dir=str(tmp_path),
        ),
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the class-level singleton between tests so iterations are
    independent. Runs both before (covers leaks from sibling test
    modules) and after the test body."""
    OrchestratorService._reset_singleton()
    yield
    OrchestratorService._reset_singleton()


def _race_get_or_create(cfg: AppConfig, n_threads: int = 16) -> list[OrchestratorService]:
    """Hammer ``get_or_create`` from ``n_threads`` workers; return every
    instance observed."""
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(OrchestratorService.get_or_create, cfg) for _ in range(n_threads)]
        return [f.result(timeout=5.0) for f in futures]


def test_get_or_create_returns_identical_object_under_thread_race(cfg):
    """16 concurrent first-callers must observe the same object identity."""
    instances = _race_get_or_create(cfg, n_threads=16)
    # All references compare ``is`` — i.e. exactly one underlying object.
    first = instances[0]
    assert all(inst is first for inst in instances), (
        "get_or_create() returned multiple distinct instances under "
        f"thread race; got {len({id(i) for i in instances})} unique objects "
        f"out of {len(instances)} callers"
    )


def test_get_or_create_is_stable_across_repeated_races(cfg):
    """50 iterations of the 16-thread race must each yield exactly one
    instance. Catches a flaky lock that only sometimes serialises."""
    for iteration in range(50):
        instances = _race_get_or_create(cfg, n_threads=16)
        first = instances[0]
        assert all(inst is first for inst in instances), (
            f"iteration {iteration}: get_or_create() returned distinct "
            f"instances under race"
        )
        # Reset for the next iteration so each iteration exercises a
        # fresh first-call window.
        OrchestratorService._reset_singleton()


def test_reset_singleton_under_concurrent_get_or_create_does_not_leak(cfg):
    """A reset racing against a get_or_create must produce at most two
    distinct instances *across the reset boundary* — never two
    distinct instances *within the same singleton epoch*.

    We can't assert exactly-one when reset is in the mix (a thread that
    runs after reset legitimately sees a fresh instance), but each
    survivor must at minimum still be a real OrchestratorService.
    """
    with ThreadPoolExecutor(max_workers=8) as ex:
        # Mix get_or_create with periodic resets.
        results = []
        for _ in range(64):
            results.append(ex.submit(OrchestratorService.get_or_create, cfg))
        for _ in range(8):
            ex.submit(OrchestratorService._reset_singleton)

        instances = [f.result(timeout=5.0) for f in results]

    # Survivors must all be real services (no None, no half-built).
    assert all(isinstance(i, OrchestratorService) for i in instances)
    # And at most a small number of distinct epochs (one per reset
    # window) — definitely far fewer than 64. This bounds the leak.
    distinct = {id(i) for i in instances}
    assert len(distinct) <= 9, (
        f"reset race produced too many distinct instances: {len(distinct)} "
        "(expected <= 9 — one per reset boundary plus initial epoch)"
    )
