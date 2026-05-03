"""L5 Release Context — filesystem backend.

Read-only thin class over a single JSON file:

- ``recent.json`` — list of release records ``{id, service, sha,
  deployed_at, author, summary}`` sorted descending by ``deployed_at``.

Accepts ``root: Path`` (the layer directory, conventionally
``incidents/releases/``) for testability. Falls back to the seed
bundle at ``examples/incident_management/asr/seeds/releases/`` when
the configured directory is missing or empty. No Postgres/pgvector
dependency in this batch.

Surface:

- ``recent_for_service(service, *, hours=24)``
- ``suspect_at(*, services, at, window_minutes=60)``
- ``context(services, incident_at)`` -> :class:`L5ReleaseContext` ready
  to attach to ``IncidentState.memory.l5_release``.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from examples.incident_management.asr.memory_state import L5ReleaseContext

_SEED_ROOT = Path(__file__).parent / "seeds" / "releases"


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp tolerating the ``Z`` suffix."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ReleaseContextStore:
    """Filesystem-backed L5 Release Context reader."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._releases: list[dict] = []
        self._load()

    # ----- loading ------------------------------------------------------

    def _load(self) -> None:
        path = self._root / "recent.json"
        if not path.exists():
            path = _SEED_ROOT / "recent.json"

        records = json.loads(path.read_text())
        cleaned: list[dict] = []
        for r in records:
            if not r.get("id") or not r.get("service") or not r.get("deployed_at"):
                continue
            try:
                _parse_iso(r["deployed_at"])
            except ValueError:
                continue
            cleaned.append(dict(r))

        cleaned.sort(
            key=lambda r: _parse_iso(r["deployed_at"]),
            reverse=True,
        )
        self._releases = cleaned

    # ----- introspection (mostly for tests) -----------------------------

    def list_all(self) -> list[dict]:
        return [dict(r) for r in self._releases]

    # ----- public read API ----------------------------------------------

    def recent_for_service(
        self,
        service: str,
        *,
        hours: int = 24,
    ) -> list[dict]:
        """Releases for ``service`` deployed within the last ``hours``.

        ``hours`` is measured against ``datetime.now(UTC)``; for
        deterministic correlation work prefer :meth:`context` /
        :meth:`suspect_at` which take an explicit ``at``/``incident_at``
        anchor.
        """
        if hours <= 0:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out = [
            dict(r)
            for r in self._releases
            if r["service"] == service and _parse_iso(r["deployed_at"]) >= cutoff
        ]
        # Already in descending order from ``_load``.
        return out

    def suspect_at(
        self,
        *,
        services: list[str],
        at: datetime,
        window_minutes: int = 60,
    ) -> list[str]:
        """Release ids for ``services`` deployed within ``window_minutes``
        of ``at``.

        The window is symmetric around ``at`` so a release shipped right
        before *or* right after that anchor time is surfaced — useful
        for both "deploy caused it" and "deploy is the rollback" cases.
        Returns release ids sorted by ``deployed_at`` descending.
        """
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        if window_minutes <= 0:
            return []

        wanted = set(services)
        delta = timedelta(minutes=window_minutes)
        lo = at - delta
        hi = at + delta

        suspects: list[tuple[datetime, str]] = []
        for r in self._releases:
            if r["service"] not in wanted:
                continue
            deployed_at = _parse_iso(r["deployed_at"])
            if lo <= deployed_at <= hi:
                suspects.append((deployed_at, r["id"]))

        suspects.sort(key=lambda t: t[0], reverse=True)
        return [rid for _, rid in suspects]

    def context(
        self,
        services: list[str],
        incident_at: datetime,
    ) -> L5ReleaseContext:
        """Assemble an :class:`L5ReleaseContext` for the investigation.

        - ``recent_releases`` — all releases for the given services in
          the last 24h relative to ``incident_at`` (descending).
        - ``suspect_releases`` — release ids inside a 60-minute window
          around ``incident_at``.
        """
        if incident_at.tzinfo is None:
            incident_at = incident_at.replace(tzinfo=timezone.utc)

        wanted = set(services)
        cutoff = incident_at - timedelta(hours=24)
        recent = [
            dict(r)
            for r in self._releases
            if r["service"] in wanted and cutoff <= _parse_iso(r["deployed_at"]) <= incident_at
        ]
        suspects = self.suspect_at(
            services=services, at=incident_at, window_minutes=60
        )
        return L5ReleaseContext(
            recent_releases=recent,
            suspect_releases=suspects,
        )
