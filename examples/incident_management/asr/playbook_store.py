"""L7 Playbook Store — filesystem backend (Phase 9 — 9d).

Read-only thin class over a directory of YAML playbooks. Each file
follows the schema:

.. code-block:: yaml

    id: pb-payments-latency
    title: "Payments service latency spike"
    match_signals:
      service: payments
      metric: p99_latency
      threshold_breach: true
    hypothesis_steps:
      - "Check recent payments deploys (L5)"
      - "Check downstream dependencies (L2)"
    remediation:
      - tool: restart_service
        args: { service: payments }
    required_approval: true

Accepts ``root: Path`` (the layer directory, conventionally
``incidents/playbooks/``) for testability. Falls back to the seed
bundle at ``examples/incident_management/asr/seeds/playbooks/`` when
the configured directory is missing or empty. No FAISS / pgvector
dependency in this batch — semantic match comes in 9d-vector later.

Surface: ``get`` / ``list_all`` / ``match``. ``match`` produces a list
of :class:`L7PlaybookSuggestion` ranked by signal-overlap score, ready
to drop onto ``IncidentState.memory.l7_playbooks``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from examples.incident_management.asr.memory_state import L7PlaybookSuggestion

_SEED_ROOT = Path(__file__).parent / "seeds" / "playbooks"


def _normalise(value: Any) -> str:
    """Lowercase a scalar for case-insensitive equality."""
    if isinstance(value, bool):
        # ``str(True) == "True"`` would never match user-supplied
        # ``"true"``; standardise on the JSON/YAML lowercase form.
        return "true" if value else "false"
    return str(value).strip().lower()


class PlaybookStore:
    """Filesystem-backed L7 Playbook reader."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._playbooks: dict[str, dict] = {}
        self._load()

    # ----- loading ------------------------------------------------------

    def _load(self) -> None:
        roots: list[Path] = [self._root]
        # Fall back to the bundled seed when the configured layer dir
        # has no playbooks yet.
        if not self._has_yaml(self._root):
            roots = [_SEED_ROOT]

        for r in roots:
            if not r.exists() or not r.is_dir():
                continue
            for path in sorted(r.iterdir()):
                if path.suffix.lower() not in {".yaml", ".yml"}:
                    continue
                try:
                    data = yaml.safe_load(path.read_text())
                except yaml.YAMLError:
                    continue
                if not isinstance(data, dict):
                    continue
                pid = data.get("id")
                if not pid or not isinstance(pid, str):
                    continue
                self._playbooks[pid] = data

    @staticmethod
    def _has_yaml(root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        return any(
            p.suffix.lower() in {".yaml", ".yml"} for p in root.iterdir()
        )

    # ----- public read API ----------------------------------------------

    def get(self, playbook_id: str) -> dict | None:
        pb = self._playbooks.get(playbook_id)
        return None if pb is None else dict(pb)

    def list_all(self) -> list[dict]:
        return [dict(p) for p in self._playbooks.values()]

    def match(self, signals: dict) -> list[L7PlaybookSuggestion]:
        """Score every playbook against ``signals`` (case-insensitive eq).

        Score = ``matched / total`` where ``total`` is the number of
        keys declared on the playbook's ``match_signals`` block. A
        playbook with no declared signals scores 0 and is dropped from
        the result. Suggestions are returned in descending score, then
        ascending ``playbook_id`` for deterministic ties.
        """
        if not signals:
            return []

        norm_signals = {
            str(k).strip().lower(): _normalise(v)
            for k, v in signals.items()
        }

        out: list[L7PlaybookSuggestion] = []
        for pid, pb in self._playbooks.items():
            declared = pb.get("match_signals") or {}
            if not isinstance(declared, dict) or not declared:
                continue

            total = len(declared)
            matched_keys: list[str] = []
            for key, expected in declared.items():
                k = str(key).strip().lower()
                if k in norm_signals and norm_signals[k] == _normalise(expected):
                    matched_keys.append(f"{key}={expected}")

            if not matched_keys:
                continue

            out.append(
                L7PlaybookSuggestion(
                    playbook_id=pid,
                    score=len(matched_keys) / total,
                    matched_signals=sorted(matched_keys),
                )
            )

        out.sort(key=lambda s: (-s.score, s.playbook_id))
        return out
