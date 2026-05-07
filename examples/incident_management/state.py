"""Incident-management state-overrides schema (DECOUPLE-05 / D-08-01).

Registered via ``OrchestratorConfig.state_overrides_schema`` in
``config/incident_management.yaml``. Validated by ``Orchestrator.start_session``
against the dict passed under the ``state_overrides=`` kwarg before any
row is written to the session store.
"""
from __future__ import annotations

from pydantic import BaseModel


class IncidentStateOverrides(BaseModel):
    """Per-session overrides for incident_management.

    All fields are Optional so v1.0 callers passing only ``environment``
    (e.g. ``tests/test_start_session_state_overrides.py``,
    ``tests/test_start_session_submitter.py``, ``src/runtime/api.py``)
    continue to validate. ``extra='forbid'`` catches typos at session-start
    rather than at gateway-eval time (PVC-06 / D-08-01).
    """

    model_config = {"extra": "forbid"}

    environment: str | None = None
    severity: str | None = None
