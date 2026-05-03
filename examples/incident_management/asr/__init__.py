"""ASR application memory-layer foundation (Phase 9 — 9a/9b/9c/9d).

App-level package that hangs the L2 / L5 / L7 memory stores off
``IncidentState``. The framework (``src/runtime/``) gains zero ASR
flavored types — every store here is a plain class over JSON files
on disk so the codebase stays air-gapped friendly.

Public surface for this batch is read-only:

- ``memory_state``  — pydantic models that ride along on
  ``IncidentState.memory`` (round-tripped via P8-J ``extra_fields``).
- ``kg_store``      — L2 Knowledge Graph (filesystem backend).
- ``release_store`` — L5 Release Context.
- ``playbook_store``— L7 Playbook Store.

Mutation paths (write from agents, playbook authoring) are deferred
to later sub-phases (9e–9g).
"""
