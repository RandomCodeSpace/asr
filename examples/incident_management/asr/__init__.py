"""ASR application memory-layer foundation.

App-level package that hangs the L2 / L5 / L7 memory stores off
``IncidentState``. The framework (``src/runtime/``) gains zero ASR
flavored types — every store here is a plain class over JSON files
on disk so the codebase stays air-gapped friendly.

Public surface is read-only:

- ``memory_state``  — pydantic models that ride along on
  ``IncidentState.memory`` (round-tripped via ``extra_fields``).
- ``kg_store``      — L2 Knowledge Graph (filesystem backend).
- ``release_store`` — L5 Release Context.
- ``playbook_store``— L7 Playbook Store.
"""
