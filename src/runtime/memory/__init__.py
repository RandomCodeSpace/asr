"""Generic memory layers (ASR L2 / L5 / L7).

These stores are app-agnostic, read-only, filesystem-backed. Apps wire
seed-directory paths via ``AppConfig`` and consume the populated
context off ``Session.extra_fields`` (or an app-specific subclass).

Public surface:

- :class:`KnowledgeGraphStore` — L2 component graph reader.
- :class:`PlaybookStore`        — L7 remediation-playbook matcher.
- :class:`ReleaseContextStore`  — L5 release-window correlation.
"""
from runtime.memory.knowledge_graph import KnowledgeGraphStore
from runtime.memory.playbook_store import PlaybookStore
from runtime.memory.release_context import ReleaseContextStore

__all__ = [
    "KnowledgeGraphStore",
    "PlaybookStore",
    "ReleaseContextStore",
]
