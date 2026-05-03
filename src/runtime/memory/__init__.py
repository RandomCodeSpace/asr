"""Generic memory layers (ASR L2 / L5 / L7).

These stores and helpers are app-agnostic. Apps configure paths via
``AppConfig`` and consume the populated context off
``Session.extra_fields`` (or an app-specific subclass).

Public surface:

- :class:`KnowledgeGraphStore` — L2 component graph reader.
- :class:`PlaybookStore`        — L7 remediation-playbook matcher.
- :class:`ReleaseContextStore`  — L5 release-window correlation.

Session-level memory state (rides on ``extra_fields``):

- :class:`L2KGContext`, :class:`L5ReleaseContext`,
  :class:`L7PlaybookSuggestion`, :class:`MemoryLayerState`.

Helpers:

- :class:`HypothesisScore`, :func:`score_hypothesis`,
  :func:`should_refine` — hypothesis-loop refinement scoring.
- :class:`ToolCallSpec`, :func:`playbook_to_tool_calls`,
  :func:`top_playbook` — playbook-to-tool-call resolution helpers.
"""
from runtime.memory.hypothesis import (
    HypothesisScore,
    score_hypothesis,
    should_refine,
)
from runtime.memory.knowledge_graph import KnowledgeGraphStore
from runtime.memory.playbook_store import PlaybookStore
from runtime.memory.release_context import ReleaseContextStore
from runtime.memory.resolution import (
    ToolCallSpec,
    playbook_to_tool_calls,
    top_playbook,
)
from runtime.memory.session_state import (
    L2KGContext,
    L5ReleaseContext,
    L7PlaybookSuggestion,
    MemoryLayerState,
)

__all__ = [
    "HypothesisScore",
    "KnowledgeGraphStore",
    "L2KGContext",
    "L5ReleaseContext",
    "L7PlaybookSuggestion",
    "MemoryLayerState",
    "PlaybookStore",
    "ReleaseContextStore",
    "ToolCallSpec",
    "playbook_to_tool_calls",
    "score_hypothesis",
    "should_refine",
    "top_playbook",
]
