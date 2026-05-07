"""Session-derived tool-arg injection (Phase 9 / FOC-01 / FOC-02).

Two responsibilities, one module:

1. :func:`strip_injected_params` — clones a ``BaseTool``'s args_schema with
   one or more parameters removed. The LLM only sees the stripped sig and
   therefore cannot hallucinate values for those params (D-09-01). The
   original tool is left untouched so direct downstream callers (tests,
   scripts, in-process MCP fixtures) keep working.

2. :func:`inject_injected_args` — at tool-invocation time, re-adds the
   real values resolved from the live :class:`runtime.state.Session` via
   the configured dotted paths. When the LLM still supplied a value for
   an injected arg, the framework's session-derived value wins and an
   INFO log captures the override (D-09-03).

The framework stays generic — apps declare which args to inject and from
where via :attr:`runtime.config.OrchestratorConfig.injected_args` (D-09-02).
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from runtime.state import Session


# Module-private logger. Tests assert against logger name
# ``"runtime.orchestrator"`` so the override-log line shows up alongside
# the rest of the orchestrator-side observability without requiring a
# separate caplog target.
_LOG = logging.getLogger("runtime.orchestrator")


def strip_injected_params(
    tool: BaseTool,
    injected_keys: frozenset[str],
) -> BaseTool:
    """Return a ``BaseTool`` whose ``args_schema`` hides every param named
    in ``injected_keys``.

    The LLM only sees the stripped sig; the framework re-adds the real
    values at invocation time via :func:`inject_injected_args` (D-09-01).

    Properties:

    * **Pure.** The original tool is left unchanged — its ``args_schema``
      is not mutated, so tests and in-process callers that hold a direct
      reference keep their full schema.
    * **Idempotent.** Calling twice with the same keys is equivalent to
      calling once. The cloned schema is structurally identical.
    * **Identity short-circuit.** Empty ``injected_keys`` (or no overlap
      between ``injected_keys`` and the tool's params) returns the tool
      unchanged so unconfigured apps and tools without any injectable
      params pay nothing.
    """
    if not injected_keys:
        return tool
    schema = getattr(tool, "args_schema", None)
    if schema is None or not hasattr(schema, "model_fields"):
        return tool
    overlap = injected_keys & set(schema.model_fields.keys())
    if not overlap:
        # No params to strip — preserve identity (no clone).
        return tool

    # Build the kwargs for ``create_model`` from the surviving fields.
    # Pydantic v2's ``create_model`` accepts ``(annotation, FieldInfo)``
    # tuples; FieldInfo carries default + description + alias so the
    # cloned schema is functionally equivalent to the original minus
    # the stripped fields.
    keep: dict[str, tuple[Any, Any]] = {
        name: (f.annotation, f)
        for name, f in schema.model_fields.items()
        if name not in injected_keys
    }
    new_schema = create_model(
        f"{schema.__name__}__StrippedForLLM",
        __base__=BaseModel,
        **keep,  # type: ignore[arg-type]
    )

    # ``BaseTool`` is itself a pydantic BaseModel — ``model_copy`` clones
    # it cheaply and lets us swap ``args_schema`` without touching the
    # original. Tools that are not pydantic models (extremely rare; only
    # custom subclasses) fall back to a regular shallow copy.
    try:
        stripped = tool.model_copy(update={"args_schema": new_schema})
    except Exception:  # pragma: no cover — defensive fallback
        import copy
        stripped = copy.copy(tool)
        stripped.args_schema = new_schema  # type: ignore[attr-defined]
    return stripped


def _resolve_dotted(root: Session, path: str) -> Any | None:
    """Walk ``path`` ('session.foo.bar') against ``root`` and return the
    terminal value or ``None`` if any segment is missing / None.

    ``path`` must start with ``session.``. The leading ``session`` token
    pins the resolution root to the live Session — config-declared paths
    cannot reach into arbitrary modules. Subsequent segments walk
    attributes (``getattr``) — for fields stored under ``extra_fields``
    apps use ``session.extra_fields.foo`` which goes through the dict
    branch below.
    """
    parts = path.split(".")
    if not parts or parts[0] != "session":
        raise ValueError(
            f"injected_args path {path!r} must start with 'session.'"
        )
    cur: Any = root
    for seg in parts[1:]:
        if cur is None:
            return None
        # Support dict-valued attrs (notably ``Session.extra_fields``)
        # transparently — ``session.extra_fields.pr_url`` resolves
        # whether ``extra_fields`` is a real attribute or a dict on
        # the model. Plain attribute walks work for typed Session
        # subclasses (``IncidentState.environment``).
        if isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            cur = getattr(cur, seg, None)
    return cur


def inject_injected_args(
    tool_args: dict[str, Any],
    *,
    session: Session,
    injected_args_cfg: dict[str, str],
    tool_name: str,
) -> dict[str, Any]:
    """Return a NEW dict with each injected arg resolved from ``session``.

    Behaviour (D-09-03):

    * Mutation-free: ``tool_args`` is never modified. Callers that need
      to keep the LLM's original call shape can compare ``tool_args`` to
      the return value.
    * Framework wins on conflict. When the LLM already supplied a value
      and the resolved framework value differs, the framework value is
      written and a single INFO record is emitted on the
      ``runtime.orchestrator`` logger with the documented payload tokens
      (``tool``, ``arg``, ``llm_value``, ``framework_value``,
      ``session_id``).
    * Missing/None resolutions are skipped. The arg is left absent so
      the tool's own default-handling (or the MCP server's required-arg
      validator) decides what to do — never silently ``None``.
    """
    out = dict(tool_args)
    for arg_name, path in injected_args_cfg.items():
        framework_value = _resolve_dotted(session, path)
        if framework_value is None:
            continue
        if arg_name in out and out[arg_name] != framework_value:
            _LOG.info(
                "tool_call.injected_arg_overridden tool=%s arg=%s "
                "llm_value=%r framework_value=%r session_id=%s",
                tool_name,
                arg_name,
                out[arg_name],
                framework_value,
                getattr(session, "id", "?"),
            )
        out[arg_name] = framework_value
    return out


__all__ = [
    "strip_injected_params",
    "inject_injected_args",
    "_LOG",
]
