"""Two-stage dedup pipeline (P7).

Stage 1 — embedding similarity over closed sessions in the same
environment via :meth:`HistoryStore.find_similar`.
Stage 2 — LLM confirmation on the top-K candidates with Pydantic-typed
structured output {is_duplicate, confidence, rationale}.

The pipeline is **framework-level** and never imports the
incident-management state class (R4 in the Phase-7 plan). Apps inject
domain-specific text via a ``text_extractor: Callable[[Session], str]``
callable.

Outcome semantics (locked):
  * Stage 2 short-circuits on the first confirmed match (R3 cost cap).
  * Stage 1 ordering already prioritises by similarity; stage 2 honours
    that order and does not re-rank.
  * Pipeline is non-mutating — the orchestrator owns mutation. This
    keeps unit tests simple and supports a future dry-run mode.

The configuration surface lives in :class:`DedupConfig` and is exposed
to the runtime via a generic provider hook
(``RuntimeConfig.dedup_config_path``). Framework default is *off*;
each example app's YAML opts in.
"""
from __future__ import annotations

import enum
import json
import logging
from typing import Any, Callable, Generic, Literal, Optional, TYPE_CHECKING, TypeVar

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover — import-only types
    from langchain_core.language_models.chat_models import BaseChatModel
    from runtime.config import FrameworkAppConfig, LLMConfig
    from runtime.storage.history_store import HistoryStore

logger = logging.getLogger(__name__)

# Framework-level state type. Permissive at ``BaseModel`` so the dedup
# layer never depends on app-level subclasses (R4 enforcement).
StateT = TypeVar("StateT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Config (P7-C)
# ---------------------------------------------------------------------------


class DedupScope(BaseModel):
    """Filter knobs that narrow the Stage 1 candidate pool."""

    same_environment: bool = True
    only_closed: bool = True


class DedupConfig(BaseModel):
    """Configuration for the two-stage dedup pipeline (P7-C).

    All numeric thresholds are inclusive at the lower bound (``>=``),
    so a candidate hitting exactly ``stage1_threshold`` is considered.

    Defaults are tuned for the incident-management example. Apps that
    want different policies override via YAML.
    """

    enabled: bool = False
    stage1_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    stage1_top_k: int = Field(default=5, ge=1, le=20)
    stage2_top_k: int = Field(default=3, ge=1, le=20)
    stage2_model: str = "cheap"
    stage2_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # Optional override of the Stage-2 system prompt. When ``None`` the
    # pipeline falls back to ``framework_cfg.dedup_system_prompt`` so
    # apps can either tune the prompt per app (FrameworkAppConfig) or
    # override it inline on the DedupConfig.
    system_prompt: str | None = None
    # Reserved for future modes; only ``post_intake`` is implemented.
    run_at: Literal["post_intake"] = "post_intake"
    scope: DedupScope = Field(default_factory=DedupScope)

    @model_validator(mode="after")
    def _validate_top_k(self) -> "DedupConfig":
        if self.stage2_top_k > self.stage1_top_k:
            raise ValueError(
                f"dedup.stage2_top_k ({self.stage2_top_k}) must be "
                f"<= dedup.stage1_top_k ({self.stage1_top_k})"
            )
        return self

    def assert_model_exists(self, llm_cfg: "LLMConfig") -> None:
        """Fail fast if ``stage2_model`` is missing from the LLM registry.

        Called at orchestrator boot when dedup is enabled. Raising here
        is preferred over discovering the typo on the first incident.
        """
        if self.stage2_model not in llm_cfg.models:
            raise ValueError(
                f"dedup.stage2_model={self.stage2_model!r} not found in "
                f"llm.models (known: {sorted(llm_cfg.models)})"
            )


# ---------------------------------------------------------------------------
# Decision schema (P7-E)
# ---------------------------------------------------------------------------


class DedupDecision(BaseModel):
    """Pydantic schema for Stage 2 LLM structured output."""

    is_duplicate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=500)


class DedupResult(BaseModel):
    """Outcome of one ``DedupPipeline.run`` call.

    ``matched=True`` iff Stage 2 confirmed a duplicate. The remaining
    fields are always populated when matched and may be partially
    populated for diagnostics when Stage 2 ran but declined.

    ``parse_failures`` counts the number of Stage 2 LLM responses that
    failed to parse into a :class:`DedupDecision` during this run. Any
    non-zero value is an operator signal that the Stage 2 model is
    drifting off-schema and dedup may be silently false-negative.
    """

    matched: bool = False
    parent_session_id: str | None = None
    candidate_id: str | None = None
    decision: DedupDecision | None = None
    stage1_score: float | None = None
    parse_failures: int = 0


# Internal tagged outcome for Stage 2 parse — distinguishes a legitimate
# "model said not-a-duplicate" from "model returned garbage we couldn't
# parse" so the pipeline can count parse failures separately.
class _Stage2Outcome(enum.Enum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    PARSE_FAILED = "parse_failed"


# ---------------------------------------------------------------------------
# Stage 2 prompt (P7-E)
# ---------------------------------------------------------------------------


# Legacy default — kept for back-compat with callers that referenced
# this module-level constant directly. New code should read the prompt
# off ``DedupConfig.system_prompt`` (when set) or the
# ``FrameworkAppConfig.dedup_system_prompt`` the pipeline holds.
_STAGE2_SYSTEM = (
    "You are deduplicating incident reports for an SRE platform. "
    "Two reports are duplicates only if they describe the same root cause "
    "AND the same service/environment AND overlap in time-of-occurrence. "
    "Surface-level keyword overlap is NOT enough. "
    "Respond with a single JSON object matching this schema: "
    '{"is_duplicate": bool, "confidence": float in [0,1], "rationale": '
    '"1-2 sentences"}.'
)


def _build_stage2_user_prompt(*, prior_text: str, new_text: str,
                              prior_id: str, new_id: str) -> str:
    """Assemble the user-side prompt for one Stage 2 comparison."""
    return (
        f"[INCIDENT A — existing, id={prior_id}]\n"
        f"{prior_text}\n\n"
        f"[INCIDENT B — new, id={new_id}]\n"
        f"{new_text}\n\n"
        "Decide: is B a duplicate of A?"
    )


def _parse_decision_tagged(
    raw: str, *, model_name: str = "<unknown>",
) -> tuple[DedupDecision | None, Exception | None]:
    """Parse the LLM's text into a ``DedupDecision`` with a failure tag.

    Returns ``(decision, None)`` on success, ``(None, exc)`` on any
    parse / validation failure (P7-E: "treat as not-duplicate, do not
    retry — budget protection"). Empty input is also a parse failure
    so the pipeline can surface model-stopped-responding as a signal.

    Emits a structured ``warning`` log on any failure with fields a
    log aggregator can pick up via the LogRecord ``extra`` namespace:
    ``event``, ``error_type``, ``error_msg``, ``model``,
    ``raw_output_excerpt``.
    """
    text = (raw or "").strip()
    if not text:
        exc = ValueError("empty Stage 2 LLM output")
        _log_parse_failure(exc, model_name=model_name, raw=raw or "")
        return None, exc
    # Tolerate ```json ... ``` fences from chatty models.
    if text.startswith("```"):
        # Strip the first fence line and a trailing fence if present.
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        _log_parse_failure(exc, model_name=model_name, raw=raw)
        return None, exc
    try:
        return DedupDecision.model_validate(payload), None
    except Exception as exc:  # noqa: BLE001 — pydantic ValidationError + fallback
        _log_parse_failure(exc, model_name=model_name, raw=raw)
        return None, exc


def _log_parse_failure(exc: Exception, *, model_name: str, raw: str) -> None:
    """Emit a structured warning for a Stage 2 parse failure.

    Fields land on the LogRecord via ``extra`` so structured log
    aggregators (Loki, Datadog, etc.) can index them, while the
    human-readable message stays useful for grep.
    """
    excerpt = (raw or "")[:200]
    logger.warning(
        "dedup stage 2 parse failure: %s (%s)",
        exc, type(exc).__name__,
        extra={
            "event": "dedup_parse_failure",
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:200],
            "model": model_name,
            "raw_output_excerpt": excerpt,
        },
    )


def _parse_decision(raw: str) -> DedupDecision | None:
    """Backward-compatible wrapper around :func:`_parse_decision_tagged`.

    Existing callers / tests that only care about the decision keep
    working; the pipeline uses the tagged variant directly so it can
    distinguish parse failures from legitimate non-matches.
    """
    decision, _err = _parse_decision_tagged(raw)
    return decision


# ---------------------------------------------------------------------------
# Pipeline (P7-D)
# ---------------------------------------------------------------------------


class DedupPipeline(Generic[StateT]):
    """Stage 1 (embedding) + Stage 2 (LLM) dedup orchestrator.

    Construction is cheap; ``run`` is the per-session entry point. The
    pipeline is stateless across runs.

    ``text_extractor`` returns the comparison text for a given session
    (the framework can't know which fields the app considers
    semantically meaningful).

    ``model_factory`` is a no-arg callable that returns a fresh
    ``BaseChatModel`` configured against ``config.stage2_model``. It is
    a callable (not a model instance) so the orchestrator can build the
    LLM lazily and so unit tests can inject a stub without importing
    LangChain.
    """

    def __init__(
        self,
        *,
        config: DedupConfig,
        text_extractor: Callable[[Any], str],
        model_factory: Callable[[], "BaseChatModel"],
        framework_cfg: "FrameworkAppConfig | None" = None,
    ) -> None:
        self.config = config
        self._text_extractor = text_extractor
        self._model_factory = model_factory
        # ``framework_cfg`` carries the cross-cutting prompt the
        # framework uses when ``DedupConfig.system_prompt`` is unset.
        # Imported lazily to avoid a circular import (``runtime.dedup``
        # is imported from ``runtime.config`` test paths).
        if framework_cfg is None:
            from runtime.config import FrameworkAppConfig as _FAC

            framework_cfg = _FAC()
        self._framework_cfg = framework_cfg

    async def run(
        self,
        *,
        session: StateT,
        history_store: "HistoryStore",
    ) -> DedupResult:
        """Run the pipeline for ``session``.

        Returns ``DedupResult(matched=False)`` when the pipeline is
        disabled, when the session text is empty, when Stage 1 finds no
        candidates above ``stage1_threshold``, or when Stage 2 declines
        every candidate (or errors out parsing structured output).
        """
        if not self.config.enabled:
            return DedupResult(matched=False)

        new_text = (self._text_extractor(session) or "").strip()
        if not new_text:
            return DedupResult(matched=False)

        candidates = self._stage1(session=session, new_text=new_text,
                                  history_store=history_store)
        if not candidates:
            return DedupResult(matched=False)

        return await self._stage2(session=session, new_text=new_text,
                                  candidates=candidates)

    # ----- Stage 1 -----

    def _stage1(
        self,
        *,
        session: StateT,
        new_text: str,
        history_store: "HistoryStore",
    ) -> list[tuple[Any, float]]:
        """Embedding similarity prefilter.

        Filters by ``scope.same_environment`` and ``scope.only_closed``,
        drops the current session id, applies the inclusive
        ``stage1_threshold``, and caps to ``stage1_top_k``.
        """
        filter_kwargs: dict[str, Any] = {}
        if self.config.scope.same_environment:
            env = getattr(session, "environment", None)
            if env:
                filter_kwargs["environment"] = env
        # ``status_filter`` is the resolved session bucket — only_closed
        # maps to "resolved" in the incident-management vocabulary.
        # Apps that disable only_closed get all statuses other than
        # in-flight via the empty filter (HistoryStore default behaviour
        # already screens deleted rows).
        status_filter = "resolved" if self.config.scope.only_closed else "*"
        try:
            raw = history_store.find_similar(
                query=new_text,
                filter_kwargs=filter_kwargs or None,
                status_filter=status_filter,
                threshold=self.config.stage1_threshold,
                limit=self.config.stage1_top_k,
            )
        except Exception as exc:  # noqa: BLE001 — never let dedup crash intake
            logger.warning("dedup stage 1: history_store failure: %s", exc)
            return []

        own_id = getattr(session, "id", None)
        out: list[tuple[Any, float]] = []
        for inc, score in raw:
            if getattr(inc, "id", None) == own_id:
                continue
            if score < self.config.stage1_threshold:
                # ``find_similar`` already screens by threshold but apps
                # may pass a custom HistoryStore — defensive double-check.
                continue
            out.append((inc, float(score)))
        return out[: self.config.stage1_top_k]

    # ----- Stage 2 -----

    async def _stage2(
        self,
        *,
        session: StateT,
        new_text: str,
        candidates: list[tuple[Any, float]],
    ) -> DedupResult:
        """LLM confirmation pass — short-circuits on the first confirm."""
        from langchain_core.messages import HumanMessage, SystemMessage

        capped = candidates[: self.config.stage2_top_k]
        # Build the model lazily so the factory error surfaces only when
        # we actually need an LLM (i.e. Stage 1 found candidates).
        try:
            llm = self._model_factory()
        except Exception as exc:  # noqa: BLE001
            logger.error("dedup stage 2: model factory failed: %s", exc)
            return DedupResult(matched=False)

        new_id = getattr(session, "id", "<new>")
        parse_failures = 0
        # Resolve the Stage-2 system prompt: per-config override wins,
        # otherwise the framework default. Apps that want incident-shaped
        # phrasing tune ``framework_cfg.dedup_system_prompt`` (the
        # incident-management example does); apps that want a one-off
        # override set it on ``DedupConfig.system_prompt``.
        system_prompt = (
            self.config.system_prompt
            or self._framework_cfg.dedup_system_prompt
        )
        for prior, stage1_score in capped:
            prior_id = getattr(prior, "id", "<unknown>")
            prior_text = (self._text_extractor(prior) or "").strip()
            user_prompt = _build_stage2_user_prompt(
                prior_text=prior_text,
                new_text=new_text,
                prior_id=str(prior_id),
                new_id=str(new_id),
            )
            try:
                msg = await llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "dedup stage 2: LLM call failed for prior=%s: %s",
                    prior_id, exc,
                )
                continue
            raw = getattr(msg, "content", "") or ""
            decision, parse_err = _parse_decision_tagged(
                raw, model_name=self.config.stage2_model,
            )
            if decision is None:
                # Parse / validation failure — count it so operators can
                # detect schema drift in dashboards / alerts. The legit
                # "model said not-duplicate" branch goes through the
                # ``decision is not None`` arm below and does NOT bump
                # the counter.
                if parse_err is not None:
                    parse_failures += 1
                continue
            if (decision.is_duplicate
                    and decision.confidence >= self.config.stage2_min_confidence):
                return DedupResult(
                    matched=True,
                    parent_session_id=str(prior_id),
                    candidate_id=str(prior_id),
                    decision=decision,
                    stage1_score=stage1_score,
                    parse_failures=parse_failures,
                )
        return DedupResult(matched=False, parse_failures=parse_failures)
