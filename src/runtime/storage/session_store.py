"""Active session lifecycle store.

``SessionStore`` owns the write path for the row schema:
create, load, save, delete, list_all, list_recent. It also owns the
vector write-through (``_persist_vector``, ``_add_vector``,
``_refresh_vector``) and the row<->model converters shared with
``HistoryStore``.

The class is parametrised as ``Generic[StateT]`` and routes row
hydration through ``self._state_cls(...)`` so apps can plug in their
own ``Session`` subclass via ``RuntimeConfig.state_class``. The row
schema remains incident-shaped, but unused fields are dropped via
Pydantic's default ``extra='ignore'`` when a narrower ``state_cls`` is
supplied.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from typing import Generic, Optional, Type, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SqlSession

from runtime.state import AgentRun, Session, TokenUsage, ToolCall
from runtime.storage.models import IncidentRow

# The legacy ``INC-YYYYMMDD-NNN`` pattern stays here for back-compat
# validation against on-disk rows minted before the ``Session.id_format``
# hook existed. New rows are validated by ``_SESSION_ID_RE`` which
# accepts any ``PREFIX-YYYYMMDD-NNN`` shape the app's ``id_format`` may
# emit (e.g. ``CR-...`` for code-review).
_INC_ID_RE = re.compile(r"^INC-\d{8}-\d{3}$")
_SESSION_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*-\d{8}-\d{3}$")

# StateT is bound to ``Session`` (not bare ``BaseModel``) because the
# store body reads typed fields (``id``, ``status``, ``version``,
# ``updated_at`` …) that are declared on ``runtime.state.Session`` and
# not on ``pydantic.BaseModel``. The resolver in
# :mod:`runtime.state_resolver` already enforces a ``Session`` subclass
# at config time, and every existing caller (production + tests) passes
# either bare ``Session`` or a ``Session`` subclass — see
# Phase 19 / HARD-03 for the rationale (was: ``bound=BaseModel`` which
# made pyright flag every typed-field access).
StateT = TypeVar("StateT", bound=Session)


def _embed_source(inc: BaseModel) -> str:
    """Produce the text that represents a session in the vector store.

    Reads ``query`` from a typed field on a Session subclass first; for
    bare ``runtime.state.Session`` instances, falls back to
    ``extra_fields["query"]``. Returns "" only when neither carries a
    value — those rows aren't vectorised.
    """
    typed = (getattr(inc, "query", "") or "").strip()
    if typed:
        return typed
    extras = getattr(inc, "extra_fields", None) or {}
    return str(extras.get("query") or "").strip()


def _embed_source_from_row(row: "IncidentRow") -> str:
    """Same as ``_embed_source`` but from a raw ORM row (used in save)."""
    return (row.query or "").strip()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _now().strftime("%Y%m%d")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """DB datetime -> Incident model ISO string (UTC, 'Z' suffix)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Incident ISO string -> DB datetime (UTC-aware)."""
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _deserialize_resolution(raw: Optional[str]):
    """Attempt JSON parse of stored resolution; return raw string on failure."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class StaleVersionError(RuntimeError):
    """Raised when ``SessionStore.save`` observes that the row has been
    updated since the in-memory copy was loaded.

    Callers should reload from the store and re-apply their mutation.
    """


class SessionStore(Generic[StateT]):
    """Active session/incident lifecycle store, parametrised on ``StateT``.

    Owns CRUD on the row schema plus the vector write-through. Read-only
    similarity search lives in ``HistoryStore``.

    Threading note: methods open short-lived sessions; safe for the
    orchestrator's coarse-grained concurrency model.

    The ``state_cls`` ctor argument controls row->model hydration. Default
    is :class:`runtime.state.Session` (the framework base). Apps inject
    their own ``Session`` subclass (e.g. ``IncidentState``) via
    ``RuntimeConfig.state_class`` to surface domain-specific fields.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        state_cls: Type[StateT] = Session,  # type: ignore[assignment]
        embedder: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        vector_path: Optional[str] = None,
        vector_index_name: str = "incidents",
        distance_strategy: str = "cosine",
        id_prefix: str = "SES",
    ) -> None:
        self.engine = engine
        self._state_cls = state_cls
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_path = vector_path
        self.vector_index_name = vector_index_name
        self.distance_strategy = distance_strategy
        # Per-app session-id namespace. Threaded into
        # ``state_cls.id_format`` so each app's rows share a stable
        # ``PREFIX-YYYYMMDD-NNN`` shape. Default ``"SES"`` keeps the
        # bare-Session path framework-neutral; apps configure this
        # via ``FrameworkAppConfig.session_id_prefix``.
        self._id_prefix = id_prefix

    # ---------- ID minting ----------
    def _next_id(self, session: SqlSession) -> str:
        """Mint a new session id via ``state_cls.id_format(seq=...)``.

        The per-app id namespace is supplied as ``self._id_prefix`` (from
        ``FrameworkAppConfig.session_id_prefix``); ``id_format`` may
        also be overridden on the state subclass for fully bespoke
        shapes. The store still owns the monotonic sequence — it scans
        for prior rows whose id starts with the same ``PREFIX-YYYYMMDD-``
        stem and returns ``max(seq) + 1``.
        """
        # Probe today's prefix by asking the state class to format seq=1
        # and stripping the ``-001`` suffix. Apps that override
        # ``id_format`` to return a non-``PREFIX-YYYYMMDD-NNN`` shape
        # (e.g. opaque ULIDs) fall through to the simple count path
        # below.
        sample = self._state_cls.id_format(seq=1, prefix=self._id_prefix)
        m = _SESSION_ID_RE.match(sample)
        if m is None:
            # Custom format — count all rows as the sequence base. Apps
            # that want collision-free ids should mint ULIDs in
            # ``id_format`` and ignore ``seq``.
            count = session.execute(
                select(IncidentRow.id)
            ).scalars().all()
            return self._state_cls.id_format(
                seq=len(count) + 1, prefix=self._id_prefix,
            )

        # Extract the ``PREFIX-YYYYMMDD-`` stem (everything up to and
        # including the second hyphen).
        stem = sample.rsplit("-", 1)[0] + "-"
        like = f"{stem}%"
        rows = session.execute(
            select(IncidentRow.id).where(IncidentRow.id.like(like))
        ).scalars().all()
        max_seq = 0
        for r in rows:
            try:
                max_seq = max(max_seq, int(r.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        return self._state_cls.id_format(
            seq=max_seq + 1, prefix=self._id_prefix,
        )

    # ---------- public API ----------
    def create(self, *, query: str, environment: str,
               reporter_id: str = "user-mock",
               reporter_team: str = "platform") -> StateT:
        with SqlSession(self.engine) as session:
            now = _now()
            inc_id = self._next_id(session)
            row = IncidentRow(
                id=inc_id,
                status="new",
                created_at=now,
                updated_at=now,
                query=query,
                environment=environment,
                reporter_id=reporter_id,
                reporter_team=reporter_team,
                summary="",
                tags=[],
                agents_run=[],
                tool_calls=[],
                findings={},
                user_inputs=[],
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            inc = self._row_to_incident(row)
        self._add_vector(inc)
        return inc

    def load(self, incident_id: str) -> StateT:
        if not _SESSION_ID_RE.match(incident_id):
            raise ValueError(
                f"Invalid incident id {incident_id!r}; expected PREFIX-YYYYMMDD-NNN"
            )
        with SqlSession(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            return self._row_to_incident(row)

    def save(self, incident: StateT) -> None:
        if not _SESSION_ID_RE.match(incident.id):
            raise ValueError(
                f"Invalid incident id {incident.id!r}; expected PREFIX-YYYYMMDD-NNN"
            )
        # ``_iso(_now())`` returns ``str`` here -- the input datetime is
        # never None -- but the helper's signature is the broader
        # ``Optional[str]``. ``or ""`` keeps pyright + the typed
        # ``Session.updated_at: str`` field consistent without changing
        # behaviour (real value is always present).
        incident.updated_at = _iso(_now()) or ""
        sess = incident  # local alias — avoids repeating the domain token in new code
        expected_version = getattr(sess, "version", 1)
        # Bump in-memory BEFORE building the row dict so the persisted
        # row reflects the new version.
        sess.version = expected_version + 1
        with SqlSession(self.engine) as session:
            existing = session.get(IncidentRow, sess.id)
            prior_text = _embed_source_from_row(existing) if existing is not None else ""
            if existing is not None and existing.version != expected_version:
                # Roll back the in-memory bump so the caller can reload + retry.
                sess.version = expected_version
                raise StaleVersionError(
                    f"session {sess.id} version is {existing.version}, "
                    f"expected {expected_version}"
                )
            data = self._incident_to_row_dict(incident)
            if existing is None:
                session.add(IncidentRow(**data))
            else:
                for k, v in data.items():
                    setattr(existing, k, v)
            session.commit()
        self._refresh_vector(incident, prior_text=prior_text)

    def delete(self, incident_id: str) -> StateT:
        with SqlSession(self.engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise FileNotFoundError(incident_id)
            if row.status != "deleted":
                row.status = "deleted"
                row.deleted_at = _now()
                row.pending_intervention = None
            session.commit()
            session.refresh(row)
            return self._row_to_incident(row)

    def list_all(self, *, include_deleted: bool = False) -> list[StateT]:
        with SqlSession(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def list_recent(self, limit: int = 20, *,
                    include_deleted: bool = False,
                    include_duplicates: bool = False) -> list[StateT]:
        """Most-recent sessions first.

        ``include_duplicates`` defaults to ``False`` so the main UI list
        stays clean of the long tail of dedup'd sessions. The UI opts in
        via ``include_duplicates=True`` to render the collapsed-
        duplicates row.
        """
        with SqlSession(self.engine) as session:
            stmt = select(IncidentRow)
            if not include_deleted:
                stmt = stmt.where(IncidentRow.deleted_at.is_(None))
            if not include_duplicates:
                stmt = stmt.where(IncidentRow.status != "duplicate")
            stmt = stmt.order_by(
                desc(IncidentRow.created_at), desc(IncidentRow.id)
            ).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def list_children(self, parent_session_id: str) -> list[StateT]:
        """Return all sessions whose ``parent_session_id`` equals the given id.

        Powers the parent-session detail pane "children" section.
        Soft-deleted children are excluded; ordering is oldest-first so
        the UI shows them in the order they were flagged.
        """
        with SqlSession(self.engine) as session:
            stmt = (
                select(IncidentRow)
                .where(IncidentRow.parent_session_id == parent_session_id)
                .where(IncidentRow.deleted_at.is_(None))
                .order_by(IncidentRow.created_at, IncidentRow.id)
            )
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_incident(r) for r in rows]

    def un_duplicate(self, session_id: str, *,
                     retracted_by: str | None = None,
                     note: str | None = None) -> StateT:
        """Retract a duplicate flag.

        Behaviour (one transaction):
          * Loads the row; raises ``FileNotFoundError`` if missing.
          * Raises ``ValueError`` if ``status != "duplicate"`` so the
            HTTP layer can return ``409 Conflict``.
          * Captures ``parent_session_id`` as ``original_match_id``.
          * Sets ``status="new"`` and clears ``parent_session_id`` /
            ``dedup_rationale``.
          * Inserts a row in ``dedup_retractions`` for the audit trail.

        Idempotent at the boundary: a second call on the same id raises
        ``ValueError`` (the row is no longer a duplicate). The retraction
        does **not** auto-rerun the agent graph — operators trigger that
        explicitly.
        """
        # Imported here so the storage layer doesn't take a hard
        # import-time dependency on the audit table for callers that
        # never invoke retraction.
        from runtime.storage.models import DedupRetractionRow
        with SqlSession(self.engine) as session:
            row = session.get(IncidentRow, session_id)
            if row is None:
                raise FileNotFoundError(session_id)
            if row.status != "duplicate":
                raise ValueError(
                    f"session {session_id!r} is not a duplicate "
                    f"(status={row.status!r})"
                )
            original_match_id = row.parent_session_id or ""
            row.status = "new"
            row.parent_session_id = None
            row.dedup_rationale = None
            row.updated_at = _now()
            session.add(DedupRetractionRow(
                session_id=session_id,
                retracted_at=_now(),
                retracted_by=retracted_by,
                original_match_id=original_match_id,
                note=note,
            ))
            session.commit()
            session.refresh(row)
            return self._row_to_incident(row)

    # ---------- vector helpers ----------
    def _persist_vector(self) -> None:
        """If FAISS-backed (has save_local) and a path is configured, persist to disk."""
        if self.vector_store is None:
            return
        if not hasattr(self.vector_store, "save_local"):
            return
        if not self.vector_path:
            return
        from pathlib import Path
        folder = Path(self.vector_path)
        folder.mkdir(parents=True, exist_ok=True)
        # ``save_local`` is FAISS-specific; the runtime ``hasattr`` guard
        # at the top of this method already ensured this codepath only
        # runs against FAISS (other VectorStores omit the method).
        # ``langchain_core.vectorstores.VectorStore`` doesn't declare it.
        self.vector_store.save_local(  # pyright: ignore[reportAttributeAccessIssue]
            folder_path=str(folder),
            index_name=self.vector_index_name,
        )

    def _add_vector(self, inc: Session) -> None:
        if self.vector_store is None or self.embedder is None:
            return
        text = _embed_source(inc)
        if not text:
            return
        from langchain_core.documents import Document
        self.vector_store.add_documents(
            [Document(page_content=text, metadata={"id": inc.id})],
            ids=[inc.id],
        )
        self._persist_vector()

    def _refresh_vector(self, inc: Session, *, prior_text: str) -> None:
        if self.vector_store is None or self.embedder is None:
            return
        text = _embed_source(inc)
        if not text:
            return
        if prior_text == text:
            return
        self.vector_store.delete(ids=[inc.id])
        from langchain_core.documents import Document
        self.vector_store.add_documents(
            [Document(page_content=text, metadata={"id": inc.id})],
            ids=[inc.id],
        )
        self._persist_vector()

    # ---------- mapping helpers ----------
    #
    # Round-trip is driven by ``state_cls.model_fields`` so any
    # ``Session`` subclass — incident-shaped, code-review-shaped, or
    # whatever a future app brings — round-trips losslessly. The
    # ``IncidentRow`` schema keeps its incident-shaped typed columns
    # for back-compat indexing (``query``, ``environment``,
    # ``severity``, ...); fields the row schema doesn't have a column
    # for land in the ``extra_fields`` JSON column on save and merge
    # back into the model on load.

    # Fields handled out-of-band by the row<->model converters; do not
    # send these to ``extra_fields`` on save.
    _STATE_TOP_LEVEL_FIELDS = frozenset({
        "id", "status", "created_at", "updated_at", "deleted_at",
        "agents_run", "tool_calls", "findings", "token_usage",
        "pending_intervention", "user_inputs",
        "parent_session_id", "dedup_rationale",
        # ``extra_fields`` is the bag itself — round-tripped via the
        # JSON column directly, never nested inside the bag.
        "extra_fields",
        # Optimistic-concurrency token — has its own typed column.
        "version",
    })

    # Incident-shaped typed columns the row carries for back-compat
    # indexing. Apps whose state declares any of these get full
    # round-trip identity through the typed column; apps without them
    # leave the column at its DB default (empty string / NULL).
    _ROW_TYPED_DOMAIN_COLUMNS = frozenset({
        "query", "environment", "summary", "tags", "severity",
        "category", "matched_prior_inc", "resolution",
    })

    def _row_to_incident(self, row: IncidentRow) -> StateT:
        """Hydrate ``row`` into ``self._state_cls``.

        Fields are pulled from typed columns when the state class
        declares them; everything else is merged in from the
        ``extra_fields`` JSON bag. ``reporter`` is reconstituted from
        the typed ``reporter_id`` / ``reporter_team`` columns *only* when
        the state class has a ``reporter`` field — otherwise it's
        omitted so apps without a reporter concept (code-review) don't
        receive an unexpected attribute.
        """
        model_fields = self._state_cls.model_fields

        agents_run = [AgentRun.model_validate(a) for a in (row.agents_run or [])]
        tool_calls = [ToolCall.model_validate(t) for t in (row.tool_calls or [])]
        token_usage = TokenUsage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
        )

        kwargs: dict[str, object] = {
            "id": row.id,
            "status": row.status,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
            "deleted_at": _iso(row.deleted_at) if row.deleted_at else None,
            "agents_run": agents_run,
            "tool_calls": tool_calls,
            "findings": dict(row.findings or {}),
            "token_usage": token_usage,
            "pending_intervention": row.pending_intervention,
            "user_inputs": list(row.user_inputs or []),
            "parent_session_id": row.parent_session_id,
            "dedup_rationale": row.dedup_rationale,
            "version": row.version if row.version is not None else 1,
        }

        # Incident-shaped typed columns: include only fields the state
        # class actually declares.
        if "query" in model_fields:
            kwargs["query"] = row.query
        if "environment" in model_fields:
            kwargs["environment"] = row.environment
        if "reporter" in model_fields:
            kwargs["reporter"] = {"id": row.reporter_id, "team": row.reporter_team}
        if "summary" in model_fields:
            kwargs["summary"] = row.summary or ""
        if "tags" in model_fields:
            kwargs["tags"] = list(row.tags or [])
        if "severity" in model_fields:
            kwargs["severity"] = row.severity
        if "category" in model_fields:
            kwargs["category"] = row.category
        if "matched_prior_inc" in model_fields:
            kwargs["matched_prior_inc"] = row.matched_prior_inc
        if "resolution" in model_fields:
            kwargs["resolution"] = _deserialize_resolution(row.resolution)

        # Merge in any non-typed-column fields from ``extra_fields``.
        # Pydantic's ``extra='ignore'`` will drop any keys the state
        # class doesn't declare (e.g. legacy fields written by an older
        # binary), keeping the round-trip robust.
        extra = dict(row.extra_fields or {})

        # Route typed-column values into extra_fields when the state
        # class does NOT declare a typed Python field AND the column
        # actually has data. This lets a bare ``runtime.state.Session``
        # surface domain values (severity, environment, query…) via
        # ``state.extra_fields`` without the app needing a Session
        # subclass.
        typed_to_extra: dict[str, object] = {}
        if "query" not in model_fields and row.query:
            typed_to_extra["query"] = row.query
        if "environment" not in model_fields and row.environment:
            typed_to_extra["environment"] = row.environment
        if "reporter" not in model_fields and (row.reporter_id or row.reporter_team):
            typed_to_extra["reporter"] = {
                "id": row.reporter_id or "", "team": row.reporter_team or "",
            }
        if "summary" not in model_fields and row.summary:
            typed_to_extra["summary"] = row.summary
        if "severity" not in model_fields and row.severity:
            typed_to_extra["severity"] = row.severity
        if "category" not in model_fields and row.category:
            typed_to_extra["category"] = row.category
        if "matched_prior_inc" not in model_fields and row.matched_prior_inc:
            typed_to_extra["matched_prior_inc"] = row.matched_prior_inc
        if "tags" not in model_fields and row.tags:
            typed_to_extra["tags"] = list(row.tags)
        if "resolution" not in model_fields and row.resolution:
            typed_to_extra["resolution"] = _deserialize_resolution(row.resolution)

        # Fan extra_fields keys out as top-level kwargs for subclasses
        # that declare typed fields for them. Pydantic's ``extra=ignore``
        # silently drops keys the subclass doesn't declare — we re-stash
        # those into the bare ``extra_fields`` kwarg below so the data
        # survives.
        for k, v in extra.items():
            if k in self._STATE_TOP_LEVEL_FIELDS:
                continue  # handled above
            kwargs[k] = v

        # If the state class itself has an ``extra_fields`` field
        # (Session and any subclass that opts in), pass the row's
        # extra_fields content + typed-column-derived values through as
        # a single dict. Subclass-specific typed fields are handled by
        # the fan-out above; bare Session collects everything here.
        if "extra_fields" in model_fields:
            merged_extras: dict[str, object] = {}
            # 1. Typed-column values (when the subclass doesn't declare them)
            merged_extras.update(typed_to_extra)
            # 2. Row's extra_fields JSON column (subclass-specific
            # fields go to top-level kwargs above; whatever the subclass
            # doesn't declare lives only here)
            for k, v in extra.items():
                if k in self._STATE_TOP_LEVEL_FIELDS:
                    continue
                if k in model_fields:
                    # Subclass declared this as a typed field — it's
                    # already routed via top-level kwargs.
                    continue
                merged_extras[k] = v
            kwargs["extra_fields"] = merged_extras

        # ``kwargs`` is built up from heterogeneous sources (typed row
        # columns + ``extra_fields`` blob) so pyright infers each value
        # as ``object``. At runtime each entry matches the concrete
        # ``state_cls`` field type by construction (the row schema is
        # the source of truth); pydantic's own validation rejects bad
        # shapes at the constructor.
        return self._state_cls(**kwargs)  # pyright: ignore[reportArgumentType]

    def _incident_to_row_dict(self, inc: StateT) -> dict:
        """Serialize a state instance into a row-shaped dict.

        Fields with a typed column on ``IncidentRow`` are written there;
        everything else (any field declared by the state class but not
        present on the row schema) lands in ``extra_fields`` JSON.
        """
        model_fields = type(inc).model_fields
        # Apps may pass either a Session subclass with the full
        # incident-shaped fields (round-trip identity) or a bare
        # Session whose app data lives in ``extra_fields``. Helper
        # ``_field`` reads from a typed attribute first, then falls back
        # to extra_fields[key] — so both subclass and bare-Session paths
        # round-trip cleanly through the typed columns.
        bare_extra = getattr(inc, "extra_fields", {}) or {}

        def _field(name: str, default=None):
            if name in model_fields:
                return getattr(inc, name, default)
            return bare_extra.get(name, default)

        reporter = _field("reporter", None)
        if isinstance(reporter, dict):
            reporter_id = reporter.get("id")
            reporter_team = reporter.get("team")
        else:
            reporter_id = getattr(reporter, "id", None) if reporter is not None else None
            reporter_team = getattr(reporter, "team", None) if reporter is not None else None
        resolution = _field("resolution", None)

        # Build ``extra_fields``: every state-class field that is *not*
        # a top-level Session field and *not* one of the incident-shaped
        # typed columns ends up here as JSON-safe dict.
        extra: dict[str, object] = {}
        for fname in model_fields:
            if fname in self._STATE_TOP_LEVEL_FIELDS:
                continue
            if fname in self._ROW_TYPED_DOMAIN_COLUMNS:
                continue
            if fname == "reporter":
                # Already projected onto reporter_id / reporter_team
                # typed columns above; do not also persist to extra.
                continue
            value = getattr(inc, fname, None)
            # Pydantic v2: prefer model_dump for nested BaseModels and
            # collections-of-BaseModels so the JSON column gets a
            # JSON-safe representation.
            if isinstance(value, BaseModel):
                extra[fname] = value.model_dump(mode="json")
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                extra[fname] = [v.model_dump(mode="json") for v in value]
            elif isinstance(value, dict):
                # Convert any embedded BaseModel values to JSON-safe form
                # via model_dump where appropriate; otherwise pass through.
                extra[fname] = {
                    k: (v.model_dump(mode="json") if isinstance(v, BaseModel) else v)
                    for k, v in value.items()
                }
            else:
                extra[fname] = value

        return {
            "id": inc.id,
            "status": inc.status,
            "created_at": _parse_iso(inc.created_at),
            "updated_at": _parse_iso(inc.updated_at),
            "deleted_at": _parse_iso(inc.deleted_at) if inc.deleted_at else None,
            "query": _field("query", "") or "",
            "environment": _field("environment", "") or "",
            "reporter_id": reporter_id or "",
            "reporter_team": reporter_team or "",
            "summary": _field("summary", "") or "",
            "severity": _field("severity", None),
            "category": _field("category", None),
            "matched_prior_inc": _field("matched_prior_inc", None),
            "resolution": (
                resolution if resolution is None or isinstance(resolution, str)
                else json.dumps(resolution)
            ),
            "tags": list(_field("tags", []) or []),
            "agents_run": [a.model_dump(mode="json") for a in inc.agents_run],
            "tool_calls": [t.model_dump(mode="json") for t in inc.tool_calls],
            "findings": dict(inc.findings),
            "pending_intervention": inc.pending_intervention,
            "user_inputs": list(inc.user_inputs),
            "input_tokens": inc.token_usage.input_tokens,
            "output_tokens": inc.token_usage.output_tokens,
            "total_tokens": inc.token_usage.total_tokens,
            # Dedup linkage + rationale columns. ``getattr`` so bare
            # ``Session`` instances (without the dedup fields) round-
            # trip with NULL.
            "parent_session_id": getattr(inc, "parent_session_id", None),
            "dedup_rationale": getattr(inc, "dedup_rationale", None),
            # Everything not covered by a typed column. Subclass fields
            # come from the loop above; bare-Session callers stash app
            # data in ``state.extra_fields`` directly. Merge both, with
            # subclass fields taking precedence (parity with load path).
            "extra_fields": ({**bare_extra, **extra}) or None,
            "version": getattr(inc, "version", 1),
        }
