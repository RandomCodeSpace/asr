# Phase 5 — Trigger Registry (api + schedule + webhook + plugin ABC) Implementation Plan

> **Status:** plan only, no implementation.
> **Predecessor:** Phase 4 (gateway hooks at session start). **Successor:** Phase 6 (agent kinds), Phase 7 (content-based dedup).
> **Locked decisions** (non-negotiable):
> - **Transports:** `api` + `schedule` + `webhook` + plugin ABC. **Defer queue consumers** (Kafka / Redis Streams) to a later phase.
> - **Webhook routing:** single generic `POST /triggers/{trigger_name}` endpoint. Each named trigger has its own Pydantic payload schema, `target_app` config, and `transform(payload) -> InitialState` hook.
> - **Scheduler / auth / dedupe:** APScheduler (asyncio) running on the FastAPI lifespan loop + bearer-token auth on incoming triggers + `Idempotency-Key` header dedup. **Content-based dedup is deferred to Phase 7.**

---

## 1. Goal + Scope

### Goal
Generalize session-start beyond the hand-coded `POST /investigate` route. After Phase 5 the framework can fire a `start_session(initial_state)` from three first-party transports (`api`, `webhook`, `schedule`) and any number of plugin-defined transports — all driven by a declarative `triggers:` block in app config.

### In scope
- `runtime/triggers/` package: `base.py` (ABC), `registry.py` (registry + dispatch), `webhook.py` (FastAPI router), `schedule.py` (APScheduler), `idempotency.py` (SQLite-backed LRU), `auth.py` (bearer middleware).
- `TriggerConfig` Pydantic model in `runtime/config.py`; new `triggers: list[TriggerConfig]` on `AppConfig`.
- `POST /triggers/{trigger_name}` route, mounted by `runtime/api.py` during lifespan.
- APScheduler started/stopped in the FastAPI lifespan (single in-process scheduler).
- A `TriggerTransport` ABC for plugin transports, with setuptools entry-point group `runtime.triggers` **and** explicit registration in app `__main__.py` (both paths supported; entry-points are optional).
- Example: `examples/incident_management/triggers.py` exporting `PagerDutyPayload` Pydantic schema and `transform_pagerduty(payload) -> IncidentState`; `examples/incident_management/config.yaml` gains a `triggers:` block.
- Tests: webhook happy-path, schedule fires, bearer auth (missing/wrong → 401), idempotency (duplicate `Idempotency-Key` returns cached), plugin transport lifecycle.
- README section documenting the trigger config shape.

### Out of scope (explicit non-goals)
- Queue/streaming consumers (Kafka, Redis Streams, Kinesis) — Phase 8+.
- HMAC signature verification for PagerDuty / Slack / Stripe — bearer is the simplest trustworthy path; signatures plug in via the `TriggerTransport` ABC later.
- Content-based deduplication (semantic / structured-key) — Phase 7's `dedup/pipeline.py`. Phase 5's `Idempotency-Key` header is a strict-match dedup, **not** content dedup.
- Trigger-level retry / DLQ — transform errors return 422 and are logged; observability and retry policy land in Phase 9.
- UI changes — the existing Streamlit UI keeps consuming `Orchestrator.list_recent_sessions()`; trigger metadata is recorded but not surfaced visually until Phase 6.
- Cron-string drift compensation — APScheduler's ±1min jitter is acceptable; any tighter SLO needs an external scheduler (out of scope).
- Multi-process schedulers (Celery beat, k8s CronJob) — single in-process APScheduler is the locked decision; horizontal scale is a Phase 9 problem.

### What stays the same
- `Orchestrator.start_session(...)` signature is the dispatch sink (already exists per Phase 2). Triggers call it; we add an optional `trigger: TriggerInfo | None` keyword for traceability but **do not change** existing positional/keyword signatures.
- The current `POST /investigate` route stays as a back-compat shim that internally constructs a synthetic `api`-transport trigger; no deprecation in Phase 5.
- `examples/incident_management/state.py` `IncidentState` is unchanged. The transform function emits an `IncidentState`; the orchestrator does not learn about transforms.

---

## 2. Target Architecture After Phase 5

```
src/runtime/
├── api.py                          ← mounts /triggers/{name} during lifespan
├── config.py                       ← AppConfig.triggers: list[TriggerConfig]   (NEW field)
├── orchestrator.py                 ← unchanged signature; accepts optional TriggerInfo
└── triggers/                       (NEW package)
    ├── __init__.py
    ├── base.py                       ← TriggerTransport ABC + TriggerInfo dataclass
    ├── config.py                     ← TriggerConfig Pydantic union (api/webhook/schedule/plugin)
    ├── registry.py                   ← TriggerRegistry: resolve dotted paths, lifecycle, dispatch
    ├── resolve.py                    ← importlib helpers for payload_schema / transform paths
    ├── webhook.py                    ← APIRouter factory; bearer auth + idempotency wired in
    ├── schedule.py                   ← APScheduler asyncio adapter
    ├── auth.py                       ← bearer-token verification (constant-time compare)
    └── idempotency.py                ← SQLite-backed LRU keyed by (trigger_name, key)

examples/incident_management/
├── config.yaml                      ← gains `triggers:` block (sample below)
└── triggers.py                      (NEW) ← PagerDutyPayload + transform_pagerduty
                                     ←       NightlySummaryPayload + transform_schedule

tests/
└── test_triggers/                  (NEW)
    ├── test_config.py
    ├── test_registry.py
    ├── test_webhook.py
    ├── test_schedule.py
    ├── test_auth.py
    ├── test_idempotency.py
    └── test_plugin_transport.py
```

### Lifespan flow

```
build_app(cfg)
  └── lifespan enter
        ├── Orchestrator.create(cfg)
        ├── TriggerRegistry.create(cfg.triggers, orchestrator=orch)
        │     ├── resolve dotted paths (payload_schema, transform) for every trigger
        │     ├── instantiate built-in transports + entry-point transports
        │     └── start() each transport (mount routes, start APScheduler, …)
        └── attach to app.state.{orchestrator, trigger_registry}

  └── lifespan exit
        ├── trigger_registry.stop()       (stop APScheduler, await jobs)
        └── orchestrator.aclose()
```

### Sample app config (locked shape)

```yaml
triggers:
  - name: pagerduty_high_severity
    transport: webhook
    target_app: incident_management
    payload_schema: examples.incident_management.triggers.PagerDutyPayload
    transform: examples.incident_management.triggers.transform_pagerduty
    auth: bearer
    auth_token_env: PAGERDUTY_WEBHOOK_TOKEN          # required when auth: bearer
    idempotency_ttl_hours: 24                         # default 24

  - name: nightly_summary
    transport: schedule
    schedule: "0 9 * * *"                             # APScheduler cron format
    timezone: UTC
    target_app: incident_management
    payload: { kind: nightly_summary }
    transform: examples.incident_management.triggers.transform_schedule
```

The `auth_token_env` field is the **only** secret-bearing field; the framework reads it via the existing `${ENV_VAR}` interpolation in `load_config`. The token never appears in YAML.

---

## 3. Naming Map / API Changes

| Before                                                       | After                                                                                         |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| (none)                                                       | `runtime.triggers.base.TriggerTransport` (ABC: `start(registry)` / `stop()`)                  |
| (none)                                                       | `runtime.triggers.base.TriggerInfo` (dataclass: `name`, `transport`, `target_app`, `received_at`) |
| (none)                                                       | `runtime.triggers.registry.TriggerRegistry`                                                   |
| (none)                                                       | `runtime.triggers.config.TriggerConfig` (discriminated union on `transport`)                  |
| `AppConfig` (no triggers field)                              | `AppConfig.triggers: list[TriggerConfig] = []`                                                |
| `POST /investigate` (only entrypoint)                        | unchanged (back-compat) **plus** `POST /triggers/{trigger_name}` (NEW)                        |
| `Orchestrator.start_session(*, query, environment, ...)`     | unchanged signature; new optional kw `trigger: TriggerInfo | None = None` for traceability    |
| (none)                                                       | `runtime.triggers.idempotency.IdempotencyStore` (SQLite-backed; same DB as session metadata)  |

No existing public symbol is renamed or deleted in Phase 5. The new module is purely additive.

---

## 4. Task Breakdown

Tasks are labeled **P5-A through P5-L**. Each task leaves the test suite green at its commit boundary. P5-A through P5-D are independent foundation; P5-E (route) requires P5-B + P5-D + P5-F; P5-G (scheduler) requires P5-B; P5-J (tests) requires E + G + H; P5-L is the sign-off.

> **Implementer guidance:** dispatch each task to a fresh subagent (`superpowers:subagent-driven-development`). Each task brief includes the goal, files touched, TDD steps, and the commit message. After each task, verify `pytest -q` is green and `pyright` clean before proceeding.

---

### P5-A — Define `TriggerConfig` Pydantic union in `runtime/triggers/config.py`

Discriminated union on the `transport` literal. Built-in variants: `api`, `webhook`, `schedule`. Plugin transports use `transport: plugin` and a `kind:` field that names the registered transport.

**Files:**
- Create: `src/runtime/triggers/__init__.py` (empty)
- Create: `src/runtime/triggers/config.py`
- Create: `tests/test_triggers/__init__.py` (empty)
- Create: `tests/test_triggers/test_config.py`
- Modify: `src/runtime/config.py` — add `triggers: list[TriggerConfig] = Field(default_factory=list)` on `AppConfig`. Import `TriggerConfig` lazily inside `runtime/config.py` to avoid a circular import (triggers package will eventually import `AppConfig` for typing).

**Steps:**

1. **Failing tests** — `tests/test_triggers/test_config.py`:
   ```python
   import pytest
   from pydantic import ValidationError

   def test_webhook_trigger_requires_payload_schema():
       from runtime.triggers.config import WebhookTriggerConfig
       with pytest.raises(ValidationError):
           WebhookTriggerConfig(name="x", target_app="incident_management")  # missing payload_schema/transform

   def test_schedule_trigger_requires_cron():
       from runtime.triggers.config import ScheduleTriggerConfig
       with pytest.raises(ValidationError):
           ScheduleTriggerConfig(name="x", target_app="incident_management",
                                 transform="examples.incident_management.triggers.transform_schedule")

   def test_dotted_path_validator_rejects_garbage():
       from runtime.triggers.config import WebhookTriggerConfig
       with pytest.raises(ValidationError):
           WebhookTriggerConfig(name="x", target_app="a", payload_schema="not a path",
                                transform="x.y", auth="bearer", auth_token_env="X")

   def test_bearer_requires_token_env():
       from runtime.triggers.config import WebhookTriggerConfig
       with pytest.raises(ValidationError):
           WebhookTriggerConfig(name="x", target_app="a",
                                payload_schema="examples.x.P", transform="examples.x.t",
                                auth="bearer")  # missing auth_token_env

   def test_app_config_accepts_triggers_block(tmp_path):
       import yaml
       from runtime.config import load_config
       p = tmp_path / "cfg.yaml"
       p.write_text(yaml.safe_dump({
           "llm": {"providers": {"primary": {"kind": "stub"}}, "embedding": {"backend": "stub"}},
           "mcp": {"servers": []},
           "triggers": [{
               "name": "nightly", "transport": "schedule", "schedule": "0 9 * * *",
               "target_app": "incident_management",
               "transform": "examples.incident_management.triggers.transform_schedule",
               "payload": {"kind": "nightly_summary"},
           }],
       }))
       cfg = load_config(p)
       assert len(cfg.triggers) == 1
       assert cfg.triggers[0].transport == "schedule"
   ```

2. **Implementation sketch** — `src/runtime/triggers/config.py`:
   ```python
   from __future__ import annotations
   import re
   from typing import Annotated, Literal, Union
   from pydantic import BaseModel, Field, field_validator, model_validator

   _DOTTED_PATH_RE = re.compile(r"^[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)+$")

   class _BaseTriggerConfig(BaseModel):
       name: str
       target_app: str
       transform: str  # dotted path to transform(payload) -> InitialState
       idempotency_ttl_hours: int = 24

       @field_validator("transform")
       @classmethod
       def _v_transform(cls, v: str) -> str:
           if not _DOTTED_PATH_RE.match(v):
               raise ValueError(f"transform must be dotted path, got {v!r}")
           return v

   class APITriggerConfig(_BaseTriggerConfig):
       transport: Literal["api"] = "api"
       mount: str = "/sessions"  # currently informational; route stays /investigate in P5

   class WebhookTriggerConfig(_BaseTriggerConfig):
       transport: Literal["webhook"] = "webhook"
       payload_schema: str  # dotted path to a Pydantic BaseModel
       auth: Literal["bearer", "none"] = "bearer"
       auth_token_env: str | None = None  # env-var name; required when auth=="bearer"

       @field_validator("payload_schema")
       @classmethod
       def _v_schema(cls, v: str) -> str:
           if not _DOTTED_PATH_RE.match(v):
               raise ValueError(f"payload_schema must be dotted path, got {v!r}")
           return v

       @model_validator(mode="after")
       def _check_bearer(self) -> "WebhookTriggerConfig":
           if self.auth == "bearer" and not self.auth_token_env:
               raise ValueError("auth: bearer requires auth_token_env")
           return self

   class ScheduleTriggerConfig(_BaseTriggerConfig):
       transport: Literal["schedule"] = "schedule"
       schedule: str  # APScheduler cron string
       timezone: str = "UTC"
       payload: dict = Field(default_factory=dict)

   class PluginTriggerConfig(_BaseTriggerConfig):
       transport: Literal["plugin"] = "plugin"
       kind: str  # named registered transport
       options: dict = Field(default_factory=dict)

   TriggerConfig = Annotated[
       Union[APITriggerConfig, WebhookTriggerConfig, ScheduleTriggerConfig, PluginTriggerConfig],
       Field(discriminator="transport"),
   ]
   ```

3. **Wire onto `AppConfig`** in `src/runtime/config.py`:
   ```python
   from runtime.triggers.config import TriggerConfig  # at top of file
   class AppConfig(BaseModel):
       # … existing fields …
       triggers: list[TriggerConfig] = Field(default_factory=list)
   ```

4. **Verify** — `pytest tests/test_triggers/test_config.py -v` green; `pytest -q` overall green; `pyright` clean.

5. **Commit** — `feat(triggers): add TriggerConfig discriminated union`

---

### P5-B — Define `TriggerRegistry` and `TriggerTransport` ABC in `runtime/triggers/registry.py` and `base.py`

The registry owns transport instances, resolves dotted paths once at startup, and exposes a single dispatch entrypoint that all transports call.

**Files:**
- Create: `src/runtime/triggers/base.py`
- Create: `src/runtime/triggers/registry.py`
- Create: `src/runtime/triggers/resolve.py`
- Create: `tests/test_triggers/test_registry.py`

**Steps:**

1. **Failing tests** — `tests/test_triggers/test_registry.py`:
   ```python
   import pytest
   from runtime.config import AppConfig
   # Smoke: an empty triggers list yields an empty, idempotent registry.
   async def test_registry_with_empty_triggers_starts_and_stops(...): ...
   # resolve_paths binds `_payload_cls` and `_transform_fn` per trigger name.
   async def test_resolve_paths_binds_callable(...): ...
   # dispatch(name, payload) -> calls transform -> calls orchestrator.start_session.
   async def test_dispatch_invokes_transform_and_starts_session(...): ...
   # Unknown trigger name raises KeyError (route layer turns this into 404).
   async def test_dispatch_unknown_name_raises(...): ...
   ```

2. **Implementation sketch** — `src/runtime/triggers/base.py`:
   ```python
   from __future__ import annotations
   from abc import ABC, abstractmethod
   from dataclasses import dataclass
   from datetime import datetime
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from runtime.triggers.registry import TriggerRegistry

   @dataclass(frozen=True)
   class TriggerInfo:
       """Provenance attached to every session started via a trigger."""
       name: str
       transport: str
       target_app: str
       received_at: datetime

   class TriggerTransport(ABC):
       """Lifecycle interface for a transport (api/webhook/schedule/plugin)."""
       @abstractmethod
       async def start(self, registry: "TriggerRegistry") -> None: ...
       @abstractmethod
       async def stop(self) -> None: ...
   ```

3. **Implementation sketch** — `src/runtime/triggers/resolve.py`:
   ```python
   import importlib
   from typing import Any, Callable, Type
   from pydantic import BaseModel

   def resolve_dotted(path: str) -> Any:
       module_path, _, attr = path.rpartition(".")
       module = importlib.import_module(module_path)
       try:
           return getattr(module, attr)
       except AttributeError as exc:
           raise ImportError(f"{module_path!r} has no attribute {attr!r}") from exc

   def resolve_payload_schema(path: str) -> Type[BaseModel]:
       cls = resolve_dotted(path)
       if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
           raise TypeError(f"{path!r} is not a Pydantic BaseModel subclass")
       return cls

   def resolve_transform(path: str) -> Callable[[BaseModel], Any]:
       fn = resolve_dotted(path)
       if not callable(fn):
           raise TypeError(f"{path!r} is not callable")
       return fn
   ```

4. **Implementation sketch** — `src/runtime/triggers/registry.py`:
   ```python
   from __future__ import annotations
   import asyncio
   from datetime import datetime, timezone
   from typing import Any
   from runtime.triggers.base import TriggerInfo, TriggerTransport
   from runtime.triggers.config import (
       APITriggerConfig, PluginTriggerConfig, ScheduleTriggerConfig,
       TriggerConfig, WebhookTriggerConfig,
   )
   from runtime.triggers.resolve import resolve_payload_schema, resolve_transform

   class TriggerRegistry:
       def __init__(self, *, orchestrator, idempotency_store, plugin_transports=None):
           self._orchestrator = orchestrator
           self._idem = idempotency_store
           self._plugin_kinds = dict(plugin_transports or {})
           self._configs: dict[str, TriggerConfig] = {}
           self._payload_cls: dict[str, type] = {}
           self._transform_fn: dict[str, Any] = {}
           self._transports: list[TriggerTransport] = []
           self._lock = asyncio.Lock()

       @classmethod
       async def create(cls, configs, *, orchestrator, idempotency_store, plugin_transports=None):
           reg = cls(orchestrator=orchestrator, idempotency_store=idempotency_store,
                     plugin_transports=plugin_transports)
           reg._resolve_paths(configs)
           reg._build_transports(configs)
           for t in reg._transports:
               await t.start(reg)
           return reg

       def _resolve_paths(self, configs: list[TriggerConfig]) -> None:
           seen: set[str] = set()
           for c in configs:
               if c.name in seen:
                   raise ValueError(f"duplicate trigger name {c.name!r}")
               seen.add(c.name)
               self._configs[c.name] = c
               self._transform_fn[c.name] = resolve_transform(c.transform)
               if isinstance(c, WebhookTriggerConfig):
                   self._payload_cls[c.name] = resolve_payload_schema(c.payload_schema)

       def _build_transports(self, configs: list[TriggerConfig]) -> None:
           # webhook: a single HTTP router covers all webhook configs (mounted in P5-E)
           # schedule: a single APScheduler covers all schedule configs (P5-G)
           # plugin: one transport per kind
           ...  # detail in P5-E / P5-G / P5-H

       async def stop(self) -> None:
           async with self._lock:
               for t in reversed(self._transports):
                   await t.stop()
               self._transports.clear()

       async def dispatch(self, *, name: str, payload: dict, idempotency_key: str | None) -> str:
           cfg = self._configs.get(name)
           if cfg is None:
               raise KeyError(name)
           if idempotency_key:
               cached = self._idem.get(name, idempotency_key)
               if cached is not None:
                   return cached  # session_id of the prior dispatch
           if isinstance(cfg, WebhookTriggerConfig):
               schema = self._payload_cls[name]
               typed = schema.model_validate(payload)
           else:
               typed = payload  # schedule/api: payload is already a dict
           initial_state = self._transform_fn[name](typed)  # may raise -> caller -> 422
           info = TriggerInfo(name=name, transport=cfg.transport,
                              target_app=cfg.target_app, received_at=datetime.now(timezone.utc))
           session_id = await self._orchestrator.start_session(
               initial_state=initial_state, trigger=info,
           )
           if idempotency_key:
               self._idem.put(name, idempotency_key, session_id, ttl_hours=cfg.idempotency_ttl_hours)
           return session_id
   ```

5. **Verify** — `pytest tests/test_triggers/test_registry.py -v` green.

6. **Commit** — `feat(triggers): add TriggerRegistry + TriggerTransport ABC`

---

### P5-C — Resolve transform / payload_schema dotted paths at registry init (covered in P5-B)

Captured here for completeness; the resolution logic lives in `runtime/triggers/resolve.py` (P5-B step 3). Acceptance:

- Bad dotted path → `ImportError` at registry creation, not at request time. **Fail fast.**
- `payload_schema` that isn't a `BaseModel` subclass → `TypeError` at startup.
- `transform` that isn't callable → `TypeError` at startup.

If P5-B already implemented `resolve.py`, P5-C is a no-op label kept for the original 12-task layout. Verification test `test_resolve_paths_binds_callable` belongs to this label.

**Commit (if separate):** `test(triggers): assert dotted-path resolution fails fast at startup`

---

### P5-D — Bearer auth middleware in `runtime/triggers/auth.py`

A small dependency that reads `Authorization: Bearer <token>` and compares constant-time against the env var named in `auth_token_env`. Returns 401 on missing/wrong header.

**Files:**
- Create: `src/runtime/triggers/auth.py`
- Create: `tests/test_triggers/test_auth.py`

**Implementation sketch:**
```python
import hmac, os
from fastapi import Header, HTTPException, status

def make_bearer_dep(token_env: str):
    """Return a FastAPI dependency that asserts Authorization matches $token_env."""
    expected = os.environ.get(token_env)  # snapshot at app startup
    if not expected:
        raise RuntimeError(f"env var {token_env!r} not set")

    async def _dep(authorization: str | None = Header(default=None)) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing bearer")
        token = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid bearer")
    return _dep
```

**Tests:**
- Missing `Authorization` header → 401.
- Wrong scheme (`Basic …`) → 401.
- Wrong token → 401.
- Correct token → dependency yields `None` (no exception).
- Token env var missing at startup → `RuntimeError` (fail fast).

**Commit:** `feat(triggers): bearer auth dependency for webhook routes`

---

### P5-E — `POST /triggers/{trigger_name}` route in `runtime/triggers/webhook.py`

Mounted by `runtime/api.py` during lifespan after `TriggerRegistry` is built.

**Files:**
- Create: `src/runtime/triggers/webhook.py`
- Modify: `src/runtime/api.py` — include router in lifespan; expose `app.state.trigger_registry`.
- Create: `tests/test_triggers/test_webhook.py`

**Implementation sketch (`webhook.py`):**
```python
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from runtime.triggers.auth import make_bearer_dep
from runtime.triggers.base import TriggerTransport
from runtime.triggers.config import WebhookTriggerConfig

class WebhookTransport(TriggerTransport):
    def __init__(self, configs: list[WebhookTriggerConfig]):
        self._configs = {c.name: c for c in configs}
        self.router = APIRouter()

    async def start(self, registry):
        for name, cfg in self._configs.items():
            deps = []
            if cfg.auth == "bearer":
                deps.append(Depends(make_bearer_dep(cfg.auth_token_env)))
            self.router.add_api_route(
                f"/triggers/{name}",
                self._make_handler(name, registry),
                methods=["POST"],
                dependencies=deps,
                status_code=202,
            )

    async def stop(self): ...

    def _make_handler(self, name: str, registry):
        async def handler(request: Request, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
            payload = await request.json()
            try:
                session_id = await registry.dispatch(
                    name=name, payload=payload, idempotency_key=idempotency_key,
                )
            except KeyError:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown trigger")
            except (ValueError, TypeError) as exc:  # pydantic validation / transform error
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
            return {"session_id": session_id}
        return handler
```

**`api.py` lifespan changes (sketch):**
```python
from runtime.triggers.registry import TriggerRegistry
from runtime.triggers.webhook import WebhookTransport
from runtime.triggers.idempotency import IdempotencyStore

@asynccontextmanager
async def _lifespan(app):
    orch = await Orchestrator.create(cfg)
    idem = IdempotencyStore.connect(cfg.storage.metadata.url)
    registry = await TriggerRegistry.create(
        cfg.triggers, orchestrator=orch, idempotency_store=idem,
    )
    # Mount webhook router(s) collected by the registry.
    for transport in registry.webhook_transports():
        app.include_router(transport.router)
    app.state.orchestrator = orch
    app.state.trigger_registry = registry
    try:
        yield
    finally:
        await registry.stop()
        await orch.aclose()
```

**Tests (`test_webhook.py`):**
- Happy path: POST valid `PagerDutyPayload` JSON to `/triggers/pagerduty_high_severity` → 202 + `session_id`. Assert `transform_pagerduty` was called and `Orchestrator.start_session` received the `IncidentState`.
- Invalid payload (missing required field) → 422.
- Transform raises `ValueError` → 422.
- Unknown trigger name → 404.
- Bearer auth: see P5-D test set, but exercised end-to-end here too.

**Verify:** `pytest tests/test_triggers/test_webhook.py -v` green.

**Commit:** `feat(triggers): generic POST /triggers/{name} webhook receiver`

---

### P5-F — Idempotency store (SQLite-backed LRU) in `runtime/triggers/idempotency.py`

In-memory LRU per trigger name, write-through to a SQLite table for cold-restart survival. Same database as the metadata store (`storage.metadata.url`); no new connection string.

**Files:**
- Create: `src/runtime/triggers/idempotency.py`
- Create: `tests/test_triggers/test_idempotency.py`
- Modify: `src/runtime/storage/models.py` — add `IdempotencyRow` table (schema below). Migration: `Base.metadata.create_all()` is already called at orchestrator startup, so no Alembic change required.

**Schema:**
```python
class IdempotencyRow(Base):
    __tablename__ = "trigger_idempotency"
    trigger_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
```

**API:**
```python
class IdempotencyStore:
    @classmethod
    def connect(cls, db_url: str) -> "IdempotencyStore": ...
    def get(self, trigger_name: str, key: str) -> str | None: ...
    def put(self, trigger_name: str, key: str, session_id: str, *, ttl_hours: int) -> None: ...
    def purge_expired(self) -> int: ...
```

- LRU is per `trigger_name`, max 1024 entries (configurable later); evicted entries stay in SQLite until `expires_at` passes.
- `purge_expired` is invoked opportunistically on every `put`.
- `get` checks LRU first, then SQLite (and refills LRU on hit).

**Tests:**
- `put` → `get` returns same `session_id`.
- `put` with `ttl_hours=0` → `get` returns `None` (already expired).
- Cold-restart: instantiate two stores against the same DB file; `put` on first, `get` on second → returns same value.
- LRU eviction does not lose data (SQLite still has it).
- Concurrent `put` of the same `(trigger, key)` is safe (last-write-wins; no duplicate-PK error — use `ON CONFLICT DO UPDATE` / merge).

**Commit:** `feat(triggers): SQLite-backed idempotency store with per-trigger LRU`

---

### P5-G — APScheduler integration in `runtime/triggers/schedule.py`

Single `AsyncIOScheduler` instance, started during lifespan, stopped on shutdown. Each `ScheduleTriggerConfig` registers a cron job that fires `registry.dispatch(name, payload, idempotency_key=None)`.

**Files:**
- Create: `src/runtime/triggers/schedule.py`
- Create: `tests/test_triggers/test_schedule.py`
- Modify: `pyproject.toml` — add dependency `apscheduler>=3.10,<4` (3.x has the stable `AsyncIOScheduler`; 4.x is alpha as of 2026 and has a different API — pin under 4 explicitly). **Note:** verify exact latest 3.x via `context7` MCP at implementation time per `rules/dependencies.md`.

**Implementation sketch:**
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from runtime.triggers.base import TriggerTransport
from runtime.triggers.config import ScheduleTriggerConfig

class ScheduleTransport(TriggerTransport):
    def __init__(self, configs: list[ScheduleTriggerConfig]):
        self._configs = configs
        self._scheduler: AsyncIOScheduler | None = None

    async def start(self, registry):
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        for cfg in self._configs:
            trigger = CronTrigger.from_crontab(cfg.schedule, timezone=cfg.timezone)
            self._scheduler.add_job(
                self._fire,
                trigger=trigger,
                kwargs={"registry": registry, "name": cfg.name, "payload": dict(cfg.payload)},
                id=f"trigger:{cfg.name}",
                replace_existing=True,
                misfire_grace_time=60,
                coalesce=True,
                max_instances=1,
            )
        self._scheduler.start()

    async def stop(self):
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=True)
            self._scheduler = None

    async def _fire(self, *, registry, name, payload):
        try:
            await registry.dispatch(name=name, payload=payload, idempotency_key=None)
        except Exception:
            # Log + swallow; Phase 9 adds retry semantics.
            logger.exception("scheduled trigger %s failed", name)
```

**Tests (`test_schedule.py`):**
- Register a `*/1 * * * *` cron, then **monkey-patch** `CronTrigger.from_crontab` (or use `DateTrigger(run_date=now+0.1s)` in the test fixture) to fire immediately. Avoid real `time.sleep(60)` in CI.
- Assert `registry.dispatch` was called with the expected name + payload.
- Assert `stop()` cancels pending jobs cleanly.
- Assert two configs with the same `name` raise at registry init (caught by P5-B's duplicate-name check, retest here for trace).

**Commit:** `feat(triggers): APScheduler-backed schedule transport`

---

### P5-H — Plugin transport ABC + entry-point loading

Defined in P5-B but tested here. Apps register transports two ways:
1. **Entry-points** in `pyproject.toml`: group `runtime.triggers`, key = `kind`, value = importable subclass of `TriggerTransport`.
2. **Explicit** registration: pass `plugin_transports={"my_kind": MyTransportClass}` to `TriggerRegistry.create`.

The registry resolves entry-points first, then merges explicit overrides (explicit wins for matching `kind`).

**Files:**
- Modify: `src/runtime/triggers/registry.py` — add entry-point loading via `importlib.metadata.entry_points(group="runtime.triggers")`.
- Create: `tests/test_triggers/test_plugin_transport.py`

**Tests:**
- Define a `StubTransport(TriggerTransport)` in the test module that records `start` / `stop` calls.
- Create a `TriggerRegistry` with `plugin_transports={"stub": StubTransport}` and a `PluginTriggerConfig(kind="stub")`.
- Assert `start` was awaited on lifespan enter; `stop` awaited on lifespan exit.
- Assert dispatch on the plugin trigger calls `transform` then `start_session` (same as built-ins).
- Smoke: monkeypatch `importlib.metadata.entry_points` to return a fake EP and verify it's loaded; this proves the entry-point path without actually publishing one.

**Commit:** `feat(triggers): plugin TriggerTransport via entry-points or explicit registration`

---

### P5-I — Example: `examples/incident_management/triggers.py`

Provide one webhook-shaped trigger (PagerDuty) and one schedule-shaped trigger (nightly summary) so the test suite has a real target_app to exercise.

**Files:**
- Create: `examples/incident_management/triggers.py`
- Modify: `examples/incident_management/config.yaml` — append the `triggers:` block from §2.

**Sketch:**
```python
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from examples.incident_management.state import IncidentState, Reporter

class PagerDutyPayload(BaseModel):
    incident_id: str
    title: str
    severity: str  # P1/P2/...
    service: str
    urgency: str

def transform_pagerduty(payload: PagerDutyPayload) -> IncidentState:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return IncidentState(
        id="",  # SessionStore assigns
        status="new",
        created_at=now, updated_at=now,
        query=f"PagerDuty: {payload.title}",
        environment="production",
        reporter=Reporter(id=f"pagerduty:{payload.incident_id}", team=payload.service),
        severity=payload.severity.lower(),  # validate via IncidentState's existing rules
    )

class NightlySummaryPayload(BaseModel):
    kind: str = Field(pattern=r"^nightly_summary$")

def transform_schedule(payload: dict | NightlySummaryPayload) -> IncidentState:
    typed = payload if isinstance(payload, NightlySummaryPayload) else NightlySummaryPayload(**payload)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return IncidentState(
        id="", status="new", created_at=now, updated_at=now,
        query="Nightly incident summary", environment="production",
        reporter=Reporter(id="scheduler", team="platform"),
        severity="info",
    )
```

> **Note:** `severity` legality, `IncidentState` field constraints, and `Reporter` shape come from `examples/incident_management/state.py` (see Phase 1 plan §P1-C). The transform functions must produce a state that round-trips through `SessionStore.create()`. If `severity="info"` is rejected by `IncidentState`, lower it to whatever the example app's `IncidentStatus` literal allows; verify by reading the state file at implementation time.

**Tests:** parts of `test_webhook.py` and `test_schedule.py` already cover these.

**Commit:** `feat(incident-management): pagerduty + schedule trigger transforms`

---

### P5-J — Tests: webhook + schedule + auth + idempotency + plugin (consolidation pass)

Each prior task ships its own focused tests. P5-J is the integration layer:

**Files:**
- Create: `tests/test_triggers/test_e2e_lifespan.py`

**Scenarios:**
1. Spin up the FastAPI app via `httpx.AsyncClient(app=…)` against a tmp SQLite DB.
2. POST a valid PagerDuty payload with a fresh `Idempotency-Key` → 202, capture `session_id`.
3. POST the **same** payload with the **same** `Idempotency-Key` → 202, **same** `session_id` returned, and `Orchestrator.start_session` was called **exactly once**.
4. POST with missing/wrong bearer → 401; `start_session` not called.
5. Stop the app (lifespan exit). Assert APScheduler is shut down (`registry._scheduler is None` post-stop).
6. Restart the app on the same DB; POST the same `Idempotency-Key` again → still returns the cached `session_id` (cold-restart survival via SQLite).
7. Plugin: register a stub plugin transport via `plugin_transports`; dispatch via the registry; assert `start`/`stop` called and `start_session` invoked.

**Verify:** `pytest tests/test_triggers/ -v` green; full suite still green.

**Commit:** `test(triggers): end-to-end lifespan + idempotency cold-restart coverage`

---

### P5-K — Document trigger config shape in README

> **Per global rule §1**: writing-side rule allows updating an *existing* README; do **not** create a new one. Verify which README to edit. Likely targets: `examples/incident_management/README.md` (Phase 1 P1-N created it) and the top-level `README.md` if it has a config-reference section.

**Files:**
- Modify: `examples/incident_management/README.md` — add a "Triggers" section: YAML shape, transport options, auth, idempotency, schedule cron format reference, security notes (where the bearer token comes from, how to rotate).
- Modify: top-level `README.md` only if it currently documents config keys; otherwise leave alone.

**Acceptance:**
- New section under `## Triggers` shows the YAML block from §2 verbatim.
- Documents the env-var convention for bearer tokens.
- Documents APScheduler cron format (5-field standard cron, `from_crontab`) and the ±1min drift caveat.
- Notes that content-based dedup is **out of scope until Phase 7**; only `Idempotency-Key` is honored in P5.

**Commit:** `docs(incident-management): document triggers config and security model`

---

### P5-L — Final verification + sign-off

**Steps:**

1. **Run full suite** — `pytest tests/ -v`. Baseline + new trigger tests all green.
2. **Type check** — `pyright src/runtime/triggers/ tests/test_triggers/` clean.
3. **Lint** — `ruff check src/runtime/triggers/ tests/test_triggers/` clean.
4. **Dependency audit** — `pip-audit` clean; per `~/.claude/rules/security.md` block on High/Critical, document Medium/Low. Pay attention to the new `apscheduler` transitive footprint.
5. **Bundle build** — `python scripts/build_single_file.py` succeeds; verify `dist/app.py` includes the new `runtime/triggers/` package.
6. **Smoke run**:
   ```bash
   PAGERDUTY_WEBHOOK_TOKEN=test123 ASR_CONFIG=examples/incident_management/config.yaml \
     uvicorn --app-dir dist app:get_app --factory --port 37776 &
   curl -X POST http://localhost:37776/triggers/pagerduty_high_severity \
     -H "Authorization: Bearer test123" -H "Idempotency-Key: smoke-1" \
     -H "Content-Type: application/json" \
     -d '{"incident_id":"P-1","title":"smoke","severity":"P2","service":"payments","urgency":"high"}'
   # → 202 {"session_id": "INC-..."}
   ```
7. **Grep checks (must all pass)**:
   ```bash
   # No raw curl in production code
   grep -rn '\bcurl\b' src/runtime/ && exit 1 || true
   # No public-internet URLs hardcoded in transports
   grep -rEn 'https?://(www|api)\.' src/runtime/triggers/ && exit 1 || true
   # Triggers package isolated from incident_management
   grep -rn 'incident_management' src/runtime/triggers/ && exit 1 || true
   ```
8. **Manual review checklist** (per `~/.claude/rules/testing.md`):
   - [ ] Re-read full diff before reporting done.
   - [ ] Each new public symbol has a docstring.
   - [ ] No `# type: ignore` without a one-line reason.
   - [ ] No bare `except:` clauses; every catch logs.
   - [ ] No unbounded queues (LRU caps the idempotency in-memory side; per `~/.claude/rules/performance.md`).
   - [ ] No secrets in YAML — only `_env` references.

**Commit:** `chore(triggers): phase 5 verification sign-off`

---

## 5. Sequencing and Dependencies

```
P5-A (TriggerConfig)
    │
    ▼
P5-B (Registry + ABC + resolve)  ←── covers P5-C resolution
    │
    ├─────────────┬───────────────┐
    ▼             ▼               ▼
 P5-D (auth)   P5-F (idempotency)  P5-H (plugin)
    │             │               │
    └─────┬───────┘               │
          ▼                       │
       P5-E (webhook route)       │
          │                       │
          │                       │
          │   P5-G (schedule)     │
          │       │               │
          ▼       ▼               ▼
       P5-I (example transforms — needed for P5-J fixtures)
          │
          ▼
       P5-J (e2e tests)
          │
          ▼
       P5-K (docs)
          │
          ▼
       P5-L (verification)
```

**Parallelizable bundles** (when dispatched via `superpowers:subagent-driven-development`):
- After P5-B lands: **P5-D, P5-F, P5-H, P5-G can be split across four subagents**. They share no files.
- P5-E lands once D + F are merged; P5-I can be drafted in parallel with E (touches different files).
- P5-J is sequential (depends on E, G, H, I all green).
- P5-K and P5-L are sequential at the tail.

---

## 6. Risks and Mitigations

**R1 — APScheduler cron format vs. unix cron.**
APScheduler supports both `CronTrigger(...)` and `CronTrigger.from_crontab(...)`; only the latter accepts standard 5-field cron strings. The plan **pins the YAML to the `from_crontab` 5-field format** and the `ScheduleTransport` calls `from_crontab` exclusively. The 6-field APScheduler-native form is rejected at config load (validate via regex in `ScheduleTriggerConfig`). Document in P5-K.

**R2 — Idempotency store survival across restart.**
Mitigation: SQLite-backed table on the **same** DB used for session metadata (`storage.metadata.url`). One DB connection pool, one filesystem path, one backup story. SQLite WAL mode (already enabled per Phase 2) handles concurrent reads from the LRU and the orchestrator. Document the operational coupling in P5-K and add a note that switching `storage.metadata.url` requires manually replaying outstanding idempotency keys (acceptable; logged, not auto-migrated).

**R3 — Transform hook errors.**
Locked policy: log + return **422 Unprocessable Entity**, do **not** auto-retry. Idempotency-Key for the failed dispatch is **not stored** (so a retry from the caller with the same key gets a fresh attempt). Observability lands in Phase 9; retry policy lands with the queue transports in Phase 8+.

**R4 — PagerDuty / Slack signature verification.**
Bearer token is the simplest secure path for Phase 5. HMAC signature transports (`x-pagerduty-signature`, `x-slack-signature`) plug in later by extending `auth: Literal["bearer", "none", "hmac_sha256"]` and adding a `signature_secret_env` field; the `WebhookTransport` selects the dependency based on `auth`. Out of scope for P5 — no code stubs.

**R5 — Schedule trigger drift.**
APScheduler in-process is good for ±1 minute accuracy under normal load. Tighter SLOs need an external scheduler (Celery beat, k8s CronJob), which is **out of scope**. Document the ±1min jitter in P5-K. `coalesce=True` and `misfire_grace_time=60` prevent duplicate fires when the loop is briefly busy.

**R6 — Race between two webhook deliveries with the same `Idempotency-Key`.**
The first request to win the SQLite UPSERT becomes canonical; the second sees the row and returns the cached `session_id` without starting a session. Test in P5-J. Use SQLite's `INSERT ... ON CONFLICT DO UPDATE` semantics to make this atomic.

**R7 — Entry-point discovery in the bundled `dist/app.py`.**
`importlib.metadata.entry_points` works on installed packages, not on a single-file bundle. Mitigation: explicit registration via `plugin_transports={...}` is the **only** supported path inside the bundled artifact; entry-points are a developer-mode convenience. Document in P5-K.

**R8 — Air-gapped APScheduler install.**
Per `~/.claude/rules/build.md`, `apscheduler` and its transitive deps must vendor cleanly. Resolve the latest 3.x via `context7` MCP at implementation time and check the wheel for any runtime network calls (it has none historically, but verify).

**R9 — Orchestrator `start_session` signature drift.**
The plan adds an optional `trigger: TriggerInfo | None = None` keyword. Verify at implementation time that no positional callers exist (search `start_session(query=`). This is keyword-only and defaults to `None`, so the back-compat `POST /investigate` path keeps working unchanged.

---

## 7. Done Criteria

1. `pytest tests/ -v` green; new tests under `tests/test_triggers/` cover each transport, auth, idempotency, plugin lifecycle, and the cold-restart e2e.
2. `pyright` and `ruff` clean on all new files.
3. `pip-audit` clean per `~/.claude/rules/security.md` (High/Critical = block; Medium/Low documented).
4. Sample webhook + schedule trigger work against `examples/incident_management/` end-to-end (P5-L smoke step).
5. `dist/app.py` builds offline and includes `runtime/triggers/` (per `~/.claude/rules/build.md`).
6. `examples/incident_management/README.md` documents the trigger YAML shape, bearer-token convention, idempotency semantics, cron drift caveat.
7. No public-internet URLs, no `curl | sh`, no telemetry phoning home (per `~/.claude/rules/build.md`).
8. Existing `POST /investigate` still works (back-compat).

---

## 8. Open Questions (for explicit user sign-off before implementation)

1. **Q:** Should `target_app` be enforced (e.g. only triggers whose `target_app` matches the running app are loaded) or treated as metadata only?
   **Tentative answer:** metadata only in P5; multi-app routing belongs to Phase 6. Confirm with user.

2. **Q:** Should the `api`-transport entry remain a no-op in YAML (informational) or actually re-mount `POST /investigate` under a configurable path?
   **Tentative answer:** no-op in P5. The legacy route is preserved verbatim. Renaming/remounting is a Phase 6 cleanup.

3. **Q:** APScheduler 3.x vs 4.x.
   **Tentative answer:** pin `>=3.10,<4`. 4.x is alpha as of 2026 cutoff and refactors the API surface. Confirm latest 3.x patch via `context7` at implementation time.

4. **Q:** Idempotency-Key max length — current sketch caps at 256 chars (DB column). Acceptable?
   **Tentative answer:** yes; matches Stripe's documented 255 max. Reject longer keys with 400 at the route layer.

5. **Q:** Should the registry expose `app.state.trigger_registry` to non-trigger routes (so an admin route can list active triggers, last fire, error counts)?
   **Tentative answer:** yes, expose a read-only `list_triggers()` method now; admin route lands in Phase 9. Cheap to bake in early.
