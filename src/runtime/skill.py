"""Skill loader.

Each agent lives in its own subdirectory under ``config/skills/``::

    config/skills/
      _common/                # OPTIONAL: prompt fragments shared by all agents
        confidence.md         # appended to every agent's system_prompt, in
        output.md             # alphabetical order, joined with blank lines
      intake/
        config.yaml           # description, tools, routes, model
        system.md             # the agent's specialty (markdown body — format is free)
        guidelines.md         # OPTIONAL extra fragments; every *.md in the
        ...                   # directory is concatenated in alphabetical order

Adding a directory under ``config/skills/`` (with a ``config.yaml`` and at
least one ``.md`` file) adds an agent. Directories whose name starts with
``_`` are reserved for shared content and never become agents.

The final ``system_prompt`` for each agent is::

    <concatenated *.md from agent_dir>
    \\n\\n
    <concatenated *.md from _common/, if present>

Structured config is validated through the :class:`Skill` /
:class:`RouteRule` Pydantic models; markdown content is loaded verbatim.

P6 — Agent kinds
----------------

Each ``Skill`` declares a ``kind`` discriminator; the loader validates
per-kind field shape so misconfigured skills fail loudly at startup
instead of at runtime. Three kinds are supported:

* ``responsive`` — the today-default LLM agent that responds inside a
  session graph (existing behaviour, preserved by default).
* ``supervisor`` — a no-LLM router that dispatches work to subordinate
  agents via LangGraph ``Send()``. No ``AgentRun`` row.
* ``monitor`` — a long-running observer that runs out-of-band on a
  schedule, evaluates an emit condition, and fires a Phase-5 trigger.
"""
from __future__ import annotations
import ast
import importlib
import re
from pathlib import Path
from typing import Any, Callable, Literal
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _validate_agent_name(name: str, *, source: str) -> None:
    """Reject directory names that can't safely become agent identifiers.

    Agent names appear in route targets (``next: triage``), LangGraph node
    IDs, and INC tags. Restricting them to ``[a-z][a-z0-9_]{0,63}`` keeps
    them grep-able, shell-safe, and free of case-sensitivity surprises
    across filesystems.
    """
    if not _AGENT_NAME_RE.match(name):
        raise ValueError(
            f"invalid agent name {name!r} (from {source}): must match "
            f"[a-z][a-z0-9_]{{0,63}} — lowercase, start with a letter, "
            f"alphanumerics + underscore only, max 64 chars"
        )


class RouteRule(BaseModel):
    when: str
    next: str
    gate: str | None = None


class DispatchRule(BaseModel):
    """One condition/target pair used by ``kind: supervisor`` skills with
    ``dispatch_strategy: rule``.

    ``when`` is a safe-eval expression (see :func:`_validate_safe_expr`)
    evaluated against the live session payload at dispatch time. The
    first matching rule wins; ``target`` names a subordinate agent.
    """

    when: str
    target: str


SkillKind = Literal["responsive", "supervisor", "monitor"]


# Cron-expression sanity check (5-field form: minute hour dom month dow).
# Each field accepts: '*', or a comma-separated list of ints / int ranges
# (a-b) / step ('* /n' or 'a-b/n'). This is intentionally a small subset
# of POSIX cron — broad enough for monitor schedules, narrow enough to
# parse with a regex (no external ``croniter`` dep, which is unavailable
# in the air-gapped target env per ``rules/build.md``).
_CRON_FIELD_RE = re.compile(
    r"^(\*|\*/\d+|\d+(-\d+)?(/\d+)?(,\d+(-\d+)?(/\d+)?)*)$"
)


def _validate_cron(expr: str) -> None:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(
            f"schedule {expr!r} is not a 5-field cron expression "
            f"(got {len(parts)} fields)"
        )
    for i, field in enumerate(parts):
        if not _CRON_FIELD_RE.match(field):
            raise ValueError(
                f"schedule {expr!r}: field #{i+1} {field!r} is not a "
                f"valid cron expression component"
            )


# Safe-eval AST whitelist for monitor ``emit_signal_when`` and
# supervisor ``DispatchRule.when``. We intentionally implement this with
# the stdlib ``ast`` module rather than depend on ``simpleeval`` —
# ``simpleeval`` is not available in the air-gapped target env. The
# whitelist is the smallest set that lets operators write conditions
# like ``observation['error_rate'] > 0.05 and status == 'open'`` without
# enabling arbitrary code execution. See plan R7.
_SAFE_AST_NODES: tuple[type, ...] = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Subscript, ast.Index, ast.Slice,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.IfExp,
)


def _validate_safe_expr(expr: str, *, source: str) -> None:
    """Reject non-whitelisted AST nodes in user-supplied expressions.

    Raises ``ValueError`` if the expression is not parseable or contains
    nodes outside :data:`_SAFE_AST_NODES`. Callable invocations,
    attribute access, comprehensions, lambdas, walrus, and the like are
    explicitly rejected.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{source}: cannot parse expression {expr!r}: {exc.msg}"
        ) from exc
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            raise ValueError(
                f"{source}: expression {expr!r} uses disallowed "
                f"construct {type(node).__name__!r} (safe-eval only "
                f"permits constants, names, comparisons, boolean ops, "
                f"arithmetic, subscripts, and literals)"
            )


def _resolve_dotted_callable(path: str, *, source: str) -> Callable[..., Any]:
    """Resolve a ``module.path:attr`` (or ``module.path.attr``) string to a callable.

    Used by ``kind: supervisor`` skills' ``runner`` field so app-level
    extension hooks can be wired in via YAML and validated at
    skill-load time. Raises ``ValueError`` on any failure mode
    (malformed path, missing module, missing attribute, non-callable
    target). The error names ``source`` so the YAML author sees which
    skill is broken.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{source}: dotted path must be a non-empty string")
    text = path.strip()
    if ":" in text:
        mod_part, _, attr_part = text.partition(":")
    elif "." in text:
        mod_part, _, attr_part = text.rpartition(".")
    else:
        raise ValueError(
            f"{source}: dotted path {path!r} must include an attribute "
            f"(use 'pkg.module:func' or 'pkg.module.func')"
        )
    if not mod_part or not attr_part:
        raise ValueError(
            f"{source}: dotted path {path!r} is missing the module or "
            f"attribute component"
        )
    try:
        module = importlib.import_module(mod_part)
    except ImportError as exc:
        raise ValueError(
            f"{source}: cannot import module {mod_part!r} from path "
            f"{path!r}: {exc}"
        ) from exc
    try:
        target = getattr(module, attr_part)
    except AttributeError as exc:
        raise ValueError(
            f"{source}: module {mod_part!r} has no attribute "
            f"{attr_part!r} (path={path!r})"
        ) from exc
    if not callable(target):
        raise ValueError(
            f"{source}: target {path!r} resolved to a non-callable "
            f"({type(target).__name__})"
        )
    return target


class Skill(BaseModel):
    """Single skill definition with a ``kind`` discriminator.

    The ``kind`` field selects the agent's execution model. Per-kind
    fields are declared at the model level for ergonomic YAML
    authoring; a ``model_validator`` rejects any combination that
    doesn't match the declared kind.

    Default kind is ``responsive`` so existing YAML (and historic
    Skill(...) construction in tests) keeps working without an explicit
    ``kind:`` field.
    """

    name: str
    description: str
    kind: SkillKind = "responsive"

    # ----- responsive (today's default behaviour) -----
    model: str | None = None
    tools: dict[str, list[str]] = Field(default_factory=dict)
    routes: list[RouteRule] = Field(default_factory=list)
    system_prompt: str = ""
    stub_response: str | None = None
    """Per-skill canned response used by ``StubChatModel`` when
    ``provider.kind == "stub"``.  Takes precedence over any entry in
    ``_DEFAULT_STUB_CANNED`` for the same agent name."""

    # ----- supervisor (no-LLM router) -----
    subordinates: list[str] = Field(default_factory=list)
    dispatch_strategy: Literal["llm", "rule"] = "llm"
    dispatch_prompt: str | None = None
    dispatch_rules: list[DispatchRule] = Field(default_factory=list)
    max_dispatch_depth: int = 3
    # P9-9h: optional dotted-path extension hook for app-specific
    # supervisor logic (e.g. memory-layer hydration, single-active-
    # investigation gates). The runner is invoked BEFORE the dispatch
    # table and may either mutate state or short-circuit to ``__end__``.
    # Resolved at skill-load time so misconfigured YAML fails fast.
    runner: str | None = None

    # ----- monitor (out-of-band, scheduled) -----
    schedule: str | None = None             # cron expression
    observe: list[str] = Field(default_factory=list)  # tool names
    emit_signal_when: str | None = None     # safe-eval expression
    trigger_target: str | None = None       # P5 trigger name
    tick_timeout_seconds: float = 30.0      # per-tick timeout (R6)

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for server, names in v.items():
            if not names:
                raise ValueError(
                    f"empty tool list for server {server!r}; "
                    f"remove the key or use ['*']"
                )
            if "*" in names and len(names) != 1:
                raise ValueError(
                    f"'*' must be the sole entry for server "
                    f"{server!r}; got {names!r}"
                )
        return v

    @field_validator("system_prompt")
    @classmethod
    def _strip_prompt(cls, v: str) -> str:
        return v.strip()

    @field_validator("max_dispatch_depth")
    @classmethod
    def _validate_max_depth(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError(
                f"max_dispatch_depth must be between 1 and 10 (got {v})"
            )
        return v

    @model_validator(mode="after")
    def _validate_kind_shape(self) -> "Skill":
        """Per-kind field-shape validation (P6-B).

        Each kind has a strict allow-list of fields. Anything from
        another kind raises ValueError naming the offending field and
        the kind. Required fields (e.g. monitor.schedule) are also
        enforced here.
        """
        kind = self.kind
        if kind == "responsive":
            self._validate_responsive()
        elif kind == "supervisor":
            self._validate_supervisor()
        elif kind == "monitor":
            self._validate_monitor()
        return self

    # ----- per-kind shape validators (called from _validate_kind_shape) -----

    def _validate_responsive(self) -> None:
        forbidden = {
            "subordinates": bool(self.subordinates),
            "dispatch_prompt": self.dispatch_prompt is not None,
            "dispatch_rules": bool(self.dispatch_rules),
            "schedule": self.schedule is not None,
            "observe": bool(self.observe),
            "emit_signal_when": self.emit_signal_when is not None,
            "trigger_target": self.trigger_target is not None,
            "runner": self.runner is not None,
        }
        # dispatch_strategy is allowed to keep its default; only flag it if
        # the user explicitly set it to non-default.
        if self.dispatch_strategy != "llm":
            forbidden["dispatch_strategy"] = True
        for field, present in forbidden.items():
            if present:
                raise ValueError(
                    f"skill {self.name!r} (kind=responsive) must not set "
                    f"{field!r} — that field belongs to another kind"
                )
        # ``system_prompt`` is sourced from the agent's *.md files at load
        # time (see ``load_skill``); the model itself permits an empty
        # string for tests and ad-hoc constructors that don't go through
        # the loader. The loader enforces .md presence for responsive
        # skills.

    def _validate_supervisor(self) -> None:
        forbidden = {
            "system_prompt": bool(self.system_prompt),
            "tools": bool(self.tools),
            "routes": bool(self.routes),
            "stub_response": self.stub_response is not None,
            "schedule": self.schedule is not None,
            "observe": bool(self.observe),
            "emit_signal_when": self.emit_signal_when is not None,
            "trigger_target": self.trigger_target is not None,
        }
        for field, present in forbidden.items():
            if present:
                raise ValueError(
                    f"skill {self.name!r} (kind=supervisor) must not set "
                    f"{field!r} — that field belongs to another kind"
                )
        if not self.subordinates:
            raise ValueError(
                f"skill {self.name!r} (kind=supervisor) requires a non-empty "
                f"subordinates list"
            )
        if self.runner is None:
            # Default every supervisor to the framework intake runner
            # (similarity retrieval + dedup gate). Apps override by
            # setting ``runner:`` in YAML.
            self.runner = "runtime.intake:default_intake_runner"
        # Resolve at skill-load time so a typo in YAML surfaces here,
        # not in the middle of a session. The resolver itself raises
        # ``ValueError`` with a helpful message — bubble that up.
        _resolve_dotted_callable(
            self.runner,
            source=f"skill {self.name!r} runner",
        )
        if self.dispatch_strategy == "llm" and not self.dispatch_prompt:
            raise ValueError(
                f"skill {self.name!r} (kind=supervisor, strategy=llm) requires "
                f"dispatch_prompt"
            )
        if self.dispatch_strategy == "rule":
            if not self.dispatch_rules:
                raise ValueError(
                    f"skill {self.name!r} (kind=supervisor, strategy=rule) "
                    f"requires dispatch_rules"
                )
            for i, rule in enumerate(self.dispatch_rules):
                _validate_safe_expr(
                    rule.when,
                    source=f"skill {self.name!r} dispatch_rules[{i}].when",
                )
                if rule.target not in self.subordinates:
                    raise ValueError(
                        f"skill {self.name!r}: dispatch_rules[{i}].target="
                        f"{rule.target!r} not found in subordinates "
                        f"({sorted(self.subordinates)})"
                    )

    def _validate_monitor(self) -> None:
        forbidden = {
            "system_prompt": bool(self.system_prompt),
            "routes": bool(self.routes),
            "stub_response": self.stub_response is not None,
            "subordinates": bool(self.subordinates),
            "dispatch_prompt": self.dispatch_prompt is not None,
            "dispatch_rules": bool(self.dispatch_rules),
            "runner": self.runner is not None,
        }
        for field, present in forbidden.items():
            if present:
                raise ValueError(
                    f"skill {self.name!r} (kind=monitor) must not set "
                    f"{field!r} — that field belongs to another kind"
                )
        if not self.schedule:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires a schedule "
                f"(5-field cron expression)"
            )
        _validate_cron(self.schedule)
        if not self.observe:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires a non-empty "
                f"observe list"
            )
        if not self.emit_signal_when:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires emit_signal_when"
            )
        _validate_safe_expr(
            self.emit_signal_when,
            source=f"skill {self.name!r} emit_signal_when",
        )
        if not self.trigger_target:
            raise ValueError(
                f"skill {self.name!r} (kind=monitor) requires trigger_target"
            )
        if self.tick_timeout_seconds <= 0:
            raise ValueError(
                f"skill {self.name!r}: tick_timeout_seconds must be positive "
                f"(got {self.tick_timeout_seconds})"
            )


def _concat_md(md_files: list[Path]) -> str:
    return "\n\n".join(p.read_text().strip() for p in md_files)


def _load_common_prompt(skills_dir: Path) -> str:
    """Read every ``*.md`` under ``<skills_dir>/_common/`` (if present) and
    return them concatenated in alphabetical order.

    Returns the empty string when no ``_common/`` directory exists or it
    contains no markdown files — ``load_all_skills`` then leaves agent
    prompts unchanged.
    """
    common_dir = skills_dir / "_common"
    if not common_dir.is_dir():
        return ""
    return _concat_md(sorted(common_dir.glob("*.md")))


def load_skill(agent_dir: str | Path, *, common: str = "") -> Skill:
    """Load one agent from its directory.

    The directory name is the agent's ``name`` (single source of truth).
    Reads ``config.yaml`` for the rest of the structured metadata and
    concatenates every ``*.md`` file (sorted alphabetically) into
    ``system_prompt``. If ``common`` is non-empty, it is appended after
    the agent's own prompt so shared sections (Confidence, Output) only
    need to be authored once.

    Raises ``ValueError`` if ``config.yaml`` declares its own ``name`` —
    the duplication used to drift silently when a directory was renamed
    without updating the config.
    """
    base = Path(agent_dir)
    config_path = base / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.yaml in skill dir: {base}")
    _validate_agent_name(base.name, source=f"directory {base}")
    cfg = yaml.safe_load(config_path.read_text()) or {}
    if "name" in cfg:
        raise ValueError(
            f"config.yaml at {config_path} must not declare 'name' — the "
            f"agent name is taken from the directory ({base.name!r})"
        )
    cfg["name"] = base.name
    # P6: only ``responsive`` skills require a system_prompt assembled from
    # the directory's .md files. ``supervisor`` and ``monitor`` skills are
    # configured purely via YAML, so the .md requirement is relaxed for
    # those kinds. The default kind is still ``responsive`` so existing
    # YAML keeps the historical "config.yaml + system.md" requirement.
    kind = cfg.get("kind", "responsive")
    md_files = sorted(base.glob("*.md"))
    if kind == "responsive":
        if not md_files:
            raise FileNotFoundError(f"no .md prompt files in skill dir: {base}")
        agent_prompt = _concat_md(md_files)
        cfg["system_prompt"] = (
            f"{agent_prompt}\n\n{common}".strip() if common else agent_prompt
        )
    else:
        # Non-responsive kinds may still ship descriptive .md alongside
        # config.yaml, but it's optional and never used as a system prompt.
        cfg.setdefault("system_prompt", "")
    return Skill(**cfg)


def load_all_skills(skills_dir: str | Path) -> dict[str, Skill]:
    base = Path(skills_dir)
    if not base.exists():
        raise FileNotFoundError(f"skills dir not found: {base}")
    common = _load_common_prompt(base)
    skills: dict[str, Skill] = {}
    for agent_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        # Reserve the leading underscore for shared content (_common,
        # _drafts, etc.) — never treat those as agents.
        if agent_dir.name.startswith("_"):
            continue
        if not (agent_dir / "config.yaml").exists():
            continue
        skill = load_skill(agent_dir, common=common)
        if skill.name in skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        skills[skill.name] = skill
    return skills
