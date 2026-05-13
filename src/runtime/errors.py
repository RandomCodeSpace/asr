"""Typed runtime errors. Phase 13 lands the LLM-call surface; future
hardening (HARD-04 silent-failure sweep, HARD-03 pyright flip,
real-LLM follow-ups) extends here.

Importable as ``from runtime.errors import LLMTimeoutError, LLMConfigError``.
"""
from __future__ import annotations


class LLMTimeoutError(TimeoutError):
    """Raised when an LLM provider HTTP call exceeds request_timeout.

    Subclasses ``TimeoutError`` so ``runtime.policy._TRANSIENT_TYPES``
    auto-classifies it as transient via ``isinstance`` -- no policy.py
    edit needed (D-13-04).

    The ``__str__`` includes the substring ``"timed out"`` so existing
    string-matchers in ``runtime.graph._TRANSIENT_MARKERS`` and
    ``runtime.orchestrator._reconstruct_last_error`` also catch it
    without modification.
    """

    def __init__(self, provider: str, model: str, elapsed_ms: int) -> None:
        self.provider = provider
        self.model = model
        self.elapsed_ms = elapsed_ms
        super().__init__(
            f"LLM request timed out after {elapsed_ms}ms "
            f"(provider={provider}, model={model})"
        )


class LLMConfigError(ValueError):
    """Raised at config-load when a provider is missing a required field.

    Subclasses ``ValueError`` so pydantic ``@model_validator(mode='after')``
    propagates it cleanly into ``ValidationError`` (D-13-05).
    """

    def __init__(self, provider: str, missing_field: str) -> None:
        self.provider = provider
        self.missing_field = missing_field
        super().__init__(
            f"{provider} provider requires {missing_field!r}"
        )


__all__ = ["LLMTimeoutError", "LLMConfigError"]
