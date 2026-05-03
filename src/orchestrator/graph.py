"""Backward-compat shim. Canonical: ``runtime.graph``."""
from runtime.graph import *  # noqa: F401,F403
from runtime.graph import _coerce_confidence, _decide_from_signal  # noqa: F401
