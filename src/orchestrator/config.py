"""Backward-compat shim. Canonical: ``runtime.config``."""
from runtime.config import *  # noqa: F401,F403
from runtime.config import _ENV_PATTERN, _interpolate  # noqa: F401  private re-export for tests
