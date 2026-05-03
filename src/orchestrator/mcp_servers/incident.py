"""Backward-compat shim. Canonical implementation:
    examples.incident_management.mcp_server
"""
from examples.incident_management.mcp_server import IncidentMCPServer  # noqa: F401
from examples.incident_management.mcp_server import normalize_severity  # noqa: F401
from examples.incident_management.mcp_server import set_state  # noqa: F401
from examples.incident_management.mcp_server import mcp  # noqa: F401
from examples.incident_management.mcp_server import lookup_similar_incidents  # noqa: F401
from examples.incident_management.mcp_server import create_incident  # noqa: F401
from examples.incident_management.mcp_server import update_incident  # noqa: F401
