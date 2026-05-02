import pytest


@pytest.mark.xfail(reason="implementations land in Tasks B–F", strict=False)
def test_storage_package_imports():
    from orchestrator.storage import (
        Base, IncidentRepository, IncidentRow, build_embedder, build_engine,
    )
    assert all([Base, IncidentRepository, IncidentRow, build_embedder, build_engine])
