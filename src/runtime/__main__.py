"""Generic runtime entry: ``python -m runtime --config config/<app>.yaml``.

Boots the long-lived ``OrchestratorService`` against an app config YAML
and blocks until the process receives SIGINT or SIGTERM. The service
runs its asyncio loop on a background thread (see
``runtime.service.OrchestratorService``); this entry point owns the
main thread and exists solely to keep the process alive and to drive
graceful shutdown on signal delivery.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from runtime.config import load_config
from runtime.service import OrchestratorService


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m runtime",
        description="Run the orchestration service against an app config YAML.",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="Path to app config YAML (e.g. config/incident_management.yaml).",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    svc = OrchestratorService(cfg)
    svc.start()

    # Block the main thread until a signal arrives. The service runs
    # on its own background thread, so we just need a wait primitive
    # signal handlers can poke. ``threading.Event`` is the simplest
    # cross-platform option.
    stop = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    try:
        stop.wait()
    finally:
        svc.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
