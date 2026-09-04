"""Granian entry point for the NEXUS ASGI dashboard.

The application remains framework-compatible with FastAPI while Granian owns
the HTTP/WebSocket serving process. Keeping this entry point separate makes it
possible to benchmark or roll back the server without duplicating route logic.
"""

from __future__ import annotations

import os

from nexus.core.engine import NexusEngine
from nexus.utils.config import load_runtime_settings
from nexus.web.litestar_app import create_litestar_app


def create_granian_app():
    settings = load_runtime_settings(os.getenv("NEXUS_CONFIG_PATH") or None)
    demo_mode = os.getenv("NEXUS_WEB_DEMO", "1").lower() in {"1", "true", "yes"}
    engine = NexusEngine(settings, demo_mode=demo_mode, auto_login=False)
    return create_litestar_app(settings, demo_mode=demo_mode, engine=engine)
