"""Litestar host for the NEXUS dashboard.

The existing FastAPI route application is mounted as an ASGI application so
the public API, WebSocket protocol, and browser contract remain unchanged
during the framework migration. Litestar owns the outer lifecycle; Granian
serves this app.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from litestar import Litestar, asgi

from nexus.core.engine import NexusEngine
from nexus.utils.config import NexusSettings
from nexus.web.app import create_app


def create_litestar_app(
    settings: NexusSettings, demo_mode: bool = True, engine: NexusEngine | None = None
) -> Litestar:
    fastapi_app = create_app(settings, demo_mode=demo_mode, engine=engine)
    state = fastapi_app.state.nexus

    @asgi(path="/", is_mount=True, copy_scope=True)
    async def dashboard(scope: Any, receive: Any, send: Any) -> None:
        # Litestar mount routes provide a relative ASGI path; Starlette/FastAPI
        # expects the path to retain its leading slash.
        if isinstance(scope.get("path"), str):
            scope = dict(scope)
            path = "/" + scope["path"].lstrip("/")
            if path != "/":
                path = path.rstrip("/")
            scope["path"] = path
            scope["raw_path"] = path.encode()
        await fastapi_app(scope, receive, send)  # type: ignore[arg-type]

    @asynccontextmanager
    async def lifespan(_: Litestar):
        await state.start()
        try:
            yield
        finally:
            await state.stop()

    return Litestar(
        route_handlers=[dashboard],
        lifespan=[lifespan],
        openapi_config=None,
    )
