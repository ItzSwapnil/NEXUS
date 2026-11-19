"""NEXUS browser application."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import create_app


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_app"]
