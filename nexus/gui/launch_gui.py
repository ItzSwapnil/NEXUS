import asyncio
import os
import sys

from PySide6.QtWidgets import QApplication

from nexus.core.engine import NexusEngine
from nexus.gui.main_window import NexusMainWindow
from nexus.utils.config import load_config


def _configure_qt_platform() -> None:
    """Select a stable Qt backend for Linux desktop sessions.

    Some Wayland compositors reject the buffer Qt creates while a window is
    transitioning into its maximized state (``xdg_surface buffer ... does
    not match the configured maximized state``).  Disabling Qt's Wayland
    client-side decorations avoids that protocol failure while preserving
    native Wayland rendering.  Respect explicit user choices.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_WAYLAND_DISABLE_WINDOWDECORATION", "1")


def launch_nexus_gui(engine: NexusEngine | None = None):
    """Launch the PySide6 GUI with an optional pre-initialized engine."""
    _configure_qt_platform()
    if engine is None:
        config = load_config()
        engine = NexusEngine(settings=config)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(engine.initialize_components())
    app = QApplication(sys.argv)
    window = NexusMainWindow(engine=engine)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _configure_qt_platform()
    config = load_config()
    engine = NexusEngine(settings=config)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(engine.initialize_components())
    app = QApplication(sys.argv)
    window = NexusMainWindow(engine=engine)
    window.show()
    sys.exit(app.exec())
