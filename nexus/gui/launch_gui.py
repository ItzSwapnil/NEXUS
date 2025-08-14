import sys
import asyncio
from PySide6.QtWidgets import QApplication
from nexus.gui.main_window import NexusMainWindow
from nexus.core.engine import NexusEngine
from nexus.utils.config import load_config


def launch_nexus_gui(engine: NexusEngine | None = None):
    """Launch the PySide6 GUI with an optional pre-initialized engine."""
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
    config = load_config()
    engine = NexusEngine(settings=config)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(engine.initialize_components())
    app = QApplication(sys.argv)
    window = NexusMainWindow(engine=engine)
    window.show()
    sys.exit(app.exec())
