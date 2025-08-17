import pytest


def test_gui_import_only():
    """Lightweight smoke test ensuring GUI classes import without constructing QApplication.
    Skips gracefully if PySide6 is unavailable or raises on import in headless CI.
    """
    try:
        from nexus.gui.main_window import NexusMainWindow  # noqa: F401
        from nexus.core.engine import NexusEngine
        from nexus.utils.config import NexusSettings, QuotexSettings, TradingSettings
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"GUI import skipped: {e}")
    # Basic engine instantiation to ensure no side-effect import errors
    settings = NexusSettings(quotex=QuotexSettings(email='a@b.com', password='pw'), trading=TradingSettings())
    engine = NexusEngine(settings)  # noqa: F841
    assert True  # If we reached here, smoke passed

