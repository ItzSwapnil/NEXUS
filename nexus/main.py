"""
NEXUS GUI Launcher (Spec-compliant single entry point)

Per SPEC.md: main.py is the single launcher and runs only the GUI + full app.
"""
from __future__ import annotations

from nexus.gui.launch_gui import launch_nexus_gui


def main() -> None:  # pragma: no cover - GUI runtime
    # launch_nexus_gui loads config & initializes engine if not provided
    launch_nexus_gui()


if __name__ == "__main__":  # pragma: no cover
    main()
