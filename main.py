"""Root launcher for NEXUS.

Allows running the app via:
  python main.py
  python -m nexus.main
  (after install) nexus
"""
from nexus.main import main

if __name__ == "__main__":  # pragma: no cover
    main()

