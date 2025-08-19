from setuptools import setup, find_packages

setup(
    name="nexus",
    version="2.0.0",
    description="Self-Evolving, AI-Only Trading Agent for Quotex",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "PySide6",
        "rich",
        "pyqtgraph",
        "pydantic",
        "pydantic-settings",
        "omegaconf",
        "pandas",
        "duckdb",
        "pyyaml",
        "loguru"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "mkdocs",
            "mkdocstrings",
            "ruff",
            "black",
            "flake8",
            "playwright"
        ],
        # Placeholder extra; user must manually install pyquotex from its source
        "quotex": []
    },
    python_requires='>=3.13.6',
    entry_points={
        "console_scripts": [
            "nexus = nexus.main:main"
        ]
    }
)