from setuptools import setup, find_packages

setup(
    name="nexus",
    version="2.0.0",
    description="Self-Evolving, AI-Only Trading Agent for Quotex",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pyquotex",
        "playwright",
        "PySide6",
        "rich",
        "pyqtgraph"
    ],
    extras_require={
        "dev": [
            "pytest",
            "mkdocs",
            "mkdocstrings",
            "ruff",
            "black",
            "flake8"
        ]
    },
    python_requires='>=3.13.6',
    entry_points={
        "console_scripts": [
            "nexus = nexus.main:main"
        ]
    }
)