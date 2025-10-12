#!/usr/bin/env python3
"""
NEXUS Setup Configuration
Author: Swapnil De Sarkar

Minimal setup.py that defers to pyproject.toml.
For interactive setup, run: python scripts/setup_wizard.py
"""

from setuptools import setup, find_packages

# All configuration is in pyproject.toml
# This file exists only for compatibility with older pip versions
setup(
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts"]),
    include_package_data=True,
)

