#!/usr/bin/env python3
"""
NEXUS - Workspace Cleaner
Removes cache files, logs, and temporary files (cross-platform)
"""

import shutil
import sys
from pathlib import Path


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


def print_header(text: str):
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def clean_pattern(pattern: str, recursive: bool = False) -> int:
    """Clean files matching a pattern."""
    count = 0
    glob_method = Path(".").rglob if recursive else Path(".").glob

    for path in glob_method(pattern):
        try:
            if path.is_file():
                path.unlink()
                print(f"  Removed: {path}")
                count += 1
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"  Removed: {path}/")
                count += 1
        except Exception as e:
            print(f"{Colors.YELLOW}  Could not remove {path}: {e}{Colors.RESET}")

    return count


def main():
    print_header("NEXUS - Workspace Cleaner")

    print(f"{Colors.YELLOW}Cleaning cache and temporary files...{Colors.RESET}\n")

    total_removed = 0

    # Python cache
    print("Cleaning Python cache...")
    total_removed += clean_pattern("**/__pycache__", recursive=True)
    total_removed += clean_pattern("**/*.pyc", recursive=True)
    total_removed += clean_pattern("**/*.pyo", recursive=True)
    total_removed += clean_pattern("**/*.pyd", recursive=True)

    # Test and coverage cache
    print("\nCleaning test cache...")
    total_removed += clean_pattern(".pytest_cache", recursive=False)
    total_removed += clean_pattern(".coverage*", recursive=False)
    total_removed += clean_pattern("htmlcov", recursive=False)
    total_removed += clean_pattern(".tox", recursive=False)
    total_removed += clean_pattern(".hypothesis", recursive=False)

    # Type checking cache
    print("\nCleaning type checking cache...")
    total_removed += clean_pattern(".mypy_cache", recursive=False)
    total_removed += clean_pattern(".pytype", recursive=False)
    total_removed += clean_pattern(".ruff_cache", recursive=False)

    # Logs (keep directory structure)
    print("\nCleaning logs...")
    for log_file in Path("logs").glob("*.log"):
        try:
            log_file.unlink()
            print(f"  Removed: {log_file}")
            total_removed += 1
        except Exception as e:
            print(f"{Colors.YELLOW}  Could not remove {log_file}: {e}{Colors.RESET}")

    # Build artifacts
    print("\nCleaning build artifacts...")
    total_removed += clean_pattern("build", recursive=False)
    total_removed += clean_pattern("dist", recursive=False)
    total_removed += clean_pattern("*.egg-info", recursive=False)

    # IDE files
    print("\nCleaning IDE files...")
    total_removed += clean_pattern(".DS_Store", recursive=True)
    total_removed += clean_pattern("Thumbs.db", recursive=True)
    total_removed += clean_pattern("*.swp", recursive=True)
    total_removed += clean_pattern("*.swo", recursive=True)

    # Summary
    print(f"\n{Colors.GREEN}✓ Cleanup complete!{Colors.RESET}")
    print(f"  Removed {total_removed} items\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Cleanup cancelled{Colors.RESET}")
        sys.exit(1)
