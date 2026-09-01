#!/usr/bin/env python3
"""
NEXUS Quick Launch CLI - Cross-platform launcher
Author: Swapnil De Sarkar
Created: 2025

Usage:
    python run.py
    or: python -m nexus.launcher
"""

import glob
import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.RESET}")


def print_menu():
    """Display main menu."""
    print_header("NEXUS AI Trader - Quick Launch")
    print("\nSelect an option:")
    print(f"  {Colors.GREEN}1.{Colors.RESET} Launch GUI (Demo Mode)")
    print(f"  {Colors.GREEN}2.{Colors.RESET} Run CLI with Stats")
    print(f"  {Colors.GREEN}3.{Colors.RESET} Run Backtest")
    print(f"  {Colors.GREEN}4.{Colors.RESET} Live Backtest (Quotex Demo)")
    print(f"  {Colors.GREEN}5.{Colors.RESET} Train AI Models")
    print(f"  {Colors.GREEN}6.{Colors.RESET} Run Tests")
    print(f"  {Colors.GREEN}7.{Colors.RESET} Clean Project")
    print(f"  {Colors.GREEN}8.{Colors.RESET} View Performance Stats")
    print(f"  {Colors.GREEN}0.{Colors.RESET} Exit")
    print()


def get_python():
    """Get Python executable from venv or system."""
    venv_python = (
        Path(".venv")
        / ("Scripts" if sys.platform == "win32" else "bin")
        / ("python.exe" if sys.platform == "win32" else "python")
    )
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_command(cmd: list, description: str = ""):
    """Run a command and handle errors."""
    if description:
        print(f"\n{Colors.YELLOW}{description}...{Colors.RESET}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Command failed with error code {e.returncode}{Colors.RESET}")
        return False
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        return False


def check_env():
    """Check if .env file exists."""
    if not Path(".env").exists():
        print(f"{Colors.RED}⚠ Warning: .env file not found!{Colors.RESET}")
        print(f"{Colors.YELLOW}Please create .env from .env.example{Colors.RESET}")
        return False
    return True


def launch_gui():
    """Launch GUI in demo mode."""
    python = get_python()
    run_command([python, "-m", "nexus.main", "--gui", "--demo"], "Launching GUI")


def run_cli_stats():
    """Run CLI with stats."""
    python = get_python()
    run_command([python, "-m", "nexus.main", "--cli", "--demo", "--stats"], "Running CLI")


def run_backtest():
    """Run backtest."""
    python = get_python()
    run_command([python, "-m", "nexus.main", "--cli", "--backtest", "--demo"], "Running backtest")


def run_live_backtest():
    """Run live backtest with Quotex demo data."""
    python = get_python()
    run_command(
        [python, "-m", "nexus.main", "--cli", "--live-backtest", "--demo"], "Running live backtest"
    )


def train_models():
    """Train AI models."""
    python = get_python()
    run_command([python, "-m", "nexus.ai.train_models"], "Training AI models")


def run_tests():
    """Run test suite."""
    python = get_python()
    run_command([python, "-m", "pytest", "-v"], "Running tests")


def clean_project():
    """Clean cache and temporary files."""
    print(f"\n{Colors.YELLOW}Cleaning project...{Colors.RESET}")

    patterns = [
        "logs/*.log",
        "data/cache/*",
        "**/__pycache__",
        "**/*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    ]

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                path_obj = Path(path)
                if path_obj.is_file():
                    path_obj.unlink()
                elif path_obj.is_dir():
                    shutil.rmtree(path_obj)
                print(f"  Removed: {path}")
            except Exception as e:
                print(f"  {Colors.YELLOW}Could not remove {path}: {e}{Colors.RESET}")

    print(f"{Colors.GREEN}✓ Cleanup complete!{Colors.RESET}")


def view_stats():
    """View performance statistics."""
    python = get_python()
    run_command(
        [
            python,
            "-c",
            """
from nexus.core.engine import NexusEngine
from nexus.utils.config import load_runtime_settings

settings = load_runtime_settings()
engine = NexusEngine(settings=settings, demo_mode=True)
stats = engine.get_performance_stats()

print("\\n" + "="*60)
print("NEXUS Performance Statistics")
print("="*60)
for key, value in stats.items():
    print(f"{key:.<40} {value}")
print("="*60)
""",
        ],
        "Loading performance stats",
    )


def main():
    """Main launcher loop."""
    if not check_env():
        response = input("\nContinue anyway? (y/N): ").lower()
        if response not in ("y", "yes"):
            sys.exit(1)

    while True:
        print_menu()

        try:
            choice = input(f"{Colors.CYAN}Enter choice (0-8): {Colors.RESET}").strip()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
            break

        if choice == "1":
            launch_gui()
        elif choice == "2":
            run_cli_stats()
        elif choice == "3":
            run_backtest()
        elif choice == "4":
            run_live_backtest()
        elif choice == "5":
            train_models()
        elif choice == "6":
            run_tests()
        elif choice == "7":
            clean_project()
        elif choice == "8":
            view_stats()
        elif choice == "0":
            print(f"{Colors.CYAN}Goodbye!{Colors.RESET}")
            break
        else:
            print(f"{Colors.RED}Invalid choice!{Colors.RESET}")

        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
        sys.exit(0)
