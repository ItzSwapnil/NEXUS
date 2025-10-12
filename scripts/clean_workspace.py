#!/usr/bin/env python3
"""
NEXUS - Workspace Cleaner
Removes cache files, logs, and temporary files (cross-platform)
"""

import sys
import shutil
from pathlib import Path
from typing import List


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def clean_pattern(pattern: str, recursive: bool = False) -> int:
    """Clean files matching a pattern."""
    count = 0
    glob_method = Path('.').rglob if recursive else Path('.').glob

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
    total_removed += clean_pattern('**/__pycache__', recursive=True)
    total_removed += clean_pattern('**/*.pyc', recursive=True)
    total_removed += clean_pattern('**/*.pyo', recursive=True)
    total_removed += clean_pattern('**/*.pyd', recursive=True)

    # Test and coverage cache
    print("\nCleaning test cache...")
    total_removed += clean_pattern('.pytest_cache', recursive=False)
    total_removed += clean_pattern('.coverage*', recursive=False)
    total_removed += clean_pattern('htmlcov', recursive=False)
    total_removed += clean_pattern('.tox', recursive=False)
    total_removed += clean_pattern('.hypothesis', recursive=False)

    # Type checking cache
    print("\nCleaning type checking cache...")
    total_removed += clean_pattern('.mypy_cache', recursive=False)
    total_removed += clean_pattern('.pytype', recursive=False)
    total_removed += clean_pattern('.ruff_cache', recursive=False)

    # Logs (keep directory structure)
    print("\nCleaning logs...")
    for log_file in Path('logs').glob('*.log'):
        try:
            log_file.unlink()
            print(f"  Removed: {log_file}")
            total_removed += 1
        except Exception as e:
            print(f"{Colors.YELLOW}  Could not remove {log_file}: {e}{Colors.RESET}")

    # Build artifacts
    print("\nCleaning build artifacts...")
    total_removed += clean_pattern('build', recursive=False)
    total_removed += clean_pattern('dist', recursive=False)
    total_removed += clean_pattern('*.egg-info', recursive=False)

    # IDE files
    print("\nCleaning IDE files...")
    total_removed += clean_pattern('.DS_Store', recursive=True)
    total_removed += clean_pattern('Thumbs.db', recursive=True)
    total_removed += clean_pattern('*.swp', recursive=True)
    total_removed += clean_pattern('*.swo', recursive=True)

    # Summary
    print(f"\n{Colors.GREEN}✓ Cleanup complete!{Colors.RESET}")
    print(f"  Removed {total_removed} items\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Cleanup cancelled{Colors.RESET}")
        sys.exit(1)
#!/usr/bin/env python3
"""
NEXUS - Project Verification Script
Verifies project integrity and readiness (cross-platform)
"""

import sys
from pathlib import Path
import subprocess


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def check(name: str, condition: bool) -> bool:
    """Check a condition and print result."""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if condition else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"  {name:.<50} {status}")
    return condition


def main():
    print_header("NEXUS - Project Verification")

    checks_passed = 0
    total_checks = 0

    # Check Python version
    print(f"{Colors.YELLOW}[1] Checking Python version...{Colors.RESET}")
    total_checks += 1
    if check("Python 3.10+", sys.version_info >= (3, 10)):
        checks_passed += 1

    # Check virtual environment
    print(f"\n{Colors.YELLOW}[2] Checking virtual environment...{Colors.RESET}")
    total_checks += 1
    if check("Virtual environment exists", Path('.venv').exists()):
        checks_passed += 1

    # Check configuration
    print(f"\n{Colors.YELLOW}[3] Checking configuration...{Colors.RESET}")
    total_checks += 1
    if check(".env file exists", Path('.env').exists()):
        checks_passed += 1

    # Check no sensitive files
    print(f"\n{Colors.YELLOW}[4] Checking security...{Colors.RESET}")
    total_checks += 1
    if check("No session.json", not Path('session.json').exists()):
        checks_passed += 1
    total_checks += 1
    if check("No settings/config.ini", not Path('settings/config.ini').exists()):
        checks_passed += 1

    # Check directories
    print(f"\n{Colors.YELLOW}[5] Checking project structure...{Colors.RESET}")
    for dir_name in ['logs', 'models', 'data', 'nexus', 'tests']:
        total_checks += 1
        if check(f"{dir_name}/ directory", Path(dir_name).exists()):
            checks_passed += 1

    # Check core files
    print(f"\n{Colors.YELLOW}[6] Checking core files...{Colors.RESET}")
    for file_name in ['pyproject.toml', 'setup.py', 'run.py', 'README.md']:
        total_checks += 1
        file_path = Path(file_name)
        if check(file_name, file_path.exists() and file_path.stat().st_size > 100):
            checks_passed += 1

    # Check core modules
    print(f"\n{Colors.YELLOW}[7] Checking Python modules...{Colors.RESET}")
    modules = [
        'nexus/ai/lstm_predictor.py',
        'nexus/ai/deep_rl_agent.py',
        'nexus/core/engine.py',
        'nexus/adapters/quotex_adapter.py'
    ]
    for module in modules:
        total_checks += 1
        if check(module, Path(module).exists()):
            checks_passed += 1

    # Run tests
    print(f"\n{Colors.YELLOW}[8] Running test suite...{Colors.RESET}")
    total_checks += 1
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v', '-q'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if check("All tests passing", result.returncode == 0):
            checks_passed += 1
    except Exception as e:
        check("All tests passing", False)
        print(f"    {Colors.RED}Error running tests: {e}{Colors.RESET}")

    # Summary
    print_header("Verification Summary")
    print(f"Total Checks: {total_checks}")
    print(f"{Colors.GREEN}Passed: {checks_passed}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {total_checks - checks_passed}{Colors.RESET}\n")

    if checks_passed == total_checks:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED!{Colors.RESET}")
        print(f"\n{Colors.CYAN}NEXUS is ready to use!{Colors.RESET}")
        print(f"  Next: {Colors.BOLD}python run.py{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Please fix the issues above.{Colors.RESET}\n")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification cancelled{Colors.RESET}")
        sys.exit(1)

