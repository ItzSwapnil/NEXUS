#!/usr/bin/env python3
"""
NEXUS Quick Launch CLI - Cross-platform launcher
Author: Swapnil De Sarkar
Created: 2025

Usage:
    python run.py
    or: python -m nexus.launcher
"""

import os
import sys
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes."""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


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
    venv_python = Path('.venv') / ('Scripts' if sys.platform == 'win32' else 'bin') / ('python.exe' if sys.platform == 'win32' else 'python')
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
    if not Path('.env').exists():
        print(f"{Colors.RED}⚠ Warning: .env file not found!{Colors.RESET}")
        print(f"{Colors.YELLOW}Please run setup.py first or create .env from .env.example{Colors.RESET}")
        return False
    return True


def launch_gui():
    """Launch GUI in demo mode."""
    python = get_python()
    run_command([python, '-m', 'nexus.main', '--gui', '--demo'], "Launching GUI")


def run_cli_stats():
    """Run CLI with stats."""
    python = get_python()
    run_command([python, '-m', 'nexus.main', '--cli', '--demo', '--stats'], "Running CLI")


def run_backtest():
    """Run backtest."""
    python = get_python()
    run_command([python, '-m', 'nexus.main', '--cli', '--backtest', '--demo'], "Running backtest")


def run_live_backtest():
    """Run live backtest with Quotex demo data."""
    python = get_python()
    run_command([python, '-m', 'nexus.main', '--cli', '--live-backtest', '--demo'], "Running live backtest")


def train_models():
    """Train AI models."""
    python = get_python()
    run_command([python, '-m', 'nexus.ai.train_models'], "Training AI models")


def run_tests():
    """Run test suite."""
    python = get_python()
    run_command([python, '-m', 'pytest', '-v'], "Running tests")


def clean_project():
    """Clean cache and temporary files."""
    print(f"\n{Colors.YELLOW}Cleaning project...{Colors.RESET}")
    
    patterns = [
        'logs/*.log',
        'data/cache/*',
        '**/__pycache__',
        '**/*.pyc',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache'
    ]
    
    import glob
    import shutil
    
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
    run_command([python, '-c', '''
from nexus.core.engine import NexusEngine
from nexus.utils.config import load_runtime_settings
import json

settings = load_runtime_settings()
engine = NexusEngine(settings=settings, demo_mode=True)
stats = engine.get_performance_stats()

print("\\n" + "="*60)
print("NEXUS Performance Statistics")
print("="*60)
for key, value in stats.items():
    print(f"{key:.<40} {value}")
print("="*60)
'''], "Loading performance stats")


def main():
    """Main launcher loop."""
    # Check for .env file
    if not check_env():
        response = input("\nContinue anyway? (y/N): ").lower()
        if response not in ('y', 'yes'):
            sys.exit(1)
    
    while True:
        print_menu()
        
        try:
            choice = input(f"{Colors.CYAN}Enter choice (0-8): {Colors.RESET}").strip()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
            break
        
        if choice == '1':
            launch_gui()
        elif choice == '2':
            run_cli_stats()
        elif choice == '3':
            run_backtest()
        elif choice == '4':
            run_live_backtest()
        elif choice == '5':
            train_models()
        elif choice == '6':
            run_tests()
        elif choice == '7':
            clean_project()
        elif choice == '8':
            view_stats()
        elif choice == '0':
            print(f"{Colors.CYAN}Goodbye!{Colors.RESET}")
            break
        else:
            print(f"{Colors.RED}Invalid choice!{Colors.RESET}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
        sys.exit(0)
#!/usr/bin/env python3
"""
NEXUS Setup CLI - Cross-platform automated setup.

Usage:
    python setup.py
    or: python -m nexus.setup
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for cross-platform terminal output."""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def run_command(cmd: List[str], check: bool = True) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_uv():
    """Check if uv is installed."""
    if run_command(['uv', '--version'], check=False):
        print_success("uv package manager found")
        return True
    else:
        print_warning("uv not found")
        print(f"{Colors.YELLOW}Install with: pip install uv{Colors.RESET}")
        return False


def create_venv():
    """Create virtual environment."""
    print(f"{Colors.YELLOW}Creating virtual environment...{Colors.RESET}")
    
    venv_path = Path('.venv')
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        return True
    
    if run_command(['uv', 'venv', '.venv']):
        print_success("Virtual environment created")
        return True
    else:
        # Fallback to standard venv
        if run_command([sys.executable, '-m', 'venv', '.venv']):
            print_success("Virtual environment created (standard)")
            return True
        print_error("Failed to create virtual environment")
        return False


def get_pip_executable() -> str:
    """Get the pip executable path in virtual environment."""
    if sys.platform == 'win32':
        return str(Path('.venv') / 'Scripts' / 'pip.exe')
    else:
        return str(Path('.venv') / 'bin' / 'pip')


def get_python_executable() -> str:
    """Get the python executable path in virtual environment."""
    if sys.platform == 'win32':
        return str(Path('.venv') / 'Scripts' / 'python.exe')
    else:
        return str(Path('.venv') / 'bin' / 'python')


def install_dependencies(options: List[str]):
    """Install dependencies with selected options."""
    print(f"\n{Colors.YELLOW}Installing dependencies...{Colors.RESET}")
    
    pip = get_pip_executable()
    
    # Upgrade pip first
    print("Upgrading pip...")
    run_command([pip, 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    
    # Install core dependencies
    print("Installing core dependencies...")
    if run_command([pip, 'install', '-e', '.']):
        print_success("Core dependencies installed")
    else:
        print_error("Failed to install core dependencies")
        return False
    
    # Install optional dependencies
    for option in options:
        if option:
            print(f"Installing {option} dependencies...")
            if run_command([pip, 'install', '-e', f'.[{option}]']):
                print_success(f"{option} dependencies installed")
            else:
                print_warning(f"Failed to install {option} dependencies")
    
    return True


def create_directories():
    """Create necessary project directories."""
    print(f"\n{Colors.YELLOW}Creating project directories...{Colors.RESET}")
    
    directories = [
        'data',
        'data/cache',
        'logs',
        'models',
        'reports',
        'settings'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_success("Directories created")


def setup_env_file():
    """Set up .env file from template."""
    print(f"\n{Colors.YELLOW}Setting up environment configuration...{Colors.RESET}")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print_warning(".env file already exists")
        return
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print_success("Created .env file from template")
        print(f"{Colors.RED}IMPORTANT: Edit .env file with your credentials!{Colors.RESET}")
    else:
        print_error(".env.example not found")


def run_tests():
    """Run test suite."""
    print(f"\n{Colors.YELLOW}Running tests...{Colors.RESET}")
    
    python = get_python_executable()
    
    if run_command([python, '-m', 'pytest', '-v', '--tb=short']):
        print_success("All tests passed!")
        return True
    else:
        print_warning("Some tests failed")
        return False


def main():
    """Main setup routine."""
    print_header("NEXUS AI Trading System - Setup")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check for uv
    has_uv = check_uv()
    
    # Create virtual environment
    if not create_venv():
        sys.exit(1)
    
    # Ask user what to install
    print(f"\n{Colors.CYAN}Optional Features Installation{Colors.RESET}")
    print("=" * 80)
    
    options = []
    
    # AI/ML
    response = input(f"{Colors.YELLOW}Install AI/ML dependencies (PyTorch, TensorFlow)? (y/N): {Colors.RESET}").lower()
    if response in ('y', 'yes'):
        options.append('ai')
    
    # GUI
    response = input(f"{Colors.YELLOW}Install GUI dependencies (PySide6)? (y/N): {Colors.RESET}").lower()
    if response in ('y', 'yes'):
        options.append('gui')
    
    # Quotex
    response = input(f"{Colors.YELLOW}Install Quotex broker adapter? (y/N): {Colors.RESET}").lower()
    if response in ('y', 'yes'):
        options.append('quotex')
    
    # Technical Analysis
    response = input(f"{Colors.YELLOW}Install Technical Analysis libraries? (y/N): {Colors.RESET}").lower()
    if response in ('y', 'yes'):
        options.append('ta')
    
    # Development tools
    response = input(f"{Colors.YELLOW}Install development tools? (y/N): {Colors.RESET}").lower()
    if response in ('y', 'yes'):
        # Install dev dependencies
        pip = get_pip_executable()
        run_command([pip, 'install', '-e', '.[dev]'])
    
    # Install selected dependencies
    if not install_dependencies(options):
        print_error("Dependency installation failed")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup .env file
    setup_env_file()
    
    # Ask to run tests
    response = input(f"\n{Colors.YELLOW}Run tests to verify installation? (Y/n): {Colors.RESET}").lower()
    if response not in ('n', 'no'):
        run_tests()
    
    # Success summary
    print_header("Setup Complete!")
    print(f"{Colors.GREEN}Next steps:{Colors.RESET}")
    print(f"  1. Edit .env file with your credentials")
    print(f"  2. Run: {Colors.CYAN}python run.py{Colors.RESET}")
    print(f"  3. Or run: {Colors.CYAN}python -m nexus.main --demo{Colors.RESET}")
    print(f"\n{Colors.GREEN}Happy Trading! 🚀{Colors.RESET}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup cancelled by user{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)

