"""
NEXUS Master Launch Panel

Single entry point providing an interactive console panel to:
- Initialize engine
- View performance stats & emotion state
- Simulate demo trades
- Run regime detection sample
- Run optional Playwright smoke test (if installed)

Also supports a non-interactive --auto-demo mode for quick CI smoke.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

from nexus.core.engine import NexusEngine
from nexus.utils.config import load_config, NexusSettings  # fixed typo
from nexus.intelligence.regime_detector import RegimeDetector

console = Console()

# --------------------------- Utility Helpers --------------------------- #

def banner(version: str = "2.0.0") -> None:
    console.print(Panel.fit(
        f"[bold cyan]NEXUS[/bold cyan] [dim]v{version}[/dim]\n[green]Self-Evolving AI Trader (Master Panel)[/green]",
        border_style="cyan"))


def build_engine(settings: NexusSettings) -> NexusEngine:
    return NexusEngine(settings=settings, demo_mode=True, auto_login=False)


async def init_engine(settings: NexusSettings) -> NexusEngine:
    engine = build_engine(settings)
    await engine.initialize_components()
    return engine


def show_stats(engine: NexusEngine) -> None:
    stats = engine.get_performance_stats()
    emotions = engine.emotion_state
    table = Table(title="Performance", box=box.SIMPLE, show_edge=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(k, str(v))
    table.add_row("greed (emotion)", f"{emotions.get('greed', 0):.2f}")
    table.add_row("fear (emotion)", f"{emotions.get('fear', 0):.2f}")
    table.add_row("confidence (emotion)", f"{emotions.get('confidence', 0):.2f}")
    console.print(table)


def simulate_trades(engine: NexusEngine, n: int = 5) -> None:
    for _ in range(n):
        success = random.random() < 0.55
        profit = round(random.uniform(1, 15), 2) if success else -round(random.uniform(1, 10), 2)
        engine.record_trade(success=success, profit=profit)
    console.print(f"[bold green]Simulated {n} trades.[/bold green]")


def run_regime_detection() -> None:
    try:
        import pandas as pd
        import numpy as np
    except Exception:  # pragma: no cover
        console.print("[yellow]pandas/numpy not available for regime detection sample[/yellow]")
        return
    detector = RegimeDetector(n_regimes=4, lookback_periods=75)
    data = pd.DataFrame({
        'open': np.random.rand(detector.lookback_periods),
        'high': np.random.rand(detector.lookback_periods),
        'low': np.random.rand(detector.lookback_periods),
        'close': np.random.rand(detector.lookback_periods),
        'volume': np.random.rand(detector.lookback_periods)
    })
    regime = asyncio.run(detector.detect_regime(data))
    console.print(f"[bold blue]Detected Regime:[/bold blue] {regime}")


def playwright_smoke() -> None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        console.print("[yellow]Playwright not installed.[/yellow]")
        return
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception:
                console.print("[yellow]Chromium browser not installed for Playwright.[/yellow]")
                return
            page = browser.new_page()
            page.goto("data:text/html,<h1>NEXUS OK</h1>")
            ok = page.text_content("h1") == "NEXUS OK"
            browser.close()
        console.print("[green]Playwright smoke:[/green] " + ("PASS" if ok else "FAIL"))
    except Exception as e:  # pragma: no cover
        console.print(f"[red]Playwright error: {e}[/red]")


# --------------------------- Interactive Panel --------------------------- #
async def interactive_panel(settings: NexusSettings) -> None:
    engine: Optional[NexusEngine] = None
    while True:
        banner(settings.version)
        console.print("[bold]Menu[/bold]:\n"
                      "1) Initialize Engine\n"
                      "2) Show Stats\n"
                      "3) Simulate Trades\n"
                      "4) Regime Detection Sample\n"
                      "5) Playwright Smoke Test\n"
                      "6) Exit")
        choice = Prompt.ask("Select", choices=["1","2","3","4","5","6"], default="2")
        if choice == "1":
            if engine is None:
                console.print("[cyan]Initializing engine...[/cyan]")
                engine = await init_engine(settings)
                console.print("[green]Engine ready.[/green]")
            else:
                console.print("[yellow]Engine already initialized.[/yellow]")
        elif choice == "2":
            if engine:
                show_stats(engine)
            else:
                console.print("[yellow]Initialize engine first (option 1).[/yellow]")
        elif choice == "3":
            if engine:
                simulate_trades(engine)
            else:
                console.print("[yellow]Initialize engine first (option 1).[/yellow]")
        elif choice == "4":
            run_regime_detection()
        elif choice == "5":
            playwright_smoke()
        elif choice == "6":
            console.print("[bold magenta]Goodbye.[/bold magenta]")
            break
        # Small separator
        console.print(Panel.fit("Press Enter to continue", style="dim"))
        try:
            input()
        except EOFError:
            break


# --------------------------- Auto Demo Mode --------------------------- #
async def auto_demo(settings: NexusSettings) -> None:
    console.print("[cyan]Running auto demo...[/cyan]")
    engine = await init_engine(settings)
    simulate_trades(engine, n=10)
    show_stats(engine)
    run_regime_detection()
    playwright_smoke()
    console.print("[green]Auto demo complete.[/green]")


# --------------------------- CLI Entry --------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEXUS Master Launch Panel")
    p.add_argument('-c', '--config', type=str, help='Path to config YAML (default: config.yaml)')
    p.add_argument('--auto-demo', action='store_true', help='Run non-interactive demo and exit')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_config(args.config) if args.config else load_config()
    if args.auto_demo:
        asyncio.run(auto_demo(settings))
    else:
        try:
            asyncio.run(interactive_panel(settings))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == '__main__':
    main()
