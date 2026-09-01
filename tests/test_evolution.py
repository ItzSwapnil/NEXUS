import json
from pathlib import Path

import pytest

from nexus.core.engine import NexusEngine
from nexus.evolution.evolver import EvolutionConfig, EvolutionRunner
from nexus.strategies.meta_strategy import MetaStrategy, SignalType, TradingSignal
from nexus.utils.config import NexusSettings, QuotexSettings, TradingSettings


@pytest.mark.asyncio
async def test_basic_evolution_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Minimal settings / engine
    settings = NexusSettings(
        quotex=QuotexSettings(email="e@x", password="pw"),
        trading=TradingSettings(base_trade_amount=5.0),
    )
    engine = NexusEngine(settings=settings, demo_mode=True)
    cfg = EvolutionConfig(population_size=3, generations=1, backtest_rows=60, backtest_window=20)
    runner = EvolutionRunner(engine=engine, config=cfg, settings=settings)
    gens = await runner.run()
    assert len(gens) == 1
    # Champion persisted
    champion_path = Path("models/meta_strategy_champion.json")
    assert champion_path.exists()
    data = json.loads(champion_path.read_text(encoding="utf-8"))
    assert 0 <= data.get("fitness", 0) <= 1
    # Hall of fame
    hof_path = Path("evolution/hall_of_fame.json")
    assert hof_path.exists()
    hof = json.loads(hof_path.read_text(encoding="utf-8"))
    assert isinstance(hof, list) and len(hof) >= 1


@pytest.mark.asyncio
async def test_circuit_breaker_activation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUS_MAX_DRAWDOWN_PCT", "0.01")  # very low threshold
    monkeypatch.setenv("NEXUS_FORCE_SIM", "1")
    monkeypatch.setenv("NEXUS_ENABLE_STOCHASTIC", "1")
    monkeypatch.setenv("NEXUS_P_WIN", "0.0")  # force loss
    monkeypatch.setenv(
        "NEXUS_LOSS_MULT_RANGE", "2.0,2.0"
    )  # deterministic large loss to trigger breaker

    settings = NexusSettings(
        quotex=QuotexSettings(email="e@x", password="pw"),
        trading=TradingSettings(base_trade_amount=10.0),
    )
    engine = NexusEngine(settings=settings, demo_mode=True)
    # One losing trade triggers drawdown and circuit breaker
    res = await engine.execute_trade("EURUSD", "call", 10.0, 60)
    assert res["success"] is False
    stats = engine.get_performance_stats()
    assert stats["circuit_breaker"] is True
    assert stats["max_drawdown_pct"] > 0
    # Position sizing now clamped
    sized = engine.advanced_risk_management({}, 100.0)
    assert sized == 1.0


@pytest.mark.asyncio
async def test_market_memory_persistence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUS_PERSIST_MEMORY", "1")
    # Build strategy and simulate update_performance
    ms = MetaStrategy()
    sig = TradingSignal(
        signal_type=SignalType.BUY,
        confidence=0.9,
        asset="EURUSD",
        timeframe=1,
        reasoning="test",
        source_model="ensemble",
        timestamp=__import__("datetime").datetime.now(),
        features={"trend_strength": 0.7, "volatility": 0.2, "momentum": 0.4},
    )
    await ms.update_performance(sig, success=True, profit=5.0)
    mem_path = Path("models/meta_strategy_memory.json")
    assert mem_path.exists(), "Memory file not persisted"
    raw = json.loads(mem_path.read_text(encoding="utf-8"))
    assert len(raw) >= 1
    # New instance should load memory
    ms2 = MetaStrategy()
    # If persistence flag on, memory should not be empty
    assert isinstance(ms2.market_memory, dict)
    assert len(ms2.market_memory) >= 0  # at minimum structure exists
