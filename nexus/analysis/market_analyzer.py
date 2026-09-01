"""Market-wide AI analysis and transparent trading scenarios.

This module estimates opportunity counts and expected P&L; it does not place
orders. Estimates are only meaningful when the input catalog and candles are
live and are deliberately labelled as scenarios rather than forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from nexus.catalog.ingest import Market


@dataclass(frozen=True)
class Scenario:
    autonomy: float
    min_confidence: float
    eligible_markets: int
    trades_10m: int
    trades_15m: int
    trades_30m: int
    trades_1h: int
    expected_profit_10m: float
    expected_profit_15m: float
    expected_profit_30m: float
    expected_profit_1h: float


@dataclass
class MarketOpportunity:
    symbol: str
    active: bool
    payout_1m: float
    payout_source: str
    signal: str
    confidence: float
    regime: str
    expected_value_per_unit: float
    recommended_entry_price: Optional[float] = None
    price_gate: str = "Disabled"
    recommended_expiration: int = 60
    reasoning: str = ""


@dataclass
class MarketAnalysisResult:
    generated_at: str
    opportunities: List[MarketOpportunity] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)
    best_scenario: Optional[Scenario] = None
    assumptions: List[str] = field(default_factory=list)


CandlesFetcher = Callable[[str], Awaitable[Optional[List[Dict[str, float]]]]]


def _price_gate(candles: Optional[List[Dict[str, float]]], signal: str) -> tuple[Optional[float], str]:
    if not candles:
        return None, "No live candles"
    try:
        closes = [float(c["close"]) for c in candles if float(c["close"]) > 0]
        if len(closes) < 5:
            return None, "Insufficient live candles"
        current = closes[-1]
        window = closes[-min(20, len(closes)) :]
        low, high = min(window), max(window)
        # A price gate is a timing suggestion, not a guaranteed price.
        if signal == "call":
            return round(low + (high - low) * 0.35, 8), "Enter only at/below pullback gate"
        if signal == "put":
            return round(high - (high - low) * 0.35, 8), "Enter only at/above rebound gate"
        return current, "Neutral price gate"
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "Invalid live candles"


class MarketAnalyzer:
    """Analyze all catalog markets using the existing AI engine."""

    def __init__(
        self,
        base_stake: float = 1.0,
        cycle_seconds: int = 60,
        max_markets_per_cycle: int = 100,
    ) -> None:
        self.base_stake = max(0.01, float(base_stake))
        self.cycle_seconds = max(1, int(cycle_seconds))
        self.max_markets_per_cycle = max(1, int(max_markets_per_cycle))

    async def analyze(
        self,
        markets: List[Market],
        ai_engine: Any,
        candles_fetcher: Optional[CandlesFetcher] = None,
        min_confidence: float = 0.70,
        autonomy: float = 0.50,
        use_price_gate: bool = True,
    ) -> MarketAnalysisResult:
        confidence_floor = min(0.99, max(0.50, float(min_confidence)))
        autonomy_value = min(1.0, max(0.0, float(autonomy)))
        opportunities: List[MarketOpportunity] = []

        for market in markets[: self.max_markets_per_cycle]:
            if not market.active:
                continue
            candles = await candles_fetcher(market.symbol) if candles_fetcher else None
            try:
                prediction = await ai_engine.analyze_market(
                    candles,
                    asset=market.symbol,
                    timeframe=60,
                    is_otc=market.otc,
                )
            except Exception as exc:  # one unavailable market must not stop the scan
                prediction = {"signal": "hold", "confidence": 0.0, "reasoning": str(exc)}

            signal = str(prediction.get("signal", "hold")).lower()
            confidence = min(1.0, max(0.0, float(prediction.get("confidence", 0.0))))
            payout = max(0.0, float(market.effective_payout("60")))
            # Net binary expected value per unit stake.
            expected_value = confidence * (payout / 100.0) - (1.0 - confidence)
            entry_price, gate_text = _price_gate(candles, signal)
            if not use_price_gate:
                gate_text = "Disabled"
                entry_price = None
            opportunities.append(
                MarketOpportunity(
                    symbol=market.symbol,
                    active=market.active,
                    payout_1m=payout,
                    payout_source=str(market.metadata.get("payout_source", "catalog data")),
                    signal=signal,
                    confidence=confidence,
                    regime=str(prediction.get("regime", "UNKNOWN")),
                    expected_value_per_unit=expected_value,
                    recommended_entry_price=entry_price,
                    price_gate=gate_text,
                    recommended_expiration=int(prediction.get("recommended_expiration", 60)),
                    reasoning=str(prediction.get("reasoning", "")),
                )
            )

        candidate_pool = [
            item
            for item in opportunities
            if item.expected_value_per_unit > 0
            and item.signal in {"call", "put"}
        ]
        scenarios = self._scenarios(candidate_pool, autonomy_value, confidence_floor)
        best = max(scenarios, key=lambda item: item.expected_profit_1h, default=None)
        return MarketAnalysisResult(
            generated_at=datetime.now(UTC).isoformat(),
            opportunities=sorted(
                opportunities,
                key=lambda item: item.expected_value_per_unit,
                reverse=True,
            ),
            scenarios=scenarios,
            best_scenario=best,
            assumptions=[
                "Expected P&L = AI confidence × payout − loss probability, per unit stake.",
                "Trade counts assume one decision cycle per configured interval across eligible markets.",
                "Projected profit is a scenario estimate, not a guarantee or financial advice.",
                "Payouts are read from the supplied catalog; stale/fallback catalog data is labelled.",
            ],
        )

    def _scenarios(
        self, eligible: List[MarketOpportunity], autonomy: float, min_confidence: float
    ) -> List[Scenario]:
        scenarios: List[Scenario] = []
        for auto in sorted({0.25, 0.50, 0.75, 1.0, round(autonomy, 2)}):
            for confidence in sorted({0.65, 0.70, 0.75, 0.80, 0.85, round(min_confidence, 2)}):
                selected = [item for item in eligible if item.confidence >= confidence]
                count = len(selected)
                avg_ev = (
                    sum(item.expected_value_per_unit for item in selected) / count
                    if count
                    else 0.0
                )
                per_cycle = count * auto
                estimates = []
                for minutes in (10, 15, 30, 60):
                    trades = int(per_cycle * (minutes * 60 / self.cycle_seconds))
                    estimates.append((trades, round(trades * avg_ev * self.base_stake, 2)))
                scenarios.append(
                    Scenario(
                        autonomy=auto,
                        min_confidence=confidence,
                        eligible_markets=count,
                        trades_10m=estimates[0][0],
                        trades_15m=estimates[1][0],
                        trades_30m=estimates[2][0],
                        trades_1h=estimates[3][0],
                        expected_profit_10m=estimates[0][1],
                        expected_profit_15m=estimates[1][1],
                        expected_profit_30m=estimates[2][1],
                        expected_profit_1h=estimates[3][1],
                    )
                )
        return scenarios


__all__ = ["MarketAnalyzer", "MarketAnalysisResult", "MarketOpportunity", "Scenario"]
