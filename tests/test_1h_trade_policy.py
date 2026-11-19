"""Unit checks for the evidence-gated demo-run policy helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_1h_trade_test.py"
_SPEC = importlib.util.spec_from_file_location("one_hour_trade_test", _SCRIPT)
assert _SPEC and _SPEC.loader
one_hour = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(one_hour)


def test_binary_option_break_even_and_expected_value() -> None:
    assert one_hour.break_even_win_rate(100.0) == pytest.approx(0.5)
    assert one_hour.break_even_win_rate(90.0) == pytest.approx(1 / 1.9)
    assert one_hour.expected_value(0.6, 90.0) == pytest.approx(0.14)
    assert one_hour.expected_value(0.5, 90.0) < 0.0


def test_position_size_is_edge_aware_and_capped() -> None:
    assert one_hour.position_size(20_000.0, 0.50, 93.0) == 0.0
    amount = one_hour.position_size(20_000.0, 0.90, 93.0)
    assert one_hour.MIN_TRADE_AMOUNT <= amount <= one_hour.MAX_TRADE_AMOUNT


def test_ai_allocation_uses_model_expiry_and_risk_capped_stake() -> None:
    allocation = one_hour.ai_allocation_decision(
        {
            "confidence": 0.8,
            "raw_confidence": 0.86,
            "recommended_expiration": 300,
            "confidence_type": "market-specific calibrated probability",
        },
        balance=20_000.0,
        payout_percent=90.0,
    )

    assert allocation["model"] == "RealAITradingEngine"
    assert allocation["ai_recommended_expiration_seconds"] == 300
    assert allocation["risk_authorized_amount"] <= one_hour.MAX_TRADE_AMOUNT
    assert allocation["expected_value_per_dollar"] > 0.0


def test_exploration_allocation_has_a_separate_small_practice_cap() -> None:
    allocation = one_hour.ai_allocation_decision(
        {"confidence": 0.9, "recommended_expiration": 30},
        balance=20_000.0,
        payout_percent=90.0,
        exploration=True,
    )

    assert allocation["allocation_mode"] == "practice_exploration"
    assert allocation["risk_authorized_amount"] <= one_hour.EXPLORATION_MAX_TRADE_AMOUNT
    assert allocation["risk_cap_fraction"] == one_hour.EXPLORATION_MAX_RISK_FRACTION


def test_exploration_stake_increases_with_the_ai_calibrated_edge() -> None:
    low = one_hour.ai_allocation_decision(
        {"confidence": 0.65, "recommended_expiration": 30}, 20_000.0, 90.0, exploration=True
    )
    high = one_hour.ai_allocation_decision(
        {"confidence": 0.85, "recommended_expiration": 300}, 20_000.0, 90.0, exploration=True
    )

    assert low["ai_recommended_expiration_seconds"] == 30
    assert high["ai_recommended_expiration_seconds"] == 300
    assert low["risk_authorized_amount"] < high["risk_authorized_amount"]


def test_ai_requested_price_is_used_unless_a_hard_risk_limit_applies() -> None:
    allocation = one_hour.ai_allocation_decision(
        {"confidence": 0.80, "recommended_expiration": 30, "recommended_risk_fraction": 0.0004},
        balance=10_000.0,
        payout_percent=90.0,
    )
    capped = one_hour.ai_allocation_decision(
        {"confidence": 0.90, "recommended_expiration": 30, "recommended_risk_fraction": 0.01},
        balance=10_000.0,
        payout_percent=90.0,
    )

    assert allocation["ai_requested_amount"] == pytest.approx(4.0)
    assert allocation["risk_authorized_amount"] == pytest.approx(4.0)
    assert not allocation["risk_limited"]
    assert capped["risk_limited"]
    assert capped["risk_authorized_amount"] <= one_hour.MAX_TRADE_AMOUNT


def test_stake_scales_with_total_balance_for_the_same_calibrated_edge() -> None:
    smaller = one_hour.ai_allocation_decision(
        {"confidence": 0.75, "recommended_expiration": 61}, 5_000.0, 90.0
    )
    larger = one_hour.ai_allocation_decision(
        {"confidence": 0.75, "recommended_expiration": 61}, 10_000.0, 90.0
    )

    assert smaller["risk_authorized_amount"] < larger["risk_authorized_amount"]
    assert larger["risk_authorized_amount"] == pytest.approx(
        2 * smaller["risk_authorized_amount"], abs=0.02
    )


def test_expiration_obeys_market_specific_minimum_with_one_second_resolution() -> None:
    assert one_hour.normalized_expiration(7.4, is_otc=True) == 7
    assert one_hour.normalized_expiration(1, is_otc=True) == 5
    assert one_hour.normalized_expiration(61.2, is_otc=False) == 61
    assert one_hour.normalized_expiration(5, is_otc=False) == 60


def test_demo_order_requires_real_broker_evidence() -> None:
    assert one_hour.is_broker_confirmed_demo_order(
        {"success": True, "real_executed": True, "order_id": "abc-123"}
    )
    assert not one_hour.is_broker_confirmed_demo_order(
        {"success": True, "real_executed": False, "order_id": "SIM-123"}
    )
    assert not one_hour.is_broker_confirmed_demo_order(
        {"success": True, "real_executed": True, "order_id": ""}
    )


def test_unpromoted_or_uncalibrated_research_cannot_open_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "research.json"
    report.write_text(
        '{"markets": [{"asset": "USDZAR_otc", "status": "complete", '
        '"promotion_status": "reject", "trades": 100, "win_rate": 0.9, '
        '"net_pnl": 100, "calibration_error": 0.01, "payout": 93}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(one_hour, "REPORT_PATH", report)

    passed, _reason, evidence = one_hour.load_walk_forward_gates()["USDZAR_OTC"]

    assert not passed
    assert evidence["promotion_status"] == "reject"
