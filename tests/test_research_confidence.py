import numpy as np
import pandas as pd

from scripts.enrich_research_report import threshold_quality
from scripts.run_robust_research import (
    _calibration_map,
    _candidate_horizons,
    _candidate_timeframes,
    _chronological,
    _ConservativeIsotonicCalibrator,
    _expected_calibration_error,
    _regime,
    _verified_history_limit,
)


def test_calibration_map_bounds_sparse_buckets() -> None:
    calibrated = _calibration_map(np.array([0.01, 0.99]), np.array([0.0, 1.0]))

    assert calibrated
    assert all(0.01 <= value <= 0.99 for value in calibrated.values())
    assert all(value not in {0.0, 1.0} for value in calibrated.values())


def test_conservative_calibrator_is_monotonic_and_does_not_extrapolate() -> None:
    calibrator = _ConservativeIsotonicCalibrator().fit(
        np.array([0.51, 0.52, 0.90, 0.91]), np.array([0.0, 1.0, 1.0, 1.0])
    )
    values = calibrator.predict(np.array([0.01, 0.5, 0.95, 0.999]))
    assert np.all(np.diff(values) >= -1e-12)
    assert values[-1] < 0.99


def test_expected_calibration_error_is_zero_for_a_calibrated_bucket() -> None:
    records = [
        {"confidence": 0.6, "win": True},
        {"confidence": 0.6, "win": True},
        {"confidence": 0.6, "win": True},
        {"confidence": 0.6, "win": False},
        {"confidence": 0.6, "win": False},
    ]

    assert _expected_calibration_error(records) == 0.0


def test_resolution_policy_does_not_fabricate_otc_precision() -> None:
    assert 41 in _candidate_horizons(5, "EURUSD_otc")
    assert _candidate_horizons(60, "EURUSD_otc") == [1, 2, 3, 5, 7]
    assert _candidate_horizons(60, "EURUSD") == [1, 2, 3, 5, 7]
    assert _candidate_timeframes(5, "EURUSD_otc") == [5, 60, 300, 900]
    assert _candidate_timeframes(60, "EURUSD") == [60, 300, 900]


def test_broker_rows_are_reordered_oldest_first() -> None:
    frame = pd.DataFrame({"time": [3, 1, 2], "close": [30, 10, 20]})
    assert _chronological(frame)["time"].tolist() == [1, 2, 3]


def test_history_request_stays_inside_verified_broker_limit() -> None:
    assert _verified_history_limit(5000) == 3200
    assert _verified_history_limit(3200) == 3200
    assert _verified_history_limit(100) == 3000


def test_regime_does_not_read_future_candles() -> None:
    close = np.concatenate([np.ones(300), np.linspace(1.0, 3.0, 300)])
    frame = pd.DataFrame({"close": close})
    before = _regime(frame, 250)
    frame.loc[300:, "close"] = np.linspace(1.0, 100.0, len(frame) - 300)
    assert _regime(frame, 250) == before


def test_confidence_quality_rejects_empty_high_confidence_buckets() -> None:
    quality = threshold_quality([{"confidence": 0.55, "win": True}] * 20)

    assert quality["monotonic_oos"] is False
    assert 0.70 in quality["insufficient_thresholds"]
    assert 0.99 in quality["insufficient_thresholds"]


def test_allocation_floor_is_separate_from_one_dollar_stake_floor() -> None:
    source = open("scripts/run_robust_research.py", encoding="utf-8").read()

    assert "max_risk * (0.05 + 0.95 * edge_scale * uncertainty_scale)" in source
    assert 'amount = float(max(1, round(amount_before)))' in source
