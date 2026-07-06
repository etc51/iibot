from __future__ import annotations

from datetime import datetime, timedelta, timezone

from samosbor.autonomy.market_regime import detect_market_regime
from samosbor.autonomy.regime_policy import resolve
from samosbor.config import LearningModeSection, LearningRiskSection
from samosbor.domain import Candle, SignalDirection


def _candles_from_closes(closes: list[float]) -> list[Candle]:
    start = datetime(2026, 7, 6, 7, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        high = max(previous, close) + 0.2
        low = min(previous, close) - 0.2
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * index),
                open=previous,
                high=high,
                low=low,
                close=close,
                volume=1_000_000,
            )
        )
        previous = close
    return candles


def test_regime_detector_clean_downtrend():
    histories = {
        f"DOWN{i}": _candles_from_closes([120.0 - step * (0.35 + i * 0.02) for step in range(90)])
        for i in range(8)
    }

    regime = detect_market_regime(histories)

    assert regime.regime == "clean_downtrend"
    assert regime.confidence > 0.6
    assert regime.features["breadth_down"] == 1.0
    assert regime.features["median_adx"] >= 24.0
    assert regime.features["chop_score"] < 0.45


def test_regime_detector_weak_down_choppy():
    base = [120.0 - step * 0.08 + (0.9 if step % 2 == 0 else -0.5) for step in range(90)]
    histories = {f"CHOP{i}": _candles_from_closes([value - i * 0.03 for value in base]) for i in range(8)}

    regime = detect_market_regime(histories)

    assert regime.regime == "weak_down_choppy"
    assert regime.confidence > 0.5
    assert regime.features["breadth_down"] >= 0.58
    assert regime.features["chop_score"] >= 0.45


def test_regime_detector_range_chop():
    histories = {}
    for index in range(8):
        drift = 0.02 if index % 2 == 0 else -0.02
        closes = [100.0 + drift * step + (1.0 if step % 2 == 0 else -1.0) for step in range(90)]
        histories[f"RANGE{index}"] = _candles_from_closes(closes)

    regime = detect_market_regime(histories)

    assert regime.regime == "range_chop"
    assert regime.confidence > 0.5
    assert max(regime.features["breadth_down"], regime.features["breadth_up"]) <= 0.58
    assert regime.features["chop_score"] >= 0.45


def test_policy_blocks_trend_short_in_weak_down_choppy():
    policy = resolve(regime="weak_down_choppy", symbol="TRNFP", side=SignalDirection.SHORT)

    assert policy.allow_trade is False
    assert policy.entry_mode == "wait"
    assert policy.risk_multiplier == 0.0
    assert "weak-down-choppy-blocks-trend-short" in policy.reasons


def test_policy_allows_pullback_short_in_weak_down_choppy():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="TRNFP",
        side=SignalDirection.SHORT,
        entry_mode="pullback_short",
    )

    assert policy.allow_trade is True
    assert policy.entry_mode == "pullback_short"
    assert 0.0 < policy.risk_multiplier <= 0.5


def test_policy_blocks_long_by_default_when_long_side_disabled():
    policy = resolve(regime="clean_uptrend", symbol="SBER", side=SignalDirection.LONG)

    assert policy.allow_trade is False
    assert policy.entry_mode == "reject"
    assert policy.risk_multiplier == 0.0
    assert "long-side-disabled" in policy.reasons


def test_ml_negative_edge_fractional_only_when_no_hard_blocks():
    policy = resolve(
        regime="clean_downtrend",
        symbol="SBER",
        side=SignalDirection.SHORT,
        ml_feedback={"blocks_entry": True},
        book={"side_imbalance": 0.1},
    )

    assert policy.allow_trade is True
    assert policy.risk_multiplier == 0.25
    assert "ml-negative-edge-fractional" in policy.reasons


def test_ml_negative_edge_plus_adverse_book_is_rejected():
    policy = resolve(
        regime="clean_downtrend",
        symbol="SBER",
        side=SignalDirection.SHORT,
        ml_feedback={"blocks_entry": True},
        book={"side_imbalance": -0.5},
    )

    assert policy.allow_trade is False
    assert policy.entry_mode == "reject"
    assert policy.risk_multiplier == 0.0
    assert "adverse-book" in policy.reasons


def test_symbol_probation_reduces_size():
    policy = resolve(
        regime="clean_downtrend",
        symbol="TRNFP",
        side=SignalDirection.SHORT,
        symbol_health="probation",
    )

    assert policy.allow_trade is True
    assert policy.risk_multiplier == 0.5
    assert "symbol-probation:TRNFP" in policy.reasons


def test_weak_symbol_observe_only_blocks_entry():
    policy = resolve(
        regime="clean_downtrend",
        symbol="TRNFP",
        side=SignalDirection.SHORT,
        symbol_health="observe_only",
    )

    assert policy.allow_trade is False
    assert policy.entry_mode == "reject"
    assert policy.risk_multiplier == 0.0
    assert "symbol-observe-only:TRNFP" in policy.reasons


def test_relaxed_learning_allows_probe_when_strict_would_wait_pullback():
    config = type(
        "PolicyConfig",
        (),
        {
            "learning_mode": LearningModeSection(enabled=True, profile="relaxed_paper_learning"),
            "learning_risk": LearningRiskSection(),
        },
    )()

    policy = resolve(
        regime="weak_down_choppy",
        symbol="TRNFP",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 8.0, "entry_liquidity_cover": 2.5, "side_imbalance": 0.1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.65,
        trend_strength=0.003,
        adx=25.0,
        config=config,
    )

    assert policy.allow_trade is True
    assert policy.decision_type == "probe_trade"
    assert policy.strict_policy_decision == "wait"
    assert policy.would_strict_policy_trade is False
    assert policy.relaxed_only_trade is True
    assert policy.risk_multiplier < 1.0
