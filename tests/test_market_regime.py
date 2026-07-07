from __future__ import annotations

from datetime import datetime, timedelta, timezone

from samosbor.autonomy.market_regime import detect_market_regime
from samosbor.autonomy.regime_policy import resolve
from samosbor.config import (
    Confirmation5mPolicySection,
    LearningModeSection,
    LearningMicrostructureSection,
    LearningRiskSection,
    MarketSelloffImpulseSection,
    PaperAlphaCaptureSection,
    RegimePolicySection,
    SidePolicySection,
)
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


def _relaxed_policy_config():
    return type(
        "PolicyConfig",
        (),
        {
            "learning_mode": LearningModeSection(enabled=True, profile="relaxed_paper_learning"),
            "learning_risk": LearningRiskSection(),
            "learning_microstructure": LearningMicrostructureSection(),
            "confirmation_5m": Confirmation5mPolicySection(),
            "regime_policy": RegimePolicySection(),
            "side_policy": SidePolicySection(),
            "market_selloff_impulse": MarketSelloffImpulseSection(),
            "paper_alpha_capture": PaperAlphaCaptureSection(enabled=True),
        },
    )()


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


def test_market_selloff_impulse_detected_before_weak_down_choppy():
    base = [120.0 - step * 0.05 + (0.8 if step % 2 == 0 else -0.45) for step in range(90)]
    histories = {}
    for index in range(12):
        closes = [value - index * 0.02 for value in base]
        closes[-1] = closes[-2] * 0.985
        histories[f"SELL{index}"] = _candles_from_closes(closes)

    regime = detect_market_regime(histories)

    assert regime.regime == "market_selloff_impulse"
    assert regime.features["previous_regime"] == "weak_down_choppy"
    assert regime.features["symbols_confirming_count"] >= 10


def test_market_selloff_impulse_detected_without_adx():
    histories = {
        f"FAST{i}": _candles_from_closes([100.0, 99.6, 99.0, 98.2, 97.6, 97.0 - i * 0.01])
        for i in range(12)
    }

    regime = detect_market_regime(histories)

    assert regime.regime == "market_selloff_impulse"
    assert regime.features["symbols_confirming_count"] == 12


def test_market_selloff_impulse_uses_15m_fallback():
    histories = {
        f"FALL{i}": _candles_from_closes([100.0, 99.3, 98.5, 97.7, 97.0, 96.4 - i * 0.01])
        for i in range(12)
    }

    regime = detect_market_regime(histories)

    assert regime.regime == "market_selloff_impulse"
    assert regime.features["used_fallback"] is True
    assert regime.features["universe_ret_15m"] <= -0.004


def test_single_symbol_drop_does_not_trigger_selloff():
    histories = {
        "SINGLE": _candles_from_closes([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    }

    regime = detect_market_regime(histories)

    assert regime.regime != "market_selloff_impulse"


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
    config = _relaxed_policy_config()

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
    assert policy.entry_mode == "weak_choppy_direct_probe_short"
    assert policy.as_metadata()["probe_now_with_pending_addon"] is True
    assert policy.as_metadata()["would_have_waited_pullback_strict"] is True


def test_current_like_ydex_signal_becomes_probe_not_wait_only():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="YDEX",
        side=SignalDirection.SHORT,
        ml_feedback={"available": True, "blocks_entry": False, "action": "allow_entry"},
        book={"available": True, "spread_bps": 8.0, "entry_liquidity_cover": 2.5, "side_imbalance": 0.1},
        confirmation={"available": True, "ret_window": 0.001, "bars": 3},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.24,
        trend_strength=0.0015,
        adx=18.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.decision_type == "probe_trade"
    assert policy.entry_mode == "weak_choppy_direct_probe_short"
    assert policy.strict_policy_decision == "wait"
    assert policy.risk_multiplier >= 0.05
    assert "weak-down-choppy-direct-probe" in policy.soft_issues


def test_current_like_ozon_probation_reduces_size_not_wait_only():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="OZON",
        side=SignalDirection.SHORT,
        ml_feedback={"available": True, "blocks_entry": False, "action": "allow_entry"},
        book={"available": True, "spread_bps": 8.0, "entry_liquidity_cover": 2.5, "side_imbalance": 0.1},
        confirmation={"available": True, "ret_window": 0.001, "bars": 3},
        symbol_health="probation",
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.24,
        trend_strength=0.0015,
        adx=18.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.decision_type == "probe_trade"
    assert policy.entry_mode == "weak_choppy_direct_probe_short"
    assert "symbol-probation:OZON" in policy.soft_issues
    assert 0.05 <= policy.risk_multiplier <= 0.15


def test_weak_down_choppy_short_strong_rebound_still_waits_pullback():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 8.0, "entry_liquidity_cover": 2.5, "side_imbalance": 0.1},
        confirmation={"available": True, "ret_window": 0.006, "bars": 3},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.7,
        trend_strength=0.003,
        adx=25.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is False
    assert policy.decision_type == "wait_pullback"
    assert policy.entry_mode == "wait"


def test_weak_down_choppy_short_extreme_book_still_rejects():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 8.0, "entry_liquidity_cover": 2.5, "side_imbalance": -0.9},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.7,
        trend_strength=0.003,
        adx=25.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is False
    assert policy.decision_type == "hard_reject"
    assert "adverse-book-below-hard-limit" in policy.hard_issues


def test_clean_uptrend_long_can_open_probe_or_normal():
    policy = resolve(
        regime="clean_uptrend",
        symbol="SBER",
        side=SignalDirection.LONG,
        confirmation={"available": True, "ret_window": 0.003, "bars": 3},
        long_side_enabled=True,
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.42,
        trend_strength=0.003,
        adx=25.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.decision_type == "normal_trade"
    assert policy.entry_mode == "clean_uptrend_direct_long"
    assert policy.risk_multiplier <= 0.35


def test_weak_down_choppy_long_rebound_can_open_tiny_probe():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="SBER",
        side=SignalDirection.LONG,
        confirmation={"available": True, "ret_window": 0.004, "bars": 3, "latest_close": 101.0},
        long_side_enabled=True,
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.28,
        trend_strength=0.0015,
        adx=18.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.decision_type == "probe_trade"
    assert policy.entry_mode == "rebound_probe_long"
    assert policy.risk_multiplier <= 0.06
    assert policy.as_metadata()["long_probe_trade"] is True


def test_weak_down_choppy_long_without_rebound_is_shadow_only():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="SBER",
        side=SignalDirection.LONG,
        confirmation={"available": True, "ret_window": -0.001, "bars": 3},
        long_side_enabled=True,
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.28,
        trend_strength=0.0015,
        adx=18.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is False
    assert policy.decision_type == "shadow_only"
    assert policy.entry_mode == "wait_failed_breakdown_reclaim_long"


def test_selloff_fresh_breakdown_short_opens_now_not_wait():
    policy = resolve(
        regime="market_selloff_impulse",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 10.0, "entry_liquidity_cover": 1.5, "side_imbalance": -0.3},
        confirmation={"available": True, "ret_window": -0.002, "bars": 1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.45,
        trend_strength=0.003,
        adx=8.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.decision_type == "normal_trade"
    assert policy.entry_mode == "market_breakdown_short"
    assert policy.risk_multiplier == 0.60
    assert policy.as_metadata()["selloff_policy_override_applied"] is True


def test_selloff_neutral_5m_allows_reduced_short():
    policy = resolve(
        regime="market_selloff_impulse",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 10.0, "entry_liquidity_cover": 1.5, "side_imbalance": -0.3},
        confirmation={"available": True, "ret_window": 0.001, "bars": 1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.35,
        trend_strength=0.003,
        adx=8.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.entry_mode != "wait"
    assert policy.risk_multiplier < 0.60
    assert policy.confirmation_5m["mode"] == "allow_reduced_short"


def test_selloff_mild_rebound_allows_reduced_short():
    policy = resolve(
        regime="market_selloff_impulse",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 10.0, "entry_liquidity_cover": 1.5, "side_imbalance": -0.3},
        confirmation={"available": True, "ret_window": 0.003, "bars": 1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.28,
        trend_strength=0.003,
        adx=8.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.entry_mode in {"selloff_momentum_short", "panic_probe_short"}
    assert policy.confirmation_5m["mode"] == "allow_reduced_short"


def test_selloff_strong_rebound_stops_chase():
    policy = resolve(
        regime="market_selloff_impulse",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 10.0, "entry_liquidity_cover": 1.5, "side_imbalance": -0.3},
        confirmation={"available": True, "ret_window": 0.006, "bars": 1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.45,
        trend_strength=0.003,
        adx=8.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is False
    assert policy.decision_type == "wait_pullback"
    assert policy.entry_mode == "wait"


def test_selloff_moderate_bad_book_reduces_size_not_rejects():
    policy = resolve(
        regime="market_selloff_impulse",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 30.0, "entry_liquidity_cover": 0.8, "side_imbalance": -0.82},
        confirmation={"available": True, "ret_window": -0.002, "bars": 1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.28,
        trend_strength=0.003,
        adx=8.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.microstructure_bucket == "probe"
    assert policy.entry_mode != "wait"


def test_selloff_extreme_book_rejects():
    policy = resolve(
        regime="market_selloff_impulse",
        symbol="SBER",
        side=SignalDirection.SHORT,
        book={"available": True, "spread_bps": 45.0, "entry_liquidity_cover": 0.3, "side_imbalance": -0.98},
        confirmation={"available": True, "ret_window": -0.002, "bars": 1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.45,
        trend_strength=0.003,
        adx=8.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is False
    assert policy.decision_type == "hard_reject"
    assert "selloff-spread-above-hard-limit" in policy.hard_issues


def test_selloff_ml_negative_edge_reduces_not_blocks():
    policy = resolve(
        regime="market_selloff_impulse",
        symbol="SBER",
        side=SignalDirection.SHORT,
        ml_feedback={"available": True, "blocks_entry": True, "action": "reject_entry"},
        book={"available": True, "spread_bps": 10.0, "entry_liquidity_cover": 1.5, "side_imbalance": -0.3},
        confirmation={"available": True, "ret_window": -0.002, "bars": 1},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.45,
        trend_strength=0.003,
        adx=8.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.ml_negative_edge is True
    assert 0.0 < policy.risk_multiplier < 0.60
    assert "selloff-ml-negative-edge-reduced" in policy.soft_issues


def test_weak_down_choppy_ml_negative_edge_can_explore_not_wait_only():
    policy = resolve(
        regime="weak_down_choppy",
        symbol="SBER",
        side=SignalDirection.SHORT,
        ml_feedback={"available": True, "blocks_entry": True, "action": "reject_entry"},
        book={"available": True, "spread_bps": 8.0, "entry_liquidity_cover": 2.5, "side_imbalance": 0.1},
        confirmation={"available": True, "ret_window": 0.001, "bars": 3},
        learning_mode_enabled=True,
        learning_profile="relaxed_paper_learning",
        signal_strength=0.16,
        trend_strength=0.0015,
        adx=18.0,
        config=_relaxed_policy_config(),
    )

    assert policy.allow_trade is True
    assert policy.decision_type == "exploration_trade"
    assert policy.entry_mode == "weak_choppy_direct_exploration_short"
