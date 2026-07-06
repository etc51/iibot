from __future__ import annotations

from datetime import datetime, timedelta, timezone

from samosbor.autonomy.ml_learning import (
    COMMISSION_EDGE_TAG,
    CONFIRMATION_AFTER_IMPULSE_TAG,
    ML_NEGATIVE_EDGE_POSITION_SCALE,
    LATE_REENTRY_TAG,
    LOW_QUALITY_TAG,
    SHORT_AFTER_EXHAUSTION_TAG,
    assess_signal_learning,
    build_entry_candle_context,
    build_setup_learning_tags,
    indicator_from_reason,
    learning_position_size_adjustment,
)
from samosbor.domain import Candle, Instrument, InstrumentType, Signal, SignalDirection, TradeRecord


def test_entry_candle_context_marks_short_after_impulse_for_learning():
    start = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    candles = []
    close = 100.0
    for index in range(20):
        close += 0.05
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * index),
                open=close - 0.03,
                high=close + 0.08,
                low=close - 0.08,
                close=close,
                volume=1000,
            )
        )
    candles[-1] = Candle(
        timestamp=candles[-1].timestamp,
        open=101.1,
        high=101.9,
        low=101.0,
        close=101.8,
        volume=1000,
    )
    candles.append(
        Candle(
            timestamp=start + timedelta(minutes=15 * 20),
            open=101.6,
            high=102.0,
            low=100.7,
            close=101.0,
            volume=2000,
        )
    )

    context = build_entry_candle_context(candles, "short")

    assert context["available"] is True
    assert context["needs_confirmation_after_impulse"] is True
    assert context["range_ratio"] > 1.5


def test_signal_learning_blocks_low_quality_entry():
    instrument = Instrument(symbol="PLZL", instrument_type=InstrumentType.STOCK, lot_size=1)
    feedback = {"pending": [], "resolved": []}
    start = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    for index in range(30):
        feedback["resolved"].append(
            _feedback_item(
                symbol="SBER",
                direction="long",
                created_at=start + timedelta(minutes=15 * index),
                signal_strength=0.85,
                net_pnl=3.0,
            )
        )
    for index in range(30):
        feedback["resolved"].append(
            _feedback_item(
                symbol="PLZL",
                direction="short",
                created_at=start + timedelta(minutes=15 * (index + 30)),
                signal_strength=0.25,
                net_pnl=-8.0,
            )
        )

    signal = Signal(
        instrument=instrument,
        direction=SignalDirection.SHORT,
        strength=0.25,
        entry_price=100.0,
        stop_price=101.0,
        take_profit=97.5,
        reason="test",
        context_score=-0.2,
        metadata={
            "entry_candle": {
                "range_pct": 0.012,
                "body_pct": 0.007,
                "ret1": -0.006,
                "ret4": 0.003,
                "range_ratio": 2.0,
                "needs_confirmation_after_impulse": True,
            }
        },
    )

    result = assess_signal_learning(
        signal,
        feedback,
        timestamp=start + timedelta(hours=20),
        quantity_lots=10,
        timezone_name="Europe/Moscow",
        slippage_bps=4.0,
        commission_bps=4.0,
        min_samples=40,
    )

    assert result["available"] is True
    assert result["blocks_entry"] is True
    assert result["action"] == "block_entry"
    assert result["probability_profit"] < 0.4
    assert LOW_QUALITY_TAG in result["learning_tags"]
    assert COMMISSION_EDGE_TAG in result["learning_tags"]
    assert CONFIRMATION_AFTER_IMPULSE_TAG in result["learning_tags"]


def test_signal_learning_normalizes_regression_target_per_lot():
    instrument = Instrument(symbol="PLZL", instrument_type=InstrumentType.STOCK, lot_size=1)
    feedback = {"pending": [], "resolved": []}
    start = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    for index in range(40):
        feedback["resolved"].append(
            _feedback_item(
                symbol="SBER",
                direction="long",
                created_at=start + timedelta(minutes=15 * index),
                signal_strength=0.85,
                quantity_lots=10,
                net_pnl=30.0,
            )
        )
    for index in range(40):
        feedback["resolved"].append(
            _feedback_item(
                symbol="PLZL",
                direction="short",
                created_at=start + timedelta(minutes=15 * (index + 40)),
                signal_strength=0.25,
                quantity_lots=10,
                net_pnl=-80.0,
            )
        )

    signal = Signal(
        instrument=instrument,
        direction=SignalDirection.SHORT,
        strength=0.25,
        entry_price=100.0,
        stop_price=101.0,
        take_profit=97.5,
        reason="test",
        context_score=-0.2,
    )

    result = assess_signal_learning(
        signal,
        feedback,
        timestamp=start + timedelta(hours=24),
        quantity_lots=10,
        timezone_name="Europe/Moscow",
        slippage_bps=4.0,
        commission_bps=4.0,
        min_samples=40,
    )

    assert result["available"] is True
    assert result["target_normalization"] == "net_pnl_per_lot"
    assert -15.0 < result["expected_pnl_per_lot_rub"] < 0.0
    assert -150.0 < result["expected_pnl_position_rub"] < 0.0


def test_learning_position_size_adjustment_reduces_ml_block_to_quarter_size():
    adjustment = learning_position_size_adjustment(
        {
            "available": True,
            "blocks_entry": True,
            "probability_profit": 0.16,
            "expected_pnl_position_rub": -50.0,
            "required_net_edge_rub": 20.0,
        },
        100,
    )

    assert adjustment["active"] is True
    assert adjustment["adjusted_quantity_lots"] == 25
    assert adjustment["requested_scale"] == ML_NEGATIVE_EDGE_POSITION_SCALE
    assert adjustment["reason"] == "reduced by ML negative edge"


def test_setup_learning_tags_detect_short_exhaustion_and_late_reentry():
    instrument = Instrument(symbol="YDEX", instrument_type=InstrumentType.STOCK, lot_size=1)
    timestamp = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    signal = Signal(
        instrument=instrument,
        direction=SignalDirection.SHORT,
        strength=0.9,
        entry_price=100.0,
        stop_price=101.0,
        take_profit=97.5,
        reason="ema-down adx=49.7 rsi=27.7 macd_hist=-4.5",
        metadata={
            "entry_candle": {
                "ret1": -0.008,
                "range_ratio": 1.66,
                "needs_confirmation_after_impulse": False,
            }
        },
    )
    prior_trade = TradeRecord(
        symbol="YDEX",
        direction=SignalDirection.SHORT,
        quantity_lots=1,
        entry_time=timestamp - timedelta(hours=2),
        exit_time=timestamp - timedelta(minutes=15),
        entry_price=102.0,
        exit_price=98.0,
        gross_pnl=4.0,
        net_pnl=3.5,
        reason="take-profit",
    )

    tags = build_setup_learning_tags(
        signal,
        [prior_trade],
        timestamp=timestamp,
        timezone_name="Europe/Moscow",
    )

    assert indicator_from_reason(signal.reason, "rsi") == 27.7
    assert SHORT_AFTER_EXHAUSTION_TAG in tags
    assert LATE_REENTRY_TAG in tags


def _feedback_item(
    *,
    symbol: str,
    direction: str,
    created_at: datetime,
    signal_strength: float,
    net_pnl: float,
    quantity_lots: int = 1,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "direction": direction,
        "created_at": created_at.isoformat(),
        "entry_price": 100.0,
        "stop_price": 99.0 if direction == "long" else 101.0,
        "take_profit": 102.5 if direction == "long" else 97.5,
        "signal_strength": signal_strength,
        "horizon_bars": 48,
        "quantity_lots": quantity_lots,
        "lot_size": 1,
        "instrument_type": "stock",
        "tick_size": 0.01,
        "currency": "rub",
        "slippage_bps": 4.0,
        "commission_bps": 4.0,
        "resolved_at": (created_at + timedelta(hours=1)).isoformat(),
        "exit_price": 102.5 if net_pnl > 0 else 101.0,
        "net_pnl": net_pnl,
        "gross_pnl": net_pnl,
        "outcome_reason": "take-profit" if net_pnl > 0 else "stop-loss",
    }
