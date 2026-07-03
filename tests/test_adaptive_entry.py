from __future__ import annotations

from datetime import datetime, timedelta, timezone

from samosbor.autonomy.adaptive_entry import build_adaptive_entry_context
from samosbor.config import StrategySection
from samosbor.domain import Candle, Instrument, InstrumentType, Signal, SignalDirection


def _candles() -> list[Candle]:
    start = datetime(2026, 7, 3, 6, 0, tzinfo=timezone.utc)
    return [
        Candle(
            timestamp=start + timedelta(minutes=15 * index),
            open=100.0 - index * 0.2,
            high=100.5 - index * 0.2,
            low=99.5 - index * 0.2,
            close=100.0 - index * 0.2,
            volume=1_000_000,
        )
        for index in range(20)
    ]


def _signal(
    *,
    context_score: float,
    entry_candle: dict[str, object],
    confirmation: dict[str, object],
) -> Signal:
    return Signal(
        instrument=Instrument("TEST", InstrumentType.STOCK, lot_size=1),
        direction=SignalDirection.SHORT,
        strength=0.8,
        entry_price=96.2,
        stop_price=98.0,
        take_profit=92.0,
        reason="test",
        context_score=context_score,
        metadata={
            "entry_candle": {"available": True, **entry_candle},
            "entry_confirmation": confirmation,
        },
    )


def test_b_setup_enters_now_when_5m_confirms():
    context = build_adaptive_entry_context(
        _signal(
            context_score=-0.05,
            entry_candle={
                "direction_confirmed_by_close": True,
                "reversal_against_direction": False,
                "close_position": 0.5,
                "ret1": -0.002,
                "range_ratio": 2.0,
            },
            confirmation={
                "against_direction": False,
                "confirmation_ok": True,
                "adverse_bars": 0,
                "ret_window": -0.003,
            },
        ),
        _candles(),
        StrategySection(adaptive_entry_enabled=True),
        prior_events=[],
        timeframe="15min",
    )

    assert context["grade"] == "B"
    assert context["action"] == "enter-now-5m-confirmed"
    assert context["block_reason"] == ""
    assert context["size_factor"] == 1.0


def test_c_setup_enters_reduced_when_5m_and_market_context_confirm():
    context = build_adaptive_entry_context(
        _signal(
            context_score=-0.2,
            entry_candle={
                "direction_confirmed_by_close": False,
                "reversal_against_direction": True,
                "close_position": 0.6,
                "ret1": 0.002,
                "range_ratio": 1.6,
            },
            confirmation={
                "against_direction": False,
                "confirmation_ok": True,
                "adverse_bars": 0,
                "ret_window": -0.004,
            },
        ),
        _candles(),
        StrategySection(
            adaptive_entry_enabled=True,
            market_context_block_threshold=0.15,
        ),
        prior_events=[],
        timeframe="15min",
    )

    assert context["grade"] == "C"
    assert context["action"] == "enter-reduced-5m-market-confirmed"
    assert context["block_reason"] == ""
    assert context["size_factor"] == 0.5


def test_c_setup_still_waits_without_strong_market_context():
    context = build_adaptive_entry_context(
        _signal(
            context_score=-0.05,
            entry_candle={
                "direction_confirmed_by_close": False,
                "reversal_against_direction": True,
                "close_position": 0.6,
                "ret1": 0.002,
                "range_ratio": 1.6,
            },
            confirmation={
                "against_direction": False,
                "confirmation_ok": True,
                "adverse_bars": 0,
                "ret_window": -0.004,
            },
        ),
        _candles(),
        StrategySection(
            adaptive_entry_enabled=True,
            market_context_block_threshold=0.15,
        ),
        prior_events=[],
        timeframe="15min",
    )

    assert context["grade"] == "C"
    assert context["action"] == "wait"
    assert str(context["block_reason"]).startswith("entry waits for adaptive confirmation")
