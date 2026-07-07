from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from samosbor.autonomy.golden_baseline import (
    is_early_5m_starter_short_signal,
    is_golden_15m_short_breakout_signal,
    passes_1m_execution_guard,
)
from samosbor.autonomy.trade_review import build_trade_review_payload
from samosbor.config import load_config
from samosbor.domain import Candle, Instrument, InstrumentType, PortfolioState, Signal, SignalDirection, TradeRecord
from samosbor.orchestrator import TradingOrchestrator


class _Regime:
    regime = "weak_down_choppy"
    confidence = 1.0
    features = {"breadth_down": 0.85, "symbols": 35}

    def as_event(self) -> dict[str, object]:
        return {"regime": self.regime, "confidence": self.confidence, **self.features}


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "golden.toml"
    path.parent.mkdir()
    path.write_text(
        "\n".join(
            [
                "[app]",
                'timezone = "Europe/Moscow"',
                "",
                "[data]",
                'timeframe = "15min"',
                "",
                "[[data.instruments]]",
                'symbol = "SBER"',
                'instrument_type = "stock"',
                "",
                "[strategy]",
                'style = "ema_adx_macd"',
                "fast_window = 20",
                "slow_window = 50",
                "require_breakout = true",
                "min_liquidity_rub = 6000000",
                "atr_stop_multiple = 1.5",
                "reward_to_risk = 2.5",
                "min_trend_strength = 0.002",
                "require_order_book = true",
                "max_entry_spread_bps = 12",
                "min_entry_liquidity_cover = 2",
                "min_entry_book_imbalance = -0.35",
                'entry_confirmation_timeframe = "5min"',
                "entry_confirmation_min_bars = 3",
                "",
                "[execution]",
                'mode = "local-paper"',
                "allow_live_trading = false",
                "",
                "[golden_baseline]",
                "enabled = true",
                'forbidden_timeframes = ["10min"]',
                "",
                "[golden_baseline.timeframes]",
                'primary = "15min"',
                'early_trigger = "5min"',
                'execution_guard = "1min"',
                "",
                "[golden_baseline.early_5m]",
                "real_trading_enabled = true",
                "starter_size_multiplier = 0.25",
                "",
                "[golden_baseline.execution_1m]",
                "enabled = true",
                "required_for_early_5m = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _bearish_candles(
    *,
    count: int,
    minutes: int,
    start: float,
    gain: float = 0.18,
    loss: float = -0.32,
    final_loss: float = -0.55,
    volume: float = 100_000,
) -> list[Candle]:
    base = datetime(2026, 7, 7, 7, 0, tzinfo=timezone.utc)
    price = start
    candles: list[Candle] = []
    for index in range(count):
        change = gain if index % 2 == 0 else loss
        if index == count - 1:
            change = final_loss
        open_price = price
        close = price + change
        high = max(open_price, close) + 0.05
        low = min(open_price, close) - 0.04
        candles.append(
            Candle(
                base + timedelta(minutes=minutes * index),
                open_price,
                high,
                low,
                close,
                volume,
            )
        )
        price = close
    return candles


def _one_minute_guard_candles(candles_5m: list[Candle], *, rebound: bool = False) -> list[Candle]:
    base = candles_5m[-1].timestamp + timedelta(minutes=1)
    price = candles_5m[-1].close
    candles: list[Candle] = []
    for index in range(3):
        open_price = price
        close = price + (0.18 if rebound else -0.02)
        candles.append(
            Candle(
                base + timedelta(minutes=index),
                open_price,
                max(open_price, close) + 0.01,
                min(open_price, close) - 0.01,
                close,
                10_000,
            )
        )
        price = close
    return candles


def _book() -> dict[str, object]:
    return {"available": True, "spread_bps": 5.0, "entry_liquidity_cover": 3.0, "side_imbalance": 0.0}


def _confirmation() -> dict[str, object]:
    return {
        "available": True,
        "timeframe": "5min",
        "bars": 3,
        "confirmation_ok": True,
        "against_direction": False,
    }


def test_golden_15m_short_breakout_passes_strict_baseline(tmp_path: Path):
    config = load_config(_write_config(tmp_path))
    candles = _bearish_candles(count=90, minutes=15, start=130.0)
    instrument = Instrument("SBER", InstrumentType.STOCK)
    signal = Signal(
        instrument,
        SignalDirection.SHORT,
        0.72,
        candles[-1].close,
        candles[-1].close + 1.0,
        candles[-1].close - 2.5,
        "ema-down",
    )

    verdict = is_golden_15m_short_breakout_signal(
        signal,
        candles,
        _confirmation(),
        _book(),
        _Regime(),
        config,
        source_strategy_direction="short",
    )

    assert verdict["passed"] is True
    assert verdict["entry_mode"] == "golden_15m_short_breakout"
    assert verdict["timeframes"] == {"primary": "15min", "early_trigger": "5min", "execution_guard": "1min"}


def test_early_5m_starter_requires_1m_guard_and_uses_reduced_size(tmp_path: Path):
    config = load_config(_write_config(tmp_path))
    candles_15m = _bearish_candles(count=90, minutes=15, start=130.0)
    candles_5m = _bearish_candles(
        count=60,
        minutes=5,
        start=126.0,
        gain=0.08,
        loss=-0.14,
        final_loss=-0.25,
    )
    guard = passes_1m_execution_guard(_one_minute_guard_candles(candles_5m), candles_5m, _book(), config)

    verdict = is_early_5m_starter_short_signal(
        candles_15m,
        candles_5m,
        _book(),
        guard,
        _Regime(),
        config,
        source_strategy_direction="none",
    )

    assert guard["passed"] is True
    assert verdict["passed"] is True
    assert verdict["entry_mode"] == "early_5m_starter_short"
    assert verdict["size_multiplier"] == 0.25


def test_1m_execution_guard_blocks_short_rebound(tmp_path: Path):
    config = load_config(_write_config(tmp_path))
    candles_5m = _bearish_candles(count=60, minutes=5, start=126.0, gain=0.08, loss=-0.14, final_loss=-0.25)

    guard = passes_1m_execution_guard(_one_minute_guard_candles(candles_5m, rebound=True), candles_5m, _book(), config)

    assert guard["passed"] is False
    assert "execution_1m_rebound_bars" in guard["failed_conditions"]


def test_non_strategy_source_is_shadow_only_for_15m_baseline(tmp_path: Path):
    config = load_config(_write_config(tmp_path))
    candles = _bearish_candles(count=90, minutes=15, start=130.0)
    instrument = Instrument("SBER", InstrumentType.STOCK)
    signal = Signal(
        instrument,
        SignalDirection.SHORT,
        0.72,
        candles[-1].close,
        candles[-1].close + 1.0,
        candles[-1].close - 2.5,
        "synthetic",
    )

    verdict = is_golden_15m_short_breakout_signal(
        signal,
        candles,
        _confirmation(),
        _book(),
        _Regime(),
        config,
        source_strategy_direction="none",
    )

    assert verdict["passed"] is False
    assert "source_not_strategy_short" in verdict["failed_conditions"]


def test_focused_config_declares_golden_3tf_runtime_profile():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")

    assert config.golden_baseline.enabled is True
    assert config.data.timeframe == "15min"
    assert config.strategy.require_breakout is True
    assert config.golden_baseline.timeframes == ["15min", "5min", "1min"]
    assert config.golden_baseline.forbidden_timeframes == ["10min"]
    assert config.execution.allow_live_trading is False
    assert config.short_only.real_trade_sources.strategy_short is True
    assert config.short_only.real_trade_sources.early_5m_starter is True
    assert config.short_only.real_trade_sources.synthetic is False


def test_execution_guard_loader_uses_one_day_1m_history(tmp_path: Path):
    config = load_config(_write_config(tmp_path))
    instrument = Instrument("SBER", InstrumentType.STOCK)

    class Provider:
        def __init__(self):
            self.calls: list[tuple[str, int | None]] = []

        def load_history_for_timeframe(self, instruments, timeframe: str, *, history_days: int | None = None):
            self.calls.append((timeframe, history_days))
            return {instrument.symbol: [] for instrument in instruments}

    provider = Provider()
    orchestrator = TradingOrchestrator(config)

    orchestrator._load_golden_execution_guard_history(provider, [instrument], {})

    assert provider.calls == [("1min", 1)]


def test_trade_review_reports_golden_3tf_counts(tmp_path: Path):
    timestamp = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    golden = {
        "enabled": True,
        "passed": False,
        "verdict": "shadow_only",
        "entry_mode": "early_5m_starter_short",
        "failed_conditions": ["execution_1m_guard_blocked"],
        "source_run": "20260707-105708",
        "source_commit": "56be2bda69876f917731f81d913fa32aa9aad8b5",
        "timeframes": {"primary": "15min", "early_trigger": "5min", "execution_guard": "1min"},
    }
    events = [
        {
            "timestamp": timestamp.isoformat(),
            "action": "golden_baseline_config",
            "source_run": golden["source_run"],
            "source_commit": golden["source_commit"],
            "primary_timeframe": "15min",
            "early_trigger_timeframe": "5min",
            "execution_guard_timeframe": "1min",
            "forbidden_timeframes": ["10min"],
            "allow_live_trading": False,
            "execution_mode": "local-paper",
        },
        {
            "timestamp": timestamp.isoformat(),
            "symbol": "SBER",
            "action": "short_only_short_candidate",
            "metadata": {"short_only": {"enabled": True, "golden_3tf": golden}},
        },
        {
            "timestamp": timestamp.isoformat(),
            "symbol": "SBER",
            "action": "golden_baseline_shadow_only",
            "metadata": {"golden_3tf": golden, "short_only": {"enabled": True}},
        },
    ]
    config = load_config(_write_config(tmp_path))
    payload = build_trade_review_payload(
        PortfolioState(cash=100_000),
        [
            TradeRecord(
                symbol="SBER",
                direction=SignalDirection.SHORT,
                quantity_lots=1,
                entry_time=timestamp,
                exit_time=timestamp + timedelta(minutes=15),
                entry_price=100.0,
                exit_price=99.0,
                gross_pnl=1.0,
                net_pnl=1.0,
                reason="take-profit",
                entry_metadata={"golden_3tf": {**golden, "passed": True, "verdict": "passed"}},
            )
        ],
        events,
        strategy=config.strategy,
        risk=config.risk,
        timezone_name="Europe/Moscow",
        generated_at=timestamp,
    )

    review = payload["golden_3tf_review"]
    assert review["enabled"] is True
    assert review["early_5m_starter_candidates"] == 1
    assert review["shadow_only_events"] == 1
    assert review["top_failed_conditions"] == {"execution_1m_guard_blocked": 2}
    assert review["pnl_total"]["net_pnl_rub"] == 1.0
