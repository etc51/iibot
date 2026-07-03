from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from samosbor.autonomy.daily_review import _simulate_plan, build_daily_review_payload, daily_review_path, write_daily_review
from samosbor.config import (
    AppConfig,
    AppSection,
    BacktestSection,
    DataSection,
    ExecutionSection,
    ReportingSection,
    ResearchSection,
    RiskSection,
    StrategySection,
    TBankSection,
)
from samosbor.domain import Candle, Instrument, InstrumentType, PortfolioState, SignalDirection, TradeMode, TradeRecord


def _config(root: Path, instrument: Instrument) -> AppConfig:
    return AppConfig(
        root_dir=root,
        app=AppSection(timezone="Europe/Moscow"),
        tbank=TBankSection(),
        data=DataSection(
            source="csv",
            timeframe="15min",
            history_days=5,
            instruments=[instrument],
        ),
        strategy=StrategySection(
            style="sma_breakout",
            fast_window=2,
            slow_window=3,
            atr_window=2,
            volume_window=2,
            breakout_window=2,
            require_breakout=False,
            atr_stop_multiple=1.0,
            reward_to_risk=2.0,
            min_signal_strength=0.0,
            min_trend_strength=0.0,
            min_liquidity_rub=1.0,
            allowed_entry_weekdays=[],
            alternative_plan_enabled=True,
            alternative_plan_entry_offset_bars=2,
            alternative_plan_atr_stop_multiple=1.0,
            alternative_plan_reward_to_risk=2.0,
        ),
        risk=RiskSection(),
        execution=ExecutionSection(
            mode=TradeMode.LOCAL_PAPER,
            slippage_bps=0.0,
            commission_bps=0.0,
            state_path="state/paper_state.json",
            allow_live_trading=False,
        ),
        backtest=BacktestSection(initial_cash=100_000, warmup_bars=3),
        reporting=ReportingSection(output_dir="runs"),
        research=ResearchSection(
            atr_stop_multipliers=[1.0, 1.5],
            reward_to_risk_values=[1.5, 2.0],
        ),
    )


def _candles() -> list[Candle]:
    start = datetime(2026, 7, 3, 6, 0, tzinfo=timezone.utc)
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 108.0, 110.0]
    rows = []
    for index, price in enumerate(prices):
        rows.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * index),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1_000_000,
            )
        )
    return rows


def test_daily_review_finds_missed_positive_candidates(tmp_path: Path):
    instrument = Instrument("TEST", InstrumentType.STOCK, lot_size=1)
    config = _config(tmp_path, instrument)

    payload = build_daily_review_payload(
        config,
        PortfolioState(cash=100_000),
        [],
        candles_by_symbol={"TEST": _candles()},
        instruments_by_symbol={"TEST": instrument},
        report_date=date(2026, 7, 3),
        feedback_payload={"resolved": [], "pending": []},
    )

    assert payload["signal_scan"]["candidate_signals"] > 0
    assert payload["signal_scan"]["missed_positive_opportunities"] > 0
    assert payload["training_examples"]
    first = payload["missed_opportunities"][0]
    assert first["symbol"] == "TEST"
    assert first["best_plan"]["net_pnl_per_lot_rub"] > 0
    tracked = payload["grid_summary"]["tracked_alternative_plan"]
    assert tracked["enabled"] is True
    assert tracked["mode"] == "observe-only"
    assert tracked["entry_offset_bars"] == 2
    assert tracked["stop_multiple"] == 1.0
    assert tracked["reward_to_risk"] == 2.0


def test_daily_review_reviews_actual_trade_and_writes_files(tmp_path: Path):
    instrument = Instrument("TEST", InstrumentType.STOCK, lot_size=1)
    config = _config(tmp_path, instrument)
    candles = _candles()
    trade = TradeRecord(
        symbol="TEST",
        direction=SignalDirection.LONG,
        quantity_lots=1,
        entry_time=candles[3].timestamp,
        exit_time=candles[4].timestamp,
        entry_price=103.0,
        exit_price=102.0,
        gross_pnl=-1.0,
        net_pnl=-1.0,
        reason="stop-loss",
        signal_strength=0.5,
        initial_stop_price=102.0,
        initial_take_profit=105.0,
    )

    payload = build_daily_review_payload(
        config,
        PortfolioState(cash=99_999, realized_pnl=-1.0),
        [trade],
        candles_by_symbol={"TEST": candles},
        instruments_by_symbol={"TEST": instrument},
        report_date=date(2026, 7, 3),
        feedback_payload={"resolved": [], "pending": []},
    )
    output_dir = tmp_path / "runs" / "daily-review" / "test"
    write_daily_review(output_dir, payload)

    assert payload["actual_day"]["opened_trades"] == 1
    assert payload["actual_trade_reviews"][0]["best_plan"] is not None
    assert (output_dir / "daily_review.json").exists()
    assert (output_dir / "daily_review.md").exists()
    assert (output_dir / "training_examples.csv").exists()
    assert str(daily_review_path(Path("state/paper_state.json"))).replace("\\", "/") == (
        "state/paper_state_daily_review.json"
    )


def test_daily_review_simulates_runner_after_take_profit_touch():
    start = datetime(2026, 7, 3, 6, 0, tzinfo=timezone.utc)
    candles = [
        Candle(start, open=100.0, high=100.5, low=99.5, close=100.0, volume=1_000_000),
        Candle(start + timedelta(minutes=15), open=100.0, high=106.0, low=100.2, close=105.0, volume=1_000_000),
        Candle(start + timedelta(minutes=30), open=105.0, high=105.2, low=102.0, close=103.0, volume=1_000_000),
    ]

    plan = _simulate_plan(
        direction=SignalDirection.LONG,
        candles=candles,
        entry_index=0,
        entry_offset_bars=0,
        entry_price=100.0,
        base_atr=2.0,
        stop_multiple=1.0,
        reward_to_risk=2.0,
        lot_size=1,
        slippage_bps=0.0,
        commission_bps=0.0,
        max_holding_bars=3,
        end_at=start + timedelta(days=1),
        timezone_info=ZoneInfo("Europe/Moscow"),
        runner_enabled=True,
        runner_breakeven_buffer_bps=10.0,
        runner_trailing_atr_multiple=1.3,
        runner_profit_lock_ratio=0.35,
        runner_atr_window=14,
    )

    assert plan["runner_activated"] is True
    assert plan["exit_reason"] == "profit-protect-stop"
    assert plan["exit_price"] > 100.0
    assert plan["final_stop_price"] > 100.0
