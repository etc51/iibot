from __future__ import annotations

from datetime import datetime, timedelta, timezone

import samosbor.autonomy.short_ev_engine as short_ev_module
from samosbor.autonomy.short_ev_engine import (
    ALLOWED_SHORT_SETUPS,
    CostEstimate,
    SetupStats,
    SetupVerdict,
    canonical_setup_id,
    classify_short_setup,
    estimate_short_setup_ev,
    short_net_breakeven_stop,
    short_trailing_stop,
)
from samosbor.config import load_config
from samosbor.domain import Candle, Instrument, InstrumentType, Position, Signal, SignalDirection


class _Regime:
    def __init__(self, regime: str = "clean_downtrend"):
        self.regime = regime
        self.confidence = 1.0
        self.features = {
            "breadth_down": 0.85,
            "symbols": 35,
            "universe_ret_15m": -0.006,
        }

    def as_event(self) -> dict[str, object]:
        return {"regime": self.regime, "confidence": self.confidence, **self.features}


def _candles(count: int = 90, *, minutes: int = 15, start: float = 130.0) -> list[Candle]:
    base = datetime(2026, 7, 7, 7, 0, tzinfo=timezone.utc)
    price = start
    candles: list[Candle] = []
    for index in range(count):
        change = 0.16 if index % 2 == 0 else -0.31
        if index == count - 1:
            change = -0.62
        open_price = price
        close = price + change
        candles.append(
            Candle(
                base + timedelta(minutes=minutes * index),
                open_price,
                max(open_price, close) + 0.05,
                min(open_price, close) - 0.04,
                close,
                100_000,
            )
        )
        price = close
    return candles


def _signal() -> Signal:
    instrument = Instrument("SBER", InstrumentType.STOCK, lot_size=10)
    return Signal(
        instrument=instrument,
        direction=SignalDirection.SHORT,
        strength=0.70,
        entry_price=100.0,
        stop_price=102.0,
        take_profit=95.0,
        reason="test",
        metadata={"microstructure": {"available": True, "spread_bps": 5.0}},
    )


def _stats(
    *,
    setup_id: str = "normal_15m_trend_short",
    sample_count: int = 30,
    win_rate: float = 0.60,
    avg_win: float = 120.0,
    avg_loss: float = -50.0,
    values_are_net: bool = True,
) -> SetupStats:
    return SetupStats(
        setup_id=setup_id,
        sample_count=sample_count,
        wins=int(sample_count * win_rate),
        losses=sample_count - int(sample_count * win_rate),
        win_rate=win_rate,
        avg_win_net_rub=avg_win,
        avg_loss_net_rub=avg_loss,
        avg_mfe_pct=0.006,
        avg_mae_pct=0.003,
        p_hit_breakeven_03_04=0.55,
        p_stop=0.40,
        p_runner=0.10,
        values_are_net=values_are_net,
        source="empirical",
    )


def _setup(setup_id: str = "normal_15m_trend_short", passed: bool = True) -> SetupVerdict:
    return SetupVerdict(setup_id, passed, "passed" if passed else "failed", [], 0.5, "test", {})


def _costs(total: float = 12.0) -> CostEstimate:
    return CostEstimate(4.0, 4.0, 2.0, total - 10.0, total, 21.0)


def _early_features(**overrides) -> dict[str, float]:
    features = {
        "ema9": 100.5,
        "ema9_slope": -0.2,
        "rsi": 32.0,
        "macd_hist": -0.3,
        "rolling_low_min": 99.0,
        "rolling_low_max": 99.0,
        "close_position": 0.20,
        "ret_window": -0.01,
    }
    features.update(overrides)
    return features


def _patch_early_inputs(monkeypatch, *, normal_failures=None, features=None):
    normal_failures = list(normal_failures or [])
    features = dict(features or _early_features())

    def fake_normal_setup(*args, **kwargs):
        del args, kwargs
        return SetupVerdict(
            "normal_15m_trend_short",
            not normal_failures,
            "passed" if not normal_failures else "; ".join(normal_failures),
            normal_failures,
            0.5,
            "test",
            {"ema_fast": 99.0, "ema_slow": 100.0, "turnover_rub": 10_000_000.0},
        )

    monkeypatch.setattr(short_ev_module, "_normal_setup", fake_normal_setup)
    monkeypatch.setattr(short_ev_module, "_trigger_features", lambda candles, trigger: features)


def _early_verdict(monkeypatch, *, normal_failures=None, features=None, execution_guard=None, golden_3tf=None, regime="market_selloff_impulse"):
    _patch_early_inputs(monkeypatch, normal_failures=normal_failures, features=features)
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    return short_ev_module._early_setup(
        _candles(3),
        _candles(3, minutes=5, start=100.0),
        execution_guard if execution_guard is not None else {"available": True, "passed": True},
        config,
        set(config.short_ev_engine.allowed_setups),
        golden_3tf=golden_3tf or {},
        market_regime=_Regime(regime),
    )


def test_allowed_setup_registry_exact():
    assert ALLOWED_SHORT_SETUPS == (
        "normal_15m_trend_short",
        "golden_15m_breakout_short",
        "early_5m_acceleration_short",
        "failed_rebound_short",
        "market_selloff_short",
    )
    assert canonical_setup_id("golden_15m_short_breakout") == "golden_15m_breakout_short"


def test_ev_gate_positive_allows_real():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    result = estimate_short_setup_ev(
        _signal(),
        setup_verdict=_setup(),
        setup_stats=_stats(),
        costs=_costs(),
        quantity_lots=1,
        config=config,
    )

    assert result.decision == "real_allowed"
    assert result.ev_net_rub > 10.0
    assert result.ev_per_risk >= 0.05


def test_ev_gate_negative_blocks():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    result = estimate_short_setup_ev(
        _signal(),
        setup_verdict=_setup(),
        setup_stats=_stats(win_rate=0.20, avg_win=50.0, avg_loss=-150.0),
        costs=_costs(),
        quantity_lots=1,
        config=config,
    )

    assert result.decision == "blocked_negative_ev"
    assert result.ev_net_rub < 0.0


def test_ev_gate_insufficient_sample_shadow_or_probe():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    result = estimate_short_setup_ev(
        _signal(),
        setup_verdict=_setup("market_selloff_short"),
        setup_stats=_stats(setup_id="market_selloff_short", sample_count=0),
        costs=_costs(),
        quantity_lots=1,
        config=config,
    )

    assert result.decision in {"probe_allowed", "shadow_only"}
    assert result.source == "shadow_only_insufficient_data"


def test_ev_gate_subtracts_costs():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    result = estimate_short_setup_ev(
        _signal(),
        setup_verdict=_setup(),
        setup_stats=_stats(sample_count=30, win_rate=1.0, avg_win=20.0, avg_loss=0.0, values_are_net=False),
        costs=_costs(total=18.0),
        quantity_lots=1,
        config=config,
    )

    assert result.ev_net_rub == 2.0
    assert result.decision == "blocked_negative_ev"


def test_ml_positive_alone_cannot_bypass_ev_gate():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    bad_signal = Signal(
        instrument=_signal().instrument,
        direction=SignalDirection.SHORT,
        strength=0.05,
        entry_price=100.0,
        stop_price=104.0,
        take_profit=99.0,
        reason="ml-only",
        metadata={},
    )
    result = estimate_short_setup_ev(
        bad_signal,
        setup_verdict=_setup("market_selloff_short"),
        setup_stats=_stats(setup_id="market_selloff_short", sample_count=0),
        costs=_costs(total=20.0),
        quantity_lots=1,
        config=config,
        ml_expected_net_edge_rub=10_000.0,
    )

    assert result.decision != "real_allowed"
    assert result.source == "shadow_only_insufficient_data"


def test_unknown_setup_shadow_only():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    result = estimate_short_setup_ev(
        _signal(),
        setup_verdict=_setup("unknown_setup"),
        setup_stats=_stats(setup_id="unknown_setup"),
        costs=_costs(),
        quantity_lots=1,
        config=config,
    )

    assert result.decision == "shadow_only"
    assert result.reason == "unknown_setup_not_allowed_real"


def test_normal_15m_trend_short_candidate():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    candles = _candles()
    verdict = classify_short_setup(
        _signal(),
        candles,
        [],
        market_regime=_Regime(),
        config=config,
        real_trade_source="setup_registry",
        golden_3tf={},
        execution_guard={"available": True, "passed": True},
    )

    assert verdict.setup_id in {"normal_15m_trend_short", "golden_15m_breakout_short"}
    assert verdict.passed is True


def test_early_5m_does_not_require_rolling_low(monkeypatch):
    verdict = _early_verdict(monkeypatch, features=_early_features(rolling_low_max=99.0))

    assert verdict.passed is True
    assert "trigger_close_not_below_rolling_low" not in verdict.failed_conditions
    early = verdict.indicators["early_5m"]
    assert early["rolling_low_broken"] is False
    assert early["rolling_low_required"] is False
    assert early["setup_quality"] == "early_acceleration_no_breakout"
    assert "trigger_close_not_below_rolling_low_quality_penalty" in early["quality_flags"]
    assert verdict.default_size_multiplier == 0.18


def test_early_5m_rolling_low_break_adds_quality_bonus(monkeypatch):
    verdict = _early_verdict(monkeypatch, features=_early_features(rolling_low_max=101.0))

    assert verdict.passed is True
    early = verdict.indicators["early_5m"]
    assert early["rolling_low_broken"] is True
    assert early["setup_quality"] == "early_breakdown"
    assert early["quality_flags"] == []
    assert early["size_multiplier_reason"] == "rolling_low_break_quality_bonus"
    assert verdict.default_size_multiplier == 0.25


def test_early_5m_without_rolling_low_can_pass_ev_gate(monkeypatch):
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    verdict = _early_verdict(monkeypatch, features=_early_features(rolling_low_max=99.0))

    result = estimate_short_setup_ev(
        _signal(),
        setup_verdict=verdict,
        setup_stats=_stats(setup_id="early_5m_acceleration_short", sample_count=40, win_rate=0.70),
        costs=_costs(),
        quantity_lots=1,
        config=config,
    )

    assert verdict.passed is True
    assert result.decision == "real_allowed"


def test_golden_breakout_still_requires_rolling_low(monkeypatch):
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    monkeypatch.setattr(
        short_ev_module,
        "_baseline_features",
        lambda candles, strategy: {
            "ema_fast": 99.0,
            "ema_slow": 100.0,
            "adx": 30.0,
            "macd_hist": -0.1,
            "rsi": 35.0,
        },
    )
    monkeypatch.setattr(short_ev_module, "average_turnover", lambda candles, window: 10_000_000.0)
    monkeypatch.setattr(short_ev_module, "rolling_low", lambda candles, window: 99.0)

    verdict = short_ev_module._normal_setup(
        _candles(3, start=101.0),
        config,
        set(config.short_ev_engine.allowed_setups),
        require_breakout=True,
    )

    assert verdict.passed is False
    assert "close_not_below_rolling_low20" in verdict.failed_conditions


def test_normal_15m_still_does_not_require_rolling_low(monkeypatch):
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    monkeypatch.setattr(
        short_ev_module,
        "_baseline_features",
        lambda candles, strategy: {
            "ema_fast": 99.0,
            "ema_slow": 100.0,
            "adx": 30.0,
            "macd_hist": -0.1,
            "rsi": 35.0,
        },
    )
    monkeypatch.setattr(short_ev_module, "average_turnover", lambda candles, window: 10_000_000.0)
    monkeypatch.setattr(short_ev_module, "rolling_low", lambda candles, window: 99.0)

    verdict = short_ev_module._normal_setup(
        _candles(3, start=98.0),
        config,
        set(config.short_ev_engine.allowed_setups),
        require_breakout=False,
    )

    assert verdict.passed is True
    assert "close_not_below_rolling_low20" not in verdict.failed_conditions


def test_early_5m_context_override_allows_strict_5m_acceleration(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        normal_failures=["ema20_not_below_ema50"],
        features=_early_features(rolling_low_max=99.0),
    )

    assert verdict.passed is True
    assert "context_ema20_not_below_ema50" not in verdict.failed_conditions
    early = verdict.indicators["early_5m"]
    assert early["context_override_used"] is True
    assert early["context_override_reason"] == "strict_5m_acceleration_down"
    assert early["original_15m_context_failed"] == ["context_ema20_not_below_ema50"]
    assert early["size_multiplier_reason"] == "context_override_smaller_size"
    assert verdict.default_size_multiplier == 0.15


def test_early_5m_context_override_requires_5m_ema9_slope_negative(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        normal_failures=["ema20_not_below_ema50"],
        features=_early_features(ema9_slope=0.1),
    )

    assert verdict.passed is False
    assert "context_ema20_not_below_ema50" in verdict.failed_conditions
    assert "trigger_ema9_slope_not_negative" in verdict.failed_conditions


def test_early_5m_context_override_requires_5m_macd_hist_negative(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        normal_failures=["ema20_not_below_ema50"],
        features=_early_features(macd_hist=0.1),
    )

    assert verdict.passed is False
    assert "context_ema20_not_below_ema50" in verdict.failed_conditions
    assert "trigger_macd_hist_not_negative" in verdict.failed_conditions


def test_early_5m_context_override_requires_negative_ret_window(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        normal_failures=["ema20_not_below_ema50"],
        features=_early_features(ret_window=0.001),
    )

    assert verdict.passed is False
    assert "context_ema20_not_below_ema50" in verdict.failed_conditions


def test_early_5m_context_override_requires_order_book(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        normal_failures=["ema20_not_below_ema50"],
        features=_early_features(),
        golden_3tf={"failed_conditions": ["order_book_imbalance_too_low"]},
    )

    assert verdict.passed is False
    assert "order_book_imbalance_too_low" in verdict.failed_conditions
    assert verdict.indicators["early_5m"]["order_book_strict_passed"] is False


def test_early_5m_context_override_requires_1m_guard(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        normal_failures=["ema20_not_below_ema50"],
        features=_early_features(),
        execution_guard={"available": True, "passed": False},
    )

    assert verdict.passed is False
    assert "execution_1m_guard_blocked" in verdict.failed_conditions
    assert verdict.indicators["early_5m"]["blocked_by_1m_after_context_override"] is True


def test_early_5m_context_override_still_blocks_range_chop(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        normal_failures=["ema20_not_below_ema50"],
        features=_early_features(),
        regime="range_chop",
    )

    assert verdict.passed is False
    assert "context_ema20_not_below_ema50" in verdict.failed_conditions
    assert verdict.indicators["early_5m"]["context_override_used"] is False


def test_1m_guard_remains_required_for_early_5m(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        features=_early_features(),
        execution_guard={"available": False, "passed": False},
    )

    assert verdict.passed is False
    assert "execution_1m_unavailable" in verdict.failed_conditions


def test_1m_guard_still_blocks_rebound(monkeypatch):
    verdict = _early_verdict(
        monkeypatch,
        features=_early_features(),
        execution_guard={
            "available": True,
            "passed": False,
            "reason": "execution_1m_rebound_bars",
        },
    )

    assert verdict.passed is False
    assert "execution_1m_guard_blocked" in verdict.failed_conditions


def test_ev_gate_required_for_early_5m(monkeypatch):
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    verdict = _early_verdict(monkeypatch, features=_early_features())

    result = estimate_short_setup_ev(
        _signal(),
        setup_verdict=verdict,
        setup_stats=_stats(setup_id="early_5m_acceleration_short", sample_count=40, win_rate=0.10),
        costs=_costs(total=30.0),
        quantity_lots=1,
        config=config,
    )

    assert verdict.passed is True
    assert result.decision == "blocked_negative_ev"
    assert result.reason == "setup_ev_below_min_after_costs"


def test_no_long_real_trade():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    long_signal = Signal(_signal().instrument, SignalDirection.LONG, 0.7, 100.0, 99.0, 102.0, "long")
    verdict = classify_short_setup(
        long_signal,
        _candles(),
        [],
        market_regime=_Regime(),
        config=config,
        real_trade_source="setup_registry",
        golden_3tf={},
        execution_guard={},
    )

    assert verdict.passed is False
    assert verdict.reason == "not_short_signal"


def test_range_chop_no_real_trade():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")
    verdict = classify_short_setup(
        _signal(),
        _candles(),
        [],
        market_regime=_Regime("range_chop"),
        config=config,
        real_trade_source="setup_registry",
        golden_3tf={},
        execution_guard={},
    )

    assert verdict.passed is False
    assert verdict.reason == "range_chop_not_tradable"


def test_only_15m_5m_1m_used():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")

    assert config.short_ev_engine.timeframes.primary == "15min"
    assert config.short_ev_engine.timeframes.trigger == "5min"
    assert config.short_ev_engine.timeframes.execution_guard == "1min"


def test_10m_forbidden():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")

    assert config.short_ev_engine.timeframes.forbidden == ["10min"]


def test_short_net_breakeven_armed_at_03_04_pct():
    position = Position(
        instrument=_signal().instrument,
        direction=SignalDirection.SHORT,
        quantity_lots=1,
        entry_price=100.0,
        entry_commission=4.0,
        margin_requirement=0.0,
        current_price=99.60,
        stop_price=102.0,
        take_profit=95.0,
        opened_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
        mfe_price=99.60,
        mae_price=100.0,
    )
    stop = short_net_breakeven_stop(position, {"estimated_total_cost_rub": 6.0}, buffer_bps=2.0)

    assert stop < position.entry_price
    assert round(stop, 2) == 99.38


def test_short_stop_moves_only_profit_direction():
    position = Position(
        instrument=_signal().instrument,
        direction=SignalDirection.SHORT,
        quantity_lots=1,
        entry_price=100.0,
        entry_commission=4.0,
        margin_requirement=0.0,
        current_price=98.8,
        stop_price=99.0,
        take_profit=95.0,
        opened_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    candles = _candles(20, minutes=5, start=99.0)
    stop, _ = short_trailing_stop(position, candles, atr_window=14, atr_multiple=1.3)

    assert stop is None or stop < position.stop_price
