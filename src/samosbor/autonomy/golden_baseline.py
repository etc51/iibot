from __future__ import annotations

import math
from collections.abc import Sequence

from ..analysis.indicators import average_turnover, rolling_low, rsi
from ..domain import Candle, SignalDirection


def is_golden_15m_short_breakout_signal(
    signal,
    candles_15m: Sequence[Candle],
    confirmation_5m: dict[str, object],
    order_book: dict[str, object],
    market_regime,
    config,
    *,
    execution_guard_1m: dict[str, object] | None = None,
    source_strategy_direction: str | None = None,
) -> dict[str, object]:
    failed: list[str] = []
    strategy = config.strategy
    baseline = config.golden_baseline
    latest = candles_15m[-1] if candles_15m else None
    features = _baseline_features(candles_15m, strategy)
    indicators = dict(features)
    indicators["turnover_rub"] = _round_or_none(average_turnover(list(candles_15m), strategy.volume_window))
    indicators["rolling_low_20"] = _round_or_none(rolling_low(list(candles_15m[:-1]), strategy.breakout_window))

    if not bool(baseline.enabled):
        failed.append("golden_baseline_disabled")
    if str(config.execution.mode.value) != "local-paper":
        failed.append("execution_mode_not_local_paper")
    if bool(config.execution.allow_live_trading):
        failed.append("allow_live_trading_true")
    if _tf(config.data.timeframe) != _tf(baseline.timeframes_config.primary):
        failed.append("primary_timeframe_not_15min")
    if _tf(config.data.timeframe) in _forbidden_timeframes(config):
        failed.append("forbidden_primary_timeframe")
    if str(strategy.style).strip().lower() != "ema_adx_macd":
        failed.append("strategy_style_not_ema_adx_macd")
    if signal is None or signal.direction != SignalDirection.SHORT:
        failed.append("not_short_signal")
    if (source_strategy_direction or _short_only_field(signal, "source_strategy_signal") or "").lower() != "short":
        failed.append("source_not_strategy_short")
    if getattr(market_regime, "regime", "") == "range_chop":
        failed.append("range_chop_block")

    close = float(latest.close) if latest is not None else None
    if latest is None:
        failed.append("missing_15m_candles")
    elif not _finite(close):
        failed.append("invalid_15m_close")

    _require_feature(failed, features, "ema_fast", "ema20_unavailable")
    _require_feature(failed, features, "ema_slow", "ema50_unavailable")
    _require_feature(failed, features, "adx", "adx_unavailable")
    _require_feature(failed, features, "macd_hist", "macd_hist_unavailable")
    _require_feature(failed, features, "rsi", "rsi_unavailable")

    if _finite(features.get("ema_fast")) and _finite(features.get("ema_slow")):
        if not float(features["ema_fast"]) < float(features["ema_slow"]):
            failed.append("ema20_not_below_ema50")
    if _finite(features.get("adx")) and float(features["adx"]) < float(strategy.adx_min):
        failed.append("adx_below_min")
    if _finite(features.get("macd_hist")) and float(features["macd_hist"]) >= 0:
        failed.append("macd_hist_not_negative")
    if _finite(features.get("rsi")):
        value = float(features["rsi"])
        if not float(strategy.rsi_short_min) <= value <= float(strategy.rsi_short_max):
            failed.append("rsi_outside_short_band")
    if close is not None and _finite(features.get("ema_fast")) and close > float(features["ema_fast"]):
        failed.append("close_above_ema20")

    breakout_low = indicators.get("rolling_low_20")
    if bool(strategy.require_breakout):
        if breakout_low is None:
            failed.append("rolling_low20_unavailable")
        elif close is not None and close > float(breakout_low):
            failed.append("close_not_below_rolling_low20")
    else:
        failed.append("require_breakout_disabled")

    turnover = indicators.get("turnover_rub")
    if turnover is None:
        failed.append("turnover_unavailable")
    elif float(turnover) < float(strategy.min_liquidity_rub):
        failed.append("liquidity_below_min")

    failed.extend(_confirmation_failures(confirmation_5m, config))
    failed.extend(_order_book_failures(order_book, config))
    if execution_guard_1m is not None:
        indicators["execution_1m"] = _execution_guard_summary(execution_guard_1m)
        if execution_guard_1m.get("available") is False and bool(config.golden_baseline.execution_1m.required_for_15m):
            failed.append("execution_1m_unavailable")
        elif bool(execution_guard_1m.get("available", False)) and not bool(execution_guard_1m.get("passed", False)):
            failed.append("execution_1m_guard_blocked")

    return _verdict(
        passed=not failed,
        failed_conditions=failed,
        indicators=indicators,
        entry_mode="golden_15m_short_breakout",
        config=config,
    )


def is_early_5m_starter_short_signal(
    candles_15m: Sequence[Candle],
    candles_5m: Sequence[Candle],
    order_book: dict[str, object],
    execution_guard_1m: dict[str, object],
    market_regime,
    config,
    *,
    source_strategy_direction: str = "none",
) -> dict[str, object]:
    failed: list[str] = []
    strategy = config.strategy
    baseline = config.golden_baseline
    early = baseline.early_5m
    trigger = early.trigger
    latest_15m = candles_15m[-1] if candles_15m else None
    latest_5m = candles_5m[-1] if candles_5m else None
    features_15m = _baseline_features(candles_15m, strategy)
    features_5m = _trigger_features(candles_5m, trigger)
    indicators = {
        "context_15m": features_15m,
        "trigger_5m": features_5m,
        "turnover_rub": _round_or_none(average_turnover(list(candles_15m), strategy.volume_window)),
        "execution_1m": _execution_guard_summary(execution_guard_1m),
    }

    if not bool(baseline.enabled):
        failed.append("golden_baseline_disabled")
    if not bool(early.enabled):
        failed.append("early_5m_disabled")
    if not bool(early.real_trading_enabled):
        failed.append("early_5m_real_disabled")
    if str(config.execution.mode.value) != "local-paper":
        failed.append("execution_mode_not_local_paper")
    if bool(config.execution.allow_live_trading):
        failed.append("allow_live_trading_true")
    if _tf(baseline.timeframes_config.early_trigger) != "5min":
        failed.append("early_trigger_timeframe_not_5min")
    if _tf(baseline.timeframes_config.execution_guard) != "1min":
        failed.append("execution_guard_timeframe_not_1min")
    if _tf(baseline.timeframes_config.early_trigger) in _forbidden_timeframes(config):
        failed.append("forbidden_early_timeframe")
    if _tf(baseline.timeframes_config.execution_guard) in _forbidden_timeframes(config):
        failed.append("forbidden_execution_guard_timeframe")
    if str(source_strategy_direction).lower() == "long":
        failed.append("strategy_long_signal_present")
    if getattr(market_regime, "regime", "") == "range_chop":
        failed.append("range_chop_block")
    if latest_15m is None:
        failed.append("missing_15m_context")
    if latest_5m is None:
        failed.append("missing_5m_candles")
    if len(candles_5m) < max(1, int(trigger.rolling_low_window_max)):
        failed.append("insufficient_5m_candles")

    close_15m = float(latest_15m.close) if latest_15m is not None else None
    if _finite(features_15m.get("ema_fast")) and _finite(features_15m.get("ema_slow")):
        if not float(features_15m["ema_fast"]) < float(features_15m["ema_slow"]):
            failed.append("context_ema20_not_below_ema50")
    else:
        failed.append("context_ema_unavailable")
    if _finite(features_15m.get("adx")):
        if float(features_15m["adx"]) < float(strategy.adx_min):
            failed.append("context_adx_below_min")
    else:
        failed.append("context_adx_unavailable")
    if _finite(features_15m.get("macd_hist")):
        if float(features_15m["macd_hist"]) >= 0:
            failed.append("context_macd_hist_not_negative")
    else:
        failed.append("context_macd_hist_unavailable")
    if _finite(features_15m.get("rsi")):
        value = float(features_15m["rsi"])
        if not float(strategy.rsi_short_min) <= value <= float(trigger.rsi_max):
            failed.append("context_rsi_outside_band")
    else:
        failed.append("context_rsi_unavailable")
    if close_15m is not None and _finite(features_15m.get("ema_fast")) and close_15m > float(features_15m["ema_fast"]):
        failed.append("context_close_above_ema20")

    turnover = indicators.get("turnover_rub")
    if turnover is None:
        failed.append("turnover_unavailable")
    elif float(turnover) < float(strategy.min_liquidity_rub):
        failed.append("liquidity_below_min")

    close_5m = float(latest_5m.close) if latest_5m is not None else None
    if _finite(features_5m.get("ema9")):
        if close_5m is not None and close_5m > float(features_5m["ema9"]):
            failed.append("trigger_close_above_ema9")
    else:
        failed.append("trigger_ema9_unavailable")
    if _finite(features_5m.get("ema9_slope")):
        if float(features_5m["ema9_slope"]) >= 0:
            failed.append("trigger_ema9_slope_not_negative")
    else:
        failed.append("trigger_ema9_slope_unavailable")
    if _finite(features_5m.get("macd_hist")):
        if float(features_5m["macd_hist"]) >= 0:
            failed.append("trigger_macd_hist_not_negative")
    else:
        failed.append("trigger_macd_hist_unavailable")
    if _finite(features_5m.get("rsi")):
        value = float(features_5m["rsi"])
        if not float(trigger.rsi_min) <= value <= float(trigger.rsi_max):
            failed.append("trigger_rsi_outside_band")
    else:
        failed.append("trigger_rsi_unavailable")
    if _finite(features_5m.get("rolling_low_max")):
        if close_5m is not None and close_5m > float(features_5m["rolling_low_max"]):
            failed.append("trigger_close_not_below_rolling_low")
    else:
        failed.append("trigger_rolling_low_unavailable")
    if _finite(features_5m.get("close_position")) and float(features_5m["close_position"]) > float(trigger.max_close_position):
        failed.append("trigger_close_position_too_high")
    if _finite(features_5m.get("ret_window")) and float(features_5m["ret_window"]) > float(trigger.max_adverse_ret):
        failed.append("trigger_adverse_rebound")

    failed.extend(_order_book_failures(order_book, config))
    if not bool(execution_guard_1m.get("available", False)):
        if bool(baseline.execution_1m.required_for_early_5m):
            failed.append("execution_1m_unavailable")
    elif not bool(execution_guard_1m.get("passed", False)):
        failed.append("execution_1m_guard_blocked")

    return _verdict(
        passed=not failed,
        failed_conditions=failed,
        indicators=indicators,
        entry_mode="early_5m_starter_short",
        config=config,
        size_multiplier=float(early.starter_size_multiplier),
    )


def passes_1m_execution_guard(
    candles_1m: Sequence[Candle],
    candles_5m: Sequence[Candle],
    order_book: dict[str, object],
    config,
) -> dict[str, object]:
    guard = config.golden_baseline.execution_1m
    if not bool(guard.enabled):
        return {
            "available": False,
            "enabled": False,
            "passed": True,
            "failed_conditions": [],
            "reason": "execution_1m_disabled",
        }
    if not candles_1m:
        return {
            "available": False,
            "enabled": True,
            "passed": False,
            "failed_conditions": ["execution_1m_unavailable"],
            "reason": "no 1m candles",
        }
    lookback = max(1, int(guard.lookback_bars))
    window = list(candles_1m)[-lookback:]
    latest = window[-1]
    first = window[0]
    failed: list[str] = []
    rebound_bars = sum(1 for candle in window if float(candle.close) > float(candle.open))
    ret_window = latest.close / first.open - 1.0 if first.open > 0 else 0.0
    price_range = latest.high - latest.low
    close_position = (latest.close - latest.low) / price_range if price_range > 0 else 0.5
    if rebound_bars >= max(1, int(guard.block_if_rebound_bars)):
        failed.append("execution_1m_rebound_bars")
    if ret_window > float(guard.max_positive_rebound_ret):
        failed.append("execution_1m_positive_rebound")
    if (rebound_bars > 0 or ret_window > 0) and close_position > float(guard.max_close_position_after_rebound):
        failed.append("execution_1m_upper_range_close")
    if bool(guard.block_if_price_above_5m_ema9):
        closes_5m = [float(candle.close) for candle in candles_5m]
        ema9 = _ema_value(closes_5m, 9)
        if ema9 is None:
            failed.append("execution_1m_5m_ema9_unavailable")
        elif latest.close > ema9:
            failed.append("execution_1m_price_above_5m_ema9")
    if bool(guard.block_if_order_book_deteriorates):
        failed.extend(_order_book_failures(order_book, config, prefix="execution_1m"))
    return {
        "available": True,
        "enabled": True,
        "passed": not failed,
        "failed_conditions": failed,
        "rebound_bars": rebound_bars,
        "ret_window": round(ret_window, 6),
        "close_position": round(close_position, 6),
        "latest_close": round(float(latest.close), 6),
        "reason": "passed" if not failed else "; ".join(failed),
    }


def _baseline_features(candles: Sequence[Candle], strategy) -> dict[str, float | None]:
    candles = list(candles)
    closes = [float(candle.close) for candle in candles]
    features = {
        "ema_fast": _round_or_none(_ema_value(closes, int(strategy.fast_window))),
        "ema_slow": _round_or_none(_ema_value(closes, int(strategy.slow_window))),
        "rsi": _round_or_none(rsi(closes, int(strategy.rsi_window))),
    }
    macd = _macd_features(closes, int(strategy.macd_fast), int(strategy.macd_slow), int(strategy.macd_signal))
    features.update(macd)
    features["adx"] = _round_or_none(_adx_value(candles, int(strategy.adx_window)))
    return features


def _trigger_features(candles: Sequence[Candle], trigger) -> dict[str, float | None]:
    candles = list(candles)
    closes = [float(candle.close) for candle in candles]
    ema_series = _ema_series(closes, int(trigger.ema_window))
    ema9 = ema_series[-1] if ema_series else None
    previous_ema9 = next((value for value in reversed(ema_series[:-1]) if value is not None), None)
    latest = candles[-1] if candles else None
    first = candles[-max(1, int(trigger.rolling_low_window_min))] if candles else None
    price_range = (latest.high - latest.low) if latest is not None else 0.0
    return {
        "ema9": _round_or_none(ema9),
        "ema9_slope": _round_or_none(ema9 - previous_ema9 if ema9 is not None and previous_ema9 is not None else None),
        "rsi": _round_or_none(rsi(closes, int(trigger.rsi_window))),
        **_macd_features(closes, int(trigger.macd_fast), int(trigger.macd_slow), int(trigger.macd_signal)),
        "rolling_low_min": _round_or_none(rolling_low(candles[:-1], int(trigger.rolling_low_window_min))),
        "rolling_low_max": _round_or_none(rolling_low(candles[:-1], int(trigger.rolling_low_window_max))),
        "close_position": _round_or_none((latest.close - latest.low) / price_range if latest is not None and price_range > 0 else None),
        "ret_window": _round_or_none(latest.close / first.open - 1.0 if latest is not None and first is not None and first.open > 0 else None),
    }


def _confirmation_failures(confirmation: dict[str, object], config) -> list[str]:
    if not isinstance(confirmation, dict) or not confirmation.get("available", False):
        return ["confirmation_5m_unavailable"]
    failed: list[str] = []
    if _tf(str(confirmation.get("timeframe", ""))) != _tf(config.golden_baseline.timeframes_config.early_trigger):
        failed.append("confirmation_timeframe_not_5min")
    if int(confirmation.get("bars", 0) or 0) < int(config.strategy.entry_confirmation_min_bars):
        failed.append("confirmation_5m_insufficient_bars")
    if bool(confirmation.get("against_direction", False)) or not bool(confirmation.get("confirmation_ok", True)):
        failed.append("confirmation_5m_against_short")
    return failed


def _order_book_failures(order_book: dict[str, object], config, *, prefix: str = "order_book") -> list[str]:
    if not isinstance(order_book, dict) or not order_book.get("available", False):
        return [f"{prefix}_unavailable"] if bool(config.strategy.require_order_book) else []
    failed: list[str] = []
    spread = _safe_float(order_book.get("spread_bps"))
    cover = _safe_float(order_book.get("entry_liquidity_cover"))
    imbalance = _safe_float(order_book.get("side_imbalance", order_book.get("imbalance")))
    if spread is None:
        failed.append(f"{prefix}_spread_unavailable")
    elif float(config.strategy.max_entry_spread_bps) > 0 and spread > float(config.strategy.max_entry_spread_bps):
        failed.append(f"{prefix}_spread_too_wide")
    if cover is None:
        failed.append(f"{prefix}_liquidity_cover_unavailable")
    elif float(config.strategy.min_entry_liquidity_cover) > 0 and cover < float(config.strategy.min_entry_liquidity_cover):
        failed.append(f"{prefix}_liquidity_cover_too_low")
    if imbalance is None:
        failed.append(f"{prefix}_imbalance_unavailable")
    elif imbalance < float(config.strategy.min_entry_book_imbalance):
        failed.append(f"{prefix}_imbalance_too_low")
    return failed


def _ema_value(values: Sequence[float], window: int) -> float | None:
    series = _ema_series(values, window)
    return series[-1] if series else None


def _ema_series(values: Sequence[float | None], window: int) -> list[float | None]:
    if window <= 0:
        return [None for _ in values]
    series: list[float | None] = []
    seed: list[float] = []
    average: float | None = None
    multiplier = 2 / (window + 1)
    for raw in values:
        if raw is None or not _finite(raw):
            series.append(None)
            continue
        value = float(raw)
        if average is None:
            seed.append(value)
            if len(seed) < window:
                series.append(None)
                continue
            average = sum(seed) / window
            series.append(average)
            continue
        average = (value - average) * multiplier + average
        series.append(average)
    return series


def _macd_features(values: Sequence[float], fast: int, slow: int, signal: int) -> dict[str, float | None]:
    fast_series = _ema_series(values, fast)
    slow_series = _ema_series(values, slow)
    macd_series: list[float | None] = []
    for fast_value, slow_value in zip(fast_series, slow_series, strict=False):
        if fast_value is None or slow_value is None:
            macd_series.append(None)
        else:
            macd_series.append(fast_value - slow_value)
    signal_series = _ema_series(macd_series, signal)
    macd = macd_series[-1] if macd_series else None
    macd_signal = signal_series[-1] if signal_series else None
    return {
        "macd": _round_or_none(macd),
        "macd_signal": _round_or_none(macd_signal),
        "macd_hist": _round_or_none(macd - macd_signal if macd is not None and macd_signal is not None else None),
    }


def _adx_value(candles: Sequence[Candle], window: int) -> float | None:
    candles = list(candles)
    if len(candles) < window * 2 or window <= 0:
        return None
    dx_values: list[float] = []
    for end in range(window, len(candles)):
        sample = candles[end - window + 1 : end + 1]
        previous_sample = candles[end - window : end]
        tr_sum = 0.0
        plus_dm_sum = 0.0
        minus_dm_sum = 0.0
        for previous, current in zip(previous_sample, sample, strict=True):
            high_diff = current.high - previous.high
            low_diff = previous.low - current.low
            plus_dm_sum += high_diff if high_diff > low_diff and high_diff > 0 else 0.0
            minus_dm_sum += low_diff if low_diff > high_diff and low_diff > 0 else 0.0
            tr_sum += max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        if tr_sum <= 0:
            continue
        plus_di = 100.0 * plus_dm_sum / tr_sum
        minus_di = 100.0 * minus_dm_sum / tr_sum
        denominator = plus_di + minus_di
        if denominator <= 0:
            continue
        dx_values.append(100.0 * abs(plus_di - minus_di) / denominator)
    if len(dx_values) < window:
        return None
    return sum(dx_values[-window:]) / window


def _short_only_field(signal, key: str) -> str:
    if signal is None:
        return ""
    metadata = getattr(signal, "metadata", {})
    if not isinstance(metadata, dict):
        return ""
    short_only = metadata.get("short_only", {})
    if not isinstance(short_only, dict):
        return ""
    return str(short_only.get(key, ""))


def _require_feature(failed: list[str], features: dict[str, object], key: str, reason: str) -> None:
    if not _finite(features.get(key)):
        failed.append(reason)


def _execution_guard_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "available": bool(payload.get("available", False)),
        "passed": bool(payload.get("passed", False)),
        "failed_conditions": list(payload.get("failed_conditions", []))
        if isinstance(payload.get("failed_conditions", []), list)
        else [],
        "reason": str(payload.get("reason", "")),
    }


def _verdict(
    *,
    passed: bool,
    failed_conditions: list[str],
    indicators: dict[str, object],
    entry_mode: str,
    config,
    size_multiplier: float = 1.0,
) -> dict[str, object]:
    return {
        "enabled": bool(config.golden_baseline.enabled),
        "passed": bool(passed),
        "verdict": "passed" if passed else "shadow_only",
        "entry_mode": entry_mode,
        "failed_conditions": failed_conditions,
        "indicators": indicators,
        "timeframes": {
            "primary": config.golden_baseline.timeframes_config.primary,
            "early_trigger": config.golden_baseline.timeframes_config.early_trigger,
            "execution_guard": config.golden_baseline.timeframes_config.execution_guard,
        },
        "source_run": config.golden_baseline.source_run,
        "source_commit": config.golden_baseline.source_commit,
        "size_multiplier": round(max(0.0, min(1.0, float(size_multiplier))), 6),
    }


def _forbidden_timeframes(config) -> set[str]:
    return {_tf(value) for value in config.golden_baseline.forbidden_timeframes}


def _tf(value: str) -> str:
    return str(value).strip().lower()


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite(value: object) -> bool:
    return _safe_float(value) is not None


def _round_or_none(value: float | None, *, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)
