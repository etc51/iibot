from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import median
from typing import Any, ClassVar

from ..analysis.indicators import atr, ema
from ..domain import Candle


@dataclass(frozen=True)
class MarketRegime:
    MARKET_SELLOFF_IMPULSE: ClassVar[str] = "market_selloff_impulse"

    regime: str
    confidence: float
    features: dict[str, object]

    def as_event(self) -> dict[str, object]:
        return {
            "event": "market_regime_detected",
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "features": {key: _event_feature_value(value) for key, value in self.features.items()},
        }


def detect_market_regime(
    histories: dict[str, list[Candle]],
    *,
    fast_window: int = 20,
    slow_window: int = 50,
    adx_window: int = 14,
    atr_window: int = 14,
    chop_window: int = 12,
) -> MarketRegime:
    symbol_features: list[dict[str, float]] = []
    selloff_features = _market_selloff_features(histories)
    required = max(slow_window + 1, adx_window * 2 + 1, atr_window + 1, chop_window + 1)
    for candles in histories.values():
        if len(candles) < required:
            continue
        closes = [candle.close for candle in candles]
        fast = ema(closes, fast_window)
        slow = ema(closes, slow_window)
        adx_value = _adx(candles, adx_window)
        atr_value = atr(candles, atr_window)
        latest_close = closes[-1]
        if (
            fast is None
            or slow is None
            or adx_value is None
            or atr_value is None
            or latest_close <= 0
            or slow <= 0
        ):
            continue
        symbol_features.append(
            {
                "down": 1.0 if fast < slow else 0.0,
                "up": 1.0 if fast > slow else 0.0,
                "adx": adx_value,
                "atr_pct": atr_value / latest_close,
                "chop": _chop_score(closes, chop_window),
            }
        )

    if not symbol_features:
        unknown_features: dict[str, object] = {
            "breadth_down": 0.0,
            "breadth_up": 0.0,
            "median_adx": 0.0,
            "chop_score": 0.0,
            "median_atr_pct": 0.0,
            "symbols": 0.0,
        }
        if bool(selloff_features.get("market_selloff_impulse")):
            return _selloff_regime(
                base_features=unknown_features,
                selloff_features=selloff_features,
                previous_regime="unknown",
            )
        return MarketRegime(
            regime="unknown",
            confidence=0.0,
            features=unknown_features,
        )

    breadth_down = sum(item["down"] for item in symbol_features) / len(symbol_features)
    breadth_up = sum(item["up"] for item in symbol_features) / len(symbol_features)
    median_adx = median(item["adx"] for item in symbol_features)
    chop_score = median(item["chop"] for item in symbol_features)
    median_atr_pct = median(item["atr_pct"] for item in symbol_features)
    features = {
        "breadth_down": breadth_down,
        "breadth_up": breadth_up,
        "median_adx": median_adx,
        "chop_score": chop_score,
        "median_atr_pct": median_atr_pct,
        "symbols": float(len(symbol_features)),
    }

    previous_regime, _ = _classify_base_regime(features)
    if bool(selloff_features.get("market_selloff_impulse")):
        return _selloff_regime(
            base_features=features,
            selloff_features=selloff_features,
            previous_regime=previous_regime,
        )

    if breadth_down >= 0.58 and (median_adx < 24.0 or chop_score >= 0.7) and chop_score >= 0.45:
        confidence = _clamp(0.35 + (breadth_down - 0.58) * 1.2 + chop_score * 0.35 + max(0.0, 24.0 - median_adx) / 100)
        return MarketRegime("weak_down_choppy", confidence, features)
    if breadth_down >= 0.62 and median_adx >= 24.0 and chop_score < 0.45:
        confidence = _clamp(0.4 + (breadth_down - 0.62) * 1.0 + min(median_adx, 50.0) / 120 + (0.45 - chop_score) * 0.3)
        return MarketRegime("clean_downtrend", confidence, features)
    if max(breadth_down, breadth_up) <= 0.58 and (median_adx < 22.0 or chop_score >= 0.7) and chop_score >= 0.45:
        confidence = _clamp(
            0.4
            + (0.58 - max(breadth_down, breadth_up)) * 0.8
            + chop_score * 0.3
            + max(0.0, 22.0 - median_adx) / 100
        )
        return MarketRegime("range_chop", confidence, features)
    if breadth_up >= 0.62 and median_adx >= 24.0 and chop_score < 0.45:
        confidence = _clamp(0.4 + (breadth_up - 0.62) * 1.0 + min(median_adx, 50.0) / 120)
        return MarketRegime("clean_uptrend", confidence, features)
    return MarketRegime("mixed", _clamp(0.35 + abs(breadth_down - breadth_up) * 0.4), features)


def _classify_base_regime(features: dict[str, object]) -> tuple[str, float]:
    breadth_down = float(features["breadth_down"])
    breadth_up = float(features["breadth_up"])
    median_adx = float(features["median_adx"])
    chop_score = float(features["chop_score"])
    if breadth_down >= 0.58 and (median_adx < 24.0 or chop_score >= 0.7) and chop_score >= 0.45:
        confidence = _clamp(0.35 + (breadth_down - 0.58) * 1.2 + chop_score * 0.35 + max(0.0, 24.0 - median_adx) / 100)
        return "weak_down_choppy", confidence
    if breadth_down >= 0.62 and median_adx >= 24.0 and chop_score < 0.45:
        confidence = _clamp(0.4 + (breadth_down - 0.62) * 1.0 + min(median_adx, 50.0) / 120 + (0.45 - chop_score) * 0.3)
        return "clean_downtrend", confidence
    if max(breadth_down, breadth_up) <= 0.58 and (median_adx < 22.0 or chop_score >= 0.7) and chop_score >= 0.45:
        confidence = _clamp(
            0.4
            + (0.58 - max(breadth_down, breadth_up)) * 0.8
            + chop_score * 0.3
            + max(0.0, 22.0 - median_adx) / 100
        )
        return "range_chop", confidence
    if breadth_up >= 0.62 and median_adx >= 24.0 and chop_score < 0.45:
        confidence = _clamp(0.4 + (breadth_up - 0.62) * 1.0 + min(median_adx, 50.0) / 120)
        return "clean_uptrend", confidence
    return "mixed", _clamp(0.35 + abs(breadth_down - breadth_up) * 0.4)


def _market_selloff_features(histories: dict[str, list[Candle]]) -> dict[str, object]:
    ret_5m: list[float] = []
    ret_15m: list[float] = []
    ret_30m: list[float] = []
    ret_60m: list[float] = []
    break_15m: list[float] = []
    break_30m: list[float] = []
    close_positions: list[float] = []
    range_expansions: list[float] = []
    rebound_from_lows: list[float] = []
    confirming_count = 0
    used_fallback = False

    for candles in histories.values():
        if len(candles) < 2:
            continue
        candles = sorted(candles, key=lambda candle: candle.timestamp)
        latest = candles[-1]
        if latest.close <= 0:
            continue
        interval_minutes = _infer_interval_minutes(candles)
        used_fallback = used_fallback or interval_minutes > 5.1
        bars_5m = _bars_for_horizon(interval_minutes, 5)
        bars_15m = _bars_for_horizon(interval_minutes, 15)
        bars_30m = _bars_for_horizon(interval_minutes, 30)
        bars_60m = _bars_for_horizon(interval_minutes, 60)

        value_5m = _return_over_bars(candles, bars_5m)
        value_15m = _return_over_bars(candles, bars_15m)
        value_30m = _return_over_bars(candles, bars_30m)
        value_60m = _return_over_bars(candles, bars_60m)
        if value_5m is not None:
            ret_5m.append(value_5m)
        if value_15m is not None:
            ret_15m.append(value_15m)
        if value_30m is not None:
            ret_30m.append(value_30m)
        if value_60m is not None:
            ret_60m.append(value_60m)

        breaking_15m = _breaks_recent_low(candles, bars_15m)
        breaking_30m = _breaks_recent_low(candles, bars_30m)
        break_15m.append(1.0 if breaking_15m else 0.0)
        break_30m.append(1.0 if breaking_30m else 0.0)
        if (
            ((value_15m is not None and value_15m <= -0.0015)
            or (value_30m is not None and value_30m <= -0.003))
            and (breaking_15m or breaking_30m)
        ):
            confirming_count += 1

        close_positions.append(_close_position(latest))
        range_expansions.append(_range_expansion(candles))
        rebound_from_lows.append(_rebound_from_recent_low(candles, bars_60m))

    universe_ret_5m = _median_or_zero(ret_5m)
    universe_ret_15m = _median_or_zero(ret_15m)
    universe_ret_30m = _median_or_zero(ret_30m)
    universe_ret_60m = _median_or_zero(ret_60m)
    breadth_down_5m = _breadth_down(ret_5m)
    breadth_down_15m = _breadth_down(ret_15m)
    breadth_breaking_15m_lows = _average_or_zero(break_15m)
    breadth_breaking_30m_lows = _average_or_zero(break_30m)
    market_selloff_impulse = (
        (
            universe_ret_15m <= -0.004
            or universe_ret_30m <= -0.007
            or universe_ret_60m <= -0.012
        )
        and (breadth_down_5m >= 0.60 or breadth_down_15m >= 0.60)
        and (breadth_breaking_15m_lows >= 0.30 or breadth_breaking_30m_lows >= 0.35)
        and confirming_count >= 10
    )

    return {
        "market_selloff_impulse": market_selloff_impulse,
        "universe_ret_5m": universe_ret_5m,
        "universe_ret_15m": universe_ret_15m,
        "universe_ret_30m": universe_ret_30m,
        "universe_ret_60m": universe_ret_60m,
        "breadth_down_5m": breadth_down_5m,
        "breadth_down_15m": breadth_down_15m,
        "breadth_breaking_15m_lows": breadth_breaking_15m_lows,
        "breadth_breaking_30m_lows": breadth_breaking_30m_lows,
        "symbols_confirming_count": confirming_count,
        "median_close_position": _median_or_zero(close_positions),
        "median_range_expansion": _median_or_zero(range_expansions),
        "market_proxy_rebound_from_low_pct": _median_or_zero(rebound_from_lows),
        "used_fallback": used_fallback,
    }


def _selloff_regime(
    *,
    base_features: dict[str, object],
    selloff_features: dict[str, object],
    previous_regime: str,
) -> MarketRegime:
    features = {
        **base_features,
        **selloff_features,
        "previous_regime": previous_regime,
        "override_reason": "broad-fast-selloff-before-weak-down-choppy",
    }
    return MarketRegime(
        MarketRegime.MARKET_SELLOFF_IMPULSE,
        _selloff_confidence(selloff_features),
        features,
    )


def _selloff_confidence(features: dict[str, object]) -> float:
    move_score = max(
        max(0.0, (-float(features.get("universe_ret_15m", 0.0)) - 0.004) / 0.016),
        max(0.0, (-float(features.get("universe_ret_30m", 0.0)) - 0.007) / 0.025),
        max(0.0, (-float(features.get("universe_ret_60m", 0.0)) - 0.012) / 0.04),
    )
    breadth_score = max(
        float(features.get("breadth_down_5m", 0.0)),
        float(features.get("breadth_down_15m", 0.0)),
    )
    low_break_score = max(
        float(features.get("breadth_breaking_15m_lows", 0.0)),
        float(features.get("breadth_breaking_30m_lows", 0.0)),
    )
    count_score = min(1.0, float(features.get("symbols_confirming_count", 0.0)) / 20.0)
    return _clamp(0.55 + min(move_score, 1.0) * 0.20 + breadth_score * 0.15 + low_break_score * 0.10 + count_score * 0.10)


def _infer_interval_minutes(candles: list[Candle]) -> float:
    if len(candles) < 2:
        return 5.0
    deltas = [
        (current.timestamp - previous.timestamp).total_seconds() / 60.0
        for previous, current in zip(candles[-6:-1], candles[-5:])
        if (current.timestamp - previous.timestamp).total_seconds() > 0
    ]
    if not deltas:
        return 5.0
    return max(1.0, median(deltas))


def _bars_for_horizon(interval_minutes: float, horizon_minutes: int) -> int:
    return max(1, int(ceil(float(horizon_minutes) / max(1.0, interval_minutes))))


def _return_over_bars(candles: list[Candle], bars: int) -> float | None:
    if len(candles) <= bars:
        return None
    reference = candles[-bars - 1].close
    latest = candles[-1].close
    if reference <= 0:
        return None
    return latest / reference - 1.0


def _breaks_recent_low(candles: list[Candle], bars: int) -> bool:
    lookback = candles[-bars - 1 : -1] if len(candles) > bars else candles[:-1]
    if not lookback:
        return False
    return candles[-1].close <= min(candle.low for candle in lookback)


def _close_position(candle: Candle) -> float:
    price_range = candle.high - candle.low
    if price_range <= 0:
        return 0.5
    return _clamp((candle.close - candle.low) / price_range)


def _range_expansion(candles: list[Candle]) -> float:
    latest_range = max(0.0, candles[-1].high - candles[-1].low)
    previous_ranges = [
        max(0.0, candle.high - candle.low)
        for candle in candles[-21:-1]
        if candle.high >= candle.low
    ]
    median_range = median(previous_ranges) if previous_ranges else latest_range
    if median_range <= 0:
        return 1.0
    return latest_range / median_range


def _rebound_from_recent_low(candles: list[Candle], bars: int) -> float:
    sample = candles[-bars - 1 :] if len(candles) > bars else candles
    low = min((candle.low for candle in sample), default=candles[-1].low)
    if low <= 0:
        return 0.0
    return candles[-1].close / low - 1.0


def _median_or_zero(values: list[float]) -> float:
    return median(values) if values else 0.0


def _average_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _breadth_down(values: list[float]) -> float:
    return sum(1 for value in values if value < 0.0) / len(values) if values else 0.0


def _event_feature_value(value: Any) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def _chop_score(closes: list[float], window: int) -> float:
    sample = closes[-window - 1 :]
    path = sum(abs(current - previous) for previous, current in zip(sample[:-1], sample[1:]))
    if path <= 0:
        return 1.0
    net = abs(sample[-1] - sample[0])
    return _clamp(1.0 - net / path)


def _adx(candles: list[Candle], window: int) -> float | None:
    if len(candles) < window * 2 + 1:
        return None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    true_ranges: list[float] = []
    for previous, current in zip(candles[-window * 2 - 1 : -1], candles[-window * 2 :]):
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    dx_values: list[float] = []
    for index in range(window, len(true_ranges) + 1):
        tr_sum = sum(true_ranges[index - window : index])
        if tr_sum <= 0:
            continue
        plus_di = 100.0 * sum(plus_dm[index - window : index]) / tr_sum
        minus_di = 100.0 * sum(minus_dm[index - window : index]) / tr_sum
        denominator = plus_di + minus_di
        if denominator <= 0:
            continue
        dx_values.append(100.0 * abs(plus_di - minus_di) / denominator)
    if not dx_values:
        return None
    return sum(dx_values[-window:]) / min(window, len(dx_values))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
