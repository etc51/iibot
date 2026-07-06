from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..analysis.indicators import atr, ema
from ..domain import Candle


@dataclass(frozen=True)
class MarketRegime:
    regime: str
    confidence: float
    features: dict[str, float]

    def as_event(self) -> dict[str, object]:
        return {
            "event": "market_regime_detected",
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "features": {key: round(value, 6) for key, value in self.features.items()},
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
        return MarketRegime(
            regime="unknown",
            confidence=0.0,
            features={
                "breadth_down": 0.0,
                "breadth_up": 0.0,
                "median_adx": 0.0,
                "chop_score": 0.0,
                "median_atr_pct": 0.0,
                "symbols": 0.0,
            },
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
