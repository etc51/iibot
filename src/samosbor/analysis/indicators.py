from __future__ import annotations

import math

from ..domain import Candle


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window or window <= 0:
        return None
    sample = values[-window:]
    return sum(sample) / window


def ema(values: list[float], window: int) -> float | None:
    if len(values) < window or window <= 0:
        return None
    multiplier = 2 / (window + 1)
    average = sum(values[:window]) / window
    for value in values[window:]:
        average = (value - average) * multiplier + average
    return average


def rsi(values: list[float], window: int = 14) -> float | None:
    if len(values) <= window:
        return None
    gains = []
    losses = []
    for previous, current in zip(values[-window - 1 : -1], values[-window:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    average_gain = sum(gains) / window
    average_loss = sum(losses) / window
    if average_loss == 0:
        return 100.0
    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))


def average_turnover(candles: list[Candle], window: int) -> float | None:
    if len(candles) < window:
        return None
    turnover = [candle.close * candle.volume for candle in candles[-window:]]
    return sum(turnover) / window


def atr(candles: list[Candle], window: int = 14) -> float | None:
    if len(candles) < window + 1:
        return None
    true_ranges: list[float] = []
    for previous, current in zip(candles[-window - 1 : -1], candles[-window:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return sum(true_ranges) / window


def rolling_high(candles: list[Candle], window: int) -> float | None:
    if len(candles) < window:
        return None
    return max(candle.high for candle in candles[-window:])


def rolling_low(candles: list[Candle], window: int) -> float | None:
    if len(candles) < window:
        return None
    return min(candle.low for candle in candles[-window:])


def annualized_volatility(candles: list[Candle], window: int, bars_per_year: int) -> float | None:
    if len(candles) < window + 1:
        return None
    returns = []
    for previous, current in zip(candles[-window - 1 : -1], candles[-window:]):
        if previous.close <= 0:
            continue
        returns.append((current.close / previous.close) - 1.0)
    if len(returns) < 2:
        return None
    average = sum(returns) / len(returns)
    variance = sum((value - average) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(bars_per_year)
