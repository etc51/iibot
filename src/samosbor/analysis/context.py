from __future__ import annotations

from bisect import bisect_right
from typing import Protocol

from .indicators import ema
from ..domain import Candle, Instrument


class ExternalContextProvider(Protocol):
    def score(self, instrument: Instrument, candles: list[Candle]) -> float:
        ...


class NeutralContextProvider:
    def score(self, instrument: Instrument, candles: list[Candle]) -> float:
        return 0.0


class StaticContextProvider:
    def __init__(self, scores: dict[str, float]):
        self.scores = {symbol.upper(): score for symbol, score in scores.items()}

    def score(self, instrument: Instrument, candles: list[Candle]) -> float:
        return self.scores.get(instrument.symbol.upper(), 0.0)


class MarketBreadthContextProvider:
    def __init__(
        self,
        histories: dict[str, list[Candle]],
        *,
        fast_window: int = 20,
        slow_window: int = 50,
        return_window: int = 4,
        min_symbols: int = 8,
        max_score: float = 0.25,
    ):
        self.fast_window = max(1, fast_window)
        self.slow_window = max(self.fast_window + 1, slow_window)
        self.return_window = max(1, return_window)
        self.min_symbols = max(1, min_symbols)
        self.max_score = max(0.0, max_score)
        self._scores = self._build_scores(histories)
        self._timestamps = sorted(self._scores)

    def score(self, instrument: Instrument, candles: list[Candle]) -> float:
        if not candles or not self._timestamps:
            return 0.0
        timestamp = candles[-1].timestamp
        index = bisect_right(self._timestamps, timestamp) - 1
        if index < 0:
            return 0.0
        return self._scores[self._timestamps[index]]

    def _build_scores(self, histories: dict[str, list[Candle]]) -> dict[object, float]:
        votes_by_timestamp: dict[object, list[float]] = {}
        required = max(self.slow_window, self.return_window + 1)
        for candles in histories.values():
            if len(candles) < required:
                continue
            closes: list[float] = []
            for candle in candles:
                closes.append(candle.close)
                if len(closes) < required:
                    continue
                fast = ema(closes, self.fast_window)
                slow = ema(closes, self.slow_window)
                if fast is None or slow is None or slow <= 0:
                    continue
                trend_vote = 1.0 if fast > slow else -1.0 if fast < slow else 0.0
                previous = closes[-self.return_window - 1]
                if previous <= 0:
                    continue
                recent_return = closes[-1] / previous - 1.0
                momentum_vote = 1.0 if recent_return > 0.001 else -1.0 if recent_return < -0.001 else 0.0
                vote = trend_vote * 0.7 + momentum_vote * 0.3
                votes_by_timestamp.setdefault(candle.timestamp, []).append(vote)

        scores: dict[object, float] = {}
        for timestamp, votes in votes_by_timestamp.items():
            if len(votes) < self.min_symbols:
                continue
            raw_score = sum(votes) / len(votes)
            score = max(-self.max_score, min(self.max_score, raw_score * self.max_score))
            scores[timestamp] = round(score, 4)
        return scores
