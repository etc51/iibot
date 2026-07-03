from __future__ import annotations

import math
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

from ..analysis.context import ExternalContextProvider, MarketBreadthContextProvider, NeutralContextProvider
from ..analysis.indicators import annualized_volatility, atr, average_turnover, rolling_high, rolling_low, rsi, sma
from ..config import StrategySection
from ..domain import Candle, Instrument, Signal, SignalDirection


def bars_per_year_for_timeframe(timeframe: str) -> int:
    mapping = {
        "day": 252,
        "hour": 2016,
        "30min": 4032,
        "15min": 8064,
        "10min": 12096,
        "5min": 24192,
        "1min": 120960,
    }
    return mapping.get(timeframe.lower(), 252)


_SUPPORTED_STYLES = {
    "sma_breakout",
    "ema_adx_macd",
    "ema_adx_donchian",
    "adx_regime_hybrid",
    "vwap_relvol_orb",
    "rsi_mean_reversion",
}
_PRECOMPUTED_TA_STYLES = {
    "ema_adx_macd",
    "ema_adx_donchian",
    "adx_regime_hybrid",
    "vwap_relvol_orb",
}
_MACD_STYLES = {"ema_adx_macd", "adx_regime_hybrid"}
_DONCHIAN_STYLES = {"ema_adx_donchian", "adx_regime_hybrid"}
_SESSION_INTRADAY_STYLES = {"vwap_relvol_orb"}


class TrendFollowingStrategy:
    def __init__(
        self,
        config: StrategySection,
        *,
        timeframe: str,
        context_provider: ExternalContextProvider | None = None,
    ):
        self.config = config
        self.timeframe = timeframe
        self.context_provider = context_provider or NeutralContextProvider()
        self.style = config.style.strip().lower()
        self._prepared_ta: dict[str, dict[str, object]] = {}
        if self.style not in _SUPPORTED_STYLES:
            raise ValueError(f"Unsupported strategy style: {config.style}")
        self.schedule_timezone = ZoneInfo(config.schedule_timezone)
        self.allowed_symbols = {
            symbol.strip().upper() for symbol in config.allowed_symbols if symbol.strip()
        }
        self.blocked_symbols = {symbol.strip().upper() for symbol in config.blocked_symbols if symbol.strip()}
        self.blocked_long_symbols = {
            symbol.strip().upper() for symbol in config.blocked_long_symbols if symbol.strip()
        }
        self.blocked_short_symbols = {
            symbol.strip().upper() for symbol in config.blocked_short_symbols if symbol.strip()
        }

    def _is_runtime_session_bar(self, timestamp: datetime) -> bool:
        localized = timestamp.astimezone(self.schedule_timezone)
        if (
            self.config.allowed_entry_weekdays
            and localized.weekday() not in self.config.allowed_entry_weekdays
        ):
            return False
        if self.config.allowed_entry_hours and localized.hour not in self.config.allowed_entry_hours:
            return False
        if self.should_force_flatten_at(timestamp):
            return False
        return True

    def _intraday_session_labels(
        self,
        timestamps: list[datetime],
    ) -> tuple[list[str | None], list[int | None]]:
        labels: list[str | None] = []
        session_bar_indexes: list[int | None] = []
        previous_active = False
        previous_session_date = None
        session_sequence = 0
        bar_index = -1
        current_label: str | None = None

        for timestamp in timestamps:
            localized = timestamp.astimezone(self.schedule_timezone)
            active = self._is_runtime_session_bar(timestamp)
            if not active:
                labels.append(None)
                session_bar_indexes.append(None)
                previous_active = False
                previous_session_date = None
                current_label = None
                bar_index = -1
                continue

            session_date = localized.date()
            if not previous_active or session_date != previous_session_date:
                session_sequence += 1
                current_label = f"{session_date.isoformat()}#{session_sequence}"
                bar_index = 0
            else:
                bar_index += 1

            labels.append(current_label)
            session_bar_indexes.append(bar_index)
            previous_active = True
            previous_session_date = session_date
        return labels, session_bar_indexes

    def prepare_history(self, instrument: Instrument, candles: list[Candle]) -> None:
        if self.style not in _PRECOMPUTED_TA_STYLES or not candles:
            return
        features = self._compute_ta_feature_series(candles)
        timestamps = [candle.timestamp for candle in candles]
        self._prepared_ta[instrument.instrument_id] = {
            "timestamps": timestamps,
            "features": features,
        }

    def prepare_market_context(self, histories: dict[str, list[Candle]]) -> None:
        if not self.config.use_market_context:
            return
        self.context_provider = MarketBreadthContextProvider(
            histories,
            fast_window=self.config.market_context_fast_window,
            slow_window=self.config.market_context_slow_window,
            return_window=self.config.market_context_return_window,
            min_symbols=self.config.market_context_min_symbols,
            max_score=self.config.market_context_max_score,
        )

    def allows_entry_at(self, timestamp: datetime) -> bool:
        return self.entry_block_reason_at(timestamp) is None

    def should_force_flatten_at(self, timestamp: datetime) -> bool:
        if not self.config.forced_flat_hours:
            return False
        localized = timestamp.astimezone(self.schedule_timezone)
        if (
            self.config.forced_flat_weekdays
            and localized.weekday() not in self.config.forced_flat_weekdays
        ):
            return False
        return localized.hour in self.config.forced_flat_hours

    def entry_block_reason_at(self, timestamp: datetime) -> str | None:
        if self.should_force_flatten_at(timestamp):
            return "entry blocked by session-flat window"
        localized = timestamp.astimezone(self.schedule_timezone)
        if (
            self.config.allowed_entry_weekdays
            and localized.weekday() not in self.config.allowed_entry_weekdays
        ):
            return "entry blocked by weekday schedule"
        if self.config.allowed_entry_hours and localized.hour not in self.config.allowed_entry_hours:
            return "entry blocked by hour schedule"
        return None

    def entry_block_reason_for_instrument(
        self,
        instrument: Instrument,
        timestamp: datetime,
        direction: SignalDirection | None = None,
    ) -> str | None:
        symbol = instrument.symbol.strip().upper()
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            return f"entry blocked by allowed universe restriction ({instrument.symbol})"
        if symbol in self.blocked_symbols:
            return f"entry blocked by symbol restriction ({instrument.symbol})"
        if direction == SignalDirection.LONG and symbol in self.blocked_long_symbols:
            return f"entry blocked by long restriction ({instrument.symbol})"
        if direction == SignalDirection.SHORT and symbol in self.blocked_short_symbols:
            return f"entry blocked by short restriction ({instrument.symbol})"
        return self.entry_block_reason_at(timestamp)

    def _required_bars(self) -> int:
        required = max(
            self.config.slow_window,
            self.config.atr_window + 1,
            self.config.volume_window,
            self.config.breakout_window + 1,
        )
        if self.style in _MACD_STYLES:
            ta_bars = max(
                self.config.adx_window * 3,
                self.config.rsi_window + 2,
                self.config.macd_slow + self.config.macd_signal + 5,
            )
            required = max(required, ta_bars)
        if self.style in _DONCHIAN_STYLES:
            ta_bars = max(
                self.config.adx_window * 3,
                self.config.rsi_window + 2,
                self.config.breakout_window + 2,
            )
            required = max(required, ta_bars)
        if self.style in _SESSION_INTRADAY_STYLES:
            ta_bars = max(
                self.config.slow_window,
                self.config.adx_window * 3,
                self.config.rsi_window + 2,
                self.config.volume_window + 2,
                self.config.opening_range_bars + 2,
            )
            required = max(required, ta_bars)
        return required

    def generate_signal(
        self,
        instrument: Instrument,
        candles: list[Candle],
    ) -> Signal | None:
        required = self._required_bars()
        if len(candles) < required:
            return None

        closes = [candle.close for candle in candles]
        fast = sma(closes, self.config.fast_window)
        slow = sma(closes, self.config.slow_window)
        atr_value = atr(candles, self.config.atr_window)
        turnover = average_turnover(candles, self.config.volume_window)
        breakout_high = rolling_high(candles[:-1], self.config.breakout_window)
        breakout_low = rolling_low(candles[:-1], self.config.breakout_window)
        volatility = annualized_volatility(
            candles,
            self.config.atr_window,
            bars_per_year_for_timeframe(self.timeframe),
        )

        if (
            fast is None
            or slow is None
            or atr_value is None
            or turnover is None
            or breakout_high is None
            or breakout_low is None
            or slow <= 0
            or atr_value <= 0
        ):
            return None

        if turnover < self.config.min_liquidity_rub:
            return None

        last = candles[-1]
        trend_strength = abs(fast - slow) / slow
        context_score = self.context_provider.score(instrument, candles)
        breakout_long = last.close >= breakout_high if self.config.require_breakout else True
        breakout_short = last.close <= breakout_low if self.config.require_breakout else True
        mean_price = slow

        if self.style == "rsi_mean_reversion":
            rsi_value = rsi(closes, self.config.rsi_window)
            if mean_price is None or mean_price <= 0 or rsi_value is None:
                return None
            deviation_strength = abs(last.close - mean_price) / mean_price
            if deviation_strength < self.config.min_trend_strength:
                return None
            if last.close < mean_price and rsi_value <= self.config.rsi_short_min:
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=deviation_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"mean-reversion long mean={mean_price:.2f} close={last.close:.2f} "
                        f"rsi={rsi_value:.2f} deviation={deviation_strength:.4f}"
                    ),
                )
            if last.close > mean_price and self.config.allow_shorts and rsi_value >= self.config.rsi_long_max:
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.SHORT,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=deviation_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"mean-reversion short mean={mean_price:.2f} close={last.close:.2f} "
                        f"rsi={rsi_value:.2f} deviation={deviation_strength:.4f}"
                    ),
                )
            return None

        if self.style == "vwap_relvol_orb":
            ta_features = self._resolved_ta_features(instrument, candles)
            if ta_features is None:
                return None
            fast = ta_features["ema_fast"]
            slow = ta_features["ema_slow"]
            session_vwap = ta_features["session_vwap"]
            relative_volume = ta_features["relative_volume"]
            session_bar_index = int(ta_features["session_bar_index"])
            if session_vwap <= 0 or relative_volume < self.config.rel_volume_threshold:
                return None
            if session_bar_index < self.config.opening_range_bars:
                return None
            ema_alignment = abs(fast - slow) / abs(slow) if slow else 0.0
            vwap_distance = abs(last.close - session_vwap) / session_vwap
            composite_strength = (
                max(ema_alignment, vwap_distance)
                + max(0.0, relative_volume - self.config.rel_volume_threshold) * 0.01
            )
            long_breakout = (
                last.close >= ta_features["opening_range_high"] if self.config.require_breakout else True
            )
            short_breakout = (
                last.close <= ta_features["opening_range_low"] if self.config.require_breakout else True
            )
            if (
                fast > slow
                and ta_features["adx"] >= self.config.adx_min
                and self.config.rsi_long_min <= ta_features["rsi"] <= self.config.rsi_long_max
                and last.close >= session_vwap
                and long_breakout
            ):
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=max(composite_strength, self.config.min_trend_strength),
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"vwap-orb long ema_fast={fast:.2f} ema_slow={slow:.2f} "
                        f"vwap={session_vwap:.2f} relvol={relative_volume:.2f} "
                        f"orb_high={ta_features['opening_range_high']:.2f} adx={ta_features['adx']:.2f}"
                    ),
                )
            if (
                self.config.allow_shorts
                and fast < slow
                and ta_features["adx"] >= self.config.adx_min
                and self.config.rsi_short_min <= ta_features["rsi"] <= self.config.rsi_short_max
                and last.close <= session_vwap
                and short_breakout
            ):
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.SHORT,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=max(composite_strength, self.config.min_trend_strength),
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"vwap-orb short ema_fast={fast:.2f} ema_slow={slow:.2f} "
                        f"vwap={session_vwap:.2f} relvol={relative_volume:.2f} "
                        f"orb_low={ta_features['opening_range_low']:.2f} adx={ta_features['adx']:.2f}"
                    ),
                )
            return None

        if trend_strength < self.config.min_trend_strength:
            return None

        if self.style == "ema_adx_macd":
            ta_features = self._resolved_ta_features(instrument, candles)
            if ta_features is None:
                return None
            fast = ta_features["ema_fast"]
            slow = ta_features["ema_slow"]
            trend_strength = abs(fast - slow) / abs(slow) if slow else 0.0
            if trend_strength < self.config.min_trend_strength:
                return None
            if (
                fast > slow
                and ta_features["adx"] >= self.config.adx_min
                and ta_features["macd_hist"] > 0
                and self.config.rsi_long_min <= ta_features["rsi"] <= self.config.rsi_long_max
                and last.close >= fast
                and breakout_long
            ):
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=trend_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"ema-up ema_fast={fast:.2f} ema_slow={slow:.2f} "
                        f"adx={ta_features['adx']:.2f} rsi={ta_features['rsi']:.2f} "
                        f"macd_hist={ta_features['macd_hist']:.4f}"
                    ),
                )
            if (
                self.config.allow_shorts
                and fast < slow
                and ta_features["adx"] >= self.config.adx_min
                and ta_features["macd_hist"] < 0
                and self.config.rsi_short_min <= ta_features["rsi"] <= self.config.rsi_short_max
                and last.close <= fast
                and breakout_short
            ):
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.SHORT,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=trend_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"ema-down ema_fast={fast:.2f} ema_slow={slow:.2f} "
                        f"adx={ta_features['adx']:.2f} rsi={ta_features['rsi']:.2f} "
                        f"macd_hist={ta_features['macd_hist']:.4f}"
                    ),
                )
            return None

        if self.style == "ema_adx_donchian":
            ta_features = self._resolved_ta_features(instrument, candles)
            if ta_features is None:
                return None
            fast = ta_features["ema_fast"]
            slow = ta_features["ema_slow"]
            trend_strength = abs(fast - slow) / abs(slow) if slow else 0.0
            if trend_strength < self.config.min_trend_strength:
                return None
            if (
                fast > slow
                and ta_features["adx"] >= self.config.adx_min
                and ta_features["dmp"] > ta_features["dmn"]
                and self.config.rsi_long_min <= ta_features["rsi"] <= self.config.rsi_long_max
                and last.close >= ta_features["donchian_upper"]
                and last.close >= fast
            ):
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=trend_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"donchian-up ema_fast={fast:.2f} ema_slow={slow:.2f} "
                        f"upper={ta_features['donchian_upper']:.2f} adx={ta_features['adx']:.2f} "
                        f"rsi={ta_features['rsi']:.2f} dmi=({ta_features['dmp']:.2f}/{ta_features['dmn']:.2f})"
                    ),
                )
            if (
                self.config.allow_shorts
                and fast < slow
                and ta_features["adx"] >= self.config.adx_min
                and ta_features["dmn"] > ta_features["dmp"]
                and self.config.rsi_short_min <= ta_features["rsi"] <= self.config.rsi_short_max
                and last.close <= ta_features["donchian_lower"]
                and last.close <= fast
            ):
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.SHORT,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=trend_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"donchian-down ema_fast={fast:.2f} ema_slow={slow:.2f} "
                        f"lower={ta_features['donchian_lower']:.2f} adx={ta_features['adx']:.2f} "
                        f"rsi={ta_features['rsi']:.2f} dmi=({ta_features['dmp']:.2f}/{ta_features['dmn']:.2f})"
                    ),
                )
            return None

        if self.style == "adx_regime_hybrid":
            ta_features = self._resolved_ta_features(instrument, candles)
            if ta_features is None or mean_price <= 0:
                return None
            fast = ta_features["ema_fast"]
            slow = ta_features["ema_slow"]
            rsi_value = ta_features["rsi"]
            trend_strength = abs(fast - slow) / abs(slow) if slow else 0.0
            trend_breakout_long = (
                last.close >= ta_features["donchian_upper"] if self.config.require_breakout else True
            )
            trend_breakout_short = (
                last.close <= ta_features["donchian_lower"] if self.config.require_breakout else True
            )
            if ta_features["adx"] >= self.config.adx_min:
                if trend_strength < self.config.min_trend_strength:
                    return None
                if (
                    fast > slow
                    and ta_features["macd_hist"] > 0
                    and ta_features["dmp"] > ta_features["dmn"]
                    and self.config.rsi_long_min <= rsi_value <= self.config.rsi_long_max
                    and trend_breakout_long
                    and last.close >= fast
                ):
                    return self._build_signal(
                        instrument=instrument,
                        direction=SignalDirection.LONG,
                        last=last,
                        atr_value=atr_value,
                        context_score=context_score,
                        trend_strength=trend_strength,
                        turnover=turnover,
                        volatility=volatility,
                        extra_reason=(
                            f"hybrid-trend long ema_fast={fast:.2f} ema_slow={slow:.2f} "
                            f"adx={ta_features['adx']:.2f} macd_hist={ta_features['macd_hist']:.4f} "
                            f"upper={ta_features['donchian_upper']:.2f}"
                        ),
                    )
                if (
                    self.config.allow_shorts
                    and fast < slow
                    and ta_features["macd_hist"] < 0
                    and ta_features["dmn"] > ta_features["dmp"]
                    and self.config.rsi_short_min <= rsi_value <= self.config.rsi_short_max
                    and trend_breakout_short
                    and last.close <= fast
                ):
                    return self._build_signal(
                        instrument=instrument,
                        direction=SignalDirection.SHORT,
                        last=last,
                        atr_value=atr_value,
                        context_score=context_score,
                        trend_strength=trend_strength,
                        turnover=turnover,
                        volatility=volatility,
                        extra_reason=(
                            f"hybrid-trend short ema_fast={fast:.2f} ema_slow={slow:.2f} "
                            f"adx={ta_features['adx']:.2f} macd_hist={ta_features['macd_hist']:.4f} "
                            f"lower={ta_features['donchian_lower']:.2f}"
                        ),
                    )
                return None

            deviation_strength = abs(last.close - mean_price) / mean_price
            if deviation_strength < self.config.min_trend_strength:
                return None
            if last.close < mean_price and rsi_value <= self.config.rsi_short_min:
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=deviation_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"hybrid-range long mean={mean_price:.2f} close={last.close:.2f} "
                        f"adx={ta_features['adx']:.2f} rsi={rsi_value:.2f}"
                    ),
                )
            if last.close > mean_price and self.config.allow_shorts and rsi_value >= self.config.rsi_long_max:
                return self._build_signal(
                    instrument=instrument,
                    direction=SignalDirection.SHORT,
                    last=last,
                    atr_value=atr_value,
                    context_score=context_score,
                    trend_strength=deviation_strength,
                    turnover=turnover,
                    volatility=volatility,
                    extra_reason=(
                        f"hybrid-range short mean={mean_price:.2f} close={last.close:.2f} "
                        f"adx={ta_features['adx']:.2f} rsi={rsi_value:.2f}"
                    ),
                )
            return None

        if fast > slow and breakout_long:
            return self._build_signal(
                instrument=instrument,
                direction=SignalDirection.LONG,
                last=last,
                atr_value=atr_value,
                context_score=context_score,
                trend_strength=trend_strength,
                turnover=turnover,
                volatility=volatility,
                extra_reason=(
                    f"trend-up fast={fast:.2f} slow={slow:.2f} "
                    f"atr={atr_value:.2f} vol={volatility or 0.0:.3f}"
                ),
            )

        if self.config.allow_shorts and fast < slow and breakout_short:
            return self._build_signal(
                instrument=instrument,
                direction=SignalDirection.SHORT,
                last=last,
                atr_value=atr_value,
                context_score=context_score,
                trend_strength=trend_strength,
                turnover=turnover,
                volatility=volatility,
                extra_reason=(
                    f"trend-down fast={fast:.2f} slow={slow:.2f} "
                    f"atr={atr_value:.2f} vol={volatility or 0.0:.3f}"
                ),
            )

        return None

    def _build_signal(
        self,
        *,
        instrument: Instrument,
        direction: SignalDirection,
        last: Candle,
        atr_value: float,
        context_score: float,
        trend_strength: float,
        turnover: float,
        volatility: float | None,
        extra_reason: str,
    ) -> Signal:
        stop_distance = atr_value * self.config.atr_stop_multiple
        if self._market_context_blocks(direction, context_score):
            return None
        if direction == SignalDirection.LONG:
            stop_distance = atr_value * self.config.atr_stop_multiple
            confidence = min(1.0, trend_strength * 80 + max(context_score, 0.0))
            if confidence < self.config.min_signal_strength:
                return None
            reason = extra_reason
            return Signal(
                instrument=instrument,
                direction=direction,
                strength=confidence,
                entry_price=last.close,
                stop_price=last.close - stop_distance,
                take_profit=last.close + stop_distance * self.config.reward_to_risk,
                reason=reason,
                context_score=context_score,
                metadata={
                    "trend_strength": trend_strength,
                    "context_score": context_score,
                    "turnover": turnover,
                    "volatility": volatility,
                },
            )
        confidence = min(1.0, trend_strength * 80 + max(-context_score, 0.0))
        if confidence < self.config.min_signal_strength:
            return None
        return Signal(
            instrument=instrument,
            direction=direction,
            strength=confidence,
            entry_price=last.close,
            stop_price=last.close + stop_distance,
            take_profit=last.close - stop_distance * self.config.reward_to_risk,
            reason=extra_reason,
            context_score=context_score,
            metadata={
                "trend_strength": trend_strength,
                "context_score": context_score,
                "turnover": turnover,
                "volatility": volatility,
            },
        )

    def _market_context_blocks(self, direction: SignalDirection, context_score: float) -> bool:
        if not self.config.use_market_context:
            return False
        threshold = max(0.0, self.config.market_context_block_threshold)
        if threshold <= 0:
            return False
        if direction == SignalDirection.LONG:
            return context_score <= -threshold
        return context_score >= threshold

    def _resolved_ta_features(
        self,
        instrument: Instrument,
        candles: list[Candle],
    ) -> dict[str, float] | None:
        prepared = self._prepared_ta.get(instrument.instrument_id)
        if prepared is not None and candles:
            timestamps = prepared["timestamps"]
            features = prepared["features"]
            index = len(candles) - 1
            if index < len(timestamps) and timestamps[index] == candles[-1].timestamp:
                return features[index]
        return self._ta_features(candles)

    def _ta_dependencies(self):
        try:
            import pandas as pd
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*mode.copy_on_write.*",
                )
                import pandas_ta as ta
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TA indicator mode requires pandas and pandas-ta to be installed."
            ) from exc
        return pd, ta

    def _ta_features(self, candles: list[Candle]) -> dict[str, float] | None:
        window = max(self._required_bars() + 5, 250)
        subset = candles[-window:]
        features = self._compute_ta_feature_series(subset)
        if not features:
            return None
        return features[-1]

    def _compute_ta_feature_series(self, candles: list[Candle]) -> list[dict[str, float] | None]:
        pd, ta = self._ta_dependencies()
        frame = pd.DataFrame(
            {
                "open": [candle.open for candle in candles],
                "high": [candle.high for candle in candles],
                "low": [candle.low for candle in candles],
                "close": [candle.close for candle in candles],
                "volume": [candle.volume for candle in candles],
            }
        )
        ema_fast = ta.ema(frame["close"], length=self.config.fast_window)
        ema_slow = ta.ema(frame["close"], length=self.config.slow_window)
        rsi = ta.rsi(frame["close"], length=self.config.rsi_window)
        adx = ta.adx(
            frame["high"],
            frame["low"],
            frame["close"],
            length=self.config.adx_window,
        )
        if adx is None:
            return [None for _ in candles]

        macd = None
        donchian = None
        if self.style in _MACD_STYLES:
            macd = ta.macd(
                frame["close"],
                fast=self.config.macd_fast,
                slow=self.config.macd_slow,
                signal=self.config.macd_signal,
            )
            if macd is None:
                return [None for _ in candles]
        if self.style in _DONCHIAN_STYLES:
            donchian = ta.donchian(
                frame["high"],
                frame["low"],
                lower_length=self.config.breakout_window,
                upper_length=self.config.breakout_window,
            )
            if donchian is None:
                return [None for _ in candles]
            lower_col = next((col for col in donchian.columns if "DCL_" in col), donchian.columns[0])
            upper_col = next((col for col in donchian.columns if "DCU_" in col), donchian.columns[-1])
            donchian_lower = donchian[lower_col].shift(1)
            donchian_upper = donchian[upper_col].shift(1)
        session_vwap = None
        relative_volume = None
        opening_range_high = None
        opening_range_low = None
        session_bar_index = None
        if self.style in _SESSION_INTRADAY_STYLES:
            session_labels, session_bar_positions = self._intraday_session_labels(
                [candle.timestamp for candle in candles]
            )
            session_keys = pd.Series(session_labels, index=frame.index, dtype="object")
            active_mask = session_keys.notna()
            typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
            active_volume = frame["volume"].where(active_mask)
            price_volume = (typical_price * frame["volume"]).where(active_mask)
            cumulative_volume = active_volume.groupby(session_keys).cumsum().replace(0.0, math.nan)
            cumulative_price_volume = price_volume.groupby(session_keys).cumsum()
            session_vwap = cumulative_price_volume / cumulative_volume
            relative_volume = active_volume / active_volume.rolling(
                self.config.volume_window,
                min_periods=self.config.volume_window,
            ).mean().shift(1)
            relative_volume = relative_volume.where(active_mask)
            session_bar_index = pd.Series(session_bar_positions, index=frame.index, dtype="float64")
            opening_range_high = frame["high"].copy() * math.nan
            opening_range_low = frame["low"].copy() * math.nan
            opening_range_bars = max(1, self.config.opening_range_bars)
            active_session_keys = [value for value in session_keys.dropna().drop_duplicates().tolist() if value]
            for session_key in active_session_keys:
                session_frame = frame.loc[session_keys == session_key]
                if len(session_frame.index) <= opening_range_bars:
                    continue
                range_high = float(session_frame["high"].iloc[:opening_range_bars].max())
                range_low = float(session_frame["low"].iloc[:opening_range_bars].min())
                tradable_index = session_frame.index[opening_range_bars:]
                opening_range_high.loc[tradable_index] = range_high
                opening_range_low.loc[tradable_index] = range_low

        series: list[dict[str, float] | None] = []
        for index in range(len(candles)):
            values = {
                "ema_fast": float(ema_fast.iloc[index]) if ema_fast is not None else math.nan,
                "ema_slow": float(ema_slow.iloc[index]) if ema_slow is not None else math.nan,
                "rsi": float(rsi.iloc[index]) if rsi is not None else math.nan,
                "adx": float(adx.iloc[index, 0]),
                "dmp": float(adx.iloc[index, 1]),
                "dmn": float(adx.iloc[index, 2]),
            }
            if self.style in _MACD_STYLES and macd is not None:
                values.update(
                    {
                        "macd": float(macd.iloc[index, 0]),
                        "macd_hist": float(macd.iloc[index, 1]),
                        "macd_signal": float(macd.iloc[index, 2]),
                    }
                )
            if self.style in _DONCHIAN_STYLES and donchian is not None:
                values.update(
                    {
                        "donchian_lower": float(donchian_lower.iloc[index]),
                        "donchian_upper": float(donchian_upper.iloc[index]),
                    }
                )
            if self.style in _SESSION_INTRADAY_STYLES and session_vwap is not None:
                values.update(
                    {
                        "session_vwap": float(session_vwap.iloc[index]),
                        "relative_volume": float(relative_volume.iloc[index]),
                        "opening_range_high": float(opening_range_high.iloc[index]),
                        "opening_range_low": float(opening_range_low.iloc[index]),
                        "session_bar_index": float(session_bar_index.iloc[index]),
                    }
                )
            if any(math.isnan(value) for value in values.values()):
                series.append(None)
                continue
            series.append(values)
        return series
