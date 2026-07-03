from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations, product

from ..backtest.engine import BacktestEngine
from ..config import BacktestSection, ResearchSection, RiskSection, StrategySection
from ..domain import Candle, Instrument
from ..reporting.metrics import compute_summary
from ..risk.manager import RiskManager
from .targets import effective_target_monthly_return_pct
from ..strategy.trend_following import TrendFollowingStrategy


@dataclass(frozen=True)
class OptimizationCandidate:
    score: float
    symbols: tuple[str, ...]
    style: str
    fast_window: int
    slow_window: int
    require_breakout: bool
    opening_range_bars: int
    rel_volume_threshold: float
    atr_stop_multiple: float
    reward_to_risk: float
    breakeven_trigger_pct: float
    trailing_profit_trigger_rub: float
    trailing_profit_lock_ratio: float
    min_trend_strength: float
    adx_min: float
    rsi_long_max: float
    rsi_short_min: float
    summary: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        return payload


class ParameterOptimizer:
    def __init__(
        self,
        *,
        base_strategy: StrategySection,
        risk: RiskSection,
        backtest: BacktestSection,
        research: ResearchSection,
        timeframe: str,
        slippage_bps: float,
        commission_bps: float,
    ):
        self.base_strategy = base_strategy
        self.risk = risk
        self.backtest = backtest
        self.research = research
        self.timeframe = timeframe
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.target_monthly_return_pct = effective_target_monthly_return_pct(research, backtest)

    def strategy_from_candidate_payload(self, payload: dict[str, object]) -> StrategySection:
        return replace(
            self.base_strategy,
            style=str(payload.get("style", self.base_strategy.style)).strip().lower(),
            fast_window=int(payload.get("fast_window", self.base_strategy.fast_window)),
            slow_window=int(payload.get("slow_window", self.base_strategy.slow_window)),
            require_breakout=bool(payload.get("require_breakout", self.base_strategy.require_breakout)),
            opening_range_bars=int(
                payload.get("opening_range_bars", self.base_strategy.opening_range_bars)
            ),
            rel_volume_threshold=float(
                payload.get("rel_volume_threshold", self.base_strategy.rel_volume_threshold)
            ),
            atr_stop_multiple=float(
                payload.get("atr_stop_multiple", self.base_strategy.atr_stop_multiple)
            ),
            reward_to_risk=float(payload.get("reward_to_risk", self.base_strategy.reward_to_risk)),
            breakeven_trigger_pct=float(
                payload.get("breakeven_trigger_pct", self.base_strategy.breakeven_trigger_pct)
            ),
            trailing_profit_trigger_rub=float(
                payload.get(
                    "trailing_profit_trigger_rub",
                    self.base_strategy.trailing_profit_trigger_rub,
                )
            ),
            trailing_profit_lock_ratio=float(
                payload.get(
                    "trailing_profit_lock_ratio",
                    self.base_strategy.trailing_profit_lock_ratio,
                )
            ),
            min_trend_strength=float(
                payload.get("min_trend_strength", self.base_strategy.min_trend_strength)
            ),
            adx_min=float(payload.get("adx_min", self.base_strategy.adx_min)),
            rsi_long_max=float(payload.get("rsi_long_max", self.base_strategy.rsi_long_max)),
            rsi_short_min=float(payload.get("rsi_short_min", self.base_strategy.rsi_short_min)),
        )

    def run(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        instruments_by_symbol: dict[str, Instrument],
    ) -> dict[str, object]:
        symbols = sorted(instruments_by_symbol)
        subset_min = max(1, self.research.subset_min_size)
        subset_max = max(subset_min, min(self.research.subset_max_size, len(symbols)))
        candidates: list[OptimizationCandidate] = []
        styles = self.research.strategy_styles or [self.base_strategy.style]
        breakout_values = (
            self.research.require_breakout_values or [self.base_strategy.require_breakout]
        )

        for subset_size in range(subset_min, subset_max + 1):
            for subset in combinations(symbols, subset_size):
                subset_candles = {symbol: candles_by_symbol[symbol] for symbol in subset}
                subset_instruments = {
                    symbol: instruments_by_symbol[symbol] for symbol in subset
                }
                for style in styles:
                    normalized_style = style.strip().lower()
                    require_breakout_values = (
                        breakout_values
                        if normalized_style
                        in {"sma_breakout", "ema_adx_macd", "adx_regime_hybrid", "vwap_relvol_orb"}
                        else [self.base_strategy.require_breakout]
                    )
                    opening_range_bars_values = (
                        self.research.opening_range_bars_values
                        if normalized_style == "vwap_relvol_orb"
                        else [self.base_strategy.opening_range_bars]
                    )
                    rel_volume_threshold_values = (
                        self.research.rel_volume_threshold_values
                        if normalized_style == "vwap_relvol_orb"
                        else [self.base_strategy.rel_volume_threshold]
                    )
                    adx_values = (
                        self.research.adx_min_values
                        if normalized_style
                        in {"ema_adx_macd", "ema_adx_donchian", "adx_regime_hybrid", "vwap_relvol_orb"}
                        else [self.base_strategy.adx_min]
                    )
                    rsi_long_max_values = (
                        self.research.rsi_long_max_values
                        if normalized_style in {"rsi_mean_reversion", "adx_regime_hybrid"}
                        else [self.base_strategy.rsi_long_max]
                    )
                    rsi_short_min_values = (
                        self.research.rsi_short_min_values
                        if normalized_style in {"rsi_mean_reversion", "adx_regime_hybrid"}
                        else [self.base_strategy.rsi_short_min]
                    )
                    for (
                        fast,
                        slow,
                        require_breakout,
                        opening_range_bars,
                        rel_volume_threshold,
                        atr_mult,
                        rr,
                        breakeven_trigger_pct,
                        trailing_trigger_rub,
                        trailing_lock_ratio,
                        trend_strength,
                        adx_min,
                        rsi_long_max,
                        rsi_short_min,
                    ) in product(
                        self.research.fast_windows,
                        self.research.slow_windows,
                        require_breakout_values,
                        opening_range_bars_values,
                        rel_volume_threshold_values,
                        self.research.atr_stop_multipliers,
                        self.research.reward_to_risk_values,
                        self.research.breakeven_trigger_pct_values,
                        self.research.trailing_profit_trigger_rub_values,
                        self.research.trailing_profit_lock_ratio_values,
                        self.research.trend_strength_values,
                        adx_values,
                        rsi_long_max_values,
                        rsi_short_min_values,
                    ):
                        if fast >= slow:
                            continue

                        strategy = replace(
                            self.base_strategy,
                            style=normalized_style,
                            fast_window=fast,
                            slow_window=slow,
                            require_breakout=require_breakout,
                            opening_range_bars=opening_range_bars,
                            rel_volume_threshold=rel_volume_threshold,
                            atr_stop_multiple=atr_mult,
                            reward_to_risk=rr,
                            breakeven_trigger_pct=breakeven_trigger_pct,
                            trailing_profit_trigger_rub=trailing_trigger_rub,
                            trailing_profit_lock_ratio=trailing_lock_ratio,
                            min_trend_strength=trend_strength,
                            adx_min=adx_min,
                            rsi_long_max=rsi_long_max,
                            rsi_short_min=rsi_short_min,
                        )
                        engine = BacktestEngine(
                            strategy=TrendFollowingStrategy(strategy, timeframe=self.timeframe),
                            risk_manager=RiskManager(self.risk),
                            backtest=self.backtest,
                            slippage_bps=self.slippage_bps,
                            commission_bps=self.commission_bps,
                        )
                        result = engine.run_with_instruments(subset_candles, subset_instruments)
                        summary = compute_summary(result, timeframe=self.timeframe)
                        candidate = OptimizationCandidate(
                            score=self._score(summary),
                            symbols=subset,
                            style=normalized_style,
                            fast_window=fast,
                            slow_window=slow,
                            require_breakout=require_breakout,
                            opening_range_bars=opening_range_bars,
                            rel_volume_threshold=rel_volume_threshold,
                            atr_stop_multiple=atr_mult,
                            reward_to_risk=rr,
                            breakeven_trigger_pct=breakeven_trigger_pct,
                            trailing_profit_trigger_rub=trailing_trigger_rub,
                            trailing_profit_lock_ratio=trailing_lock_ratio,
                            min_trend_strength=trend_strength,
                            adx_min=adx_min,
                            rsi_long_max=rsi_long_max,
                            rsi_short_min=rsi_short_min,
                            summary=summary,
                        )
                        candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)
        top = candidates[: self.research.top_n]
        best = top[0] if top else None
        return {
            "evaluated_candidates": len(candidates),
            "best_candidate": best.to_dict() if best else None,
            "top_candidates": [candidate.to_dict() for candidate in top],
        }

    def _score(self, summary: dict[str, float | int]) -> float:
        total_return = float(summary["total_return_pct"])
        max_drawdown = float(summary["max_drawdown_pct"])
        sharpe = float(summary["sharpe_ratio"])
        avg_monthly = float(summary["avg_monthly_return_pct"])
        profit_factor = float(summary["profit_factor"])
        trades = int(summary["trades"])
        target_gap = max(0.0, self.target_monthly_return_pct - avg_monthly)
        trade_penalty = max(0, self.research.min_trades - trades) * 0.6
        return (
            total_return
            + avg_monthly * 2.0
            + sharpe * 3.0
            + profit_factor
            - max_drawdown * 0.9
            - target_gap * 0.8
            - trade_penalty
        )
