from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..backtest.engine import BacktestEngine
from ..config import BacktestSection, ResearchSection, RiskSection, StrategySection
from ..domain import BacktestResult, Candle, Instrument
from ..reporting.metrics import compute_summary
from ..risk.manager import RiskManager
from ..strategy.trend_following import TrendFollowingStrategy
from .optimizer import ParameterOptimizer
from .targets import (
    effective_target_monthly_profit_rub,
    effective_target_monthly_return_pct,
    effective_target_payload,
)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_months: tuple[str, ...]
    test_months: tuple[str, ...]
    best_candidate: dict[str, object]
    train_summary: dict[str, float | int]
    test_summary: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_months": list(self.train_months),
            "test_months": list(self.test_months),
            "best_candidate": self.best_candidate,
            "train_summary": self.train_summary,
            "test_summary": self.test_summary,
        }


class WalkForwardValidator:
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
        self.target = effective_target_payload(research, backtest)
        self.target_monthly_return_pct = effective_target_monthly_return_pct(research, backtest)
        self.target_monthly_profit_rub = effective_target_monthly_profit_rub(research, backtest)

    def run(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        instruments_by_symbol: dict[str, Instrument],
    ) -> dict[str, object]:
        grouped = _group_candles_by_month(candles_by_symbol)
        available_months = _available_months(grouped)
        train_months = max(1, self.research.walk_forward_train_months)
        test_months = max(1, self.research.walk_forward_test_months)
        step_months = max(1, self.research.walk_forward_step_months)
        minimum_months = train_months + test_months
        if len(available_months) < minimum_months:
            raise ValueError(
                "Not enough monthly history for walk-forward validation: "
                f"need at least {minimum_months} months, got {len(available_months)}."
            )

        folds: list[WalkForwardFold] = []
        skipped_folds = 0

        for start_index in range(0, len(available_months) - minimum_months + 1, step_months):
            train_keys = tuple(available_months[start_index : start_index + train_months])
            test_keys = tuple(
                available_months[
                    start_index + train_months : start_index + train_months + test_months
                ]
            )
            train_bundle = _slice_grouped_candles(grouped, train_keys)
            test_bundle = _slice_grouped_candles(grouped, test_keys)
            eligible_symbols = [
                symbol
                for symbol in sorted(instruments_by_symbol)
                if len(train_bundle.get(symbol, [])) >= self.backtest.warmup_bars
                and len(test_bundle.get(symbol, [])) > 0
            ]
            if not eligible_symbols:
                skipped_folds += 1
                continue

            train_candles = {symbol: train_bundle[symbol] for symbol in eligible_symbols}
            test_candles = {symbol: test_bundle[symbol] for symbol in eligible_symbols}
            eligible_instruments = {
                symbol: instruments_by_symbol[symbol] for symbol in eligible_symbols
            }

            optimizer = ParameterOptimizer(
                base_strategy=self.base_strategy,
                risk=self.risk,
                backtest=self.backtest,
                research=self.research,
                timeframe=self.timeframe,
                slippage_bps=self.slippage_bps,
                commission_bps=self.commission_bps,
            )
            optimization = optimizer.run(train_candles, eligible_instruments)
            best_candidate = optimization.get("best_candidate")
            if not best_candidate:
                skipped_folds += 1
                continue

            selected_symbols = [
                symbol
                for symbol in best_candidate["symbols"]
                if symbol in test_candles and test_candles[symbol]
            ]
            if not selected_symbols:
                skipped_folds += 1
                continue

            strategy = optimizer.strategy_from_candidate_payload(best_candidate)
            combined_candles = {
                symbol: [*train_candles[symbol], *test_candles[symbol]] for symbol in selected_symbols
            }
            selected_instruments = {
                symbol: eligible_instruments[symbol] for symbol in selected_symbols
            }
            test_start_at = min(
                candle.timestamp
                for symbol in selected_symbols
                for candle in test_candles[symbol]
            )
            engine = BacktestEngine(
                strategy=TrendFollowingStrategy(strategy, timeframe=self.timeframe),
                risk_manager=RiskManager(self.risk),
                backtest=self.backtest,
                slippage_bps=self.slippage_bps,
                commission_bps=self.commission_bps,
            )
            combined_result = engine.run_with_instruments(
                combined_candles,
                selected_instruments,
                trade_start_at=test_start_at,
            )
            test_result = _trim_backtest_result(combined_result, test_start_at)
            test_summary = compute_summary(test_result, timeframe=self.timeframe)
            test_summary["normalized_monthly_return_pct"] = round(
                _normalized_monthly_return_pct(
                    float(test_summary["total_return_pct"]),
                    len(test_keys),
                ),
                3,
            )
            train_summary = dict(best_candidate["summary"])
            train_summary["normalized_monthly_return_pct"] = round(
                _normalized_monthly_return_pct(
                    float(train_summary["total_return_pct"]),
                    len(train_keys),
                ),
                3,
            )
            folds.append(
                WalkForwardFold(
                    fold_index=len(folds) + 1,
                    train_months=train_keys,
                    test_months=test_keys,
                    best_candidate=best_candidate,
                    train_summary=train_summary,
                    test_summary=test_summary,
                )
            )

        if not folds:
            raise ValueError("Walk-forward validation produced no evaluable folds.")

        return {
            "config": {
                "train_months": train_months,
                "test_months": test_months,
                "step_months": step_months,
                "target_daily_profit_rub": self.target["daily_profit_rub"],
                "target_monthly_return_pct": self.target_monthly_return_pct,
                "target_monthly_profit_rub": self.target_monthly_profit_rub,
                "trading_days_per_month": self.target["trading_days_per_month"],
            },
            "summary": _walk_forward_summary(
                folds,
                target_monthly_return_pct=self.target_monthly_return_pct,
            ),
            "available_months": available_months,
            "skipped_folds": skipped_folds,
            "folds": [fold.to_dict() for fold in folds],
        }


def _group_candles_by_month(
    candles_by_symbol: dict[str, list[Candle]],
) -> dict[str, dict[str, list[Candle]]]:
    grouped: dict[str, dict[str, list[Candle]]] = {}
    for symbol, candles in candles_by_symbol.items():
        symbol_months: dict[str, list[Candle]] = {}
        for candle in candles:
            key = candle.timestamp.strftime("%Y-%m")
            symbol_months.setdefault(key, []).append(candle)
        grouped[symbol] = symbol_months
    return grouped


def _available_months(grouped: dict[str, dict[str, list[Candle]]]) -> list[str]:
    months = {month for symbol_months in grouped.values() for month in symbol_months}
    return sorted(months)


def _slice_grouped_candles(
    grouped: dict[str, dict[str, list[Candle]]],
    months: tuple[str, ...],
) -> dict[str, list[Candle]]:
    sliced: dict[str, list[Candle]] = {}
    for symbol, symbol_months in grouped.items():
        candles: list[Candle] = []
        for month in months:
            candles.extend(symbol_months.get(month, []))
        sliced[symbol] = candles
    return sliced


def _trim_backtest_result(result: BacktestResult, start_at) -> BacktestResult:
    equity_curve = [point for point in result.equity_curve if point.timestamp >= start_at]
    trades = [trade for trade in result.trades if trade.entry_time >= start_at]
    return BacktestResult(
        portfolio=result.portfolio,
        trades=trades,
        equity_curve=equity_curve,
        events=[],
    )


def _normalized_monthly_return_pct(total_return_pct: float, months: int) -> float:
    if months <= 0:
        return 0.0
    gross_return = 1.0 + total_return_pct / 100.0
    if gross_return <= 0:
        return -100.0
    return ((gross_return ** (1 / months)) - 1.0) * 100.0


def _walk_forward_summary(
    folds: list[WalkForwardFold],
    *,
    target_monthly_return_pct: float,
) -> dict[str, float | int]:
    test_total_returns = [float(fold.test_summary["total_return_pct"]) for fold in folds]
    test_drawdowns = [float(fold.test_summary["max_drawdown_pct"]) for fold in folds]
    test_sharpes = [float(fold.test_summary["sharpe_ratio"]) for fold in folds]
    test_monthlies = [
        float(fold.test_summary["normalized_monthly_return_pct"]) for fold in folds
    ]
    train_total_returns = [float(fold.train_summary["total_return_pct"]) for fold in folds]
    train_monthlies = [
        float(fold.train_summary["normalized_monthly_return_pct"]) for fold in folds
    ]
    compounded_equity = 1.0
    for value in test_total_returns:
        compounded_equity *= 1.0 + value / 100.0

    return {
        "folds_evaluated": len(folds),
        "average_train_total_return_pct": round(_mean(train_total_returns), 3),
        "average_train_normalized_monthly_return_pct": round(_mean(train_monthlies), 3),
        "average_test_total_return_pct": round(_mean(test_total_returns), 3),
        "median_test_total_return_pct": round(median(test_total_returns), 3),
        "average_test_max_drawdown_pct": round(_mean(test_drawdowns), 3),
        "average_test_sharpe_ratio": round(_mean(test_sharpes), 3),
        "average_test_normalized_monthly_return_pct": round(_mean(test_monthlies), 3),
        "probability_positive_pct": round(
            _probability(value > 0 for value in test_total_returns),
            3,
        ),
        "probability_target_monthly_return_pct": round(
            _probability(value >= target_monthly_return_pct for value in test_monthlies),
            3,
        ),
        "compounded_test_return_pct": round((compounded_equity - 1.0) * 100.0, 3),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _probability(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items) * 100
