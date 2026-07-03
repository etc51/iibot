from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from ..domain import BacktestResult


@dataclass(frozen=True)
class MonteCarloSummary:
    iterations: int
    horizon_months: int
    average_total_return_pct: float
    median_total_return_pct: float
    probability_positive_pct: float
    probability_target_monthly_return_pct: float
    average_max_drawdown_pct: float
    p05_total_return_pct: float
    p95_total_return_pct: float
    source_months: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "iterations": self.iterations,
            "horizon_months": self.horizon_months,
            "average_total_return_pct": round(self.average_total_return_pct, 3),
            "median_total_return_pct": round(self.median_total_return_pct, 3),
            "probability_positive_pct": round(self.probability_positive_pct, 3),
            "probability_target_monthly_return_pct": round(
                self.probability_target_monthly_return_pct, 3
            ),
            "average_max_drawdown_pct": round(self.average_max_drawdown_pct, 3),
            "p05_total_return_pct": round(self.p05_total_return_pct, 3),
            "p95_total_return_pct": round(self.p95_total_return_pct, 3),
            "source_months": self.source_months,
        }


def monthly_returns_from_result(result: BacktestResult) -> list[float]:
    monthly_equity: dict[str, tuple[float, float]] = {}
    for point in result.equity_curve:
        key = point.timestamp.strftime("%Y-%m")
        if key not in monthly_equity:
            monthly_equity[key] = (point.equity, point.equity)
        else:
            first, _ = monthly_equity[key]
            monthly_equity[key] = (first, point.equity)

    returns = []
    for key in sorted(monthly_equity):
        first, last = monthly_equity[key]
        if first <= 0:
            continue
        returns.append((last / first) - 1.0)
    return returns


class MonteCarloSimulator:
    def __init__(
        self,
        *,
        iterations: int,
        horizon_months: int,
        target_monthly_return_pct: float,
        seed: int,
    ):
        self.iterations = iterations
        self.horizon_months = horizon_months
        self.target_monthly_return_pct = target_monthly_return_pct
        self.seed = seed

    def run(self, result: BacktestResult) -> dict[str, object]:
        monthly_returns = monthly_returns_from_result(result)
        if not monthly_returns:
            raise ValueError("Not enough monthly data for Monte Carlo simulation.")

        rng = random.Random(self.seed)
        total_returns: list[float] = []
        avg_monthlies: list[float] = []
        max_drawdowns: list[float] = []

        for _ in range(self.iterations):
            sampled = [rng.choice(monthly_returns) for _ in range(self.horizon_months)]
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for monthly_return in sampled:
                equity *= 1 + monthly_return
                peak = max(peak, equity)
                if peak > 0:
                    max_dd = max(max_dd, 1 - (equity / peak))
            total_returns.append((equity - 1.0) * 100)
            avg_monthlies.append(sum(sampled) / len(sampled) * 100)
            max_drawdowns.append(max_dd * 100)

        summary = MonteCarloSummary(
            iterations=self.iterations,
            horizon_months=self.horizon_months,
            average_total_return_pct=_mean(total_returns),
            median_total_return_pct=_percentile(total_returns, 0.5),
            probability_positive_pct=_probability(value > 0 for value in total_returns),
            probability_target_monthly_return_pct=_probability(
                value >= self.target_monthly_return_pct for value in avg_monthlies
            ),
            average_max_drawdown_pct=_mean(max_drawdowns),
            p05_total_return_pct=_percentile(total_returns, 0.05),
            p95_total_return_pct=_percentile(total_returns, 0.95),
            source_months=len(monthly_returns),
        )
        return {
            "summary": summary.to_dict(),
            "base_monthly_returns_pct": [round(value * 100, 3) for value in monthly_returns],
        }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _probability(condition_results: Iterable[bool]) -> float:
    results = list(condition_results)
    if not results:
        return 0.0
    return sum(1 for value in results if value) / len(results) * 100


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]
