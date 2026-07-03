from __future__ import annotations

import math
from collections import OrderedDict

from ..domain import BacktestResult
from ..strategy.trend_following import bars_per_year_for_timeframe


def compute_summary(result: BacktestResult, *, timeframe: str) -> dict[str, float | int]:
    equity_values = [point.equity for point in result.equity_curve]
    if not equity_values:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_monthly_return_pct": 0.0,
            "trades": 0,
        }

    starting_equity = equity_values[0]
    ending_equity = equity_values[-1]
    total_return_pct = ((ending_equity / starting_equity) - 1) * 100 if starting_equity else 0.0
    max_drawdown_pct = _max_drawdown_pct(equity_values)
    sharpe_ratio = _sharpe_ratio(equity_values, bars_per_year_for_timeframe(timeframe))
    wins = [trade for trade in result.trades if trade.net_pnl > 0]
    losses = [trade for trade in result.trades if trade.net_pnl < 0]
    gross_profit = sum(trade.net_pnl for trade in wins)
    gross_loss = abs(sum(trade.net_pnl for trade in losses))
    profit_factor = gross_profit / gross_loss if gross_loss else 0.0
    win_rate_pct = (len(wins) / len(result.trades) * 100) if result.trades else 0.0
    avg_monthly_return_pct = _average_monthly_return_pct(result)

    return {
        "total_return_pct": round(total_return_pct, 3),
        "max_drawdown_pct": round(max_drawdown_pct, 3),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "win_rate_pct": round(win_rate_pct, 3),
        "profit_factor": round(profit_factor, 3),
        "avg_monthly_return_pct": round(avg_monthly_return_pct, 3),
        "trades": len(result.trades),
        "ending_equity_rub": round(ending_equity, 2),
        "realized_pnl_rub": round(result.portfolio.realized_pnl, 2),
    }


def _max_drawdown_pct(values: list[float]) -> float:
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, 1 - (value / peak))
    return max_drawdown * 100


def _sharpe_ratio(values: list[float], bars_per_year: int) -> float:
    if len(values) < 3:
        return 0.0
    returns = []
    for previous, current in zip(values[:-1], values[1:]):
        if previous <= 0:
            continue
        returns.append((current / previous) - 1.0)
    if len(returns) < 2:
        return 0.0
    average = sum(returns) / len(returns)
    variance = sum((value - average) ** 2 for value in returns) / (len(returns) - 1)
    if variance <= 0:
        return 0.0
    return average / math.sqrt(variance) * math.sqrt(bars_per_year)


def _average_monthly_return_pct(result: BacktestResult) -> float:
    monthly_equity: OrderedDict[str, float] = OrderedDict()
    for point in result.equity_curve:
        monthly_equity[point.timestamp.strftime("%Y-%m")] = point.equity
    values = list(monthly_equity.values())
    if len(values) < 2:
        return 0.0
    monthly_returns = []
    for previous, current in zip(values[:-1], values[1:]):
        if previous <= 0:
            continue
        monthly_returns.append((current / previous - 1.0) * 100)
    if not monthly_returns:
        return 0.0
    return sum(monthly_returns) / len(monthly_returns)
