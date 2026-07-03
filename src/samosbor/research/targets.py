from __future__ import annotations

from ..config import BacktestSection, ResearchSection


def trading_days_per_month(research: ResearchSection) -> int:
    return max(1, int(research.trading_days_per_month))


def effective_target_daily_profit_rub(
    research: ResearchSection,
    backtest: BacktestSection,
) -> float:
    if research.target_daily_profit_rub > 0:
        return float(research.target_daily_profit_rub)
    return effective_target_monthly_profit_rub(research, backtest) / float(
        trading_days_per_month(research)
    )


def effective_target_monthly_profit_rub(
    research: ResearchSection,
    backtest: BacktestSection,
) -> float:
    if research.target_daily_profit_rub > 0:
        return float(research.target_daily_profit_rub) * float(trading_days_per_month(research))
    if research.target_monthly_profit_rub > 0:
        return float(research.target_monthly_profit_rub)
    return float(backtest.initial_cash) * float(research.target_monthly_return_pct) / 100.0


def effective_target_monthly_return_pct(
    research: ResearchSection,
    backtest: BacktestSection,
) -> float:
    monthly_profit = effective_target_monthly_profit_rub(research, backtest)
    if monthly_profit > 0 and backtest.initial_cash > 0:
        return monthly_profit / float(backtest.initial_cash) * 100.0
    return float(research.target_monthly_return_pct)


def effective_target_daily_return_pct(
    research: ResearchSection,
    backtest: BacktestSection,
) -> float:
    if backtest.initial_cash <= 0:
        return 0.0
    return effective_target_daily_profit_rub(research, backtest) / float(backtest.initial_cash) * 100.0


def effective_target_payload(
    research: ResearchSection,
    backtest: BacktestSection,
) -> dict[str, float | int]:
    days = trading_days_per_month(research)
    daily_profit = effective_target_daily_profit_rub(research, backtest)
    monthly_profit = effective_target_monthly_profit_rub(research, backtest)
    monthly_return_pct = effective_target_monthly_return_pct(research, backtest)
    daily_return_pct = effective_target_daily_return_pct(research, backtest)
    return {
        "daily_profit_rub": round(daily_profit, 2),
        "daily_return_pct": round(daily_return_pct, 3),
        "monthly_profit_rub": round(monthly_profit, 2),
        "monthly_return_pct": round(monthly_return_pct, 3),
        "trading_days_per_month": days,
    }


def render_target_label(target: dict[str, object]) -> str:
    return (
        f"{target['daily_profit_rub']} RUB/day "
        f"(~{target['monthly_profit_rub']} RUB/month, "
        f"{target['trading_days_per_month']} trading days/month, "
        f"{target['monthly_return_pct']}%)"
    )
