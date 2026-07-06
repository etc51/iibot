from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Iterable
from zoneinfo import ZoneInfo

from ..domain import PortfolioState, Position, TradeRecord
from ..runtime_metadata import with_runtime_metadata


def build_paper_report_payload(
    portfolio: PortfolioState,
    trades: list[TradeRecord],
    *,
    timezone_name: str,
    report_date: date | None = None,
    days: int = 1,
) -> dict[str, object]:
    if days < 1:
        raise ValueError("days must be >= 1")

    timezone = ZoneInfo(timezone_name)
    anchor_date = report_date or datetime.now(timezone).date()
    start_at, end_at = _date_window(anchor_date, days=days, timezone=timezone)
    previous_start_at = start_at - timedelta(days=days)
    previous_end_at = start_at

    current_trades = _select_trades(trades, start_at=start_at, end_at=end_at, timezone=timezone)
    previous_trades = _select_trades(
        trades,
        start_at=previous_start_at,
        end_at=previous_end_at,
        timezone=timezone,
    )

    current_summary = _trade_summary(current_trades)
    previous_summary = _trade_summary(previous_trades)
    current_portfolio = _portfolio_snapshot(portfolio)
    current_positions = _open_positions(portfolio.positions.values(), timezone=timezone)

    return with_runtime_metadata({
        "period": {
            "timezone": timezone_name,
            "days": days,
            "report_date": anchor_date.isoformat(),
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
        },
        "portfolio": current_portfolio,
        "summary": current_summary,
        "comparison_to_previous_window": {
            "days": days,
            "summary": previous_summary,
            "delta": _summary_delta(current_summary, previous_summary),
        },
        "closed_trades": _trade_rows(current_trades),
        "symbol_breakdown": _symbol_breakdown(current_trades),
        "entry_hour_breakdown": _entry_hour_breakdown(current_trades, timezone=timezone),
        "exit_reason_breakdown": _exit_reason_breakdown(current_trades),
        "best_trades": _trade_rows(sorted(current_trades, key=lambda trade: trade.net_pnl, reverse=True)[:3]),
        "worst_trades": _trade_rows(sorted(current_trades, key=lambda trade: trade.net_pnl)[:3]),
        "open_positions": current_positions,
    })


def write_paper_report(output_dir: Path, payload: dict[str, object]) -> None:
    payload = with_runtime_metadata(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_trade_csv(output_dir / "trades.csv", list(payload.get("closed_trades", [])))
    _write_breakdown_csv(output_dir / "symbol_breakdown.csv", list(payload.get("symbol_breakdown", [])))
    _write_breakdown_csv(
        output_dir / "entry_hour_breakdown.csv",
        list(payload.get("entry_hour_breakdown", [])),
    )
    _write_breakdown_csv(
        output_dir / "exit_reason_breakdown.csv",
        list(payload.get("exit_reason_breakdown", [])),
    )
    (output_dir / "open_positions.json").write_text(
        json.dumps(payload.get("open_positions", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(payload), encoding="utf-8")


def _date_window(anchor_date: date, *, days: int, timezone: ZoneInfo) -> tuple[datetime, datetime]:
    start_date = anchor_date - timedelta(days=days - 1)
    start_at = datetime.combine(start_date, time.min, tzinfo=timezone)
    end_at = datetime.combine(anchor_date + timedelta(days=1), time.min, tzinfo=timezone)
    return start_at, end_at


def _select_trades(
    trades: Iterable[TradeRecord],
    *,
    start_at: datetime,
    end_at: datetime,
    timezone: ZoneInfo,
) -> list[TradeRecord]:
    selected: list[TradeRecord] = []
    for trade in trades:
        exit_time = trade.exit_time.astimezone(timezone)
        if start_at <= exit_time < end_at:
            selected.append(trade)
    return selected


def _trade_summary(trades: list[TradeRecord]) -> dict[str, float | int]:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "net_pnl_rub": 0.0,
            "gross_profit_rub": 0.0,
            "gross_loss_rub": 0.0,
            "profit_factor": 0.0,
            "expectancy_rub": 0.0,
            "median_trade_rub": 0.0,
            "avg_win_rub": 0.0,
            "avg_loss_rub": 0.0,
            "payout_ratio": 0.0,
            "best_trade_rub": 0.0,
            "worst_trade_rub": 0.0,
        }

    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in trades if trade.net_pnl < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss else 0.0
    payout_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else 0.0
    net_pnl = sum(trade.net_pnl for trade in trades)

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 3),
        "net_pnl_rub": round(net_pnl, 2),
        "gross_profit_rub": round(gross_profit, 2),
        "gross_loss_rub": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "expectancy_rub": round(net_pnl / len(trades), 2),
        "median_trade_rub": round(float(median(trade.net_pnl for trade in trades)), 2),
        "avg_win_rub": round(avg_win, 2),
        "avg_loss_rub": round(avg_loss, 2),
        "payout_ratio": round(payout_ratio, 3),
        "best_trade_rub": round(max(trade.net_pnl for trade in trades), 2),
        "worst_trade_rub": round(min(trade.net_pnl for trade in trades), 2),
    }


def _summary_delta(
    current_summary: dict[str, float | int],
    previous_summary: dict[str, float | int],
) -> dict[str, float | int]:
    delta: dict[str, float | int] = {}
    numeric_fields = [
        "trades",
        "wins",
        "losses",
        "win_rate_pct",
        "net_pnl_rub",
        "profit_factor",
        "expectancy_rub",
        "avg_win_rub",
        "avg_loss_rub",
        "best_trade_rub",
        "worst_trade_rub",
    ]
    for key in numeric_fields:
        current = current_summary.get(key, 0)
        previous = previous_summary.get(key, 0)
        delta[key] = round(float(current) - float(previous), 3)
    return delta


def _portfolio_snapshot(portfolio: PortfolioState) -> dict[str, float | int | bool]:
    return {
        "cash_rub": round(portfolio.cash, 2),
        "realized_pnl_rub": round(portfolio.realized_pnl, 2),
        "equity_rub": round(portfolio.equity({}), 2),
        "gross_exposure_rub": round(portfolio.gross_exposure({}), 2),
        "margin_reserved_rub": round(portfolio.margin_reserved(), 2),
        "open_positions": len(portfolio.positions),
        "trading_halted": portfolio.trading_halted,
    }


def _open_positions(positions: Iterable[Position], *, timezone: ZoneInfo) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position in positions:
        rows.append(
            {
                "symbol": position.instrument.symbol,
                "direction": position.direction.value,
                "quantity_lots": position.quantity_lots,
                "entry_price": round(position.entry_price, 6),
                "current_price": round(position.current_price, 6),
                "unrealized_pnl_rub": round(position.unrealized_pnl(), 2),
                "margin_requirement_rub": round(position.margin_requirement, 2),
                "stop_price": round(position.stop_price, 6),
                "take_profit": round(position.take_profit, 6),
                "opened_at": position.opened_at.astimezone(timezone).isoformat(),
                "updated_at": position.updated_at.astimezone(timezone).isoformat(),
            }
        )
    return rows


def _symbol_breakdown(trades: list[TradeRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        grouped[trade.symbol].append(trade)
    rows = [
        _group_summary(symbol, group)
        for symbol, group in grouped.items()
    ]
    rows.sort(key=lambda item: float(item["net_pnl_rub"]), reverse=True)
    return rows


def _entry_hour_breakdown(trades: list[TradeRecord], *, timezone: ZoneInfo) -> list[dict[str, object]]:
    grouped: dict[int, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        grouped[trade.entry_time.astimezone(timezone).hour].append(trade)
    rows = []
    for hour, group in grouped.items():
        summary = _group_summary(str(hour), group)
        summary["entry_hour"] = int(summary.pop("group"))
        rows.append(summary)
    rows.sort(key=lambda item: int(item["entry_hour"]))
    return rows


def _exit_reason_breakdown(trades: list[TradeRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        grouped[trade.reason].append(trade)
    rows = []
    for reason, group in grouped.items():
        summary = _group_summary(reason, group)
        summary["reason"] = str(summary.pop("group"))
        rows.append(summary)
    rows.sort(key=lambda item: float(item["net_pnl_rub"]))
    return rows


def _group_summary(group_name: str, trades: list[TradeRecord]) -> dict[str, object]:
    summary = _trade_summary(trades)
    return {
        "group": group_name,
        **summary,
    }


def _trade_rows(trades: list[TradeRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trade in trades:
        rows.append(
            {
                "symbol": trade.symbol,
                "direction": trade.direction.value,
                "quantity_lots": trade.quantity_lots,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "gross_pnl": round(trade.gross_pnl, 2),
                "net_pnl": round(trade.net_pnl, 2),
                "reason": trade.reason,
                "signal_strength": round(trade.signal_strength, 4),
            }
        )
    return rows


def _write_trade_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "direction",
                "quantity_lots",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "gross_pnl",
                "net_pnl",
                "reason",
                "signal_strength",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_breakdown_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(payload: dict[str, object]) -> str:
    period = payload["period"]
    summary = payload["summary"]
    portfolio = payload["portfolio"]
    previous = payload["comparison_to_previous_window"]
    symbols = payload.get("symbol_breakdown", [])[:3]
    bad_hours = sorted(
        payload.get("entry_hour_breakdown", []),
        key=lambda item: float(item["net_pnl_rub"]),
    )[:3]

    lines = [
        "# Samosbor Paper Report",
        "",
        f"- Commit: {payload.get('commit_hash', 'unknown')}",
        f"- Period: {period['start_at']} .. {period['end_at']} ({period['timezone']})",
        f"- Net PnL: {summary['net_pnl_rub']} RUB",
        f"- Trades: {summary['trades']}",
        f"- Win rate: {summary['win_rate_pct']}%",
        f"- Profit factor: {summary['profit_factor']}",
        f"- Expectancy: {summary['expectancy_rub']} RUB/trade",
        f"- Open positions: {portfolio['open_positions']}",
        f"- Margin reserved: {portfolio['margin_reserved_rub']} RUB",
        f"- Vs previous {previous['days']} day window: {previous['delta']['net_pnl_rub']} RUB, {previous['delta']['trades']} trades",
        "",
        "## Top Symbols",
    ]
    if symbols:
        for row in symbols:
            lines.append(
                f"- {row['group']}: {row['net_pnl_rub']} RUB, {row['trades']} trades, win rate {row['win_rate_pct']}%"
            )
    else:
        lines.append("- No closed trades in this window")

    lines.append("")
    lines.append("## Weak Entry Hours")
    if bad_hours:
        for row in bad_hours:
            lines.append(
                f"- {row['entry_hour']:02d}:00: {row['net_pnl_rub']} RUB, {row['trades']} trades"
            )
    else:
        lines.append("- No closed trades in this window")
    lines.append("")
    return "\n".join(lines)
