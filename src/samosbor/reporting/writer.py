from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from ..domain import BacktestResult, PortfolioState


def write_backtest_report(
    output_dir: Path,
    result: BacktestResult,
    summary: dict[str, float | int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_trades(output_dir / "trades.csv", result)
    _write_equity(output_dir / "equity.csv", result)
    _write_jsonl(output_dir / "events.jsonl", result.events)
    write_portfolio_snapshot(output_dir / "portfolio.json", result.portfolio)


def write_portfolio_snapshot(path: Path, portfolio: PortfolioState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portfolio.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_trades(path: Path, result: BacktestResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
            ]
        )
        for trade in result.trades:
            writer.writerow(
                [
                    trade.symbol,
                    trade.direction.value,
                    trade.quantity_lots,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    trade.entry_price,
                    trade.exit_price,
                    trade.gross_pnl,
                    trade.net_pnl,
                    trade.reason,
                    trade.signal_strength,
                ]
            )


def _write_equity(path: Path, result: BacktestResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "equity", "cash", "gross_exposure"])
        for point in result.equity_curve:
            writer.writerow(
                [
                    point.timestamp.isoformat(),
                    point.equity,
                    point.cash,
                    point.gross_exposure,
                ]
            )


def _write_jsonl(path: Path, events: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False))
            handle.write("\n")
