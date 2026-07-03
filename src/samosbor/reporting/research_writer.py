from __future__ import annotations

import csv
import json
from pathlib import Path


def write_optimizer_report(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimizer.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = payload.get("top_candidates", [])
    with (output_dir / "optimizer_top_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "score",
                "symbols",
                "style",
                "fast_window",
                "slow_window",
                "require_breakout",
                "atr_stop_multiple",
                "reward_to_risk",
                "min_trend_strength",
                "adx_min",
                "rsi_long_max",
                "rsi_short_min",
                "total_return_pct",
                "max_drawdown_pct",
                "sharpe_ratio",
                "avg_monthly_return_pct",
                "profit_factor",
                "trades",
            ]
        )
        for row in rows:
            summary = row["summary"]
            writer.writerow(
                [
                    row["score"],
                    ",".join(row["symbols"]),
                    row["style"],
                    row["fast_window"],
                    row["slow_window"],
                    row["require_breakout"],
                    row["atr_stop_multiple"],
                    row["reward_to_risk"],
                    row["min_trend_strength"],
                    row["adx_min"],
                    row.get("rsi_long_max", ""),
                    row.get("rsi_short_min", ""),
                    summary["total_return_pct"],
                    summary["max_drawdown_pct"],
                    summary["sharpe_ratio"],
                    summary["avg_monthly_return_pct"],
                    summary["profit_factor"],
                    summary["trades"],
                ]
            )


def write_monte_carlo_report(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monte_carlo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_walk_forward_report(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "walk_forward.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = payload.get("folds", [])
    with (output_dir / "walk_forward_folds.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "fold_index",
                "train_months",
                "test_months",
                "symbols",
                "style",
                "fast_window",
                "slow_window",
                "require_breakout",
                "atr_stop_multiple",
                "reward_to_risk",
                "min_trend_strength",
                "adx_min",
                "rsi_long_max",
                "rsi_short_min",
                "train_total_return_pct",
                "train_normalized_monthly_return_pct",
                "test_total_return_pct",
                "test_normalized_monthly_return_pct",
                "test_max_drawdown_pct",
                "test_sharpe_ratio",
                "test_trades",
            ]
        )
        for row in rows:
            candidate = row["best_candidate"]
            train_summary = row["train_summary"]
            test_summary = row["test_summary"]
            writer.writerow(
                [
                    row["fold_index"],
                    ",".join(row["train_months"]),
                    ",".join(row["test_months"]),
                    ",".join(candidate["symbols"]),
                    candidate["style"],
                    candidate["fast_window"],
                    candidate["slow_window"],
                    candidate["require_breakout"],
                    candidate["atr_stop_multiple"],
                    candidate["reward_to_risk"],
                    candidate["min_trend_strength"],
                    candidate["adx_min"],
                    candidate.get("rsi_long_max", ""),
                    candidate.get("rsi_short_min", ""),
                    train_summary["total_return_pct"],
                    train_summary["normalized_monthly_return_pct"],
                    test_summary["total_return_pct"],
                    test_summary["normalized_monthly_return_pct"],
                    test_summary["max_drawdown_pct"],
                    test_summary["sharpe_ratio"],
                    test_summary["trades"],
                ]
            )
