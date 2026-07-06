from __future__ import annotations

from collections import defaultdict
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..domain import PortfolioState, TradeRecord
from ..reporting.paper_report import build_paper_report_payload
from ..runtime_metadata import with_runtime_metadata


def build_entry_symbol_tuning_payload(
    portfolio: PortfolioState,
    trades: list[TradeRecord],
    *,
    timezone_name: str,
    current_blocked_symbols: list[str],
    current_blocked_long_symbols: list[str] | None = None,
    current_blocked_short_symbols: list[str] | None = None,
    evidence_source: str = "closed-trades",
    report_date: date | None = None,
    lookback_days: int = 45,
    min_trades_per_symbol: int = 4,
    min_trades_per_direction_symbol: int = 4,
    max_symbols_to_block: int = 1,
    max_total_blocked_symbols: int = 4,
    max_symbols_to_unblock: int = 1,
    max_long_symbols_to_block: int = 1,
    max_short_symbols_to_block: int = 1,
    max_total_blocked_long_symbols: int = 4,
    max_total_blocked_short_symbols: int = 4,
    max_long_symbols_to_unblock: int = 1,
    max_short_symbols_to_unblock: int = 1,
    runtime_symbols: list[str] | None = None,
    min_runtime_symbols_for_restrictions: int = 8,
    max_restricted_symbol_share: float = 0.25,
) -> dict[str, object]:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if min_trades_per_symbol < 1:
        raise ValueError("min_trades_per_symbol must be >= 1")
    if min_trades_per_direction_symbol < 1:
        raise ValueError("min_trades_per_direction_symbol must be >= 1")
    if max_symbols_to_block < 0:
        raise ValueError("max_symbols_to_block must be >= 0")
    if max_total_blocked_symbols < 0:
        raise ValueError("max_total_blocked_symbols must be >= 0")
    if max_symbols_to_unblock < 0:
        raise ValueError("max_symbols_to_unblock must be >= 0")
    if max_long_symbols_to_block < 0:
        raise ValueError("max_long_symbols_to_block must be >= 0")
    if max_short_symbols_to_block < 0:
        raise ValueError("max_short_symbols_to_block must be >= 0")
    if max_total_blocked_long_symbols < 0:
        raise ValueError("max_total_blocked_long_symbols must be >= 0")
    if max_total_blocked_short_symbols < 0:
        raise ValueError("max_total_blocked_short_symbols must be >= 0")
    if max_long_symbols_to_unblock < 0:
        raise ValueError("max_long_symbols_to_unblock must be >= 0")
    if max_short_symbols_to_unblock < 0:
        raise ValueError("max_short_symbols_to_unblock must be >= 0")
    if min_runtime_symbols_for_restrictions < 1:
        raise ValueError("min_runtime_symbols_for_restrictions must be >= 1")
    if max_restricted_symbol_share <= 0 or max_restricted_symbol_share > 1:
        raise ValueError("max_restricted_symbol_share must be within (0, 1]")

    report = build_paper_report_payload(
        portfolio,
        trades,
        timezone_name=timezone_name,
        report_date=report_date,
        days=lookback_days,
    )
    closed_trade_rows = _closed_trade_rows_for_window(
        trades,
        timezone_name=timezone_name,
        start_at_iso=str(report["period"]["start_at"]),
        end_at_iso=str(report["period"]["end_at"]),
    )
    rows = _symbol_breakdown(closed_trade_rows)
    direction_rows = _symbol_direction_breakdown(closed_trade_rows)
    current_set = _normalized_symbol_set(current_blocked_symbols)
    current_long_set = _normalized_symbol_set(current_blocked_long_symbols or [])
    current_short_set = _normalized_symbol_set(current_blocked_short_symbols or [])
    runtime_set = _normalized_symbol_set(runtime_symbols or [])
    runtime_count = len(runtime_set)
    removals = [
        str(row["group"]).strip().upper()
        for row in _recovery_candidates(
            rows,
            current_set=current_set,
            min_trades=min_trades_per_symbol,
        )[:max_symbols_to_unblock]
    ]
    effective_current_set = current_set - set(removals)

    candidates = [
        row
        for row in rows
        if str(row["group"]).strip().upper() not in effective_current_set
        and int(row["trades"]) >= min_trades_per_symbol
        and float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])) < 0
        and float(row.get("raw_expectancy_rub", row["expectancy_rub"])) < 0
        and float(row.get("raw_profit_factor", row["profit_factor"])) < 1.0
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])),
            float(row.get("raw_expectancy_rub", row["expectancy_rub"])),
            float(row.get("raw_profit_factor", row["profit_factor"])),
        )
    )

    available_slots = max(0, max_total_blocked_symbols - len(effective_current_set))
    additions = [
        str(row["group"]).strip().upper()
        for row in candidates[: min(max_symbols_to_block, available_slots)]
    ]
    proposed_blocked_symbols = sorted(effective_current_set | set(additions))
    globally_blocked_symbols = set(proposed_blocked_symbols)
    long_removals = [
        str(row["symbol"]).strip().upper()
        for row in _direction_recovery_candidates(
            direction_rows,
            direction="long",
            current_set=current_long_set,
            globally_blocked_symbols=globally_blocked_symbols,
            min_trades=min_trades_per_direction_symbol,
        )[:max_long_symbols_to_unblock]
    ]
    short_removals = [
        str(row["symbol"]).strip().upper()
        for row in _direction_recovery_candidates(
            direction_rows,
            direction="short",
            current_set=current_short_set,
            globally_blocked_symbols=globally_blocked_symbols,
            min_trades=min_trades_per_direction_symbol,
        )[:max_short_symbols_to_unblock]
    ]
    effective_current_long_set = current_long_set - set(long_removals)
    effective_current_short_set = current_short_set - set(short_removals)

    long_candidates = _direction_candidates(
        direction_rows,
        direction="long",
        current_set=effective_current_long_set,
        globally_blocked_symbols=globally_blocked_symbols,
        min_trades=min_trades_per_direction_symbol,
    )
    short_candidates = _direction_candidates(
        direction_rows,
        direction="short",
        current_set=effective_current_short_set,
        globally_blocked_symbols=globally_blocked_symbols,
        min_trades=min_trades_per_direction_symbol,
    )

    available_long_slots = max(0, max_total_blocked_long_symbols - len(effective_current_long_set))
    available_short_slots = max(0, max_total_blocked_short_symbols - len(effective_current_short_set))
    long_additions = [
        str(row["symbol"]).strip().upper()
        for row in long_candidates[: min(max_long_symbols_to_block, available_long_slots)]
    ]
    short_additions = [
        str(row["symbol"]).strip().upper()
        for row in short_candidates[: min(max_short_symbols_to_block, available_short_slots)]
    ]
    proposed_blocked_long_symbols = sorted(effective_current_long_set | set(long_additions))
    proposed_blocked_short_symbols = sorted(effective_current_short_set | set(short_additions))
    restricted_runtime_symbols = (
        {
            *proposed_blocked_symbols,
            *proposed_blocked_long_symbols,
            *proposed_blocked_short_symbols,
        }
        & runtime_set
    )
    restriction_share = len(restricted_runtime_symbols) / runtime_count if runtime_count else 0.0
    guardrail_reason = ""
    if runtime_count and runtime_count < min_runtime_symbols_for_restrictions:
        guardrail_reason = (
            f"runtime universe is too small ({runtime_count} symbols) for entry symbol restrictions"
        )
    elif runtime_count and restriction_share > max_restricted_symbol_share:
        guardrail_reason = (
            "candidate restrictions would over-concentrate the runtime universe "
            f"({len(restricted_runtime_symbols)}/{runtime_count} symbols)"
        )

    if guardrail_reason:
        additions = []
        removals = []
        long_additions = []
        long_removals = []
        short_additions = []
        short_removals = []
        proposed_blocked_symbols = sorted(current_set)
        proposed_blocked_long_symbols = sorted(current_long_set)
        proposed_blocked_short_symbols = sorted(current_short_set)

    changed = (
        proposed_blocked_symbols != sorted(current_set)
        or proposed_blocked_long_symbols != sorted(current_long_set)
        or proposed_blocked_short_symbols != sorted(current_short_set)
    )

    if changed:
        reason = "entry symbol restrictions updated from paper results"
    elif guardrail_reason:
        reason = guardrail_reason
    elif available_slots <= 0 and available_long_slots <= 0 and available_short_slots <= 0:
        reason = "entry restriction budgets are already exhausted"
    else:
        reason = "insufficient evidence for symbol restriction change"

    return {
        "analysis_window": report["period"],
        "guardrails": {
            "min_trades_per_symbol": min_trades_per_symbol,
            "min_trades_per_direction_symbol": min_trades_per_direction_symbol,
            "max_symbols_to_block": max_symbols_to_block,
            "max_total_blocked_symbols": max_total_blocked_symbols,
            "max_symbols_to_unblock": max_symbols_to_unblock,
            "max_long_symbols_to_block": max_long_symbols_to_block,
            "max_short_symbols_to_block": max_short_symbols_to_block,
            "max_total_blocked_long_symbols": max_total_blocked_long_symbols,
            "max_total_blocked_short_symbols": max_total_blocked_short_symbols,
            "max_long_symbols_to_unblock": max_long_symbols_to_unblock,
            "max_short_symbols_to_unblock": max_short_symbols_to_unblock,
            "min_runtime_symbols_for_restrictions": min_runtime_symbols_for_restrictions,
            "max_restricted_symbol_share": max_restricted_symbol_share,
        },
        "evidence_source": evidence_source,
        "runtime_symbols": sorted(runtime_set),
        "current_blocked_symbols": sorted(current_set),
        "proposed_blocked_symbols": proposed_blocked_symbols,
        "additions": additions,
        "removals": removals,
        "current_blocked_long_symbols": sorted(current_long_set),
        "proposed_blocked_long_symbols": proposed_blocked_long_symbols,
        "long_additions": long_additions,
        "long_removals": long_removals,
        "current_blocked_short_symbols": sorted(current_short_set),
        "proposed_blocked_short_symbols": proposed_blocked_short_symbols,
        "short_additions": short_additions,
        "short_removals": short_removals,
        "changed": changed,
        "reason": reason,
        "symbol_breakdown": rows,
        "symbol_direction_breakdown": direction_rows,
    }


def write_entry_symbol_tuning(output_dir: Path, payload: dict[str, object]) -> None:
    payload = with_runtime_metadata(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "symbol_restrictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "symbol_restrictions_patch.toml").write_text(
        _render_patch(payload),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )


def _render_patch(payload: dict[str, object]) -> str:
    symbols = ", ".join(f'"{symbol}"' for symbol in payload["proposed_blocked_symbols"])
    long_symbols = ", ".join(f'"{symbol}"' for symbol in payload["proposed_blocked_long_symbols"])
    short_symbols = ", ".join(f'"{symbol}"' for symbol in payload["proposed_blocked_short_symbols"])
    return "\n".join(
        [
            "# Candidate patch generated from paper-trading symbol results",
            "[strategy]",
            f"blocked_symbols = [{symbols}]",
            f"blocked_long_symbols = [{long_symbols}]",
            f"blocked_short_symbols = [{short_symbols}]",
            "",
        ]
    )


def _render_markdown(payload: dict[str, object]) -> str:
    rows = sorted(payload["symbol_breakdown"], key=lambda row: float(row["net_pnl_rub"]))
    worst_rows = rows[:3]
    best_rows = list(reversed(rows[-3:])) if rows else []
    direction_rows = sorted(
        payload["symbol_direction_breakdown"],
        key=lambda row: float(row["net_pnl_rub"]),
    )
    weak_long_rows = [row for row in direction_rows if str(row["direction"]) == "long"][:3]
    weak_short_rows = [row for row in direction_rows if str(row["direction"]) == "short"][:3]
    lines = [
        "# Entry Symbol Tuning",
        "",
        f"- Commit: {payload.get('commit_hash', 'unknown')}",
        f"- Lookback: {payload['analysis_window']['days']} day(s)",
        f"- Evidence source: {payload['evidence_source']}",
        f"- Evidence counts: {_render_evidence_counts(payload.get('evidence_counts', {}))}",
        f"- Current blocked symbols: {payload['current_blocked_symbols']}",
        f"- Proposed blocked symbols: {payload['proposed_blocked_symbols']}",
        f"- Additions: {payload['additions']}",
        f"- Removals: {payload['removals']}",
        f"- Current blocked long symbols: {payload['current_blocked_long_symbols']}",
        f"- Proposed blocked long symbols: {payload['proposed_blocked_long_symbols']}",
        f"- Long additions: {payload['long_additions']}",
        f"- Long removals: {payload['long_removals']}",
        f"- Current blocked short symbols: {payload['current_blocked_short_symbols']}",
        f"- Proposed blocked short symbols: {payload['proposed_blocked_short_symbols']}",
        f"- Short additions: {payload['short_additions']}",
        f"- Short removals: {payload['short_removals']}",
        f"- Changed: {payload['changed']}",
        f"- Reason: {payload['reason']}",
        "",
        "## Strong Symbols",
    ]
    if best_rows:
        for row in best_rows:
            lines.append(
                f"- {row['group']}: {row['net_pnl_rub']} RUB, {row['trades']} trades, expectancy {row['expectancy_rub']}"
            )
    else:
        lines.append("- No eligible paper trades in this window")

    lines.append("")
    lines.append("## Weak Symbols")
    if worst_rows:
        for row in worst_rows:
            lines.append(
                f"- {row['group']}: {row['net_pnl_rub']} RUB, {row['trades']} trades, expectancy {row['expectancy_rub']}"
            )
    else:
        lines.append("- No eligible paper trades in this window")
    lines.append("")
    lines.append("## Weak Long Sleeves")
    if weak_long_rows:
        for row in weak_long_rows:
            lines.append(
                f"- {row['symbol']} long: {row['net_pnl_rub']} RUB, {row['trades']} trades, expectancy {row['expectancy_rub']}"
            )
    else:
        lines.append("- No eligible long-direction paper trades in this window")
    lines.append("")
    lines.append("## Weak Short Sleeves")
    if weak_short_rows:
        for row in weak_short_rows:
            lines.append(
                f"- {row['symbol']} short: {row['net_pnl_rub']} RUB, {row['trades']} trades, expectancy {row['expectancy_rub']}"
            )
    else:
        lines.append("- No eligible short-direction paper trades in this window")
    lines.append("")
    return "\n".join(lines)


def _render_evidence_counts(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        return "-"
    parts: list[str] = []
    if "combined_trades" in payload:
        parts.append(f"combined={payload['combined_trades']}")
    if "closed_trades" in payload:
        parts.append(f"closed={payload['closed_trades']}")
    if "deduplicated_feedback_trades" in payload:
        parts.append(f"feedback={payload['deduplicated_feedback_trades']}")
    if payload.get("duplicate_feedback_trades"):
        parts.append(f"dupes={payload['duplicate_feedback_trades']}")
    return ", ".join(parts) if parts else "-"


def _normalized_symbol_set(values: list[str]) -> set[str]:
    return {
        str(symbol).strip().upper()
        for symbol in values
        if str(symbol).strip()
    }


def _closed_trade_rows_for_window(
    trades: list[TradeRecord],
    *,
    timezone_name: str,
    start_at_iso: str,
    end_at_iso: str,
) -> list[dict[str, object]]:
    timezone = ZoneInfo(timezone_name)
    start_at = datetime.fromisoformat(start_at_iso)
    end_at = datetime.fromisoformat(end_at_iso)
    rows: list[dict[str, object]] = []
    for trade in trades:
        exit_time = trade.exit_time.astimezone(timezone)
        if start_at <= exit_time < end_at:
            rows.append(
                {
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "net_pnl": trade.net_pnl,
                }
            )
    return rows


def _direction_candidates(
    rows: list[dict[str, object]],
    *,
    direction: str,
    current_set: set[str],
    globally_blocked_symbols: set[str],
    min_trades: int,
) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if str(row["direction"]).strip().lower() == direction
        and str(row["symbol"]).strip().upper() not in globally_blocked_symbols
        and str(row["symbol"]).strip().upper() not in current_set
        and int(row["trades"]) >= min_trades
        and float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])) < 0
        and float(row.get("raw_expectancy_rub", row["expectancy_rub"])) < 0
        and float(row.get("raw_profit_factor", row["profit_factor"])) < 1.0
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])),
            float(row.get("raw_expectancy_rub", row["expectancy_rub"])),
            float(row.get("raw_profit_factor", row["profit_factor"])),
        )
    )
    return candidates


def _recovery_candidates(
    rows: list[dict[str, object]],
    *,
    current_set: set[str],
    min_trades: int,
) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if str(row["group"]).strip().upper() in current_set
        and int(row["trades"]) >= min_trades
        and float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])) > 0
        and float(row.get("raw_expectancy_rub", row["expectancy_rub"])) > 0
        and float(row.get("raw_profit_factor", row["profit_factor"])) >= 1.0
        and float(row.get("win_rate_pct", 0.0)) >= 50.0
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])),
            float(row.get("raw_expectancy_rub", row["expectancy_rub"])),
            float(row.get("raw_profit_factor", row["profit_factor"])),
            float(row.get("win_rate_pct", 0.0)),
        ),
        reverse=True,
    )
    return candidates


def _direction_recovery_candidates(
    rows: list[dict[str, object]],
    *,
    direction: str,
    current_set: set[str],
    globally_blocked_symbols: set[str],
    min_trades: int,
) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if str(row["direction"]).strip().lower() == direction
        and str(row["symbol"]).strip().upper() not in globally_blocked_symbols
        and str(row["symbol"]).strip().upper() in current_set
        and int(row["trades"]) >= min_trades
        and float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])) > 0
        and float(row.get("raw_expectancy_rub", row["expectancy_rub"])) > 0
        and float(row.get("raw_profit_factor", row["profit_factor"])) >= 1.0
        and float(row.get("win_rate_pct", 0.0)) >= 50.0
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("raw_net_pnl_rub", row["net_pnl_rub"])),
            float(row.get("raw_expectancy_rub", row["expectancy_rub"])),
            float(row.get("raw_profit_factor", row["profit_factor"])),
            float(row.get("win_rate_pct", 0.0)),
        ),
        reverse=True,
    )
    return candidates


def _symbol_breakdown(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        grouped[symbol].append(row)
    return _grouped_rows_to_breakdown(grouped, group_name_builder=lambda symbol: {"group": symbol})


def _symbol_direction_breakdown(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        direction = str(row.get("direction", "")).strip().lower()
        if not symbol or direction not in {"long", "short"}:
            continue
        grouped[(symbol, direction)].append(row)

    breakdown = _grouped_rows_to_breakdown(
        grouped,
        group_name_builder=lambda key: {
            "group": f"{key[0]}:{key[1]}",
            "symbol": key[0],
            "direction": key[1],
        },
    )
    breakdown.sort(key=lambda row: (str(row["symbol"]), str(row["direction"])))
    return breakdown


def _grouped_rows_to_breakdown(grouped, *, group_name_builder) -> list[dict[str, object]]:
    breakdown: list[dict[str, object]] = []
    for key, group_rows in grouped.items():
        pnl_values = [float(row.get("net_pnl", 0.0)) for row in group_rows]
        wins = [value for value in pnl_values if value > 0]
        losses = [value for value in pnl_values if value < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        net_pnl = sum(pnl_values)
        trades = len(pnl_values)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss else (999.0 if gross_profit > 0 else 0.0)
        expectancy = net_pnl / trades if trades else 0.0
        breakdown.append(
            {
                **group_name_builder(key),
                "trades": trades,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": round(len(wins) / trades * 100, 3) if trades else 0.0,
                "net_pnl_rub": round(net_pnl, 2),
                "gross_profit_rub": round(gross_profit, 2),
                "gross_loss_rub": round(gross_loss, 2),
                "profit_factor": round(profit_factor, 3),
                "expectancy_rub": round(expectancy, 2),
                "avg_win_rub": round(avg_win, 2),
                "avg_loss_rub": round(avg_loss, 2),
                "raw_net_pnl_rub": net_pnl,
                "raw_profit_factor": profit_factor,
                "raw_expectancy_rub": expectancy,
            }
        )
    return breakdown
