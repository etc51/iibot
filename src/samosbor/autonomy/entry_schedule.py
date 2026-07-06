from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

from ..domain import PortfolioState, TradeRecord
from ..reporting.paper_report import build_paper_report_payload
from ..runtime_metadata import with_runtime_metadata


def build_entry_schedule_tuning_payload(
    portfolio: PortfolioState,
    trades: list[TradeRecord],
    *,
    timezone_name: str,
    current_hours: list[int],
    evidence_source: str = "closed-trades",
    report_date: date | None = None,
    lookback_days: int = 45,
    min_trades_per_hour: int = 3,
    max_hours_to_add: int = 2,
    max_hours_to_remove: int = 2,
    min_active_hours: int = 6,
    min_active_hour_coverage_ratio: float = 0.75,
) -> dict[str, object]:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if min_trades_per_hour < 1:
        raise ValueError("min_trades_per_hour must be >= 1")
    if min_active_hours < 1:
        raise ValueError("min_active_hours must be >= 1")
    if min_active_hour_coverage_ratio <= 0 or min_active_hour_coverage_ratio > 1:
        raise ValueError("min_active_hour_coverage_ratio must be within (0, 1]")

    report = build_paper_report_payload(
        portfolio,
        trades,
        timezone_name=timezone_name,
        report_date=report_date,
        days=lookback_days,
    )
    entry_hours = list(report["entry_hour_breakdown"])
    current_set = {int(hour) for hour in current_hours}
    if not current_set:
        return {
            "analysis_window": report["period"],
            "guardrails": {
                "min_trades_per_hour": min_trades_per_hour,
                "max_hours_to_add": max_hours_to_add,
                "max_hours_to_remove": max_hours_to_remove,
                "min_active_hours": min_active_hours,
                "min_active_hour_coverage_ratio": min_active_hour_coverage_ratio,
            },
            "evidence_source": evidence_source,
            "current_hours": [],
            "proposed_hours": [],
            "additions": [],
            "removals": [],
            "changed": False,
            "reason": "runtime is configured to trade without hour filters",
            "entry_hour_breakdown": entry_hours,
        }

    removable = [
        row
        for row in entry_hours
        if int(row["entry_hour"]) in current_set
        and int(row["trades"]) >= min_trades_per_hour
        and float(row["net_pnl_rub"]) < 0
        and float(row["expectancy_rub"]) < 0
    ]
    removable.sort(key=lambda row: (float(row["net_pnl_rub"]), float(row["expectancy_rub"])))
    removals = [int(row["entry_hour"]) for row in removable[:max_hours_to_remove]]

    addable = [
        row
        for row in entry_hours
        if int(row["entry_hour"]) not in current_set
        and int(row["trades"]) >= min_trades_per_hour
        and float(row["net_pnl_rub"]) > 0
        and float(row["expectancy_rub"]) > 0
        and float(row["win_rate_pct"]) >= 50.0
    ]
    addable.sort(
        key=lambda row: (
            float(row["net_pnl_rub"]),
            float(row["expectancy_rub"]),
            float(row["win_rate_pct"]),
        ),
        reverse=True,
    )
    additions = [int(row["entry_hour"]) for row in addable[:max_hours_to_add]]

    proposed_hours = sorted((current_set - set(removals)) | set(additions))
    minimum_active_hours = (
        min(
            len(current_set),
            max(
                min_active_hours,
                math.ceil(len(current_set) * min_active_hour_coverage_ratio),
            ),
        )
        if current_set
        else min_active_hours
    )
    guardrail_reason = ""
    if removals and current_set and len(proposed_hours) < minimum_active_hours:
        additions = []
        removals = []
        proposed_hours = sorted(current_set)
        guardrail_reason = (
            "candidate hours would over-narrow the runtime schedule "
            f"({len(proposed_hours)}/{len(current_set)} hours)"
        )
    no_change = proposed_hours == sorted(current_set)

    return {
        "analysis_window": report["period"],
        "guardrails": {
            "min_trades_per_hour": min_trades_per_hour,
            "max_hours_to_add": max_hours_to_add,
            "max_hours_to_remove": max_hours_to_remove,
            "min_active_hours": min_active_hours,
            "min_active_hour_coverage_ratio": min_active_hour_coverage_ratio,
        },
        "evidence_source": evidence_source,
        "current_hours": sorted(current_set),
        "proposed_hours": proposed_hours,
        "additions": additions,
        "removals": removals,
        "changed": not no_change,
        "reason": (
            guardrail_reason
            if guardrail_reason
            else "insufficient evidence for change"
            if no_change
            else "hours updated from paper results"
        ),
        "entry_hour_breakdown": entry_hours,
    }


def write_entry_schedule_tuning(output_dir: Path, payload: dict[str, object]) -> None:
    payload = with_runtime_metadata(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schedule_tuning.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "schedule_patch.toml").write_text(
        _render_schedule_patch(payload),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )


def _render_schedule_patch(payload: dict[str, object]) -> str:
    proposed_hours = ", ".join(str(hour) for hour in payload["proposed_hours"])
    return "\n".join(
        [
            "# Candidate patch generated from paper-trading results",
            "[strategy]",
            f"allowed_entry_hours = [{proposed_hours}]",
            "",
        ]
    )


def _render_markdown(payload: dict[str, object]) -> str:
    rows = sorted(payload["entry_hour_breakdown"], key=lambda row: float(row["net_pnl_rub"]))
    worst_rows = rows[:3]
    best_rows = list(reversed(rows[-3:])) if rows else []
    lines = [
        "# Entry Schedule Tuning",
        "",
        f"- Commit: {payload.get('commit_hash', 'unknown')}",
        f"- Lookback: {payload['analysis_window']['days']} day(s)",
        f"- Evidence source: {payload['evidence_source']}",
        f"- Current hours: {payload['current_hours']}",
        f"- Proposed hours: {payload['proposed_hours']}",
        f"- Additions: {payload['additions']}",
        f"- Removals: {payload['removals']}",
        f"- Changed: {payload['changed']}",
        f"- Reason: {payload['reason']}",
        "",
        "## Strong Hours",
    ]
    if best_rows:
        for row in best_rows:
            lines.append(
                f"- {int(row['entry_hour']):02d}:00: {row['net_pnl_rub']} RUB, {row['trades']} trades, expectancy {row['expectancy_rub']}"
            )
    else:
        lines.append("- No eligible paper trades in this window")

    lines.append("")
    lines.append("## Weak Hours")
    if worst_rows:
        for row in worst_rows:
            lines.append(
                f"- {int(row['entry_hour']):02d}:00: {row['net_pnl_rub']} RUB, {row['trades']} trades, expectancy {row['expectancy_rub']}"
            )
    else:
        lines.append("- No eligible paper trades in this window")
    lines.append("")
    return "\n".join(lines)
