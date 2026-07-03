from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import BacktestSection, ResearchSection
from ..domain import TradeRecord
from ..research.targets import effective_target_payload, render_target_label


def build_signal_strength_breakdown(
    trades: list[TradeRecord],
    *,
    bucket_size: float = 0.1,
) -> list[dict[str, object]]:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be > 0")

    buckets: dict[float, list[TradeRecord]] = {}
    for trade in trades:
        if trade.signal_strength <= 0:
            continue
        bucket = math.floor(trade.signal_strength / bucket_size) * bucket_size
        bucket = round(min(bucket, 1.0 - bucket_size), 4)
        buckets.setdefault(bucket, []).append(trade)

    rows = []
    for bucket_start in sorted(buckets):
        group = buckets[bucket_start]
        rows.append(
            {
                "bucket_start": round(bucket_start, 4),
                "bucket_end": round(min(1.0, bucket_start + bucket_size), 4),
                **_trade_summary(group),
            }
        )
    return rows


def build_entry_quality_tuning_payload(
    *,
    trades: list[TradeRecord],
    evidence_source: str,
    current_min_signal_strength: float,
    backtest: BacktestSection,
    research: ResearchSection,
    lookback_trades: int = 40,
    min_trades: int = 8,
    min_trade_retention_ratio: float = 0.5,
    min_expectancy_improvement_rub: float = 0.0,
    bucket_step: float = 0.05,
) -> dict[str, object]:
    ordered = sorted(trades, key=lambda trade: trade.exit_time)
    recent = ordered[-lookback_trades:] if lookback_trades > 0 else ordered
    evidence_trades = [trade for trade in recent if trade.signal_strength > 0]
    breakdown = build_signal_strength_breakdown(evidence_trades)

    target = effective_target_payload(research, backtest)
    if not evidence_trades:
        empty_summary = _trade_summary([])
        return {
            "target": target,
            "evidence_source": evidence_source,
            "lookback": {
                "requested_trades": lookback_trades,
                "eligible_trades": len(evidence_trades),
                "min_trades": min_trades,
            },
            "current_min_signal_strength": round(current_min_signal_strength, 4),
            "recommended_min_signal_strength": round(current_min_signal_strength, 4),
            "changed": False,
            "reason": "insufficient paper trades with signal strength evidence",
            "baseline_summary": empty_summary,
            "recommended_summary": empty_summary,
            "signal_strength_breakdown": breakdown,
            "candidate_thresholds": [],
        }

    thin_evidence = len(evidence_trades) < min_trades
    baseline = _trade_summary(
        [trade for trade in evidence_trades if trade.signal_strength >= current_min_signal_strength]
    )
    thresholds = _candidate_thresholds(
        evidence_trades,
        current_min_signal_strength=current_min_signal_strength,
        bucket_step=bucket_step,
    )
    should_consider_lower_thresholds = (
        baseline["trades"] < min_trades
        or float(baseline["net_pnl_rub"]) <= 0
        or float(baseline["expectancy_rub"]) <= 0
    )

    candidates = []
    best_candidate: dict[str, object] | None = None
    for threshold in thresholds:
        selected = [trade for trade in evidence_trades if trade.signal_strength >= threshold]
        summary = _trade_summary(selected)
        retention_ratio = summary["trades"] / baseline["trades"] if baseline["trades"] else 0.0
        expectancy_delta = round(
            float(summary["expectancy_rub"]) - float(baseline["expectancy_rub"]),
            2,
        )
        net_pnl_delta = round(
            float(summary["net_pnl_rub"]) - float(baseline["net_pnl_rub"]),
            2,
        )
        profit_factor_delta = round(
            float(summary["profit_factor"]) - float(baseline["profit_factor"]),
            3,
        )
        if threshold > current_min_signal_strength:
            eligible = (
                summary["trades"] >= min_trades
                and retention_ratio >= min_trade_retention_ratio
                and expectancy_delta >= min_expectancy_improvement_rub
                and net_pnl_delta > 0
                and profit_factor_delta >= 0
            )
        else:
            eligible = (
                should_consider_lower_thresholds
                and (
                    summary["trades"] >= min_trades
                    or (
                        thin_evidence
                        and current_min_signal_strength > 0
                        and summary["trades"] > baseline["trades"]
                    )
                )
                and net_pnl_delta > 0
                and float(summary["expectancy_rub"]) > 0
                and float(summary["profit_factor"]) >= 1.0
            )
        row = {
            "threshold": round(threshold, 4),
            "eligible": eligible,
            "retention_ratio": round(retention_ratio, 3),
            "expectancy_delta_rub": expectancy_delta,
            "net_pnl_delta_rub": net_pnl_delta,
            "profit_factor_delta": profit_factor_delta,
            "summary": summary,
        }
        candidates.append(row)
        if eligible and (
            best_candidate is None
            or (
                threshold > current_min_signal_strength
                and float(row["expectancy_delta_rub"]) > float(best_candidate["expectancy_delta_rub"])
            )
            or (
                threshold > current_min_signal_strength
                and float(row["expectancy_delta_rub"]) == float(best_candidate["expectancy_delta_rub"])
                and float(row["net_pnl_delta_rub"]) > float(best_candidate["net_pnl_delta_rub"])
            )
            or (
                threshold < current_min_signal_strength
                and float(row["net_pnl_delta_rub"]) > float(best_candidate["net_pnl_delta_rub"])
            )
            or (
                threshold < current_min_signal_strength
                and float(row["net_pnl_delta_rub"]) == float(best_candidate["net_pnl_delta_rub"])
                and float(row["expectancy_delta_rub"]) > float(best_candidate["expectancy_delta_rub"])
            )
        ):
            best_candidate = row

    if best_candidate is None:
        recommended = current_min_signal_strength
        changed = False
        reason = (
            "insufficient paper trades with signal strength evidence"
            if thin_evidence
            else "no signal-strength threshold passed the safety guardrails"
        )
        best_summary = baseline
    else:
        recommended = float(best_candidate["threshold"])
        changed = recommended != current_min_signal_strength
        if changed and recommended < current_min_signal_strength:
            reason = "lower signal-strength threshold restored recent paper-trade coverage"
        elif changed:
            reason = "signal-strength threshold improved recent paper expectancy"
        else:
            reason = "current signal-strength threshold already matches the best candidate"
        best_summary = best_candidate["summary"]

    return {
        "target": target,
        "evidence_source": evidence_source,
        "lookback": {
            "requested_trades": lookback_trades,
            "eligible_trades": len(evidence_trades),
            "min_trades": min_trades,
        },
        "guardrails": {
            "min_trade_retention_ratio": min_trade_retention_ratio,
            "min_expectancy_improvement_rub": min_expectancy_improvement_rub,
            "bucket_step": bucket_step,
        },
        "current_min_signal_strength": round(current_min_signal_strength, 4),
        "recommended_min_signal_strength": round(recommended, 4),
        "changed": changed,
        "reason": reason,
        "baseline_summary": baseline,
        "recommended_summary": best_summary,
        "signal_strength_breakdown": breakdown,
        "candidate_thresholds": candidates,
    }


def write_entry_quality_tuning(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "entry_quality_tuning.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "entry_quality_patch.toml").write_text(
        _render_patch(payload),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )


def _candidate_thresholds(
    trades: list[TradeRecord],
    *,
    current_min_signal_strength: float,
    bucket_step: float,
) -> list[float]:
    thresholds = {
        round(
            min(1.0, math.floor(trade.signal_strength / bucket_step) * bucket_step),
            4,
        )
        for trade in trades
        if trade.signal_strength > 0
    }
    return sorted(value for value in thresholds if value != round(current_min_signal_strength, 4))


def _trade_summary(trades: list[TradeRecord]) -> dict[str, float | int]:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "net_pnl_rub": 0.0,
            "expectancy_rub": 0.0,
            "profit_factor": 0.0,
            "avg_signal_strength": 0.0,
        }
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl < 0]
    gross_profit = sum(trade.net_pnl for trade in wins)
    gross_loss = abs(sum(trade.net_pnl for trade in losses))
    net_pnl = sum(trade.net_pnl for trade in trades)
    avg_signal_strength = sum(trade.signal_strength for trade in trades) / len(trades)
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 3),
        "net_pnl_rub": round(net_pnl, 2),
        "expectancy_rub": round(net_pnl / len(trades), 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else (999.0 if gross_profit > 0 else 0.0),
        "avg_signal_strength": round(avg_signal_strength, 4),
    }


def _render_patch(payload: dict[str, object]) -> str:
    if not payload.get("changed"):
        return "\n".join(
            [
                "# No entry-quality change recommended by the latest paper feedback pass",
                "",
            ]
        )
    return "\n".join(
        [
            "# Candidate entry-quality patch generated from paper-trading results",
            "[strategy]",
            f"min_signal_strength = {payload['recommended_min_signal_strength']}",
            "",
        ]
    )


def _render_markdown(payload: dict[str, object]) -> str:
    baseline = payload["baseline_summary"]
    recommended = payload["recommended_summary"]
    lines = [
        "# Entry Quality Tuning",
        "",
        f"- Target: {render_target_label(payload['target'])}",
        f"- Evidence source: {payload['evidence_source']}",
        f"- Evidence counts: {_render_evidence_counts(payload.get('evidence_counts', {}))}",
        f"- Changed: {payload['changed']}",
        f"- Reason: {payload['reason']}",
        f"- Current min signal strength: {payload['current_min_signal_strength']}",
        f"- Recommended min signal strength: {payload['recommended_min_signal_strength']}",
        "",
        "## Baseline",
        f"- Trades: {baseline['trades']}",
        f"- Net PnL: {baseline['net_pnl_rub']} RUB",
        f"- Expectancy: {baseline['expectancy_rub']} RUB",
        f"- Profit factor: {baseline['profit_factor']}",
        "",
        "## Recommended",
        f"- Trades: {recommended['trades']}",
        f"- Net PnL: {recommended['net_pnl_rub']} RUB",
        f"- Expectancy: {recommended['expectancy_rub']} RUB",
        f"- Profit factor: {recommended['profit_factor']}",
        "",
    ]
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
