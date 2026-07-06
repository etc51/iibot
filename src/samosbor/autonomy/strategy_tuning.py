from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ..config import BacktestSection, ResearchSection, StrategySection
from ..research.targets import effective_target_payload, render_target_label
from ..runtime_metadata import with_runtime_metadata


def adapt_strategy_tuning_research(
    research: ResearchSection,
    *,
    available_months: int,
    fixed_subset_size: int,
) -> tuple[ResearchSection | None, dict[str, object]]:
    if available_months < 2:
        return None, {
            "available_months": available_months,
            "usable": False,
            "reason": "need at least 2 months of history for strategy tuning",
        }

    train_months = max(1, research.walk_forward_train_months)
    test_months = max(1, research.walk_forward_test_months)
    step_months = max(1, research.walk_forward_step_months)

    if available_months < train_months + test_months:
        test_months = 1
        train_months = max(1, available_months - test_months)
        step_months = 1

    adjusted = replace(
        research,
        subset_min_size=max(1, fixed_subset_size),
        subset_max_size=max(1, fixed_subset_size),
        walk_forward_train_months=train_months,
        walk_forward_test_months=test_months,
        walk_forward_step_months=max(1, min(step_months, test_months)),
    )
    return adjusted, {
        "available_months": available_months,
        "usable": True,
        "configured_train_months": research.walk_forward_train_months,
        "configured_test_months": research.walk_forward_test_months,
        "configured_step_months": research.walk_forward_step_months,
        "train_months": adjusted.walk_forward_train_months,
        "test_months": adjusted.walk_forward_test_months,
        "step_months": adjusted.walk_forward_step_months,
        "subset_size": fixed_subset_size,
        "history_was_adapted": (
            adjusted.walk_forward_train_months != research.walk_forward_train_months
            or adjusted.walk_forward_test_months != research.walk_forward_test_months
            or adjusted.walk_forward_step_months != research.walk_forward_step_months
        ),
    }


def build_strategy_tuning_payload(
    *,
    current_strategy: StrategySection,
    candidate_strategy: StrategySection,
    baseline_latest_test_summary: dict[str, float | int],
    candidate_latest_test_summary: dict[str, float | int],
    walk_forward_summary: dict[str, float | int],
    walk_forward_config: dict[str, object],
    backtest: BacktestSection,
    research: ResearchSection,
    research_window: dict[str, object],
    min_monthly_improvement_pct: float = 0.05,
    max_extra_drawdown_pct: float = 1.0,
    min_positive_fold_probability_pct: float = 55.0,
) -> dict[str, object]:
    target = effective_target_payload(research, backtest)
    current_view = _strategy_view(current_strategy)
    candidate_view = _strategy_view(candidate_strategy)
    patch_values = {
        key: value
        for key, value in candidate_view.items()
        if current_view.get(key) != value
    }

    monthly_delta = round(
        float(candidate_latest_test_summary["normalized_monthly_return_pct"])
        - float(baseline_latest_test_summary["normalized_monthly_return_pct"]),
        3,
    )
    total_return_delta = round(
        float(candidate_latest_test_summary["total_return_pct"])
        - float(baseline_latest_test_summary["total_return_pct"]),
        3,
    )
    drawdown_delta = round(
        float(candidate_latest_test_summary["max_drawdown_pct"])
        - float(baseline_latest_test_summary["max_drawdown_pct"]),
        3,
    )
    sharpe_delta = round(
        float(candidate_latest_test_summary["sharpe_ratio"])
        - float(baseline_latest_test_summary["sharpe_ratio"]),
        3,
    )

    changed = bool(patch_values) and (
        monthly_delta >= min_monthly_improvement_pct
        and total_return_delta > 0
        and drawdown_delta <= max_extra_drawdown_pct
        and float(walk_forward_summary["average_test_normalized_monthly_return_pct"]) > 0
        and float(walk_forward_summary["probability_positive_pct"])
        >= min_positive_fold_probability_pct
    )

    if not patch_values:
        reason = "current strategy already matches the latest walk-forward winner"
    elif not changed:
        reason = "candidate failed one or more safety guardrails"
    else:
        reason = "candidate improved the latest OOS window without breaking guardrails"

    return {
        "target": target,
        "research_window": research_window,
        "walk_forward": {
            "config": walk_forward_config,
            "summary": walk_forward_summary,
        },
        "guardrails": {
            "min_monthly_improvement_pct": min_monthly_improvement_pct,
            "max_extra_drawdown_pct": max_extra_drawdown_pct,
            "min_positive_fold_probability_pct": min_positive_fold_probability_pct,
        },
        "current_strategy": current_view,
        "candidate_strategy": candidate_view,
        "patch_values": patch_values,
        "baseline_latest_test_summary": baseline_latest_test_summary,
        "candidate_latest_test_summary": candidate_latest_test_summary,
        "comparison": {
            "monthly_return_delta_pct": monthly_delta,
            "total_return_delta_pct": total_return_delta,
            "max_drawdown_delta_pct": drawdown_delta,
            "sharpe_delta": sharpe_delta,
        },
        "changed": changed,
        "reason": reason,
    }


def write_strategy_tuning(output_dir: Path, payload: dict[str, object]) -> None:
    payload = with_runtime_metadata(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strategy_tuning.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "strategy_patch.toml").write_text(
        _render_strategy_patch(payload),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(payload), encoding="utf-8")


def _strategy_view(strategy: StrategySection) -> dict[str, object]:
    return {
        "style": strategy.style,
        "fast_window": strategy.fast_window,
        "slow_window": strategy.slow_window,
        "require_breakout": strategy.require_breakout,
        "opening_range_bars": strategy.opening_range_bars,
        "rel_volume_threshold": strategy.rel_volume_threshold,
        "atr_stop_multiple": strategy.atr_stop_multiple,
        "reward_to_risk": strategy.reward_to_risk,
        "breakeven_trigger_pct": strategy.breakeven_trigger_pct,
        "trailing_profit_trigger_rub": strategy.trailing_profit_trigger_rub,
        "trailing_profit_lock_ratio": strategy.trailing_profit_lock_ratio,
        "min_trend_strength": strategy.min_trend_strength,
        "adx_min": strategy.adx_min,
        "rsi_long_max": strategy.rsi_long_max,
        "rsi_short_min": strategy.rsi_short_min,
    }


def _render_strategy_patch(payload: dict[str, object]) -> str:
    patch_values = payload.get("patch_values", {})
    if not patch_values:
        return "\n".join(
            [
                "# No strategy parameter changes recommended by the latest tuning pass",
                "",
            ]
        )

    lines = [
        "# Candidate strategy patch generated from walk-forward tuning",
        "[strategy]",
    ]
    for key, value in patch_values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = f'"{value}"'
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    lines.append("")
    return "\n".join(lines)


def _render_markdown(payload: dict[str, object]) -> str:
    comparison = payload["comparison"]
    baseline = payload["baseline_latest_test_summary"]
    candidate = payload["candidate_latest_test_summary"]
    lines = [
        "# Strategy Tuning",
        "",
        f"- Commit: {payload.get('commit_hash', 'unknown')}",
        f"- Target: {render_target_label(payload['target'])}",
        f"- Changed: {payload['changed']}",
        f"- Reason: {payload['reason']}",
        f"- Current strategy: {payload['current_strategy']}",
        f"- Candidate strategy: {payload['candidate_strategy']}",
        f"- Latest OOS monthly delta: {comparison['monthly_return_delta_pct']} pct",
        f"- Latest OOS total return delta: {comparison['total_return_delta_pct']} pct",
        f"- Latest OOS drawdown delta: {comparison['max_drawdown_delta_pct']} pct",
        "",
        "## Baseline OOS",
        f"- Return: {baseline['total_return_pct']}%",
        f"- Monthly: {baseline['normalized_monthly_return_pct']}%",
        f"- Drawdown: {baseline['max_drawdown_pct']}%",
        f"- Sharpe: {baseline['sharpe_ratio']}",
        "",
        "## Candidate OOS",
        f"- Return: {candidate['total_return_pct']}%",
        f"- Monthly: {candidate['normalized_monthly_return_pct']}%",
        f"- Drawdown: {candidate['max_drawdown_pct']}%",
        f"- Sharpe: {candidate['sharpe_ratio']}",
        "",
    ]
    return "\n".join(lines)
