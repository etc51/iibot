from __future__ import annotations

import json
import math
from pathlib import Path


def build_universe_selection_tuning_payload(
    *,
    configured_symbols: list[str],
    current_allowed_symbols: list[str],
    optimizer_payload: dict[str, object],
    walk_forward_payload: dict[str, object],
    max_allowed_symbols: int = 3,
    min_walk_forward_positive_probability_pct: float = 55.0,
    min_latest_fold_monthly_return_pct: float = 0.0,
    min_walk_forward_folds: int = 3,
    min_latest_fold_trades: int = 4,
    require_optimizer_overlap: bool = True,
    min_effective_symbols: int = 5,
    min_configured_coverage_ratio: float = 0.5,
) -> dict[str, object]:
    if max_allowed_symbols < 1:
        raise ValueError("max_allowed_symbols must be >= 1")
    if min_walk_forward_positive_probability_pct < 0:
        raise ValueError("min_walk_forward_positive_probability_pct must be >= 0")
    if min_walk_forward_folds < 1:
        raise ValueError("min_walk_forward_folds must be >= 1")
    if min_latest_fold_trades < 1:
        raise ValueError("min_latest_fold_trades must be >= 1")
    if min_effective_symbols < 1:
        raise ValueError("min_effective_symbols must be >= 1")
    if min_configured_coverage_ratio <= 0 or min_configured_coverage_ratio > 1:
        raise ValueError("min_configured_coverage_ratio must be within (0, 1]")

    configured = _normalized_symbols(configured_symbols)
    current_raw = _normalized_symbols(current_allowed_symbols)
    current_effective = current_raw or list(configured)

    optimizer_best = optimizer_payload.get("best_candidate", {})
    optimizer_symbols = _filter_to_configured(
        optimizer_best.get("symbols", []) if isinstance(optimizer_best, dict) else [],
        configured,
    )

    folds = walk_forward_payload.get("folds", [])
    latest_fold = folds[-1] if isinstance(folds, list) and folds else {}
    latest_candidate = latest_fold.get("best_candidate", {}) if isinstance(latest_fold, dict) else {}
    latest_test_summary = latest_fold.get("test_summary", {}) if isinstance(latest_fold, dict) else {}
    walk_forward_symbols = _filter_to_configured(
        latest_candidate.get("symbols", []) if isinstance(latest_candidate, dict) else [],
        configured,
    )

    walk_forward_summary = walk_forward_payload.get("summary", {})
    positive_probability_pct = float(
        walk_forward_summary.get("probability_positive_pct", 0.0)
        if isinstance(walk_forward_summary, dict)
        else 0.0
    )
    folds_evaluated = int(
        walk_forward_summary.get("folds_evaluated", 0)
        if isinstance(walk_forward_summary, dict)
        else 0
    )
    latest_fold_monthly_return_pct = float(
        latest_test_summary.get("normalized_monthly_return_pct", 0.0)
        if isinstance(latest_test_summary, dict)
        else 0.0
    )
    latest_fold_trades = int(
        latest_test_summary.get("trades", 0)
        if isinstance(latest_test_summary, dict)
        else 0
    )

    consensus_symbols: list[str] = []
    reason = ""
    if not configured:
        reason = "runtime config has no configured symbols"
    elif not optimizer_symbols:
        reason = "optimizer did not produce a usable symbol subset"
    elif not walk_forward_symbols:
        reason = "walk-forward did not produce a usable latest-fold symbol subset"
    elif folds_evaluated < min_walk_forward_folds:
        reason = "walk-forward history is too short for a universe change"
    elif positive_probability_pct < min_walk_forward_positive_probability_pct:
        reason = "walk-forward positive-fold probability is too weak for a universe change"
    elif latest_fold_monthly_return_pct < min_latest_fold_monthly_return_pct:
        reason = "latest walk-forward fold is too weak for a universe change"
    elif latest_fold_trades < min_latest_fold_trades:
        reason = "latest walk-forward fold has too few trades for a universe change"
    else:
        if require_optimizer_overlap:
            consensus_symbols = [
                symbol for symbol in walk_forward_symbols if symbol in set(optimizer_symbols)
            ]
            if not consensus_symbols:
                reason = "optimizer and latest walk-forward fold do not agree on a stable universe"
        else:
            consensus_symbols = list(walk_forward_symbols)

    proposed_effective_symbols = list(current_effective)
    if consensus_symbols:
        proposed_effective_symbols = sorted(dict.fromkeys(consensus_symbols))[:max_allowed_symbols]
        reason = "runtime universe updated from optimizer and walk-forward consensus"

    proposed_allowed_symbols = (
        []
        if proposed_effective_symbols == list(configured)
        else list(proposed_effective_symbols)
    )
    guardrail_blocked = False
    minimum_effective_symbols = (
        min(
            len(configured),
            max(
                min_effective_symbols,
                math.ceil(len(configured) * min_configured_coverage_ratio),
            ),
        )
        if configured
        else min_effective_symbols
    )
    if consensus_symbols and len(proposed_effective_symbols) < minimum_effective_symbols:
        proposed_effective_symbols = list(current_effective)
        proposed_allowed_symbols = list(current_raw)
        guardrail_blocked = True
        reason = (
            "research consensus is too narrow for a diversified runtime "
            f"({len(consensus_symbols)}/{len(configured)} symbols)"
        )
    additions = [
        symbol for symbol in proposed_effective_symbols if symbol not in set(current_effective)
    ]
    removals = [
        symbol for symbol in current_effective if symbol not in set(proposed_effective_symbols)
    ]
    changed = proposed_allowed_symbols != current_raw
    if not changed and consensus_symbols and not guardrail_blocked:
        reason = "current allowed symbol set already matches research consensus"
    elif not changed and not reason:
        reason = "insufficient evidence for runtime universe change"

    return {
        "guardrails": {
            "max_allowed_symbols": max_allowed_symbols,
            "min_walk_forward_positive_probability_pct": min_walk_forward_positive_probability_pct,
            "min_latest_fold_monthly_return_pct": min_latest_fold_monthly_return_pct,
            "min_walk_forward_folds": min_walk_forward_folds,
            "min_latest_fold_trades": min_latest_fold_trades,
            "require_optimizer_overlap": require_optimizer_overlap,
            "min_effective_symbols": min_effective_symbols,
            "min_configured_coverage_ratio": min_configured_coverage_ratio,
        },
        "configured_symbols": list(configured),
        "current_allowed_symbols": current_raw,
        "current_effective_symbols": current_effective,
        "optimizer_best_symbols": optimizer_symbols,
        "walk_forward_latest_symbols": walk_forward_symbols,
        "consensus_symbols": consensus_symbols,
        "proposed_allowed_symbols": proposed_allowed_symbols,
        "proposed_effective_symbols": proposed_effective_symbols,
        "additions": additions,
        "removals": removals,
        "changed": changed,
        "reason": reason or "insufficient evidence for runtime universe change",
        "optimizer_summary": {
            "evaluated_candidates": optimizer_payload.get("evaluated_candidates", 0),
            "score": optimizer_best.get("score", 0.0) if isinstance(optimizer_best, dict) else 0.0,
            "avg_monthly_return_pct": (
                optimizer_best.get("summary", {}).get("avg_monthly_return_pct", 0.0)
                if isinstance(optimizer_best, dict)
                else 0.0
            ),
            "max_drawdown_pct": (
                optimizer_best.get("summary", {}).get("max_drawdown_pct", 0.0)
                if isinstance(optimizer_best, dict)
                else 0.0
            ),
            "profit_factor": (
                optimizer_best.get("summary", {}).get("profit_factor", 0.0)
                if isinstance(optimizer_best, dict)
                else 0.0
            ),
            "trades": (
                optimizer_best.get("summary", {}).get("trades", 0)
                if isinstance(optimizer_best, dict)
                else 0
            ),
        },
        "walk_forward_summary": {
            "folds_evaluated": folds_evaluated,
            "average_test_normalized_monthly_return_pct": (
                walk_forward_summary.get("average_test_normalized_monthly_return_pct", 0.0)
                if isinstance(walk_forward_summary, dict)
                else 0.0
            ),
            "probability_positive_pct": positive_probability_pct,
            "latest_fold_monthly_return_pct": latest_fold_monthly_return_pct,
            "latest_fold_test_trades": latest_fold_trades,
        },
    }


def write_universe_selection_tuning(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "universe_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "universe_selection_patch.toml").write_text(
        _render_patch(payload),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )


def _render_patch(payload: dict[str, object]) -> str:
    symbols = ", ".join(f'"{symbol}"' for symbol in payload["proposed_allowed_symbols"])
    return "\n".join(
        [
            "# Candidate patch generated from optimizer + walk-forward universe consensus",
            "[strategy]",
            f"allowed_symbols = [{symbols}]",
            "",
        ]
    )


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Runtime Universe Tuning",
        "",
        f"- Configured symbols: {payload['configured_symbols']}",
        f"- Current allowed symbols: {payload['current_allowed_symbols']}",
        f"- Current effective symbols: {payload['current_effective_symbols']}",
        f"- Optimizer winner symbols: {payload['optimizer_best_symbols']}",
        f"- Walk-forward latest symbols: {payload['walk_forward_latest_symbols']}",
        f"- Consensus symbols: {payload['consensus_symbols']}",
        f"- Proposed allowed symbols: {payload['proposed_allowed_symbols']}",
        f"- Proposed effective symbols: {payload['proposed_effective_symbols']}",
        f"- Additions: {payload['additions']}",
        f"- Removals: {payload['removals']}",
        f"- Changed: {payload['changed']}",
        f"- Reason: {payload['reason']}",
        "",
        "## Research Guardrails",
        f"- Optimizer candidates: {payload['optimizer_summary']['evaluated_candidates']}",
        f"- Walk-forward folds: {payload['walk_forward_summary']['folds_evaluated']}",
        f"- Walk-forward positive probability: {payload['walk_forward_summary']['probability_positive_pct']}%",
        f"- Latest fold normalized monthly return: {payload['walk_forward_summary']['latest_fold_monthly_return_pct']}%",
        f"- Latest fold trades: {payload['walk_forward_summary']['latest_fold_test_trades']}",
        "",
    ]
    return "\n".join(lines)


def _normalized_symbols(values: list[str]) -> list[str]:
    normalized = {
        str(symbol).strip().upper()
        for symbol in values
        if str(symbol).strip()
    }
    return sorted(normalized)


def _filter_to_configured(values: list[object], configured: list[str]) -> list[str]:
    configured_set = set(configured)
    return [
        symbol
        for symbol in _normalized_symbols([str(value) for value in values])
        if symbol in configured_set
    ]
