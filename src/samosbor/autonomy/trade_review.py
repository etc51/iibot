from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import RiskSection, StrategySection
from ..domain import PortfolioState, TradeRecord
from ..runtime_metadata import with_runtime_metadata
from .short_ev_engine import ALLOWED_SHORT_SETUPS, canonical_setup_id
from .ml_learning import (
    COMMISSION_EDGE_TAG,
    CONFIRMATION_AFTER_IMPULSE_TAG,
    LATE_REENTRY_TAG,
    LOW_QUALITY_TAG,
    NEGATIVE_EXPECTANCY_TAG,
    SHORT_AFTER_EXHAUSTION_TAG,
)


def trade_review_path(state_path: Path) -> Path:
    suffix = state_path.suffix or ".json"
    return state_path.with_name(f"{state_path.stem}_trade_review{suffix}")


def build_trade_review_payload(
    portfolio: PortfolioState,
    trades: list[TradeRecord],
    events: list[dict[str, object]],
    *,
    strategy: StrategySection,
    risk: RiskSection,
    timezone_name: str,
    lookback_trades: int = 100,
    generated_at: datetime | None = None,
    microstructure_dir: str | Path | None = None,
) -> dict[str, object]:
    generated_at = generated_at or datetime.now(timezone.utc)
    timezone_info = ZoneInfo(timezone_name)
    closed_trades = sorted(trades, key=lambda trade: (trade.exit_time, trade.entry_time))
    recent_trades = closed_trades[-max(1, lookback_trades) :]
    parsed_events = _parsed_events(events)
    microstructure_root = Path(microstructure_dir) if microstructure_dir else None
    reviews = [
        _review_trade(
            trade,
            parsed_events,
            strategy=strategy,
            timezone_info=timezone_info,
            microstructure_dir=microstructure_root,
        )
        for trade in recent_trades
    ]
    recommendations = _build_recommendations(
        reviews,
        strategy=strategy,
        risk=risk,
    )
    summary = _summary(reviews)
    short_only_enabled = _short_only_enabled(reviews, parsed_events)
    long_learning = _long_learning_review(reviews, parsed_events)
    if short_only_enabled:
        long_learning = {
            **long_learning,
            "active": False,
            "legacy_only": True,
        }
    return with_runtime_metadata({
        "generated_at": generated_at.isoformat(),
        "timezone": timezone_name,
        "lookback_trades": lookback_trades,
        "reviewed_trades": len(reviews),
        "total_closed_trades": len(closed_trades),
        "portfolio": {
            "realized_pnl_rub": round(portfolio.realized_pnl, 2),
            "trading_halted": portfolio.trading_halted,
            "open_positions": len(portfolio.positions),
        },
        "summary": summary,
        "breakdowns": {
            "symbol": _group_breakdown(reviews, "symbol"),
            "direction": _group_breakdown(reviews, "direction"),
            "entry_hour": _group_breakdown(reviews, "entry_hour"),
            "exit_reason": _group_breakdown(reviews, "exit_reason"),
            "signal_strength_bucket": _group_breakdown(reviews, "signal_strength_bucket"),
            "microstructure_quality": _group_breakdown(reviews, "microstructure_quality"),
            "market_regime": _group_breakdown(reviews, "market_regime"),
            "entry_mode": _group_breakdown(reviews, "entry_mode"),
            "symbol_health": _group_breakdown(reviews, "symbol_health"),
            "rebound_outcome": _group_breakdown(reviews, "rebound_outcome"),
            "trade_source": _group_breakdown(reviews, "trade_source"),
            "actual_policy_decision": _group_breakdown(reviews, "actual_policy_decision"),
            "strict_policy_decision": _group_breakdown(reviews, "strict_policy_decision"),
        },
        "policy_signal_distribution": _policy_signal_distribution(parsed_events),
        "policy_outcomes": _policy_outcomes(reviews),
        "weak_choppy_direct_probe_review": _weak_choppy_direct_probe_review(reviews, parsed_events),
        "short_only_review": _short_only_review(portfolio, reviews, parsed_events),
        "short_ev_review": _short_ev_review(portfolio, reviews, parsed_events),
        "golden_3tf_review": _golden_3tf_review(reviews, parsed_events),
        "selloff_capture_review": _selloff_capture_review(reviews, parsed_events),
        "underallocation_review": _underallocation_review(parsed_events),
        "long_during_selloff_review": _long_during_selloff_review(reviews, parsed_events),
        "long_learning_review": long_learning,
        "strict_vs_relaxed": _strict_vs_relaxed_review(reviews, parsed_events),
        "blocker_review": _blocker_review(parsed_events),
        "learning_caps": _learning_cap_metrics(reviews, parsed_events),
        "selloff_learning_caps": _selloff_learning_cap_metrics(reviews, parsed_events),
        "mistake_breakdown": _mistake_breakdown(reviews),
        "recommendations": recommendations["items"],
        "config_patch_candidates": recommendations["config_patch_candidates"],
        "reviews": reviews,
    })


def save_trade_review(path: Path, payload: dict[str, object]) -> None:
    payload = with_runtime_metadata(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_trade_review(output_dir: Path, payload: dict[str, object]) -> None:
    payload = with_runtime_metadata(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_trade_review(output_dir / "trade_review.json", payload)
    (output_dir / "trade_review.md").write_text(_render_markdown(payload), encoding="utf-8")


def _review_trade(
    trade: TradeRecord,
    events: list[dict[str, object]],
    *,
    strategy: StrategySection,
    timezone_info: ZoneInfo,
    microstructure_dir: Path | None,
) -> dict[str, object]:
    entry_event = _matching_entry_event(trade, events)
    protection_events = _matching_protection_events(trade, events)
    initial_stop = _float_or_none(trade.initial_stop_price) or _float_or_none(entry_event.get("initial_stop_price"))
    initial_take_profit = (
        _float_or_none(trade.initial_take_profit)
        or _float_or_none(entry_event.get("initial_take_profit"))
    )
    inferred_units = _infer_units(trade)
    planned_risk_rub = _planned_distance_rub(trade.entry_price, initial_stop, inferred_units)
    planned_reward_rub = _planned_distance_rub(trade.entry_price, initial_take_profit, inferred_units)
    realized_r = trade.net_pnl / planned_risk_rub if planned_risk_rub and planned_risk_rub > 0 else None
    planned_rr = planned_reward_rub / planned_risk_rub if planned_risk_rub and planned_reward_rub else None
    holding_minutes = max(0.0, (trade.exit_time - trade.entry_time).total_seconds() / 60.0)
    localized_entry = trade.entry_time.astimezone(timezone_info)
    microstructure = _microstructure_for_review(trade, microstructure_dir=microstructure_dir)
    ml_learning = _ml_learning_summary(trade.entry_metadata)
    post_close_analysis = _post_close_analysis_summary(trade.entry_metadata)
    trade_excursion = _trade_excursion_summary(trade.entry_metadata, post_close_analysis)
    market_regime = _market_regime_summary(trade.entry_metadata)
    regime_policy = _regime_policy_summary(trade.entry_metadata)
    learning_caps = _learning_caps_summary(trade.entry_metadata, regime_policy)
    entry_mode = _entry_mode_summary(trade.entry_metadata, regime_policy)
    symbol_health = _symbol_health_summary(trade.entry_metadata, regime_policy)
    rebound_outcome = _rebound_outcome_summary(trade.entry_metadata)
    trade_source = _trade_source(trade)
    actual_policy_decision = str(regime_policy.get("actual_policy_decision", "unknown") or "unknown")
    strict_policy_decision = str(regime_policy.get("strict_policy_decision", "unknown") or "unknown")
    mistake_tags = _mistake_tags(
        trade,
        holding_minutes=holding_minutes,
        planned_risk_rub=planned_risk_rub,
        planned_rr=planned_rr,
        current_min_signal_strength=strategy.min_signal_strength,
        microstructure=microstructure,
        ml_learning=ml_learning,
        market_regime=market_regime,
        entry_mode=entry_mode,
        symbol_health=symbol_health,
        rebound_outcome=rebound_outcome,
    )
    lessons = [_lesson_for_tag(tag) for tag in mistake_tags]
    return {
        "review_id": _trade_review_id(trade),
        "symbol": trade.symbol,
        "direction": trade.direction.value,
        "quantity_lots": trade.quantity_lots,
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "entry_hour": localized_entry.hour,
        "entry_weekday": localized_entry.weekday(),
        "entry_price": round(trade.entry_price, 6),
        "exit_price": round(trade.exit_price, 6),
        "gross_pnl_rub": round(trade.gross_pnl, 2),
        "net_pnl_rub": round(trade.net_pnl, 2),
        "outcome": _outcome(trade.net_pnl),
        "exit_reason": trade.reason,
        "entry_reason": trade.entry_reason or str(entry_event.get("reason", "")),
        "signal_strength": round(trade.signal_strength, 4),
        "signal_strength_bucket": _strength_bucket(trade.signal_strength),
        "entry_context_score": round(trade.entry_context_score, 4),
        "entry_metadata": trade.entry_metadata,
        "market_regime": market_regime,
        "entry_mode": entry_mode,
        "symbol_health": symbol_health,
        "rebound_outcome": rebound_outcome,
        "trade_source": trade_source,
        "actual_policy_decision": actual_policy_decision,
        "strict_policy_decision": strict_policy_decision,
        "relaxed_only_trade": bool(regime_policy.get("relaxed_only_trade", False)),
        "effective_risk_multiplier": regime_policy.get("effective_risk_multiplier", regime_policy.get("risk_multiplier", 1.0)),
        "soft_issues": regime_policy.get("soft_issues", []),
        "hard_issues": regime_policy.get("hard_issues", []),
        "regime_policy": regime_policy,
        "learning_caps": learning_caps,
        "entry_ml_learning": ml_learning,
        "post_close_analysis": post_close_analysis,
        "trade_excursion": trade_excursion,
        "mfe_price": trade_excursion.get("mfe_price"),
        "mae_price": trade_excursion.get("mae_price"),
        "mfe_pnl_rub": trade_excursion.get("mfe_pnl_rub"),
        "mae_pnl_rub": trade_excursion.get("mae_pnl_rub"),
        "mae_abs_pnl_rub": trade_excursion.get("mae_abs_pnl_rub"),
        "mfe_r": trade_excursion.get("mfe_r"),
        "mae_r": trade_excursion.get("mae_r"),
        "mae_abs_r": trade_excursion.get("mae_abs_r"),
        "entry_microstructure": microstructure,
        "microstructure_quality": _microstructure_quality(microstructure),
        "initial_stop_price": _rounded_or_none(initial_stop),
        "initial_take_profit": _rounded_or_none(initial_take_profit),
        "planned_risk_rub": _rounded_or_none(planned_risk_rub, digits=2),
        "planned_reward_rub": _rounded_or_none(planned_reward_rub, digits=2),
        "planned_reward_risk": _rounded_or_none(planned_rr, digits=3),
        "realized_r": _rounded_or_none(realized_r, digits=3),
        "holding_minutes": round(holding_minutes, 2),
        "protection_updates": len(protection_events),
        "mistake_tags": mistake_tags,
        "lessons": lessons,
    }


def _parsed_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for event in events:
        timestamp = _parse_timestamp(event.get("timestamp"))
        if timestamp is None:
            continue
        parsed.append({**event, "_timestamp": timestamp})
    return parsed


def _matching_entry_event(trade: TradeRecord, events: list[dict[str, object]]) -> dict[str, object]:
    for event in events:
        if event.get("action") != "open":
            continue
        if str(event.get("symbol", "")) != trade.symbol:
            continue
        if str(event.get("direction", "")) != trade.direction.value:
            continue
        timestamp = event.get("_timestamp")
        if isinstance(timestamp, datetime) and abs((timestamp - trade.entry_time).total_seconds()) <= 1:
            return event
    return {}


def _matching_protection_events(
    trade: TradeRecord,
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    matched = []
    for event in events:
        if event.get("action") != "protect":
            continue
        if str(event.get("symbol", "")) != trade.symbol:
            continue
        timestamp = event.get("_timestamp")
        if isinstance(timestamp, datetime) and trade.entry_time <= timestamp <= trade.exit_time:
            matched.append(event)
    return matched


def _microstructure_summary(metadata: dict[str, object]) -> dict[str, object]:
    raw = dict(metadata.get("microstructure", {})) if isinstance(metadata.get("microstructure", {}), dict) else {}
    if not raw:
        return {
            "available": False,
            "reason": "not captured",
        }
    if not raw.get("available"):
        return {
            "available": False,
            "reason": str(raw.get("reason", "not available")),
        }
    keys = [
        "source",
        "collected_at",
        "timestamp",
        "collector_lag_seconds",
        "entry_event_microstructure_timestamp",
        "entry_event_microstructure_lag_seconds",
        "depth_requested",
        "depth_returned",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread_bps",
        "estimated_spread_cost_bps",
        "bid_depth_lots",
        "ask_depth_lots",
        "bid_depth_rub",
        "ask_depth_rub",
        "imbalance",
        "side_imbalance",
        "requested_lots",
        "entry_depth_lots",
        "entry_depth_rub",
        "entry_liquidity_cover",
        "best_executable_price",
    ]
    result: dict[str, object] = {"available": True}
    for key in keys:
        if key in raw:
            result[key] = raw[key]
    return result


def _microstructure_for_review(
    trade: TradeRecord,
    *,
    microstructure_dir: Path | None,
) -> dict[str, object]:
    event_microstructure = _microstructure_summary(trade.entry_metadata)
    event_timestamp = _parse_timestamp(event_microstructure.get("timestamp"))
    microstructure_anchor = event_timestamp or trade.entry_time
    collector_row = _nearest_microstructure_row(
        microstructure_dir,
        symbol=trade.symbol,
        entry_time=microstructure_anchor,
        max_lag_seconds=120.0,
    )
    if collector_row:
        microstructure = _microstructure_summary({"microstructure": collector_row})
        microstructure["source"] = "collector"
        collected_at = _parse_timestamp(collector_row.get("collected_at"))
        if collected_at is not None:
            microstructure["collector_lag_seconds"] = round(
                abs((collected_at - microstructure_anchor).total_seconds()),
                3,
            )
        if event_timestamp is not None:
            microstructure["entry_event_microstructure_timestamp"] = event_timestamp.isoformat()
            microstructure["entry_event_microstructure_lag_seconds"] = round(
                (event_timestamp - trade.entry_time).total_seconds(),
                3,
            )
        return _with_trade_side_microstructure(microstructure, trade)

    if event_microstructure.get("available"):
        event_microstructure["source"] = "entry_metadata"
        event_timestamp = _parse_timestamp(event_microstructure.get("timestamp"))
        if event_timestamp is not None:
            event_microstructure["entry_event_microstructure_lag_seconds"] = round(
                (event_timestamp - trade.entry_time).total_seconds(),
                3,
            )
    return _with_trade_side_microstructure(event_microstructure, trade)


def _ml_learning_summary(metadata: dict[str, object]) -> dict[str, object]:
    raw = metadata.get("ml_learning", {}) if isinstance(metadata, dict) else {}
    if not isinstance(raw, dict) or not raw:
        return {
            "available": False,
            "reason": "not captured",
            "blocks_entry": False,
        }
    keys = [
        "available",
        "action",
        "blocks_entry",
        "model",
        "reason",
        "resolved_samples",
        "usable_samples",
        "training_win_rate_pct",
        "training_expectancy_per_lot_rub",
        "probability_profit",
        "expected_pnl_per_lot_rub",
        "expected_pnl_position_rub",
        "low_quality_probability_threshold",
        "learning_tags",
    ]
    result: dict[str, object] = {}
    for key in keys:
        if key in raw:
            result[key] = raw[key]
    result.setdefault("available", False)
    result.setdefault("blocks_entry", False)
    return result


def _post_close_analysis_summary(metadata: dict[str, object]) -> dict[str, object]:
    raw = metadata.get("post_close_analysis", {}) if isinstance(metadata, dict) else {}
    if not isinstance(raw, dict) or not raw:
        return {
            "available": False,
            "reason": "not captured",
        }
    keys = [
        "available",
        "stage",
        "outcome",
        "is_error",
        "net_pnl_rub",
        "gross_pnl_rub",
        "exit_reason",
        "planned_risk_rub",
        "realized_r",
        "mfe_price",
        "mae_price",
        "mfe_pnl_rub",
        "mae_pnl_rub",
        "mae_abs_pnl_rub",
        "mfe_r",
        "mae_r",
        "mae_abs_r",
        "ml_available",
        "ml_probability_profit",
        "ml_expected_pnl_position_rub",
        "ml_entry_bias",
        "ml_verdict",
        "tags",
        "summary",
    ]
    return {key: raw[key] for key in keys if key in raw}


def _trade_excursion_summary(
    metadata: dict[str, object],
    post_close_analysis: dict[str, object],
) -> dict[str, object]:
    raw = metadata.get("trade_excursion", {}) if isinstance(metadata, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    keys = [
        "mfe_price",
        "mae_price",
        "mfe_pnl_rub",
        "mae_pnl_rub",
        "mae_abs_pnl_rub",
        "mfe_r",
        "mae_r",
        "mae_abs_r",
        "planned_risk_rub",
    ]
    result = {key: raw[key] for key in keys if key in raw}
    for key in keys:
        if key not in result and key in post_close_analysis:
            result[key] = post_close_analysis[key]
    if not result:
        return {
            "available": False,
            "reason": "not captured",
        }
    return {
        "available": True,
        **result,
    }


def _market_regime_summary(metadata: dict[str, object]) -> str:
    raw = metadata.get("market_regime", {}) if isinstance(metadata, dict) else {}
    if isinstance(raw, dict):
        return str(raw.get("regime", "unknown") or "unknown")
    if raw:
        return str(raw)
    return "unknown"


def _regime_policy_summary(metadata: dict[str, object]) -> dict[str, object]:
    raw = {}
    if isinstance(metadata, dict):
        raw = metadata.get("regime_policy", metadata.get("regime_policy_audit", {}))
    if not isinstance(raw, dict) or not raw:
        return {
            "available": False,
            "entry_mode": "unknown",
            "risk_multiplier": 1.0,
            "effective_risk_multiplier": 1.0,
            "symbol_health": "unknown",
            "actual_policy_profile": "unknown",
            "actual_policy_decision": "unknown",
            "strict_policy_decision": "unknown",
            "would_strict_policy_trade": True,
            "relaxed_only_trade": False,
            "soft_issues": [],
            "hard_issues": [],
            "tags": [],
            "reasons": [],
        }
    return {
        "available": True,
        "allow_trade": bool(raw.get("allow_trade", False)),
        "entry_mode": str(raw.get("entry_mode", "unknown") or "unknown"),
        "risk_multiplier": float(raw.get("risk_multiplier", 1.0) or 0.0),
        "effective_risk_multiplier": float(raw.get("effective_risk_multiplier", raw.get("risk_multiplier", 1.0)) or 0.0),
        "symbol_health": str(raw.get("symbol_health", "unknown") or "unknown"),
        "actual_policy_profile": str(raw.get("actual_policy_profile", "strict") or "strict"),
        "actual_policy_decision": str(
            raw.get("actual_policy_decision", raw.get("decision_type", "unknown")) or "unknown"
        ),
        "strict_policy_decision": str(raw.get("strict_policy_decision", "unknown") or "unknown"),
        "would_strict_policy_trade": bool(raw.get("would_strict_policy_trade", True)),
        "would_strict_policy_risk_multiplier": float(raw.get("would_strict_policy_risk_multiplier", 0.0) or 0.0),
        "relaxed_only_trade": bool(raw.get("relaxed_only_trade", False)),
        "soft_issues": (
            list(raw.get("soft_issues", []))
            if isinstance(raw.get("soft_issues", []), list)
            else []
        ),
        "hard_issues": (
            list(raw.get("hard_issues", []))
            if isinstance(raw.get("hard_issues", []), list)
            else []
        ),
        "tags": (
            list(raw.get("tags", []))
            if isinstance(raw.get("tags", []), list)
            else []
        ),
        "reasons": (
            list(raw.get("reasons", []))
            if isinstance(raw.get("reasons", []), list)
            else []
        ),
        "risk_components": (
            dict(raw.get("risk_components", {}))
            if isinstance(raw.get("risk_components", {}), dict)
            else {}
        ),
    }


def _learning_caps_summary(metadata: dict[str, object], regime_policy: dict[str, object]) -> dict[str, object]:
    raw: object = {}
    if isinstance(metadata, dict):
        raw = metadata.get("learning_caps", {})
    if not isinstance(raw, dict) or not raw:
        raw = regime_policy.get("learning_caps", {}) if isinstance(regime_policy, dict) else {}
    if not isinstance(raw, dict) or not raw:
        return {
            "available": False,
            "oversampling_tags": [],
            "daily_cap_hit": False,
            "same_symbol_cap_hit": False,
            "same_entry_mode_cap_hit": False,
            "same_regime_cap_hit": False,
            "cap_behavior_applied": "none",
        }
    tags = raw.get("oversampling_tags", [])
    return {
        "available": True,
        "mode": str(raw.get("mode", "")),
        "decision_type": str(raw.get("decision_type", "")),
        "probe_count_today": int(raw.get("probe_count_today", 0) or 0),
        "exploration_count_today": int(raw.get("exploration_count_today", 0) or 0),
        "same_symbol_count_today": int(raw.get("same_symbol_count_today", 0) or 0),
        "same_entry_mode_count_today": int(raw.get("same_entry_mode_count_today", 0) or 0),
        "same_regime_count_today": int(raw.get("same_regime_count_today", 0) or 0),
        "daily_cap_hit": bool(raw.get("daily_cap_hit", False)),
        "same_symbol_cap_hit": bool(raw.get("same_symbol_cap_hit", False)),
        "same_entry_mode_cap_hit": bool(raw.get("same_entry_mode_cap_hit", False)),
        "same_regime_cap_hit": bool(raw.get("same_regime_cap_hit", False)),
        "cap_behavior_applied": str(raw.get("cap_behavior_applied", "none") or "none"),
        "oversampling_tags": list(tags) if isinstance(tags, list) else [],
        "original_quantity_lots": int(raw.get("original_quantity_lots", 0) or 0),
        "adjusted_quantity_lots": int(raw.get("adjusted_quantity_lots", 0) or 0),
        "size_multiplier": float(raw.get("size_multiplier", 1.0) or 0.0),
    }


def _entry_mode_summary(metadata: dict[str, object], regime_policy: dict[str, object]) -> str:
    raw = metadata.get("entry_mode") if isinstance(metadata, dict) else None
    if raw:
        return str(raw)
    policy_mode = regime_policy.get("entry_mode")
    if policy_mode:
        return str(policy_mode)
    pending = metadata.get("pending_entry", {}) if isinstance(metadata, dict) else {}
    if isinstance(pending, dict) and pending.get("outcome") == "triggered":
        return "pullback_short"
    return "unknown"


def _symbol_health_summary(metadata: dict[str, object], regime_policy: dict[str, object]) -> str:
    raw = metadata.get("symbol_health") if isinstance(metadata, dict) else None
    if raw:
        return str(raw)
    policy_health = regime_policy.get("symbol_health")
    if policy_health:
        return str(policy_health)
    return "unknown"


def _rebound_outcome_summary(metadata: dict[str, object]) -> str:
    pending = metadata.get("pending_entry", {}) if isinstance(metadata, dict) else {}
    if not isinstance(pending, dict) or not pending:
        return "none"
    if pending.get("outcome"):
        return str(pending.get("outcome"))
    if pending.get("triggered_at"):
        return "triggered"
    return "pending"


def _trade_source(trade: TradeRecord) -> str:
    metadata = trade.entry_metadata if isinstance(trade.entry_metadata, dict) else {}
    shadow_type = str(metadata.get("shadow_type", "") or "")
    if shadow_type == "policy_rejected":
        return "policy_rejected_shadow"
    if metadata.get("shadow_trade_id"):
        return "policy_shadow"
    if trade.reason == "shadow-feedback" or trade.entry_reason == "shadow-feedback":
        return "strategy_shadow"
    return "executed"


def _nearest_microstructure_row(
    microstructure_dir: Path | None,
    *,
    symbol: str,
    entry_time: datetime,
    max_lag_seconds: float,
) -> dict[str, object] | None:
    if microstructure_dir is None:
        return None
    entry_utc = entry_time.astimezone(timezone.utc)
    candidates = [
        microstructure_dir / entry_utc.strftime("%Y%m%d") / f"{symbol}.jsonl",
        microstructure_dir / (entry_utc - timedelta(days=1)).strftime("%Y%m%d") / f"{symbol}.jsonl",
        microstructure_dir / (entry_utc + timedelta(days=1)).strftime("%Y%m%d") / f"{symbol}.jsonl",
    ]

    best_row: dict[str, object] | None = None
    best_lag: float | None = None
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            row_time = _parse_timestamp(row.get("collected_at")) or _parse_timestamp(row.get("timestamp"))
            if row_time is None:
                continue
            lag = abs((row_time - entry_utc).total_seconds())
            if best_lag is None or lag < best_lag:
                best_lag = lag
                best_row = row
    if best_row is None or best_lag is None or best_lag > max_lag_seconds:
        return None
    return best_row


def _with_trade_side_microstructure(
    microstructure: dict[str, object],
    trade: TradeRecord,
) -> dict[str, object]:
    if not microstructure.get("available"):
        return microstructure
    result = dict(microstructure)
    requested_lots = int(float(result.get("requested_lots", 0.0) or 0.0))
    existing_cover = float(result.get("entry_liquidity_cover", 0.0) or 0.0)
    has_side_imbalance = "side_imbalance" in result
    if result.get("source") != "collector" and existing_cover > 0 and has_side_imbalance:
        if float(result.get("estimated_spread_cost_bps", 0.0) or 0.0) <= 0:
            result["estimated_spread_cost_bps"] = round(float(result.get("spread_bps", 0.0) or 0.0) / 2, 4)
        return result

    if requested_lots <= 0:
        requested_lots = max(1, int(trade.quantity_lots))
        result["requested_lots"] = requested_lots

    bid_depth_lots = float(result.get("bid_depth_lots", 0.0) or 0.0)
    ask_depth_lots = float(result.get("ask_depth_lots", 0.0) or 0.0)
    bid_depth_rub = float(result.get("bid_depth_rub", 0.0) or 0.0)
    ask_depth_rub = float(result.get("ask_depth_rub", 0.0) or 0.0)
    best_bid = float(result.get("best_bid", 0.0) or 0.0)
    best_ask = float(result.get("best_ask", 0.0) or 0.0)
    imbalance = float(result.get("imbalance", 0.0) or 0.0)

    if trade.direction.value == "short":
        entry_depth_lots = bid_depth_lots
        entry_depth_rub = bid_depth_rub
        best_executable_price = best_bid
        side_imbalance = -imbalance
    else:
        entry_depth_lots = ask_depth_lots
        entry_depth_rub = ask_depth_rub
        best_executable_price = best_ask
        side_imbalance = imbalance

    result["entry_depth_lots"] = round(entry_depth_lots, 4)
    result["entry_depth_rub"] = round(entry_depth_rub, 2)
    result["entry_liquidity_cover"] = round(entry_depth_lots / requested_lots, 4)
    result["best_executable_price"] = round(best_executable_price, 8)
    result["side_imbalance"] = round(side_imbalance, 4)
    if float(result.get("estimated_spread_cost_bps", 0.0) or 0.0) <= 0:
        result["estimated_spread_cost_bps"] = round(float(result.get("spread_bps", 0.0) or 0.0) / 2, 4)
    return result


def _microstructure_quality(microstructure: dict[str, object]) -> str:
    if not microstructure.get("available"):
        return "missing"
    spread_bps = float(microstructure.get("spread_bps", 0.0))
    cover = float(microstructure.get("entry_liquidity_cover", 0.0))
    side_imbalance = float(microstructure.get("side_imbalance", 0.0))
    if spread_bps > 12.0:
        return "wide-spread"
    if cover < 2.0:
        return "thin-book"
    if side_imbalance < -0.35:
        return "adverse-imbalance"
    return "clean"


def _mistake_tags(
    trade: TradeRecord,
    *,
    holding_minutes: float,
    planned_risk_rub: float | None,
    planned_rr: float | None,
    current_min_signal_strength: float,
    microstructure: dict[str, object],
    ml_learning: dict[str, object],
    market_regime: str,
    entry_mode: str,
    symbol_health: str,
    rebound_outcome: str,
) -> list[str]:
    tags: list[str] = []
    if planned_rr is not None and planned_rr < 1.2:
        tags.append("poor-reward-risk-plan")

    if trade.net_pnl > 0:
        if trade.reason == "take-profit":
            tags.append("planned-take-profit")
        elif trade.reason in {"breakeven-stop", "profit-protect-stop"}:
            tags.append("protected-profit")
        elif trade.reason == "stop-loss":
            tags.append("profitable-stop-exit")
        return tags

    if trade.net_pnl == 0:
        tags.append("flat-trade")
        return tags

    if trade.reason == "stop-loss":
        tags.append("stop-loss")
    elif trade.reason == "signal-flip":
        tags.append("trend-flip-after-entry")
    elif trade.reason == "session-flat":
        tags.append("forced-session-exit-loss")
    elif trade.reason == "risk-halt":
        tags.append("risk-halt-loss")

    if trade.signal_strength <= max(0.55, current_min_signal_strength + 0.05):
        tags.append("weak-signal-loss")
    if planned_risk_rub is not None and planned_risk_rub > 0 and abs(trade.net_pnl) >= planned_risk_rub * 0.9:
        tags.append("full-risk-loss")
    if holding_minutes <= 90:
        tags.append("fast-loss")
    if abs(trade.entry_context_score) < 0.05:
        tags.append("low-context-edge-loss")
    if market_regime == "weak_down_choppy" and trade.direction.value == "short" and entry_mode == "trend_short":
        tags.append("weak-down-choppy-trend-short-loss")
    if rebound_outcome == "triggered" and entry_mode == "pullback_short":
        tags.append("failed-rebound-trigger-loss")
    if symbol_health == "probation":
        tags.append("symbol-probation-loss")
    learning_tags = ml_learning.get("learning_tags", [])
    if isinstance(learning_tags, list):
        if LOW_QUALITY_TAG in learning_tags:
            tags.append("low-quality-ml-learning-loss")
        if NEGATIVE_EXPECTANCY_TAG in learning_tags:
            tags.append("negative-expectancy-ml-learning-loss")
        if COMMISSION_EDGE_TAG in learning_tags:
            tags.append("commission-edge-ml-learning-loss")
        if CONFIRMATION_AFTER_IMPULSE_TAG in learning_tags:
            tags.append("confirmation-after-impulse-loss")
        if SHORT_AFTER_EXHAUSTION_TAG in learning_tags:
            tags.append("short-after-exhaustion-loss")
        if LATE_REENTRY_TAG in learning_tags:
            tags.append("late-reentry-loss")
    if not microstructure.get("available"):
        tags.append("no-order-book-at-entry")
    else:
        if float(microstructure.get("spread_bps", 0.0)) > 12.0:
            tags.append("wide-spread-entry")
        if float(microstructure.get("entry_liquidity_cover", 0.0)) < 2.0:
            tags.append("thin-book-entry")
        if float(microstructure.get("side_imbalance", 0.0)) < -0.35:
            tags.append("adverse-book-imbalance")
    return tags


def _lesson_for_tag(tag: str) -> str:
    lessons = {
        "poor-reward-risk-plan": (
            "Entry had weak planned reward/risk; prefer setups with cleaner upside versus stop distance."
        ),
        "planned-take-profit": "Exit matched the initial profit plan; keep this setup in the evidence pool.",
        "protected-profit": (
            "Protection logic preserved gains; keep tracking whether trailing parameters are too tight."
        ),
        "profitable-stop-exit": "Stop acted as profit protection; classify separately from true stop-losses.",
        "flat-trade": "Trade did not pay enough to cover opportunity cost; review entry timing and exit rule.",
        "stop-loss": "Initial trade idea was invalidated by price; review entry filter and stop placement.",
        "trend-flip-after-entry": "Signal reversed after entry; trend filter or confirmation may be too loose.",
        "forced-session-exit-loss": (
            "Session flattening closed a loser; check whether entries occur too late in the window."
        ),
        "risk-halt-loss": "Loss happened during risk halt flow; reduce risk until recent expectancy recovers.",
        "weak-signal-loss": "Loss came from a low-strength signal; candidate fix is raising min_signal_strength.",
        "full-risk-loss": (
            "Trade consumed nearly the full planned risk; position size or setup filter needs tightening."
        ),
        "fast-loss": "Trade failed quickly after entry; entry timing or breakout confirmation is likely weak.",
        "low-context-edge-loss": "Context score was neutral; require stronger context before taking similar trades.",
        "low-quality-ml-learning-loss": (
            "ML marked this setup as low-quality; keep trading it only as observe-only evidence until enough "
            "samples accumulate."
        ),
        "negative-expectancy-ml-learning-loss": (
            "ML expected negative per-lot expectancy; compare future samples before converting this into a hard rule."
        ),
        "commission-edge-ml-learning-loss": (
            "ML expected profit did not clear the required net edge after round-turnover commission."
        ),
        "confirmation-after-impulse-loss": (
            "Entry followed an impulse/reversal candle; require evidence that the next candle confirms continuation "
            "before scaling this pattern."
        ),
        "short-after-exhaustion-loss": (
            "Short was opened after an exhausted down move or rebound; require continuation confirmation or "
            "reduce size."
        ),
        "late-reentry-loss": (
            "Repeated same-direction entry after an earlier winner failed; require a fresh confirmed setup before "
            "re-entering."
        ),
        "weak-down-choppy-trend-short-loss": (
            "Weak down/choppy regime punished direct trend-short entry; defer until a rebound fails before entering."
        ),
        "failed-rebound-trigger-loss": (
            "Pullback-short trigger still failed; review trigger distance, rebound quality, and stop placement before "
            "scaling it."
        ),
        "symbol-probation-loss": (
            "Symbol was already under probation; keep reduced size until recent expectancy improves."
        ),
        "no-order-book-at-entry": (
            "Entry had no order book snapshot; the bot could not judge spread, depth, or imbalance."
        ),
        "wide-spread-entry": "Entry spread was wide; require a tighter book before taking similar trades.",
        "thin-book-entry": "Entry-side depth was thin versus order size; reduce size or skip similar trades.",
        "adverse-book-imbalance": "Order book imbalance was against the trade direction at entry.",
    }
    return lessons.get(tag, "Review this pattern before allowing the same setup to scale.")


def _build_recommendations(
    reviews: list[dict[str, object]],
    *,
    strategy: StrategySection,
    risk: RiskSection,
) -> dict[str, object]:
    recommendations: list[dict[str, object]] = []
    patch: dict[str, object] = {"strategy": {}, "risk": {}}
    losing_reviews = [review for review in reviews if float(review["net_pnl_rub"]) < 0]
    if not reviews:
        return {
            "items": [
                {
                    "action": "collect-more-trades",
                    "confidence": "low",
                    "reason": "No closed trades are available for error review.",
                }
            ],
            "config_patch_candidates": {},
        }

    tag_counts = Counter(tag for review in reviews for tag in review.get("mistake_tags", []))
    weak_signal_losses = [
        review
        for review in losing_reviews
        if "weak-signal-loss" in review.get("mistake_tags", [])
        and float(review["signal_strength"]) >= strategy.min_signal_strength
    ]
    if weak_signal_losses:
        worst_kept_strength = max(float(review["signal_strength"]) for review in weak_signal_losses)
        recommended_strength = min(0.95, max(strategy.min_signal_strength + 0.05, worst_kept_strength + 0.01))
        recommended_strength = _ceil_to_step(recommended_strength, 0.05)
        patch["strategy"]["min_signal_strength"] = recommended_strength
        recommendations.append(
            {
                "action": "raise-min-signal-strength",
                "confidence": _confidence(len(weak_signal_losses)),
                "reason": f"{len(weak_signal_losses)} losing trade(s) came from weak signals.",
                "patch": {"strategy.min_signal_strength": recommended_strength},
            }
        )

    stop_loss_count = tag_counts.get("stop-loss", 0)
    full_risk_count = tag_counts.get("full-risk-loss", 0)
    if stop_loss_count >= 2 or full_risk_count >= 2:
        reduced_risk = max(0.0025, round(risk.max_risk_per_trade * 0.8, 4))
        patch["risk"]["max_risk_per_trade"] = reduced_risk
        recommendations.append(
            {
                "action": "reduce-risk-per-trade",
                "confidence": _confidence(max(stop_loss_count, full_risk_count)),
                "reason": "Recent trades include repeated stop/full-risk losses.",
                "patch": {"risk.max_risk_per_trade": reduced_risk},
            }
        )

    poor_rr_count = tag_counts.get("poor-reward-risk-plan", 0)
    if poor_rr_count >= 2:
        reward_to_risk = round(max(strategy.reward_to_risk, 1.5), 2)
        patch["strategy"]["reward_to_risk"] = reward_to_risk
        recommendations.append(
            {
                "action": "enforce-better-reward-risk",
                "confidence": _confidence(poor_rr_count),
                "reason": "Multiple trades had weak planned reward/risk.",
                "patch": {"strategy.reward_to_risk": reward_to_risk},
            }
        )

    weak_choppy_trend_short_count = tag_counts.get("weak-down-choppy-trend-short-loss", 0)
    if weak_choppy_trend_short_count >= 1:
        recommendations.append(
            {
                "action": "defer-weak-down-choppy-trend-shorts",
                "confidence": _confidence(weak_choppy_trend_short_count),
                "reason": (
                    "Weak down/choppy losers came from direct trend-short entries; wait for a failed rebound "
                    "before entry."
                ),
                "runtime_policy": {
                    "regime": "weak_down_choppy",
                    "blocked_entry_mode": "trend_short",
                    "allowed_entry_mode": "pullback_short",
                },
            }
        )

    failed_rebound_count = tag_counts.get("failed-rebound-trigger-loss", 0)
    if failed_rebound_count >= 1:
        recommendations.append(
            {
                "action": "tighten-pullback-short-trigger",
                "confidence": _confidence(failed_rebound_count),
                "reason": (
                    "A deferred pullback-short trigger still lost; require better rebound failure or smaller size."
                ),
                "runtime_policy": {
                    "entry_mode": "pullback_short",
                    "risk_multiplier_cap": 0.5,
                    "review_trigger_distance": True,
                },
            }
        )

    weak_symbols = _negative_groups(reviews, "symbol", min_trades=2)
    if weak_symbols:
        additions = [item["group"] for item in weak_symbols[:2]]
        recommendations.append(
            {
                "action": "observe-weak-symbols",
                "confidence": _confidence(max(int(item["trades"]) for item in weak_symbols)),
                "reason": (
                    "One or more symbols show negative recent expectancy; keep them in learning mode instead "
                    "of blocking."
                ),
                "symbols": additions,
            }
        )

    impulse_loss_count = tag_counts.get("confirmation-after-impulse-loss", 0)
    if impulse_loss_count >= 1:
        recommendations.append(
            {
                "action": "learn-confirmation-after-impulse",
                "confidence": _confidence(impulse_loss_count),
                "reason": (
                    "Recent loss followed an impulse/reversal entry; collect more observe-only samples before "
                    "enforcing confirmation."
                ),
                "mode": "observe_only",
                "blocks_entry": False,
            }
        )

    no_book_count = tag_counts.get("no-order-book-at-entry", 0)
    microstructure_loss_count = (
        tag_counts.get("wide-spread-entry", 0)
        + tag_counts.get("thin-book-entry", 0)
        + tag_counts.get("adverse-book-imbalance", 0)
    )
    if no_book_count >= 1:
        recommendations.append(
            {
                "action": "collect-order-book-evidence",
                "confidence": "medium",
                "reason": f"{no_book_count} losing trade(s) had no order book snapshot.",
            }
        )
    if microstructure_loss_count >= 1:
        recommendations.append(
            {
                "action": "require-clean-microstructure",
                "confidence": _confidence(microstructure_loss_count),
                "reason": "Recent losers show weak spread/depth/imbalance at entry.",
                "rules": {
                    "max_entry_spread_bps": 12.0,
                    "min_entry_liquidity_cover": 2.0,
                    "min_entry_book_imbalance": -0.35,
                },
            }
        )

    weak_hours = _negative_groups(reviews, "entry_hour", min_trades=2)
    current_hours = list(strategy.allowed_entry_hours)
    removable_hours = [
        int(item["group"])
        for item in weak_hours
        if current_hours and int(item["group"]) in current_hours
    ]
    if removable_hours:
        allowed_hours = [hour for hour in current_hours if hour not in removable_hours[:2]]
        patch["strategy"]["allowed_entry_hours"] = allowed_hours
        recommendations.append(
            {
                "action": "remove-weak-entry-hours",
                "confidence": _confidence(max(int(item["trades"]) for item in weak_hours)),
                "reason": "Recent entry-hour expectancy is negative.",
                "hours": removable_hours[:2],
                "patch": {"strategy.allowed_entry_hours": allowed_hours},
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "action": "keep-current-rules",
                "confidence": "medium" if len(reviews) >= 5 else "low",
                "reason": "No repeated loss pattern is strong enough to change config yet.",
            }
        )

    clean_patch = {
        section: values
        for section, values in patch.items()
        if values
    }
    return {"items": recommendations, "config_patch_candidates": clean_patch}


def _summary(reviews: list[dict[str, object]]) -> dict[str, object]:
    if not reviews:
        return {
            "trades": 0,
            "net_pnl_rub": 0.0,
            "win_rate_pct": 0.0,
            "expectancy_rub": 0.0,
            "mistake_trades": 0,
        }
    pnl_values = [float(review["net_pnl_rub"]) for review in reviews]
    wins = [value for value in pnl_values if value > 0]
    mistake_trades = [
        review
        for review in reviews
        if any(_is_error_tag(tag) for tag in review.get("mistake_tags", []))
    ]
    return {
        "trades": len(reviews),
        "wins": len(wins),
        "losses": len([value for value in pnl_values if value < 0]),
        "net_pnl_rub": round(sum(pnl_values), 2),
        "win_rate_pct": round(len(wins) / len(reviews) * 100, 3),
        "expectancy_rub": round(sum(pnl_values) / len(reviews), 2),
        "mistake_trades": len(mistake_trades),
    }


def _group_breakdown(reviews: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[object, list[dict[str, object]]] = defaultdict(list)
    for review in reviews:
        grouped[review.get(key, "")].append(review)
    rows = [_group_row(group, items) for group, items in grouped.items()]
    rows.sort(key=lambda item: (float(item["net_pnl_rub"]), str(item["group"])))
    return rows


def _group_row(group: object, reviews: list[dict[str, object]]) -> dict[str, object]:
    pnl_values = [float(review["net_pnl_rub"]) for review in reviews]
    wins = [value for value in pnl_values if value > 0]
    return {
        "group": group,
        "trades": len(reviews),
        "net_pnl_rub": round(sum(pnl_values), 2),
        "expectancy_rub": round(sum(pnl_values) / len(reviews), 2),
        "win_rate_pct": round(len(wins) / len(reviews) * 100, 3),
    }


def _policy_signal_distribution(events: list[dict[str, object]]) -> dict[str, object]:
    rows: dict[tuple[str, str, str], int] = Counter()
    relaxed_only = 0
    total = 0
    for event in events:
        if event.get("action") != "signal":
            continue
        policy_fields = _event_policy_fields(event)
        if not policy_fields:
            continue
        total += 1
        if policy_fields.get("relaxed_only_trade"):
            relaxed_only += 1
        rows[
            (
                str(policy_fields.get("actual_policy_decision", "unknown")),
                str(policy_fields.get("strict_policy_decision", "unknown")),
                "approved" if bool(event.get("approved")) else "rejected",
            )
        ] += 1
    return {
        "total_policy_signals": total,
        "relaxed_only_signals": relaxed_only,
        "rows": [
            {
                "actual_policy_decision": actual,
                "strict_policy_decision": strict,
                "outcome": outcome,
                "signals": count,
            }
            for (actual, strict, outcome), count in sorted(rows.items())
        ],
    }


def _policy_outcomes(reviews: list[dict[str, object]]) -> dict[str, object]:
    return {
        "by_trade_source": _group_breakdown(reviews, "trade_source"),
        "by_actual_policy_decision": _group_breakdown(reviews, "actual_policy_decision"),
        "by_strict_policy_decision": _group_breakdown(reviews, "strict_policy_decision"),
        "resolved_shadow_trades": sum(
            1
            for review in reviews
            if str(review.get("trade_source", "")).endswith("shadow")
            or "shadow" in str(review.get("trade_source", ""))
        ),
        "relaxed_only_trades": sum(1 for review in reviews if bool(review.get("relaxed_only_trade"))),
    }


def _selloff_capture_review(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    detections = [event for event in events if event.get("action") == "market_selloff_impulse_detected"]
    selloff_candidates = [event for event in events if event.get("action") == "selloff_short_candidate"]
    opened_events = [event for event in events if event.get("action") == "selloff_short_opened"]
    rejected_events = [event for event in events if event.get("action") == "selloff_short_rejected"]
    budget_events = [
        event
        for event in events
        if event.get("action") in {"selloff_budget_used", "selloff_budget_unused", "selloff_underallocated"}
    ]
    selloff_reviews = [review for review in reviews if review.get("market_regime") == "market_selloff_impulse"]
    selloff_short_reviews = [
        review
        for review in selloff_reviews
        if review.get("direction") == "short"
    ]
    detected_at = _first_event_iso(detections)
    first_trade_at = _first_event_iso(opened_events) or _first_review_entry_iso(selloff_short_reviews)
    latency_minutes = _latency_minutes(detected_at, first_trade_at)
    diagnostics = _latest_selloff_diagnostics(budget_events)
    return {
        "selloff_windows": len(detections),
        "selloff_detected_at": detected_at,
        "first_selloff_trade_at": first_trade_at,
        "latency_minutes": latency_minutes,
        "candidates_count": len(selloff_candidates),
        "trades_opened": len(opened_events),
        "gross_exposure_at_detection": diagnostics.get("gross_exposure_pct", 0.0),
        "gross_exposure_peak_during_selloff": max(
            [float(_event_diagnostics(event).get("gross_exposure_pct", 0.0) or 0.0) for event in budget_events],
            default=0.0,
        ),
        "target_gross_exposure": diagnostics.get("selloff_target_gross_exposure", 0.0),
        "budget_used_pct": diagnostics.get("budget_used_pct", 0.0),
        "missed_budget_pct": round(max(0.0, 1.0 - float(diagnostics.get("budget_used_pct", 0.0) or 0.0)), 6),
        "selloff_pnl": _pnl_for_reviews(selloff_short_reviews),
        "selloff_pnl_by_entry_mode": _pnl_by_entry_modes(
            selloff_short_reviews,
            [
                "market_breakdown_short",
                "selloff_momentum_short",
                "panic_probe_short",
                "post_selloff_failed_rebound_short",
            ],
        ),
        "rejection_reasons": dict(
            sorted(Counter(str(event.get("reason", "")) for event in rejected_events if event.get("reason")).items())
        ),
        "wait_only_count_during_selloff": sum(
            1
            for event in selloff_candidates
            if _event_entry_mode(event) == "wait"
            or _event_policy_fields(event).get("actual_policy_decision") == "wait_pullback"
        ),
    }


def _short_only_review(
    portfolio: PortfolioState,
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    enabled = _short_only_enabled(reviews, events)
    candidate_events = [
        event
        for event in events
        if event.get("action") in {"short_only_short_candidate", "short_only_upsize_candidate"}
    ]
    short_only_signals = [
        event
        for event in events
        if event.get("action") == "signal" and _event_short_only(event).get("enabled") is True
    ]
    approved = [event for event in short_only_signals if bool(event.get("approved", False))]
    blocked = [event for event in short_only_signals if not bool(event.get("approved", False))]
    short_only_reviews = [
        review
        for review in reviews
        if isinstance(review.get("entry_metadata", {}), dict)
        and isinstance(review.get("entry_metadata", {}).get("short_only", {}), dict)
        and bool(review.get("entry_metadata", {}).get("short_only", {}).get("enabled", False))
    ]
    expected_edges = [
        float(event.get("expected_net_edge_rub", 0.0) or 0.0)
        for event in candidate_events
        if bool(event.get("edge_gate_passed", False))
    ]
    budget_events = [event for event in events if event.get("action") == "short_only_budget_allocation"]
    latest_budget = _latest_short_only_budget(budget_events)
    hard_reasons = Counter(
        str(reason)
        for event in blocked
        for reason in _short_only_hard_reasons(event)
        if str(reason).strip()
    )
    open_short_only_positions = [
        position
        for position in portfolio.positions.values()
        if position.direction.value == "short"
    ]
    soft_reduced = [
        event
        for event in candidate_events
        if (
            _event_short_only(event).get("size_multiplier") is not None
            and float(_event_short_only(event).get("size_multiplier", 1.0) or 1.0) < 1.0
            and not _short_only_hard_reasons(event)
        )
    ]
    return {
        "short_only_enabled": enabled,
        "active": enabled,
        "live_disabled": True,
        "active_regime": _latest_market_regime(events),
        "effective_regime": _latest_effective_regime(events),
        "long_signals_ignored": sum(1 for event in events if event.get("action") == "long_signal_ignored_short_only"),
        "longs_flattened": sum(1 for event in events if event.get("action") == "long_position_flattened_short_only"),
        "range_chop_no_trade_count": sum(1 for event in events if event.get("action") == "range_chop_no_trade_short_only"),
        "no_trade_range_chop_count": sum(1 for event in events if event.get("action") == "range_chop_no_trade_short_only"),
        "short_candidates_total": len(candidate_events),
        "strategy_short_candidates": sum(
            1 for event in candidate_events if not bool(_event_short_only(event).get("synthetic_candidate", False))
        ),
        "synthetic_short_candidates": sum(
            1 for event in candidate_events if bool(_event_short_only(event).get("synthetic_candidate", False))
        ),
        "upsize_candidates": sum(1 for event in events if event.get("action") == "short_only_upsize_candidate"),
        "positive_ev_short_candidates": sum(1 for event in candidate_events if bool(event.get("edge_gate_passed", False))),
        "positive_ev_candidates": sum(1 for event in candidate_events if bool(event.get("edge_gate_passed", False))),
        "shorts_opened": len(approved),
        "shorts_upsized": sum(1 for event in events if event.get("action") == "short_only_upsize_opened"),
        "shorts_blocked_hard": len(blocked),
        "hard_blocked_candidates": len(blocked),
        "soft_reduced_candidates": len(soft_reduced),
        "average_expected_edge": round(sum(expected_edges) / len(expected_edges), 2) if expected_edges else 0.0,
        "realized_pnl_short_only": _pnl_for_reviews(short_only_reviews),
        "open_pnl_short_only": round(sum(position.unrealized_pnl(position.current_price) for position in open_short_only_positions), 2),
        "source_validation": _short_only_source_validation(short_only_reviews, events),
        "disabled_sources": _short_only_disabled_sources(events),
        "pnl_by_source": _pnl_by_short_only_source(short_only_reviews),
        "pnl_by_regime": _group_breakdown(short_only_reviews, "market_regime") if short_only_reviews else [],
        "pnl_by_effective_regime": _pnl_by_short_only_metadata_key(short_only_reviews, "effective_regime"),
        "pnl_by_edge_source": _pnl_by_short_only_metadata_key(short_only_reviews, "edge_source"),
        "pnl_by_edge_bucket": _pnl_by_short_only_edge_bucket(short_only_reviews),
        "pnl_by_confirmation": _pnl_by_short_only_metadata_key(short_only_reviews, "confirmation_status"),
        "pnl_by_expansion": _pnl_by_short_only_expansion_bucket(short_only_reviews),
        "pnl_by_synthetic_reason": _pnl_by_short_only_metadata_key(short_only_reviews, "synthetic_reason"),
        "budget_target": latest_budget.get("budget_target_gross_rub", 0.0),
        "budget_used": latest_budget.get("budget_used_gross_rub", 0.0),
        "budget_target_gross": latest_budget.get("budget_target_gross_rub", 0.0),
        "budget_peak_gross": latest_budget.get("budget_used_gross_rub", 0.0),
        "budget_used_pct": latest_budget.get("budget_used_pct", 0.0),
        "underallocated_count": sum(1 for event in events if event.get("action") == "short_only_underallocated"),
        "underallocated_reasons": _short_only_underallocated_reasons(events),
        "strong_rebound_reduced_count": sum(
            1
            for event in candidate_events
            if _event_short_only(event).get("confirmation_status") == "strong_rebound"
            and not _short_only_hard_reasons(event)
        ),
        "extreme_adverse_block_count": sum(
            1 for event in blocked if "extreme adverse" in str(event.get("reason", "")).lower()
        ),
        "microstructure_soft_reduced_count": sum(
            1 for event in candidate_events if bool(_event_short_only(event).get("microstructure_soft_reasons", []))
        ),
        "microstructure_hard_block_count": sum(
            1 for event in blocked if "microstructure" in str(event.get("reason", "")).lower()
        ),
        "early_loss_guard_exits": sum(1 for event in events if event.get("action") == "short_only_early_loss_guard_exit"),
        "breakeven_stop_armed_count": sum(1 for event in events if event.get("action") == "short_only_breakeven_stop_armed"),
        "top_hard_block_reasons": dict(sorted(hard_reasons.items(), key=lambda item: (-item[1], item[0]))[:10]),
    }


def _short_ev_review(
    portfolio: PortfolioState,
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    del portfolio
    config_events = [event for event in events if event.get("action") == "short_ev_engine_config"]
    latest_config = config_events[-1] if config_events else {}
    allowed = latest_config.get("allowed_setups", list(ALLOWED_SHORT_SETUPS))
    if not isinstance(allowed, list):
        allowed = list(ALLOWED_SHORT_SETUPS)
    allowed_setups = [canonical_setup_id(item) for item in allowed if canonical_setup_id(item)]
    candidate_events = [
        event
        for event in events
        if event.get("action") in {"short_only_short_candidate", "short_only_upsize_candidate"}
        and _event_short_only(event).get("short_ev_engine_enabled") is True
    ]
    signal_events = [
        event
        for event in events
        if event.get("action") == "signal" and _event_short_only(event).get("short_ev_engine_enabled") is True
    ]
    short_ev_reviews = [
        review
        for review in reviews
        if _review_short_ev_engine(review)
        or canonical_setup_id(_review_short_only(review).get("setup_id", "")) in set(allowed_setups)
    ]
    shadow_events = [
        event
        for event in events
        if "shadow" in str(event.get("action", ""))
        and _event_short_only(event).get("short_ev_engine_enabled") is True
    ]
    setup_registry = []
    ev_by_setup = []
    for setup_id in allowed_setups:
        setup_candidates = [event for event in candidate_events if _event_setup_id(event) == setup_id]
        setup_signals = [event for event in signal_events if _event_setup_id(event) == setup_id]
        setup_reviews = [review for review in short_ev_reviews if _review_setup_id(review) == setup_id]
        real_count = sum(
            1
            for event in setup_signals
            if bool(event.get("approved", False))
            and _event_short_only(event).get("short_ev_decision") == "real_allowed"
        )
        probe_count = sum(
            1
            for event in setup_signals
            if bool(event.get("approved", False))
            and _event_short_only(event).get("short_ev_decision") == "probe_allowed"
        )
        shadow_count = sum(1 for event in setup_signals if not bool(event.get("approved", False)))
        setup_registry.append(
            {
                "setup_id": setup_id,
                "enabled_real": True,
                "enabled_probe": True,
                "enabled_shadow": True,
                "candidate_count": len(setup_candidates),
                "real_count": real_count,
                "probe_count": probe_count,
                "shadow_count": shadow_count,
            }
        )
        pnl = _pnl_for_reviews(setup_reviews)
        ev_rows = [_event_short_only(event) for event in setup_candidates]
        avg_ev = _avg([_float(row.get("ev_net_rub")) for row in ev_rows])
        avg_ev_per_risk = _avg([_float(row.get("ev_per_risk")) for row in ev_rows])
        avg_cost = _avg(
            [
                _float(row.get("costs", {}).get("total_cost_rub"))
                for row in ev_rows
                if isinstance(row.get("costs", {}), dict)
            ]
        )
        avg_confidence = _avg([_float(row.get("ev_confidence")) for row in ev_rows])
        wins = int(pnl.get("wins", 0)) if isinstance(pnl, dict) else 0
        losses = int(pnl.get("losses", 0)) if isinstance(pnl, dict) else 0
        trades = int(pnl.get("trades", 0)) if isinstance(pnl, dict) else 0
        ev_by_setup.append(
            {
                "setup_id": setup_id,
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / trades, 6) if trades else 0.0,
                "avg_win_net": _avg([float(review.get("net_pnl_rub", 0.0)) for review in setup_reviews if float(review.get("net_pnl_rub", 0.0)) > 0]),
                "avg_loss_net": _avg([float(review.get("net_pnl_rub", 0.0)) for review in setup_reviews if float(review.get("net_pnl_rub", 0.0)) < 0]),
                "ev_net": round(avg_ev, 2),
                "ev_per_risk": round(avg_ev_per_risk, 6),
                "costs_avg": round(avg_cost, 2),
                "confidence": round(avg_confidence, 6),
                "real_enabled_reason": _setup_real_enabled_reason(setup_candidates, setup_reviews),
                "pnl": pnl,
                "avg_mfe_pct": _avg([_review_mfe_pct(review) for review in setup_reviews]),
                "avg_mae_pct": _avg([_review_mae_pct(review) for review in setup_reviews]),
            }
        )
    cost_rows = [
        _event_short_only(event).get("costs", {})
        for event in candidate_events
        if isinstance(_event_short_only(event).get("costs", {}), dict)
    ]
    gross_pnl = sum(abs(float(review.get("gross_pnl_rub", 0.0) or 0.0)) for review in short_ev_reviews)
    costs_total = {
        "commission_total": round(
            sum(_float(row.get("entry_commission_rub")) + _float(row.get("exit_commission_rub")) for row in cost_rows),
            2,
        ),
        "spread_cost_total": round(sum(_float(row.get("spread_cost_rub")) for row in cost_rows), 2),
        "slippage_total": round(sum(_float(row.get("slippage_cost_rub")) for row in cost_rows), 2),
    }
    total_costs = sum(float(value) for value in costs_total.values())
    costs_total["costs_as_pct_of_gross"] = round(total_costs / gross_pnl * 100.0, 4) if gross_pnl > 0 else 0.0
    return {
        "enabled": bool(config_events or candidate_events or short_ev_reviews),
        "mode": latest_config.get("mode", ""),
        "allow_live_trading": latest_config.get("allow_live_trading"),
        "execution_mode": latest_config.get("execution_mode"),
        "allowed_setups": allowed_setups,
        "setup_registry": setup_registry,
        "ev_by_setup": ev_by_setup,
        "entry_source": {
            setup_id: sum(1 for event in candidate_events if _event_setup_id(event) == setup_id)
            for setup_id in allowed_setups
        },
        "exit_performance": {
            "breakeven_armed_count": sum(1 for event in events if event.get("action") == "short_net_breakeven_armed"),
            "breakeven_saved_losses": sum(
                1
                for review in short_ev_reviews
                if str(review.get("exit_reason", "")) in {"breakeven-stop", "profit-protect-stop"}
            ),
            "trailing_activated_count": sum(1 for event in events if event.get("action") == "short_ev_trailing_stop_updated"),
            "trailing_exit_pnl": _pnl_for_reviews(
                [
                    review
                    for review in short_ev_reviews
                    if str(review.get("exit_reason", "")) in {"profit-protect-stop", "take-profit-runner"}
                ]
            ),
            "order_book_tightening_count": sum(1 for event in events if event.get("action") == "short_ev_order_book_tightening"),
            "full_risk_loss_count": sum(1 for review in short_ev_reviews if str(review.get("exit_reason", "")) == "stop-loss"),
            "fast_loss_count": sum(1 for review in short_ev_reviews if "fast" in str(review.get("exit_reason", ""))),
        },
        "costs": costs_total,
        "shadow_validation": {
            "shadow_count": len(shadow_events),
            "shadow_by_setup": dict(sorted(Counter(_event_setup_id(event) for event in shadow_events).items())),
            "setups_close_to_real_enable": [
                row["setup_id"]
                for row in ev_by_setup
                if float(row.get("ev_net", 0.0) or 0.0) > 0 and float(row.get("confidence", 0.0) or 0.0) < 0.55
            ],
            "setups_to_disable": [
                row["setup_id"]
                for row in ev_by_setup
                if int(row.get("trades", 0) or 0) >= 10 and float(row.get("ev_net", 0.0) or 0.0) < 0
            ],
        },
        "what_to_change_next": _short_ev_recommendations(ev_by_setup),
    }


def _golden_3tf_review(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    config_events = [event for event in events if event.get("action") == "golden_baseline_config"]
    latest_config = config_events[-1] if config_events else {}
    candidate_events = [
        event
        for event in events
        if event.get("action") in {"short_only_short_candidate", "short_only_upsize_candidate"}
        and _event_golden_3tf(event)
    ]
    signal_events = [
        event
        for event in events
        if event.get("action") == "signal" and _event_golden_3tf(event)
    ]
    approved = [event for event in signal_events if bool(event.get("approved", False))]
    rejected = [event for event in signal_events if not bool(event.get("approved", False))]
    shadow_events = [event for event in events if event.get("action") == "golden_baseline_shadow_only"]
    golden_reviews = [review for review in reviews if _review_golden_3tf(review)]
    failed_conditions = Counter(
        str(reason)
        for event in candidate_events + signal_events + shadow_events
        for reason in _golden_failed_conditions(_event_golden_3tf(event))
        if str(reason).strip()
    )
    verdict_counts = Counter(
        str(_event_golden_3tf(event).get("verdict", "unknown") or "unknown")
        for event in candidate_events
    )
    entry_mode_counts = Counter(
        str(_event_golden_3tf(event).get("entry_mode", "unknown") or "unknown")
        for event in candidate_events
    )
    review_modes = sorted(
        {
            str(_review_golden_3tf(review).get("entry_mode", "unknown") or "unknown")
            for review in golden_reviews
        }
    )
    return {
        "enabled": bool(config_events or candidate_events or golden_reviews),
        "source_run": latest_config.get("source_run") or _first_golden_field(candidate_events, "source_run"),
        "source_commit": latest_config.get("source_commit") or _first_golden_field(candidate_events, "source_commit"),
        "timeframes": {
            "primary": latest_config.get("primary_timeframe") or _first_golden_timeframe(candidate_events, "primary"),
            "early_trigger": latest_config.get("early_trigger_timeframe") or _first_golden_timeframe(candidate_events, "early_trigger"),
            "execution_guard": latest_config.get("execution_guard_timeframe") or _first_golden_timeframe(candidate_events, "execution_guard"),
            "forbidden": list(latest_config.get("forbidden_timeframes", []))
            if isinstance(latest_config.get("forbidden_timeframes", []), list)
            else [],
        },
        "allow_live_trading": latest_config.get("allow_live_trading"),
        "execution_mode": latest_config.get("execution_mode"),
        "candidate_events": len(candidate_events),
        "golden_15m_candidates": entry_mode_counts.get("golden_15m_short_breakout", 0),
        "early_5m_starter_candidates": entry_mode_counts.get("early_5m_starter_short", 0),
        "passed_candidates": verdict_counts.get("passed", 0),
        "shadow_only_candidates": verdict_counts.get("shadow_only", 0),
        "approved_signals": len(approved),
        "rejected_signals": len(rejected),
        "shadow_only_events": len(shadow_events),
        "top_failed_conditions": dict(
            sorted(failed_conditions.items(), key=lambda item: (-item[1], item[0]))[:15]
        ),
        "candidate_entry_modes": dict(sorted(entry_mode_counts.items())),
        "candidate_verdicts": dict(sorted(verdict_counts.items())),
        "pnl_total": _pnl_for_reviews(golden_reviews),
        "pnl_by_entry_mode": {
            mode: _pnl_for_reviews(
                [review for review in golden_reviews if str(_review_golden_3tf(review).get("entry_mode", "unknown")) == mode]
            )
            for mode in review_modes
        },
        "pnl_by_verdict": _golden_pnl_by_verdict(golden_reviews),
    }


def _underallocation_review(events: list[dict[str, object]]) -> dict[str, object]:
    underallocated = [event for event in events if event.get("action") == "selloff_underallocated"]
    budget_events = [
        event
        for event in events
        if event.get("action") in {"selloff_budget_used", "selloff_budget_unused", "selloff_underallocated"}
    ]
    budget_used = [float(_event_diagnostics(event).get("budget_used_pct", 0.0) or 0.0) for event in budget_events]
    skipped_symbols = sorted(
        {
            str(event.get("symbol", ""))
            for event in events
            if event.get("action") == "selloff_short_rejected" and str(event.get("symbol", "")).strip()
        }
    )
    reason_counts = Counter(
        str(_event_diagnostics(event).get("unused_budget_reason", event.get("unused_budget_reason", "")))
        for event in budget_events
        if str(_event_diagnostics(event).get("unused_budget_reason", event.get("unused_budget_reason", ""))).strip()
    )
    blocker_counts: Counter[str] = Counter()
    for event in budget_events:
        blockers = _event_diagnostics(event).get("selloff_budget_blockers", {})
        if isinstance(blockers, dict):
            for key, value in blockers.items():
                blocker_counts[str(key)] += int(value or 0)
    return {
        "selloff_underallocated_count": len(underallocated),
        "avg_budget_used_pct": round(sum(budget_used) / len(budget_used), 6) if budget_used else 0.0,
        "max_budget_used_pct": round(max(budget_used), 6) if budget_used else 0.0,
        "reasons_for_unused_budget": dict(sorted(reason_counts.items())),
        "selloff_budget_blockers": dict(sorted(blocker_counts.items())),
        "symbols_skipped_despite_selloff": skipped_symbols,
        "shadow_pnl_for_skipped_candidates": {"available": False, "reason": "shadow PnL unavailable in current cycle events"},
    }


def _long_during_selloff_review(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    long_signal_events = [
        event
        for event in events
        if str(event.get("direction", event.get("side", ""))) == "long"
        and _event_regime(event) == "market_selloff_impulse"
    ]
    selloff_long_reviews = [
        review
        for review in reviews
        if review.get("direction") == "long" and review.get("market_regime") == "market_selloff_impulse"
    ]
    return {
        "long_signals_during_selloff": len(long_signal_events),
        "long_shadow_count": sum(
            1
            for event in long_signal_events
            if _event_policy_fields(event).get("actual_policy_decision") == "shadow_only"
            or event.get("action") == "long_shadow_only_created"
        ),
        "capitulation_bounce_probes": sum(
            1
            for event in long_signal_events
            if _event_entry_mode(event) == "capitulation_bounce_probe_long"
        ),
        "long_pnl_after_reclaim": _pnl_for_reviews(
            [
                review
                for review in selloff_long_reviews
                if review.get("entry_mode") == "capitulation_bounce_probe_long"
            ]
        ),
        "false_bounce_losses": sum(
            1
            for review in selloff_long_reviews
            if review.get("entry_mode") == "capitulation_bounce_probe_long"
            and float(review.get("net_pnl_rub", 0.0) or 0.0) < 0.0
        ),
    }


def _weak_choppy_direct_probe_review(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    signal_events = [event for event in events if event.get("action") == "signal"]
    weak_signals = [event for event in signal_events if _event_regime(event) == "weak_down_choppy"]
    pending_events = [event for event in events if event.get("action") == "pending-entry"]
    weak_reviews = [review for review in reviews if review.get("market_regime") == "weak_down_choppy"]
    return {
        "weak_down_choppy_signals_total": len(weak_signals),
        "weak_down_choppy_wait_only_count": sum(
            1
            for event in weak_signals
            if _event_entry_mode(event) == "wait"
            or _event_policy_fields(event).get("actual_policy_decision") == "wait_pullback"
        ),
        "weak_down_choppy_probe_now_count": sum(
            1 for event in weak_signals if _event_entry_mode(event) == "weak_choppy_direct_probe_short"
        ),
        "weak_down_choppy_exploration_now_count": sum(
            1 for event in weak_signals if _event_entry_mode(event) == "weak_choppy_direct_exploration_short"
        ),
        "weak_down_choppy_pending_addon_created_count": sum(
            1 for event in pending_events if _pending_event_is_addon(event) and event.get("status") == "created"
        ),
        "weak_down_choppy_pending_addon_executed_count": sum(
            1
            for review in reviews
            if review.get("entry_mode") == "pullback_short"
            and _review_pending_entry(review).get("is_addon", False)
        ),
        "weak_down_choppy_pending_addon_expired_count": sum(
            1 for event in pending_events if _pending_event_is_addon(event) and event.get("status") == "expired"
        ),
        "strict_wait_relaxed_probe_count": sum(
            1
            for event in weak_signals
            if _event_policy_fields(event).get("strict_policy_decision") == "wait"
            and _event_policy_fields(event).get("actual_policy_decision")
            in {"probe_trade", "exploration_trade"}
        ),
        "pnl_by_entry_mode": _pnl_by_entry_modes(
            weak_reviews,
            [
                "weak_choppy_direct_probe_short",
                "weak_choppy_direct_exploration_short",
                "wait_pullback_short",
                "pullback_short",
            ],
        ),
    }


def _long_learning_review(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    long_signal_events = [
        event
        for event in events
        if event.get("action") == "signal" and str(event.get("direction", event.get("side", ""))) == "long"
    ]
    long_reviews = [review for review in reviews if review.get("direction") == "long"]
    return {
        "long_signals_total": len(long_signal_events),
        "long_normal_count": _count_signal_decision(long_signal_events, "normal_trade"),
        "long_probe_count": _count_signal_decision(long_signal_events, "probe_trade"),
        "long_exploration_count": _count_signal_decision(long_signal_events, "exploration_trade"),
        "long_shadow_only_count": _count_signal_decision(long_signal_events, "shadow_only"),
        "long_rejected_count": sum(1 for event in long_signal_events if not bool(event.get("approved", False))),
        "long_pnl_total": _pnl_for_reviews(long_reviews),
        "long_pnl_by_regime": _group_breakdown(long_reviews, "market_regime") if long_reviews else [],
        "long_pnl_by_entry_mode": _group_breakdown(long_reviews, "entry_mode") if long_reviews else [],
        "long_win_rate_by_entry_mode": _group_breakdown(long_reviews, "entry_mode") if long_reviews else [],
        "long_probe_pnl": _pnl_for_reviews(
            [review for review in long_reviews if review.get("actual_policy_decision") == "probe_trade"]
        ),
        "long_exploration_pnl": _pnl_for_reviews(
            [review for review in long_reviews if review.get("actual_policy_decision") == "exploration_trade"]
        ),
        "clean_uptrend_long_pnl": _pnl_for_reviews(
            [review for review in long_reviews if review.get("market_regime") == "clean_uptrend"]
        ),
        "weak_down_choppy_rebound_long_pnl": _pnl_for_reviews(
            [
                review
                for review in long_reviews
                if review.get("entry_mode") in {"rebound_probe_long", "rebound_exploration_long"}
            ]
        ),
        "range_chop_failed_breakdown_long_pnl": _pnl_for_reviews(
            [review for review in long_reviews if review.get("entry_mode") == "range_failed_breakdown_long"]
        ),
        "capitulation_bounce_long_pnl": _pnl_for_reviews(
            [review for review in long_reviews if review.get("entry_mode") == "capitulation_bounce_probe_long"]
        ),
    }


def _strict_vs_relaxed_review(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    relaxed_reviews = [review for review in reviews if bool(review.get("relaxed_only_trade", False))]
    strict_wait_relaxed = [
        review
        for review in relaxed_reviews
        if review.get("strict_policy_decision") == "wait"
        and review.get("actual_policy_decision") in {"probe_trade", "exploration_trade", "normal_trade"}
    ]
    strict_reject_long_probe = [
        review
        for review in relaxed_reviews
        if review.get("direction") == "long"
        and review.get("strict_policy_decision") == "reject"
        and review.get("actual_policy_decision") in {"probe_trade", "exploration_trade"}
    ]
    signal_events = [event for event in events if event.get("action") == "signal"]
    return {
        "strict_wait_but_relaxed_traded_count": len(strict_wait_relaxed),
        "strict_wait_but_relaxed_traded_pnl": _pnl_for_reviews(strict_wait_relaxed),
        "strict_reject_but_relaxed_long_probe_count": len(strict_reject_long_probe),
        "strict_reject_but_relaxed_long_probe_pnl": _pnl_for_reviews(strict_reject_long_probe),
        "relaxed_only_trade_count": len(relaxed_reviews),
        "relaxed_only_trade_pnl": _pnl_for_reviews(relaxed_reviews),
        "strict_wait_relaxed_signal_count": sum(
            1
            for event in signal_events
            if _event_policy_fields(event).get("strict_policy_decision") == "wait"
            and _event_policy_fields(event).get("actual_policy_decision")
            in {"probe_trade", "exploration_trade", "normal_trade"}
        ),
    }


def _blocker_review(events: list[dict[str, object]]) -> dict[str, object]:
    signal_events = [event for event in events if event.get("action") == "signal"]
    cap_actions = {"learning_cap_warning", "learning_cap_shadow_only", "learning_cap_reduce_size"}
    return {
        "ml_blocked_count": sum(
            1
            for event in signal_events
            if _event_ml_blocks(event) and not bool(event.get("approved", False))
        ),
        "caps_blocked_count": sum(1 for event in events if str(event.get("action", "")) in cap_actions),
        "regime_wait_only_count": sum(
            1
            for event in signal_events
            if _event_entry_mode(event) == "wait"
            or _event_policy_fields(event).get("actual_policy_decision") == "wait_pullback"
        ),
        "hard_risk_blocked_count": sum(
            1
            for event in signal_events
            if not bool(event.get("approved", False))
            and any(token in str(event.get("reason", "")) for token in ["risk", "exposure", "cash", "positions"])
        ),
        "hard_execution_blocked_count": sum(
            1
            for event in signal_events
            if bool(_event_hard_issues(event)) or "execution" in str(event.get("reason", ""))
        ),
        "soft_issue_reduced_size_count": sum(
            1
            for event in signal_events
            if bool(_event_soft_issues(event))
            and _event_policy_fields(event).get("actual_policy_decision") in {"probe_trade", "exploration_trade"}
        ),
    }


def _learning_cap_metrics(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    cap_events = _learning_cap_events(events)
    event_caps = [_event_learning_caps(event) for event in cap_events]
    review_caps = [
        review.get("learning_caps", {})
        for review in reviews
        if isinstance(review.get("learning_caps", {}), dict)
        and bool(review.get("learning_caps", {}).get("available"))
    ]
    all_caps = [caps for caps in [*event_caps, *review_caps] if isinstance(caps, dict) and caps]
    behavior_counts = Counter(str(caps.get("cap_behavior_applied", "none") or "none") for caps in all_caps)
    tag_counts = Counter(
        tag
        for caps in all_caps
        for tag in caps.get("oversampling_tags", [])
        if isinstance(caps.get("oversampling_tags", []), list)
    )
    cap_event_counts = _count_cap_events(cap_events)
    same_symbol_reviews = _reviews_with_learning_tag(reviews, "same_symbol_learning_cap_hit")
    same_entry_mode_reviews = _reviews_with_learning_tag(reviews, "same_entry_mode_learning_cap_hit")
    same_regime_reviews = _reviews_with_learning_tag(reviews, "same_regime_learning_cap_hit")
    return {
        "available": bool(all_caps or cap_events),
        "trades_with_learning_caps": len(review_caps),
        "events_with_learning_caps": len(cap_events),
        "probe_trades_per_day": max(
            [int(caps.get("probe_count_today", 0) or 0) for caps in all_caps],
            default=0,
        ),
        "exploration_trades_per_day": max(
            [int(caps.get("exploration_count_today", 0) or 0) for caps in all_caps],
            default=0,
        ),
        "probe_daily_cap_hits": sum(
            1
            for caps in all_caps
            if bool(caps.get("daily_cap_hit")) and str(caps.get("mode", "")) == "probe"
        ),
        "exploration_daily_cap_hits": sum(
            1
            for caps in all_caps
            if bool(caps.get("daily_cap_hit")) and str(caps.get("mode", "")) == "exploration"
        ),
        "same_symbol_cap_hits": sum(1 for caps in all_caps if bool(caps.get("same_symbol_cap_hit"))),
        "same_entry_mode_cap_hits": sum(1 for caps in all_caps if bool(caps.get("same_entry_mode_cap_hit"))),
        "same_regime_cap_hits": sum(1 for caps in all_caps if bool(caps.get("same_regime_cap_hit"))),
        "oversampling_tags": dict(sorted(tag_counts.items())),
        "cap_behavior_counts": dict(sorted(behavior_counts.items())),
        "warning_events": cap_event_counts.get("learning_cap_warning", 0),
        "shadow_only_events": cap_event_counts.get("learning_cap_shadow_only", 0),
        "reduce_size_events": cap_event_counts.get("learning_cap_reduce_size", 0),
        "unresolved_shadow_only_events": _unresolved_shadow_count(cap_events),
        "same_symbol_cap_pnl": _pnl_for_reviews(same_symbol_reviews),
        "same_entry_mode_cap_pnl": _pnl_for_reviews(same_entry_mode_reviews),
        "same_regime_cap_pnl": _pnl_for_reviews(same_regime_reviews),
    }


def _selloff_learning_cap_metrics(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    selloff_events = [
        event
        for event in events
        if str(event.get("action", "")).startswith("selloff_learning_cap")
        or bool(_event_learning_caps(event, key="selloff_learning_caps"))
    ]
    event_caps = [_event_learning_caps(event, key="selloff_learning_caps") for event in selloff_events]
    review_caps = [
        review.get("selloff_learning_caps", {})
        for review in reviews
        if isinstance(review.get("selloff_learning_caps", {}), dict)
    ]
    all_caps = [caps for caps in [*event_caps, *review_caps] if isinstance(caps, dict) and caps]
    cap_event_counts = _count_cap_events(selloff_events)
    return {
        "available": bool(all_caps or selloff_events),
        "events_with_selloff_learning_caps": len(selloff_events),
        "selloff_positions_used": max(
            [int(caps.get("selloff_positions_count", caps.get("selloff_positions_used", 0)) or 0) for caps in all_caps],
            default=0,
        ),
        "selloff_new_shorts_per_cycle_max_observed": max(
            [
                int(caps.get("new_selloff_shorts_this_cycle", caps.get("new_shorts_this_cycle", 0)) or 0)
                for caps in all_caps
            ],
            default=0,
        ),
        "selloff_same_symbol_cap_hits": sum(
            1
            for caps in all_caps
            if bool(caps.get("same_symbol_selloff_cap_hit", caps.get("same_symbol_cap_hit", False)))
        ),
        "selloff_same_entry_mode_cap_hits": sum(
            1
            for caps in all_caps
            if bool(caps.get("same_entry_mode_selloff_cap_hit", caps.get("same_entry_mode_cap_hit", False)))
        ),
        "selloff_same_regime_cap_hits": sum(
            1
            for caps in all_caps
            if bool(caps.get("same_regime_selloff_cap_hit", caps.get("same_regime_cap_hit", False)))
        ),
        "warning_events": cap_event_counts.get("selloff_learning_cap_warning", 0),
        "shadow_only_events": cap_event_counts.get("selloff_learning_cap_shadow_only", 0),
        "reduce_size_events": cap_event_counts.get("selloff_learning_cap_reduce_size", 0),
        "unresolved_shadow_only_events": _unresolved_shadow_count(selloff_events),
    }


def _learning_cap_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    actions = {
        "learning_cap_warning",
        "learning_cap_shadow_only",
        "learning_cap_reduce_size",
    }
    return [
        event
        for event in events
        if str(event.get("action", "")) in actions or bool(_event_learning_caps(event))
    ]


def _event_learning_caps(
    event: dict[str, object],
    *,
    key: str = "learning_caps",
) -> dict[str, object]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    candidates: list[object] = [
        event.get(key),
        metadata.get(key),
    ]
    policy = metadata.get("regime_policy", metadata.get("regime_policy_audit", {}))
    if isinstance(policy, dict):
        candidates.append(policy.get(key))
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return dict(candidate)
    return {}


def _count_cap_events(events: list[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(event.get("action", "")) for event in events).items()))


def _reviews_with_learning_tag(
    reviews: list[dict[str, object]],
    tag: str,
) -> list[dict[str, object]]:
    matching: list[dict[str, object]] = []
    for review in reviews:
        caps = review.get("learning_caps", {})
        if not isinstance(caps, dict):
            continue
        tags = caps.get("oversampling_tags", [])
        if isinstance(tags, list) and tag in tags:
            matching.append(review)
    return matching


def _pnl_for_reviews(reviews: list[dict[str, object]]) -> dict[str, object]:
    if not reviews:
        return {
            "trades": 0,
            "net_pnl_rub": 0.0,
            "expectancy_rub": 0.0,
            "win_rate_pct": 0.0,
        }
    pnl_values = [float(review.get("net_pnl_rub", 0.0) or 0.0) for review in reviews]
    wins = [value for value in pnl_values if value > 0]
    return {
        "trades": len(reviews),
        "net_pnl_rub": round(sum(pnl_values), 2),
        "expectancy_rub": round(sum(pnl_values) / len(reviews), 2),
        "win_rate_pct": round(len(wins) / len(reviews) * 100, 3),
    }


def _unresolved_shadow_count(events: list[dict[str, object]]) -> int:
    return sum(1 for event in events if str(event.get("action", "")).endswith("_shadow_only"))


def _event_policy_fields(event: dict[str, object]) -> dict[str, object]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    policy = metadata.get("regime_policy", metadata.get("regime_policy_audit", {}))
    if not isinstance(policy, dict):
        policy = {}
    actual = event.get("actual_policy_decision", policy.get("actual_policy_decision", policy.get("decision_type")))
    strict = event.get("strict_policy_decision", policy.get("strict_policy_decision"))
    if actual is None and strict is None:
        return {}
    return {
        "actual_policy_decision": actual or "unknown",
        "strict_policy_decision": strict or "unknown",
        "relaxed_only_trade": event.get("relaxed_only_trade", policy.get("relaxed_only_trade", False)),
    }


def _event_regime(event: dict[str, object]) -> str:
    if event.get("regime"):
        return str(event.get("regime"))
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    market_regime = metadata.get("market_regime", {})
    if isinstance(market_regime, dict):
        return str(market_regime.get("regime", ""))
    return ""


def _event_entry_mode(event: dict[str, object]) -> str:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    policy = metadata.get("regime_policy", metadata.get("regime_policy_audit", {}))
    if not isinstance(policy, dict):
        policy = {}
    return str(event.get("entry_mode", metadata.get("entry_mode", policy.get("entry_mode", ""))) or "")


def _event_short_only(event: dict[str, object]) -> dict[str, object]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    short_only = metadata.get("short_only", {})
    return dict(short_only) if isinstance(short_only, dict) else {}


def _review_short_only(review: dict[str, object]) -> dict[str, object]:
    metadata = review.get("entry_metadata", {})
    if not isinstance(metadata, dict):
        return {}
    short_only = metadata.get("short_only", {})
    return dict(short_only) if isinstance(short_only, dict) else {}


def _review_short_ev_engine(review: dict[str, object]) -> dict[str, object]:
    metadata = review.get("entry_metadata", {})
    if not isinstance(metadata, dict):
        return {}
    engine = metadata.get("short_ev_engine", {})
    return dict(engine) if isinstance(engine, dict) else {}


def _event_setup_id(event: dict[str, object]) -> str:
    short_only = _event_short_only(event)
    return canonical_setup_id(short_only.get("setup_id", ""))


def _review_setup_id(review: dict[str, object]) -> str:
    engine = _review_short_ev_engine(review)
    if engine.get("setup_id"):
        return canonical_setup_id(engine.get("setup_id"))
    short_only = _review_short_only(review)
    return canonical_setup_id(short_only.get("setup_id", short_only.get("entry_mode", "")))


def _avg(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else 0.0


def _float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _review_mfe_pct(review: dict[str, object]) -> float | None:
    entry = _float(review.get("entry_price"))
    mfe = _float(review.get("mfe_price"))
    if entry <= 0 or mfe <= 0:
        return None
    if str(review.get("direction", "")) == "short":
        return max(0.0, (entry - mfe) / entry)
    return max(0.0, (mfe - entry) / entry)


def _review_mae_pct(review: dict[str, object]) -> float | None:
    entry = _float(review.get("entry_price"))
    mae = _float(review.get("mae_price"))
    if entry <= 0 or mae <= 0:
        return None
    if str(review.get("direction", "")) == "short":
        return max(0.0, (mae - entry) / entry)
    return max(0.0, (entry - mae) / entry)


def _setup_real_enabled_reason(
    setup_candidates: list[dict[str, object]],
    setup_reviews: list[dict[str, object]],
) -> str:
    if setup_reviews:
        return "closed_trade_stats_available"
    for event in setup_candidates:
        reason = str(_event_short_only(event).get("short_ev_reason", ""))
        if reason:
            return reason
    return "waiting_for_candidates"


def _short_ev_recommendations(ev_by_setup: list[dict[str, object]]) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    for row in ev_by_setup:
        setup_id = str(row.get("setup_id", ""))
        trades = int(row.get("trades", 0) or 0)
        ev_net = float(row.get("ev_net", 0.0) or 0.0)
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if trades >= 10 and ev_net < 0:
            recommendations.append(
                {
                    "action": "disable-negative-setup",
                    "setup_id": setup_id,
                    "reason": f"{setup_id} has {trades} trades and EV {ev_net:.2f} RUB",
                }
            )
        elif ev_net > 0 and confidence < 0.55:
            recommendations.append(
                {
                    "action": "keep-shadow-until-confidence",
                    "setup_id": setup_id,
                    "reason": f"{setup_id} EV {ev_net:.2f} RUB but confidence {confidence:.2f}",
                }
            )
        elif trades == 0:
            recommendations.append(
                {
                    "action": "collect-shadow-samples",
                    "setup_id": setup_id,
                    "reason": f"{setup_id} has no closed setup stats yet",
                }
            )
        if len(recommendations) >= 5:
            break
    return recommendations


def _event_golden_3tf(event: dict[str, object]) -> dict[str, object]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    direct = metadata.get("golden_3tf", {})
    if isinstance(direct, dict) and direct:
        return dict(direct)
    short_only = _event_short_only(event)
    nested = short_only.get("golden_3tf", {})
    return dict(nested) if isinstance(nested, dict) else {}


def _review_golden_3tf(review: dict[str, object]) -> dict[str, object]:
    metadata = review.get("entry_metadata", {})
    if not isinstance(metadata, dict):
        return {}
    direct = metadata.get("golden_3tf", {})
    if isinstance(direct, dict) and direct:
        return dict(direct)
    short_only = metadata.get("short_only", {})
    if not isinstance(short_only, dict):
        return {}
    nested = short_only.get("golden_3tf", {})
    return dict(nested) if isinstance(nested, dict) else {}


def _golden_failed_conditions(payload: dict[str, object]) -> list[object]:
    if not isinstance(payload, dict):
        return []
    conditions = payload.get("failed_conditions", [])
    return conditions if isinstance(conditions, list) else []


def _first_golden_field(events: list[dict[str, object]], key: str) -> object:
    for event in events:
        value = _event_golden_3tf(event).get(key)
        if value:
            return value
    return ""


def _first_golden_timeframe(events: list[dict[str, object]], key: str) -> object:
    for event in events:
        timeframes = _event_golden_3tf(event).get("timeframes", {})
        if isinstance(timeframes, dict) and timeframes.get(key):
            return timeframes[key]
    return ""


def _golden_pnl_by_verdict(reviews: list[dict[str, object]]) -> dict[str, object]:
    verdicts = sorted(
        {
            str(_review_golden_3tf(review).get("verdict", "unknown") or "unknown")
            for review in reviews
        }
    )
    return {
        verdict: _pnl_for_reviews(
            [review for review in reviews if str(_review_golden_3tf(review).get("verdict", "unknown")) == verdict]
        )
        for verdict in verdicts
    }


def _short_only_hard_reasons(event: dict[str, object]) -> list[object]:
    short_only = _event_short_only(event)
    reasons = short_only.get("hard_reasons", [])
    if isinstance(reasons, list):
        return reasons
    reason = event.get("reason", "")
    return [reason] if reason else []


def _short_only_enabled(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> bool:
    if any(event.get("action") == "short_only_cycle_start" for event in events):
        return True
    for review in reviews:
        metadata = review.get("entry_metadata", {})
        if not isinstance(metadata, dict):
            continue
        short_only = metadata.get("short_only", {})
        if isinstance(short_only, dict) and bool(short_only.get("enabled", False)):
            return True
    return False


def _latest_short_only_budget(events: list[dict[str, object]]) -> dict[str, object]:
    if not events:
        return {}
    ordered = sorted(
        events,
        key=lambda event: (
            event["_timestamp"].timestamp()
            if isinstance(event.get("_timestamp"), datetime)
            else 0.0
        ),
    )
    metadata = ordered[-1].get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("short_only_budget"), dict):
        return dict(metadata["short_only_budget"])
    return dict(ordered[-1])


def _pnl_by_short_only_edge_bucket(reviews: list[dict[str, object]]) -> dict[str, object]:
    buckets = sorted(
        {
            str(review.get("entry_metadata", {}).get("short_only", {}).get("edge_bucket", "unknown"))
            for review in reviews
            if isinstance(review.get("entry_metadata", {}), dict)
            and isinstance(review.get("entry_metadata", {}).get("short_only", {}), dict)
        }
    )
    return {
        bucket: _pnl_for_reviews(
            [
                review
                for review in reviews
                if isinstance(review.get("entry_metadata", {}), dict)
                and isinstance(review.get("entry_metadata", {}).get("short_only", {}), dict)
                and str(review.get("entry_metadata", {}).get("short_only", {}).get("edge_bucket", "unknown")) == bucket
            ]
        )
        for bucket in buckets
    }


def _pnl_by_short_only_metadata_key(reviews: list[dict[str, object]], key: str) -> dict[str, object]:
    buckets = sorted(
        {
            str(review.get("entry_metadata", {}).get("short_only", {}).get(key, "unknown"))
            for review in reviews
            if isinstance(review.get("entry_metadata", {}), dict)
            and isinstance(review.get("entry_metadata", {}).get("short_only", {}), dict)
        }
    )
    return {
        bucket: _pnl_for_reviews(
            [
                review
                for review in reviews
                if isinstance(review.get("entry_metadata", {}), dict)
                and isinstance(review.get("entry_metadata", {}).get("short_only", {}), dict)
                and str(review.get("entry_metadata", {}).get("short_only", {}).get(key, "unknown")) == bucket
            ]
        )
        for bucket in buckets
    }


def _pnl_by_short_only_source(reviews: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for review in reviews:
        grouped[_short_only_review_source(review)].append(review)
    return {source: _pnl_for_reviews(items) for source, items in sorted(grouped.items())}


def _pnl_by_short_only_expansion_bucket(reviews: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for review in reviews:
        short_only = review.get("entry_metadata", {}).get("short_only", {})
        if not isinstance(short_only, dict):
            grouped["unknown"].append(review)
            continue
        expansion = _safe_float(short_only.get("expansion_factor_used"), default=0.0)
        if expansion >= 2.0:
            bucket = ">=2.0x"
        elif expansion >= 1.2:
            bucket = "1.2-2.0x"
        elif expansion >= 0.8:
            bucket = "0.8-1.2x"
        elif expansion > 0.0:
            bucket = "<0.8x"
        else:
            bucket = "unknown"
        grouped[bucket].append(review)
    return {bucket: _pnl_for_reviews(items) for bucket, items in sorted(grouped.items())}


def _short_only_source_validation(
    reviews: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    groups = {
        "strategy_short_real": [
            review for review in reviews if _short_only_review_source(review) == "strategy_short"
        ],
        "synthetic_shadow": [
            event for event in events if event.get("action") == "short_only_synthetic_shadow_only"
        ],
        "ml_only_shadow": [
            event
            for event in events
            if event.get("action") == "short_only_synthetic_shadow_only"
            and _event_short_only(event).get("edge_source") == "ml"
        ],
        "mixed_bearish_shadow": [
            event for event in events if event.get("action") == "short_only_mixed_bearish_shadow_only"
        ],
        "upsize_shadow": [
            event for event in events if event.get("action") == "short_only_upsize_shadow_only"
        ],
    }
    return {
        "strategy_short_real": _source_validation_real_row(groups["strategy_short_real"]),
        "synthetic_shadow": _source_validation_shadow_row(groups["synthetic_shadow"]),
        "ml_only_shadow": _source_validation_shadow_row(groups["ml_only_shadow"]),
        "mixed_bearish_shadow": _source_validation_shadow_row(groups["mixed_bearish_shadow"]),
        "upsize_shadow": _source_validation_shadow_row(groups["upsize_shadow"]),
    }


def _short_only_disabled_sources(events: list[dict[str, object]]) -> list[dict[str, object]]:
    event_counts = Counter(str(event.get("action", "")) for event in events)
    return [
        {
            "source": "synthetic",
            "real_trading_enabled": False,
            "shadow_only": True,
            "reason": "disabled_after_0_53_short_only_winners",
            "criteria_to_reenable": ">=100 shadow trades, positive expectancy, profit factor >= 1.15",
            "shadow_events": event_counts.get("short_only_synthetic_shadow_only", 0),
        },
        {
            "source": "upsize",
            "real_trading_enabled": False,
            "shadow_only": True,
            "reason": "disabled_until_strategy_short_baseline_positive",
            "criteria_to_reenable": "strategy_short baseline positive first",
            "shadow_events": event_counts.get("short_only_upsize_shadow_only", 0),
        },
        {
            "source": "mixed_bearish",
            "real_trading_enabled": False,
            "shadow_only": True,
            "reason": "mixed_bearish_not_proven",
            "criteria_to_reenable": ">=100 shadow trades and positive expectancy",
            "shadow_events": event_counts.get("short_only_mixed_bearish_shadow_only", 0),
        },
        {
            "source": "paper_exposure_sizing",
            "real_trading_enabled": False,
            "shadow_only": False,
            "reason": "disabled_after_expanded_size_losses",
            "criteria_to_reenable": "short_only realized cohort positive with no expansion",
            "shadow_events": 0,
        },
    ]


def _source_validation_real_row(reviews: list[dict[str, object]]) -> dict[str, object]:
    pnl_values = [_safe_float(review.get("net_pnl_rub"), default=0.0) for review in reviews]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    return {
        **_pnl_for_reviews(reviews),
        "profit_factor": round(profit_factor, 3) if profit_factor is not None and math.isfinite(profit_factor) else profit_factor,
        "average_mfe_r": _avg_review_float(reviews, "mfe_r"),
        "average_mae_r": _avg_review_float(reviews, "mae_r"),
    }


def _source_validation_shadow_row(events: list[dict[str, object]]) -> dict[str, object]:
    return {
        "count": len(events),
        "shadow_pnl": {"available": False, "reason": "shadow exits are not resolved as trades"},
        "expectancy_rub": None,
        "win_rate_pct": None,
        "profit_factor": None,
        "average_mfe_r": None,
        "average_mae_r": None,
    }


def _short_only_review_source(review: dict[str, object]) -> str:
    metadata = review.get("entry_metadata", {})
    short_only = metadata.get("short_only", {}) if isinstance(metadata, dict) else {}
    if not isinstance(short_only, dict):
        return "unknown"
    if bool(short_only.get("is_upsize", False)):
        return "upsize"
    if bool(short_only.get("synthetic_candidate", False)):
        return "synthetic"
    if str(short_only.get("source_strategy_signal", "")) == "short":
        return "strategy_short"
    return str(short_only.get("real_trade_source", "strategy_short") or "strategy_short")


def _avg_review_float(reviews: list[dict[str, object]], key: str) -> float | None:
    values = [
        _safe_float(review.get(key), default=None)
        for review in reviews
        if _safe_float(review.get(key), default=None) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _safe_float(value: object, *, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _latest_market_regime(events: list[dict[str, object]]) -> str:
    for event in reversed(events):
        if event.get("action") == "market_regime":
            return str(event.get("regime", ""))
    return ""


def _latest_effective_regime(events: list[dict[str, object]]) -> str:
    for event in reversed(events):
        if event.get("action") == "short_only_mixed_bearish_override":
            return str(event.get("effective_regime", ""))
        if event.get("action") in {"short_only_budget_allocation", "short_only_no_trade_regime"}:
            return str(event.get("regime", ""))
    return _latest_market_regime(events)


def _short_only_underallocated_reasons(events: list[dict[str, object]]) -> dict[str, int]:
    reasons = Counter(
        str(event.get("reason", "unknown"))
        for event in events
        if event.get("action") == "short_only_underallocated"
    )
    return dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))


def _pending_event_is_addon(event: dict[str, object]) -> bool:
    metadata = event.get("metadata", {})
    return isinstance(metadata, dict) and bool(metadata.get("is_addon", False))


def _first_event_iso(events: list[dict[str, object]]) -> str:
    timestamps = [
        event.get("_timestamp")
        for event in events
        if isinstance(event.get("_timestamp"), datetime)
    ]
    if not timestamps:
        return ""
    return min(timestamps).isoformat()


def _first_review_entry_iso(reviews: list[dict[str, object]]) -> str:
    timestamps = [
        _parse_timestamp(review.get("entry_time"))
        for review in reviews
        if review.get("entry_time")
    ]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if not timestamps:
        return ""
    return min(timestamps).isoformat()


def _latency_minutes(start_iso: str, end_iso: str) -> float | None:
    start = _parse_timestamp(start_iso)
    end = _parse_timestamp(end_iso)
    if start is None or end is None:
        return None
    return round(max(0.0, (end - start).total_seconds() / 60.0), 3)


def _event_diagnostics(event: dict[str, object]) -> dict[str, object]:
    metadata = event.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("selloff_budget_diagnostics"), dict):
        return dict(metadata["selloff_budget_diagnostics"])
    keys = {
        "equity",
        "gross_exposure",
        "gross_exposure_pct",
        "target_gross_exposure",
        "selloff_target_gross_exposure",
        "budget_used_pct",
        "unused_budget_reason",
        "candidates_count",
        "approved_count",
        "rejected_count",
        "wait_count",
        "shadow_count",
        "selloff_budget_blockers",
    }
    return {key: event[key] for key in keys if key in event}


def _latest_selloff_diagnostics(events: list[dict[str, object]]) -> dict[str, object]:
    if not events:
        return {}
    ordered = sorted(
        events,
        key=lambda event: (
            event["_timestamp"].timestamp()
            if isinstance(event.get("_timestamp"), datetime)
            else 0.0
        ),
    )
    return _event_diagnostics(ordered[-1])


def _review_pending_entry(review: dict[str, object]) -> dict[str, object]:
    metadata = review.get("entry_metadata", {})
    if not isinstance(metadata, dict):
        return {}
    pending = metadata.get("pending_entry", {})
    return dict(pending) if isinstance(pending, dict) else {}


def _pnl_by_entry_modes(
    reviews: list[dict[str, object]],
    modes: list[str],
) -> dict[str, object]:
    return {
        mode: _pnl_for_reviews([review for review in reviews if review.get("entry_mode") == mode])
        for mode in modes
    }


def _count_signal_decision(events: list[dict[str, object]], decision: str) -> int:
    return sum(1 for event in events if _event_policy_fields(event).get("actual_policy_decision") == decision)


def _event_ml_blocks(event: dict[str, object]) -> bool:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    ml = metadata.get("ml_learning", {})
    return isinstance(ml, dict) and bool(ml.get("blocks_entry", False))


def _event_soft_issues(event: dict[str, object]) -> list[object]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    policy = metadata.get("regime_policy", metadata.get("regime_policy_audit", {}))
    if not isinstance(policy, dict):
        policy = {}
    issues = event.get("soft_issues", policy.get("soft_issues", []))
    return issues if isinstance(issues, list) else []


def _event_hard_issues(event: dict[str, object]) -> list[object]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    policy = metadata.get("regime_policy", metadata.get("regime_policy_audit", {}))
    if not isinstance(policy, dict):
        policy = {}
    issues = event.get("hard_issues", policy.get("hard_issues", []))
    return issues if isinstance(issues, list) else []


def _negative_groups(
    reviews: list[dict[str, object]],
    key: str,
    *,
    min_trades: int,
) -> list[dict[str, object]]:
    groups = [
        row
        for row in _group_breakdown(reviews, key)
        if int(row["trades"]) >= min_trades and float(row["expectancy_rub"]) < 0
    ]
    groups.sort(key=lambda row: float(row["expectancy_rub"]))
    return groups


def _mistake_breakdown(reviews: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(
        tag
        for review in reviews
        for tag in review.get("mistake_tags", [])
        if _is_error_tag(tag)
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _is_error_tag(tag: str) -> bool:
    return tag not in {"planned-take-profit", "protected-profit", "profitable-stop-exit"}


def _trade_review_id(trade: TradeRecord) -> str:
    return "|".join(
        [
            trade.symbol,
            trade.direction.value,
            trade.entry_time.isoformat(),
            trade.exit_time.isoformat(),
            f"{trade.entry_price:.8f}",
            f"{trade.exit_price:.8f}",
        ]
    )


def _outcome(net_pnl: float) -> str:
    if net_pnl > 0:
        return "win"
    if net_pnl < 0:
        return "loss"
    return "flat"


def _strength_bucket(strength: float) -> str:
    bucket = math.floor(max(0.0, min(1.0, strength)) * 10) / 10
    upper = min(1.0, bucket + 0.1)
    return f"{bucket:.1f}-{upper:.1f}"


def _infer_units(trade: TradeRecord) -> float:
    if trade.direction.value == "long":
        price_delta = trade.exit_price - trade.entry_price
    else:
        price_delta = trade.entry_price - trade.exit_price
    if abs(price_delta) > 1e-12 and abs(trade.gross_pnl) > 1e-12:
        return abs(trade.gross_pnl / price_delta)
    return max(1, trade.quantity_lots)


def _planned_distance_rub(
    entry_price: float,
    target_price: float | None,
    units: float,
) -> float | None:
    if target_price is None or target_price <= 0:
        return None
    distance = abs(entry_price - target_price)
    if distance <= 0:
        return None
    return distance * max(1.0, units)


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _rounded_or_none(value: float | None, *, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        return round(value, 4)
    return round(math.ceil(value / step) * step, 4)


def _confidence(sample_size: int) -> str:
    if sample_size >= 8:
        return "high"
    if sample_size >= 4:
        return "medium"
    return "low"


def _render_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {})
    recommendations = payload.get("recommendations", [])
    mistakes = payload.get("mistake_breakdown", {})
    lines = [
        "# Trade Review",
        "",
        f"- Commit: {payload.get('commit_hash', 'unknown')}",
        f"- Reviewed trades: {payload.get('reviewed_trades', 0)}",
        f"- Net PnL: {summary.get('net_pnl_rub', 0.0)} RUB",
        f"- Win rate: {summary.get('win_rate_pct', 0.0)}%",
        f"- Expectancy: {summary.get('expectancy_rub', 0.0)} RUB/trade",
        f"- Mistake trades: {summary.get('mistake_trades', 0)}",
        "",
        "## Mistakes",
    ]
    if mistakes:
        for tag, count in mistakes.items():
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- No repeated mistake pattern found")

    breakdowns = payload.get("breakdowns", {})
    dimension_keys = ["market_regime", "entry_mode", "symbol_health", "rebound_outcome"]
    lines.append("")
    lines.append("## Dimensions")
    if isinstance(breakdowns, dict):
        for key in dimension_keys:
            rows = breakdowns.get(key, [])
            if not rows:
                continue
            preview = []
            for row in rows[:3]:
                preview.append(
                    f"{row.get('group')}: {row.get('expectancy_rub')} RUB/trade "
                    f"({row.get('trades')} trades)"
                )
            lines.append(f"- {key}: " + "; ".join(preview))

    policy = payload.get("policy_outcomes", {})
    if isinstance(policy, dict):
        lines.append("")
        lines.append("## Policy Outcomes")
        lines.append(f"- Resolved shadow trades: {policy.get('resolved_shadow_trades', 0)}")
        lines.append(f"- Relaxed-only trades: {policy.get('relaxed_only_trades', 0)}")
        for key, label in [
            ("by_trade_source", "trade_source"),
            ("by_actual_policy_decision", "actual_policy_decision"),
        ]:
            rows = policy.get(key, [])
            if not rows:
                continue
            preview = [
                f"{row.get('group')}: {row.get('expectancy_rub')} RUB/trade ({row.get('trades')} trades)"
                for row in rows[:3]
            ]
            lines.append(f"- {label}: " + "; ".join(preview))

    short_only = payload.get("short_only_review", {})
    if isinstance(short_only, dict) and short_only.get("short_only_enabled"):
        lines.append("")
        lines.append("## Short Only Review")
        lines.append(f"- Long signals ignored: {short_only.get('long_signals_ignored', 0)}")
        lines.append(f"- Longs flattened: {short_only.get('longs_flattened', 0)}")
        lines.append(f"- Short candidates: {short_only.get('short_candidates_total', 0)}")
        lines.append(f"- Positive-EV candidates: {short_only.get('positive_ev_short_candidates', 0)}")
        lines.append(f"- Shorts opened: {short_only.get('shorts_opened', 0)}")
        lines.append(f"- Budget used: {short_only.get('budget_used', 0.0)} / {short_only.get('budget_target', 0.0)}")
        lines.append(f"- Underallocated events: {short_only.get('underallocated_count', 0)}")

    short_ev = payload.get("short_ev_review", {})
    if isinstance(short_ev, dict) and short_ev.get("enabled"):
        lines.append("")
        lines.append("## Short EV Review")
        lines.append(f"- Mode: {short_ev.get('mode', '')}")
        lines.append(f"- Execution: {short_ev.get('execution_mode', '')}, live={short_ev.get('allow_live_trading')}")
        rows = short_ev.get("ev_by_setup", [])
        if isinstance(rows, list):
            for row in rows[:5]:
                if isinstance(row, dict):
                    lines.append(
                        "- "
                        f"{row.get('setup_id')}: EV {row.get('ev_net')} RUB, "
                        f"confidence {row.get('confidence')}, trades {row.get('trades')}, "
                        f"costs avg {row.get('costs_avg')} RUB"
                    )
        exits = short_ev.get("exit_performance", {})
        if isinstance(exits, dict):
            lines.append(
                "- Exits: "
                f"BE {exits.get('breakeven_armed_count', 0)}, "
                f"trailing {exits.get('trailing_activated_count', 0)}, "
                f"book tightening {exits.get('order_book_tightening_count', 0)}"
            )

    golden = payload.get("golden_3tf_review", {})
    if isinstance(golden, dict) and golden.get("enabled"):
        lines.append("")
        lines.append("## Golden 3TF Baseline Review")
        lines.append(f"- Source: {golden.get('source_run', '')} / {str(golden.get('source_commit', ''))[:12]}")
        lines.append(f"- Timeframes: {golden.get('timeframes', {})}")
        lines.append(f"- Candidates: {golden.get('candidate_events', 0)}")
        lines.append(f"- 15m full: {golden.get('golden_15m_candidates', 0)}")
        lines.append(f"- Early 5m starters: {golden.get('early_5m_starter_candidates', 0)}")
        lines.append(f"- Passed / shadow-only: {golden.get('passed_candidates', 0)} / {golden.get('shadow_only_candidates', 0)}")
        lines.append(f"- Approved / rejected signals: {golden.get('approved_signals', 0)} / {golden.get('rejected_signals', 0)}")
        failed = golden.get("top_failed_conditions", {})
        if isinstance(failed, dict) and failed:
            lines.append("- Top failed conditions: " + "; ".join(f"{key}: {value}" for key, value in failed.items()))

    selloff = payload.get("selloff_capture_review", {})
    if isinstance(selloff, dict):
        lines.append("")
        lines.append("## Selloff Capture Review")
        lines.append(f"- Selloff windows: {selloff.get('selloff_windows', 0)}")
        lines.append(f"- Candidates: {selloff.get('candidates_count', 0)}")
        lines.append(f"- Trades opened: {selloff.get('trades_opened', 0)}")
        lines.append(f"- Budget used: {selloff.get('budget_used_pct', 0.0)}")
        lines.append(f"- Wait-only count: {selloff.get('wait_only_count_during_selloff', 0)}")

    underallocation = payload.get("underallocation_review", {})
    if isinstance(underallocation, dict):
        lines.append("")
        lines.append("## Underallocation Review")
        lines.append(f"- Underallocated events: {underallocation.get('selloff_underallocated_count', 0)}")
        lines.append(f"- Avg budget used: {underallocation.get('avg_budget_used_pct', 0.0)}")
        lines.append(f"- Max budget used: {underallocation.get('max_budget_used_pct', 0.0)}")
        reasons = underallocation.get("reasons_for_unused_budget", {})
        if isinstance(reasons, dict) and reasons:
            lines.append("- Reasons: " + "; ".join(f"{key}: {value}" for key, value in reasons.items()))

    selloff_long = payload.get("long_during_selloff_review", {})
    if isinstance(selloff_long, dict):
        lines.append("")
        lines.append("## Long During Selloff Review")
        lines.append(f"- Long signals: {selloff_long.get('long_signals_during_selloff', 0)}")
        lines.append(f"- Long shadow count: {selloff_long.get('long_shadow_count', 0)}")
        lines.append(f"- Capitulation bounce probes: {selloff_long.get('capitulation_bounce_probes', 0)}")

    lines.append("")
    lines.append("## Recommendations")
    for item in recommendations:
        lines.append(f"- {item.get('action')}: {item.get('reason')}")
    lines.append("")
    return "\n".join(lines)
