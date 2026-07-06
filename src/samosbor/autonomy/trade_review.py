from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import RiskSection, StrategySection
from ..domain import PortfolioState, TradeRecord
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
    return {
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
        "mistake_breakdown": _mistake_breakdown(reviews),
        "recommendations": recommendations["items"],
        "config_patch_candidates": recommendations["config_patch_candidates"],
        "reviews": reviews,
    }


def save_trade_review(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_trade_review(output_dir: Path, payload: dict[str, object]) -> None:
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
    market_regime = _market_regime_summary(trade.entry_metadata)
    regime_policy = _regime_policy_summary(trade.entry_metadata)
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
        "entry_ml_learning": ml_learning,
        "post_close_analysis": post_close_analysis,
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
        "ml_available",
        "ml_probability_profit",
        "ml_expected_pnl_position_rub",
        "ml_entry_bias",
        "ml_verdict",
        "tags",
        "summary",
    ]
    return {key: raw[key] for key in keys if key in raw}


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

    lines.append("")
    lines.append("## Recommendations")
    for item in recommendations:
        lines.append(f"- {item.get('action')}: {item.get('reason')}")
    lines.append("")
    return "\n".join(lines)
