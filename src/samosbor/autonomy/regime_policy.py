from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..domain import SignalDirection


class PolicyDecisionType(str, Enum):
    NORMAL_TRADE = "normal_trade"
    PROBE_TRADE = "probe_trade"
    EXPLORATION_TRADE = "exploration_trade"
    WAIT_PULLBACK = "wait_pullback"
    SHADOW_ONLY = "shadow_only"
    HARD_REJECT = "hard_reject"


@dataclass(frozen=True)
class TradePolicy:
    allow_trade: bool
    entry_mode: str
    risk_multiplier: float
    reasons: list[str] = field(default_factory=list)
    symbol_health: str = "normal"
    risk_components: dict[str, float] = field(default_factory=dict)
    decision_type: str = PolicyDecisionType.NORMAL_TRADE.value
    strict_policy_decision: str = "unknown"
    strict_policy_reasons: list[str] = field(default_factory=list)
    actual_policy_reasons: list[str] = field(default_factory=list)
    soft_issues: list[str] = field(default_factory=list)
    hard_issues: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    actual_policy_profile: str = "strict"
    would_strict_policy_trade: bool = True
    would_strict_policy_risk_multiplier: float = 1.0
    relaxed_only_trade: bool = False
    microstructure_bucket: str = "unknown"
    confirmation_5m_status: str = "unknown"
    ml_negative_edge: bool = False
    effective_risk_multiplier: float = 1.0
    strict_policy_metadata: dict[str, object] = field(default_factory=dict)
    side_policy: dict[str, object] = field(default_factory=dict)
    symbol_health_metadata: dict[str, object] = field(default_factory=dict)
    confirmation_5m: dict[str, object] = field(default_factory=dict)
    policy_flags: dict[str, object] = field(default_factory=dict)
    long_context: dict[str, object] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, object]:
        metadata = {
            "allow_trade": self.allow_trade,
            "entry_mode": self.entry_mode,
            "risk_multiplier": self.risk_multiplier,
            "effective_risk_multiplier": self.effective_risk_multiplier,
            "reasons": list(self.reasons),
            "symbol_health": self.symbol_health,
            "risk_components": dict(self.risk_components),
            "decision_type": self.decision_type,
            "actual_policy_profile": self.actual_policy_profile,
            "actual_policy_decision": self.decision_type,
            "actual_policy_reasons": list(self.actual_policy_reasons or self.reasons),
            "strict_policy_decision": self.strict_policy_decision,
            "strict_policy_reasons": list(self.strict_policy_reasons),
            "would_strict_policy_trade": self.would_strict_policy_trade,
            "would_strict_policy_risk_multiplier": self.would_strict_policy_risk_multiplier,
            "relaxed_only_trade": self.relaxed_only_trade,
            "soft_issues": list(self.soft_issues),
            "hard_issues": list(self.hard_issues),
            "tags": list(self.tags),
            "microstructure_bucket": self.microstructure_bucket,
            "confirmation_5m_status": self.confirmation_5m_status,
            "ml_negative_edge": self.ml_negative_edge,
            "strict_policy": dict(self.strict_policy_metadata),
            "side_policy": dict(self.side_policy),
            "symbol_health_policy": dict(self.symbol_health_metadata),
            "confirmation_5m": dict(self.confirmation_5m),
            "policy_flags": dict(self.policy_flags),
            "long_context": dict(self.long_context),
        }
        metadata.update(dict(self.policy_flags))
        if self.long_context:
            metadata["long_context"] = dict(self.long_context)
        return metadata


@dataclass(frozen=True)
class BucketResult:
    bucket: str
    multiplier: float
    issue: str
    hard: bool = False


def bucket_spread(spread_bps: float | None) -> BucketResult:
    if spread_bps is None:
        return BucketResult("unknown", 1.0, "")
    if spread_bps <= 12.0:
        return BucketResult("normal", 1.0, "")
    if spread_bps <= 18.0:
        return BucketResult("probe", 0.35, "slightly-wide-spread")
    if spread_bps <= 25.0:
        return BucketResult("exploration", 0.10, "wide-spread-bounded")
    return BucketResult("reject", 0.0, "spread-above-hard-limit", hard=True)


def bucket_liquidity_cover(cover: float | None) -> BucketResult:
    if cover is None:
        return BucketResult("unknown", 1.0, "")
    if cover >= 2.0:
        return BucketResult("normal", 1.0, "")
    if cover >= 1.2:
        return BucketResult("probe", 0.35, "thin-book-probe")
    if cover >= 0.8:
        return BucketResult("exploration", 0.10, "thin-book-exploration")
    return BucketResult("reject", 0.0, "liquidity-cover-below-hard-limit", hard=True)


def bucket_imbalance(side_imbalance: float | None) -> BucketResult:
    if side_imbalance is None:
        return BucketResult("unknown", 1.0, "")
    if side_imbalance >= -0.35:
        return BucketResult("normal", 1.0, "")
    if side_imbalance >= -0.60:
        return BucketResult("probe", 0.35, "mild-adverse-book")
    if side_imbalance >= -0.80:
        return BucketResult("exploration", 0.10, "medium-adverse-book")
    return BucketResult("reject", 0.0, "adverse-book-below-hard-limit", hard=True)


def resolve(
    *,
    regime: str,
    symbol: str,
    side: SignalDirection | str,
    ml_feedback: dict[str, object] | None = None,
    book: dict[str, object] | None = None,
    confirmation: dict[str, object] | None = None,
    entry_mode: str | None = None,
    symbol_health: str = "normal",
    long_side_enabled: bool = False,
    learning_mode_enabled: bool = False,
    learning_profile: str = "strict",
    signal_strength: float | None = None,
    trend_strength: float | None = None,
    adx: float | None = None,
    require_order_book: bool = False,
    observe_only_configured: bool = False,
    config: object | None = None,
) -> TradePolicy:
    strict_policy = _resolve_strict(
        regime=regime,
        symbol=symbol,
        side=side,
        ml_feedback=ml_feedback,
        book=book,
        entry_mode=entry_mode,
        symbol_health=symbol_health,
        long_side_enabled=long_side_enabled,
    )
    if not learning_mode_enabled:
        return strict_policy
    return _resolve_relaxed(
        regime=regime,
        symbol=symbol,
        side=side,
        ml_feedback=ml_feedback,
        book=book,
        confirmation=confirmation,
        entry_mode=entry_mode,
        symbol_health=symbol_health,
        learning_profile=learning_profile,
        signal_strength=signal_strength,
        trend_strength=trend_strength,
        adx=adx,
        require_order_book=require_order_book,
        observe_only_configured=observe_only_configured,
        config=config,
        strict_policy=strict_policy,
    )


def _resolve_strict(
    *,
    regime: str,
    symbol: str,
    side: SignalDirection | str,
    ml_feedback: dict[str, object] | None,
    book: dict[str, object] | None,
    entry_mode: str | None,
    symbol_health: str,
    long_side_enabled: bool,
) -> TradePolicy:
    normalized_side = _side_value(side)
    requested_mode = (entry_mode or "").strip()
    reasons: list[str] = []
    components: dict[str, float] = {"base": 1.0}

    if normalized_side == SignalDirection.LONG.value and not long_side_enabled:
        return _strict_policy(
            False,
            "reject",
            0.0,
            ["long-side-disabled"],
            symbol_health=symbol_health,
            components=components,
        )
    if symbol_health == "observe_only":
        return _strict_policy(
            False,
            "reject",
            0.0,
            [f"symbol-observe-only:{symbol}"],
            symbol_health=symbol_health,
            components=components,
        )

    risk_multiplier = 1.0
    if symbol_health == "probation":
        risk_multiplier *= 0.5
        components["symbol_health"] = 0.5
        reasons.append(f"symbol-probation:{symbol}")

    if _adverse_book(book):
        return _strict_policy(
            False,
            "reject",
            0.0,
            reasons + ["adverse-book"],
            symbol_health=symbol_health,
            components=components,
        )

    if _ml_blocks(ml_feedback):
        risk_multiplier *= 0.25
        components["ml_negative_edge"] = 0.25
        reasons.append("ml-negative-edge-fractional")

    if regime == "weak_down_choppy":
        if normalized_side == SignalDirection.SHORT.value and requested_mode == "pullback_short":
            components["regime"] = 0.5
            return _strict_policy(
                True,
                "pullback_short",
                max(0.0, min(0.5, risk_multiplier * 0.5)),
                reasons + ["weak-down-choppy-pullback-only"],
                symbol_health=symbol_health,
                components=components,
            )
        if normalized_side == SignalDirection.SHORT.value:
            return _strict_policy(
                False,
                "wait",
                0.0,
                reasons + ["weak-down-choppy-blocks-trend-short"],
                symbol_health=symbol_health,
                components=components,
            )
        return _strict_policy(
            False,
            "reject",
            0.0,
            reasons + ["weak-down-choppy-no-long"],
            symbol_health=symbol_health,
            components=components,
        )

    if normalized_side == SignalDirection.SHORT.value:
        mode = requested_mode or "trend_short"
    elif normalized_side == SignalDirection.LONG.value:
        mode = requested_mode or "trend_long"
    else:
        mode = requested_mode or "wait"
    return _strict_policy(
        True,
        mode,
        max(0.0, min(1.0, risk_multiplier)),
        reasons or ["default-allow"],
        symbol_health=symbol_health,
        components=components,
    )


def _strict_policy(
    allow_trade: bool,
    entry_mode: str,
    risk_multiplier: float,
    reasons: list[str],
    *,
    symbol_health: str,
    components: dict[str, float],
) -> TradePolicy:
    if allow_trade:
        decision_type = PolicyDecisionType.NORMAL_TRADE.value
    elif entry_mode == "wait":
        decision_type = PolicyDecisionType.WAIT_PULLBACK.value
    else:
        decision_type = PolicyDecisionType.HARD_REJECT.value
    strict_decision = "allow" if allow_trade else ("wait" if entry_mode == "wait" else "reject")
    return TradePolicy(
        allow_trade=allow_trade,
        entry_mode=entry_mode,
        risk_multiplier=risk_multiplier,
        reasons=reasons,
        symbol_health=symbol_health,
        risk_components=components,
        decision_type=decision_type,
        strict_policy_decision=strict_decision,
        strict_policy_reasons=list(reasons),
        actual_policy_reasons=list(reasons),
        actual_policy_profile="strict",
        would_strict_policy_trade=allow_trade,
        would_strict_policy_risk_multiplier=risk_multiplier,
        effective_risk_multiplier=risk_multiplier,
    )


def _resolve_relaxed(
    *,
    regime: str,
    symbol: str,
    side: SignalDirection | str,
    ml_feedback: dict[str, object] | None,
    book: dict[str, object] | None,
    confirmation: dict[str, object] | None,
    entry_mode: str | None,
    symbol_health: str,
    learning_profile: str,
    signal_strength: float | None,
    trend_strength: float | None,
    adx: float | None,
    require_order_book: bool,
    observe_only_configured: bool,
    config: object | None,
    strict_policy: TradePolicy,
) -> TradePolicy:
    normalized_side = _side_value(side)
    requested_mode = (entry_mode or "").strip() or _default_entry_mode(normalized_side)
    components: dict[str, float] = {"base": _mode_risk_multiplier(config, "normal", 1.0)}
    soft_issues: list[str] = []
    hard_issues: list[str] = []
    tags: list[str] = []
    multipliers: list[tuple[str, float]] = [("base", components["base"])]
    entry_mode_out = requested_mode
    force_wait_pullback = False
    force_shadow_only = False
    decision_type_override: str | None = None
    final_multiplier_override: float | None = None
    policy_flags: dict[str, object] = {}
    long_context: dict[str, object] = {}
    ml_negative_edge = _ml_blocks(ml_feedback)

    if book is not None and not book.get("available", True) and require_order_book:
        hard_issues.append("missing-order-book")
        tags.append("technical-execution-issue")

    micro = _microstructure_assessment(book, regime=regime, config=config)
    if micro["hard_issue"]:
        hard_issues.append(str(micro["hard_issue"]))
    for issue in micro["soft_issues"]:
        soft_issues.append(issue)
    if float(micro["multiplier"]) < 1.0:
        multipliers.append(("microstructure", float(micro["multiplier"])))
    tags.extend(micro["tags"])

    confirmation_result = _confirmation_assessment(
        confirmation,
        side=normalized_side,
        regime=regime,
        config=config,
    )
    if confirmation_result["hard_issue"]:
        hard_issues.append(str(confirmation_result["hard_issue"]))
    for issue in confirmation_result["soft_issues"]:
        soft_issues.append(issue)
    if float(confirmation_result["multiplier"]) < 1.0:
        multipliers.append(("confirmation_5m", float(confirmation_result["multiplier"])))
    if confirmation_result["wait_pullback"]:
        force_wait_pullback = True
    tags.extend(confirmation_result["tags"])

    signal_issue = _signal_quality_issue(
        signal_strength=signal_strength,
        trend_strength=trend_strength,
        adx=adx,
        config=config,
    )
    if signal_issue:
        soft_issues.append(signal_issue["issue"])
        multipliers.append(("signal_quality", float(signal_issue["multiplier"])))
        tags.append(str(signal_issue["tag"]))

    symbol_status = symbol_health
    symbol_multiplier = 1.0
    if symbol_health == "observe_only" and observe_only_configured:
        hard_issues.append(f"symbol-observe-only:{symbol}")
    elif symbol_health in {"observe_only", "weak"}:
        symbol_status = "weak"
        symbol_multiplier = _symbol_health_multiplier(config, "weak", 0.10)
        soft_issues.append(f"symbol-weak:{symbol}")
        tags.append("weak-symbol")
    elif symbol_health == "probation":
        symbol_multiplier = _symbol_health_multiplier(config, "probation", 0.25)
        soft_issues.append(f"symbol-probation:{symbol}")
        tags.append("probation-symbol")
    if symbol_multiplier < 1.0:
        multipliers.append(("symbol_health", symbol_multiplier))

    side_policy = _side_policy_metadata(
        normalized_side,
        config=config,
        current_soft_issues=len(soft_issues),
    )
    if side_policy["hard_issue"]:
        hard_issues.append(str(side_policy["hard_issue"]))
    if side_policy["soft_issue"]:
        soft_issues.append(str(side_policy["soft_issue"]))
    side_multiplier = float(side_policy["multiplier"])
    if side_multiplier < 1.0:
        multipliers.append(("side", side_multiplier))
    tags.extend(side_policy["tags"])

    if normalized_side == SignalDirection.LONG.value:
        long_decision = _long_regime_decision(
            regime=regime,
            signal_strength=signal_strength,
            confirmation=confirmation,
            confirmation_result=confirmation_result,
            side_policy=side_policy,
            config=config,
            strict_policy=strict_policy,
        )
        if long_decision["hard_issue"]:
            hard_issues.append(str(long_decision["hard_issue"]))
        if long_decision["soft_issue"]:
            soft_issues.append(str(long_decision["soft_issue"]))
        tags.extend([str(tag) for tag in long_decision["tags"]])
        entry_mode_out = str(long_decision["entry_mode"])
        long_context = dict(long_decision["long_context"])
        policy_flags.update(dict(long_decision["policy_flags"]))
        if bool(long_decision["force_wait_pullback"]):
            force_wait_pullback = True
        if bool(long_decision["force_shadow_only"]):
            force_shadow_only = True
        if long_decision["decision_type"]:
            decision_type_override = str(long_decision["decision_type"])
        if long_decision["final_multiplier"] is not None:
            final_multiplier_override = float(long_decision["final_multiplier"])
            components["long_regime_policy"] = final_multiplier_override
    elif regime == "market_selloff_impulse":
        if normalized_side == SignalDirection.SHORT.value:
            short_decision = _market_selloff_short_decision(
                signal_strength=signal_strength,
                requested_mode=requested_mode,
                confirmation_result=confirmation_result,
                micro_bucket=str(micro["bucket"]),
                current_soft_issues=len(soft_issues),
                symbol_health=symbol_health,
                ml_negative_edge=ml_negative_edge,
                config=config,
                strict_policy=strict_policy,
            )
            soft_issues.extend([str(issue) for issue in short_decision["soft_issues"]])
            tags.extend([str(tag) for tag in short_decision["tags"]])
            entry_mode_out = str(short_decision["entry_mode"])
            policy_flags.update(dict(short_decision["policy_flags"]))
            if bool(short_decision["force_wait_pullback"]):
                force_wait_pullback = True
            if bool(short_decision["force_shadow_only"]):
                force_shadow_only = True
            if short_decision["decision_type"]:
                decision_type_override = str(short_decision["decision_type"])
            if short_decision["final_multiplier"] is not None:
                final_multiplier_override = float(short_decision["final_multiplier"])
                components["market_selloff_impulse"] = final_multiplier_override
    elif regime == "weak_down_choppy":
        if normalized_side == SignalDirection.SHORT.value and requested_mode == "pullback_short":
            multipliers.append(("regime", 0.5))
            soft_issues.append("weak-down-choppy-pullback-short")
            tags.append("wait-pullback-trigger")
            entry_mode_out = "pullback_short"
        elif normalized_side == SignalDirection.SHORT.value:
            short_decision = _weak_down_choppy_short_decision(
                signal_strength=signal_strength,
                confirmation_result=confirmation_result,
                micro_bucket=str(micro["bucket"]),
                current_soft_issues=len(soft_issues),
                symbol_health=symbol_health,
                ml_negative_edge=ml_negative_edge,
                config=config,
                strict_policy=strict_policy,
            )
            soft_issues.extend([str(issue) for issue in short_decision["soft_issues"]])
            tags.extend([str(tag) for tag in short_decision["tags"]])
            entry_mode_out = str(short_decision["entry_mode"])
            policy_flags.update(dict(short_decision["policy_flags"]))
            if bool(short_decision["force_wait_pullback"]):
                force_wait_pullback = True
            if short_decision["decision_type"]:
                decision_type_override = str(short_decision["decision_type"])
            if short_decision["final_multiplier"] is not None:
                final_multiplier_override = float(short_decision["final_multiplier"])
                components["weak_choppy_direct"] = final_multiplier_override
            else:
                multipliers.append(("regime", 0.35))
    elif regime == "range_chop":
        if _learning_mode_value(config, "allow_range_chop_exploration", True):
            multipliers.append(("regime", 0.10))
            soft_issues.append("range-chop-exploration")
            tags.append("range-chop")
        else:
            force_wait_pullback = normalized_side == SignalDirection.SHORT.value
            soft_issues.append("range-chop-no-trend-follow")
            tags.append("range-chop")

    ml_multiplier = 1.0
    if ml_negative_edge and hard_issues:
        hard_issues.append("ml-negative-edge-plus-hard-execution-issue")
    elif ml_negative_edge:
        soft_count_without_ml = len(soft_issues)
        if soft_count_without_ml == 0:
            ml_multiplier = _ml_policy_value(config, "negative_edge_only_multiplier", 0.35)
            soft_issues.append("ml-negative-edge")
            tags.append("ml-negative-edge")
        elif soft_count_without_ml == 1:
            ml_multiplier = _ml_policy_value(
                config,
                "negative_edge_plus_one_soft_issue_multiplier",
                0.15,
            )
            soft_issues.append("ml-negative-edge-plus-one-soft-issue")
            tags.append("ml-negative-edge")
        else:
            ml_multiplier = _ml_policy_value(
                config,
                "negative_edge_plus_multiple_soft_issues_multiplier",
                0.08,
            )
            soft_issues.append("ml-negative-edge-plus-multiple-soft-issues")
            tags.append("ml-negative-edge")
        multipliers.append(("ml_negative_edge", ml_multiplier))

    effective_multiplier = _clamped_product([value for _, value in multipliers])
    components.update({name: value for name, value in multipliers if name != "base"})
    if final_multiplier_override is not None and not hard_issues and not force_wait_pullback:
        effective_multiplier = max(0.0, min(1.0, float(final_multiplier_override)))

    if hard_issues:
        decision_type = PolicyDecisionType.HARD_REJECT.value
        allow_trade = False
        effective_multiplier = 0.0
        entry_mode_out = "reject"
    elif force_wait_pullback:
        decision_type = PolicyDecisionType.WAIT_PULLBACK.value
        allow_trade = False
        effective_multiplier = 0.0
        entry_mode_out = "wait"
    elif force_shadow_only:
        decision_type = PolicyDecisionType.SHADOW_ONLY.value
        allow_trade = False
        effective_multiplier = 0.0
    else:
        decision_type = decision_type_override or _decision_from_issues(soft_issues, tags)
        allow_trade = True
        if decision_type == PolicyDecisionType.PROBE_TRADE.value and not _learning_mode_value(
            config,
            "allow_probe_trades",
            True,
        ):
            decision_type = PolicyDecisionType.SHADOW_ONLY.value
            allow_trade = False
        elif decision_type == PolicyDecisionType.EXPLORATION_TRADE.value and not _learning_mode_value(
            config,
            "allow_exploration_trades",
            True,
        ):
            decision_type = PolicyDecisionType.SHADOW_ONLY.value
            allow_trade = False

        if allow_trade and effective_multiplier < _learning_mode_value(
            config,
            "min_effective_risk_multiplier_to_trade",
            0.05,
        ):
            decision_type = PolicyDecisionType.SHADOW_ONLY.value
            allow_trade = False

    if final_multiplier_override is None and decision_type == PolicyDecisionType.PROBE_TRADE.value:
        effective_multiplier *= _mode_risk_multiplier(config, "probe", 0.25)
    elif final_multiplier_override is None and decision_type == PolicyDecisionType.EXPLORATION_TRADE.value:
        effective_multiplier *= _mode_risk_multiplier(config, "exploration", 0.10)
    effective_multiplier = max(0.0, min(1.0, effective_multiplier if allow_trade else 0.0))
    if allow_trade and effective_multiplier < _learning_mode_value(
        config,
        "min_effective_risk_multiplier_to_trade",
        0.05,
    ):
        decision_type = PolicyDecisionType.SHADOW_ONLY.value
        allow_trade = False
        effective_multiplier = 0.0
    components["decision_mode"] = (
        _mode_risk_multiplier(config, "probe", 0.25)
        if final_multiplier_override is None and decision_type == PolicyDecisionType.PROBE_TRADE.value
        else _mode_risk_multiplier(config, "exploration", 0.10)
        if final_multiplier_override is None and decision_type == PolicyDecisionType.EXPLORATION_TRADE.value
        else 1.0
    )

    reasons = _dedupe([*soft_issues, *hard_issues]) or ["relaxed-default-allow"]
    strict_decision = strict_policy.strict_policy_decision
    relaxed_only = allow_trade and not strict_policy.allow_trade
    if allow_trade and strict_decision == "wait":
        policy_flags.setdefault("would_have_waited_pullback_strict", True)
        if decision_type in {
            PolicyDecisionType.PROBE_TRADE.value,
            PolicyDecisionType.EXPLORATION_TRADE.value,
        }:
            policy_flags.setdefault("strict_wait_overridden_by_relaxed_probe", True)
    if policy_flags.get("probe_now_with_pending_addon"):
        policy_flags.setdefault("pending_addon_created", False)
    confirmation_metadata = {
        "status": confirmation_result["status"],
        "mode": confirmation_result["mode"],
        "adverse_ret": confirmation_result["adverse_ret"],
        "bars_used": confirmation_result["bars_used"],
        "relaxed_decision": decision_type,
        "strict_decision": "reject" if _strict_confirmation_blocks(confirmation) else "allow",
    }
    symbol_metadata = {
        "status": symbol_status,
        "strict_policy_decision": (
            "reject" if strict_policy.strict_policy_decision == "reject" else "allow"
        ),
        "relaxed_policy_decision": decision_type,
        "multiplier": symbol_multiplier,
        "reason": _symbol_health_reason(symbol_status, symbol, observe_only_configured),
    }
    actual_reasons = _dedupe(reasons + (list(strict_policy.reasons) if relaxed_only else []))
    return TradePolicy(
        allow_trade=allow_trade,
        entry_mode=entry_mode_out,
        risk_multiplier=round(effective_multiplier, 6),
        reasons=reasons,
        symbol_health=symbol_status,
        risk_components=components,
        decision_type=decision_type,
        strict_policy_decision=strict_decision,
        strict_policy_reasons=list(strict_policy.reasons),
        actual_policy_reasons=actual_reasons,
        soft_issues=_dedupe(soft_issues),
        hard_issues=_dedupe(hard_issues),
        tags=_dedupe(tags),
        actual_policy_profile=learning_profile or "relaxed_paper_learning",
        would_strict_policy_trade=strict_policy.allow_trade,
        would_strict_policy_risk_multiplier=strict_policy.risk_multiplier,
        relaxed_only_trade=relaxed_only,
        microstructure_bucket=str(micro["bucket"]),
        confirmation_5m_status=str(confirmation_result["status"]),
        ml_negative_edge=ml_negative_edge,
        effective_risk_multiplier=round(effective_multiplier, 6),
        strict_policy_metadata=strict_policy.as_metadata(),
        side_policy={
            "side": normalized_side,
            "regime": regime,
            "long_mode": long_context.get("long_mode", "") if long_context else "",
            "normal_enabled": side_policy["normal_enabled"],
            "probe_enabled": side_policy["probe_enabled"],
            "exploration_enabled": side_policy["exploration_enabled"],
            "multiplier": side_multiplier,
            "reason": side_policy["reason"],
        },
        symbol_health_metadata=symbol_metadata,
        confirmation_5m=confirmation_metadata,
        policy_flags=policy_flags,
        long_context=long_context,
    )


def _market_selloff_short_decision(
    *,
    signal_strength: float | None,
    requested_mode: str,
    confirmation_result: dict[str, object],
    micro_bucket: str,
    current_soft_issues: int,
    symbol_health: str,
    ml_negative_edge: bool,
    config: object | None,
    strict_policy: TradePolicy,
) -> dict[str, object]:
    status = str(confirmation_result.get("status", ""))
    base_flags = {
        "selloff_policy_override_applied": True,
        "do_not_wait_for_pullback_in_broad_selloff": bool(
            getattr(
                getattr(config, "paper_alpha_capture", None),
                "do_not_wait_for_pullback_in_broad_selloff",
                True,
            )
        ),
        "would_have_waited_pullback_strict": strict_policy.strict_policy_decision == "wait",
    }
    if bool(confirmation_result.get("wait_pullback")):
        return {
            "decision_type": "",
            "entry_mode": "wait",
            "final_multiplier": None,
            "force_wait_pullback": True,
            "force_shadow_only": False,
            "soft_issues": ["selloff-strong-rebound-stop-chase"],
            "tags": ["selloff-wait-pullback", "stop-chase"],
            "policy_flags": {**base_flags, "selloff_wait_pullback_selected": True},
        }
    if status.startswith("extreme_"):
        return {
            "decision_type": PolicyDecisionType.SHADOW_ONLY.value,
            "entry_mode": "selloff_extreme_rebound_shadow_short",
            "final_multiplier": 0.0,
            "force_wait_pullback": False,
            "force_shadow_only": True,
            "soft_issues": ["selloff-extreme-adverse-confirmation"],
            "tags": ["selloff-shadow-only"],
            "policy_flags": {**base_flags, "selloff_wait_pullback_selected": False},
        }

    signal_value = float(signal_strength or 0.0)
    risk_cfg = getattr(getattr(config, "market_selloff_impulse", None), "risk", None)
    if requested_mode == "pullback_short":
        entry_mode = "post_selloff_failed_rebound_short"
        decision_type = PolicyDecisionType.PROBE_TRADE.value
        multiplier = float(getattr(risk_cfg, "post_selloff_failed_rebound_short_multiplier", 0.35))
        issue = "market-selloff-post-rebound-short"
        tag = "selloff-failed-rebound-short"
    elif micro_bucket == "exploration" or current_soft_issues >= 5 or signal_value < 0.15:
        entry_mode = "panic_probe_short"
        decision_type = PolicyDecisionType.EXPLORATION_TRADE.value
        multiplier = float(getattr(risk_cfg, "panic_probe_short_multiplier", 0.25))
        issue = "market-selloff-panic-probe-short"
        tag = "selloff-panic-probe-short"
    elif signal_value >= 0.30 and micro_bucket in {"normal", "unknown", "unavailable"} and current_soft_issues <= 2:
        entry_mode = "market_breakdown_short"
        decision_type = PolicyDecisionType.NORMAL_TRADE.value
        multiplier = float(getattr(risk_cfg, "market_breakdown_short_multiplier", 0.60))
        issue = "market-selloff-breakdown-short"
        tag = "selloff-breakdown-short"
    else:
        entry_mode = "selloff_momentum_short"
        decision_type = PolicyDecisionType.PROBE_TRADE.value
        multiplier = float(getattr(risk_cfg, "selloff_momentum_short_multiplier", 0.45))
        issue = "market-selloff-momentum-short"
        tag = "selloff-momentum-short"

    soft_issues = [issue]
    tags = [tag, "selloff-short-now", "strict-wait-overridden"]
    if status == "neutral":
        multiplier *= 0.85
        soft_issues.append("selloff-neutral-5m-reduced-short")
    elif status.startswith("mild_"):
        multiplier *= 0.75
        soft_issues.append("selloff-mild-rebound-reduced-short")
    if symbol_health == "probation":
        multiplier *= 0.75
    elif symbol_health in {"weak", "observe_only"}:
        multiplier *= 0.50
    if ml_negative_edge:
        if current_soft_issues <= 1:
            multiplier *= 0.50
            soft_issues.append("selloff-ml-negative-edge-reduced")
        else:
            multiplier *= 0.35
            soft_issues.append("selloff-ml-negative-edge-plus-soft-reduced")
            decision_type = PolicyDecisionType.EXPLORATION_TRADE.value
        tags.append("selloff-ml-size-reduced")

    return {
        "decision_type": decision_type,
        "entry_mode": entry_mode,
        "final_multiplier": round(max(0.0, min(1.0, multiplier)), 6),
        "force_wait_pullback": False,
        "force_shadow_only": False,
        "soft_issues": soft_issues,
        "tags": tags,
        "policy_flags": {
            **base_flags,
            "selloff_direct_short_selected": True,
            "selloff_entry_mode": entry_mode,
            "strict_wait_overridden_by_relaxed_probe": strict_policy.strict_policy_decision == "wait",
        },
    }


def _weak_down_choppy_short_decision(
    *,
    signal_strength: float | None,
    confirmation_result: dict[str, object],
    micro_bucket: str,
    current_soft_issues: int,
    symbol_health: str,
    ml_negative_edge: bool,
    config: object | None,
    strict_policy: TradePolicy,
) -> dict[str, object]:
    cfg = getattr(getattr(config, "regime_policy", None), "weak_down_choppy", None)
    signal_value = float(signal_strength or 0.0)
    confirmation_mode = _weak_choppy_short_confirmation_mode(config, str(confirmation_result.get("status", "")))
    addon_multiplier = _weak_choppy_value(config, "pullback_addon_multiplier", 0.15)
    base_flags = {
        "would_have_waited_pullback_strict": strict_policy.strict_policy_decision == "wait",
        "strict_policy_keeps_wait_pullback": bool(
            getattr(cfg, "strict_policy_keeps_wait_pullback", True)
        ),
    }

    if confirmation_mode == "wait_pullback_only" or bool(confirmation_result.get("wait_pullback")):
        return {
            "decision_type": "",
            "entry_mode": "wait",
            "final_multiplier": None,
            "force_wait_pullback": True,
            "soft_issues": ["weak-down-choppy-wait-pullback"],
            "tags": ["wait-pullback", "weak-choppy-wait-only"],
            "policy_flags": {**base_flags, "weak_choppy_wait_only_selected": True},
        }
    if confirmation_mode == "shadow_or_reject":
        return {
            "decision_type": PolicyDecisionType.SHADOW_ONLY.value,
            "entry_mode": "weak_choppy_adverse_confirmation_shadow_short",
            "final_multiplier": 0.0,
            "force_wait_pullback": False,
            "soft_issues": ["weak-down-choppy-extreme-adverse-confirmation"],
            "tags": ["weak-choppy-shadow-only"],
            "policy_flags": {**base_flags, "weak_choppy_wait_only_selected": False},
        }

    probe_enabled = bool(_weak_choppy_value(config, "short_direct_probe_enabled", True))
    exploration_enabled = bool(_weak_choppy_value(config, "short_direct_exploration_enabled", True))
    enable_addon = bool(_weak_choppy_value(config, "enable_probe_now_with_pending_addon", True))
    create_addon = bool(_weak_choppy_value(config, "create_pullback_addon_after_direct_probe", True))
    probe_min = _weak_choppy_value(config, "short_direct_probe_min_signal_strength", 0.20)
    exploration_min = _weak_choppy_value(config, "short_direct_exploration_min_signal_strength", 0.12)
    max_soft = int(_weak_choppy_value(config, "short_direct_probe_max_soft_issues", 4))
    allow_ml_exploration = bool(_weak_choppy_value(config, "allow_ml_negative_edge_exploration", True))
    book_ok = micro_bucket in {"normal", "probe", "exploration", "unknown", "unavailable"}
    can_probe = (
        probe_enabled
        and book_ok
        and not ml_negative_edge
        and signal_value >= probe_min
        and current_soft_issues <= max_soft
    )
    can_explore = (
        exploration_enabled
        and book_ok
        and (allow_ml_exploration or not ml_negative_edge)
        and signal_value >= exploration_min
    )
    if not can_probe and not can_explore:
        return {
            "decision_type": "",
            "entry_mode": "wait",
            "final_multiplier": None,
            "force_wait_pullback": True,
            "soft_issues": ["weak-down-choppy-wait-pullback"],
            "tags": ["wait-pullback", "weak-choppy-wait-only"],
            "policy_flags": {**base_flags, "weak_choppy_wait_only_selected": True},
        }

    if can_probe:
        decision_type = PolicyDecisionType.PROBE_TRADE.value
        entry_mode = "weak_choppy_direct_probe_short"
        multiplier = _weak_choppy_value(config, "short_direct_probe_multiplier", 0.25)
        issue = "weak-down-choppy-direct-probe"
        tag = "weak-choppy-probe-now"
    else:
        decision_type = PolicyDecisionType.EXPLORATION_TRADE.value
        entry_mode = "weak_choppy_direct_exploration_short"
        multiplier = _weak_choppy_value(config, "short_direct_exploration_multiplier", 0.10)
        issue = "weak-down-choppy-direct-exploration"
        tag = "weak-choppy-exploration-now"

    if symbol_health == "probation":
        multiplier *= 0.5
    elif symbol_health in {"weak", "observe_only"}:
        multiplier *= 0.35
    status = str(confirmation_result.get("status", ""))
    if status == "neutral":
        multiplier *= 0.75
    elif status.startswith("mild_"):
        multiplier = min(multiplier * 0.75, _weak_choppy_value(config, "short_direct_exploration_multiplier", 0.10))
    if ml_negative_edge:
        multiplier = min(
            multiplier,
            max(
                _learning_mode_value(config, "min_effective_risk_multiplier_to_trade", 0.05),
                _weak_choppy_value(config, "short_direct_exploration_multiplier", 0.25),
            ),
        )
        decision_type = PolicyDecisionType.EXPLORATION_TRADE.value
        entry_mode = "weak_choppy_direct_exploration_short"
        issue = "weak-down-choppy-negative-edge-exploration"
        tag = "weak-choppy-negative-edge-exploration"

    return {
        "decision_type": decision_type,
        "entry_mode": entry_mode,
        "final_multiplier": round(max(0.0, min(1.0, multiplier)), 6),
        "force_wait_pullback": False,
        "soft_issues": [issue],
        "tags": [tag, "strict-wait-overridden"] if strict_policy.strict_policy_decision == "wait" else [tag],
        "policy_flags": {
            **base_flags,
            "probe_now_with_pending_addon": bool(enable_addon and create_addon),
            "pending_addon_type": "wait_pullback_short",
            "pending_addon_multiplier": round(addon_multiplier, 6),
            "pending_addon_created": False,
            "strict_wait_overridden_by_relaxed_probe": strict_policy.strict_policy_decision == "wait",
            "weak_choppy_probe_now_selected": decision_type == PolicyDecisionType.PROBE_TRADE.value,
            "weak_choppy_exploration_now_selected": decision_type == PolicyDecisionType.EXPLORATION_TRADE.value,
        },
    }


def _long_regime_decision(
    *,
    regime: str,
    signal_strength: float | None,
    confirmation: dict[str, object] | None,
    confirmation_result: dict[str, object],
    side_policy: dict[str, object],
    config: object | None,
    strict_policy: TradePolicy,
) -> dict[str, object]:
    signal_value = float(signal_strength or 0.0)
    ret_window = _float_or_none((confirmation or {}).get("ret_window"))
    recovery_confirmed = bool(ret_window is not None and ret_window > 0.0)
    strong_recovery = bool(ret_window is not None and ret_window >= _confirmation_value(config, "mild_adverse_ret", 0.0025))
    aligned = str(confirmation_result.get("status", "")) == "aligned"
    base_context = {
        "regime": regime,
        "rebound_confirmed": recovery_confirmed,
        "failed_breakdown": recovery_confirmed,
        "reclaim_level": (confirmation or {}).get("latest_close") if isinstance(confirmation, dict) else None,
        "strict_policy_decision": strict_policy.strict_policy_decision,
        "normal_long_allowed": False,
        "size_reason": "long_learning_small_size",
    }

    if side_policy.get("hard_issue"):
        return _long_decision_result(
            decision_type=PolicyDecisionType.HARD_REJECT.value,
            entry_mode="reject",
            final_multiplier=0.0,
            soft_issue="",
            hard_issue=str(side_policy["hard_issue"]),
            tags=[],
            force_shadow_only=False,
            long_context={**base_context, "long_mode": "reject"},
            strict_policy=strict_policy,
        )

    if regime == "clean_uptrend":
        long_cfg = getattr(getattr(getattr(config, "regime_policy", None), "clean_uptrend", None), "long", None)
        normal_min = float(getattr(long_cfg, "long_direct_normal_min_signal_strength", 0.35))
        probe_min = float(getattr(long_cfg, "long_direct_probe_min_signal_strength", 0.20))
        if bool(getattr(long_cfg, "allow_direct_trend_long", True)) and bool(side_policy.get("normal_enabled")) and signal_value >= normal_min:
            multiplier = float(getattr(long_cfg, "long_normal_multiplier", 0.25))
            return _long_decision_result(
                decision_type=PolicyDecisionType.NORMAL_TRADE.value,
                entry_mode="clean_uptrend_direct_long",
                final_multiplier=multiplier,
                soft_issue="clean-uptrend-long-normal-small-size",
                tags=["long-normal"],
                force_shadow_only=False,
                long_context={
                    **base_context,
                    "long_mode": "normal",
                    "normal_long_allowed": True,
                    "relaxed_policy_decision": PolicyDecisionType.NORMAL_TRADE.value,
                },
                strict_policy=strict_policy,
            )
        if bool(side_policy.get("probe_enabled")) and signal_value >= probe_min:
            multiplier = float(getattr(long_cfg, "long_probe_multiplier", 0.15))
            return _long_decision_result(
                decision_type=PolicyDecisionType.PROBE_TRADE.value,
                entry_mode="clean_uptrend_direct_probe_long",
                final_multiplier=multiplier,
                soft_issue="clean-uptrend-long-probe",
                tags=["long-probe"],
                force_shadow_only=False,
                long_context={**base_context, "long_mode": "probe"},
                strict_policy=strict_policy,
            )
        return _long_shadow_result(base_context, strict_policy, "clean_uptrend_long_signal_too_weak")

    if regime == "mixed":
        long_cfg = getattr(getattr(getattr(config, "regime_policy", None), "mixed", None), "long", None)
        if bool(getattr(long_cfg, "require_5m_aligned_or_recovery", True)) and not (aligned and recovery_confirmed):
            return _long_shadow_result(base_context, strict_policy, "mixed_long_requires_recovery")
        if bool(getattr(long_cfg, "allow_long_probe", True)) and signal_value >= 0.20:
            multiplier = float(getattr(long_cfg, "long_probe_multiplier", 0.10))
            return _long_decision_result(
                decision_type=PolicyDecisionType.PROBE_TRADE.value,
                entry_mode="rebound_probe_long",
                final_multiplier=multiplier,
                soft_issue="mixed-long-recovery-probe",
                tags=["long-probe"],
                force_shadow_only=False,
                long_context={**base_context, "long_mode": "probe"},
                strict_policy=strict_policy,
            )
        if bool(getattr(long_cfg, "allow_long_exploration", True)):
            multiplier = float(getattr(long_cfg, "long_exploration_multiplier", 0.05))
            return _long_decision_result(
                decision_type=PolicyDecisionType.EXPLORATION_TRADE.value,
                entry_mode="rebound_exploration_long",
                final_multiplier=multiplier,
                soft_issue="mixed-long-recovery-exploration",
                tags=["long-exploration"],
                force_shadow_only=False,
                long_context={**base_context, "long_mode": "exploration"},
                strict_policy=strict_policy,
            )

    if regime == "range_chop":
        long_cfg = getattr(getattr(getattr(config, "regime_policy", None), "range_chop", None), "long", None)
        if bool(getattr(long_cfg, "require_failed_breakdown_or_reclaim", True)) and not recovery_confirmed:
            return _long_shadow_result(base_context, strict_policy, "range_chop_long_requires_reclaim")
        if bool(getattr(long_cfg, "allow_mean_reversion_long_exploration", True)):
            multiplier = float(getattr(long_cfg, "long_exploration_multiplier", 0.05))
            return _long_decision_result(
                decision_type=PolicyDecisionType.EXPLORATION_TRADE.value,
                entry_mode="range_failed_breakdown_long",
                final_multiplier=multiplier,
                soft_issue="range-chop-long-reclaim-exploration",
                tags=["long-exploration", "range-chop"],
                force_shadow_only=False,
                long_context={**base_context, "long_mode": "exploration"},
                strict_policy=strict_policy,
            )

    if regime == "weak_down_choppy":
        long_cfg = getattr(getattr(getattr(config, "regime_policy", None), "weak_down_choppy", None), "long", None)
        if long_cfg is None:
            long_cfg = getattr(getattr(config, "regime_policy", None), "weak_down_choppy_long", None)
        if not strong_recovery:
            return _long_shadow_result(base_context, strict_policy, "weak_down_choppy_long_requires_rebound")
        if bool(getattr(long_cfg, "allow_rebound_probe_long", True)) and signal_value >= 0.20:
            multiplier = _at_least_min_trade_multiplier(
                config,
                float(getattr(long_cfg, "long_probe_multiplier", 0.05)),
            )
            return _long_decision_result(
                decision_type=PolicyDecisionType.PROBE_TRADE.value,
                entry_mode="rebound_probe_long",
                final_multiplier=multiplier,
                soft_issue="weak-down-choppy-rebound-long-probe",
                tags=["long-probe", "weak-down-choppy-long"],
                force_shadow_only=False,
                long_context={**base_context, "long_mode": "probe"},
                strict_policy=strict_policy,
            )
        if bool(getattr(long_cfg, "allow_rebound_exploration_long", True)):
            multiplier = _at_least_min_trade_multiplier(
                config,
                float(getattr(long_cfg, "long_exploration_multiplier", 0.03)),
            )
            return _long_decision_result(
                decision_type=PolicyDecisionType.EXPLORATION_TRADE.value,
                entry_mode="rebound_exploration_long",
                final_multiplier=multiplier,
                soft_issue="weak-down-choppy-rebound-long-exploration",
                tags=["long-exploration", "weak-down-choppy-long"],
                force_shadow_only=False,
                long_context={**base_context, "long_mode": "exploration"},
                strict_policy=strict_policy,
            )

    if regime == "market_selloff_impulse":
        long_cfg = getattr(getattr(config, "market_selloff_impulse", None), "long", None)
        if bool(getattr(long_cfg, "require_reclaim_confirmation", True)) and not strong_recovery:
            return _long_shadow_result(base_context, strict_policy, "market_selloff_long_requires_reclaim")
        if bool(getattr(long_cfg, "allow_capitulation_bounce_probe", True)):
            multiplier = _at_least_min_trade_multiplier(
                config,
                float(getattr(long_cfg, "capitulation_bounce_probe_multiplier", 0.03)),
            )
            return _long_decision_result(
                decision_type=PolicyDecisionType.PROBE_TRADE.value,
                entry_mode="capitulation_bounce_probe_long",
                final_multiplier=multiplier,
                soft_issue="market-selloff-capitulation-bounce-long-probe",
                tags=["long-probe", "selloff-bounce"],
                force_shadow_only=False,
                long_context={**base_context, "long_mode": "probe"},
                strict_policy=strict_policy,
            )

    if bool(side_policy.get("probe_enabled")) and signal_value >= 0.20:
        return _long_decision_result(
            decision_type=PolicyDecisionType.PROBE_TRADE.value,
            entry_mode="trend_long",
            final_multiplier=float(side_policy.get("multiplier", 0.10) or 0.10),
            soft_issue="long-generic-probe",
            tags=["long-probe"],
            force_shadow_only=False,
            long_context={**base_context, "long_mode": "probe"},
            strict_policy=strict_policy,
        )
    return _long_shadow_result(base_context, strict_policy, "long_policy_shadow_only")


def _long_decision_result(
    *,
    decision_type: str,
    entry_mode: str,
    final_multiplier: float | None,
    soft_issue: str,
    tags: list[str],
    force_shadow_only: bool,
    long_context: dict[str, object],
    strict_policy: TradePolicy,
    hard_issue: str = "",
) -> dict[str, object]:
    context = {
        **long_context,
        "relaxed_policy_decision": decision_type,
    }
    return {
        "decision_type": decision_type,
        "entry_mode": entry_mode,
        "final_multiplier": final_multiplier,
        "force_wait_pullback": False,
        "force_shadow_only": force_shadow_only,
        "soft_issue": soft_issue,
        "hard_issue": hard_issue,
        "tags": tags,
        "long_context": context,
        "policy_flags": {
            "long_probe_trade": decision_type == PolicyDecisionType.PROBE_TRADE.value,
            "long_exploration_trade": decision_type == PolicyDecisionType.EXPLORATION_TRADE.value,
            "long_normal_trade": decision_type == PolicyDecisionType.NORMAL_TRADE.value,
            "strict_reject_overridden_by_relaxed_long_probe": (
                decision_type in {PolicyDecisionType.PROBE_TRADE.value, PolicyDecisionType.EXPLORATION_TRADE.value}
                and strict_policy.strict_policy_decision == "reject"
            ),
        },
    }


def _long_shadow_result(
    base_context: dict[str, object],
    strict_policy: TradePolicy,
    reason: str,
) -> dict[str, object]:
    return _long_decision_result(
        decision_type=PolicyDecisionType.SHADOW_ONLY.value,
        entry_mode="wait_failed_breakdown_reclaim_long",
        final_multiplier=0.0,
        soft_issue=reason.replace("_", "-"),
        tags=["long-shadow-only"],
        force_shadow_only=True,
        long_context={**base_context, "long_mode": "shadow_only", "reason": reason},
        strict_policy=strict_policy,
    )


def _side_value(side: SignalDirection | str) -> str:
    return side.value if isinstance(side, SignalDirection) else str(side)


def _default_entry_mode(side: str) -> str:
    if side == SignalDirection.SHORT.value:
        return "trend_short"
    if side == SignalDirection.LONG.value:
        return "trend_long"
    return "wait"


def _ml_blocks(ml_feedback: dict[str, object] | None) -> bool:
    return bool(ml_feedback and ml_feedback.get("blocks_entry"))


def _adverse_book(book: dict[str, object] | None) -> bool:
    if not book:
        return False
    if book.get("blocks_entry"):
        return True
    imbalance = book.get("side_imbalance", book.get("imbalance"))
    try:
        return float(imbalance) < -0.35
    except (TypeError, ValueError):
        return False


def _bucket_spread_selloff(spread_bps: float | None, config: object | None) -> BucketResult:
    if spread_bps is None:
        return BucketResult("unknown", 1.0, "")
    cfg = _selloff_microstructure_config(config)
    normal = float(getattr(cfg, "max_entry_spread_bps_normal", 20.0))
    probe = float(getattr(cfg, "max_entry_spread_bps_probe", 32.0))
    exploration = float(getattr(cfg, "max_entry_spread_bps_exploration", 40.0))
    if spread_bps <= normal:
        return BucketResult("normal", 1.0, "")
    if spread_bps <= probe:
        return BucketResult("probe", 0.60, "selloff-slightly-wide-spread")
    if spread_bps <= exploration:
        return BucketResult("exploration", 0.35, "selloff-wide-spread-bounded")
    return BucketResult("reject", 0.0, "selloff-spread-above-hard-limit", hard=True)


def _bucket_liquidity_cover_selloff(cover: float | None, config: object | None) -> BucketResult:
    if cover is None:
        return BucketResult("unknown", 1.0, "")
    cfg = _selloff_microstructure_config(config)
    normal = float(getattr(cfg, "min_entry_liquidity_cover_normal", 1.0))
    probe = float(getattr(cfg, "min_entry_liquidity_cover_probe", 0.6))
    exploration = float(getattr(cfg, "min_entry_liquidity_cover_exploration", 0.4))
    if cover >= normal:
        return BucketResult("normal", 1.0, "")
    if cover >= probe:
        return BucketResult("probe", 0.60, "selloff-thin-book-probe")
    if cover >= exploration:
        return BucketResult("exploration", 0.35, "selloff-thin-book-exploration")
    return BucketResult("reject", 0.0, "selloff-liquidity-cover-below-hard-limit", hard=True)


def _bucket_imbalance_selloff(side_imbalance: float | None, config: object | None) -> BucketResult:
    if side_imbalance is None:
        return BucketResult("unknown", 1.0, "")
    cfg = _selloff_microstructure_config(config)
    normal = float(getattr(cfg, "min_entry_book_imbalance_normal", -0.60))
    probe = float(getattr(cfg, "min_entry_book_imbalance_probe", -0.85))
    exploration = float(getattr(cfg, "min_entry_book_imbalance_exploration", -0.95))
    if side_imbalance >= normal:
        return BucketResult("normal", 1.0, "")
    if side_imbalance >= probe:
        return BucketResult("probe", 0.60, "selloff-mild-adverse-book")
    if side_imbalance >= exploration:
        return BucketResult("exploration", 0.35, "selloff-medium-adverse-book")
    return BucketResult("reject", 0.0, "selloff-adverse-book-below-hard-limit", hard=True)


def _microstructure_assessment(
    book: dict[str, object] | None,
    *,
    regime: str,
    config: object | None,
) -> dict[str, object]:
    if not book:
        return {
            "bucket": "unknown",
            "multiplier": 1.0,
            "soft_issues": [],
            "hard_issue": "",
            "tags": [],
        }
    if not book.get("available", True):
        return {
            "bucket": "unavailable",
            "multiplier": 1.0,
            "soft_issues": [],
            "hard_issue": "",
            "tags": ["order-book-unavailable"],
        }
    spread = _float_or_none(book.get("spread_bps"))
    cover = _float_or_none(book.get("entry_liquidity_cover"))
    imbalance = _float_or_none(book.get("side_imbalance", book.get("imbalance")))
    if regime == "market_selloff_impulse":
        buckets = [
            _bucket_spread_selloff(spread, config),
            _bucket_liquidity_cover_selloff(cover, config),
            _bucket_imbalance_selloff(imbalance, config),
        ]
    else:
        buckets = [
            bucket_spread(spread),
            bucket_liquidity_cover(cover),
            bucket_imbalance(imbalance),
        ]
    hard = next((bucket.issue for bucket in buckets if bucket.hard), "")
    soft = [bucket.issue for bucket in buckets if bucket.issue and not bucket.hard]
    worst_bucket = _worst_bucket([bucket.bucket for bucket in buckets])
    multiplier = _clamped_product(
        [bucket.multiplier for bucket in buckets if not bucket.hard]
    )
    tags = []
    if soft or hard:
        tags.append("adverse-book")
    return {
        "bucket": worst_bucket,
        "multiplier": multiplier,
        "soft_issues": soft,
        "hard_issue": hard,
        "tags": tags,
    }


def _confirmation_assessment(
    confirmation: dict[str, object] | None,
    *,
    side: str,
    regime: str,
    config: object | None,
) -> dict[str, object]:
    if not confirmation or not confirmation.get("available"):
        return {
            "status": "unavailable",
            "mode": "none",
            "adverse_ret": None,
            "bars_used": 0,
            "multiplier": 1.0,
            "soft_issues": [],
            "hard_issue": "",
            "wait_pullback": False,
            "tags": [],
        }

    ret_window = _float_or_none(confirmation.get("ret_window")) or 0.0
    if side == SignalDirection.SHORT.value:
        adverse_ret = max(0.0, ret_window)
        rebound_label = "rebound_against_short"
    else:
        adverse_ret = max(0.0, -ret_window)
        rebound_label = "pullback_against_long"

    bars_used = int(confirmation.get("bars", 0) or 0)
    mild_threshold = _confirmation_value(config, "mild_adverse_ret", 0.0025)
    strong_threshold = _confirmation_value(config, "strong_adverse_ret", 0.005)
    extreme_threshold = _confirmation_value(config, "extreme_adverse_ret", 0.012)
    selloff_confirmation = (
        _selloff_confirmation_config(config)
        if regime == "market_selloff_impulse" and side == SignalDirection.SHORT.value
        else None
    )

    if adverse_ret <= 0:
        status = "aligned"
        mode = "normal"
        multiplier = 1.0
        soft: list[str] = []
        hard = ""
        wait = False
    elif adverse_ret < mild_threshold:
        status = "neutral"
        mode = (
            str(getattr(selloff_confirmation, "neutral_confirmation_mode", "allow_reduced_short"))
            if selloff_confirmation is not None
            else _confirmation_value(config, "neutral_confirmation_mode", "probe")
        )
        multiplier = 0.65 if mode == "allow_reduced_short" else 0.35
        soft = ["neutral-5m-confirmation"]
        hard = ""
        wait = False
    elif adverse_ret < strong_threshold:
        status = f"mild_{rebound_label}"
        mode = (
            str(getattr(selloff_confirmation, "mild_rebound_against_short_mode", "allow_reduced_short"))
            if selloff_confirmation is not None
            else _confirmation_value(config, "mild_rebound_against_short_mode", "exploration_or_wait")
        )
        multiplier = 0.45 if mode == "allow_reduced_short" else 0.10
        soft = [f"mild-5m-{rebound_label.replace('_', '-')}"]
        hard = ""
        wait = mode == "wait_pullback"
    elif adverse_ret < extreme_threshold:
        status = f"strong_{rebound_label}"
        mode = (
            str(getattr(selloff_confirmation, "strong_rebound_against_short_mode", "stop_chase_wait_pullback"))
            if selloff_confirmation is not None
            else _confirmation_value(config, "strong_rebound_against_short_mode", "wait_pullback")
        )
        wait_modes = {"wait_pullback", "stop_chase_wait_pullback"}
        multiplier = 0.0 if mode in wait_modes else 0.08
        soft = [f"strong-5m-{rebound_label.replace('_', '-')}"]
        hard = ""
        wait = mode in wait_modes
    else:
        status = f"extreme_{rebound_label}"
        if selloff_confirmation is not None:
            mode = str(getattr(selloff_confirmation, "extreme_adverse_mode", "shadow_or_reject"))
        else:
            mode = "reject" if _confirmation_value(config, "hard_block_rebound_against_short", False) else "shadow_only"
        multiplier = 0.0
        soft = []
        hard = "extreme-5m-adverse-move" if mode == "reject" else ""
        wait = False

    return {
        "status": status,
        "mode": mode,
        "adverse_ret": round(adverse_ret, 6),
        "bars_used": bars_used,
        "multiplier": multiplier,
        "soft_issues": soft,
        "hard_issue": hard,
        "wait_pullback": wait,
        "tags": ["entry-confirmation"] if status not in {"aligned", "unavailable"} else [],
    }


def _selloff_microstructure_config(config: object | None) -> object | None:
    return getattr(getattr(config, "learning_microstructure", None), "market_selloff_impulse", None)


def _selloff_confirmation_config(config: object | None) -> object | None:
    return getattr(getattr(config, "confirmation_5m", None), "market_selloff_impulse", None)


def _signal_quality_issue(
    *,
    signal_strength: float | None,
    trend_strength: float | None,
    adx: float | None,
    config: object | None,
) -> dict[str, object] | None:
    normal_signal = _signal_mode_value(config, "normal", "min_signal_strength", 0.30)
    probe_signal = _signal_mode_value(config, "probe", "min_signal_strength", 0.20)
    exploration_signal = _signal_mode_value(config, "exploration", "min_signal_strength", 0.12)
    if signal_strength is not None:
        if signal_strength < exploration_signal:
            return {"issue": "signal-below-exploration-threshold", "multiplier": 0.0, "tag": "weak-signal"}
        if signal_strength < probe_signal:
            return {"issue": "signal-exploration-strength", "multiplier": 0.10, "tag": "weak-signal"}
        if signal_strength < normal_signal:
            return {"issue": "signal-probe-strength", "multiplier": 0.35, "tag": "weak-signal"}

    normal_trend = _signal_mode_value(config, "normal", "min_trend_strength", 0.002)
    probe_trend = _signal_mode_value(config, "probe", "min_trend_strength", 0.001)
    exploration_trend = _signal_mode_value(config, "exploration", "min_trend_strength", 0.0005)
    if trend_strength is not None:
        if trend_strength < exploration_trend:
            return {"issue": "trend-below-exploration-threshold", "multiplier": 0.0, "tag": "weak-trend"}
        if trend_strength < probe_trend:
            return {"issue": "trend-exploration-strength", "multiplier": 0.10, "tag": "weak-trend"}
        if trend_strength < normal_trend:
            return {"issue": "trend-probe-strength", "multiplier": 0.35, "tag": "weak-trend"}

    normal_adx = _signal_mode_value(config, "normal", "adx_min", 20.0)
    probe_adx = _signal_mode_value(config, "probe", "adx_min", 16.0)
    exploration_adx = _signal_mode_value(config, "exploration", "adx_min", 12.0)
    if adx is not None:
        if adx < exploration_adx:
            return {"issue": "adx-below-exploration-threshold", "multiplier": 0.0, "tag": "weak-adx"}
        if adx < probe_adx:
            return {"issue": "adx-exploration-strength", "multiplier": 0.10, "tag": "weak-adx"}
        if adx < normal_adx:
            return {"issue": "adx-probe-strength", "multiplier": 0.35, "tag": "weak-adx"}
    return None


def _side_policy_metadata(
    side: str,
    *,
    config: object | None,
    current_soft_issues: int,
) -> dict[str, object]:
    if side != SignalDirection.LONG.value:
        return {
            "normal_enabled": True,
            "probe_enabled": True,
            "exploration_enabled": True,
            "multiplier": 1.0,
            "reason": "short-side-default",
            "soft_issue": "",
            "hard_issue": "",
            "tags": [],
        }
    long_policy = getattr(getattr(config, "side_policy", None), "long", None)
    normal_enabled = bool(getattr(long_policy, "normal_enabled", False))
    probe_enabled = bool(getattr(long_policy, "probe_enabled", True))
    exploration_enabled = bool(getattr(long_policy, "exploration_enabled", True))
    multiplier = float(getattr(long_policy, "probe_risk_multiplier", 0.10))
    if normal_enabled:
        return {
            "normal_enabled": normal_enabled,
            "probe_enabled": probe_enabled,
            "exploration_enabled": exploration_enabled,
            "multiplier": 1.0,
            "reason": "long-normal-enabled",
            "soft_issue": "",
            "hard_issue": "",
            "tags": [],
        }
    if probe_enabled or exploration_enabled:
        return {
            "normal_enabled": normal_enabled,
            "probe_enabled": probe_enabled,
            "exploration_enabled": exploration_enabled,
            "multiplier": multiplier,
            "reason": "long-normal-disabled-relaxed-probe",
            "soft_issue": "long-normal-disabled",
            "hard_issue": "",
            "tags": ["long-probe"],
        }
    return {
        "normal_enabled": normal_enabled,
        "probe_enabled": probe_enabled,
        "exploration_enabled": exploration_enabled,
        "multiplier": 0.0,
        "reason": "long-side-disabled",
        "soft_issue": "",
        "hard_issue": "long-side-disabled",
        "tags": [],
    }


def _decision_from_issues(soft_issues: list[str], tags: list[str]) -> str:
    if not soft_issues:
        return PolicyDecisionType.NORMAL_TRADE.value
    exploration_markers = (
        "exploration",
        "range-chop",
        "medium-adverse",
        "wide-spread-bounded",
        "multiple-soft",
        "strong-5m",
    )
    if len(soft_issues) >= 2 or any(
        any(marker in issue for marker in exploration_markers) for issue in soft_issues
    ):
        return PolicyDecisionType.EXPLORATION_TRADE.value
    if "long-probe" in tags:
        return PolicyDecisionType.PROBE_TRADE.value
    return PolicyDecisionType.PROBE_TRADE.value


def _worst_bucket(buckets: list[str]) -> str:
    order = {
        "unknown": 0,
        "normal": 1,
        "probe": 2,
        "exploration": 3,
        "reject": 4,
    }
    return max(buckets, key=lambda bucket: order.get(bucket, 0))


def _clamped_product(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= max(0.0, min(1.0, float(value)))
    return max(0.0, min(1.0, result))


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _learning_mode_value(config: object | None, name: str, default: Any) -> Any:
    return getattr(getattr(config, "learning_mode", None), name, default)


def _weak_choppy_value(config: object | None, name: str, default: Any) -> Any:
    policy = getattr(getattr(config, "regime_policy", None), "weak_down_choppy", None)
    return getattr(policy, name, default)


def _weak_choppy_short_confirmation_mode(config: object | None, status: str) -> str:
    policy = getattr(getattr(config, "regime_policy", None), "weak_down_choppy", None)
    confirmation = getattr(policy, "short_confirmation", None)
    normalized = status.strip().lower()
    if normalized in {"aligned", "unavailable", ""}:
        return str(getattr(confirmation, "aligned_5m_mode", "probe"))
    if normalized == "neutral":
        return str(getattr(confirmation, "neutral_5m_mode", "probe"))
    if normalized.startswith("mild_"):
        return str(getattr(confirmation, "mild_rebound_5m_mode", "exploration_or_wait_addon"))
    if normalized.startswith("strong_"):
        return str(getattr(confirmation, "strong_rebound_5m_mode", "wait_pullback_only"))
    if normalized.startswith("extreme_"):
        return str(getattr(confirmation, "extreme_adverse_5m_mode", "shadow_or_reject"))
    return "probe"


def _at_least_min_trade_multiplier(config: object | None, value: float) -> float:
    min_value = float(_learning_mode_value(config, "min_effective_risk_multiplier_to_trade", 0.05))
    return round(max(0.0, min(1.0, max(float(value), min_value))), 6)


def _ml_policy_value(config: object | None, name: str, default: float) -> float:
    return float(getattr(getattr(config, "ml_learning_policy", None), name, default))


def _confirmation_value(config: object | None, name: str, default: Any) -> Any:
    return getattr(getattr(config, "confirmation_5m", None), name, default)


def _mode_risk_multiplier(config: object | None, mode: str, default: float) -> float:
    learning_risk = getattr(config, "learning_risk", None)
    mode_config = getattr(learning_risk, mode, None)
    return float(getattr(mode_config, "risk_multiplier", default))


def _signal_mode_value(config: object | None, mode: str, key: str, default: float) -> float:
    learning_signals = getattr(config, "learning_signals", None)
    mode_config = getattr(learning_signals, mode, None)
    return float(getattr(mode_config, key, default))


def _symbol_health_multiplier(config: object | None, status: str, default: float) -> float:
    policy = getattr(config, "symbol_health_policy", None)
    status_policy = getattr(policy, status, None)
    return float(getattr(status_policy, "risk_multiplier", default))


def _symbol_health_reason(status: str, symbol: str, observe_only_configured: bool) -> str:
    if status == "weak":
        return f"symbol weak/probation observation for {symbol}"
    if status == "probation":
        return f"symbol probation for {symbol}"
    if status == "observe_only" and observe_only_configured:
        return f"symbol observe-only configured for {symbol}"
    return "normal symbol health"


def _strict_confirmation_blocks(confirmation: dict[str, object] | None) -> bool:
    return bool(confirmation and confirmation.get("available") and confirmation.get("against_direction"))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
