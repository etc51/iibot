from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from samosbor.autonomy.trade_review import build_trade_review_payload, trade_review_path
from samosbor.config import RiskSection, StrategySection
from samosbor.domain import PortfolioState, SignalDirection, TradeRecord


def _trade(
    *,
    symbol: str,
    entry_time: datetime,
    net_pnl: float,
    signal_strength: float,
    direction: SignalDirection = SignalDirection.LONG,
    entry_reason: str = "trend-up fast=101 slow=100",
    entry_metadata: dict[str, object] | None = None,
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        direction=direction,
        quantity_lots=10,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(minutes=30),
        entry_price=100.0,
        exit_price=95.0,
        gross_pnl=-50.0,
        net_pnl=net_pnl,
        reason="stop-loss",
        signal_strength=signal_strength,
        entry_reason=entry_reason,
        entry_context_score=0.0,
        entry_metadata=entry_metadata or {"trend_strength": 0.004},
        initial_stop_price=95.0,
        initial_take_profit=110.0,
    )


def test_trade_review_path_uses_state_stem():
    assert str(trade_review_path(Path("state/paper_state.json"))).replace(
        "\\",
        "/",
    ) == "state/paper_state_trade_review.json"


def test_trade_review_classifies_errors_and_recommends_patches():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(symbol="SBER", entry_time=opened, net_pnl=-50.0, signal_strength=0.41),
        _trade(symbol="SBER", entry_time=opened + timedelta(hours=1), net_pnl=-60.0, signal_strength=0.44),
    ]
    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=-110.0),
        trades,
        [],
        strategy=StrategySection(min_signal_strength=0.4, allowed_entry_hours=[10, 11, 12]),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    assert payload["reviewed_trades"] == 2
    assert "commit_hash" in payload
    assert payload["summary"]["mistake_trades"] == 2
    assert payload["mistake_breakdown"]["stop-loss"] == 2
    assert payload["mistake_breakdown"]["weak-signal-loss"] == 2
    assert payload["config_patch_candidates"]["strategy"]["min_signal_strength"] > 0.4
    assert all(
        item["action"] != "block-weak-symbols"
        for item in payload["recommendations"]
    )
    assert any(
        item["action"] == "observe-weak-symbols"
        and item["symbols"] == ["SBER"]
        for item in payload["recommendations"]
    )
    assert payload["config_patch_candidates"]["risk"]["max_risk_per_trade"] == 0.008


def test_trade_review_uses_order_book_microstructure_tags():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=-50.0,
            signal_strength=0.7,
            entry_metadata={
                "trend_strength": 0.004,
                "microstructure": {
                    "available": True,
                    "spread_bps": 15.0,
                    "entry_liquidity_cover": 1.2,
                    "side_imbalance": -0.5,
                },
            },
        )
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=-50.0),
        trades,
        [],
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    review = payload["reviews"][0]
    assert review["entry_microstructure"]["available"] is True
    assert review["microstructure_quality"] == "wide-spread"
    assert "wide-spread-entry" in review["mistake_tags"]
    assert "thin-book-entry" in review["mistake_tags"]
    assert "adverse-book-imbalance" in review["mistake_tags"]
    assert payload["breakdowns"]["microstructure_quality"][0]["group"] == "wide-spread"


def test_trade_review_reports_selloff_capture_and_underallocation():
    opened = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened + timedelta(minutes=2),
            net_pnl=120.0,
            signal_strength=0.5,
            direction=SignalDirection.SHORT,
            entry_metadata={
                "market_regime": {"regime": "market_selloff_impulse", "confidence": 0.82},
                "entry_mode": "market_breakdown_short",
                "regime_policy": {
                    "entry_mode": "market_breakdown_short",
                    "actual_policy_decision": "normal_trade",
                    "strict_policy_decision": "allow",
                    "risk_multiplier": 0.6,
                },
            },
        )
    ]
    events = [
        {
            "timestamp": opened.isoformat(),
            "action": "market_selloff_impulse_detected",
            "regime": "market_selloff_impulse",
        },
        {
            "timestamp": (opened + timedelta(minutes=1)).isoformat(),
            "action": "selloff_short_candidate",
            "symbol": "SBER",
            "direction": "short",
            "regime": "market_selloff_impulse",
            "entry_mode": "market_breakdown_short",
            "actual_policy_decision": "normal_trade",
            "approved": True,
        },
        {
            "timestamp": (opened + timedelta(minutes=1)).isoformat(),
            "action": "selloff_short_opened",
            "symbol": "SBER",
            "direction": "short",
            "regime": "market_selloff_impulse",
            "entry_mode": "market_breakdown_short",
            "actual_policy_decision": "normal_trade",
            "approved": True,
        },
        {
            "timestamp": (opened + timedelta(minutes=1)).isoformat(),
            "action": "selloff_short_rejected",
            "symbol": "GAZP",
            "direction": "short",
            "regime": "market_selloff_impulse",
            "entry_mode": "wait",
            "actual_policy_decision": "wait_pullback",
            "approved": False,
            "reason": "entry deferred for pullback short",
        },
        {
            "timestamp": (opened + timedelta(minutes=3)).isoformat(),
            "action": "selloff_underallocated",
            "regime": "market_selloff_impulse",
            "gross_exposure_pct": 0.12,
            "selloff_target_gross_exposure": 1.0,
            "budget_used_pct": 0.12,
            "unused_budget_reason": "confirmation blocked",
            "candidates_count": 2,
            "approved_count": 1,
            "rejected_count": 1,
            "wait_count": 1,
            "shadow_count": 0,
            "selloff_budget_blockers": {"confirmation blocked": 1},
        },
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=120.0),
        trades,
        events,
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    assert payload["selloff_capture_review"]["selloff_windows"] == 1
    assert payload["selloff_capture_review"]["candidates_count"] == 1
    assert payload["selloff_capture_review"]["trades_opened"] == 1
    assert payload["selloff_capture_review"]["budget_used_pct"] == 0.12
    assert payload["underallocation_review"]["selloff_underallocated_count"] == 1
    assert payload["underallocation_review"]["reasons_for_unused_budget"]["confirmation blocked"] == 1
    assert payload["long_during_selloff_review"]["long_signals_during_selloff"] == 0


def test_trade_review_short_only_section_and_legacy_long_learning_marker():
    opened = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=80.0,
            signal_strength=0.7,
            direction=SignalDirection.SHORT,
            entry_metadata={
                "market_regime": {"regime": "clean_downtrend", "confidence": 0.8},
                "short_only": {
                    "enabled": True,
                    "expected_net_edge_rub": 120.0,
                    "edge_bucket": "medium_positive",
                },
            },
        )
    ]
    events = [
        {"timestamp": opened.isoformat(), "action": "short_only_cycle_start"},
        {
            "timestamp": opened.isoformat(),
            "action": "long_signal_ignored_short_only",
            "symbol": "GAZP",
        },
        {
            "timestamp": opened.isoformat(),
            "action": "short_only_short_candidate",
            "symbol": "SBER",
            "edge_gate_passed": True,
            "expected_net_edge_rub": 120.0,
        },
        {
            "timestamp": opened.isoformat(),
            "action": "signal",
            "symbol": "SBER",
            "direction": "short",
            "approved": True,
            "metadata": {"short_only": {"enabled": True, "hard_reasons": []}},
        },
        {
            "timestamp": opened.isoformat(),
            "action": "short_only_budget_allocation",
            "budget_target_gross_rub": 100000.0,
            "budget_used_gross_rub": 80000.0,
            "metadata": {
                "short_only_budget": {
                    "budget_target_gross_rub": 100000.0,
                    "budget_used_gross_rub": 80000.0,
                }
            },
        },
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=80.0),
        trades,
        events,
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    assert payload["short_only_review"]["short_only_enabled"] is True
    assert payload["short_only_review"]["long_signals_ignored"] == 1
    assert payload["short_only_review"]["positive_ev_short_candidates"] == 1
    assert payload["short_only_review"]["shorts_opened"] == 1
    assert payload["short_only_review"]["pnl_by_edge_bucket"]["medium_positive"]["net_pnl_rub"] == 80.0
    assert payload["long_learning_review"]["active"] is False
    assert payload["long_learning_review"]["legacy_only"] is True


def test_trade_review_breaks_down_regime_policy_and_pending_rebound():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=-50.0,
            signal_strength=0.7,
            direction=SignalDirection.SHORT,
            entry_metadata={
                "market_regime": {"regime": "weak_down_choppy", "confidence": 0.8},
                "entry_mode": "pullback_short",
                "symbol_health": "probation",
                "regime_policy": {
                    "allow_trade": True,
                    "entry_mode": "pullback_short",
                    "risk_multiplier": 0.5,
                    "symbol_health": "probation",
                    "reasons": ["weak-down-choppy-pullback-short"],
                },
                "pending_entry": {
                    "state": "WAIT_PULLBACK_SHORT",
                    "outcome": "triggered",
                },
            },
        )
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=-50.0),
        trades,
        [],
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    review = payload["reviews"][0]
    assert review["market_regime"] == "weak_down_choppy"
    assert review["entry_mode"] == "pullback_short"
    assert review["symbol_health"] == "probation"
    assert review["rebound_outcome"] == "triggered"
    assert "failed-rebound-trigger-loss" in review["mistake_tags"]
    assert "symbol-probation-loss" in review["mistake_tags"]
    assert payload["breakdowns"]["market_regime"][0]["group"] == "weak_down_choppy"
    assert payload["breakdowns"]["entry_mode"][0]["group"] == "pullback_short"
    assert payload["breakdowns"]["symbol_health"][0]["group"] == "probation"
    assert payload["breakdowns"]["rebound_outcome"][0]["group"] == "triggered"
    assert any(item["action"] == "tighten-pullback-short-trigger" for item in payload["recommendations"])


def test_trade_review_reports_policy_decisions_and_shadow_outcomes():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=25.0,
            signal_strength=0.7,
            direction=SignalDirection.SHORT,
            entry_metadata={
                "market_regime": {"regime": "weak_down_choppy", "confidence": 0.8},
                "entry_mode": "weak_choppy_direct_probe_short",
                "regime_policy": {
                    "allow_trade": True,
                    "entry_mode": "weak_choppy_direct_probe_short",
                    "decision_type": "probe_trade",
                    "actual_policy_decision": "probe_trade",
                    "strict_policy_decision": "wait",
                    "would_strict_policy_trade": False,
                    "relaxed_only_trade": True,
                    "effective_risk_multiplier": 0.12,
                    "soft_issues": ["weak-down-choppy-trend-short-probe"],
                },
            },
        ),
        _trade(
            symbol="GAZP",
            entry_time=opened + timedelta(hours=1),
            net_pnl=-15.0,
            signal_strength=0.6,
            direction=SignalDirection.SHORT,
            entry_metadata={
                "shadow_trade_id": "shadow|GAZP|short|2025-01-01T11:00:00+00:00",
                "shadow_type": "policy_rejected",
                "regime_policy": {
                    "allow_trade": False,
                    "decision_type": "hard_reject",
                    "actual_policy_decision": "hard_reject",
                    "strict_policy_decision": "reject",
                    "would_strict_policy_trade": False,
                    "relaxed_only_trade": False,
                    "hard_issues": ["adverse-book-below-hard-limit"],
                },
            },
        ),
    ]
    events = [
        {
            "timestamp": opened.isoformat(),
            "action": "signal",
            "symbol": "SBER",
            "approved": True,
            "actual_policy_decision": "probe_trade",
            "strict_policy_decision": "wait",
            "relaxed_only_trade": True,
            "metadata": {
                "market_regime": {"regime": "weak_down_choppy"},
                "entry_mode": "weak_choppy_direct_probe_short",
                "regime_policy": {
                    "entry_mode": "weak_choppy_direct_probe_short",
                    "actual_policy_decision": "probe_trade",
                    "strict_policy_decision": "wait",
                    "relaxed_only_trade": True,
                },
            },
        },
        {
            "timestamp": (opened + timedelta(hours=1)).isoformat(),
            "action": "signal",
            "symbol": "GAZP",
            "approved": False,
            "actual_policy_decision": "hard_reject",
            "strict_policy_decision": "reject",
        },
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=10.0),
        trades,
        events,
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    reviews = {review["symbol"]: review for review in payload["reviews"]}
    assert reviews["SBER"]["trade_source"] == "executed"
    assert reviews["SBER"]["actual_policy_decision"] == "probe_trade"
    assert reviews["SBER"]["strict_policy_decision"] == "wait"
    assert reviews["SBER"]["relaxed_only_trade"] is True
    assert reviews["GAZP"]["trade_source"] == "policy_rejected_shadow"
    assert payload["policy_signal_distribution"]["relaxed_only_signals"] == 1
    assert payload["policy_outcomes"]["resolved_shadow_trades"] == 1
    assert payload["policy_outcomes"]["relaxed_only_trades"] == 1
    assert payload["weak_choppy_direct_probe_review"]["weak_down_choppy_probe_now_count"] == 1
    assert payload["weak_choppy_direct_probe_review"]["strict_wait_relaxed_probe_count"] == 1
    assert payload["strict_vs_relaxed"]["strict_wait_but_relaxed_traded_count"] == 1
    assert {row["group"] for row in payload["breakdowns"]["actual_policy_decision"]} == {
        "probe_trade",
        "hard_reject",
    }


def test_trade_review_reports_long_learning_section_and_old_trades():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=12.0,
            signal_strength=0.5,
            direction=SignalDirection.LONG,
            entry_metadata={
                "market_regime": {"regime": "clean_uptrend", "confidence": 0.9},
                "entry_mode": "clean_uptrend_direct_long",
                "regime_policy": {
                    "entry_mode": "clean_uptrend_direct_long",
                    "actual_policy_decision": "normal_trade",
                    "strict_policy_decision": "allow",
                    "long_context": {"regime": "clean_uptrend", "long_mode": "normal"},
                },
            },
        ),
        _trade(
            symbol="OLD",
            entry_time=opened + timedelta(hours=1),
            net_pnl=-5.0,
            signal_strength=0.4,
            direction=SignalDirection.LONG,
            entry_metadata={"trend_strength": 0.004},
        ),
    ]
    events = [
        {
            "timestamp": opened.isoformat(),
            "action": "signal",
            "symbol": "SBER",
            "direction": "long",
            "approved": True,
            "metadata": {
                "market_regime": {"regime": "clean_uptrend"},
                "entry_mode": "clean_uptrend_direct_long",
                "regime_policy": {"actual_policy_decision": "normal_trade"},
            },
        }
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=7.0),
        trades,
        events,
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    review = payload["long_learning_review"]
    assert review["long_signals_total"] == 1
    assert review["long_normal_count"] == 1
    assert review["long_pnl_total"]["trades"] == 2
    assert review["clean_uptrend_long_pnl"]["net_pnl_rub"] == 12.0


def test_trade_review_reports_learning_cap_metrics():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=-25.0,
            signal_strength=0.7,
            direction=SignalDirection.SHORT,
            entry_metadata={
                "learning_caps": {
                    "mode": "probe",
                    "decision_type": "probe_trade",
                    "probe_count_today": 40,
                    "exploration_count_today": 0,
                    "same_symbol_count_today": 3,
                    "same_entry_mode_count_today": 1,
                    "same_regime_count_today": 1,
                    "daily_cap_hit": True,
                    "same_symbol_cap_hit": True,
                    "same_entry_mode_cap_hit": False,
                    "same_regime_cap_hit": False,
                    "cap_behavior_applied": "shadow_only",
                    "oversampling_tags": [
                        "probe_daily_cap_soft_warning",
                        "same_symbol_learning_cap_hit",
                    ],
                    "original_quantity_lots": 4,
                    "adjusted_quantity_lots": 0,
                    "size_multiplier": 1.0,
                }
            },
        )
    ]
    events = [
        {
            "timestamp": opened.isoformat(),
            "action": "learning_cap_reduce_size",
            "symbol": "GAZP",
            "metadata": {
                "learning_caps": {
                    "mode": "exploration",
                    "decision_type": "exploration_trade",
                    "probe_count_today": 0,
                    "exploration_count_today": 40,
                    "daily_cap_hit": True,
                    "same_symbol_cap_hit": False,
                    "same_entry_mode_cap_hit": False,
                    "same_regime_cap_hit": True,
                    "cap_behavior_applied": "reduce_size",
                    "oversampling_tags": ["same_regime_learning_cap_hit"],
                }
            },
        }
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=-25.0),
        trades,
        events,
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    review_caps = payload["reviews"][0]["learning_caps"]
    metrics = payload["learning_caps"]
    assert review_caps["available"] is True
    assert metrics["available"] is True
    assert metrics["probe_trades_per_day"] == 40
    assert metrics["exploration_trades_per_day"] == 40
    assert metrics["probe_daily_cap_hits"] == 1
    assert metrics["exploration_daily_cap_hits"] == 1
    assert metrics["same_symbol_cap_hits"] == 1
    assert metrics["same_regime_cap_hits"] == 1
    assert metrics["reduce_size_events"] == 1
    assert metrics["same_symbol_cap_pnl"]["trades"] == 1


def test_trade_review_reports_selloff_learning_cap_metrics():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    events = [
        {
            "timestamp": opened.isoformat(),
            "action": "selloff_learning_cap_shadow_only",
            "symbol": "SBER",
            "metadata": {
                "selloff_learning_caps": {
                    "selloff_positions_count": 9,
                    "new_selloff_shorts_this_cycle": 8,
                    "same_symbol_selloff_cap_hit": True,
                    "same_entry_mode_selloff_cap_hit": False,
                    "same_regime_selloff_cap_hit": True,
                }
            },
        }
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=0.0),
        [],
        events,
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    metrics = payload["selloff_learning_caps"]
    assert metrics["available"] is True
    assert metrics["selloff_positions_used"] == 9
    assert metrics["selloff_new_shorts_per_cycle_max_observed"] == 8
    assert metrics["selloff_same_symbol_cap_hits"] == 1
    assert metrics["selloff_same_regime_cap_hits"] == 1
    assert metrics["shadow_only_events"] == 1


def test_trade_review_exposes_post_close_analysis():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=-50.0,
            signal_strength=0.7,
            entry_metadata={
                "post_close_analysis": {
                    "available": True,
                    "stage": "post_close",
                    "outcome": "error",
                    "is_error": True,
                    "ml_verdict": "ml_warned_loss",
                    "summary": "error via stop-loss, -1.00R; ml_warned_loss",
                }
            },
        )
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=-50.0),
        trades,
        [],
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    analysis = payload["reviews"][0]["post_close_analysis"]
    assert analysis["available"] is True
    assert analysis["outcome"] == "error"
    assert analysis["ml_verdict"] == "ml_warned_loss"


def test_trade_review_does_not_raise_signal_threshold_for_already_filtered_loss():
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trades = [
        _trade(symbol="ROSN", entry_time=opened, net_pnl=-50.0, signal_strength=0.25),
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=-50.0),
        trades,
        [],
        strategy=StrategySection(min_signal_strength=0.3),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
    )

    assert "strategy" not in payload["config_patch_candidates"]


def test_trade_review_prefers_nearest_collector_microstructure(tmp_path: Path):
    opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    micro_dir = tmp_path / "runs" / "microstructure" / "20250101"
    micro_dir.mkdir(parents=True)
    (micro_dir / "SBER.jsonl").write_text(
        json.dumps(
            {
                "collected_at": "2025-01-01T10:00:12+00:00",
                "available": True,
                "symbol": "SBER",
                "depth_requested": 10,
                "depth_returned": 10,
                "timestamp": "2025-01-01T10:00:11+00:00",
                "best_bid": 99.9,
                "best_ask": 100.0,
                "mid_price": 99.95,
                "spread_bps": 10.005,
                "bid_depth_lots": 100.0,
                "ask_depth_lots": 50.0,
                "bid_depth_rub": 9990.0,
                "ask_depth_rub": 5000.0,
                "imbalance": 0.3333,
                "requested_lots": 0,
                "entry_liquidity_cover": 0.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    trades = [
        _trade(
            symbol="SBER",
            entry_time=opened,
            net_pnl=-50.0,
            signal_strength=0.7,
            entry_metadata={
                "microstructure": {
                    "available": True,
                    "timestamp": "2025-01-01T10:00:20+00:00",
                    "spread_bps": 1.0,
                    "entry_liquidity_cover": 10.0,
                    "side_imbalance": 0.0,
                }
            },
        )
    ]

    payload = build_trade_review_payload(
        PortfolioState(cash=100_000, realized_pnl=-50.0),
        trades,
        [],
        strategy=StrategySection(min_signal_strength=0.4),
        risk=RiskSection(max_risk_per_trade=0.01),
        timezone_name="Europe/Moscow",
        microstructure_dir=tmp_path / "runs" / "microstructure",
    )

    microstructure = payload["reviews"][0]["entry_microstructure"]
    assert microstructure["source"] == "collector"
    assert microstructure["collector_lag_seconds"] == 8.0
    assert microstructure["entry_event_microstructure_lag_seconds"] == 20.0
    assert microstructure["requested_lots"] == 10
    assert microstructure["entry_depth_lots"] == 50.0
    assert microstructure["entry_liquidity_cover"] == 5.0
