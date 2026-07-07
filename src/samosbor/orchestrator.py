from __future__ import annotations

import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from .analysis.indicators import atr
from .autonomy.entry_schedule import (
    build_entry_schedule_tuning_payload,
    write_entry_schedule_tuning,
)
from .autonomy.runner import runner_breakeven_stop, runner_extreme_price
from .autonomy.entry_confirmation import build_entry_confirmation_context
from .autonomy.entry_quality_tuning import (
    build_entry_quality_tuning_payload,
    write_entry_quality_tuning,
)
from .autonomy.entry_symbols import (
    build_entry_symbol_tuning_payload,
    write_entry_symbol_tuning,
)
from .autonomy.daily_review import (
    build_daily_review_payload,
    daily_review_path,
    save_daily_review,
    write_daily_review,
)
from .autonomy.effective_config import (
    align_effective_config_sources,
    base_strategy_values,
    build_effective_config_guardrail_payload,
    build_effective_strategy_overrides,
    default_effective_config_path,
    summarize_effective_config_sources,
    write_effective_config,
)
from .autonomy.ml_learning import (
    assess_signal_learning,
    build_entry_candle_context,
    build_setup_learning_tags,
    indicator_from_reason,
    learning_position_size_adjustment,
)
from .autonomy.market_regime import detect_market_regime
from .autonomy.pending_entries import (
    evaluate_pending_entries,
    pending_entry_expired_event,
    pending_entry_quantity_lots,
    pending_entry_signal,
    record_pending_pullback_short,
)
from .autonomy.regime_policy import PolicyDecisionType, resolve as resolve_regime_policy
from .autonomy.signal_feedback import (
    backfill_signal_feedback_for_symbol,
    build_trade_evidence,
    default_signal_horizon_bars,
    load_signal_feedback,
    record_rejected_shadow_signal,
    record_shadow_signal,
    resolve_pending_signals,
    save_signal_feedback,
    signal_feedback_path,
)
from .autonomy.exit_tuning import (
    build_exit_reason_breakdown,
    build_exit_tuning_payload,
    specialize_exit_tuning_research,
    write_exit_tuning,
)
from .autonomy.strategy_tuning import (
    adapt_strategy_tuning_research,
    build_strategy_tuning_payload,
    write_strategy_tuning,
)
from .autonomy.trade_review import (
    build_trade_review_payload,
    save_trade_review,
    trade_review_path,
    write_trade_review,
)
from .autonomy.universe_selection import (
    build_universe_selection_tuning_payload,
    write_universe_selection_tuning,
)
from .config import AppConfig
from .config import StrategySection
from .data.csv_provider import CSVMarketDataProvider
from .data.moex_data_pack import MoexDataPackProvider
from .data.parquet_directory import ParquetDirectoryProvider
from .data.tbank import TBankMarketDataProvider
from .domain import ExitReason, SignalDirection
from .execution.paper import LocalPaperBroker
from .execution.sandbox import TBankSandboxExecutor
from .reporting.metrics import compute_summary
from .reporting.paper_report import build_paper_report_payload, write_paper_report
from .reporting.research_writer import (
    write_monte_carlo_report,
    write_optimizer_report,
    write_walk_forward_report,
)
from .reporting.writer import write_backtest_report, write_json_payload, write_portfolio_snapshot
from .research.monte_carlo import MonteCarloSimulator
from .research.optimizer import ParameterOptimizer
from .research.targets import (
    effective_target_monthly_return_pct,
    effective_target_payload,
)
from .research.walk_forward import (
    WalkForwardValidator,
    _available_months,
    _group_candles_by_month,
    _normalized_monthly_return_pct,
    _slice_grouped_candles,
    _trim_backtest_result,
)
from .risk.manager import RiskManager
from .runtime_metadata import add_runtime_metadata
from .safety import assert_paper_only_mode
from .strategy.trend_following import TrendFollowingStrategy
from .backtest.engine import BacktestEngine

LOGGER = logging.getLogger(__name__)


class TradingOrchestrator:
    def __init__(self, config: AppConfig):
        self.config = config
        self._market_bundle_cache = None
        self._market_bundle_cache_depth = 0

    def _autotune_dir(self) -> Path:
        return self.config.autotune_dir()

    @contextmanager
    def _market_bundle_cache_scope(self):
        self._market_bundle_cache_depth += 1
        try:
            yield
        finally:
            self._market_bundle_cache_depth = max(0, self._market_bundle_cache_depth - 1)
            if self._market_bundle_cache_depth == 0:
                self._market_bundle_cache = None

    def _data_provider(self):
        if self.config.data.source == "csv":
            return CSVMarketDataProvider(self.config.resolve_path(self.config.data.csv_path))
        if self.config.data.source == "parquet-directory":
            return ParquetDirectoryProvider(
                self.config.resolve_path(self.config.data.parquet_dir_path),
                timeframe=self.config.data.timeframe,
                history_days=self.config.data.history_days,
            )
        if self.config.data.source == "moex-data-pack":
            return MoexDataPackProvider(
                self.config.resolve_path(self.config.data.local_data_pack_path),
                timeframe=self.config.data.timeframe,
                history_days=self.config.data.history_days,
            )
        if self.config.data.source == "tbank":
            return TBankMarketDataProvider(self.config)
        raise ValueError(f"Unsupported data source: {self.config.data.source}")

    def _strategy(self) -> TrendFollowingStrategy:
        strategy_config = self.config.strategy
        if self._relaxed_learning_enabled():
            exploration = self.config.learning_signals.exploration
            strategy_config = replace(
                strategy_config,
                min_signal_strength=exploration.min_signal_strength,
                min_trend_strength=exploration.min_trend_strength,
                adx_min=exploration.adx_min,
            )
        return TrendFollowingStrategy(
            strategy_config,
            timeframe=self.config.data.timeframe,
        )

    def _adaptation_strategy(self) -> TrendFollowingStrategy:
        # Diagnostic shadow evidence should not inherit runtime entry restrictions,
        # otherwise the bot cannot learn which blocked windows or symbols deserve reopening.
        return TrendFollowingStrategy(
            replace(
                self.config.strategy,
                min_signal_strength=0.0,
                allowed_entry_hours=[],
                allowed_entry_weekdays=[],
                allowed_symbols=[],
                blocked_symbols=[],
                blocked_long_symbols=[],
                blocked_short_symbols=[],
                forced_flat_hours=[],
                forced_flat_weekdays=[],
            ),
            timeframe=self.config.data.timeframe,
        )

    def _risk_manager(self) -> RiskManager:
        return RiskManager(self.config.risk)

    def _relaxed_learning_enabled(self) -> bool:
        return (
            bool(getattr(self.config.learning_mode, "enabled", False))
            and self.config.execution.mode.value == "local-paper"
        )

    def _short_only_enabled(self) -> bool:
        return (
            bool(getattr(getattr(self.config, "short_only", None), "enabled", False))
            and self.config.execution.mode.value == "local-paper"
        )

    def _run_short_only_cycle(self) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        provider = self._data_provider()
        instruments = provider.resolve_universe(self.config.data.instruments)
        history = provider.load_history(instruments)
        confirmation_history = self._load_entry_confirmation_history(provider, instruments, history)
        marks = {symbol: candles[-1].close for symbol, candles in history.items() if candles}

        state_path = self.config.resolve_path(self.config.execution.state_path)
        feedback_path = signal_feedback_path(state_path)
        broker = self._load_paper_broker()
        signal_feedback = load_signal_feedback(feedback_path)

        timestamp = datetime.now(timezone.utc)
        strategy = self._strategy()
        if hasattr(strategy, "prepare_market_context"):
            strategy.prepare_market_context(history)
        for instrument in instruments:
            strategy.prepare_history(instrument, history.get(instrument.symbol, []))
        risk_manager = self._risk_manager()
        cycle_events: list[dict[str, object]] = [
            {
                "timestamp": timestamp.isoformat(),
                "action": "short_only_cycle_start",
                "short_only_enabled": True,
                "disable_all_longs": bool(self.config.short_only.disable_all_longs),
                "allow_live_trading": self.config.execution.allow_live_trading,
            }
        ]
        market_regime = detect_market_regime(history)
        cycle_events.append(
            {
                "timestamp": timestamp.isoformat(),
                "action": "market_regime",
                **market_regime.as_event(),
            }
        )
        if market_regime.regime == "market_selloff_impulse":
            cycle_events.append(self._market_selloff_detected_event(market_regime, timestamp))

        resolve_pending_signals(signal_feedback, history)
        broker.mark_to_market(marks, timestamp)
        risk_manager.update_drawdown_state(broker.portfolio, marks)
        cycle_events.extend(
            self._short_only_manage_existing_positions(
                broker,
                risk_manager,
                strategy,
                history,
                marks,
                timestamp=timestamp,
            )
        )

        mode = self._short_only_mode_for_regime(market_regime.regime)
        if mode == "NO_TRADE":
            cycle_events.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "action": (
                        "range_chop_no_trade_short_only"
                        if market_regime.regime == "range_chop"
                        else "short_only_no_trade_regime"
                    ),
                    "regime": market_regime.regime,
                    "reason": "short-only no-trade regime",
                    "metadata": {"market_regime": market_regime.as_event()},
                }
            )
        else:
            candidates = self._short_only_collect_candidates(
                provider,
                strategy,
                risk_manager,
                broker,
                signal_feedback,
                instruments,
                history,
                confirmation_history,
                marks,
                market_regime=market_regime,
                timestamp=timestamp,
                cycle_events=cycle_events,
            )
            cycle_events.extend(
                self._short_only_allocate_and_open(
                    broker,
                    candidates,
                    marks,
                    market_regime=market_regime,
                    timestamp=timestamp,
                )
            )

        broker.mark_to_market(marks, timestamp)
        broker.events.extend(cycle_events)
        broker.save(state_path)
        save_signal_feedback(feedback_path, signal_feedback)
        trade_review = build_trade_review_payload(
            broker.portfolio,
            broker.trades,
            broker.events,
            strategy=self.config.strategy,
            risk=self.config.risk,
            timezone_name=self.config.app.timezone,
            lookback_trades=100,
            generated_at=timestamp,
            microstructure_dir=self.config.resolve_path(self.config.reporting.output_dir) / "microstructure",
        )
        latest_trade_review_path = trade_review_path(state_path)
        save_trade_review(latest_trade_review_path, trade_review)
        trade_review["latest_path"] = str(latest_trade_review_path)
        add_runtime_metadata(trade_review)
        signal_activity = _summarize_signal_activity(cycle_events)
        short_only_activity = _summarize_short_only_activity(cycle_events, broker.portfolio, marks)
        stamp = timestamp.strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "paper" / stamp
        trade_review["output_dir"] = str(output_dir)
        summary = {
            "timestamp": timestamp.isoformat(),
            "equity_rub": round(broker.portfolio.equity(marks), 2),
            "cash_rub": round(broker.portfolio.cash, 2),
            "gross_exposure_rub": round(broker.portfolio.gross_exposure(marks), 2),
            "open_positions": len(broker.portfolio.positions),
            "trading_halted": broker.portfolio.trading_halted,
            "short_only": short_only_activity,
            "trade_review": _trade_review_view(trade_review),
            **signal_activity,
        }
        add_runtime_metadata(summary)
        cycle_events_payload = {"events": cycle_events}
        add_runtime_metadata(cycle_events_payload)
        write_json_payload(output_dir / "cycle_summary.json", summary)
        write_json_payload(output_dir / "cycle_events.json", cycle_events_payload)
        write_trade_review(output_dir, trade_review)
        write_portfolio_snapshot(output_dir / "portfolio.json", broker.portfolio)
        return {"summary": summary, "output_dir": str(output_dir)}

    def _short_only_manage_existing_positions(
        self,
        broker,
        risk_manager: RiskManager,
        strategy,
        history: dict[str, list],
        marks: dict[str, float],
        *,
        timestamp: datetime,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for symbol, position in list(broker.portfolio.positions.items()):
            candles = history.get(symbol, [])
            latest = candles[-1] if candles else None
            mark = marks.get(symbol, position.current_price)
            event_timestamp = latest.timestamp if latest is not None else timestamp
            if position.direction == SignalDirection.LONG:
                if bool(self.config.short_only.flatten_existing_longs):
                    broker.close_position(
                        symbol,
                        price=mark,
                        timestamp=event_timestamp,
                        reason=ExitReason.SHORT_ONLY_POLICY_FLATTEN_LONG,
                    )
                    events.append(
                        {
                            "timestamp": event_timestamp.isoformat(),
                            "symbol": symbol,
                            "action": "long_position_flattened_short_only",
                            "direction": "long",
                            "reason": "short_only_policy_flatten_long",
                        }
                    )
                continue
            if latest is None:
                continue
            if latest.high >= position.stop_price:
                broker.close_position(
                    symbol,
                    price=position.stop_price,
                    timestamp=latest.timestamp,
                    reason=ExitReason.STOP_LOSS,
                )
                continue
            if latest.low <= position.take_profit and not position.runner_active:
                if self.config.short_only.exits.use_existing_runner and self._take_profit_activates_runner(
                    broker,
                    symbol,
                    position,
                    latest,
                ):
                    position = broker.portfolio.positions.get(symbol)
                else:
                    broker.close_position(
                        symbol,
                        price=position.take_profit,
                        timestamp=latest.timestamp,
                        reason=ExitReason.TAKE_PROFIT,
                    )
                    continue
            if position and position.runner_active and self.config.short_only.exits.use_existing_runner:
                self._update_runner_extreme(broker, symbol, position, latest)
                position = broker.portfolio.positions.get(symbol)
            if position is not None and strategy.should_force_flatten_at(latest.timestamp):
                broker.close_position(
                    symbol,
                    price=latest.close,
                    timestamp=latest.timestamp,
                    reason=ExitReason.SESSION_FLAT,
                )
                continue
            if position is not None and self.config.short_only.exits.use_existing_atr_stop:
                if position.runner_active:
                    new_stop = risk_manager.runner_trailing_stop_price(
                        position,
                        atr_value=atr(candles, self.config.strategy.atr_window),
                        strategy=self.config.strategy,
                    )
                else:
                    new_stop = risk_manager.trailing_stop_price(
                        position,
                        latest.close,
                        self.config.strategy,
                    )
                if new_stop is not None:
                    broker.update_position_protection(
                        symbol,
                        timestamp=latest.timestamp,
                        stop_price=new_stop,
                        reason="short-only-trailing-profit-protection",
                    )
        return events

    def _short_only_collect_candidates(
        self,
        provider,
        strategy,
        risk_manager: RiskManager,
        broker,
        signal_feedback: dict[str, list[dict[str, object]]],
        instruments: list,
        history: dict[str, list],
        confirmation_history: dict[str, list],
        marks: dict[str, float],
        *,
        market_regime,
        timestamp: datetime,
        cycle_events: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for instrument in instruments:
            candles = history.get(instrument.symbol, [])
            if not candles:
                continue
            latest = candles[-1]
            if instrument.symbol in broker.portfolio.positions:
                continue
            signal = strategy.generate_signal(instrument, candles)
            if signal is None:
                continue
            if signal.direction == SignalDirection.LONG:
                cycle_events.append(
                    {
                        "timestamp": latest.timestamp.isoformat(),
                        "symbol": instrument.symbol,
                        "action": "long_signal_ignored_short_only",
                        "direction": "long",
                        "strength": signal.strength,
                        "reason": "short_only disables active long trading",
                    }
                )
                continue
            if signal.direction != SignalDirection.SHORT:
                continue
            entry_block_reason = strategy.entry_block_reason_for_instrument(
                instrument,
                latest.timestamp,
                signal.direction,
            )
            if entry_block_reason is not None:
                cycle_events.append(
                    self._short_only_signal_event(
                        signal,
                        market_regime=market_regime,
                        timestamp=latest.timestamp,
                        approved=False,
                        reason=entry_block_reason,
                        quantity_lots=0,
                    )
                )
                continue

            decision = risk_manager.approve(broker.portfolio, signal, marks, broker.trades)
            signal_for_entry = self._signal_with_entry_candle_context(signal, candles)
            signal_for_entry = self._signal_with_entry_confirmation_context(
                signal_for_entry,
                confirmation_history.get(instrument.symbol, []),
                signal_timestamp=latest.timestamp,
            )
            signal_for_entry = self._signal_with_setup_learning_tags(
                signal_for_entry,
                broker.trades,
                timestamp=latest.timestamp,
            )
            quantity_lots = max(0, int(decision.quantity_lots))
            if decision.approved:
                signal_for_entry = self._signal_with_entry_microstructure(
                    provider,
                    signal_for_entry,
                    quantity_lots=quantity_lots,
                )
                signal_for_entry = self._signal_with_learning_assessment(
                    signal_for_entry,
                    signal_feedback,
                    timestamp=latest.timestamp,
                    quantity_lots=quantity_lots,
                )
            edge = self._short_only_edge_gate(
                signal_for_entry,
                market_regime=market_regime,
                quantity_lots=quantity_lots,
            )
            micro = self._short_only_microstructure_gate(signal_for_entry)
            confirmation = self._short_only_confirmation_gate(signal_for_entry, market_regime=market_regime)
            hard_reasons = []
            if not decision.approved:
                hard_reasons.append(decision.reason)
            if not edge["passed"]:
                hard_reasons.append(str(edge["reason"]))
            if micro["hard_reason"]:
                hard_reasons.append(str(micro["hard_reason"]))
            if confirmation["hard_reason"]:
                hard_reasons.append(str(confirmation["hard_reason"]))
            metadata = dict(signal_for_entry.metadata)
            multiplier = _bounded_multiplier(
                float(edge["multiplier"]) * float(micro["multiplier"]) * float(confirmation["multiplier"])
            )
            metadata["market_regime"] = market_regime.as_event()
            metadata["short_only"] = {
                "enabled": True,
                "mode": self._short_only_mode_for_regime(market_regime.regime),
                "expected_net_edge_rub": edge["expected_net_edge_rub"],
                "required_net_edge_rub": edge["required_net_edge_rub"],
                "edge_source": edge["source"],
                "edge_gate_passed": edge["passed"],
                "edge_gate_reason": edge["reason"],
                "edge_bucket": edge["edge_bucket"],
                "microstructure_multiplier": micro["multiplier"],
                "confirmation_multiplier": confirmation["multiplier"],
                "size_multiplier": multiplier,
                "hard_reasons": list(hard_reasons),
            }
            signal_for_entry = replace(signal_for_entry, metadata=metadata)
            cycle_events.append(
                {
                    "timestamp": latest.timestamp.isoformat(),
                    "symbol": instrument.symbol,
                    "action": "short_only_short_candidate",
                    "regime": market_regime.regime,
                    "direction": "short",
                    "strength": signal.strength,
                    "expected_net_edge_rub": edge["expected_net_edge_rub"],
                    "required_net_edge_rub": edge["required_net_edge_rub"],
                    "edge_source": edge["source"],
                    "edge_gate_passed": edge["passed"],
                    "edge_gate_reason": edge["reason"],
                    "hard_reasons": list(hard_reasons),
                    "metadata": {"short_only": metadata["short_only"]},
                }
            )
            if hard_reasons:
                cycle_events.append(
                    self._short_only_signal_event(
                        signal_for_entry,
                        market_regime=market_regime,
                        timestamp=latest.timestamp,
                        approved=False,
                        reason="; ".join(hard_reasons),
                        quantity_lots=0,
                        original_quantity_lots=quantity_lots,
                    )
                )
                continue
            candidates.append(
                {
                    "signal": signal_for_entry,
                    "timestamp": latest.timestamp,
                    "risk_quantity_lots": quantity_lots,
                    "size_multiplier": multiplier,
                    "expected_net_edge_rub": float(edge["expected_net_edge_rub"]),
                    "required_net_edge_rub": float(edge["required_net_edge_rub"]),
                    "edge_source": str(edge["source"]),
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["expected_net_edge_rub"]),
                float(item["signal"].strength),
            ),
            reverse=True,
        )
        return candidates

    def _short_only_allocate_and_open(
        self,
        broker,
        candidates: list[dict[str, object]],
        marks: dict[str, float],
        *,
        market_regime,
        timestamp: datetime,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        sizing = self._short_only_sizing_for_regime(market_regime.regime)
        equity = broker.portfolio.equity(marks)
        target_gross = max(0.0, equity * float(getattr(sizing, "target_gross_exposure", 0.0)))
        regime_max_gross = float(getattr(sizing, "max_gross_exposure", 0.0) or self.config.risk.max_gross_exposure)
        max_gross = equity * min(float(self.config.risk.max_gross_exposure), regime_max_gross)
        max_positions = min(
            int(self.config.risk.max_positions),
            int(getattr(sizing, "max_positions", 0) or self.config.risk.max_positions),
        )
        max_new = int(getattr(sizing, "max_new_shorts_per_cycle", 0) or len(candidates))
        opened = 0
        blocked: Counter[str] = Counter()
        for candidate in candidates:
            signal = candidate["signal"]
            if opened >= max_new:
                blocked["max_new_shorts_per_cycle"] += 1
                events.append(
                    self._short_only_signal_event(
                        signal,
                        market_regime=market_regime,
                        timestamp=candidate["timestamp"],
                        approved=False,
                        reason="short_only max new shorts per cycle reached",
                        quantity_lots=0,
                        original_quantity_lots=int(candidate["risk_quantity_lots"]),
                    )
                )
                continue
            if len(broker.portfolio.positions) >= max_positions:
                blocked["max_positions"] += 1
                events.append(
                    self._short_only_signal_event(
                        signal,
                        market_regime=market_regime,
                        timestamp=candidate["timestamp"],
                        approved=False,
                        reason="short_only max positions reached",
                        quantity_lots=0,
                        original_quantity_lots=int(candidate["risk_quantity_lots"]),
                    )
                )
                continue
            current_gross = broker.portfolio.gross_exposure(marks)
            if target_gross > 0 and current_gross >= target_gross:
                blocked["target_gross_exposure_reached"] += 1
                events.append(
                    self._short_only_signal_event(
                        signal,
                        market_regime=market_regime,
                        timestamp=candidate["timestamp"],
                        approved=False,
                        reason="short_only target gross exposure reached",
                        quantity_lots=0,
                        original_quantity_lots=int(candidate["risk_quantity_lots"]),
                    )
                )
                continue
            if current_gross >= max_gross:
                blocked["max_gross_exposure"] += 1
                events.append(
                    self._short_only_signal_event(
                        signal,
                        market_regime=market_regime,
                        timestamp=candidate["timestamp"],
                        approved=False,
                        reason="short_only max gross exposure reached",
                        quantity_lots=0,
                        original_quantity_lots=int(candidate["risk_quantity_lots"]),
                    )
                )
                continue
            quantity_lots = self._short_only_allocated_quantity(
                candidate,
                equity=equity,
                current_gross=current_gross,
                target_gross=target_gross,
                max_gross=max_gross,
                sizing=sizing,
            )
            if quantity_lots < 1:
                blocked["lot sizing failed"] += 1
                events.append(
                    self._short_only_signal_event(
                        signal,
                        market_regime=market_regime,
                        timestamp=candidate["timestamp"],
                        approved=False,
                        reason="short_only allocation produced < 1 lot",
                        quantity_lots=0,
                        original_quantity_lots=int(candidate["risk_quantity_lots"]),
                    )
                )
                continue
            metadata = dict(signal.metadata)
            short_only = dict(metadata.get("short_only", {}))
            short_only["allocated_quantity_lots"] = quantity_lots
            short_only["budget_target_gross_rub"] = round(target_gross, 2)
            short_only["budget_current_gross_rub"] = round(current_gross, 2)
            metadata["short_only"] = short_only
            signal = replace(signal, metadata=metadata)
            events.append(
                self._short_only_signal_event(
                    signal,
                    market_regime=market_regime,
                    timestamp=candidate["timestamp"],
                    approved=True,
                    reason="approved",
                    quantity_lots=quantity_lots,
                    original_quantity_lots=int(candidate["risk_quantity_lots"]),
                )
            )
            broker.open_position(signal, quantity_lots, candidate["timestamp"])
            opened += 1

        final_gross = broker.portfolio.gross_exposure(marks)
        budget_used = final_gross / target_gross if target_gross > 0 else 0.0
        diagnostics = {
            "regime": market_regime.regime,
            "target_gross_exposure": round(float(getattr(sizing, "target_gross_exposure", 0.0)), 6),
            "budget_target_gross_rub": round(target_gross, 2),
            "budget_used_gross_rub": round(final_gross, 2),
            "budget_used_pct": round(budget_used, 6),
            "positive_ev_candidates": len(candidates),
            "shorts_opened": opened,
            "blocked_reasons": dict(sorted(blocked.items())),
        }
        events.append(
            {
                "timestamp": timestamp.isoformat(),
                "action": "short_only_budget_allocation",
                **diagnostics,
                "metadata": {"short_only_budget": diagnostics},
            }
        )
        if candidates and target_gross > 0 and final_gross < target_gross * 0.50:
            events.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "action": "short_only_underallocated",
                    "severity": "warning",
                    **diagnostics,
                    "reason": "gross exposure after allocation below 50% of target",
                    "metadata": {"short_only_budget": diagnostics},
                }
            )
        return events

    def _short_only_allocated_quantity(
        self,
        candidate: dict[str, object],
        *,
        equity: float,
        current_gross: float,
        target_gross: float,
        max_gross: float,
        sizing,
    ) -> int:
        signal = candidate["signal"]
        notional_per_lot = signal.entry_price * signal.instrument.lot_size
        if notional_per_lot <= 0:
            return 0
        risk_quantity = self._quantity_after_multiplier(
            int(candidate["risk_quantity_lots"]),
            float(candidate["size_multiplier"]),
        )
        if risk_quantity < 1:
            return 0
        per_symbol_target = equity * float(getattr(sizing, "per_symbol_exposure_target", 0.0) or 1.0)
        per_symbol_max = equity * float(getattr(sizing, "per_symbol_exposure_max", 0.0) or 1.0)
        remaining_target = max(0.0, target_gross - current_gross) if target_gross > 0 else max_gross - current_gross
        remaining_global = max(0.0, max_gross - current_gross)
        budget = max(0.0, min(per_symbol_max, per_symbol_target, remaining_target, remaining_global))
        budget_lots = int(budget // notional_per_lot)
        return max(0, min(risk_quantity, budget_lots))

    def _short_only_edge_gate(
        self,
        signal,
        *,
        market_regime,
        quantity_lots: int,
    ) -> dict[str, object]:
        cfg = self.config.short_only
        edge_cfg = cfg.edge
        ml = signal.metadata.get("ml_learning", {})
        expected: float | None = None
        required_ml = 0.0
        source = "none"
        multiplier = 1.0
        reason = ""
        if isinstance(ml, dict) and bool(ml.get("available", False)):
            expected = self._object_float(ml.get("expected_pnl_position_rub"))
            required_ml = self._object_float(ml.get("required_net_edge_rub")) or 0.0
            source = "ml"
        elif (
            str(getattr(cfg.ml, "missing_model_action", "")) == "price_action_fallback"
            and bool(edge_cfg.allow_ml_fallback_when_model_missing)
            and self._price_action_fallback_allowed(signal, market_regime)
        ):
            expected = self._price_action_expected_edge(signal, quantity_lots)
            source = "price_action_fallback"
        buffer_rub = signal.entry_price * signal.instrument.lot_size * max(1, int(quantity_lots)) * (
            float(edge_cfg.required_edge_buffer_bps) / 10000.0
        )
        required = max(float(edge_cfg.min_expected_net_edge_rub), required_ml + buffer_rub)
        if expected is None:
            return {
                "passed": False,
                "expected_net_edge_rub": 0.0,
                "required_net_edge_rub": round(required, 2),
                "source": source,
                "multiplier": 0.0,
                "edge_bucket": "missing",
                "reason": "edge model missing and price action fallback not allowed",
            }
        expected_per_lot = expected / max(1, int(quantity_lots))
        passed = expected > required and expected_per_lot >= float(edge_cfg.min_expected_net_edge_per_lot_rub)
        if not passed:
            multiplier = 0.0
            reason = "expected net edge is not positive after required buffer"
        elif expected < required * 2:
            multiplier = float(cfg.ml.weak_positive_edge_multiplier)
            reason = "weak positive edge"
        else:
            multiplier = float(cfg.ml.positive_edge_multiplier)
            reason = "positive edge"
        return {
            "passed": passed,
            "expected_net_edge_rub": round(expected, 2),
            "required_net_edge_rub": round(required, 2),
            "source": source,
            "multiplier": _bounded_multiplier(multiplier),
            "edge_bucket": self._short_only_edge_bucket(expected - required),
            "reason": reason,
        }

    def _price_action_fallback_allowed(self, signal, market_regime) -> bool:
        if market_regime.regime not in {"market_selloff_impulse", "clean_downtrend"}:
            return False
        if market_regime.regime == "market_selloff_impulse" and not self.config.short_only.edge.allow_price_action_edge_in_selloff:
            return False
        candle = signal.metadata.get("entry_candle", {})
        if not isinstance(candle, dict):
            candle = {}
        ret1 = self._object_float(candle.get("ret1")) or 0.0
        ret4 = self._object_float(candle.get("ret4")) or 0.0
        return signal.direction == SignalDirection.SHORT and (ret1 < 0.0 or ret4 < 0.0 or signal.strength >= 0.30)

    def _price_action_expected_edge(self, signal, quantity_lots: int) -> float:
        quantity_units = max(1, int(quantity_lots)) * signal.instrument.lot_size
        planned_risk = abs(signal.stop_price - signal.entry_price) * quantity_units
        planned_reward = abs(signal.entry_price - signal.take_profit) * quantity_units
        win_probability = max(0.05, min(0.95, float(signal.strength)))
        return planned_reward * win_probability - planned_risk * (1.0 - win_probability)

    def _short_only_microstructure_gate(self, signal) -> dict[str, object]:
        cfg = self.config.short_only.microstructure
        micro = signal.metadata.get("microstructure", {})
        if not isinstance(micro, dict) or not micro.get("available", False):
            return {"multiplier": 1.0, "hard_reason": "", "soft_reasons": ["microstructure-unavailable"]}
        spread = self._object_float(micro.get("spread_bps"))
        cover = self._object_float(micro.get("entry_liquidity_cover"))
        imbalance = self._object_float(micro.get("side_imbalance", micro.get("imbalance")))
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        if spread is not None:
            if spread > float(cfg.hard_max_spread_bps):
                hard_reasons.append("short_only extreme spread")
            elif spread > float(cfg.soft_spread_bps):
                soft_reasons.append("short_only soft spread")
        if cover is not None:
            if cover < float(cfg.hard_min_liquidity_cover):
                hard_reasons.append("short_only extreme liquidity")
            elif cover < float(cfg.soft_liquidity_cover):
                soft_reasons.append("short_only soft liquidity")
        if imbalance is not None:
            if imbalance < float(cfg.hard_min_book_imbalance):
                hard_reasons.append("short_only extreme imbalance")
            elif imbalance < float(cfg.soft_book_imbalance):
                soft_reasons.append("short_only soft imbalance")
        multiplier = 1.0
        if soft_reasons:
            multiplier = float(cfg.bad_but_allowed_multiplier if len(soft_reasons) > 1 else cfg.soft_multiplier)
        return {
            "multiplier": _bounded_multiplier(multiplier),
            "hard_reason": "; ".join(hard_reasons),
            "soft_reasons": soft_reasons,
        }

    def _short_only_confirmation_gate(self, signal, *, market_regime) -> dict[str, object]:
        cfg = self.config.short_only.confirmation
        confirmation = signal.metadata.get("entry_confirmation", {})
        if not isinstance(confirmation, dict) or not confirmation.get("available", False):
            return {"multiplier": 1.0, "hard_reason": "", "status": "unavailable"}
        bars = int(confirmation.get("bars", 0) or 0)
        min_bars = int(cfg.selloff_min_5m_bars if market_regime.regime == "market_selloff_impulse" else cfg.normal_min_5m_bars)
        if bars < min_bars:
            return {"multiplier": float(cfg.neutral_5m_multiplier), "hard_reason": "", "status": "insufficient_bars"}
        ret_window = self._object_float(confirmation.get("ret_window")) or 0.0
        adverse_ret = max(0.0, ret_window)
        mild = float(getattr(self.config.confirmation_5m, "mild_adverse_ret", 0.0025))
        strong = float(getattr(self.config.confirmation_5m, "strong_adverse_ret", 0.005))
        extreme = float(getattr(self.config.confirmation_5m, "extreme_adverse_ret", 0.012))
        if adverse_ret <= 0.0:
            return {"multiplier": 1.0, "hard_reason": "", "status": "aligned"}
        if adverse_ret < mild:
            return {"multiplier": float(cfg.neutral_5m_multiplier), "hard_reason": "", "status": "neutral"}
        if adverse_ret < strong:
            return {"multiplier": float(cfg.mild_rebound_multiplier), "hard_reason": "", "status": "mild_rebound"}
        if adverse_ret < extreme:
            hard = "short_only strong rebound against short" if cfg.strong_rebound_action == "no_trade" else ""
            return {"multiplier": 0.0 if hard else float(cfg.mild_rebound_multiplier), "hard_reason": hard, "status": "strong_rebound"}
        hard = "short_only extreme adverse confirmation" if cfg.extreme_adverse_action == "no_trade" else ""
        return {"multiplier": 0.0 if hard else float(cfg.mild_rebound_multiplier), "hard_reason": hard, "status": "extreme_adverse"}

    def _short_only_mode_for_regime(self, regime: str) -> str:
        allowed = set(self.config.short_only.allow_shorts_only_in_regimes)
        if regime not in allowed:
            if regime == "mixed" and bool(self.config.short_only.allow_mixed_regime_shorts):
                return "SHORT_SELECTIVE"
            return "NO_TRADE"
        if regime == "market_selloff_impulse":
            return "SHORT_AGGRESSIVE"
        if regime == "clean_downtrend":
            return "SHORT_AGGRESSIVE"
        if regime == "weak_down_choppy":
            return "SHORT_SELECTIVE"
        return "NO_TRADE"

    def _short_only_sizing_for_regime(self, regime: str):
        sizing = self.config.short_only.sizing
        return getattr(sizing, regime, sizing.range_chop)

    def _short_only_signal_event(
        self,
        signal,
        *,
        market_regime,
        timestamp: datetime,
        approved: bool,
        reason: str,
        quantity_lots: int,
        original_quantity_lots: int = 0,
    ) -> dict[str, object]:
        metadata = dict(signal.metadata)
        metadata.setdefault("market_regime", market_regime.as_event())
        metadata.setdefault("short_only", {"enabled": True})
        return {
            "timestamp": timestamp.isoformat(),
            "symbol": signal.instrument.symbol,
            "action": "signal",
            "event_type": "short_only_policy_decision",
            "approved": approved,
            "reason": reason,
            "direction": signal.direction.value,
            "strength": signal.strength,
            "quantity_lots": int(quantity_lots),
            "original_quantity_lots": int(original_quantity_lots or quantity_lots),
            "metadata": metadata,
        }

    @staticmethod
    def _short_only_edge_bucket(edge_after_required: float) -> str:
        if edge_after_required <= 0:
            return "non_positive"
        if edge_after_required < 50:
            return "small_positive"
        if edge_after_required < 250:
            return "medium_positive"
        return "large_positive"

    @staticmethod
    def _object_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_paper_broker(self) -> LocalPaperBroker:
        state_path = self.config.resolve_path(self.config.execution.state_path)
        return LocalPaperBroker.load(
            state_path,
            initial_cash=self.config.backtest.initial_cash,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )

    def _load_trade_evidence(self, broker: LocalPaperBroker) -> dict[str, object]:
        feedback = load_signal_feedback(
            signal_feedback_path(self.config.resolve_path(self.config.execution.state_path))
        )
        evidence = build_trade_evidence(broker.trades, feedback)
        return {
            "trades": evidence["trades"],
            "evidence_source": evidence["evidence_source"],
            "evidence_counts": evidence["counts"],
        }

    def _load_market_bundle(self):
        if self._market_bundle_cache_depth > 0 and self._market_bundle_cache is not None:
            return self._market_bundle_cache

        provider = self._data_provider()
        instruments = provider.resolve_universe(self.config.data.instruments)
        candles_by_symbol = provider.load_history(instruments)
        instruments_by_symbol = {instrument.symbol: instrument for instrument in instruments}
        bundle = (provider, instruments, candles_by_symbol, instruments_by_symbol)
        if self._market_bundle_cache_depth > 0:
            self._market_bundle_cache = bundle
        return bundle

    def _run_backtest_bundle(
        self,
        candles_by_symbol: dict[str, list],
        instruments_by_symbol: dict[str, object],
        *,
        strategy_config: StrategySection | None = None,
    ):
        engine = BacktestEngine(
            strategy=TrendFollowingStrategy(
                strategy_config or self.config.strategy,
                timeframe=self.config.data.timeframe,
            ),
            risk_manager=self._risk_manager(),
            backtest=self.config.backtest,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        result = engine.run_with_instruments(candles_by_symbol, instruments_by_symbol)
        summary = compute_summary(result, timeframe=self.config.data.timeframe)
        return result, summary

    def list_accounts(self) -> list[dict[str, str]]:
        provider = TBankMarketDataProvider(self.config)
        return provider.list_accounts()

    def init_sandbox(self, amount_rub: float) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        executor = TBankSandboxExecutor(self.config)
        account_id = executor.ensure_account()
        executor.fund_account(amount_rub)
        return {"account_id": account_id, "funded_rub": amount_rub}

    def run_backtest(self) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        _, _, candles_by_symbol, instruments_by_symbol = self._load_market_bundle()
        result, summary = self._run_backtest_bundle(candles_by_symbol, instruments_by_symbol)
        add_runtime_metadata(summary)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "backtests" / stamp
        write_backtest_report(output_dir, result, summary)
        LOGGER.info("Backtest report written to %s", output_dir)
        return {"summary": summary, "output_dir": str(output_dir)}

    def optimize_strategy(self) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        _, _, candles_by_symbol, instruments_by_symbol = self._load_market_bundle()
        optimizer = ParameterOptimizer(
            base_strategy=self.config.strategy,
            risk=self.config.risk,
            backtest=self.config.backtest,
            research=self.config.research,
            timeframe=self.config.data.timeframe,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        payload = optimizer.run(candles_by_symbol, instruments_by_symbol)
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "optimizer" / stamp
        write_optimizer_report(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def run_monte_carlo(self) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        _, _, candles_by_symbol, instruments_by_symbol = self._load_market_bundle()
        result, summary = self._run_backtest_bundle(candles_by_symbol, instruments_by_symbol)
        simulator = MonteCarloSimulator(
            iterations=self.config.research.monte_carlo_iterations,
            horizon_months=self.config.research.monte_carlo_horizon_months,
            target_monthly_return_pct=effective_target_monthly_return_pct(
                self.config.research,
                self.config.backtest,
            ),
            seed=self.config.research.random_seed,
        )
        payload = {
            "backtest_summary": summary,
            "target": effective_target_payload(self.config.research, self.config.backtest),
            "monte_carlo": simulator.run(result),
        }
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "monte-carlo" / stamp
        write_monte_carlo_report(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def run_walk_forward(self, *, adaptive_history: bool = False) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        _, instruments, candles_by_symbol, instruments_by_symbol = self._load_market_bundle()
        research = self.config.research
        research_window: dict[str, object] | None = None
        if adaptive_history:
            grouped = _group_candles_by_month(candles_by_symbol)
            available_months = _available_months(grouped)
            tuned_research, research_window = adapt_strategy_tuning_research(
                self.config.research,
                available_months=len(available_months),
                fixed_subset_size=len(instruments),
            )
            if tuned_research is None:
                payload = {
                    "config": {},
                    "summary": {},
                    "available_months": available_months,
                    "skipped_folds": 0,
                    "folds": [],
                    "research_window": research_window,
                    "reason": research_window["reason"],
                }
                add_runtime_metadata(payload)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "walk-forward" / stamp
                write_walk_forward_report(output_dir, payload)
                payload["output_dir"] = str(output_dir)
                return payload
            research = tuned_research
        validator = WalkForwardValidator(
            base_strategy=self.config.strategy,
            risk=self.config.risk,
            backtest=self.config.backtest,
            research=research,
            timeframe=self.config.data.timeframe,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        payload = validator.run(candles_by_symbol, instruments_by_symbol)
        if research_window is not None:
            payload["research_window"] = research_window
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "walk-forward" / stamp
        write_walk_forward_report(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def tune_strategy(
        self,
        *,
        min_monthly_improvement_pct: float = 0.05,
        max_extra_drawdown_pct: float = 1.0,
        min_positive_fold_probability_pct: float = 55.0,
        walk_forward_payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        _, instruments, candles_by_symbol, instruments_by_symbol = self._load_market_bundle()
        grouped = _group_candles_by_month(candles_by_symbol)
        available_months = _available_months(grouped)
        tuned_research, research_window = adapt_strategy_tuning_research(
            self.config.research,
            available_months=len(available_months),
            fixed_subset_size=len(instruments),
        )
        if tuned_research is None:
            payload = {
                "target": effective_target_payload(self.config.research, self.config.backtest),
                "research_window": research_window,
                "changed": False,
                "reason": research_window["reason"],
            }
            add_runtime_metadata(payload)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            output_dir = self._autotune_dir() / "strategy" / stamp
            write_json_payload(output_dir / "strategy_tuning.json", payload)
            payload["output_dir"] = str(output_dir)
            return payload

        walk_forward = walk_forward_payload
        if walk_forward is None:
            validator = WalkForwardValidator(
                base_strategy=self.config.strategy,
                risk=self.config.risk,
                backtest=self.config.backtest,
                research=tuned_research,
                timeframe=self.config.data.timeframe,
                slippage_bps=self.config.execution.slippage_bps,
                commission_bps=self.config.execution.commission_bps,
            )
            walk_forward = validator.run(candles_by_symbol, instruments_by_symbol)
        research_window = walk_forward.get("research_window", research_window)
        latest_fold = walk_forward["folds"][-1]

        optimizer = ParameterOptimizer(
            base_strategy=self.config.strategy,
            risk=self.config.risk,
            backtest=self.config.backtest,
            research=tuned_research,
            timeframe=self.config.data.timeframe,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        candidate_strategy = optimizer.strategy_from_candidate_payload(latest_fold["best_candidate"])
        baseline_summary = self._evaluate_strategy_test_window(
            strategy_config=self.config.strategy,
            grouped=grouped,
            instruments_by_symbol=instruments_by_symbol,
            symbols=latest_fold["best_candidate"]["symbols"],
            train_months=latest_fold["train_months"],
            test_months=latest_fold["test_months"],
        )
        payload = build_strategy_tuning_payload(
            current_strategy=self.config.strategy,
            candidate_strategy=candidate_strategy,
            baseline_latest_test_summary=baseline_summary,
            candidate_latest_test_summary=latest_fold["test_summary"],
            walk_forward_summary=walk_forward["summary"],
            walk_forward_config=walk_forward["config"],
            backtest=self.config.backtest,
            research=tuned_research,
            research_window=research_window,
            min_monthly_improvement_pct=min_monthly_improvement_pct,
            max_extra_drawdown_pct=max_extra_drawdown_pct,
            min_positive_fold_probability_pct=min_positive_fold_probability_pct,
        )
        payload["latest_fold"] = latest_fold
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "strategy" / stamp
        write_strategy_tuning(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def tune_exits(
        self,
        *,
        min_monthly_improvement_pct: float = 0.03,
        max_extra_drawdown_pct: float = 1.0,
        min_positive_fold_probability_pct: float = 55.0,
    ) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        _, instruments, candles_by_symbol, instruments_by_symbol = self._load_market_bundle()
        grouped = _group_candles_by_month(candles_by_symbol)
        available_months = _available_months(grouped)
        tuned_research, research_window = adapt_strategy_tuning_research(
            self.config.research,
            available_months=len(available_months),
            fixed_subset_size=len(instruments),
        )
        if tuned_research is None:
            payload = {
                "target": effective_target_payload(self.config.research, self.config.backtest),
                "research_window": research_window,
                "changed": False,
                "reason": research_window["reason"],
            }
            add_runtime_metadata(payload)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            output_dir = self._autotune_dir() / "exits" / stamp
            write_json_payload(output_dir / "exit_tuning.json", payload)
            payload["output_dir"] = str(output_dir)
            return payload

        exit_research = specialize_exit_tuning_research(tuned_research, self.config.strategy)
        validator = WalkForwardValidator(
            base_strategy=self.config.strategy,
            risk=self.config.risk,
            backtest=self.config.backtest,
            research=exit_research,
            timeframe=self.config.data.timeframe,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        walk_forward = validator.run(candles_by_symbol, instruments_by_symbol)
        latest_fold = walk_forward["folds"][-1]

        optimizer = ParameterOptimizer(
            base_strategy=self.config.strategy,
            risk=self.config.risk,
            backtest=self.config.backtest,
            research=exit_research,
            timeframe=self.config.data.timeframe,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        candidate_strategy = optimizer.strategy_from_candidate_payload(latest_fold["best_candidate"])
        baseline_result, baseline_summary = self._evaluate_strategy_test_window_bundle(
            strategy_config=self.config.strategy,
            grouped=grouped,
            instruments_by_symbol=instruments_by_symbol,
            symbols=latest_fold["best_candidate"]["symbols"],
            train_months=latest_fold["train_months"],
            test_months=latest_fold["test_months"],
        )
        candidate_result, candidate_summary = self._evaluate_strategy_test_window_bundle(
            strategy_config=candidate_strategy,
            grouped=grouped,
            instruments_by_symbol=instruments_by_symbol,
            symbols=latest_fold["best_candidate"]["symbols"],
            train_months=latest_fold["train_months"],
            test_months=latest_fold["test_months"],
        )
        payload = build_exit_tuning_payload(
            current_strategy=self.config.strategy,
            candidate_strategy=candidate_strategy,
            baseline_latest_test_summary=baseline_summary,
            candidate_latest_test_summary=candidate_summary,
            baseline_exit_breakdown=build_exit_reason_breakdown(baseline_result.trades),
            candidate_exit_breakdown=build_exit_reason_breakdown(candidate_result.trades),
            walk_forward_summary=walk_forward["summary"],
            walk_forward_config=walk_forward["config"],
            backtest=self.config.backtest,
            research=exit_research,
            research_window=research_window,
            min_monthly_improvement_pct=min_monthly_improvement_pct,
            max_extra_drawdown_pct=max_extra_drawdown_pct,
            min_positive_fold_probability_pct=min_positive_fold_probability_pct,
        )
        payload["latest_fold"] = latest_fold
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "exits" / stamp
        write_exit_tuning(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def run_paper_report(
        self,
        *,
        days: int = 1,
        report_date: str | None = None,
        timezone_name: str | None = None,
    ) -> dict[str, object]:
        broker = self._load_paper_broker()
        parsed_date = date.fromisoformat(report_date) if report_date else None
        payload = build_paper_report_payload(
            broker.portfolio,
            broker.trades,
            timezone_name=timezone_name or self.config.app.timezone,
            report_date=parsed_date,
            days=days,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "paper-reports" / stamp
        write_paper_report(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def run_trade_review(self, *, lookback_trades: int = 100) -> dict[str, object]:
        broker = self._load_paper_broker()
        state_path = self.config.resolve_path(self.config.execution.state_path)
        payload = build_trade_review_payload(
            broker.portfolio,
            broker.trades,
            broker.events,
            strategy=self.config.strategy,
            risk=self.config.risk,
            timezone_name=self.config.app.timezone,
            lookback_trades=lookback_trades,
            microstructure_dir=self.config.resolve_path(self.config.reporting.output_dir) / "microstructure",
        )
        latest_path = trade_review_path(state_path)
        save_trade_review(latest_path, payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "trade-review" / stamp
        write_trade_review(output_dir, payload)
        payload["latest_path"] = str(latest_path)
        payload["output_dir"] = str(output_dir)
        return payload

    def run_daily_review(
        self,
        *,
        report_date: str | None = None,
        days: int = 1,
        max_signal_rows: int = 250,
        max_ml_candidates: int = 60,
        max_holding_bars: int = 32,
    ) -> dict[str, object]:
        broker = self._load_paper_broker()
        state_path = self.config.resolve_path(self.config.execution.state_path)
        provider = self._data_provider()
        instruments = provider.resolve_universe(self.config.data.instruments)
        candles_by_symbol = provider.load_history(instruments)
        instruments_by_symbol = {instrument.symbol: instrument for instrument in instruments}
        confirmation_history = self._load_entry_confirmation_history(provider, instruments, candles_by_symbol)
        feedback = load_signal_feedback(signal_feedback_path(state_path))
        parsed_date = date.fromisoformat(report_date) if report_date else None
        payload = build_daily_review_payload(
            self.config,
            broker.portfolio,
            broker.trades,
            candles_by_symbol=candles_by_symbol,
            instruments_by_symbol=instruments_by_symbol,
            confirmation_history_by_symbol=confirmation_history,
            feedback_payload=feedback,
            report_date=parsed_date,
            days=days,
            max_signal_rows=max_signal_rows,
            max_ml_candidates=max_ml_candidates,
            max_holding_bars=max_holding_bars,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "daily-review" / stamp
        latest_path = daily_review_path(state_path)
        payload["latest_path"] = str(latest_path)
        payload["output_dir"] = str(output_dir)
        save_daily_review(latest_path, payload)
        write_daily_review(output_dir, payload)
        return payload

    def tune_entry_schedule(
        self,
        *,
        lookback_days: int = 45,
        report_date: str | None = None,
        timezone_name: str | None = None,
        min_trades_per_hour: int = 3,
        max_hours_to_add: int = 2,
        max_hours_to_remove: int = 2,
    ) -> dict[str, object]:
        broker = self._load_paper_broker()
        trade_evidence = self._load_trade_evidence(broker)
        parsed_date = date.fromisoformat(report_date) if report_date else None
        payload = build_entry_schedule_tuning_payload(
            broker.portfolio,
            trade_evidence["trades"],
            timezone_name=timezone_name or self.config.app.timezone,
            current_hours=self.config.strategy.allowed_entry_hours,
            evidence_source=str(trade_evidence["evidence_source"]),
            report_date=parsed_date,
            lookback_days=lookback_days,
            min_trades_per_hour=min_trades_per_hour,
            max_hours_to_add=max_hours_to_add,
            max_hours_to_remove=max_hours_to_remove,
        )
        payload["evidence_counts"] = trade_evidence["evidence_counts"]
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "entry-schedule" / stamp
        write_entry_schedule_tuning(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def tune_entry_quality(
        self,
        *,
        lookback_trades: int = 40,
        min_trades: int = 8,
        min_trade_retention_ratio: float = 0.5,
        min_expectancy_improvement_rub: float = 0.0,
        bucket_step: float = 0.05,
    ) -> dict[str, object]:
        broker = self._load_paper_broker()
        trade_evidence = self._load_trade_evidence(broker)
        payload = build_entry_quality_tuning_payload(
            trades=trade_evidence["trades"],
            evidence_source=str(trade_evidence["evidence_source"]),
            current_min_signal_strength=self.config.strategy.min_signal_strength,
            backtest=self.config.backtest,
            research=self.config.research,
            lookback_trades=lookback_trades,
            min_trades=min_trades,
            min_trade_retention_ratio=min_trade_retention_ratio,
            min_expectancy_improvement_rub=min_expectancy_improvement_rub,
            bucket_step=bucket_step,
        )
        payload["evidence_counts"] = trade_evidence["evidence_counts"]
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "entry-quality" / stamp
        write_entry_quality_tuning(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def tune_entry_symbols(
        self,
        *,
        lookback_days: int = 45,
        report_date: str | None = None,
        timezone_name: str | None = None,
        min_trades_per_symbol: int = 8,
        min_trades_per_direction_symbol: int = 8,
        max_symbols_to_block: int = 0,
        max_total_blocked_symbols: int = 4,
        max_long_symbols_to_block: int = 0,
        max_short_symbols_to_block: int = 0,
        max_total_blocked_long_symbols: int = 4,
        max_total_blocked_short_symbols: int = 4,
    ) -> dict[str, object]:
        broker = self._load_paper_broker()
        trade_evidence = self._load_trade_evidence(broker)
        parsed_date = date.fromisoformat(report_date) if report_date else None
        payload = build_entry_symbol_tuning_payload(
            broker.portfolio,
            trade_evidence["trades"],
            timezone_name=timezone_name or self.config.app.timezone,
            current_blocked_symbols=self.config.strategy.blocked_symbols,
            current_blocked_long_symbols=self.config.strategy.blocked_long_symbols,
            current_blocked_short_symbols=self.config.strategy.blocked_short_symbols,
            evidence_source=str(trade_evidence["evidence_source"]),
            report_date=parsed_date,
            lookback_days=lookback_days,
            min_trades_per_symbol=min_trades_per_symbol,
            min_trades_per_direction_symbol=min_trades_per_direction_symbol,
            max_symbols_to_block=max_symbols_to_block,
            max_total_blocked_symbols=max_total_blocked_symbols,
            max_long_symbols_to_block=max_long_symbols_to_block,
            max_short_symbols_to_block=max_short_symbols_to_block,
            max_total_blocked_long_symbols=max_total_blocked_long_symbols,
            max_total_blocked_short_symbols=max_total_blocked_short_symbols,
            runtime_symbols=[instrument.symbol for instrument in self.config.data.instruments],
        )
        payload["evidence_counts"] = trade_evidence["evidence_counts"]
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "entry-symbols" / stamp
        write_entry_symbol_tuning(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def tune_runtime_universe(
        self,
        *,
        optimizer_payload: dict[str, object] | None = None,
        walk_forward_payload: dict[str, object] | None = None,
        max_allowed_symbols: int | None = None,
        min_walk_forward_positive_probability_pct: float = 55.0,
        min_latest_fold_monthly_return_pct: float = 0.0,
        min_walk_forward_folds: int = 3,
        min_latest_fold_trades: int = 4,
        require_optimizer_overlap: bool = True,
    ) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        with self._market_bundle_cache_scope():
            optimizer = optimizer_payload or self.optimize_strategy()
            walk_forward = walk_forward_payload or self.run_walk_forward(adaptive_history=True)
        allowed_cap = max_allowed_symbols or max(
            1,
            min(self.config.risk.max_positions, self.config.research.subset_max_size),
        )
        payload = build_universe_selection_tuning_payload(
            configured_symbols=[instrument.symbol for instrument in self.config.data.instruments],
            current_allowed_symbols=self.config.strategy.allowed_symbols,
            optimizer_payload=optimizer,
            walk_forward_payload=walk_forward,
            max_allowed_symbols=allowed_cap,
            min_walk_forward_positive_probability_pct=min_walk_forward_positive_probability_pct,
            min_latest_fold_monthly_return_pct=min_latest_fold_monthly_return_pct,
            min_walk_forward_folds=min_walk_forward_folds,
            min_latest_fold_trades=min_latest_fold_trades,
            require_optimizer_overlap=require_optimizer_overlap,
        )
        add_runtime_metadata(payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "universe-selection" / stamp
        write_universe_selection_tuning(output_dir, payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def bootstrap_entry_feedback(
        self,
        *,
        replace_existing: bool = False,
        max_signals_per_symbol: int = 0,
    ) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        _, instruments, candles_by_symbol, _ = self._load_market_bundle()
        state_path = self.config.resolve_path(self.config.execution.state_path)
        feedback_path = signal_feedback_path(state_path)
        payload = {"pending": [], "resolved": []} if replace_existing else load_signal_feedback(feedback_path)
        shadow_strategy = self._adaptation_strategy()
        horizon_bars = default_signal_horizon_bars(self.config.data.timeframe)
        counts: dict[str, int] = {}
        generated_total = 0

        for instrument in instruments:
            candles = candles_by_symbol.get(instrument.symbol, [])
            if len(candles) <= self.config.backtest.warmup_bars:
                counts[instrument.symbol] = 0
                continue
            generated = backfill_signal_feedback_for_symbol(
                payload,
                instrument=instrument,
                candles=candles,
                strategy=shadow_strategy,
                warmup_bars=self.config.backtest.warmup_bars,
                horizon_bars=horizon_bars,
                max_signals=max_signals_per_symbol,
                slippage_bps=self.config.execution.slippage_bps,
                commission_bps=self.config.execution.commission_bps,
            )
            counts[instrument.symbol] = generated
            generated_total += generated

        save_signal_feedback(feedback_path, payload)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "entry-feedback-bootstrap" / stamp
        result = {
            "feedback_path": str(feedback_path),
            "replace_existing": replace_existing,
            "max_signals_per_symbol": max_signals_per_symbol,
            "generated_total": generated_total,
            "generated_by_symbol": counts,
            "pending_signals": len(payload.get("pending", [])),
            "resolved_signals": len(payload.get("resolved", [])),
        }
        add_runtime_metadata(result)
        write_json_payload(output_dir / "bootstrap_summary.json", result)
        result["output_dir"] = str(output_dir)
        return result

    def refresh_effective_config(
        self,
        *,
        source_config_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        source_path = Path(source_config_path).resolve()
        target_path = Path(output_path).resolve() if output_path else default_effective_config_path(source_path)
        autotune_dir = self._autotune_dir()
        sources = align_effective_config_sources(
            self.config,
            summarize_effective_config_sources(autotune_dir),
        )
        broker = self._load_paper_broker()
        rollback_report = build_paper_report_payload(
            broker.portfolio,
            broker.trades,
            timezone_name=self.config.app.timezone,
            days=3,
        )
        guardrail = build_effective_config_guardrail_payload(
            base_values=base_strategy_values(self.config),
            source_summaries=sources,
            paper_report=rollback_report,
            guardrail_days=3,
            min_recent_trades=6,
        )
        if guardrail["rollback_to_base"]:
            overrides = base_strategy_values(self.config)
        else:
            overrides = build_effective_strategy_overrides(
                self.config,
                source_summaries=sources,
            )
        write_effective_config(
            source_path,
            target_path,
            strategy_overrides=overrides,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "effective-config" / stamp
        result = {
            "source_config_path": str(source_path),
            "effective_config_path": str(target_path),
            "paper_only_mode": self.config.execution.mode.value,
            "allow_live_trading": self.config.execution.allow_live_trading,
            "applied_strategy_overrides": overrides,
            "sources": sources,
            "rollback_guardrail": guardrail,
        }
        add_runtime_metadata(result)
        write_json_payload(output_dir / "effective_config.json", result)
        result["output_dir"] = str(output_dir)
        return result

    def run_nightly_autonomy(
        self,
        *,
        active_config_path: str | Path,
        base_config_path: str | Path,
        effective_output_path: str | Path,
    ) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        active_config = Path(active_config_path).resolve()
        base_config = Path(base_config_path).resolve()
        effective_config = Path(effective_output_path).resolve()

        with self._market_bundle_cache_scope():
            paper_report = self.run_paper_report(days=1)
            trade_review = self.run_trade_review(lookback_trades=100)
            feedback_bootstrap = self.bootstrap_entry_feedback()
            entry_schedule = self.tune_entry_schedule(
                lookback_days=45,
                min_trades_per_hour=3,
            )
            entry_symbols = self.tune_entry_symbols(
                lookback_days=45,
                min_trades_per_symbol=8,
                max_symbols_to_block=0,
                max_total_blocked_symbols=4,
            )
            entry_quality = self.tune_entry_quality(
                lookback_trades=40,
                min_trades=8,
            )
            optimizer = self.optimize_strategy()
            walk_forward = self.run_walk_forward(adaptive_history=True)
            runtime_universe = self.tune_runtime_universe(
                optimizer_payload=optimizer,
                walk_forward_payload=walk_forward,
            )
            monte_carlo = self.run_monte_carlo()
            strategy_tuning = self.tune_strategy(walk_forward_payload=walk_forward)
            exit_tuning = self.tune_exits()
            effective_config_result = self.refresh_effective_config(
                source_config_path=base_config,
                output_path=effective_config,
            )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = self._autotune_dir() / "nightly-autonomy" / stamp
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_config_path": str(active_config),
            "base_config_path": str(base_config),
            "effective_output_path": str(effective_config),
            "steps_executed": [
                "paper-report",
                "trade-review",
                "bootstrap-entry-feedback",
                "tune-entry-hours",
                "tune-entry-symbols",
                "tune-entry-quality",
                "optimize",
                "walk-forward",
                "tune-universe",
                "monte-carlo",
                "tune-strategy",
                "tune-exits",
                "refresh-effective-config",
            ],
            "analysis": {
                "paper_report": _paper_report_view(paper_report),
                "trade_review": _trade_review_view(trade_review),
            },
            "restrictions": {
                "entry_schedule": _entry_schedule_view(entry_schedule),
                "entry_symbols": _entry_symbols_view(entry_symbols),
                "entry_quality": _entry_quality_view(entry_quality),
                "signal_feedback_bootstrap": _feedback_bootstrap_view(feedback_bootstrap),
            },
            "research": {
                "optimizer": _optimizer_view(optimizer),
                "walk_forward": _walk_forward_view(walk_forward),
                "monte_carlo": _monte_carlo_view(monte_carlo),
            },
            "tuning": {
                "strategy": _strategy_tuning_view(strategy_tuning),
                "exits": _exit_tuning_view(exit_tuning),
            },
            "runtime": {
                "universe_selection": _runtime_universe_view(runtime_universe),
                "effective_config": _effective_config_view(effective_config_result),
            },
        }
        add_runtime_metadata(result)
        write_json_payload(output_dir / "nightly_autonomy.json", result)
        (output_dir / "summary.md").write_text(_render_nightly_autonomy_markdown(result), encoding="utf-8")
        result["output_dir"] = str(output_dir)
        return result

    def run_paper_cycle(self) -> dict[str, object]:
        assert_paper_only_mode(
            self.config.execution.mode,
            allow_live_trading=self.config.execution.allow_live_trading,
            live_flag=False,
        )
        if self._short_only_enabled():
            return self._run_short_only_cycle()
        provider = self._data_provider()
        instruments = provider.resolve_universe(self.config.data.instruments)
        history = provider.load_history(instruments)
        confirmation_history = self._load_entry_confirmation_history(provider, instruments, history)
        marks = {symbol: candles[-1].close for symbol, candles in history.items() if candles}

        state_path = self.config.resolve_path(self.config.execution.state_path)
        feedback_path = signal_feedback_path(state_path)
        broker = self._load_paper_broker()
        signal_feedback = load_signal_feedback(feedback_path)

        timestamp = datetime.now(timezone.utc)
        strategy = self._strategy()
        shadow_strategy = self._adaptation_strategy()
        if hasattr(strategy, "prepare_market_context"):
            strategy.prepare_market_context(history)
        for instrument in instruments:
            strategy.prepare_history(instrument, history.get(instrument.symbol, []))
            shadow_strategy.prepare_history(instrument, history.get(instrument.symbol, []))
        risk_manager = self._risk_manager()
        cycle_events: list[dict[str, object]] = []
        market_regime = detect_market_regime(history)
        cycle_events.append(
            {
                "timestamp": timestamp.isoformat(),
                "action": "market_regime",
                **market_regime.as_event(),
            }
        )
        if market_regime.regime == "market_selloff_impulse":
            cycle_events.append(self._market_selloff_detected_event(market_regime, timestamp))
        resolve_pending_signals(signal_feedback, history)
        pending_entry_results = evaluate_pending_entries(signal_feedback, history)
        triggered_pending_entries_by_symbol: dict[str, list[dict[str, object]]] = {}
        for pending_result in pending_entry_results:
            if pending_result.get("status") == "expired":
                cycle_events.append(pending_entry_expired_event(pending_result))
                continue
            item = dict(pending_result.get("item", {}))
            symbol = str(item.get("symbol", ""))
            if symbol:
                triggered_pending_entries_by_symbol.setdefault(symbol, []).append(pending_result)

        broker.mark_to_market(marks, timestamp)
        risk_manager.update_drawdown_state(broker.portfolio, marks)

        for instrument in instruments:
            candles = history.get(instrument.symbol, [])
            if not candles:
                continue
            latest = candles[-1]
            position = broker.portfolio.positions.get(instrument.symbol)
            if position is not None:
                if position.direction.value == "long":
                    if latest.low <= position.stop_price:
                        broker.close_position(
                            instrument.symbol,
                            price=position.stop_price,
                            timestamp=latest.timestamp,
                            reason=ExitReason.STOP_LOSS,
                        )
                        position = None
                    elif latest.high >= position.take_profit and not position.runner_active:
                        if self._take_profit_activates_runner(
                            broker,
                            instrument.symbol,
                            position,
                            latest,
                        ):
                            position = broker.portfolio.positions.get(instrument.symbol)
                        else:
                            broker.close_position(
                                instrument.symbol,
                                price=position.take_profit,
                                timestamp=latest.timestamp,
                                reason=ExitReason.TAKE_PROFIT,
                            )
                            position = None
                    elif position.runner_active:
                        self._update_runner_extreme(broker, instrument.symbol, position, latest)
                        position = broker.portfolio.positions.get(instrument.symbol)
                else:
                    if latest.high >= position.stop_price:
                        broker.close_position(
                            instrument.symbol,
                            price=position.stop_price,
                            timestamp=latest.timestamp,
                            reason=ExitReason.STOP_LOSS,
                        )
                        position = None
                    elif latest.low <= position.take_profit and not position.runner_active:
                        if self._take_profit_activates_runner(
                            broker,
                            instrument.symbol,
                            position,
                            latest,
                        ):
                            position = broker.portfolio.positions.get(instrument.symbol)
                        else:
                            broker.close_position(
                                instrument.symbol,
                                price=position.take_profit,
                                timestamp=latest.timestamp,
                                reason=ExitReason.TAKE_PROFIT,
                            )
                            position = None
                    elif position.runner_active:
                        self._update_runner_extreme(broker, instrument.symbol, position, latest)
                        position = broker.portfolio.positions.get(instrument.symbol)
                if position is not None and strategy.should_force_flatten_at(latest.timestamp):
                    broker.close_position(
                        instrument.symbol,
                        price=latest.close,
                        timestamp=latest.timestamp,
                        reason=ExitReason.SESSION_FLAT,
                    )
                    position = None
                if position is not None:
                    if position.runner_active:
                        new_stop = risk_manager.runner_trailing_stop_price(
                            position,
                            atr_value=atr(candles, self.config.strategy.atr_window),
                            strategy=self.config.strategy,
                        )
                    else:
                        new_stop = risk_manager.trailing_stop_price(
                            position,
                            latest.close,
                            self.config.strategy,
                        )
                    if new_stop is not None:
                        reason = (
                            "runner-trailing-profit-protection"
                            if position.runner_active
                            else "trailing-profit-protection"
                        )
                        broker.update_position_protection(
                            instrument.symbol,
                            timestamp=latest.timestamp,
                            stop_price=new_stop,
                            reason=reason,
                        )

            opened_pending_entry = False
            if position is None:
                for pending_result in triggered_pending_entries_by_symbol.get(instrument.symbol, []):
                    pending_events = self._process_pending_entry_trigger(
                        provider,
                        broker,
                        risk_manager,
                        signal_feedback,
                        pending_result,
                        marks,
                        market_regime=market_regime,
                        timestamp=latest.timestamp,
                    )
                    cycle_events.extend(pending_events)
                    position = broker.portfolio.positions.get(instrument.symbol)
                    if position is not None:
                        opened_pending_entry = True
                        break
            if opened_pending_entry:
                continue

            signal = strategy.generate_signal(instrument, candles)
            shadow_signal = shadow_strategy.generate_signal(instrument, candles)
            if signal is not None:
                signal = self._signal_with_entry_candle_context(signal, candles)
                signal = self._signal_with_entry_confirmation_context(
                    signal,
                    confirmation_history.get(instrument.symbol, []),
                    signal_timestamp=latest.timestamp,
                )
                signal = self._signal_with_setup_learning_tags(
                    signal,
                    broker.trades,
                    timestamp=latest.timestamp,
                )
            if shadow_signal is not None:
                shadow_signal = self._signal_with_entry_candle_context(shadow_signal, candles)
                shadow_signal = self._signal_with_entry_confirmation_context(
                    shadow_signal,
                    confirmation_history.get(instrument.symbol, []),
                    signal_timestamp=latest.timestamp,
                )
            if signal is None:
                if (
                    shadow_signal is not None
                    and position is None
                    and shadow_strategy.allows_entry_at(latest.timestamp)
                ):
                    record_shadow_signal(
                        signal_feedback,
                        shadow_signal,
                        timestamp=latest.timestamp,
                        horizon_bars=default_signal_horizon_bars(self.config.data.timeframe),
                        slippage_bps=self.config.execution.slippage_bps,
                        commission_bps=self.config.execution.commission_bps,
                        **self._signal_feedback_runner_kwargs(),
                    )
                continue

            if position and position.direction != signal.direction:
                broker.close_position(
                    instrument.symbol,
                    price=latest.close,
                    timestamp=latest.timestamp,
                    reason=ExitReason.SIGNAL_FLIP,
                )
                position = None

            if position is None:
                if shadow_signal is not None and shadow_strategy.allows_entry_at(latest.timestamp):
                    record_shadow_signal(
                        signal_feedback,
                        shadow_signal,
                        timestamp=latest.timestamp,
                        horizon_bars=default_signal_horizon_bars(self.config.data.timeframe),
                        slippage_bps=self.config.execution.slippage_bps,
                        commission_bps=self.config.execution.commission_bps,
                        **self._signal_feedback_runner_kwargs(),
                    )
                entry_block_reason = strategy.entry_block_reason_for_instrument(
                    instrument,
                    latest.timestamp,
                    signal.direction,
                )
                if entry_block_reason is not None:
                    cycle_events.append(
                        {
                            "timestamp": latest.timestamp.isoformat(),
                            "symbol": instrument.symbol,
                            "action": "signal",
                            "approved": False,
                            "reason": entry_block_reason,
                            "direction": signal.direction.value,
                            "strength": signal.strength,
                            "quantity_lots": 0,
                        }
                    )
                    continue
                decision = risk_manager.approve(broker.portfolio, signal, marks, broker.trades)
                signal_for_entry = signal
                entry_quantity_lots = decision.quantity_lots
                microstructure_block_reason = None
                if decision.approved:
                    microstructure_block_reason = self._entry_confirmation_block_reason(signal_for_entry)
                    if microstructure_block_reason is None:
                        signal_for_entry = self._signal_with_entry_microstructure(
                            provider,
                            signal,
                            quantity_lots=entry_quantity_lots,
                        )
                        signal_for_entry = self._signal_with_learning_assessment(
                            signal_for_entry,
                            signal_feedback,
                            timestamp=latest.timestamp,
                            quantity_lots=entry_quantity_lots,
                        )
                        microstructure_block_reason = self._microstructure_block_reason(signal_for_entry)
                        if microstructure_block_reason is None:
                            (
                                signal_for_entry,
                                entry_quantity_lots,
                                microstructure_block_reason,
                            ) = self._signal_with_runtime_policy(
                                signal_for_entry,
                                entry_quantity_lots,
                                market_regime=market_regime,
                                symbol_health=self._symbol_health(instrument.symbol, broker.trades),
                                entry_mode=(
                                    "trend_short"
                                    if signal.direction == SignalDirection.SHORT
                                    else "trend_long"
                                ),
                            )
                            if microstructure_block_reason is None:
                                (
                                    signal_for_entry,
                                    entry_quantity_lots,
                                    microstructure_block_reason,
                                    learning_cap_events,
                                ) = self._apply_learning_caps(
                                    broker,
                                    signal_for_entry,
                                    quantity_lots=entry_quantity_lots,
                                    timestamp=latest.timestamp,
                                    extra_events=cycle_events,
                                )
                                cycle_events.extend(learning_cap_events)
                                if microstructure_block_reason is not None:
                                    entry_quantity_lots = 0
                            if microstructure_block_reason is None:
                                pending_created_event = self._record_waiting_pullback_short(
                                    signal_feedback,
                                    signal_for_entry,
                                    candles=candles,
                                    timestamp=latest.timestamp,
                                    quantity_lots=decision.quantity_lots,
                                    market_regime=market_regime,
                                )
                                if pending_created_event is not None:
                                    signal_for_entry = self._signal_with_pending_addon_metadata(
                                        signal_for_entry,
                                        pending_created_event,
                                    )
                                    cycle_events.append(pending_created_event)
                                policy = signal_for_entry.metadata.get("regime_policy", {})
                                if isinstance(policy, dict) and policy.get("entry_mode") == "wait":
                                    microstructure_block_reason = "entry deferred for pullback short"
                            if microstructure_block_reason is None and entry_quantity_lots < 1:
                                microstructure_block_reason = "entry blocked by adaptive risk size < 1 lot"

                event = {
                    "timestamp": latest.timestamp.isoformat(),
                    "symbol": instrument.symbol,
                    "action": "signal",
                    "event_type": "policy_decision",
                    "approved": decision.approved and microstructure_block_reason is None,
                    "reason": microstructure_block_reason or decision.reason,
                    "direction": signal.direction.value,
                    "strength": signal.strength,
                    "quantity_lots": entry_quantity_lots if microstructure_block_reason is None else 0,
                    "original_quantity_lots": decision.quantity_lots,
                    "metadata": dict(signal_for_entry.metadata),
                }
                event["metadata"].setdefault("market_regime", market_regime.as_event())
                if "regime_policy" in event["metadata"]:
                    event["metadata"]["regime_policy_audit"] = event["metadata"]["regime_policy"]
                event.update(self._policy_event_fields(signal_for_entry))
                shadow_trade_id = self._record_rejected_shadow_if_needed(
                    signal_feedback,
                    signal_for_entry,
                    timestamp=latest.timestamp,
                    quantity_lots=decision.quantity_lots,
                    block_reason=microstructure_block_reason,
                )
                if shadow_trade_id:
                    event["shadow_trade_id"] = shadow_trade_id
                    event["metadata"]["shadow_trade_id"] = shadow_trade_id
                strict_shadow_id = self._record_strict_policy_shadow_if_needed(
                    signal_feedback,
                    signal_for_entry,
                    timestamp=latest.timestamp,
                )
                if strict_shadow_id:
                    event["strict_shadow_trade_id"] = strict_shadow_id
                    event["metadata"]["strict_shadow_trade_id"] = strict_shadow_id
                cycle_events.extend(
                    self._policy_auxiliary_events(
                        signal_for_entry,
                        latest.timestamp,
                        approved=bool(decision.approved and microstructure_block_reason is None),
                        block_reason=microstructure_block_reason,
                    )
                )
                cycle_events.extend(self._selloff_signal_events(event, market_regime))
                cycle_events.append(event)
                if decision.approved:
                    if microstructure_block_reason is None:
                        broker.open_position(signal_for_entry, entry_quantity_lots, latest.timestamp)

        if market_regime.regime == "market_selloff_impulse":
            cycle_events.extend(
                self._selloff_cycle_events(
                    broker,
                    marks,
                    cycle_events,
                    market_regime=market_regime,
                    timestamp=timestamp,
                )
            )
        broker.mark_to_market(marks, timestamp)
        broker.events.extend(cycle_events)
        broker.save(state_path)
        save_signal_feedback(feedback_path, signal_feedback)
        trade_review = build_trade_review_payload(
            broker.portfolio,
            broker.trades,
            broker.events,
            strategy=self.config.strategy,
            risk=self.config.risk,
            timezone_name=self.config.app.timezone,
            lookback_trades=100,
            generated_at=timestamp,
            microstructure_dir=self.config.resolve_path(self.config.reporting.output_dir) / "microstructure",
        )
        latest_trade_review_path = trade_review_path(state_path)
        save_trade_review(latest_trade_review_path, trade_review)
        trade_review["latest_path"] = str(latest_trade_review_path)
        add_runtime_metadata(trade_review)
        signal_activity = _summarize_signal_activity(cycle_events)
        stamp = timestamp.strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.resolve_path(self.config.reporting.output_dir) / "paper" / stamp
        trade_review["output_dir"] = str(output_dir)

        summary = {
            "timestamp": timestamp.isoformat(),
            "equity_rub": round(broker.portfolio.equity(marks), 2),
            "cash_rub": round(broker.portfolio.cash, 2),
            "gross_exposure_rub": round(broker.portfolio.gross_exposure(marks), 2),
            "open_positions": len(broker.portfolio.positions),
            "trading_halted": broker.portfolio.trading_halted,
            "trade_review": _trade_review_view(trade_review),
            **signal_activity,
        }
        add_runtime_metadata(summary)
        cycle_events_payload = {"events": cycle_events}
        add_runtime_metadata(cycle_events_payload)
        write_json_payload(output_dir / "cycle_summary.json", summary)
        write_json_payload(output_dir / "cycle_events.json", cycle_events_payload)
        write_trade_review(output_dir, trade_review)
        write_portfolio_snapshot(output_dir / "portfolio.json", broker.portfolio)
        return {"summary": summary, "output_dir": str(output_dir)}

    def _load_entry_confirmation_history(self, provider, instruments, primary_history):
        timeframe = self.config.strategy.entry_confirmation_timeframe.strip()
        if not timeframe:
            return {}
        if timeframe.lower() == self.config.data.timeframe.lower():
            return primary_history
        if hasattr(provider, "load_history_for_timeframe"):
            try:
                return provider.load_history_for_timeframe(instruments, timeframe)
            except Exception as exc:  # pragma: no cover - live API dependent
                LOGGER.warning("Entry confirmation history failed for %s: %s", timeframe, exc)
                return {}

        confirmation_config = replace(
            self.config,
            data=replace(self.config.data, timeframe=timeframe),
        )
        confirmation_provider = TradingOrchestrator(confirmation_config)._data_provider()
        try:
            return confirmation_provider.load_history(instruments)
        except Exception as exc:  # pragma: no cover - live API dependent
            LOGGER.warning("Entry confirmation history failed for %s: %s", timeframe, exc)
            return {}

    def _signal_with_entry_candle_context(self, signal, candles):
        metadata = dict(signal.metadata)
        metadata["entry_candle"] = build_entry_candle_context(candles, signal.direction.value)
        return replace(signal, metadata=metadata)

    def _signal_with_entry_confirmation_context(self, signal, candles, *, signal_timestamp: datetime):
        metadata = dict(signal.metadata)
        metadata["entry_confirmation"] = build_entry_confirmation_context(
            candles,
            signal.direction.value,
            signal_timestamp=signal_timestamp,
            primary_timeframe=self.config.data.timeframe,
            confirmation_timeframe=self.config.strategy.entry_confirmation_timeframe,
            min_bars=self.config.strategy.entry_confirmation_min_bars,
            max_adverse_ret=self.config.strategy.entry_confirmation_max_adverse_ret,
        )
        return replace(signal, metadata=metadata)

    def _signal_with_setup_learning_tags(self, signal, recent_trades, *, timestamp: datetime):
        tags = build_setup_learning_tags(
            signal,
            recent_trades,
            timestamp=timestamp,
            timezone_name=self.config.app.timezone,
        )
        if not tags:
            return signal
        metadata = dict(signal.metadata)
        existing_tags = [
            str(tag)
            for tag in metadata.get("setup_learning_tags", [])
            if str(tag)
        ]
        metadata["setup_learning_tags"] = sorted(set(existing_tags).union(tags))
        return replace(signal, metadata=metadata)

    def _signal_with_learning_assessment(
        self,
        signal,
        feedback_payload: dict[str, list[dict[str, object]]],
        *,
        timestamp: datetime,
        quantity_lots: int,
    ):
        metadata = dict(signal.metadata)
        metadata["ml_learning"] = assess_signal_learning(
            signal,
            feedback_payload,
            timestamp=timestamp,
            quantity_lots=quantity_lots,
            timezone_name=self.config.app.timezone,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        return replace(signal, metadata=metadata)

    def _process_pending_entry_trigger(
        self,
        provider,
        broker,
        risk_manager,
        feedback_payload: dict[str, object],
        pending_result: dict[str, object],
        marks: dict[str, float],
        *,
        market_regime,
        timestamp: datetime,
    ) -> list[dict[str, object]]:
        item = dict(pending_result.get("item", {}))
        event_timestamp = str(pending_result.get("timestamp", timestamp.isoformat()))
        fill_timestamp = self._parse_event_timestamp(event_timestamp, fallback=timestamp)
        if bool(item.get("addon_shadow_only_due_to_no_pyramiding", False)):
            return [
                {
                    "timestamp": event_timestamp,
                    "symbol": item.get("symbol", ""),
                    "action": "pending-entry",
                    "status": "shadow-triggered",
                    "state": item.get("state", ""),
                    "reason": "pending add-on observed only because pyramiding is unsupported",
                    "metadata": {
                        "id": item.get("id", ""),
                        "created_at": item.get("created_at", ""),
                        "triggered_at": item.get("triggered_at", ""),
                        "bars_seen": item.get("bars_seen", 0),
                        "quantity_lots": item.get("quantity_lots", 0),
                        "is_addon": item.get("is_addon", False),
                        "parent_entry_mode": item.get("parent_entry_mode", ""),
                        "parent_decision_type": item.get("parent_decision_type", ""),
                        "addon_multiplier": item.get("addon_multiplier", 0.0),
                        "addon_shadow_only_due_to_no_pyramiding": True,
                    },
                }
            ]
        signal = pending_entry_signal(
            item,
            reward_to_risk=self.config.strategy.reward_to_risk,
        )
        approved_pending_quantity = pending_entry_quantity_lots(item)
        decision = risk_manager.approve(broker.portfolio, signal, marks, broker.trades)
        signal_for_entry = signal
        entry_quantity_lots = min(decision.quantity_lots, approved_pending_quantity)
        block_reason = None
        learning_cap_events: list[dict[str, object]] = []
        if decision.approved and entry_quantity_lots < 1:
            block_reason = "entry blocked by pending-entry quantity limit"
        if decision.approved and block_reason is None:
            signal_for_entry = self._signal_with_entry_microstructure(
                provider,
                signal_for_entry,
                quantity_lots=entry_quantity_lots,
            )
            signal_for_entry = self._signal_with_learning_assessment(
                signal_for_entry,
                feedback_payload,
                timestamp=fill_timestamp,
                quantity_lots=entry_quantity_lots,
            )
            block_reason = self._microstructure_block_reason(signal_for_entry)
            if block_reason is None:
                (
                    signal_for_entry,
                    entry_quantity_lots,
                    block_reason,
                ) = self._signal_with_runtime_policy(
                    signal_for_entry,
                    entry_quantity_lots,
                    market_regime=market_regime,
                    symbol_health=self._symbol_health(signal.instrument.symbol, broker.trades),
                    entry_mode="pullback_short",
                )
                if block_reason is None:
                    (
                        signal_for_entry,
                        entry_quantity_lots,
                        block_reason,
                        learning_cap_events,
                    ) = self._apply_learning_caps(
                        broker,
                        signal_for_entry,
                        quantity_lots=entry_quantity_lots,
                        timestamp=fill_timestamp,
                    )
                    if block_reason is not None:
                        entry_quantity_lots = 0
                if block_reason is None and entry_quantity_lots < 1:
                    block_reason = "entry blocked by adaptive risk size < 1 lot"

        pending_event = {
            "timestamp": event_timestamp,
            "symbol": signal.instrument.symbol,
            "action": "pending-entry",
            "status": "triggered",
            "state": item.get("state", ""),
            "reason": pending_result.get("reason", "pending entry triggered"),
            "metadata": {
                "id": item.get("id", ""),
                "created_at": item.get("created_at", ""),
                "triggered_at": item.get("triggered_at", ""),
                "bars_seen": item.get("bars_seen", 0),
                "rebound_high": item.get("rebound_high", 0.0),
                "quantity_lots": approved_pending_quantity,
            },
        }
        signal_event = {
            "timestamp": event_timestamp,
            "symbol": signal.instrument.symbol,
            "action": "signal",
            "event_type": "policy_decision",
            "approved": decision.approved and block_reason is None,
            "reason": block_reason or decision.reason,
            "direction": signal.direction.value,
            "strength": signal.strength,
            "quantity_lots": entry_quantity_lots if block_reason is None else 0,
            "original_quantity_lots": decision.quantity_lots,
            "metadata": dict(signal_for_entry.metadata),
        }
        signal_event["metadata"].setdefault("market_regime", market_regime.as_event())
        if "regime_policy" in signal_event["metadata"]:
            signal_event["metadata"]["regime_policy_audit"] = signal_event["metadata"]["regime_policy"]
        signal_event.update(self._policy_event_fields(signal_for_entry))
        shadow_trade_id = self._record_rejected_shadow_if_needed(
            feedback_payload,
            signal_for_entry,
            timestamp=fill_timestamp,
            quantity_lots=approved_pending_quantity,
            block_reason=block_reason,
        )
        if shadow_trade_id:
            signal_event["shadow_trade_id"] = shadow_trade_id
            signal_event["metadata"]["shadow_trade_id"] = shadow_trade_id
        strict_shadow_id = self._record_strict_policy_shadow_if_needed(
            feedback_payload,
            signal_for_entry,
            timestamp=fill_timestamp,
        )
        if strict_shadow_id:
            signal_event["strict_shadow_trade_id"] = strict_shadow_id
            signal_event["metadata"]["strict_shadow_trade_id"] = strict_shadow_id
        if decision.approved and block_reason is None:
            broker.open_position(signal_for_entry, entry_quantity_lots, fill_timestamp)
        return [pending_event, *learning_cap_events, signal_event]

    def _signal_with_runtime_policy(
        self,
        signal,
        quantity_lots: int,
        *,
        market_regime,
        symbol_health: str,
        entry_mode: str,
    ):
        policy = resolve_regime_policy(
            regime=market_regime.regime,
            symbol=signal.instrument.symbol,
            side=signal.direction,
            ml_feedback=signal.metadata.get("ml_learning", {}),
            book=signal.metadata.get("microstructure", {}),
            confirmation=signal.metadata.get("entry_confirmation", {}),
            entry_mode=entry_mode,
            symbol_health=symbol_health,
            long_side_enabled=True,
            learning_mode_enabled=self._relaxed_learning_enabled(),
            learning_profile=self.config.learning_mode.profile,
            signal_strength=signal.strength,
            trend_strength=self._signal_metadata_float(signal, "trend_strength"),
            adx=indicator_from_reason(signal.reason, "adx"),
            require_order_book=self.config.strategy.require_order_book,
            config=self.config,
        )
        metadata = dict(signal.metadata)
        metadata["market_regime"] = market_regime.as_event()
        metadata["symbol_health"] = symbol_health
        metadata["regime_policy"] = policy.as_metadata()
        metadata["regime_policy_audit"] = policy.as_metadata()
        metadata["entry_mode"] = policy.entry_mode
        metadata["actual_policy_profile"] = policy.actual_policy_profile
        metadata["actual_policy_decision"] = policy.decision_type
        metadata["strict_policy_decision"] = policy.strict_policy_decision
        metadata["strict_policy_reasons"] = list(policy.strict_policy_reasons)
        metadata["actual_policy_reasons"] = list(policy.actual_policy_reasons)
        metadata["would_strict_policy_trade"] = policy.would_strict_policy_trade
        metadata["would_strict_policy_risk_multiplier"] = policy.would_strict_policy_risk_multiplier
        metadata["relaxed_only_trade"] = policy.relaxed_only_trade
        metadata["risk_multiplier"] = policy.risk_multiplier
        metadata["effective_risk_multiplier"] = policy.effective_risk_multiplier
        metadata["soft_issues"] = list(policy.soft_issues)
        metadata["hard_issues"] = list(policy.hard_issues)
        metadata["policy_tags"] = list(policy.tags)
        metadata["side_policy"] = dict(policy.side_policy)
        metadata["symbol_health_policy"] = dict(policy.symbol_health_metadata)
        metadata["confirmation_5m"] = dict(policy.confirmation_5m)

        original_quantity = max(0, int(quantity_lots))
        adjusted_quantity = original_quantity
        block_reason = None
        if policy.decision_type == PolicyDecisionType.WAIT_PULLBACK.value or policy.entry_mode == "wait":
            adjusted_quantity = 0
        elif not policy.allow_trade:
            adjusted_quantity = 0
            block_reason = self._policy_block_reason(policy)
        else:
            adjusted_quantity = self._quantity_after_multiplier(original_quantity, policy.risk_multiplier)
            if adjusted_quantity < 1:
                block_reason = "entry blocked by adaptive risk size < 1 lot"

        metadata["adaptive_risk_sizing"] = {
            "original_quantity_lots": original_quantity,
            "adjusted_quantity_lots": adjusted_quantity,
            "risk_multiplier": round(float(policy.risk_multiplier), 4),
            "entry_mode": policy.entry_mode,
            "symbol_health": symbol_health,
            "reasons": list(policy.reasons),
            "risk_components": dict(policy.risk_components),
            "decision_type": policy.decision_type,
            "actual_policy_profile": policy.actual_policy_profile,
            "soft_issues": list(policy.soft_issues),
            "hard_issues": list(policy.hard_issues),
        }
        ml_adjustment = learning_position_size_adjustment(
            signal.metadata.get("ml_learning", {}),
            original_quantity,
        )
        metadata["ml_sizing"] = {
            "original_quantity_lots": original_quantity,
            "final_quantity_lots": adjusted_quantity,
            "scale": (
                round(adjusted_quantity / original_quantity, 4)
                if original_quantity
                else 0.0
            ),
            "ml_negative_edge": policy.ml_negative_edge,
            "soft_issue_count": len(policy.soft_issues),
            "hard_issue_count": len(policy.hard_issues),
            "policy_multiplier": policy.risk_multiplier,
        }
        if ml_adjustment.get("active"):
            metadata["ml_sizing"] = {
                **metadata["ml_sizing"],
                **ml_adjustment,
                "adjusted_quantity_lots": adjusted_quantity,
            }

        return replace(signal, metadata=metadata), adjusted_quantity, block_reason

    def _record_waiting_pullback_short(
        self,
        feedback_payload: dict[str, object],
        signal,
        *,
        candles,
        timestamp: datetime,
        quantity_lots: int,
        market_regime,
    ) -> dict[str, object] | None:
        policy = signal.metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            return None
        is_wait_only = policy.get("entry_mode") == "wait"
        create_addon = bool(policy.get("probe_now_with_pending_addon", False))
        if not is_wait_only and not create_addon:
            return None
        addon_multiplier = float(policy.get("pending_addon_multiplier", 0.0) or 0.0)
        pending_quantity_lots = int(quantity_lots)
        addon_metadata: dict[str, object] = {}
        if create_addon:
            pending_quantity_lots = self._quantity_after_multiplier(
                max(1, int(quantity_lots)),
                addon_multiplier or 0.15,
            )
            addon_metadata = {
                "is_addon": True,
                "parent_entry_mode": policy.get("entry_mode", ""),
                "parent_decision_type": policy.get("actual_policy_decision", policy.get("decision_type", "")),
                "addon_multiplier": addon_multiplier or 0.15,
                "strict_policy_original_decision": policy.get("strict_policy_decision", "unknown"),
                "relaxed_probe_opened": True,
                "addon_shadow_only_due_to_no_pyramiding": True,
            }
        item = record_pending_pullback_short(
            feedback_payload,
            signal,
            candles=candles,
            timestamp=timestamp,
            quantity_lots=pending_quantity_lots,
            policy_metadata=policy,
            market_regime=market_regime.as_event(),
            addon_metadata=addon_metadata,
        )
        if item is None:
            return None
        reason = (
            "weak choppy probe opened; pending pullback add-on created"
            if create_addon
            else "entry deferred for pullback short"
        )
        return {
            "timestamp": timestamp.isoformat(),
            "symbol": signal.instrument.symbol,
            "action": "pending-entry",
            "status": "created",
            "state": item.get("state", ""),
            "reason": reason,
            "metadata": {
                "id": item.get("id", ""),
                "entry_price": item.get("entry_price", 0.0),
                "pullback_trigger_price": item.get("pullback_trigger_price", 0.0),
                "failed_rebound_price": item.get("failed_rebound_price", 0.0),
                "expires_after_bars": item.get("expires_after_bars", 0),
                "quantity_lots": item.get("quantity_lots", 0),
                "is_addon": item.get("is_addon", False),
                "parent_entry_mode": item.get("parent_entry_mode", ""),
                "parent_decision_type": item.get("parent_decision_type", ""),
                "addon_multiplier": item.get("addon_multiplier", 0.0),
                "strict_policy_original_decision": item.get("strict_policy_original_decision", ""),
                "relaxed_probe_opened": item.get("relaxed_probe_opened", False),
                "addon_shadow_only_due_to_no_pyramiding": item.get(
                    "addon_shadow_only_due_to_no_pyramiding",
                    False,
                ),
                "event_type": (
                    "weak_choppy_probe_now_pending_addon_created"
                    if create_addon
                    else "weak_choppy_wait_only_selected"
                ),
                "regime_policy": policy,
            },
        }

    @staticmethod
    def _signal_with_pending_addon_metadata(signal, pending_event: dict[str, object]):
        event_metadata = pending_event.get("metadata", {})
        if not isinstance(event_metadata, dict):
            return signal
        if not bool(event_metadata.get("is_addon", False)):
            return signal
        metadata = dict(signal.metadata)
        policy = dict(metadata.get("regime_policy", {})) if isinstance(metadata.get("regime_policy"), dict) else {}
        addon_fields = {
            "pending_addon_created": True,
            "pending_addon_id": event_metadata.get("id", ""),
            "pending_addon_type": policy.get("pending_addon_type", "wait_pullback_short"),
            "pending_addon_multiplier": event_metadata.get(
                "addon_multiplier",
                policy.get("pending_addon_multiplier", 0.0),
            ),
            "addon_shadow_only_due_to_no_pyramiding": event_metadata.get(
                "addon_shadow_only_due_to_no_pyramiding",
                False,
            ),
        }
        metadata.update(addon_fields)
        if policy:
            policy.update(addon_fields)
            metadata["regime_policy"] = policy
            metadata["regime_policy_audit"] = policy
        return replace(signal, metadata=metadata)

    def _policy_event_fields(self, signal) -> dict[str, object]:
        policy = signal.metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            return {}
        return {
            "actual_policy_profile": policy.get("actual_policy_profile", "strict"),
            "actual_policy_decision": policy.get("actual_policy_decision", policy.get("decision_type", "")),
            "strict_policy_decision": policy.get("strict_policy_decision", "unknown"),
            "would_strict_policy_trade": policy.get("would_strict_policy_trade", True),
            "risk_multiplier": policy.get("risk_multiplier", 1.0),
            "effective_risk_multiplier": policy.get("effective_risk_multiplier", 1.0),
            "soft_issues": policy.get("soft_issues", []),
            "hard_issues": policy.get("hard_issues", []),
            "tags": policy.get("tags", []),
            "microstructure_bucket": policy.get("microstructure_bucket", "unknown"),
            "confirmation_5m_status": policy.get("confirmation_5m_status", "unknown"),
            "ml_negative_edge": policy.get("ml_negative_edge", False),
            "symbol_health_status": policy.get("symbol_health", "unknown"),
            "relaxed_only_trade": policy.get("relaxed_only_trade", False),
        }

    def _policy_auxiliary_events(
        self,
        signal,
        timestamp: datetime,
        *,
        approved: bool,
        block_reason: str | None,
    ) -> list[dict[str, object]]:
        policy = signal.metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            return []
        metadata = dict(signal.metadata)
        base = {
            "timestamp": timestamp.isoformat(),
            "symbol": signal.instrument.symbol,
            "side": signal.direction.value,
            "regime": metadata.get("market_regime", {}).get("regime", "")
            if isinstance(metadata.get("market_regime"), dict)
            else "",
            "signal_strength": signal.strength,
            "actual_policy_decision": policy.get("actual_policy_decision", policy.get("decision_type", "")),
            "strict_policy_decision": policy.get("strict_policy_decision", "unknown"),
            "relaxed_only_trade": policy.get("relaxed_only_trade", False),
            "entry_mode": policy.get("entry_mode", ""),
            "risk_multiplier": policy.get("risk_multiplier", 0.0),
            "effective_risk_multiplier": policy.get("effective_risk_multiplier", 0.0),
            "soft_issues": policy.get("soft_issues", []),
            "hard_issues": policy.get("hard_issues", []),
            "reasons": policy.get("reasons", []),
            "tags": policy.get("tags", []),
            "pending_addon_created": policy.get("pending_addon_created", False),
            "pending_addon_id": policy.get("pending_addon_id", ""),
            "long_context": policy.get("long_context", {}),
            "ml_action": (
                metadata.get("ml_learning", {}).get("action", "")
                if isinstance(metadata.get("ml_learning"), dict)
                else ""
            ),
            "ml_expected_position": (
                metadata.get("ml_learning", {}).get("expected_pnl_position_rub", 0.0)
                if isinstance(metadata.get("ml_learning"), dict)
                else 0.0
            ),
            "block_reason": block_reason or "",
        }
        events: list[dict[str, object]] = []
        entry_mode = str(policy.get("entry_mode", ""))
        decision = str(policy.get("actual_policy_decision", policy.get("decision_type", "")))
        if entry_mode.startswith("weak_choppy_direct_"):
            events.append({**base, "action": "weak_choppy_probe_now_selected"})
            if bool(policy.get("pending_addon_created", False)):
                events.append({**base, "action": "weak_choppy_probe_now_pending_addon_created"})
            if approved:
                events.append({**base, "action": "weak_choppy_probe_now_opened"})
            else:
                events.append({**base, "action": "weak_choppy_probe_now_rejected"})
        elif entry_mode == "wait":
            events.append({**base, "action": "weak_choppy_wait_only_selected"})
        if bool(policy.get("strict_wait_overridden_by_relaxed_probe", False)):
            events.append({**base, "action": "strict_wait_overridden_by_relaxed_probe"})
        if signal.direction == SignalDirection.LONG:
            if decision == PolicyDecisionType.PROBE_TRADE.value:
                events.append({**base, "action": "long_probe_selected"})
                events.append({**base, "action": "long_probe_opened" if approved else "long_probe_rejected"})
            elif decision == PolicyDecisionType.EXPLORATION_TRADE.value:
                events.append({**base, "action": "long_exploration_opened" if approved else "long_probe_rejected"})
            elif decision == PolicyDecisionType.SHADOW_ONLY.value:
                events.append({**base, "action": "long_shadow_only_created"})
        return events

    @staticmethod
    def _market_selloff_detected_event(market_regime, timestamp: datetime) -> dict[str, object]:
        features = dict(market_regime.as_event().get("features", {}))
        return {
            "timestamp": timestamp.isoformat(),
            "action": "market_selloff_impulse_detected",
            "event_type": "market_regime_override",
            "regime": market_regime.regime,
            "confidence": round(float(market_regime.confidence), 4),
            "universe_ret_15m": features.get("universe_ret_15m", 0.0),
            "universe_ret_30m": features.get("universe_ret_30m", 0.0),
            "universe_ret_60m": features.get("universe_ret_60m", 0.0),
            "breadth_down_5m": features.get("breadth_down_5m", 0.0),
            "breadth_down_15m": features.get("breadth_down_15m", 0.0),
            "breadth_breaking_15m_lows": features.get("breadth_breaking_15m_lows", 0.0),
            "breadth_breaking_30m_lows": features.get("breadth_breaking_30m_lows", 0.0),
            "symbols_confirming_count": features.get("symbols_confirming_count", 0),
            "used_fallback": features.get("used_fallback", False),
            "previous_regime": features.get("previous_regime", "unknown"),
            "override_reason": features.get("override_reason", ""),
            "metadata": {"market_regime": market_regime.as_event()},
        }

    def _selloff_signal_events(self, event: dict[str, object], market_regime) -> list[dict[str, object]]:
        if market_regime.regime != "market_selloff_impulse":
            return []
        if str(event.get("direction", event.get("side", ""))) != SignalDirection.SHORT.value:
            return []
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        policy = metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        base = {
            "timestamp": event.get("timestamp", ""),
            "symbol": event.get("symbol", ""),
            "action": "selloff_short_candidate",
            "regime": "market_selloff_impulse",
            "direction": SignalDirection.SHORT.value,
            "strength": event.get("strength", 0.0),
            "approved": bool(event.get("approved", False)),
            "reason": event.get("reason", ""),
            "entry_mode": event.get("entry_mode", metadata.get("entry_mode", policy.get("entry_mode", ""))),
            "actual_policy_decision": event.get(
                "actual_policy_decision",
                policy.get("actual_policy_decision", policy.get("decision_type", "")),
            ),
            "strict_policy_decision": event.get("strict_policy_decision", policy.get("strict_policy_decision", "")),
            "quantity_lots": event.get("quantity_lots", 0),
            "risk_multiplier": event.get("risk_multiplier", policy.get("risk_multiplier", 0.0)),
            "soft_issues": event.get("soft_issues", policy.get("soft_issues", [])),
            "hard_issues": event.get("hard_issues", policy.get("hard_issues", [])),
            "metadata": {
                "market_regime": market_regime.as_event(),
                "regime_policy": policy,
                "original_signal_event": {
                    "approved": bool(event.get("approved", False)),
                    "reason": event.get("reason", ""),
                    "quantity_lots": event.get("quantity_lots", 0),
                    "original_quantity_lots": event.get("original_quantity_lots", 0),
                },
            },
        }
        events: list[dict[str, object]] = [base]
        if bool(policy.get("selloff_policy_override_applied", False)):
            events.append({**base, "action": "selloff_policy_override_applied"})
        events.append(
            {
                **base,
                "action": "selloff_short_opened" if bool(event.get("approved", False)) else "selloff_short_rejected",
            }
        )
        return events

    def _selloff_cycle_events(
        self,
        broker,
        marks: dict[str, float],
        cycle_events: list[dict[str, object]],
        *,
        market_regime,
        timestamp: datetime,
    ) -> list[dict[str, object]]:
        candidates = [event for event in cycle_events if event.get("action") == "selloff_short_candidate"]
        approved = [event for event in cycle_events if event.get("action") == "selloff_short_opened"]
        rejected = [event for event in cycle_events if event.get("action") == "selloff_short_rejected"]
        wait_count = sum(
            1
            for event in candidates
            if str(event.get("entry_mode", "")) == "wait"
            or str(event.get("actual_policy_decision", "")) == PolicyDecisionType.WAIT_PULLBACK.value
        )
        shadow_count = sum(
            1
            for event in candidates
            if str(event.get("actual_policy_decision", "")) == PolicyDecisionType.SHADOW_ONLY.value
        )
        equity = broker.portfolio.equity(marks)
        gross_exposure = broker.portfolio.gross_exposure(marks)
        gross_exposure_pct = gross_exposure / equity if equity > 0 else 0.0
        normal_target = self._paper_alpha_target_gross_exposure(normal=True)
        selloff_target = self._paper_alpha_target_gross_exposure(normal=False)
        budget_used_pct = gross_exposure_pct / selloff_target if selloff_target > 0 else 0.0
        blockers = self._selloff_budget_blockers(candidates, rejected, wait_count=wait_count, shadow_count=shadow_count)
        unused_reason = self._selloff_unused_budget_reason(
            candidates_count=len(candidates),
            approved_count=len(approved),
            gross_exposure_pct=gross_exposure_pct,
            selloff_target=selloff_target,
            blockers=blockers,
        )
        diagnostics = {
            "equity": round(equity, 2),
            "gross_exposure": round(gross_exposure, 2),
            "gross_exposure_pct": round(gross_exposure_pct, 6),
            "target_gross_exposure": round(normal_target, 6),
            "selloff_target_gross_exposure": round(selloff_target, 6),
            "budget_used_pct": round(budget_used_pct, 6),
            "unused_budget_reason": unused_reason,
            "candidates_count": len(candidates),
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "wait_count": wait_count,
            "shadow_count": shadow_count,
            "selloff_budget_blockers": blockers,
        }
        selected_symbols = [str(event.get("symbol", "")) for event in approved if str(event.get("symbol", ""))]
        events = [
            {
                "timestamp": timestamp.isoformat(),
                "action": "selloff_basket_selected",
                "regime": market_regime.regime,
                "selected_symbols": selected_symbols,
                "candidates_count": len(candidates),
                "selected_count": len(selected_symbols),
                "metadata": {"selloff_budget_diagnostics": diagnostics},
            }
        ]
        budget_action = "selloff_budget_used" if budget_used_pct >= 0.80 else "selloff_budget_unused"
        events.append(
            {
                "timestamp": timestamp.isoformat(),
                "action": budget_action,
                "regime": market_regime.regime,
                **diagnostics,
                "metadata": {"selloff_budget_diagnostics": diagnostics},
            }
        )
        if gross_exposure_pct < 0.30:
            events.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "action": "selloff_underallocated",
                    "severity": "warning",
                    "regime": market_regime.regime,
                    **diagnostics,
                    "metadata": {"selloff_budget_diagnostics": diagnostics},
                }
            )
        return events

    def _paper_alpha_target_gross_exposure(self, *, normal: bool) -> float:
        paper_alpha = getattr(self.config, "paper_alpha_capture", None)
        if bool(getattr(paper_alpha, "enabled", False)) and self.config.execution.mode.value == "local-paper":
            field = "target_gross_exposure_normal" if normal else "target_gross_exposure_selloff"
            return max(0.0, float(getattr(paper_alpha, field, 0.40 if normal else 1.00)))
        if normal:
            return min(float(self.config.risk.max_gross_exposure), 0.40)
        basket = getattr(getattr(self.config, "market_selloff_impulse", None), "basket", None)
        return max(0.0, float(getattr(basket, "max_total_selloff_gross_exposure", 1.00)))

    @staticmethod
    def _selloff_budget_blockers(
        candidates: list[dict[str, object]],
        rejected: list[dict[str, object]],
        *,
        wait_count: int,
        shadow_count: int,
    ) -> dict[str, int]:
        blockers: Counter[str] = Counter()
        if not candidates:
            blockers["no candidates"] += 1
        if wait_count:
            blockers["wait_pullback"] += wait_count
        if shadow_count:
            blockers["shadow_only"] += shadow_count
        for event in rejected:
            reason = str(event.get("reason", "")).lower()
            hard_issues = " ".join(str(issue).lower() for issue in event.get("hard_issues", []) if issue)
            soft_issues = " ".join(str(issue).lower() for issue in event.get("soft_issues", []) if issue)
            text = " ".join([reason, hard_issues, soft_issues])
            if "risk" in text or "exposure" in text or "cash" in text or "positions" in text:
                blockers["risk blocked"] += 1
            elif "micro" in text or "book" in text or "spread" in text or "liquidity" in text:
                blockers["microstructure blocked"] += 1
            elif "confirmation" in text or "rebound" in text or "pullback" in text:
                blockers["confirmation blocked"] += 1
            elif "ml" in text:
                blockers["ML blocked"] += 1
            elif "lot" in text or "size" in text:
                blockers["lot sizing failed"] += 1
            elif "policy" in text or "shadow" in text:
                blockers["policy blocked"] += 1
            else:
                blockers["unknown"] += 1
        return dict(sorted(blockers.items()))

    @staticmethod
    def _selloff_unused_budget_reason(
        *,
        candidates_count: int,
        approved_count: int,
        gross_exposure_pct: float,
        selloff_target: float,
        blockers: dict[str, int],
    ) -> str:
        if selloff_target > 0 and gross_exposure_pct >= selloff_target * 0.80:
            return "target gross exposure reached"
        if candidates_count <= 0:
            return "no candidates"
        if approved_count <= 0:
            if blockers:
                return max(blockers.items(), key=lambda item: item[1])[0]
            return "policy blocked"
        if blockers:
            return "partial allocation: " + max(blockers.items(), key=lambda item: item[1])[0]
        return "partial allocation: risk manager sizing or position slots"

    def _record_rejected_shadow_if_needed(
        self,
        feedback_payload: dict[str, object],
        signal,
        *,
        timestamp: datetime,
        quantity_lots: int,
        block_reason: str | None,
    ) -> str:
        if not self._relaxed_learning_enabled():
            return ""
        if not bool(getattr(self.config.learning_mode, "record_rejected_shadow", False)):
            return ""
        policy = signal.metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            return ""
        decision_type = str(policy.get("decision_type", ""))
        if decision_type not in {
            PolicyDecisionType.HARD_REJECT.value,
            PolicyDecisionType.SHADOW_ONLY.value,
        }:
            return ""
        return record_rejected_shadow_signal(
            feedback_payload,
            signal,
            timestamp=timestamp,
            horizon_bars=default_signal_horizon_bars(self.config.data.timeframe),
            quantity_lots=max(1, int(quantity_lots or 1)),
            rejection_reason=block_reason or "; ".join(str(item) for item in policy.get("reasons", [])),
            decision_type=decision_type,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
            **self._signal_feedback_runner_kwargs(),
        )

    def _record_strict_policy_shadow_if_needed(
        self,
        feedback_payload: dict[str, object],
        signal,
        *,
        timestamp: datetime,
    ) -> str:
        if not self._relaxed_learning_enabled():
            return ""
        if not bool(getattr(self.config.learning_mode, "record_strict_policy_shadow", False)):
            return ""
        policy = signal.metadata.get("regime_policy", {})
        if not isinstance(policy, dict) or not policy.get("relaxed_only_trade"):
            return ""
        strict_shadow_id = "|".join(
            [
                "strict-shadow",
                signal.instrument.symbol,
                signal.direction.value,
                timestamp.isoformat(),
            ]
        )
        record = {
            "shadow_trade_id": strict_shadow_id,
            "symbol": signal.instrument.symbol,
            "direction": signal.direction.value,
            "created_at": timestamp.isoformat(),
            "decision_type": policy.get("strict_policy_decision", "unknown"),
            "reason": "; ".join(str(item) for item in policy.get("strict_policy_reasons", [])),
            "status": "strict_policy_shadow_for_relaxed_trade",
            "metadata": {
                "actual_policy_decision": policy.get("actual_policy_decision", ""),
                "strict_policy_decision": policy.get("strict_policy_decision", ""),
                "relaxed_only_trade": True,
                "regime_policy": policy,
            },
        }
        shadows = feedback_payload.setdefault("shadow_rejected", [])
        if not any(item.get("shadow_trade_id") == strict_shadow_id for item in shadows):
            shadows.append(record)
        return strict_shadow_id

    def _learning_mode_limit_reason(
        self,
        broker,
        signal,
        *,
        timestamp: datetime,
        extra_events: list[dict[str, object]] | None = None,
    ) -> str | None:
        _, _, reason, _ = self._apply_learning_caps(
            broker,
            signal,
            quantity_lots=1,
            timestamp=timestamp,
            extra_events=extra_events,
        )
        return reason

    def _apply_learning_caps(
        self,
        broker,
        signal,
        *,
        quantity_lots: int,
        timestamp: datetime,
        extra_events: list[dict[str, object]] | None = None,
    ):
        if not self._relaxed_learning_enabled():
            return signal, quantity_lots, None, []
        policy = signal.metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            return signal, quantity_lots, None, []
        decision_type = str(policy.get("decision_type", ""))
        mode_name = {
            PolicyDecisionType.NORMAL_TRADE.value: "normal",
            PolicyDecisionType.PROBE_TRADE.value: "probe",
            PolicyDecisionType.EXPLORATION_TRADE.value: "exploration",
        }.get(decision_type)
        if mode_name is None:
            return signal, quantity_lots, None, []
        mode_config = getattr(self.config.learning_risk, mode_name)
        selloff_active = (
            self._signal_market_regime(signal) == "market_selloff_impulse"
            and signal.direction == SignalDirection.SHORT
        )
        selloff_basket = getattr(getattr(self.config, "market_selloff_impulse", None), "basket", None)
        selloff_caps = getattr(getattr(self.config, "market_selloff_impulse", None), "learning_caps", None)
        mode_open_positions = sum(
            1
            for position in broker.portfolio.positions.values()
            if self._position_policy_mode(position) == decision_type
        )
        selloff_positions_count = sum(
            1
            for position in broker.portfolio.positions.values()
            if position.direction == SignalDirection.SHORT
            and self._position_market_regime(position) == "market_selloff_impulse"
        )
        use_global_slots = (
            mode_name == "probe"
            and bool(self.config.learning_mode.allow_probe_to_use_global_position_slots)
        ) or (
            mode_name == "exploration"
            and bool(self.config.learning_mode.allow_exploration_to_use_global_position_slots)
        )
        events = [{**event, "_historical": True} for event in broker.events] + list(extra_events or [])
        counts = self._learning_cap_counts(
            events,
            timestamp=timestamp,
            signal=signal,
            decision_type=decision_type,
        )
        selloff_cycle_count = (
            sum(
                1
                for event in events
                if event.get("action") == "signal"
                and bool(event.get("approved"))
                and self._event_market_regime(event) == "market_selloff_impulse"
                and str(event.get("direction", event.get("side", ""))) == SignalDirection.SHORT.value
            )
            if selloff_active
            else 0
        )
        tags: list[str] = []
        cap_behavior = "allow"
        multiplier = 1.0
        block_reason: str | None = None
        cap_events: list[dict[str, object]] = []

        if (
            selloff_active
            and int(getattr(selloff_basket, "max_selloff_positions", 0) or 0) > 0
            and selloff_positions_count >= int(getattr(selloff_basket, "max_selloff_positions", 0) or 0)
        ):
            block_reason = "entry blocked by selloff max positions"
            cap_behavior = "shadow_only"
            tags.append("selloff_positions_cap_hit")

        if (
            block_reason is None
            and not use_global_slots
            and int(mode_config.max_positions) > 0
            and mode_open_positions >= int(mode_config.max_positions)
        ):
            block_reason = f"entry blocked by {mode_name} learning max positions"
            cap_behavior = "block"
            tags.append(f"{mode_name}_positions_cap_hit")

        max_new_trades = (
            int(getattr(selloff_basket, "max_new_shorts_per_cycle", 0) or 0)
            if selloff_active
            else int(mode_config.max_new_trades_per_cycle)
        )
        current_cycle_count = selloff_cycle_count if selloff_active else counts["cycle_mode_count"]
        if block_reason is None and max_new_trades > 0 and current_cycle_count >= max_new_trades:
            block_reason = (
                "entry blocked by selloff per-cycle cap"
                if selloff_active
                else f"entry blocked by {mode_name} learning per-cycle cap"
            )
            cap_behavior = "shadow_only"
            tags.append("new_selloff_shorts_per_cycle_cap_hit" if selloff_active else "new_trades_per_cycle_learning_cap_hit")

        max_daily = int(mode_config.max_trades_per_day)
        if max_daily > 0 and counts["mode_count_today"] >= max_daily:
            tags.append(f"{mode_name}_daily_cap_soft_warning")
            daily_behavior = self._learning_cap_behavior("daily_cap_behavior", selloff_active=selloff_active)
            if block_reason is None and daily_behavior in {"shadow_only", "block"}:
                block_reason = f"entry blocked by {mode_name} learning daily trade cap"
                cap_behavior = daily_behavior
            elif cap_behavior == "allow":
                cap_behavior = "warn_only"

        same_symbol_limit = (
            int(getattr(selloff_caps, "max_same_symbol_selloff_trades_per_day", 0) or 0)
            if selloff_active
            else int(mode_config.max_same_symbol_trades_per_day)
        )
        if (
            block_reason is None
            and same_symbol_limit > 0
            and counts["same_symbol_count_today"] >= same_symbol_limit
        ):
            tags.extend(
                ["same_symbol_selloff_cap_hit", "oversampled_symbol"]
                if selloff_active
                else ["same_symbol_learning_cap_hit", "oversampled_symbol"]
            )
            behavior = self._learning_cap_behavior("same_symbol_cap_behavior", selloff_active=selloff_active)
            if behavior in {"shadow_only", "block"}:
                block_reason = "same_symbol_selloff_cap_hit" if selloff_active else "same_symbol_learning_cap_hit"
                cap_behavior = behavior
            elif behavior == "reduce_size":
                multiplier *= self._learning_cap_same_regime_multiplier()
                cap_behavior = "reduce_size"

        same_entry_limit = (
            int(getattr(selloff_caps, "max_same_entry_mode_selloff_trades_per_day", 0) or 0)
            if selloff_active
            else int(mode_config.max_same_entry_mode_trades_per_day)
        )
        if (
            block_reason is None
            and same_entry_limit > 0
            and counts["same_entry_mode_count_today"] >= same_entry_limit
        ):
            tags.extend(
                ["same_entry_mode_selloff_cap_hit", "oversampled_entry_mode"]
                if selloff_active
                else ["same_entry_mode_learning_cap_hit", "oversampled_entry_mode"]
            )
            behavior = self._learning_cap_behavior("same_entry_mode_cap_behavior", selloff_active=selloff_active)
            if behavior in {"shadow_only", "block"}:
                block_reason = "same_entry_mode_selloff_cap_hit" if selloff_active else "same_entry_mode_learning_cap_hit"
                cap_behavior = behavior
            elif behavior == "reduce_size":
                multiplier *= self._learning_cap_same_regime_multiplier()
                cap_behavior = "reduce_size"

        same_regime_limit = (
            int(getattr(selloff_caps, "max_same_regime_selloff_trades_per_day", 0) or 0)
            if selloff_active
            else int(mode_config.max_same_regime_trades_per_day)
        )
        if (
            block_reason is None
            and same_regime_limit > 0
            and counts["same_regime_count_today"] >= same_regime_limit
        ):
            tags.extend(
                ["same_regime_selloff_cap_hit", "oversampled_regime"]
                if selloff_active
                else ["same_regime_learning_cap_hit", "oversampled_regime"]
            )
            behavior = self._learning_cap_behavior("same_regime_cap_behavior", selloff_active=selloff_active)
            if behavior in {"shadow_only", "block"}:
                block_reason = "same_regime_selloff_cap_hit" if selloff_active else "same_regime_learning_cap_hit"
                cap_behavior = behavior
            elif behavior == "reduce_size":
                multiplier *= self._learning_cap_same_regime_multiplier()
                cap_behavior = "reduce_size"

        adjusted_quantity = max(0, int(quantity_lots))
        if block_reason is None and multiplier < 1.0:
            adjusted_quantity = self._quantity_after_multiplier(adjusted_quantity, multiplier)
            if adjusted_quantity < 1:
                block_reason = "entry blocked by learning cap reduced size < 1 lot"
                cap_behavior = "shadow_only"

        metadata = self._learning_cap_metadata(
            mode_name=mode_name,
            decision_type=decision_type,
            counts=counts,
            mode_open_positions=mode_open_positions,
            daily_cap_hit=max_daily > 0 and counts["mode_count_today"] >= max_daily,
            same_symbol_cap_hit="same_symbol_learning_cap_hit" in tags,
            same_entry_mode_cap_hit="same_entry_mode_learning_cap_hit" in tags,
            same_regime_cap_hit="same_regime_learning_cap_hit" in tags,
            cap_behavior=cap_behavior,
            tags=tags,
            quantity_lots=quantity_lots,
            adjusted_quantity_lots=0 if block_reason is not None else adjusted_quantity,
            size_multiplier=multiplier,
        )
        if selloff_active:
            metadata.update(
                {
                    "selloff_active": True,
                    "selloff_positions_count": selloff_positions_count,
                    "new_selloff_shorts_this_cycle": selloff_cycle_count,
                    "same_symbol_selloff_cap_hit": "same_symbol_selloff_cap_hit" in tags,
                    "same_entry_mode_selloff_cap_hit": "same_entry_mode_selloff_cap_hit" in tags,
                    "same_regime_selloff_cap_hit": "same_regime_selloff_cap_hit" in tags,
                }
            )
        signal = self._signal_with_learning_cap_metadata(signal, metadata)
        cap_events = self._learning_cap_events(
            signal,
            timestamp=timestamp,
            metadata=metadata,
            block_reason=block_reason,
        )
        if block_reason is not None:
            return signal, 0, block_reason, cap_events
        return signal, adjusted_quantity, None, cap_events

    def _learning_cap_counts(
        self,
        events: list[dict[str, object]],
        *,
        timestamp: datetime,
        signal,
        decision_type: str,
    ) -> dict[str, int]:
        today = timestamp.date()
        symbol = signal.instrument.symbol
        entry_mode = self._signal_entry_mode(signal)
        regime = self._signal_market_regime(signal)
        counts = {
            "mode_count_today": 0,
            "same_symbol_count_today": 0,
            "same_entry_mode_count_today": 0,
            "same_regime_count_today": 0,
            "cycle_mode_count": 0,
        }
        for event in events:
            if event.get("action") != "signal" or not bool(event.get("approved")):
                continue
            event_decision = self._event_policy_decision(event)
            if event_decision != decision_type:
                continue
            event_timestamp = self._parse_event_timestamp(str(event.get("timestamp", "")), fallback=timestamp)
            same_day = event_timestamp.date() == today
            if not same_day:
                continue
            counts["mode_count_today"] += 1
            if str(event.get("symbol", "")) == symbol:
                counts["same_symbol_count_today"] += 1
            if self._event_entry_mode(event) == entry_mode:
                counts["same_entry_mode_count_today"] += 1
            if self._event_market_regime(event) == regime:
                counts["same_regime_count_today"] += 1
            if not bool(event.get("_historical")):
                counts["cycle_mode_count"] += 1
        return counts

    @staticmethod
    def _event_policy_decision(event: dict[str, object]) -> str:
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        policy = metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        return str(
            event.get(
                "actual_policy_decision",
                metadata.get(
                    "actual_policy_decision",
                    policy.get("actual_policy_decision", policy.get("decision_type", "")),
                ),
            )
        )

    @staticmethod
    def _event_entry_mode(event: dict[str, object]) -> str:
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        policy = metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        return str(metadata.get("entry_mode", policy.get("entry_mode", "")))

    @staticmethod
    def _event_market_regime(event: dict[str, object]) -> str:
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        market_regime = metadata.get("market_regime", {})
        if isinstance(market_regime, dict):
            return str(market_regime.get("regime", "unknown"))
        return str(market_regime or "unknown")

    @staticmethod
    def _signal_entry_mode(signal) -> str:
        policy = signal.metadata.get("regime_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        return str(signal.metadata.get("entry_mode", policy.get("entry_mode", "")))

    @staticmethod
    def _signal_market_regime(signal) -> str:
        market_regime = signal.metadata.get("market_regime", {})
        if isinstance(market_regime, dict):
            return str(market_regime.get("regime", "unknown"))
        return str(market_regime or "unknown")

    def _learning_cap_metadata(
        self,
        *,
        mode_name: str,
        decision_type: str,
        counts: dict[str, int],
        mode_open_positions: int,
        daily_cap_hit: bool,
        same_symbol_cap_hit: bool,
        same_entry_mode_cap_hit: bool,
        same_regime_cap_hit: bool,
        cap_behavior: str,
        tags: list[str],
        quantity_lots: int,
        adjusted_quantity_lots: int,
        size_multiplier: float,
    ) -> dict[str, object]:
        return {
            "mode": mode_name,
            "decision_type": decision_type,
            "mode_open_positions": mode_open_positions,
            "probe_count_today": counts["mode_count_today"] if mode_name == "probe" else 0,
            "exploration_count_today": (
                counts["mode_count_today"] if mode_name == "exploration" else 0
            ),
            "same_symbol_count_today": counts["same_symbol_count_today"],
            "same_entry_mode_count_today": counts["same_entry_mode_count_today"],
            "same_regime_count_today": counts["same_regime_count_today"],
            "new_trades_this_cycle": counts["cycle_mode_count"],
            "daily_cap_hit": daily_cap_hit,
            "same_symbol_cap_hit": same_symbol_cap_hit,
            "same_entry_mode_cap_hit": same_entry_mode_cap_hit,
            "same_regime_cap_hit": same_regime_cap_hit,
            "cap_behavior_applied": cap_behavior,
            "oversampling_tags": list(tags),
            "original_quantity_lots": int(quantity_lots),
            "adjusted_quantity_lots": int(adjusted_quantity_lots),
            "size_multiplier": round(float(size_multiplier), 4),
        }

    @staticmethod
    def _signal_with_learning_cap_metadata(signal, learning_caps: dict[str, object]):
        metadata = dict(signal.metadata)
        metadata["learning_caps"] = dict(learning_caps)
        if bool(learning_caps.get("selloff_active", False)):
            metadata["selloff_learning_caps"] = dict(learning_caps)
        policy = metadata.get("regime_policy", {})
        if isinstance(policy, dict):
            policy = dict(policy)
            policy["learning_caps"] = dict(learning_caps)
            if bool(learning_caps.get("selloff_active", False)):
                policy["selloff_learning_caps"] = dict(learning_caps)
            policy["oversampling_tags"] = list(learning_caps.get("oversampling_tags", []))
            metadata["regime_policy"] = policy
            metadata["regime_policy_audit"] = policy
        return replace(signal, metadata=metadata)

    @staticmethod
    def _learning_cap_events(
        signal,
        *,
        timestamp: datetime,
        metadata: dict[str, object],
        block_reason: str | None,
    ) -> list[dict[str, object]]:
        behavior = str(metadata.get("cap_behavior_applied", "allow"))
        if behavior == "allow":
            return []
        if behavior == "warn_only":
            action = "learning_cap_warning"
        elif behavior == "reduce_size":
            action = "learning_cap_reduce_size"
        else:
            action = "learning_cap_shadow_only"
        if bool(metadata.get("selloff_active", False)):
            action = f"selloff_{action}"
        return [
            {
                "timestamp": timestamp.isoformat(),
                "symbol": signal.instrument.symbol,
                "action": action,
                "reason": block_reason or behavior,
                "metadata": {
                    "learning_caps": dict(metadata),
                    "selloff_learning_caps": dict(metadata)
                    if bool(metadata.get("selloff_active", False))
                    else {},
                },
            }
        ]

    def _learning_cap_behavior(self, name: str, *, selloff_active: bool = False) -> str:
        source = (
            getattr(getattr(self.config, "market_selloff_impulse", None), "learning_caps", None)
            if selloff_active
            else self.config.learning_caps
        )
        value = str(getattr(source, name, "warn_only"))
        if value not in {"warn_only", "shadow_only", "block", "reduce_size"}:
            return "warn_only"
        return value

    def _learning_cap_same_regime_multiplier(self) -> float:
        return max(0.0, min(1.0, float(self.config.learning_caps.same_regime_cap_multiplier)))

    @staticmethod
    def _position_policy_mode(position) -> str:
        metadata = dict(getattr(position, "entry_metadata", {}) or {})
        policy = metadata.get("regime_policy", {})
        if isinstance(policy, dict):
            return str(policy.get("decision_type", policy.get("actual_policy_decision", "")))
        return str(metadata.get("actual_policy_decision", ""))

    @staticmethod
    def _position_market_regime(position) -> str:
        metadata = dict(getattr(position, "entry_metadata", {}) or {})
        market_regime = metadata.get("market_regime", {})
        if isinstance(market_regime, dict):
            return str(market_regime.get("regime", "unknown"))
        return str(market_regime or "unknown")

    def _symbol_health(self, symbol: str, trades) -> str:
        recent = [trade for trade in reversed(trades) if trade.symbol == symbol][:6]
        if len(recent) < 3:
            return "normal"
        losses = sum(1 for trade in recent if trade.net_pnl < 0)
        expectancy = sum(float(trade.net_pnl) for trade in recent) / len(recent)
        if len(recent) >= 5 and losses >= 4 and expectancy < 0:
            return "observe_only"
        if losses >= 2 and expectancy < 0:
            return "probation"
        return "normal"

    @staticmethod
    def _quantity_after_multiplier(quantity_lots: int, multiplier: float) -> int:
        quantity = max(0, int(quantity_lots))
        bounded_multiplier = max(0.0, min(1.0, float(multiplier)))
        if quantity < 1 or bounded_multiplier <= 0.0:
            return 0
        adjusted = int(quantity * bounded_multiplier)
        return min(quantity, max(1, adjusted))

    @staticmethod
    def _signal_metadata_float(signal, key: str) -> float | None:
        try:
            return float(signal.metadata.get(key))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _policy_block_reason(policy) -> str:
        reasons = ", ".join(str(reason) for reason in policy.reasons) or "policy rejected entry"
        if policy.decision_type == PolicyDecisionType.SHADOW_ONLY.value:
            return f"entry shadowed by relaxed policy ({reasons})"
        if policy.decision_type == PolicyDecisionType.HARD_REJECT.value:
            return f"entry hard rejected by regime policy ({reasons})"
        return f"entry blocked by regime policy ({reasons})"

    @staticmethod
    def _parse_event_timestamp(value: str, *, fallback: datetime) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return fallback

    def _take_profit_activates_runner(self, broker, symbol: str, position, candle) -> bool:
        if not self.config.strategy.take_profit_activates_runner:
            return False
        stop_price = runner_breakeven_stop(
            direction=position.direction,
            entry_price=position.entry_price,
            buffer_bps=self.config.strategy.runner_breakeven_buffer_bps,
        )
        extreme_price = runner_extreme_price(
            direction=position.direction,
            current_extreme=position.runner_extreme_price,
            candle=candle,
            activation_price=position.take_profit,
        )
        return broker.activate_position_runner(
            symbol,
            timestamp=candle.timestamp,
            activation_price=position.take_profit,
            stop_price=stop_price,
            extreme_price=extreme_price,
        )

    def _update_runner_extreme(self, broker, symbol: str, position, candle) -> None:
        if not position.runner_active:
            return
        extreme_price = runner_extreme_price(
            direction=position.direction,
            current_extreme=position.runner_extreme_price,
            candle=candle,
            activation_price=position.runner_activation_price or position.take_profit,
        )
        broker.update_position_runner_extreme(
            symbol,
            timestamp=candle.timestamp,
            extreme_price=extreme_price,
        )

    def _signal_feedback_runner_kwargs(self) -> dict[str, object]:
        strategy = self.config.strategy
        return {
            "runner_enabled": strategy.take_profit_activates_runner,
            "runner_breakeven_buffer_bps": strategy.runner_breakeven_buffer_bps,
            "runner_trailing_atr_multiple": strategy.runner_trailing_atr_multiple,
            "runner_profit_lock_ratio": strategy.runner_profit_lock_ratio,
            "runner_atr_window": strategy.atr_window,
        }

    def _entry_confirmation_block_reason(self, signal) -> str | None:
        if self._relaxed_learning_enabled():
            return None
        confirmation = signal.metadata.get("entry_confirmation", {})
        if not isinstance(confirmation, dict):
            return None
        if not confirmation.get("available"):
            return None
        if confirmation.get("against_direction"):
            timeframe = confirmation.get("timeframe", "lower timeframe")
            reason = str(confirmation.get("reason", "against entry direction"))
            return f"entry blocked by {timeframe} confirmation ({reason})"
        return None

    def _signal_with_entry_microstructure(self, provider, signal, *, quantity_lots: int):
        metadata = dict(signal.metadata)
        snapshot = self._entry_microstructure_snapshot(provider, signal, quantity_lots=quantity_lots)
        metadata["microstructure"] = snapshot
        return replace(signal, metadata=metadata)

    def _entry_microstructure_snapshot(self, provider, signal, *, quantity_lots: int) -> dict[str, object]:
        if not hasattr(provider, "get_order_book_snapshot"):
            return {
                "available": False,
                "reason": "provider has no order book support",
            }
        try:
            snapshot = provider.get_order_book_snapshot(
                signal.instrument,
                depth=self.config.strategy.order_book_depth,
                quantity_lots=quantity_lots,
                direction=signal.direction.value,
            )
        except Exception as exc:  # pragma: no cover - live broker/API dependent
            LOGGER.warning("Order book snapshot failed for %s: %s", signal.instrument.symbol, exc)
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        return dict(snapshot)

    def _microstructure_block_reason(self, signal) -> str | None:
        if self._relaxed_learning_enabled():
            return None
        microstructure = dict(signal.metadata.get("microstructure", {}))
        if not microstructure.get("available"):
            if self.config.strategy.require_order_book:
                return "entry blocked by missing order book"
            return None

        max_spread = float(self.config.strategy.max_entry_spread_bps)
        spread = float(microstructure.get("spread_bps", 0.0))
        if max_spread > 0 and spread > max_spread:
            return f"entry blocked by wide spread ({spread:.2f} bps)"

        min_cover = float(self.config.strategy.min_entry_liquidity_cover)
        cover = float(microstructure.get("entry_liquidity_cover", 0.0))
        if min_cover > 0 and cover < min_cover:
            return f"entry blocked by thin order book (cover {cover:.2f}x)"

        min_imbalance = float(self.config.strategy.min_entry_book_imbalance)
        side_imbalance = float(microstructure.get("side_imbalance", 0.0))
        if side_imbalance < min_imbalance:
            return f"entry blocked by adverse book imbalance ({side_imbalance:.2f})"
        return None

    def _evaluate_strategy_test_window(
        self,
        *,
        strategy_config: StrategySection,
        grouped: dict[str, dict[str, list]],
        instruments_by_symbol: dict[str, object],
        symbols: list[str],
        train_months: list[str],
        test_months: list[str],
    ) -> dict[str, float | int]:
        _, summary = self._evaluate_strategy_test_window_bundle(
            strategy_config=strategy_config,
            grouped=grouped,
            instruments_by_symbol=instruments_by_symbol,
            symbols=symbols,
            train_months=train_months,
            test_months=test_months,
        )
        return summary

    def _evaluate_strategy_test_window_bundle(
        self,
        *,
        strategy_config: StrategySection,
        grouped: dict[str, dict[str, list]],
        instruments_by_symbol: dict[str, object],
        symbols: list[str],
        train_months: list[str],
        test_months: list[str],
    ) -> tuple[object, dict[str, float | int]]:
        selected_symbols = [symbol for symbol in symbols if symbol in instruments_by_symbol]
        combined_months = tuple([*train_months, *test_months])
        combined_bundle = _slice_grouped_candles(grouped, combined_months)
        test_bundle = _slice_grouped_candles(grouped, tuple(test_months))
        selected_candles = {symbol: combined_bundle[symbol] for symbol in selected_symbols}
        selected_instruments = {
            symbol: instruments_by_symbol[symbol] for symbol in selected_symbols
        }
        test_start_at = min(
            candle.timestamp
            for symbol in selected_symbols
            for candle in test_bundle.get(symbol, [])
        )
        engine = BacktestEngine(
            strategy=TrendFollowingStrategy(strategy_config, timeframe=self.config.data.timeframe),
            risk_manager=self._risk_manager(),
            backtest=self.config.backtest,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
        )
        combined_result = engine.run_with_instruments(
            selected_candles,
            selected_instruments,
            trade_start_at=test_start_at,
        )
        test_result = _trim_backtest_result(combined_result, test_start_at)
        summary = compute_summary(test_result, timeframe=self.config.data.timeframe)
        summary["normalized_monthly_return_pct"] = round(
            _normalized_monthly_return_pct(
                float(summary["total_return_pct"]),
                len(test_months),
            ),
            3,
        )
        return test_result, summary


def _paper_report_view(payload: dict[str, object]) -> dict[str, object]:
    summary = payload.get("summary", {})
    comparison = payload.get("comparison_to_previous_window", {})
    delta = comparison.get("delta", {})
    return {
        "output_dir": payload.get("output_dir", ""),
        "period": payload.get("period", {}),
        "portfolio": payload.get("portfolio", {}),
        "summary": {
            "trades": summary.get("trades", 0),
            "net_pnl_rub": summary.get("net_pnl_rub", 0.0),
            "win_rate_pct": summary.get("win_rate_pct", 0.0),
            "profit_factor": summary.get("profit_factor", 0.0),
            "expectancy_rub": summary.get("expectancy_rub", 0.0),
        },
        "comparison_delta": {
            "trades": delta.get("trades", 0.0),
            "net_pnl_rub": delta.get("net_pnl_rub", 0.0),
        },
    }


def _trade_review_view(payload: dict[str, object]) -> dict[str, object]:
    summary = dict(payload.get("summary", {}))
    return {
        "output_dir": payload.get("output_dir", ""),
        "latest_path": payload.get("latest_path", ""),
        "reviewed_trades": payload.get("reviewed_trades", 0),
        "total_closed_trades": payload.get("total_closed_trades", 0),
        "summary": {
            "net_pnl_rub": summary.get("net_pnl_rub", 0.0),
            "win_rate_pct": summary.get("win_rate_pct", 0.0),
            "expectancy_rub": summary.get("expectancy_rub", 0.0),
            "mistake_trades": summary.get("mistake_trades", 0),
        },
        "mistake_breakdown": payload.get("mistake_breakdown", {}),
        "config_patch_candidates": payload.get("config_patch_candidates", {}),
        "short_only_review": payload.get("short_only_review", {}),
        "recommendations": payload.get("recommendations", []),
    }


def _summarize_signal_activity(events: list[dict[str, object]]) -> dict[str, object]:
    signal_events = [event for event in events if event.get("action") == "signal"]
    approved_signals = [event for event in signal_events if bool(event.get("approved", False))]
    rejected_signals = [event for event in signal_events if not bool(event.get("approved", False))]
    rejection_reasons = Counter(
        str(event.get("reason", ""))
        for event in rejected_signals
        if str(event.get("reason", "")).strip()
    )
    return {
        "signals_total": len(signal_events),
        "signals_approved": len(approved_signals),
        "signals_rejected": len(rejected_signals),
        "signal_rejection_reason_breakdown": dict(sorted(rejection_reasons.items())),
    }


def _summarize_short_only_activity(
    events: list[dict[str, object]],
    portfolio,
    marks: dict[str, float],
) -> dict[str, object]:
    candidate_events = [event for event in events if event.get("action") == "short_only_short_candidate"]
    signal_events = [
        event
        for event in events
        if event.get("action") == "signal"
        and _event_is_short_only_signal(event)
    ]
    approved = [event for event in signal_events if bool(event.get("approved", False))]
    underallocated = [event for event in events if event.get("action") == "short_only_underallocated"]
    equity = portfolio.equity(marks)
    gross = portfolio.gross_exposure(marks)
    return {
        "enabled": any(event.get("action") == "short_only_cycle_start" for event in events),
        "long_signals_ignored": sum(1 for event in events if event.get("action") == "long_signal_ignored_short_only"),
        "longs_flattened": sum(1 for event in events if event.get("action") == "long_position_flattened_short_only"),
        "no_trade_range_chop_count": sum(1 for event in events if event.get("action") == "range_chop_no_trade_short_only"),
        "short_candidates_total": len(candidate_events),
        "positive_ev_short_candidates": sum(1 for event in candidate_events if bool(event.get("edge_gate_passed", False))),
        "shorts_opened": len(approved),
        "shorts_blocked_hard": len(signal_events) - len(approved),
        "gross_exposure_pct": round(gross / equity, 6) if equity > 0 else 0.0,
        "underallocated_count": len(underallocated),
    }


def _event_is_short_only_signal(event: dict[str, object]) -> bool:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    short_only = metadata.get("short_only", {})
    return isinstance(short_only, dict) and bool(short_only.get("enabled", False))


def _bounded_multiplier(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _entry_schedule_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "evidence_source": payload.get("evidence_source", ""),
        "evidence_counts": payload.get("evidence_counts", {}),
        "changed": payload.get("changed", False),
        "reason": payload.get("reason", ""),
        "current_hours": payload.get("current_hours", []),
        "proposed_hours": payload.get("proposed_hours", []),
        "additions": payload.get("additions", []),
        "removals": payload.get("removals", []),
    }


def _entry_symbols_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "evidence_source": payload.get("evidence_source", ""),
        "evidence_counts": payload.get("evidence_counts", {}),
        "changed": payload.get("changed", False),
        "reason": payload.get("reason", ""),
        "current_blocked_symbols": payload.get("current_blocked_symbols", []),
        "proposed_blocked_symbols": payload.get("proposed_blocked_symbols", []),
        "additions": payload.get("additions", []),
        "current_blocked_long_symbols": payload.get("current_blocked_long_symbols", []),
        "proposed_blocked_long_symbols": payload.get("proposed_blocked_long_symbols", []),
        "long_additions": payload.get("long_additions", []),
        "current_blocked_short_symbols": payload.get("current_blocked_short_symbols", []),
        "proposed_blocked_short_symbols": payload.get("proposed_blocked_short_symbols", []),
        "short_additions": payload.get("short_additions", []),
    }


def _entry_quality_view(payload: dict[str, object]) -> dict[str, object]:
    lookback = payload.get("lookback", {})
    return {
        "output_dir": payload.get("output_dir", ""),
        "evidence_source": payload.get("evidence_source", ""),
        "evidence_counts": payload.get("evidence_counts", {}),
        "changed": payload.get("changed", False),
        "reason": payload.get("reason", ""),
        "current_min_signal_strength": payload.get("current_min_signal_strength", 0.0),
        "recommended_min_signal_strength": payload.get("recommended_min_signal_strength", 0.0),
        "eligible_trades": lookback.get("eligible_trades", 0),
    }


def _feedback_bootstrap_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "generated_total": payload.get("generated_total", 0),
        "generated_by_symbol": payload.get("generated_by_symbol", {}),
        "resolved_signals": payload.get("resolved_signals", 0),
        "pending_signals": payload.get("pending_signals", 0),
    }


def _runtime_universe_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "changed": payload.get("changed", False),
        "reason": payload.get("reason", ""),
        "configured_symbols": payload.get("configured_symbols", []),
        "current_allowed_symbols": payload.get("current_allowed_symbols", []),
        "proposed_allowed_symbols": payload.get("proposed_allowed_symbols", []),
        "proposed_effective_symbols": payload.get("proposed_effective_symbols", []),
        "optimizer_best_symbols": payload.get("optimizer_best_symbols", []),
        "walk_forward_latest_symbols": payload.get("walk_forward_latest_symbols", []),
        "consensus_symbols": payload.get("consensus_symbols", []),
    }


def _optimizer_view(payload: dict[str, object]) -> dict[str, object]:
    best = payload.get("best_candidate") or {}
    summary = best.get("summary", {}) if isinstance(best, dict) else {}
    return {
        "output_dir": payload.get("output_dir", ""),
        "evaluated_candidates": payload.get("evaluated_candidates", 0),
        "best_candidate": {
            "symbols": best.get("symbols", []),
            "style": best.get("style", ""),
            "score": best.get("score", 0.0),
            "total_return_pct": summary.get("total_return_pct", 0.0),
            "avg_monthly_return_pct": summary.get("avg_monthly_return_pct", 0.0),
            "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
            "profit_factor": summary.get("profit_factor", 0.0),
            "trades": summary.get("trades", 0),
        },
    }


def _walk_forward_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "config": payload.get("config", {}),
        "summary": payload.get("summary", {}),
        "available_months": payload.get("available_months", []),
        "skipped_folds": payload.get("skipped_folds", 0),
    }


def _monte_carlo_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "target": payload.get("target", {}),
        "backtest_summary": payload.get("backtest_summary", {}),
        "monte_carlo_summary": payload.get("monte_carlo", {}).get("summary", {}),
    }


def _strategy_tuning_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "changed": payload.get("changed", False),
        "reason": payload.get("reason", ""),
        "patch_values": payload.get("patch_values", {}),
        "comparison": payload.get("comparison", {}),
    }


def _exit_tuning_view(payload: dict[str, object]) -> dict[str, object]:
    return {
        "output_dir": payload.get("output_dir", ""),
        "changed": payload.get("changed", False),
        "reason": payload.get("reason", ""),
        "patch_values": payload.get("patch_values", {}),
        "comparison": payload.get("comparison", {}),
    }


def _effective_config_view(payload: dict[str, object]) -> dict[str, object]:
    sources = []
    for source in payload.get("sources", []):
        activation = source.get("activation", {})
        sources.append(
            {
                "source": source.get("source", ""),
                "changed": source.get("changed", False),
                "selected_values": source.get("selected_values", {}),
                "activation": activation,
            }
        )
    return {
        "output_dir": payload.get("output_dir", ""),
        "source_config_path": payload.get("source_config_path", ""),
        "effective_config_path": payload.get("effective_config_path", ""),
        "paper_only_mode": payload.get("paper_only_mode", ""),
        "allow_live_trading": payload.get("allow_live_trading", False),
        "applied_strategy_overrides": payload.get("applied_strategy_overrides", {}),
        "rollback_guardrail": payload.get("rollback_guardrail", {}),
        "sources": sources,
    }


def _render_nightly_autonomy_markdown(payload: dict[str, object]) -> str:
    analysis = payload["analysis"]["paper_report"]
    trade_review = payload["analysis"]["trade_review"]
    restrictions = payload["restrictions"]
    research = payload["research"]
    tuning = payload["tuning"]
    universe = payload["runtime"]["universe_selection"]
    runtime = payload["runtime"]["effective_config"]
    lines = [
        "# Nightly Autonomy",
        "",
        f"- Commit: {payload.get('commit_hash', 'unknown')}",
        f"- Active config: {payload['active_config_path']}",
        f"- Base config: {payload['base_config_path']}",
        f"- Effective output: {payload['effective_output_path']}",
        "",
        "## Analyze",
        f"- Trades: {analysis['summary']['trades']}",
        f"- Net PnL: {analysis['summary']['net_pnl_rub']} RUB",
        f"- Profit factor: {analysis['summary']['profit_factor']}",
        f"- Trade-review mistakes: {trade_review['summary']['mistake_trades']}",
        f"- Trade-review patch candidates: {trade_review['config_patch_candidates']}",
        "",
        "## Restrictions",
        f"- Entry hours changed: {restrictions['entry_schedule']['changed']} ({restrictions['entry_schedule']['reason']}, source={restrictions['entry_schedule']['evidence_source']})",
        f"- Entry symbols changed: {restrictions['entry_symbols']['changed']} ({restrictions['entry_symbols']['reason']}, source={restrictions['entry_symbols']['evidence_source']})",
        f"- Directional symbol blocks: long={restrictions['entry_symbols']['proposed_blocked_long_symbols']} short={restrictions['entry_symbols']['proposed_blocked_short_symbols']}",
        f"- Entry quality changed: {restrictions['entry_quality']['changed']} ({restrictions['entry_quality']['reason']})",
        f"- Signal feedback resolved: {restrictions['signal_feedback_bootstrap']['resolved_signals']}",
        "",
        "## Research",
        f"- Optimizer candidates: {research['optimizer']['evaluated_candidates']}",
        f"- Walk-forward folds: {research['walk_forward']['summary'].get('folds_evaluated', 0)}",
        f"- Monte Carlo positive probability: {research['monte_carlo']['monte_carlo_summary'].get('probability_positive_pct', 0.0)}%",
        "",
        "## Tuning",
        f"- Strategy candidate accepted: {tuning['strategy']['changed']}",
        f"- Exit candidate accepted: {tuning['exits']['changed']}",
        "",
        "## Runtime",
        f"- Universe changed: {universe['changed']} ({universe['reason']})",
        f"- Proposed effective symbols: {universe['proposed_effective_symbols']}",
        f"- Rollback guardrail: {runtime['rollback_guardrail'].get('rollback_to_base', False)} ({runtime['rollback_guardrail'].get('reason', '')})",
        f"- Active override keys: {sorted(runtime['applied_strategy_overrides'])}",
        "",
    ]
    return "\n".join(lines)
