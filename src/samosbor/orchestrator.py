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
from .autonomy.adaptive_entry import (
    adaptive_entry_block_reason,
    build_adaptive_entry_context,
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
    COMMISSION_EDGE_TAG,
    LATE_REENTRY_TAG,
    LOW_QUALITY_PROBABILITY_THRESHOLD,
    LOW_QUALITY_TAG,
    NEGATIVE_EXPECTANCY_TAG,
    SHORT_AFTER_EXHAUSTION_TAG,
    CONFIRMATION_AFTER_IMPULSE_TAG,
    assess_signal_learning,
    build_entry_candle_context,
    build_setup_learning_tags,
)
from .autonomy.signal_feedback import (
    backfill_signal_feedback_for_symbol,
    build_trade_evidence,
    default_signal_horizon_bars,
    load_signal_feedback,
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
from .domain import ExitReason
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
        return TrendFollowingStrategy(
            self.config.strategy,
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
        resolve_pending_signals(signal_feedback, history)

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
                signal = self._signal_with_adaptive_entry_context(signal, candles, broker.events)
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
                    self._record_blocked_signal_feedback(
                        signal_feedback,
                        signal,
                        timestamp=latest.timestamp,
                        reason=entry_block_reason,
                    )
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
                adaptive_block_reason = adaptive_entry_block_reason(signal)
                if adaptive_block_reason is not None:
                    self._record_blocked_signal_feedback(
                        signal_feedback,
                        signal,
                        timestamp=latest.timestamp,
                        reason=adaptive_block_reason,
                    )
                    cycle_events.append(
                        {
                            "timestamp": latest.timestamp.isoformat(),
                            "symbol": instrument.symbol,
                            "action": "signal",
                            "approved": False,
                            "reason": adaptive_block_reason,
                            "direction": signal.direction.value,
                            "strength": signal.strength,
                            "quantity_lots": 0,
                            "metadata": dict(signal.metadata),
                        }
                    )
                    continue
                decision = risk_manager.approve(broker.portfolio, signal, marks, broker.trades)
                signal_for_entry = signal
                microstructure_block_reason = None
                if decision.approved:
                    microstructure_block_reason = self._entry_confirmation_block_reason(signal_for_entry)
                    if microstructure_block_reason is None:
                        signal_for_entry = self._signal_with_entry_microstructure(
                            provider,
                            signal,
                            quantity_lots=decision.quantity_lots,
                        )
                        signal_for_entry = self._signal_with_learning_assessment(
                            signal_for_entry,
                            signal_feedback,
                            timestamp=latest.timestamp,
                            quantity_lots=decision.quantity_lots,
                        )
                        microstructure_block_reason = self._microstructure_block_reason(signal_for_entry)
                        if microstructure_block_reason is None:
                            microstructure_block_reason = self._learning_entry_block_reason(
                                signal_for_entry,
                                quantity_lots=decision.quantity_lots,
                            )
                    if microstructure_block_reason is None:
                        signal_for_entry, decision = self._apply_learning_size_adjustment(
                            signal_for_entry,
                            decision,
                        )

                event = {
                    "timestamp": latest.timestamp.isoformat(),
                    "symbol": instrument.symbol,
                    "action": "signal",
                    "approved": decision.approved and microstructure_block_reason is None,
                    "reason": microstructure_block_reason or decision.reason,
                    "direction": signal.direction.value,
                    "strength": signal.strength,
                    "quantity_lots": decision.quantity_lots if microstructure_block_reason is None else 0,
                    "metadata": dict(signal_for_entry.metadata),
                }
                cycle_events.append(event)
                if not event["approved"]:
                    self._record_blocked_signal_feedback(
                        signal_feedback,
                        signal_for_entry,
                        timestamp=latest.timestamp,
                        reason=str(event["reason"]),
                        quantity_lots=max(1, int(decision.quantity_lots or 1)),
                    )
                if decision.approved:
                    if microstructure_block_reason is None:
                        broker.open_position(signal_for_entry, decision.quantity_lots, latest.timestamp)

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
        write_json_payload(output_dir / "cycle_summary.json", summary)
        write_json_payload(output_dir / "cycle_events.json", {"events": cycle_events})
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

    def _signal_with_adaptive_entry_context(self, signal, candles, prior_events):
        metadata = dict(signal.metadata)
        metadata["adaptive_entry"] = build_adaptive_entry_context(
            signal,
            candles,
            self.config.strategy,
            prior_events=prior_events,
            timeframe=self.config.data.timeframe,
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

    def _record_blocked_signal_feedback(
        self,
        feedback_payload,
        signal,
        *,
        timestamp: datetime,
        reason: str,
        quantity_lots: int = 1,
    ) -> None:
        metadata = dict(signal.metadata)
        metadata["runtime_feedback"] = {
            "source": "runtime-blocked-signal",
            "blocked": True,
            "reason": reason,
        }
        record_shadow_signal(
            feedback_payload,
            replace(signal, metadata=metadata),
            timestamp=timestamp,
            horizon_bars=default_signal_horizon_bars(self.config.data.timeframe),
            quantity_lots=max(1, int(quantity_lots or 1)),
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps,
            **self._signal_feedback_runner_kwargs(),
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

    def _apply_learning_size_adjustment(self, signal, decision):
        metadata = dict(signal.metadata)
        learning = metadata.get("ml_learning", {})
        if not isinstance(learning, dict) or not learning.get("available"):
            return signal, decision

        tags = set(str(tag) for tag in learning.get("learning_tags", []) if str(tag))
        should_reduce = (
            LOW_QUALITY_TAG in tags
            or NEGATIVE_EXPECTANCY_TAG in tags
            or float(learning.get("probability_profit", 1.0) or 1.0) < LOW_QUALITY_PROBABILITY_THRESHOLD
        )
        if not should_reduce or decision.quantity_lots <= 1:
            return signal, decision

        original_quantity = int(decision.quantity_lots)
        adjusted_quantity = max(1, original_quantity // 2)
        if adjusted_quantity >= original_quantity:
            return signal, decision

        ratio = adjusted_quantity / original_quantity
        metadata["learning_size_adjustment"] = {
            "reason": "low-quality ML setup",
            "original_quantity_lots": original_quantity,
            "adjusted_quantity_lots": adjusted_quantity,
            "factor": round(ratio, 4),
        }
        adjusted_decision = replace(
            decision,
            quantity_lots=adjusted_quantity,
            estimated_notional_rub=decision.estimated_notional_rub * ratio,
        )
        return replace(signal, metadata=metadata), adjusted_decision

    def _learning_entry_block_reason(self, signal, *, quantity_lots: int) -> str | None:
        metadata = dict(signal.metadata)
        entry_candle = metadata.get("entry_candle", {})
        if not isinstance(entry_candle, dict):
            return None
        learning = metadata.get("ml_learning", {})
        learning_tags = set()
        setup_tags = set(str(tag) for tag in metadata.get("setup_learning_tags", []) if str(tag))
        if isinstance(learning, dict):
            learning_tags = set(str(tag) for tag in learning.get("learning_tags", []) if str(tag))
        learning_tags.update(setup_tags)

        if SHORT_AFTER_EXHAUSTION_TAG in learning_tags:
            return "entry waits for next candle after short exhaustion"

        direction_confirmed = bool(entry_candle.get("direction_confirmed_by_close"))

        if not direction_confirmed and CONFIRMATION_AFTER_IMPULSE_TAG in learning_tags:
            return "entry waits for confirmation after impulse"

        if isinstance(learning, dict) and learning.get("available") and learning.get("blocks_entry"):
            probability = float(learning.get("probability_profit", 1.0) or 1.0)
            expected_position = float(learning.get("expected_pnl_position_rub", 0.0) or 0.0)
            required_edge = float(learning.get("required_net_edge_rub", 0.0) or 0.0)
            if probability < LOW_QUALITY_PROBABILITY_THRESHOLD or LOW_QUALITY_TAG in learning_tags:
                return (
                    "entry blocked by low ML probability "
                    f"({probability:.2f} < {LOW_QUALITY_PROBABILITY_THRESHOLD:.2f})"
                )
            if expected_position < 0 or NEGATIVE_EXPECTANCY_TAG in learning_tags:
                return f"entry blocked by negative ML expectancy ({expected_position:.2f} RUB)"
            if COMMISSION_EDGE_TAG in learning_tags:
                return (
                    "entry blocked by ML edge below commission floor "
                    f"({expected_position:.2f} <= {required_edge:.2f} RUB)"
                )

        if not direction_confirmed and LATE_REENTRY_TAG in learning_tags:
            probability = float(learning.get("probability_profit", 0.0) or 0.0) if isinstance(learning, dict) else 0.0
            if probability < 0.50:
                return "entry waits for late reentry confirmation"
        return None

    def _learning_confirmation_block_reason(self, signal) -> str | None:
        return self._learning_entry_block_reason(signal, quantity_lots=1)

    def _entry_confirmation_block_reason(self, signal) -> str | None:
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
