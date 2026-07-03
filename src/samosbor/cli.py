from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .logging_utils import configure_logging
from .orchestrator import TradingOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MOEX AI Trader paper-trading system")
    parser.add_argument(
        "--config",
        default="configs/server_tbank_stocks_intraday_300k_focused.toml",
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("accounts", help="List T-Bank accounts visible to the configured token")
    subparsers.add_parser("backtest", help="Run a historical backtest")
    subparsers.add_parser("paper-cycle", help="Run one paper-trading cycle")
    microstructure_parser = subparsers.add_parser(
        "collect-microstructure",
        help="Continuously collect order-book microstructure for every configured instrument",
    )
    microstructure_parser.add_argument("--interval-sec", type=float, default=15.0)
    microstructure_parser.add_argument("--depth", type=int, default=0)
    microstructure_parser.add_argument("--output-dir", default="")
    microstructure_parser.add_argument("--once", action="store_true")
    microstructure_parser.add_argument("--cycles", type=int, default=0)
    refresh_effective_parser = subparsers.add_parser(
        "refresh-effective-config",
        help="Build the derived paper runtime config from the latest autotune artifacts",
    )
    refresh_effective_parser.add_argument(
        "--output",
        help="Optional path for the generated effective config, defaults next to the input config",
    )
    nightly_parser = subparsers.add_parser(
        "nightly-autonomy",
        help="Run the full nightly analyze/research/optimizer/restrictions autonomy cycle",
    )
    nightly_parser.add_argument(
        "--base-config",
        required=True,
        help="Base runtime config used to rebuild the effective config at the end of the nightly cycle",
    )
    nightly_parser.add_argument(
        "--effective-output",
        required=True,
        help="Path to the effective runtime config that nightly autonomy should refresh",
    )
    paper_report_parser = subparsers.add_parser("paper-report", help="Build a summary from paper-trading state")
    paper_report_parser.add_argument("--days", type=int, default=1, help="Lookback window in days")
    paper_report_parser.add_argument("--date", help="Anchor ISO date in report timezone, defaults to today")
    paper_report_parser.add_argument("--timezone", help="IANA timezone override, defaults to config app.timezone")
    trade_review_parser = subparsers.add_parser("trade-review", help="Analyze closed paper trades for mistakes")
    trade_review_parser.add_argument(
        "--lookback-trades",
        type=int,
        default=100,
        help="Number of latest closed trades to review",
    )
    subparsers.add_parser("optimize", help="Search parameter sets and instrument subsets")
    subparsers.add_parser("monte-carlo", help="Run Monte Carlo robustness analysis on a fresh backtest")
    walk_forward_parser = subparsers.add_parser(
        "walk-forward",
        help="Run rolling walk-forward validation with re-optimization",
    )
    walk_forward_parser.add_argument(
        "--adaptive-history",
        action="store_true",
        help="Shrink the walk-forward window to fit the available monthly history when needed",
    )
    bootstrap_feedback_parser = subparsers.add_parser(
        "bootstrap-entry-feedback",
        help="Backfill the shadow signal feedback journal from recent historical candles",
    )
    bootstrap_feedback_parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Rebuild the signal feedback journal from scratch",
    )
    bootstrap_feedback_parser.add_argument(
        "--max-signals-per-symbol",
        type=int,
        default=0,
        help="Optional cap on generated resolved signals per symbol, 0 means no cap",
    )
    tune_schedule_parser = subparsers.add_parser(
        "tune-entry-hours",
        help="Recommend safer entry hours from recent paper-trading results",
    )
    tune_schedule_parser.add_argument("--days", type=int, default=45, help="Lookback window in days")
    tune_schedule_parser.add_argument("--date", help="Anchor ISO date in report timezone, defaults to today")
    tune_schedule_parser.add_argument("--timezone", help="IANA timezone override, defaults to config app.timezone")
    tune_schedule_parser.add_argument(
        "--min-trades-per-hour",
        type=int,
        default=3,
        help="Minimum closed trades per hour before considering a schedule change",
    )
    tune_schedule_parser.add_argument(
        "--max-hours-to-add",
        type=int,
        default=2,
        help="Maximum number of positive hours to add in one recommendation pass",
    )
    tune_schedule_parser.add_argument(
        "--max-hours-to-remove",
        type=int,
        default=2,
        help="Maximum number of negative hours to remove in one recommendation pass",
    )
    tune_entry_symbols_parser = subparsers.add_parser(
        "tune-entry-symbols",
        help="Recommend safer blocked_symbols from recent paper-trading results",
    )
    tune_entry_symbols_parser.add_argument("--days", type=int, default=45, help="Lookback window in days")
    tune_entry_symbols_parser.add_argument("--date", help="Anchor ISO date in report timezone, defaults to today")
    tune_entry_symbols_parser.add_argument("--timezone", help="IANA timezone override, defaults to config app.timezone")
    tune_entry_symbols_parser.add_argument(
        "--min-trades-per-symbol",
        type=int,
        default=8,
        help="Minimum closed trades per symbol before considering a restriction change",
    )
    tune_entry_symbols_parser.add_argument(
        "--max-symbols-to-block",
        type=int,
        default=0,
        help="Maximum number of new weak symbols to block in one recommendation pass",
    )
    tune_entry_symbols_parser.add_argument(
        "--max-total-blocked-symbols",
        type=int,
        default=4,
        help="Maximum total blocked symbols after applying the recommendation",
    )
    tune_entry_symbols_parser.add_argument(
        "--min-trades-per-direction-symbol",
        type=int,
        default=8,
        help="Minimum paper trades per symbol+direction before blocking only long or only short",
    )
    tune_entry_symbols_parser.add_argument(
        "--max-long-symbols-to-block",
        type=int,
        default=0,
        help="Maximum number of long-direction symbol blocks to add in one recommendation pass",
    )
    tune_entry_symbols_parser.add_argument(
        "--max-short-symbols-to-block",
        type=int,
        default=0,
        help="Maximum number of short-direction symbol blocks to add in one recommendation pass",
    )
    tune_entry_symbols_parser.add_argument(
        "--max-total-blocked-long-symbols",
        type=int,
        default=4,
        help="Maximum total blocked_long_symbols after applying the recommendation",
    )
    tune_entry_symbols_parser.add_argument(
        "--max-total-blocked-short-symbols",
        type=int,
        default=4,
        help="Maximum total blocked_short_symbols after applying the recommendation",
    )
    tune_entry_quality_parser = subparsers.add_parser(
        "tune-entry-quality",
        help="Recommend a safer min_signal_strength from recent paper-trading results",
    )
    tune_entry_quality_parser.add_argument(
        "--lookback-trades",
        type=int,
        default=40,
        help="Number of latest closed paper trades to analyze",
    )
    tune_entry_quality_parser.add_argument(
        "--min-trades",
        type=int,
        default=8,
        help="Minimum number of signal-tagged trades required before suggesting a patch",
    )
    tune_entry_quality_parser.add_argument(
        "--min-trade-retention-ratio",
        type=float,
        default=0.5,
        help="Minimum retained trade ratio after raising the threshold",
    )
    tune_entry_quality_parser.add_argument(
        "--min-expectancy-improvement-rub",
        type=float,
        default=0.0,
        help="Minimum expectancy improvement required before suggesting a patch",
    )
    tune_entry_quality_parser.add_argument(
        "--bucket-step",
        type=float,
        default=0.05,
        help="Step used to test candidate signal-strength thresholds",
    )
    tune_universe_parser = subparsers.add_parser(
        "tune-universe",
        help="Recommend an active runtime symbol subset from optimizer and walk-forward consensus",
    )
    tune_universe_parser.add_argument(
        "--max-allowed-symbols",
        type=int,
        help="Maximum number of active symbols to keep in the runtime universe",
    )
    tune_universe_parser.add_argument(
        "--min-walk-forward-positive-probability-pct",
        type=float,
        default=55.0,
        help="Minimum walk-forward positive-fold probability required before changing the active universe",
    )
    tune_universe_parser.add_argument(
        "--min-latest-fold-monthly-return-pct",
        type=float,
        default=0.0,
        help="Minimum normalized monthly return required on the latest walk-forward fold",
    )
    tune_universe_parser.add_argument(
        "--min-walk-forward-folds",
        type=int,
        default=3,
        help="Minimum number of walk-forward folds required before changing the active universe",
    )
    tune_universe_parser.add_argument(
        "--min-latest-fold-trades",
        type=int,
        default=4,
        help="Minimum number of latest-fold test trades required before changing the active universe",
    )
    tune_universe_parser.add_argument(
        "--skip-optimizer-overlap",
        action="store_true",
        help="Allow the latest walk-forward winner to drive the universe even without optimizer overlap",
    )
    tune_strategy_parser = subparsers.add_parser(
        "tune-strategy",
        help="Recommend safer strategy parameters from recent walk-forward results",
    )
    tune_strategy_parser.add_argument(
        "--min-monthly-improvement-pct",
        type=float,
        default=0.05,
        help="Minimum latest OOS monthly improvement required before suggesting a patch",
    )
    tune_strategy_parser.add_argument(
        "--max-extra-drawdown-pct",
        type=float,
        default=1.0,
        help="Maximum tolerated increase in latest OOS drawdown versus the current strategy",
    )
    tune_strategy_parser.add_argument(
        "--min-positive-fold-probability-pct",
        type=float,
        default=55.0,
        help="Minimum walk-forward positive-fold probability required before suggesting a patch",
    )
    tune_exits_parser = subparsers.add_parser(
        "tune-exits",
        help="Recommend safer exit parameters from recent walk-forward results",
    )
    tune_exits_parser.add_argument(
        "--min-monthly-improvement-pct",
        type=float,
        default=0.03,
        help="Minimum latest OOS monthly improvement required before suggesting an exit patch",
    )
    tune_exits_parser.add_argument(
        "--max-extra-drawdown-pct",
        type=float,
        default=1.0,
        help="Maximum tolerated increase in latest OOS drawdown versus current exits",
    )
    tune_exits_parser.add_argument(
        "--min-positive-fold-probability-pct",
        type=float,
        default=55.0,
        help="Minimum walk-forward positive-fold probability required before suggesting exit changes",
    )

    sandbox_parser = subparsers.add_parser("sandbox-init", help="Create/fund a sandbox account")
    sandbox_parser.add_argument("--fund-rub", type=float, default=1_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    log_path = config.resolve_path(config.reporting.output_dir) / "logs" / "samosbor.log"
    configure_logging(log_path, verbose=args.verbose)
    orchestrator = TradingOrchestrator(config)

    if args.command == "accounts":
        print(json.dumps(orchestrator.list_accounts(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "backtest":
        print(json.dumps(orchestrator.run_backtest(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "paper-cycle":
        print(json.dumps(orchestrator.run_paper_cycle(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "collect-microstructure":
        from .microstructure_collector import run_microstructure_collector

        cycles = 1 if args.once else (args.cycles if args.cycles > 0 else None)
        print(
            json.dumps(
                run_microstructure_collector(
                    config,
                    interval_sec=args.interval_sec,
                    depth=args.depth or None,
                    output_dir=args.output_dir or None,
                    cycles=cycles,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "refresh-effective-config":
        print(
            json.dumps(
                orchestrator.refresh_effective_config(
                    source_config_path=args.config,
                    output_path=args.output,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "nightly-autonomy":
        print(
            json.dumps(
                orchestrator.run_nightly_autonomy(
                    active_config_path=args.config,
                    base_config_path=args.base_config,
                    effective_output_path=args.effective_output,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "paper-report":
        print(
            json.dumps(
                orchestrator.run_paper_report(
                    days=args.days,
                    report_date=args.date,
                    timezone_name=args.timezone,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "trade-review":
        print(
            json.dumps(
                orchestrator.run_trade_review(lookback_trades=args.lookback_trades),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "optimize":
        print(json.dumps(orchestrator.optimize_strategy(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "monte-carlo":
        print(json.dumps(orchestrator.run_monte_carlo(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "walk-forward":
        print(
            json.dumps(
                orchestrator.run_walk_forward(adaptive_history=args.adaptive_history),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "bootstrap-entry-feedback":
        print(
            json.dumps(
                orchestrator.bootstrap_entry_feedback(
                    replace_existing=args.replace_existing,
                    max_signals_per_symbol=args.max_signals_per_symbol,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "tune-entry-hours":
        print(
            json.dumps(
                orchestrator.tune_entry_schedule(
                    lookback_days=args.days,
                    report_date=args.date,
                    timezone_name=args.timezone,
                    min_trades_per_hour=args.min_trades_per_hour,
                    max_hours_to_add=args.max_hours_to_add,
                    max_hours_to_remove=args.max_hours_to_remove,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "tune-entry-symbols":
        print(
            json.dumps(
                orchestrator.tune_entry_symbols(
                    lookback_days=args.days,
                    report_date=args.date,
                    timezone_name=args.timezone,
                    min_trades_per_symbol=args.min_trades_per_symbol,
                    min_trades_per_direction_symbol=args.min_trades_per_direction_symbol,
                    max_symbols_to_block=args.max_symbols_to_block,
                    max_total_blocked_symbols=args.max_total_blocked_symbols,
                    max_long_symbols_to_block=args.max_long_symbols_to_block,
                    max_short_symbols_to_block=args.max_short_symbols_to_block,
                    max_total_blocked_long_symbols=args.max_total_blocked_long_symbols,
                    max_total_blocked_short_symbols=args.max_total_blocked_short_symbols,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "tune-entry-quality":
        print(
            json.dumps(
                orchestrator.tune_entry_quality(
                    lookback_trades=args.lookback_trades,
                    min_trades=args.min_trades,
                    min_trade_retention_ratio=args.min_trade_retention_ratio,
                    min_expectancy_improvement_rub=args.min_expectancy_improvement_rub,
                    bucket_step=args.bucket_step,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "tune-universe":
        print(
            json.dumps(
                orchestrator.tune_runtime_universe(
                    max_allowed_symbols=args.max_allowed_symbols,
                    min_walk_forward_positive_probability_pct=args.min_walk_forward_positive_probability_pct,
                    min_latest_fold_monthly_return_pct=args.min_latest_fold_monthly_return_pct,
                    min_walk_forward_folds=args.min_walk_forward_folds,
                    min_latest_fold_trades=args.min_latest_fold_trades,
                    require_optimizer_overlap=not args.skip_optimizer_overlap,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "tune-strategy":
        print(
            json.dumps(
                orchestrator.tune_strategy(
                    min_monthly_improvement_pct=args.min_monthly_improvement_pct,
                    max_extra_drawdown_pct=args.max_extra_drawdown_pct,
                    min_positive_fold_probability_pct=args.min_positive_fold_probability_pct,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "tune-exits":
        print(
            json.dumps(
                orchestrator.tune_exits(
                    min_monthly_improvement_pct=args.min_monthly_improvement_pct,
                    max_extra_drawdown_pct=args.max_extra_drawdown_pct,
                    min_positive_fold_probability_pct=args.min_positive_fold_probability_pct,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "sandbox-init":
        print(
            json.dumps(
                orchestrator.init_sandbox(args.fund_rub),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
