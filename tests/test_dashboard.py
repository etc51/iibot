from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from samosbor.dashboard import (
    _sanitize_entry_schedule_payload,
    _sanitize_entry_symbols_payload,
    build_dashboard_payload,
    render_dashboard_html,
)


class DashboardTest(unittest.TestCase):
    def test_dashboard_exposes_breakeven_trailing_details_for_open_position(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            state_dir = root / "state"
            runs_dir = root / "runs"
            config_dir.mkdir(parents=True)
            state_dir.mkdir(parents=True)
            runs_dir.mkdir(parents=True)

            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[tbank]",
                        'account_name = "Акции"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "breakeven_trigger_pct = 0.5",
                        "trailing_profit_lock_ratio = 0.5",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/demo_state.json"',
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        'output_dir = "runs"',
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (state_dir / "demo_state.json").write_text(
                json.dumps(
                    {
                        "portfolio": {
                            "cash": 300000.0,
                            "positions": {
                                "SBER": {
                                    "instrument": {
                                        "symbol": "SBER",
                                        "instrument_type": "stock",
                                        "lot_size": 1,
                                    },
                                    "direction": "long",
                                    "quantity_lots": 10,
                                    "entry_price": 100.0,
                                    "current_price": 101.0,
                                    "stop_price": 100.25,
                                    "take_profit": 103.0,
                                    "margin_requirement": 0.0,
                                    "signal_strength": 0.8,
                                    "opened_at": "2026-06-15T09:00:00+00:00",
                                    "updated_at": "2026-06-15T09:30:00+00:00",
                                }
                            },
                        },
                        "events": [
                            {
                                "timestamp": "2026-06-15T09:30:00+00:00",
                                "symbol": "SBER",
                                "action": "protect",
                                "direction": "long",
                                "stop_price": 100.25,
                                "take_profit": 103.0,
                                "reason": "trailing-profit-protection",
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (state_dir / "demo_state_signal_feedback.json").write_text(
                json.dumps({"pending": [], "resolved": []}),
                encoding="utf-8",
            )

            payload = build_dashboard_payload(config_path)
            html = render_dashboard_html(payload)

            position = payload["runtime"]["positions"][0]
            self.assertEqual(position["trailing_status"], "active")
            self.assertAlmostEqual(position["trailing_breakeven_trigger_pct"], 0.5)
            self.assertAlmostEqual(position["trailing_trigger_price"], 100.5)
            self.assertAlmostEqual(position["trailing_first_lock_price"], 100.0)
            self.assertAlmostEqual(position["trailing_protected_profit_rub"], 2.5)
            self.assertIn("breakeven 0.50%", html)
            self.assertIn("Break-even trigger", html)

    def test_dashboard_reads_latest_samosbor_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            state_dir = root / "state"
            runs_dir = root / "runs"
            config_dir.mkdir(parents=True)
            state_dir.mkdir(parents=True)
            runs_dir.mkdir(parents=True)

            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[tbank]",
                        'account_name = "Акции"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "LKOH"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "TATN"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/demo_state.json"',
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        'output_dir = "runs"',
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (state_dir / "demo_state.json").write_text(
                json.dumps(
                    {
                        "portfolio": {
                            "cash": 299999.59,
                            "realized_pnl": -4.9,
                            "peak_equity": 300100.0,
                            "trading_halted": False,
                            "positions": {
                                "LKOH": {
                                    "direction": "long",
                                    "quantity_lots": 1,
                                    "entry_price": 7025.0,
                                    "current_price": 7080.0,
                                    "stop_price": 6960.0,
                                    "take_profit": 7155.0,
                                    "margin_requirement": 0.0,
                                    "signal_strength": 0.77,
                                    "opened_at": "2026-06-15T09:00:00+00:00",
                                    "updated_at": "2026-06-15T09:35:00+00:00",
                                }
                            },
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (state_dir / "demo_state_signal_feedback.json").write_text(
                json.dumps({"pending": [{"symbol": "TATN"}], "resolved": [{"symbol": "LKOH"}]}),
                encoding="utf-8",
            )

            self._write_json(
                runs_dir / "paper" / "20260615-093644" / "cycle_summary.json",
                {
                    "timestamp": "2026-06-15T09:36:44.918301+00:00",
                    "equity_rub": 299995.1,
                    "cash_rub": 299999.59,
                    "gross_exposure_rub": 2058.39,
                    "open_positions": 2,
                    "trading_halted": False,
                    "signals_total": 5,
                    "signals_approved": 2,
                    "signals_rejected": 3,
                    "signal_rejection_reason_breakdown": {
                        "outside_entry_hours": 2,
                        "blocked_symbol_direction": 1,
                    },
                },
            )
            self._write_json(
                runs_dir / "paper-reports" / "20260615-091418" / "summary.json",
                {
                    "summary": {
                        "trades": 4,
                        "net_pnl_rub": 125.0,
                        "win_rate_pct": 50.0,
                        "profit_factor": 1.2,
                        "expectancy_rub": 31.25,
                    },
                    "portfolio": {"open_positions": 2},
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "effective-config" / "20260615-093614" / "effective_config.json",
                {
                    "effective_config_path": str(config_dir / "paper.effective.toml"),
                    "applied_strategy_overrides": {
                        "allowed_symbols": ["LKOH"],
                        "allowed_entry_hours": [9, 10, 12],
                        "blocked_long_symbols": ["LKOH"],
                    },
                    "rollback_guardrail": {"rollback_to_base": False, "reason": "ok"},
                    "sources": [
                        {
                            "source": "universe-selection",
                            "changed": True,
                            "selected_values": {"allowed_symbols": ["LKOH"]},
                            "activation": {
                                "confirmed": True,
                                "pending_activation": False,
                                "reason": "candidate confirmed across 2 consecutive tuning runs",
                            },
                        },
                        {
                            "source": "entry-symbols",
                            "changed": True,
                            "selected_values": {"blocked_long_symbols": ["LKOH"]},
                            "activation": {"reason": "confirmed"},
                        }
                    ],
                    "output_dir": str(runs_dir / "autotune" / "demo" / "effective-config" / "20260615-093614"),
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-symbols" / "20260615-093603" / "symbol_restrictions.json",
                {
                    "changed": True,
                    "reason": "entry symbol restrictions updated from paper results",
                    "evidence_source": "signal-feedback",
                    "proposed_blocked_symbols": [],
                    "proposed_blocked_long_symbols": ["LKOH"],
                    "proposed_blocked_short_symbols": [],
                    "symbol_direction_breakdown": [
                        {
                            "symbol": "LKOH",
                            "direction": "long",
                            "trades": 8,
                            "win_rate_pct": 37.5,
                            "net_pnl_rub": -0.01,
                            "profit_factor": 0.98,
                            "expectancy_rub": -0.001,
                        }
                    ],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-schedule" / "20260615-091418" / "schedule_tuning.json",
                {
                    "changed": True,
                    "reason": "hours updated from paper results",
                    "evidence_source": "signal-feedback",
                    "proposed_hours": [9, 10, 12],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-quality" / "20260615-081025" / "entry_quality_tuning.json",
                {
                    "changed": True,
                    "reason": "lower signal-strength threshold restored recent paper-trade coverage",
                    "evidence_source": "closed-trades+signal-feedback",
                    "evidence_counts": {
                        "closed_trades": 2,
                        "feedback_trades": 3,
                        "deduplicated_feedback_trades": 2,
                        "duplicate_feedback_trades": 1,
                        "combined_trades": 4,
                    },
                    "current_min_signal_strength": 0.7,
                    "recommended_min_signal_strength": 0.55,
                    "lookback": {
                        "requested_trades": 40,
                        "eligible_trades": 12,
                        "min_trades": 8,
                    },
                    "baseline_summary": {
                        "trades": 4,
                        "expectancy_rub": -40.0,
                        "profit_factor": 0.82,
                    },
                    "recommended_summary": {
                        "trades": 12,
                        "expectancy_rub": 38.5,
                        "profit_factor": 1.24,
                    },
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "universe-selection" / "20260615-092500" / "universe_selection.json",
                {
                    "changed": True,
                    "reason": "runtime universe updated from optimizer and walk-forward consensus",
                    "guardrails": {
                        "min_effective_symbols": 1,
                        "min_configured_coverage_ratio": 0.5,
                    },
                    "configured_symbols": ["LKOH", "TATN"],
                    "current_allowed_symbols": [],
                    "current_effective_symbols": ["LKOH", "TATN"],
                    "optimizer_best_symbols": ["LKOH"],
                    "walk_forward_latest_symbols": ["LKOH"],
                    "consensus_symbols": ["LKOH"],
                    "proposed_allowed_symbols": ["LKOH"],
                    "proposed_effective_symbols": ["LKOH"],
                    "optimizer_summary": {"evaluated_candidates": 12},
                    "walk_forward_summary": {
                        "folds_evaluated": 4,
                        "probability_positive_pct": 75.0,
                        "latest_fold_test_trades": 6,
                    },
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "nightly-autonomy" / "20260615-081500" / "nightly_autonomy.json",
                {
                    "timestamp": "2026-06-15T08:15:00+00:00",
                    "steps_executed": ["paper-report", "bootstrap-entry-feedback", "tune-entry-symbols"],
                    "output_dir": str(runs_dir / "autotune" / "demo" / "nightly-autonomy" / "20260615-081500"),
                },
            )

            payload = build_dashboard_payload(config_path, effective_config_path=config_dir / "paper.effective.toml")
            html = render_dashboard_html(payload)

            self.assertEqual(payload["runtime"]["latest_cycle"]["equity_rub"], 299995.1)
            self.assertEqual(
                payload["autonomy"]["effective_runtime"]["applied_strategy_overrides"]["blocked_long_symbols"],
                ["LKOH"],
            )
            self.assertEqual(
                payload["autonomy"]["runtime_universe"]["active_allowed_symbols_override"],
                ["LKOH"],
            )
            self.assertEqual(
                payload["autonomy"]["runtime_universe"]["active_effective_symbols"],
                ["LKOH"],
            )
            self.assertTrue(payload["autonomy"]["runtime_universe"]["activation"]["confirmed"])
            self.assertEqual(payload["runtime"]["signal_feedback"]["resolved_signals"], 1)
            self.assertEqual(payload["autonomy"]["entry_quality"]["current_min_signal_strength"], 0.7)
            self.assertEqual(payload["autonomy"]["entry_quality"]["lookback"]["eligible_trades"], 12)
            self.assertIn("Samosbor Paper Dashboard", html)
            self.assertIn("Effective Runtime Universe", html)
            self.assertIn("Signal Diagnostics", html)
            self.assertIn("outside_entry_hours", html)
            self.assertIn("Lookback eligible trades", html)
            self.assertIn("LKOH", html)
            self.assertIn("blocked_long_symbols", html)

    def test_dashboard_ignores_stale_futures_autonomy_artifacts_for_stock_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            state_dir = root / "state"
            runs_dir = root / "runs"
            config_dir.mkdir(parents=True)
            state_dir.mkdir(parents=True)
            runs_dir.mkdir(parents=True)

            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[tbank]",
                        'account_name = "Акции"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "LKOH"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "TATN"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "allowed_entry_hours = [10, 11, 12]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/demo_state.json"',
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        'output_dir = "runs"',
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (state_dir / "demo_state.json").write_text(
                json.dumps({"portfolio": {"cash": 300000.0, "positions": {}}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (state_dir / "demo_state_signal_feedback.json").write_text(
                json.dumps({"pending": [], "resolved": []}),
                encoding="utf-8",
            )

            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-symbols" / "20260615-090000" / "symbol_restrictions.json",
                {
                    "changed": True,
                    "reason": "stock restriction",
                    "evidence_source": "signal-feedback",
                    "proposed_blocked_symbols": [],
                    "proposed_blocked_long_symbols": ["LKOH"],
                    "proposed_blocked_short_symbols": [],
                    "symbol_direction_breakdown": [
                        {
                            "symbol": "LKOH",
                            "direction": "long",
                            "trades": 4,
                            "win_rate_pct": 25.0,
                            "net_pnl_rub": -120.0,
                            "profit_factor": 0.7,
                            "expectancy_rub": -30.0,
                        }
                    ],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-symbols" / "20260615-100000" / "symbol_restrictions.json",
                {
                    "changed": True,
                    "reason": "stale futures restriction",
                    "evidence_source": "signal-feedback",
                    "proposed_blocked_symbols": [],
                    "proposed_blocked_long_symbols": ["CNYRUBF"],
                    "proposed_blocked_short_symbols": [],
                    "symbol_direction_breakdown": [
                        {
                            "symbol": "CNYRUBF",
                            "direction": "long",
                            "trades": 12,
                            "win_rate_pct": 33.3,
                            "net_pnl_rub": -420.0,
                            "profit_factor": 0.5,
                            "expectancy_rub": -35.0,
                        }
                    ],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-schedule" / "20260615-090000" / "schedule_tuning.json",
                {
                    "changed": True,
                    "reason": "stock schedule",
                    "evidence_source": "signal-feedback",
                    "proposed_hours": [10, 11],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-schedule" / "20260615-100000" / "schedule_tuning.json",
                {
                    "changed": True,
                    "reason": "stale futures schedule",
                    "evidence_source": "signal-feedback",
                    "proposed_hours": [9, 20],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "universe-selection" / "20260615-090000" / "universe_selection.json",
                {
                    "changed": True,
                    "reason": "stock universe",
                    "configured_symbols": ["LKOH", "TATN"],
                    "current_allowed_symbols": [],
                    "current_effective_symbols": ["LKOH", "TATN"],
                    "optimizer_best_symbols": ["LKOH"],
                    "walk_forward_latest_symbols": ["LKOH"],
                    "consensus_symbols": ["LKOH"],
                    "proposed_allowed_symbols": ["LKOH"],
                    "proposed_effective_symbols": ["LKOH"],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "universe-selection" / "20260615-100000" / "universe_selection.json",
                {
                    "changed": True,
                    "reason": "stale futures universe",
                    "configured_symbols": ["CNYRUBF"],
                    "current_allowed_symbols": ["CNYRUBF"],
                    "current_effective_symbols": ["CNYRUBF"],
                    "optimizer_best_symbols": ["CNYRUBF"],
                    "walk_forward_latest_symbols": ["CNYRUBF"],
                    "consensus_symbols": ["CNYRUBF"],
                    "proposed_allowed_symbols": ["CNYRUBF"],
                    "proposed_effective_symbols": ["CNYRUBF"],
                },
            )

            payload = build_dashboard_payload(config_path)
            html = render_dashboard_html(payload)

            self.assertEqual(
                payload["autonomy"]["entry_symbols"]["proposed_blocked_long_symbols"],
                ["LKOH"],
            )
            self.assertEqual(
                payload["autonomy"]["entry_schedule"]["proposed_hours"],
                [10, 11],
            )
            self.assertEqual(
                payload["autonomy"]["runtime_universe"]["proposed_effective_symbols"],
                ["LKOH"],
            )
            self.assertNotIn("CNYRUBF", html)
            self.assertIn("LKOH", html)

    def test_dashboard_sanitize_helpers_preserve_evidence_counts_for_compatible_payloads(self):
        symbol_payload = _sanitize_entry_symbols_payload(
            {
                "changed": True,
                "reason": "mixed evidence",
                "evidence_source": "closed-trades+signal-feedback",
                "evidence_counts": {
                    "closed_trades": 2,
                    "feedback_trades": 3,
                    "deduplicated_feedback_trades": 2,
                    "duplicate_feedback_trades": 1,
                    "combined_trades": 4,
                },
                "proposed_blocked_symbols": ["LKOH", "CNYRUBF"],
                "proposed_blocked_long_symbols": [],
                "proposed_blocked_short_symbols": [],
                "symbol_direction_breakdown": [
                    {"symbol": "LKOH", "direction": "long"},
                    {"symbol": "CNYRUBF", "direction": "short"},
                ],
            },
            {"LKOH", "TATN"},
        )
        schedule_payload = _sanitize_entry_schedule_payload(
            {
                "changed": True,
                "reason": "mixed evidence",
                "evidence_source": "closed-trades+signal-feedback",
                "evidence_counts": {
                    "closed_trades": 2,
                    "feedback_trades": 3,
                    "deduplicated_feedback_trades": 2,
                    "duplicate_feedback_trades": 1,
                    "combined_trades": 4,
                },
                "proposed_hours": [10, 12, 20],
            },
            {10, 11, 12},
        )

        self.assertEqual(symbol_payload["evidence_source"], "closed-trades+signal-feedback")
        self.assertEqual(symbol_payload["evidence_counts"]["combined_trades"], 4)
        self.assertEqual(symbol_payload["proposed_blocked_symbols"], ["LKOH"])
        self.assertEqual(schedule_payload["evidence_source"], "closed-trades+signal-feedback")
        self.assertEqual(schedule_payload["evidence_counts"]["deduplicated_feedback_trades"], 2)
        self.assertEqual(schedule_payload["proposed_hours"], [10, 12])

    def test_dashboard_sanitizes_incompatible_only_autonomy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            state_dir = root / "state"
            runs_dir = root / "runs"
            config_dir.mkdir(parents=True)
            state_dir.mkdir(parents=True)
            runs_dir.mkdir(parents=True)

            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[tbank]",
                        'account_name = "Акции"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "LKOH"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "TATN"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "allowed_entry_hours = [10, 11, 12]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/demo_state.json"',
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        'output_dir = "runs"',
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (state_dir / "demo_state.json").write_text(
                json.dumps({"portfolio": {"cash": 300000.0, "positions": {}}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (state_dir / "demo_state_signal_feedback.json").write_text(
                json.dumps({"pending": [], "resolved": []}),
                encoding="utf-8",
            )

            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-symbols" / "20260615-100000" / "symbol_restrictions.json",
                {
                    "changed": True,
                    "reason": "stale futures restriction",
                    "evidence_source": "signal-feedback",
                    "proposed_blocked_symbols": [],
                    "proposed_blocked_long_symbols": ["CNYRUBF"],
                    "proposed_blocked_short_symbols": [],
                    "symbol_direction_breakdown": [
                        {
                            "symbol": "CNYRUBF",
                            "direction": "long",
                            "trades": 12,
                            "win_rate_pct": 33.3,
                            "net_pnl_rub": -420.0,
                            "profit_factor": 0.5,
                            "expectancy_rub": -35.0,
                        }
                    ],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "entry-schedule" / "20260615-100000" / "schedule_tuning.json",
                {
                    "changed": True,
                    "reason": "stale futures schedule",
                    "evidence_source": "signal-feedback",
                    "proposed_hours": [9, 20],
                },
            )
            self._write_json(
                runs_dir / "autotune" / "demo" / "universe-selection" / "20260615-100000" / "universe_selection.json",
                {
                    "changed": True,
                    "reason": "stale futures universe",
                    "configured_symbols": ["CNYRUBF"],
                    "current_allowed_symbols": ["CNYRUBF"],
                    "current_effective_symbols": ["CNYRUBF"],
                    "optimizer_best_symbols": ["CNYRUBF"],
                    "walk_forward_latest_symbols": ["CNYRUBF"],
                    "consensus_symbols": ["CNYRUBF"],
                    "proposed_allowed_symbols": ["CNYRUBF"],
                    "proposed_effective_symbols": ["CNYRUBF"],
                },
            )

            payload = build_dashboard_payload(config_path)
            html = render_dashboard_html(payload)

            self.assertEqual(payload["autonomy"]["entry_symbols"]["proposed_blocked_long_symbols"], [])
            self.assertEqual(payload["autonomy"]["entry_symbols"]["symbol_direction_breakdown"], [])
            self.assertIn(
                "not compatible with current runtime universe",
                payload["autonomy"]["entry_symbols"]["reason"],
            )
            self.assertEqual(payload["autonomy"]["entry_schedule"]["proposed_hours"], [])
            self.assertIn(
                "not compatible with current runtime hours",
                payload["autonomy"]["entry_schedule"]["reason"],
            )
            self.assertEqual(payload["autonomy"]["runtime_universe"]["proposed_effective_symbols"], [])
            self.assertIn(
                "not compatible with current runtime universe",
                payload["autonomy"]["runtime_universe"]["reason"],
            )
            self.assertNotIn("CNYRUBF", html)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
