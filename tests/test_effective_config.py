from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from samosbor.autonomy.effective_config import (
    align_effective_config_sources,
    base_strategy_values,
    build_effective_config_guardrail_payload,
    build_effective_strategy_overrides,
    default_effective_config_path,
    summarize_effective_config_sources,
    write_effective_config,
)
from samosbor.config import load_config
from samosbor.domain import PortfolioState, SignalDirection, TradeRecord
from samosbor.reporting.paper_report import build_paper_report_payload


def _trade(
    *,
    entry_time: datetime,
    exit_time: datetime,
    net_pnl: float,
) -> TradeRecord:
    return TradeRecord(
        symbol="CNYRUBF",
        direction=SignalDirection.LONG,
        quantity_lots=1,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        reason="stop-loss" if net_pnl < 0 else "take-profit",
        signal_strength=0.5,
    )


class EffectiveConfigTest(unittest.TestCase):
    def test_default_effective_config_path_reuses_existing_effective_file(self):
        original = Path("configs/server_tbank_cnyrubf_premium.toml")
        effective = Path("configs/server_tbank_cnyrubf_premium.effective.toml")
        self.assertEqual(
            default_effective_config_path(original).name,
            "server_tbank_cnyrubf_premium.effective.toml",
        )
        self.assertEqual(default_effective_config_path(effective).name, effective.name)

    def test_latest_candidate_waits_for_second_confirmation_before_activation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [9, 10]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            self._write_json(
                autotune_dir / "strategy" / "20260101-000001" / "strategy_tuning.json",
                {
                    "changed": True,
                    "reason": "strategy candidate passed guardrails",
                    "current_strategy": {
                        "style": "ema_adx_macd",
                        "fast_window": 10,
                        "slow_window": 40,
                        "require_breakout": False,
                        "opening_range_bars": 2,
                        "rel_volume_threshold": 1.1,
                        "breakeven_trigger_pct": 0.0,
                        "min_trend_strength": 0.002,
                        "adx_min": 20.0,
                    },
                    "candidate_strategy": {
                        "style": "ema_adx_macd",
                        "fast_window": 12,
                        "slow_window": 48,
                        "require_breakout": True,
                        "opening_range_bars": 3,
                        "rel_volume_threshold": 1.2,
                        "min_trend_strength": 0.003,
                        "adx_min": 18.0,
                    },
                },
            )
            self._write_json(
                autotune_dir / "exits" / "20260101-000001" / "exit_tuning.json",
                {
                    "changed": True,
                    "reason": "exit candidate passed guardrails",
                    "current_exit_settings": {
                        "atr_stop_multiple": 1.5,
                        "reward_to_risk": 2.0,
                    },
                    "candidate_exit_settings": {
                        "atr_stop_multiple": 1.75,
                        "reward_to_risk": 2.5,
                    },
                },
            )
            self._write_json(
                autotune_dir / "entry-schedule" / "20260101-000001" / "schedule_tuning.json",
                {
                    "changed": False,
                    "reason": "insufficient evidence for change",
                    "current_hours": [9, 10, 12],
                    "proposed_hours": [9, 10, 12],
                },
            )
            self._write_json(
                autotune_dir / "entry-quality" / "20260101-000001" / "entry_quality_tuning.json",
                {
                    "changed": False,
                    "reason": "current threshold already matches the best candidate",
                    "current_min_signal_strength": 0.0,
                    "recommended_min_signal_strength": 0.0,
                },
            )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertEqual(loaded.strategy.fast_window, 10)
            self.assertEqual(loaded.strategy.slow_window, 40)
            self.assertFalse(loaded.strategy.require_breakout)
            self.assertAlmostEqual(loaded.strategy.atr_stop_multiple, 1.5)
            self.assertAlmostEqual(loaded.strategy.reward_to_risk, 2.0)
            self.assertAlmostEqual(loaded.strategy.min_signal_strength, 0.0)
            self.assertAlmostEqual(loaded.strategy.min_trend_strength, 0.002)
            self.assertAlmostEqual(loaded.strategy.adx_min, 20.0)
            self.assertEqual(loaded.strategy.opening_range_bars, 2)
            self.assertAlmostEqual(loaded.strategy.rel_volume_threshold, 1.1)
            self.assertEqual(loaded.strategy.allowed_entry_hours, [9, 10])
            self.assertFalse(sources[0]["activation"]["confirmed"])
            self.assertTrue(sources[0]["activation"]["pending_activation"])
            self.assertEqual(sources[0]["activation"]["confirmation_count"], 1)
            self.assertEqual(sources[1]["selected_values"], {})
            self.assertEqual(sources[2]["selected_values"], {})
            self.assertEqual(sources[3]["selected_values"], {})
            self.assertEqual(sources[4]["selected_values"], {})
            self.assertEqual(sources[5]["selected_values"], {})

    def test_effective_config_uses_confirmed_autotune_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [9, 10]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "strategy" / stamp / "strategy_tuning.json",
                    {
                        "changed": True,
                        "reason": "strategy candidate passed guardrails",
                        "current_strategy": {
                            "style": "ema_adx_macd",
                            "fast_window": 10,
                            "slow_window": 40,
                            "require_breakout": False,
                            "opening_range_bars": 2,
                            "rel_volume_threshold": 1.1,
                            "min_trend_strength": 0.002,
                            "adx_min": 20.0,
                        },
                        "candidate_strategy": {
                            "style": "ema_adx_macd",
                            "fast_window": 12,
                            "slow_window": 48,
                            "require_breakout": True,
                        "opening_range_bars": 3,
                        "rel_volume_threshold": 1.25,
                        "breakeven_trigger_pct": 0.5,
                        "min_trend_strength": 0.003,
                        "adx_min": 18.0,
                    },
                    },
                )
                self._write_json(
                    autotune_dir / "exits" / stamp / "exit_tuning.json",
                    {
                        "changed": True,
                        "reason": "exit candidate passed guardrails",
                        "current_exit_settings": {
                            "atr_stop_multiple": 1.5,
                            "reward_to_risk": 2.0,
                        },
                        "candidate_exit_settings": {
                            "atr_stop_multiple": 1.75,
                            "reward_to_risk": 2.5,
                        },
                    },
                )
            self._write_json(
                autotune_dir / "entry-schedule" / "20260102-000001" / "schedule_tuning.json",
                {
                    "changed": False,
                    "reason": "insufficient evidence for change",
                    "current_hours": [9, 10, 12],
                    "proposed_hours": [9, 10, 12],
                },
            )
            self._write_json(
                autotune_dir / "entry-quality" / "20260102-000001" / "entry_quality_tuning.json",
                {
                    "changed": False,
                    "reason": "current threshold already matches the best candidate",
                    "current_min_signal_strength": 0.2,
                    "recommended_min_signal_strength": 0.2,
                },
            )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertEqual(loaded.strategy.fast_window, 12)
            self.assertEqual(loaded.strategy.slow_window, 48)
            self.assertTrue(loaded.strategy.require_breakout)
            self.assertAlmostEqual(loaded.strategy.atr_stop_multiple, 1.75)
            self.assertAlmostEqual(loaded.strategy.reward_to_risk, 2.5)
            self.assertAlmostEqual(loaded.strategy.breakeven_trigger_pct, 0.5)
            self.assertAlmostEqual(loaded.strategy.min_signal_strength, 0.0)
            self.assertAlmostEqual(loaded.strategy.min_trend_strength, 0.003)
            self.assertAlmostEqual(loaded.strategy.adx_min, 18.0)
            self.assertEqual(loaded.strategy.opening_range_bars, 3)
            self.assertAlmostEqual(loaded.strategy.rel_volume_threshold, 1.25)
            self.assertEqual(loaded.strategy.allowed_entry_hours, [9, 10])
            self.assertEqual(loaded.execution.mode.value, "local-paper")
            self.assertFalse(loaded.execution.allow_live_trading)
            self.assertEqual(len(sources), 6)
            self.assertEqual(sources[0]["source"], "strategy")
            self.assertTrue(sources[0]["activation"]["confirmed"])
            self.assertEqual(sources[0]["activation"]["confirmation_count"], 2)
            self.assertEqual(sources[0]["selected_values"]["breakeven_trigger_pct"], 0.5)
            self.assertEqual(sources[1]["selected_values"]["atr_stop_multiple"], 1.75)
            self.assertEqual(sources[2]["selected_values"], {})
            self.assertEqual(sources[3]["selected_values"], {})
            self.assertEqual(sources[4]["selected_values"], {})
            self.assertEqual(sources[5]["selected_values"], {})

    def test_effective_config_can_apply_confirmed_trailing_exit_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "trailing_profit_trigger_rub = 0.0",
                        "trailing_profit_lock_ratio = 0.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [9, 10]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "exits" / stamp / "exit_tuning.json",
                    {
                        "changed": True,
                        "reason": "exit candidate passed guardrails",
                        "current_exit_settings": {
                        "atr_stop_multiple": 1.5,
                        "reward_to_risk": 2.0,
                        "breakeven_trigger_pct": 0.0,
                        "trailing_profit_trigger_rub": 0.0,
                        "trailing_profit_lock_ratio": 0.0,
                    },
                        "candidate_exit_settings": {
                        "atr_stop_multiple": 1.5,
                        "reward_to_risk": 2.0,
                        "breakeven_trigger_pct": 0.5,
                        "trailing_profit_trigger_rub": 1200.0,
                        "trailing_profit_lock_ratio": 0.5,
                    },
                    },
                )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config, source_summaries=sources)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertAlmostEqual(loaded.strategy.breakeven_trigger_pct, 0.5)
            self.assertAlmostEqual(loaded.strategy.trailing_profit_trigger_rub, 1200.0)
            self.assertAlmostEqual(loaded.strategy.trailing_profit_lock_ratio, 0.5)
            self.assertEqual(sources[1]["selected_values"]["breakeven_trigger_pct"], 0.5)
            self.assertEqual(sources[1]["selected_values"]["trailing_profit_trigger_rub"], 1200.0)
            self.assertEqual(sources[1]["selected_values"]["trailing_profit_lock_ratio"], 0.5)

    def test_stale_current_values_do_not_override_manual_base_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.25",
                        "reward_to_risk = 1.5",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 15.0",
                        "allowed_entry_hours = [9, 10, 11, 12]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            self._write_json(
                autotune_dir / "strategy" / "20260101-000001" / "strategy_tuning.json",
                {
                    "changed": False,
                    "reason": "candidate failed one or more safety guardrails",
                    "current_strategy": {
                        "style": "ema_adx_macd",
                        "fast_window": 10,
                        "slow_window": 40,
                        "require_breakout": False,
                        "min_trend_strength": 0.002,
                        "adx_min": 20.0,
                    },
                    "candidate_strategy": {
                        "style": "ema_adx_macd",
                        "fast_window": 12,
                        "slow_window": 48,
                        "require_breakout": True,
                        "min_trend_strength": 0.003,
                        "adx_min": 18.0,
                    },
                },
            )
            self._write_json(
                autotune_dir / "exits" / "20260101-000001" / "exit_tuning.json",
                {
                    "changed": False,
                    "reason": "candidate exit settings failed one or more safety guardrails",
                    "current_exit_settings": {
                        "atr_stop_multiple": 1.5,
                        "reward_to_risk": 2.0,
                    },
                    "candidate_exit_settings": {
                        "atr_stop_multiple": 1.75,
                        "reward_to_risk": 2.5,
                    },
                },
            )
            self._write_json(
                autotune_dir / "entry-schedule" / "20260101-000001" / "schedule_tuning.json",
                {
                    "changed": False,
                    "reason": "insufficient evidence for change",
                    "current_hours": [9, 10, 12],
                    "proposed_hours": [9, 10, 12],
                },
            )
            self._write_json(
                autotune_dir / "entry-quality" / "20260101-000001" / "entry_quality_tuning.json",
                {
                    "changed": False,
                    "reason": "current threshold already matches the best candidate",
                    "current_min_signal_strength": 0.2,
                    "recommended_min_signal_strength": 0.2,
                },
            )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config, source_summaries=sources)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertEqual(overrides, {})
            self.assertAlmostEqual(loaded.strategy.atr_stop_multiple, 1.25)
            self.assertAlmostEqual(loaded.strategy.reward_to_risk, 1.5)
            self.assertAlmostEqual(loaded.strategy.adx_min, 15.0)
            self.assertAlmostEqual(loaded.strategy.min_signal_strength, 0.0)
            self.assertEqual(loaded.strategy.allowed_entry_hours, [9, 10, 11, 12])

    def test_guardrail_rolls_back_to_base_when_recent_window_turns_negative(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [9, 10]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            config = load_config(base_config)
            paper_report = build_paper_report_payload(
                PortfolioState(cash=100_000.0, realized_pnl=-400.0, peak_equity=100_500.0),
                trades=[
                    _trade(
                        entry_time=datetime(2025, 1, 15, 8, 0, tzinfo=timezone.utc),
                        exit_time=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
                        net_pnl=-120.0,
                    ),
                    _trade(
                        entry_time=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
                        exit_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
                        net_pnl=-110.0,
                    ),
                    _trade(
                        entry_time=datetime(2025, 1, 16, 8, 0, tzinfo=timezone.utc),
                        exit_time=datetime(2025, 1, 16, 10, 0, tzinfo=timezone.utc),
                        net_pnl=-100.0,
                    ),
                    _trade(
                        entry_time=datetime(2025, 1, 16, 12, 0, tzinfo=timezone.utc),
                        exit_time=datetime(2025, 1, 16, 14, 0, tzinfo=timezone.utc),
                        net_pnl=-90.0,
                    ),
                    _trade(
                        entry_time=datetime(2025, 1, 17, 8, 0, tzinfo=timezone.utc),
                        exit_time=datetime(2025, 1, 17, 10, 0, tzinfo=timezone.utc),
                        net_pnl=-80.0,
                    ),
                    _trade(
                        entry_time=datetime(2025, 1, 17, 12, 0, tzinfo=timezone.utc),
                        exit_time=datetime(2025, 1, 17, 14, 0, tzinfo=timezone.utc),
                        net_pnl=-70.0,
                    ),
                ],
                timezone_name="Europe/Moscow",
                report_date=date(2025, 1, 17),
                days=3,
            )
            decision = build_effective_config_guardrail_payload(
                base_values=base_strategy_values(config),
                source_summaries=[
                    {
                        "source": "strategy",
                        "selected_values": {
                            "fast_window": 12,
                            "slow_window": 48,
                        },
                    },
                    {
                        "source": "entry-quality",
                        "selected_values": {
                            "min_signal_strength": 0.25,
                        },
                    },
                ],
                paper_report=paper_report,
                guardrail_days=3,
                min_recent_trades=6,
            )

            self.assertTrue(decision["rollback_to_base"])
            self.assertEqual(decision["active_sources"], ["strategy", "entry-quality"])
            self.assertIn("fast_window", decision["active_override_keys"])
            self.assertIn("min_signal_strength", decision["active_override_keys"])
            self.assertLess(decision["recent_summary"]["net_pnl_rub"], 0)

    def test_effective_config_can_apply_confirmed_symbol_restrictions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "GAZP"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "LKOH"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "NVTK"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "ROSN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "TATN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "PLZL"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "GMKN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "CHMF"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "POSI"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "VTBR"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "IMOEXF"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [9, 10]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "entry-symbols" / stamp / "symbol_restrictions.json",
                    {
                        "changed": True,
                        "reason": "entry symbol restrictions updated from paper results",
                        "current_blocked_symbols": [],
                        "proposed_blocked_symbols": ["IMOEXF"],
                        "additions": ["IMOEXF"],
                        "current_blocked_long_symbols": [],
                        "proposed_blocked_long_symbols": ["SBER"],
                        "long_additions": ["SBER"],
                        "current_blocked_short_symbols": [],
                        "proposed_blocked_short_symbols": ["GAZP"],
                        "short_additions": ["GAZP"],
                    },
                )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config, source_summaries=sources)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertEqual(loaded.strategy.blocked_symbols, ["IMOEXF"])
            self.assertEqual(loaded.strategy.blocked_long_symbols, ["SBER"])
            self.assertEqual(loaded.strategy.blocked_short_symbols, ["GAZP"])
            self.assertEqual(sources[5]["source"], "entry-symbols")
            self.assertEqual(
                sources[5]["selected_values"],
                {
                    "blocked_symbols": ["IMOEXF"],
                    "blocked_long_symbols": ["SBER"],
                    "blocked_short_symbols": ["GAZP"],
                },
            )

    def test_effective_config_can_apply_confirmed_entry_symbol_unblocks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "GAZP"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "LKOH"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "NVTK"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "ROSN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "TATN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "PLZL"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "IMOEXF"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        'blocked_symbols = ["IMOEXF"]',
                        'blocked_long_symbols = ["SBER"]',
                        'blocked_short_symbols = ["GAZP"]',
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "entry-symbols" / stamp / "symbol_restrictions.json",
                    {
                        "changed": True,
                        "reason": "entry symbol restrictions updated from paper results",
                        "current_blocked_symbols": ["IMOEXF"],
                        "proposed_blocked_symbols": [],
                        "additions": [],
                        "removals": ["IMOEXF"],
                        "current_blocked_long_symbols": ["SBER"],
                        "proposed_blocked_long_symbols": [],
                        "long_additions": [],
                        "long_removals": ["SBER"],
                        "current_blocked_short_symbols": ["GAZP"],
                        "proposed_blocked_short_symbols": [],
                        "short_additions": [],
                        "short_removals": ["GAZP"],
                    },
                )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config, source_summaries=sources)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertEqual(loaded.strategy.blocked_symbols, [])
            self.assertEqual(loaded.strategy.blocked_long_symbols, [])
            self.assertEqual(loaded.strategy.blocked_short_symbols, [])
            self.assertEqual(
                sources[5]["selected_values"],
                {
                    "blocked_symbols": [],
                    "blocked_long_symbols": [],
                    "blocked_short_symbols": [],
                },
            )

    def test_effective_config_can_apply_confirmed_allowed_symbol_universe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "GAZP"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "LKOH"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "NVTK"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "ROSN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "TATN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "PLZL"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "GMKN"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [9, 10]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "universe-selection" / stamp / "universe_selection.json",
                    {
                        "changed": True,
                        "reason": "runtime universe updated from optimizer and walk-forward consensus",
                        "configured_symbols": ["SBER", "GAZP", "LKOH", "NVTK", "ROSN", "TATN", "PLZL", "GMKN"],
                        "current_allowed_symbols": [],
                        "current_effective_symbols": ["GAZP", "GMKN", "LKOH", "NVTK", "PLZL", "ROSN", "SBER", "TATN"],
                        "optimizer_best_symbols": ["SBER", "GAZP", "LKOH", "ROSN", "NVTK"],
                        "walk_forward_latest_symbols": ["SBER", "GAZP", "LKOH", "ROSN", "NVTK"],
                        "consensus_symbols": ["SBER", "GAZP", "LKOH", "ROSN", "NVTK"],
                        "proposed_allowed_symbols": ["GAZP", "LKOH", "NVTK", "ROSN", "SBER"],
                        "proposed_effective_symbols": ["GAZP", "LKOH", "NVTK", "ROSN", "SBER"],
                        "additions": [],
                        "removals": ["GMKN", "PLZL", "TATN"],
                    },
                )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config, source_summaries=sources)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertEqual(loaded.strategy.allowed_symbols, ["GAZP", "LKOH", "NVTK", "ROSN", "SBER"])
            self.assertEqual(sources[4]["source"], "universe-selection")
            self.assertEqual(
                sources[4]["selected_values"],
                {"allowed_symbols": ["GAZP", "LKOH", "NVTK", "ROSN", "SBER"]},
            )

    def test_effective_config_can_apply_confirmed_expanded_entry_schedule(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [10, 11, 12, 13, 14, 15, 16, 17]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "entry-schedule" / stamp / "schedule_tuning.json",
                    {
                        "changed": True,
                        "reason": "hours updated from paper results",
                        "current_hours": [10, 11, 12, 13, 14, 15, 16, 17],
                        "proposed_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                    },
                )

            config = load_config(base_config)
            sources = summarize_effective_config_sources(autotune_dir)
            overrides = build_effective_strategy_overrides(config, source_summaries=sources)
            output_config = config_dir / "paper.effective.toml"
            write_effective_config(base_config, output_config, strategy_overrides=overrides)
            loaded = load_config(output_config)

            self.assertEqual(
                loaded.strategy.allowed_entry_hours,
                [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            )
            self.assertEqual(sources[2]["source"], "entry-schedule")
            self.assertEqual(
                sources[2]["selected_values"],
                {"allowed_entry_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
            )

    def test_effective_config_rejects_overconcentrated_symbol_restrictions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "GAZP"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "LKOH"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "NVTK"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "ROSN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "TATN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "PLZL"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "GMKN"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "CHMF"',
                        'instrument_type = "stock"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "POSI"',
                        'instrument_type = "stock"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [10, 11, 12, 13, 14, 15, 16, 17]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "entry-symbols" / stamp / "symbol_restrictions.json",
                    {
                        "changed": True,
                        "reason": "entry symbol restrictions updated from paper results",
                        "current_blocked_symbols": [],
                        "proposed_blocked_symbols": ["TATN"],
                        "current_blocked_long_symbols": [],
                        "proposed_blocked_long_symbols": ["LKOH"],
                        "current_blocked_short_symbols": [],
                        "proposed_blocked_short_symbols": ["SBER"],
                    },
                )

            config = load_config(base_config)
            raw_sources = summarize_effective_config_sources(autotune_dir)
            aligned = align_effective_config_sources(config, raw_sources)
            overrides = build_effective_strategy_overrides(config, source_summaries=raw_sources)

            self.assertEqual(aligned[5]["selected_values"], {})
            self.assertIn("over-concentrate", aligned[5]["activation"]["reason"])
            self.assertEqual(overrides, {})

    def test_aligned_sources_ignore_stale_schedule_and_foreign_symbols(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            autotune_dir = root / "runs" / "autotune" / "paper"
            config_dir.mkdir(parents=True)

            base_config = config_dir / "paper.toml"
            base_config.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
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
                        "fast_window = 10",
                        "slow_window = 40",
                        "require_breakout = false",
                        "atr_stop_multiple = 1.5",
                        "reward_to_risk = 2.0",
                        "min_signal_strength = 0.0",
                        "min_trend_strength = 0.002",
                        "adx_min = 20.0",
                        "allowed_entry_hours = [10, 11, 12, 13, 14, 15, 16, 17]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
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

            for stamp in ("20260101-000001", "20260102-000001"):
                self._write_json(
                    autotune_dir / "entry-schedule" / stamp / "schedule_tuning.json",
                    {
                        "changed": True,
                        "reason": "hours updated from paper results",
                        "current_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                        "proposed_hours": [9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                    },
                )
                self._write_json(
                    autotune_dir / "entry-symbols" / stamp / "symbol_restrictions.json",
                    {
                        "changed": True,
                        "reason": "entry symbol restrictions updated from paper results",
                        "current_blocked_symbols": [],
                        "proposed_blocked_symbols": [],
                        "current_blocked_long_symbols": [],
                        "proposed_blocked_long_symbols": ["CNYRUBF"],
                        "current_blocked_short_symbols": [],
                        "proposed_blocked_short_symbols": [],
                    },
                )

            config = load_config(base_config)
            raw_sources = summarize_effective_config_sources(autotune_dir)
            aligned = align_effective_config_sources(config, raw_sources)
            overrides = build_effective_strategy_overrides(config, source_summaries=raw_sources)

            self.assertEqual(aligned[2]["source"], "entry-schedule")
            self.assertEqual(aligned[2]["selected_values"], {})
            self.assertIn("does not match", aligned[2]["activation"]["reason"])
            self.assertEqual(aligned[5]["source"], "entry-symbols")
            self.assertEqual(aligned[5]["selected_values"], {})
            self.assertIn("not compatible", aligned[5]["activation"]["reason"])
            self.assertEqual(overrides, {})

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
