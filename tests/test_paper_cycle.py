from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from samosbor.autonomy.market_regime import MarketRegime
from samosbor.autonomy.regime_policy import PolicyDecisionType
from samosbor.autonomy.signal_feedback import signal_feedback_path
from samosbor.autonomy.trade_review import trade_review_path
from samosbor.config import load_config
from samosbor.domain import Candle, ExitReason, Instrument, InstrumentType, Signal, SignalDirection
from samosbor.execution.paper import LocalPaperBroker
from samosbor.orchestrator import TradingOrchestrator


def _write_basic_paper_config(
    root: Path,
    *,
    symbol: str = "SBER",
    strategy_lines: list[str] | None = None,
) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "paper.toml"
    strategy_lines = strategy_lines or ["min_liquidity_rub = 1.0"]
    config_path.write_text(
        "\n".join(
            [
                "[app]",
                'timezone = "Europe/Moscow"',
                "",
                "[data]",
                'source = "csv"',
                'csv_path = "data/demo.csv"',
                'timeframe = "30min"',
                "",
                "[[data.instruments]]",
                f'symbol = "{symbol}"',
                'instrument_type = "stock"',
                "lot_size = 1",
                "",
                "[strategy]",
                *strategy_lines,
                "",
                "[execution]",
                'mode = "local-paper"',
                "allow_live_trading = false",
                'state_path = "state/paper_state.json"',
                "",
                "[backtest]",
                "initial_cash = 100000",
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
    return config_path


class _FakeProvider:
    def __init__(
        self,
        instruments: list[Instrument],
        history: dict[str, list[Candle]],
        confirmation_history: dict[str, list[Candle]] | None = None,
    ):
        self._instruments = instruments
        self._history = history
        self._confirmation_history = confirmation_history or {}

    def resolve_universe(self, instruments: list[Instrument]) -> list[Instrument]:
        return self._instruments

    def load_history(self, instruments: list[Instrument]) -> dict[str, list[Candle]]:
        return self._history

    def load_history_for_timeframe(self, instruments: list[Instrument], timeframe: str) -> dict[str, list[Candle]]:
        return self._confirmation_history


class _OrderBookProvider(_FakeProvider):
    def __init__(self, instruments: list[Instrument], history: dict[str, list[Candle]], snapshot: dict[str, object]):
        super().__init__(instruments, history)
        self.snapshot = snapshot

    def get_order_book_snapshot(self, instrument, *, depth, quantity_lots=0, direction=""):
        return {
            **self.snapshot,
            "requested_lots": quantity_lots,
        }


class _PaperCycleOrchestrator(TradingOrchestrator):
    def __init__(self, config, provider):
        super().__init__(config)
        self._provider = provider

    def _data_provider(self):
        return self._provider


class _BlockedSignalStrategy:
    def prepare_history(self, instrument, candles):
        return None

    def generate_signal(self, instrument, candles):
        last = candles[-1]
        return Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.82,
            entry_price=last.close,
            stop_price=last.close - 1.0,
            take_profit=last.close + 2.0,
            reason="forced-test-signal",
        )

    def allows_entry_at(self, timestamp):
        return True

    def should_force_flatten_at(self, timestamp):
        return False

    def entry_block_reason_for_instrument(self, instrument, timestamp, direction=None):
        return "entry blocked by hour schedule"


class _BlockedSignalCycleOrchestrator(_PaperCycleOrchestrator):
    def _strategy(self):
        return _BlockedSignalStrategy()


class _EntrySignalStrategy(_BlockedSignalStrategy):
    def entry_block_reason_for_instrument(self, instrument, timestamp, direction=None):
        return None


class _EntrySignalCycleOrchestrator(_PaperCycleOrchestrator):
    def _strategy(self):
        return _EntrySignalStrategy()


class _MlBlockedCycleOrchestrator(_EntrySignalCycleOrchestrator):
    def _signal_with_learning_assessment(self, signal, feedback_payload, *, timestamp, quantity_lots):
        metadata = dict(signal.metadata)
        metadata["ml_learning"] = {
            "available": True,
            "blocks_entry": True,
            "probability_profit": 0.16,
            "expected_pnl_position_rub": -50.0,
            "required_net_edge_rub": 20.0,
            "learning_tags": ["low-quality-learning", "negative-expectancy-learning"],
        }
        return replace(signal, metadata=metadata)


class _ZeroRuntimeSizeCycleOrchestrator(_EntrySignalCycleOrchestrator):
    def _signal_with_runtime_policy(self, signal, quantity_lots, *, market_regime, symbol_health, entry_mode):
        metadata = dict(signal.metadata)
        metadata["adaptive_risk_sizing"] = {
            "original_quantity_lots": int(quantity_lots),
            "adjusted_quantity_lots": 0,
            "risk_multiplier": 0.0,
            "entry_mode": entry_mode,
            "symbol_health": symbol_health,
        }
        return replace(signal, metadata=metadata), 0, None


class _ShortExhaustionSignalStrategy(_EntrySignalStrategy):
    def generate_signal(self, instrument, candles):
        last = candles[-1]
        return Signal(
            instrument=instrument,
            direction=SignalDirection.SHORT,
            strength=0.82,
            entry_price=last.close,
            stop_price=last.close + 1.0,
            take_profit=last.close - 2.0,
            reason="ema-down adx=49.7 rsi=27.7 macd_hist=-4.5",
        )


class _ShortExhaustionCycleOrchestrator(_PaperCycleOrchestrator):
    def _strategy(self):
        return _ShortExhaustionSignalStrategy()


class _PlainShortSignalStrategy(_EntrySignalStrategy):
    def generate_signal(self, instrument, candles):
        last = candles[-1]
        return Signal(
            instrument=instrument,
            direction=SignalDirection.SHORT,
            strength=0.82,
            entry_price=last.close,
            stop_price=last.close + 1.0,
            take_profit=last.close - 2.0,
            reason="ema-down adx=32.0 rsi=42.0 macd_hist=-0.2",
        )


class _PlainShortCycleOrchestrator(_PaperCycleOrchestrator):
    def _strategy(self):
        return _PlainShortSignalStrategy()


class _NoSignalStrategy:
    def prepare_history(self, instrument, candles):
        return None

    def generate_signal(self, instrument, candles):
        return None

    def allows_entry_at(self, timestamp):
        return False

    def should_force_flatten_at(self, timestamp):
        return False


class _AlwaysSignalAdaptationStrategy:
    def prepare_history(self, instrument, candles):
        return None

    def generate_signal(self, instrument, candles):
        last = candles[-1]
        return Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.77,
            entry_price=last.close,
            stop_price=last.close - 1.0,
            take_profit=last.close + 2.0,
            reason="shadow-adaptation",
        )

    def allows_entry_at(self, timestamp):
        return True

    def should_force_flatten_at(self, timestamp):
        return False


class _AdaptationPaperCycleOrchestrator(_PaperCycleOrchestrator):
    def _strategy(self):
        return _NoSignalStrategy()

    def _adaptation_strategy(self):
        return _AlwaysSignalAdaptationStrategy()


class _NoSignalPaperCycleOrchestrator(_PaperCycleOrchestrator):
    def _strategy(self):
        return _NoSignalStrategy()

    def _adaptation_strategy(self):
        return _NoSignalStrategy()


class PaperCycleSessionFlatTest(unittest.TestCase):
    def test_paper_cycle_flattens_existing_position_in_session_window(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "allowed_entry_hours = [10, 11, 12, 13, 14, 15, 16, 17]",
                        "allowed_entry_weekdays = [0, 1, 2, 3, 4]",
                        "forced_flat_hours = [18, 19, 20, 21, 22, 23]",
                        "forced_flat_weekdays = [0, 1, 2, 3, 4]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            state_path = config.resolve_path(config.execution.state_path)
            broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
            broker.open_position(
                Signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    strength=0.8,
                    entry_price=100.0,
                    stop_price=95.0,
                    take_profit=110.0,
                    reason="bootstrap-position",
                ),
                10,
                datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc),
            )
            broker.save(state_path)

            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc),
                open=100.0,
                high=100.4,
                low=99.9,
                close=100.1,
                volume=5_000_000,
            )
            orchestrator = _PaperCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )

            orchestrator.run_paper_cycle()
            reloaded = LocalPaperBroker.load(
                state_path,
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(len(reloaded.portfolio.positions), 0)
            self.assertEqual(len(reloaded.trades), 1)
            self.assertEqual(reloaded.trades[0].reason, ExitReason.SESSION_FLAT.value)
            review = json.loads(trade_review_path(state_path).read_text(encoding="utf-8"))
            self.assertEqual(review["reviewed_trades"], 1)
            self.assertEqual(review["reviews"][0]["symbol"], "SBER")


class PaperCycleTrailingProtectionTest(unittest.TestCase):
    def test_paper_cycle_records_profitable_trailing_stop_as_profit_protection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_basic_paper_config(root)
            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            state_path = config.resolve_path(config.execution.state_path)
            broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
            broker.open_position(
                Signal(
                    instrument=instrument,
                    direction=SignalDirection.SHORT,
                    strength=0.8,
                    entry_price=100.0,
                    stop_price=105.0,
                    take_profit=90.0,
                    reason="bootstrap-position",
                ),
                10,
                datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
            )
            broker.update_position_protection(
                "SBER",
                timestamp=datetime(2025, 1, 1, 10, 30, tzinfo=timezone.utc),
                stop_price=98.0,
                reason="trailing-profit-protection",
            )
            broker.save(state_path)

            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=97.0,
                high=98.5,
                low=96.5,
                close=97.4,
                volume=5_000_000,
            )
            orchestrator = _NoSignalPaperCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )

            orchestrator.run_paper_cycle()
            reloaded = LocalPaperBroker.load(
                state_path,
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(len(reloaded.portfolio.positions), 0)
            self.assertEqual(len(reloaded.trades), 1)
            self.assertEqual(reloaded.trades[0].reason, ExitReason.PROFIT_PROTECT_STOP.value)
            self.assertGreater(reloaded.trades[0].net_pnl, 0.0)

    def test_paper_cycle_persists_trailing_stop_after_profit_threshold(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "trailing_profit_trigger_rub = 50.0",
                        "trailing_profit_lock_ratio = 0.5",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            state_path = config.resolve_path(config.execution.state_path)
            broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
            broker.open_position(
                Signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    strength=0.8,
                    entry_price=100.0,
                    stop_price=95.0,
                    take_profit=120.0,
                    reason="bootstrap-position",
                ),
                10,
                datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
            )
            broker.save(state_path)

            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=100.0,
                high=107.0,
                low=101.0,
                close=106.0,
                volume=5_000_000,
            )
            orchestrator = _PaperCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )

            orchestrator.run_paper_cycle()
            reloaded = LocalPaperBroker.load(
                state_path,
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(len(reloaded.trades), 0)
            self.assertEqual(len(reloaded.portfolio.positions), 1)
            position = reloaded.portfolio.positions["SBER"]
            self.assertEqual(position.current_price, 106.0)
            self.assertEqual(position.stop_price, 103.0)
            self.assertEqual(position.take_profit, 120.0)

            protect_events = [event for event in reloaded.events if event.get("action") == "protect"]
            self.assertEqual(len(protect_events), 1)
            self.assertEqual(protect_events[0]["timestamp"], latest_candle.timestamp.isoformat())
            self.assertEqual(protect_events[0]["reason"], "trailing-profit-protection")
            self.assertEqual(protect_events[0]["stop_price"], 103.0)

    def test_paper_cycle_take_profit_activates_runner_instead_of_closing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "take_profit_activates_runner = true",
                        "runner_breakeven_buffer_bps = 10.0",
                        "runner_trailing_atr_multiple = 1.3",
                        "runner_profit_lock_ratio = 0.35",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            state_path = config.resolve_path(config.execution.state_path)
            broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
            broker.open_position(
                Signal(
                    instrument=instrument,
                    direction=SignalDirection.LONG,
                    strength=0.8,
                    entry_price=100.0,
                    stop_price=95.0,
                    take_profit=104.0,
                    reason="bootstrap-position",
                ),
                10,
                datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
            )
            broker.save(state_path)

            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=100.0,
                high=106.0,
                low=100.2,
                close=105.0,
                volume=5_000_000,
            )
            orchestrator = _NoSignalPaperCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )

            orchestrator.run_paper_cycle()
            reloaded = LocalPaperBroker.load(
                state_path,
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(len(reloaded.trades), 0)
            position = reloaded.portfolio.positions["SBER"]
            self.assertTrue(position.runner_active)
            self.assertEqual(position.runner_activation_price, 104.0)
            self.assertEqual(position.runner_extreme_price, 106.0)
            self.assertGreater(position.stop_price, 100.0)
            self.assertNotEqual(position.stop_price, 95.0)

            runner_events = [event for event in reloaded.events if event.get("action") == "runner-activate"]
            protect_events = [event for event in reloaded.events if event.get("action") == "protect"]
            self.assertEqual(len(runner_events), 1)
            self.assertEqual(len(protect_events), 1)
            self.assertEqual(protect_events[0]["reason"], "runner-trailing-profit-protection")


class PaperCycleSignalDiagnosticsTest(unittest.TestCase):
    def _load_config_with_extra_lines(self, root: Path, lines: list[str]):
        config_path = _write_basic_paper_config(root)
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return load_config(config_path)

    def _broker_for_config(self, config):
        return LocalPaperBroker.fresh(
            config.backtest.initial_cash,
            slippage_bps=config.execution.slippage_bps,
            commission_bps=config.execution.commission_bps,
        )

    def _policy_signal(
        self,
        decision_type: str,
        *,
        symbol: str = "SBER",
        entry_mode: str = "trend_short",
        market_regime: str = "weak_down",
    ) -> Signal:
        return Signal(
            instrument=Instrument(symbol=symbol, instrument_type=InstrumentType.STOCK, lot_size=1),
            direction=SignalDirection.SHORT,
            strength=0.6,
            entry_price=100.0,
            stop_price=101.0,
            take_profit=98.0,
            reason="probe",
            metadata={
                "entry_mode": entry_mode,
                "market_regime": {"regime": market_regime},
                "regime_policy": {
                    "decision_type": decision_type,
                    "entry_mode": entry_mode,
                },
            },
        )

    def _approved_policy_event(
        self,
        timestamp: datetime,
        decision_type: str,
        *,
        symbol: str = "SBER",
        entry_mode: str = "trend_short",
        market_regime: str = "weak_down",
    ) -> dict[str, object]:
        return {
            "timestamp": timestamp.isoformat(),
            "action": "signal",
            "symbol": symbol,
            "approved": True,
            "actual_policy_decision": decision_type,
            "metadata": {
                "entry_mode": entry_mode,
                "market_regime": {"regime": market_regime},
                "regime_policy": {
                    "decision_type": decision_type,
                    "entry_mode": entry_mode,
                },
            },
        }

    def test_probe_allows_up_to_40_trades_per_day_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            for index in range(39):
                broker.events.append(
                    self._approved_policy_event(
                        timestamp,
                        PolicyDecisionType.PROBE_TRADE.value,
                        symbol=f"SYM{index}",
                        entry_mode=f"mode-{index}",
                        market_regime=f"regime-{index}",
                    )
                )

            reason = TradingOrchestrator(config)._learning_mode_limit_reason(
                broker,
                self._policy_signal(
                    PolicyDecisionType.PROBE_TRADE.value,
                    entry_mode="fresh-mode",
                    market_regime="fresh-regime",
                ),
                timestamp=timestamp,
            )

            self.assertIsNone(reason)

    def test_exploration_allows_up_to_40_trades_per_day_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            for index in range(39):
                broker.events.append(
                    self._approved_policy_event(
                        timestamp,
                        PolicyDecisionType.EXPLORATION_TRADE.value,
                        symbol=f"SYM{index}",
                        entry_mode=f"mode-{index}",
                        market_regime=f"regime-{index}",
                    )
                )

            reason = TradingOrchestrator(config)._learning_mode_limit_reason(
                broker,
                self._policy_signal(
                    PolicyDecisionType.EXPLORATION_TRADE.value,
                    entry_mode="fresh-mode",
                    market_regime="fresh-regime",
                ),
                timestamp=timestamp,
            )

            self.assertIsNone(reason)

    def test_probe_uses_global_position_slots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                    "[learning_risk.probe]",
                    "max_positions = 1",
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            broker.open_position(
                self._policy_signal(PolicyDecisionType.PROBE_TRADE.value, symbol="GAZP"),
                1,
                timestamp,
            )

            reason = TradingOrchestrator(config)._learning_mode_limit_reason(
                broker,
                self._policy_signal(PolicyDecisionType.PROBE_TRADE.value),
                timestamp=timestamp,
            )

            self.assertIsNone(reason)

    def test_exploration_uses_global_position_slots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                    "[learning_risk.exploration]",
                    "max_positions = 1",
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            broker.open_position(
                self._policy_signal(PolicyDecisionType.EXPLORATION_TRADE.value, symbol="GAZP"),
                1,
                timestamp,
            )

            reason = TradingOrchestrator(config)._learning_mode_limit_reason(
                broker,
                self._policy_signal(PolicyDecisionType.EXPLORATION_TRADE.value),
                timestamp=timestamp,
            )

            self.assertIsNone(reason)

    def test_daily_cap_warn_only_does_not_hard_reject(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                    "[learning_risk.probe]",
                    "max_trades_per_day = 1",
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            broker.events.append(
                self._approved_policy_event(timestamp, PolicyDecisionType.PROBE_TRADE.value)
            )

            signal, quantity_lots, reason, cap_events = TradingOrchestrator(config)._apply_learning_caps(
                broker,
                self._policy_signal(PolicyDecisionType.PROBE_TRADE.value),
                quantity_lots=4,
                timestamp=timestamp,
            )

            self.assertIsNone(reason)
            self.assertEqual(quantity_lots, 4)
            self.assertEqual(cap_events[0]["action"], "learning_cap_warning")
            self.assertTrue(signal.metadata["learning_caps"]["daily_cap_hit"])

    def test_daily_cap_can_shadow_only_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                    "[learning_risk.probe]",
                    "risk_multiplier = 0.3",
                    "max_positions = 5",
                    "max_trades_per_day = 1",
                    "",
                    "[learning_caps]",
                    'daily_cap_behavior = "shadow_only"',
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            broker.events.append(
                self._approved_policy_event(timestamp, PolicyDecisionType.PROBE_TRADE.value)
            )

            reason = TradingOrchestrator(config)._learning_mode_limit_reason(
                broker,
                self._policy_signal(PolicyDecisionType.PROBE_TRADE.value),
                timestamp=timestamp,
            )

            self.assertEqual(reason, "entry blocked by probe learning daily trade cap")

    def test_same_symbol_probe_cap_creates_shadow_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                    "[learning_risk.probe]",
                    "max_same_symbol_trades_per_day = 1",
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            broker.events.append(
                self._approved_policy_event(
                    timestamp,
                    PolicyDecisionType.PROBE_TRADE.value,
                    symbol="SBER",
                    entry_mode="other-mode",
                    market_regime="other-regime",
                )
            )

            signal, quantity_lots, reason, cap_events = TradingOrchestrator(config)._apply_learning_caps(
                broker,
                self._policy_signal(PolicyDecisionType.PROBE_TRADE.value),
                quantity_lots=4,
                timestamp=timestamp,
            )

            self.assertEqual(reason, "same_symbol_learning_cap_hit")
            self.assertEqual(quantity_lots, 0)
            self.assertEqual(cap_events[0]["action"], "learning_cap_shadow_only")
            self.assertTrue(signal.metadata["learning_caps"]["same_symbol_cap_hit"])

    def test_same_entry_mode_probe_cap_creates_shadow_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                    "[learning_risk.probe]",
                    "max_same_entry_mode_trades_per_day = 1",
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            broker.events.append(
                self._approved_policy_event(
                    timestamp,
                    PolicyDecisionType.PROBE_TRADE.value,
                    symbol="GAZP",
                    entry_mode="trend_short",
                    market_regime="other-regime",
                )
            )

            signal, quantity_lots, reason, cap_events = TradingOrchestrator(config)._apply_learning_caps(
                broker,
                self._policy_signal(PolicyDecisionType.PROBE_TRADE.value),
                quantity_lots=4,
                timestamp=timestamp,
            )

            self.assertEqual(reason, "same_entry_mode_learning_cap_hit")
            self.assertEqual(quantity_lots, 0)
            self.assertEqual(cap_events[0]["action"], "learning_cap_shadow_only")
            self.assertTrue(signal.metadata["learning_caps"]["same_entry_mode_cap_hit"])

    def test_same_regime_cap_reduces_size_not_hard_reject(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = self._load_config_with_extra_lines(
                root,
                [
                    "[learning_mode]",
                    "enabled = true",
                    'profile = "relaxed_paper_learning"',
                    "",
                    "[learning_risk.probe]",
                    "max_same_regime_trades_per_day = 1",
                    "",
                ],
            )
            broker = self._broker_for_config(config)
            timestamp = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            broker.events.append(
                self._approved_policy_event(
                    timestamp,
                    PolicyDecisionType.PROBE_TRADE.value,
                    symbol="GAZP",
                    entry_mode="other-mode",
                    market_regime="weak_down",
                )
            )

            signal, quantity_lots, reason, cap_events = TradingOrchestrator(config)._apply_learning_caps(
                broker,
                self._policy_signal(PolicyDecisionType.PROBE_TRADE.value),
                quantity_lots=5,
                timestamp=timestamp,
            )

            self.assertIsNone(reason)
            self.assertEqual(quantity_lots, 2)
            self.assertEqual(cap_events[0]["action"], "learning_cap_reduce_size")
            self.assertTrue(signal.metadata["learning_caps"]["same_regime_cap_hit"])
            self.assertEqual(signal.metadata["learning_caps"]["size_multiplier"], 0.5)

    def test_paper_cycle_summary_includes_signal_rejection_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=100.0,
                high=101.0,
                low=99.8,
                close=100.7,
                volume=5_000_000,
            )
            orchestrator = _BlockedSignalCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )

            result = orchestrator.run_paper_cycle()
            summary_path = Path(result["output_dir"]) / "cycle_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            cycle_events = json.loads(
                (Path(result["output_dir"]) / "cycle_events.json").read_text(encoding="utf-8")
            )
            review = json.loads(
                trade_review_path(config.resolve_path(config.execution.state_path)).read_text(encoding="utf-8")
            )

            self.assertIn("commit_hash", summary)
            self.assertEqual(cycle_events["commit_hash"], summary["commit_hash"])
            self.assertEqual(review["commit_hash"], summary["commit_hash"])
            self.assertEqual(summary["signals_total"], 1)
            self.assertEqual(summary["signals_approved"], 0)
            self.assertEqual(summary["signals_rejected"], 1)
            self.assertEqual(
                summary["signal_rejection_reason_breakdown"],
                {"entry blocked by hour schedule": 1},
            )

    def test_paper_cycle_blocks_entry_when_order_book_spread_is_wide(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "order_book_depth = 10",
                        "require_order_book = true",
                        "max_entry_spread_bps = 5.0",
                        "min_entry_liquidity_cover = 1.0",
                        "min_entry_book_imbalance = -1.0",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=100.0,
                high=101.0,
                low=99.8,
                close=100.7,
                volume=5_000_000,
            )
            snapshot = {
                "available": True,
                "spread_bps": 20.0,
                "entry_liquidity_cover": 5.0,
                "side_imbalance": 0.0,
            }
            orchestrator = _EntrySignalCycleOrchestrator(
                config,
                _OrderBookProvider([instrument], {"SBER": [latest_candle]}, snapshot),
            )

            result = orchestrator.run_paper_cycle()
            summary_path = Path(result["output_dir"]) / "cycle_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            state = LocalPaperBroker.load(
                config.resolve_path(config.execution.state_path),
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(summary["signals_total"], 1)
            self.assertEqual(summary["signals_approved"], 0)
            self.assertEqual(summary["signals_rejected"], 1)
            self.assertEqual(len(state.portfolio.positions), 0)
            self.assertEqual(
                summary["signal_rejection_reason_breakdown"],
                {"entry blocked by wide spread (20.00 bps)": 1},
            )

    def test_paper_cycle_reduces_entry_size_when_ml_edge_is_negative(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=100.0,
                high=101.0,
                low=99.8,
                close=100.7,
                volume=5_000_000,
            )
            orchestrator = _MlBlockedCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )

            result = orchestrator.run_paper_cycle()
            summary = json.loads((Path(result["output_dir"]) / "cycle_summary.json").read_text(encoding="utf-8"))
            cycle_events = json.loads(
                (Path(result["output_dir"]) / "cycle_events.json").read_text(encoding="utf-8")
            )["events"]
            signal_event = next(event for event in cycle_events if event["action"] == "signal")
            state = LocalPaperBroker.load(
                config.resolve_path(config.execution.state_path),
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(summary["signals_total"], 1)
            self.assertEqual(summary["signals_approved"], 1)
            self.assertEqual(summary["signal_rejection_reason_breakdown"], {})
            self.assertTrue(signal_event["approved"])
            self.assertEqual(signal_event["quantity_lots"], max(1, int(signal_event["original_quantity_lots"] * 0.25)))
            position = state.portfolio.positions["SBER"]
            self.assertEqual(position.quantity_lots, signal_event["quantity_lots"])
            self.assertEqual(position.entry_metadata["ml_sizing"]["requested_scale"], 0.25)
            self.assertEqual(
                position.entry_metadata["ml_sizing"]["adjusted_quantity_lots"],
                signal_event["quantity_lots"],
            )

    def test_paper_cycle_rejects_runtime_policy_zero_lot_entry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_basic_paper_config(root)
            config = load_config(config_path)
            instrument = Instrument(symbol="PLZL", instrument_type=InstrumentType.STOCK, lot_size=1)
            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=1857.0,
                high=1860.0,
                low=1850.0,
                close=1854.0,
                volume=5_000_000,
            )
            orchestrator = _ZeroRuntimeSizeCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"PLZL": [latest_candle]}),
            )

            result = orchestrator.run_paper_cycle()
            summary = json.loads((Path(result["output_dir"]) / "cycle_summary.json").read_text(encoding="utf-8"))
            cycle_events = json.loads(
                (Path(result["output_dir"]) / "cycle_events.json").read_text(encoding="utf-8")
            )["events"]
            signal_event = next(event for event in cycle_events if event["action"] == "signal")
            state = LocalPaperBroker.load(
                config.resolve_path(config.execution.state_path),
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(summary["signals_total"], 1)
            self.assertEqual(summary["signals_approved"], 0)
            self.assertEqual(summary["signals_rejected"], 1)
            self.assertFalse(signal_event["approved"])
            self.assertEqual(signal_event["quantity_lots"], 0)
            self.assertEqual(signal_event["reason"], "entry blocked by adaptive risk size < 1 lot")
            self.assertEqual(state.portfolio.positions, {})

    def test_paper_cycle_opens_weak_choppy_probe_and_creates_pending_addon(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_basic_paper_config(root)
            config_path.write_text(
                config_path.read_text(encoding="utf-8")
                + "\n".join(
                    [
                        "[learning_mode]",
                        "enabled = true",
                        'profile = "relaxed_paper_learning"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            latest_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=100.4,
                high=100.6,
                low=99.8,
                close=100.0,
                volume=5_000_000,
            )
            orchestrator = _PlainShortCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )
            regime = MarketRegime("weak_down_choppy", 0.82, {"breadth_down": 0.7, "chop_score": 0.8})

            with patch("samosbor.orchestrator.detect_market_regime", return_value=regime):
                result = orchestrator.run_paper_cycle()

            summary = json.loads((Path(result["output_dir"]) / "cycle_summary.json").read_text(encoding="utf-8"))
            cycle_events = json.loads(
                (Path(result["output_dir"]) / "cycle_events.json").read_text(encoding="utf-8")
            )["events"]
            signal_event = next(event for event in cycle_events if event["action"] == "signal")
            pending_event = next(event for event in cycle_events if event["action"] == "pending-entry")
            feedback = json.loads(
                signal_feedback_path(config.resolve_path(config.execution.state_path)).read_text(encoding="utf-8")
            )
            state = LocalPaperBroker.load(
                config.resolve_path(config.execution.state_path),
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(summary["signals_total"], 1)
            self.assertEqual(summary["signals_approved"], 1)
            self.assertTrue(signal_event["approved"])
            self.assertEqual(
                signal_event["metadata"]["regime_policy"]["entry_mode"],
                "weak_choppy_direct_probe_short",
            )
            self.assertEqual(
                signal_event["metadata"]["regime_policy"]["strict_policy_decision"],
                "wait",
            )
            self.assertTrue(signal_event["metadata"]["regime_policy"]["pending_addon_created"])
            self.assertEqual(pending_event["status"], "created")
            self.assertTrue(pending_event["metadata"]["is_addon"])
            self.assertTrue(pending_event["metadata"]["addon_shadow_only_due_to_no_pyramiding"])
            self.assertEqual(feedback["pending_entries"][0]["state"], "WAIT_PULLBACK_SHORT")
            self.assertTrue(feedback["pending_entries"][0]["is_addon"])
            self.assertEqual(len(state.portfolio.positions), 1)
            self.assertEqual(state.portfolio.positions["SBER"].direction, SignalDirection.SHORT)

    def test_paper_cycle_opens_pending_pullback_short_after_failed_rebound(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_basic_paper_config(root)
            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            first_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                open=100.4,
                high=100.6,
                low=99.8,
                close=100.0,
                volume=5_000_000,
            )
            regime = MarketRegime("weak_down_choppy", 0.82, {"breadth_down": 0.7, "chop_score": 0.8})
            first_orchestrator = _PlainShortCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [first_candle]}),
            )
            with patch("samosbor.orchestrator.detect_market_regime", return_value=regime):
                first_orchestrator.run_paper_cycle()

            trigger_candle = Candle(
                timestamp=datetime(2025, 1, 1, 11, 30, tzinfo=timezone.utc),
                open=100.35,
                high=100.45,
                low=99.75,
                close=99.9,
                volume=5_000_000,
            )
            second_orchestrator = _NoSignalPaperCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [first_candle, trigger_candle]}),
            )
            with patch("samosbor.orchestrator.detect_market_regime", return_value=regime):
                result = second_orchestrator.run_paper_cycle()

            cycle_events = json.loads(
                (Path(result["output_dir"]) / "cycle_events.json").read_text(encoding="utf-8")
            )["events"]
            pending_event = next(event for event in cycle_events if event["action"] == "pending-entry")
            signal_event = next(event for event in cycle_events if event["action"] == "signal")
            feedback = json.loads(
                signal_feedback_path(config.resolve_path(config.execution.state_path)).read_text(encoding="utf-8")
            )
            state = LocalPaperBroker.load(
                config.resolve_path(config.execution.state_path),
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(pending_event["status"], "triggered")
            self.assertTrue(signal_event["approved"])
            self.assertEqual(signal_event["metadata"]["entry_mode"], "pullback_short")
            self.assertEqual(signal_event["metadata"]["pending_entry"]["outcome"], "triggered")
            self.assertEqual(feedback["pending_entries"], [])
            position = state.portfolio.positions["SBER"]
            self.assertEqual(position.direction, SignalDirection.SHORT)
            expected_fill = trigger_candle.close / (1 + config.execution.slippage_bps / 10000)
            self.assertAlmostEqual(position.entry_price, expected_fill)

    def test_paper_cycle_observes_short_exhaustion_without_blocking_entry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "YDEX"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "order_book_depth = 10",
                        "require_order_book = true",
                        "max_entry_spread_bps = 12.0",
                        "min_entry_liquidity_cover = 1.0",
                        "min_entry_book_imbalance = -1.0",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="YDEX", instrument_type=InstrumentType.STOCK, lot_size=1)
            candles = [
                Candle(
                    timestamp=datetime(2025, 1, 1, 10, 30, tzinfo=timezone.utc),
                    open=100.0,
                    high=100.4,
                    low=99.8,
                    close=100.0,
                    volume=5_000_000,
                ),
                Candle(
                    timestamp=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
                    open=100.0,
                    high=100.2,
                    low=98.8,
                    close=99.0,
                    volume=5_000_000,
                ),
            ]
            snapshot = {
                "available": True,
                "spread_bps": 1.0,
                "entry_liquidity_cover": 10.0,
                "side_imbalance": 0.0,
            }
            orchestrator = _ShortExhaustionCycleOrchestrator(
                config,
                _OrderBookProvider([instrument], {"YDEX": candles}, snapshot),
            )

            result = orchestrator.run_paper_cycle()
            summary = json.loads((Path(result["output_dir"]) / "cycle_summary.json").read_text(encoding="utf-8"))
            state = LocalPaperBroker.load(
                config.resolve_path(config.execution.state_path),
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(summary["signals_total"], 1)
            self.assertEqual(summary["signals_approved"], 1)
            self.assertEqual(len(state.portfolio.positions), 1)
            position = state.portfolio.positions["YDEX"]
            self.assertIn("short-after-exhaustion-learning", position.entry_metadata["setup_learning_tags"])
            self.assertEqual(summary["signal_rejection_reason_breakdown"], {})

    def test_paper_cycle_blocks_short_when_5m_confirmation_rebounds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "15min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "YDEX"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        'entry_confirmation_timeframe = "5min"',
                        "entry_confirmation_min_bars = 3",
                        "entry_confirmation_max_adverse_ret = 0.005",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="YDEX", instrument_type=InstrumentType.STOCK, lot_size=1)
            signal_time = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)
            candles_15m = [
                Candle(
                    timestamp=datetime(2025, 1, 1, 10, 45, tzinfo=timezone.utc),
                    open=101.0,
                    high=101.2,
                    low=99.8,
                    close=100.0,
                    volume=5_000_000,
                ),
                Candle(
                    timestamp=signal_time,
                    open=100.0,
                    high=100.2,
                    low=98.9,
                    close=99.0,
                    volume=5_000_000,
                ),
            ]
            candles_5m = [
                Candle(
                    timestamp=signal_time,
                    open=99.0,
                    high=99.4,
                    low=98.8,
                    close=99.2,
                    volume=1_000_000,
                ),
                Candle(
                    timestamp=datetime(2025, 1, 1, 11, 5, tzinfo=timezone.utc),
                    open=99.2,
                    high=99.8,
                    low=99.1,
                    close=99.7,
                    volume=1_000_000,
                ),
                Candle(
                    timestamp=datetime(2025, 1, 1, 11, 10, tzinfo=timezone.utc),
                    open=99.7,
                    high=100.1,
                    low=99.6,
                    close=100.0,
                    volume=1_000_000,
                ),
            ]
            orchestrator = _PlainShortCycleOrchestrator(
                config,
                _FakeProvider(
                    [instrument],
                    {"YDEX": candles_15m},
                    confirmation_history={"YDEX": candles_5m},
                ),
            )

            result = orchestrator.run_paper_cycle()
            summary = json.loads((Path(result["output_dir"]) / "cycle_summary.json").read_text(encoding="utf-8"))
            state = LocalPaperBroker.load(
                config.resolve_path(config.execution.state_path),
                initial_cash=config.backtest.initial_cash,
                slippage_bps=config.execution.slippage_bps,
                commission_bps=config.execution.commission_bps,
            )

            self.assertEqual(summary["signals_total"], 1)
            self.assertEqual(summary["signals_approved"], 0)
            self.assertEqual(len(state.portfolio.positions), 0)
            self.assertEqual(
                summary["signal_rejection_reason_breakdown"],
                {"entry blocked by 5min confirmation (5m rebound against short)": 1},
            )

    def test_paper_cycle_records_market_regime_audit_event(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "15min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "YDEX"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "min_liquidity_rub = 1.0",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="YDEX", instrument_type=InstrumentType.STOCK, lot_size=1)
            start = datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc)
            candles = [
                Candle(
                    timestamp=start + timedelta(minutes=15 * index),
                    open=100.0 + index * 0.1,
                    high=101.0 + index * 0.1,
                    low=99.0 + index * 0.1,
                    close=100.0 + index * 0.1,
                    volume=5_000_000,
                )
                for index in range(60)
            ]
            orchestrator = _EntrySignalCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"YDEX": candles}),
            )

            result = orchestrator.run_paper_cycle()
            cycle_events = json.loads(
                (Path(result["output_dir"]) / "cycle_events.json").read_text(encoding="utf-8")
            )["events"]

            regime_event = cycle_events[0]
            self.assertEqual(regime_event["event"], "market_regime_detected")
            self.assertIn(regime_event["regime"], {"unknown", "clean_uptrend", "mixed"})
            signal_event = next(event for event in cycle_events if event.get("action") == "signal")
            self.assertIn("market_regime", signal_event["metadata"])
            self.assertIn("regime_policy_audit", signal_event["metadata"])

    def test_paper_cycle_records_shadow_evidence_outside_runtime_entry_hours(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        'timeframe = "30min"',
                        "",
                        "[[data.instruments]]",
                        'symbol = "SBER"',
                        'instrument_type = "stock"',
                        "lot_size = 1",
                        "",
                        "[strategy]",
                        "allowed_entry_hours = [1]",
                        "allowed_entry_weekdays = [0]",
                        "min_liquidity_rub = 1.0",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "initial_cash = 100000",
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

            config = load_config(config_path)
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
            latest_candle = Candle(
                timestamp=datetime(2025, 1, 6, 11, 0, tzinfo=timezone.utc),
                open=100.0,
                high=101.0,
                low=99.8,
                close=100.7,
                volume=5_000_000,
            )
            orchestrator = _AdaptationPaperCycleOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": [latest_candle]}),
            )

            orchestrator.run_paper_cycle()
            feedback = json.loads(
                signal_feedback_path(
                    config.resolve_path(config.execution.state_path)
                ).read_text(encoding="utf-8")
            )

            self.assertEqual(len(feedback["pending"]), 1)
            self.assertEqual(feedback["pending"][0]["symbol"], "SBER")


if __name__ == "__main__":
    unittest.main()
