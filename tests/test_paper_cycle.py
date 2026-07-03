from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from samosbor.autonomy.signal_feedback import signal_feedback_path
from samosbor.autonomy.trade_review import trade_review_path
from samosbor.config import load_config
from samosbor.domain import Candle, ExitReason, Instrument, InstrumentType, RiskDecision, Signal, SignalDirection
from samosbor.execution.paper import LocalPaperBroker
from samosbor.orchestrator import TradingOrchestrator


class _FakeProvider:
    def __init__(self, instruments: list[Instrument], history: dict[str, list[Candle]]):
        self._instruments = instruments
        self._history = history

    def resolve_universe(self, instruments: list[Instrument]) -> list[Instrument]:
        return self._instruments

    def load_history(self, instruments: list[Instrument]) -> dict[str, list[Candle]]:
        return self._history


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


class PaperCycleSignalDiagnosticsTest(unittest.TestCase):
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

    def test_paper_cycle_waits_for_short_exhaustion_confirmation(self):
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
            self.assertEqual(summary["signals_approved"], 0)
            self.assertEqual(len(state.portfolio.positions), 0)
            self.assertEqual(
                summary["signal_rejection_reason_breakdown"],
                {"entry waits for next candle after short exhaustion": 1},
            )

    def test_learning_entry_block_rejects_low_quality_ml(self):
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
                        "",
                        "[strategy]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            instrument = Instrument(symbol="MTLR", instrument_type=InstrumentType.STOCK, lot_size=1)
            signal = Signal(
                instrument=instrument,
                direction=SignalDirection.SHORT,
                strength=0.9,
                entry_price=35.5,
                stop_price=35.9,
                take_profit=34.5,
                reason="test",
                metadata={
                    "entry_candle": {"direction_confirmed_by_close": True},
                    "ml_learning": {
                        "available": True,
                        "blocks_entry": True,
                        "probability_profit": 0.224,
                        "expected_pnl_position_rub": -1603.45,
                        "learning_tags": ["low-quality-learning", "negative-expectancy-learning"],
                    },
                },
            )
            orchestrator = _PaperCycleOrchestrator(config, _FakeProvider([], {}))

            reason = orchestrator._learning_entry_block_reason(signal, quantity_lots=660)

            self.assertEqual(reason, "entry blocked by low ML probability (0.22 < 0.40)")

    def test_learning_entry_block_requires_edge_above_commission_floor(self):
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
                        "",
                        "[strategy]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        "commission_bps = 4.0",
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            instrument = Instrument(symbol="TATN", instrument_type=InstrumentType.STOCK, lot_size=1)
            signal = Signal(
                instrument=instrument,
                direction=SignalDirection.SHORT,
                strength=0.9,
                entry_price=500.0,
                stop_price=505.0,
                take_profit=487.5,
                reason="test",
                metadata={
                    "entry_candle": {"direction_confirmed_by_close": True},
                    "ml_learning": {
                        "available": True,
                        "blocks_entry": True,
                        "probability_profit": 0.62,
                        "expected_pnl_position_rub": 90.0,
                        "required_net_edge_rub": 160.0,
                        "learning_tags": ["commission-edge-learning"],
                    },
                },
            )
            orchestrator = _PaperCycleOrchestrator(config, _FakeProvider([], {}))

            reason = orchestrator._learning_entry_block_reason(signal, quantity_lots=100)

            self.assertEqual(
                reason,
                "entry blocked by ML edge below commission floor (90.00 <= 160.00 RUB)",
            )

    def test_learning_size_adjustment_halves_low_quality_entry(self):
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
                        "",
                        "[strategy]",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            instrument = Instrument(symbol="OZON", instrument_type=InstrumentType.STOCK, lot_size=1)
            signal = Signal(
                instrument=instrument,
                direction=SignalDirection.SHORT,
                strength=0.8,
                entry_price=100.0,
                stop_price=101.0,
                take_profit=97.5,
                reason="test",
                metadata={
                    "ml_learning": {
                        "available": True,
                        "probability_profit": 0.32,
                        "learning_tags": ["low-quality-learning"],
                    }
                },
            )
            decision = RiskDecision(
                True,
                "approved",
                quantity_lots=11,
                risk_budget_rub=1000.0,
                estimated_notional_rub=1100.0,
            )
            orchestrator = _PaperCycleOrchestrator(config, _FakeProvider([], {}))

            adjusted_signal, adjusted_decision = orchestrator._apply_learning_size_adjustment(signal, decision)

            self.assertEqual(adjusted_decision.quantity_lots, 5)
            self.assertEqual(
                adjusted_signal.metadata["learning_size_adjustment"]["original_quantity_lots"],
                11,
            )
            self.assertEqual(
                adjusted_signal.metadata["learning_size_adjustment"]["adjusted_quantity_lots"],
                5,
            )

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
