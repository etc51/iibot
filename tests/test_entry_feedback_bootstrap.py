from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from samosbor.autonomy.signal_feedback import load_signal_feedback, signal_feedback_path
from samosbor.config import load_config
from samosbor.domain import Candle, Instrument, InstrumentType, Signal, SignalDirection
from samosbor.orchestrator import TradingOrchestrator


class _FakeProvider:
    def __init__(self, instruments: list[Instrument], history: dict[str, list[Candle]]):
        self._instruments = instruments
        self._history = history

    def resolve_universe(self, instruments: list[Instrument]) -> list[Instrument]:
        return self._instruments

    def load_history(self, instruments: list[Instrument]) -> dict[str, list[Candle]]:
        return self._history


class _RestrictedNoSignalStrategy:
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
            strength=0.66,
            entry_price=last.close,
            stop_price=last.close - 1.0,
            take_profit=last.close + 1.5,
            reason="adaptation-shadow",
        )

    def allows_entry_at(self, timestamp):
        return True

    def should_force_flatten_at(self, timestamp):
        return False


class _AdaptationBootstrapOrchestrator(TradingOrchestrator):
    def __init__(self, config, provider):
        super().__init__(config)
        self._provider = provider

    def _data_provider(self):
        return self._provider

    def _strategy(self):
        return _RestrictedNoSignalStrategy()

    def _adaptation_strategy(self):
        return _AlwaysSignalAdaptationStrategy()


class EntryFeedbackBootstrapTest(unittest.TestCase):
    def test_bootstrap_entry_feedback_uses_unfiltered_adaptation_strategy(self):
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
                        "warmup_bars = 3",
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
            ts = datetime(2025, 1, 6, 7, 0, tzinfo=timezone.utc)
            candles: list[Candle] = []
            for price in (100.0, 100.4, 100.8, 101.2, 101.6, 102.0):
                candles.append(
                    Candle(
                        timestamp=ts,
                        open=price - 0.2,
                        high=price + 0.6,
                        low=price - 0.4,
                        close=price,
                        volume=1_000_000,
                    )
                )
                ts += timedelta(minutes=30)

            orchestrator = _AdaptationBootstrapOrchestrator(
                config,
                _FakeProvider([instrument], {"SBER": candles}),
            )

            result = orchestrator.bootstrap_entry_feedback()
            feedback = load_signal_feedback(
                signal_feedback_path(config.resolve_path(config.execution.state_path))
            )

            self.assertGreater(result["generated_total"], 0)
            self.assertGreater(result["generated_by_symbol"]["SBER"], 0)
            self.assertGreater(len(feedback["resolved"]), 0)


if __name__ == "__main__":
    unittest.main()
