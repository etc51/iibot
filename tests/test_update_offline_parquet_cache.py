from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from samosbor.domain import Candle, Instrument, InstrumentType
from samosbor.offline_parquet_cache import (
    candles_to_frame,
    load_instrument_metadata,
    write_candle_frame,
)


def _load_update_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "server" / "update-offline-parquet-cache.py"
    spec = importlib.util.spec_from_file_location("test_update_offline_parquet_cache_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UpdateOfflineParquetCacheTest(unittest.TestCase):
    def test_main_uses_config_history_days_for_bootstrap_by_default(self):
        module = _load_update_script()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "paper.toml"
            config_path.write_text("", encoding="utf-8")
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            config = _DummyConfig(root, [instrument], history_days=17)

            class Provider:
                def __init__(self, _config):
                    self._config = _config

                def resolve_universe(self, instruments):
                    return list(instruments)

                def get_candles_range(self, instrument, *, timeframe, from_dt):
                    return []

            stdout = io.StringIO()
            with (
                patch.object(module, "load_config", return_value=config),
                patch.object(module, "TBankMarketDataProvider", Provider),
                patch.object(
                    module,
                    "incremental_fetch_start",
                    return_value=datetime(2026, 6, 1, tzinfo=timezone.utc),
                ) as incremental_fetch_start,
                patch.object(sys, "argv", ["prog", "--config", str(config_path), "--parquet-dir", "data"]),
                contextlib.redirect_stdout(stdout),
            ):
                rc = module.main()

            self.assertEqual(rc, 0)
            incremental_fetch_start.assert_called_once_with(
                None,
                bootstrap_history_days=17,
                overlap_hours=48,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["bootstrap_history_days"], 17)

    def test_main_returns_nonzero_when_bootstrap_fetch_fails_without_cache(self):
        module = _load_update_script()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "paper.toml"
            config_path.write_text("", encoding="utf-8")
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            config = _DummyConfig(root, [instrument])

            class FailingProvider:
                def __init__(self, _config):
                    self._config = _config

                def resolve_universe(self, instruments):
                    return list(instruments)

                def get_candles_range(self, instrument, *, timeframe, from_dt):
                    raise RuntimeError(f"cannot fetch {instrument.symbol}")

            stdout = io.StringIO()
            with (
                patch.object(module, "load_config", return_value=config),
                patch.object(module, "TBankMarketDataProvider", FailingProvider),
                patch.object(sys, "argv", ["prog", "--config", str(config_path), "--parquet-dir", "data"]),
                contextlib.redirect_stdout(stdout),
            ):
                rc = module.main()

            self.assertEqual(rc, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["safe_to_continue"])
            self.assertEqual(payload["missing_symbols"], ["SBER"])
            self.assertEqual(payload["results"][0]["status"], "bootstrap-failed")

    def test_main_reuses_existing_cache_when_fetch_fails(self):
        module = _load_update_script()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "paper.toml"
            config_path.write_text("", encoding="utf-8")
            instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            config = _DummyConfig(root, [instrument])
            parquet_path = root / "data" / "SBER_30min.parquet"
            write_candle_frame(
                parquet_path,
                candles_to_frame(
                    [
                        Candle(
                            datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc),
                            100.0,
                            101.0,
                            99.0,
                            100.5,
                            10.0,
                        )
                    ]
                ),
            )

            class FailingProvider:
                def __init__(self, _config):
                    self._config = _config

                def resolve_universe(self, instruments):
                    return list(instruments)

                def get_candles_range(self, instrument, *, timeframe, from_dt):
                    raise RuntimeError(f"cannot fetch {instrument.symbol}")

            stdout = io.StringIO()
            with (
                patch.object(module, "load_config", return_value=config),
                patch.object(module, "TBankMarketDataProvider", FailingProvider),
                patch.object(sys, "argv", ["prog", "--config", str(config_path), "--parquet-dir", "data"]),
                contextlib.redirect_stdout(stdout),
            ):
                rc = module.main()

            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["safe_to_continue"])
            self.assertEqual(payload["missing_symbols"], [])
            self.assertEqual(payload["results"][0]["status"], "stale-cache-reused")

    def test_main_writes_instrument_metadata_sidecar(self):
        module = _load_update_script()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "paper.toml"
            config_path.write_text("", encoding="utf-8")
            configured = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            resolved = Instrument(
                symbol="SBER",
                instrument_type=InstrumentType.STOCK,
                uid="uid-sber",
                figi="figi-sber",
                class_code="TQBR",
                lot_size=10,
                tick_size=0.01,
            )
            config = _DummyConfig(root, [configured])

            class Provider:
                def __init__(self, _config):
                    self._config = _config

                def resolve_universe(self, instruments):
                    return [resolved]

                def get_candles_range(self, instrument, *, timeframe, from_dt):
                    return []

            stdout = io.StringIO()
            with (
                patch.object(module, "load_config", return_value=config),
                patch.object(module, "TBankMarketDataProvider", Provider),
                patch.object(sys, "argv", ["prog", "--config", str(config_path), "--parquet-dir", "data"]),
                contextlib.redirect_stdout(stdout),
            ):
                rc = module.main()

            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["instrument_metadata_path"].endswith("instrument_metadata.json"))
            metadata = load_instrument_metadata(root / "data")
            self.assertEqual(metadata["SBER"].uid, "uid-sber")
            self.assertEqual(metadata["SBER"].lot_size, 10)


class _DummyConfig:
    def __init__(self, root: Path, instruments: list[object], *, history_days: int = 30):
        self._root = root
        self.data = SimpleNamespace(timeframe="30min", history_days=history_days, instruments=instruments)

    def resolve_path(self, path: str) -> Path:
        return (self._root / path).resolve()

    def autotune_dir(self) -> Path:
        return self._root / "runs" / "autotune" / "demo"


if __name__ == "__main__":
    unittest.main()
