from __future__ import annotations

from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path

from ..domain import Candle, Instrument
from ..offline_parquet_cache import load_instrument_metadata


class ParquetDirectoryDependencyError(RuntimeError):
    """Raised when parquet-directory dependencies are unavailable."""


def _imports():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ParquetDirectoryDependencyError(
            "Parquet directory support requires pandas and pyarrow."
        ) from exc
    return pd


class ParquetDirectoryProvider:
    def __init__(self, base_path: Path, *, timeframe: str, history_days: int):
        self.base_path = base_path
        self.timeframe = timeframe
        self.history_days = history_days
        self._instrument_metadata = load_instrument_metadata(base_path)

    def resolve_universe(self, instruments: list[Instrument]) -> list[Instrument]:
        return [self.resolve_instrument(instrument) for instrument in instruments]

    def resolve_instrument(self, instrument: Instrument) -> Instrument:
        if not self._path_for_symbol(instrument.symbol).exists():
            raise FileNotFoundError(
                f"Parquet file not found for symbol {instrument.symbol}: {self._path_for_symbol(instrument.symbol)}"
            )
        metadata = self._instrument_metadata.get(instrument.symbol.upper())
        if metadata is None:
            return instrument
        return self._merge_instrument_metadata(instrument, metadata)

    def load_history(self, instruments: list[Instrument]) -> dict[str, list[Candle]]:
        resolved = self.resolve_universe(instruments)
        return {instrument.symbol: self.get_candles(instrument) for instrument in resolved}

    def get_candles(self, instrument: Instrument) -> list[Candle]:
        pd = _imports()
        path = self._path_for_symbol(instrument.symbol)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found for symbol {instrument.symbol}: {path}")

        frame = pd.read_parquet(path)
        timestamp_series = self._timestamp_series(frame)
        frame = frame.copy()
        frame["__timestamp"] = pd.to_datetime(timestamp_series, utc=True)
        frame = frame.sort_values("__timestamp")

        if self.history_days > 0 and not frame.empty:
            cutoff = frame["__timestamp"].max() - timedelta(days=self.history_days)
            frame = frame[frame["__timestamp"] >= cutoff]

        candles = [
            Candle(
                timestamp=row["__timestamp"].to_pydatetime().astimezone(timezone.utc),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in frame.to_dict(orient="records")
        ]
        return candles

    def _path_for_symbol(self, symbol: str) -> Path:
        suffix = self.timeframe.lower()
        return self.base_path / f"{symbol.upper()}_{suffix}.parquet"

    @staticmethod
    def _merge_instrument_metadata(instrument: Instrument, metadata: Instrument) -> Instrument:
        return replace(
            instrument,
            figi=metadata.figi or instrument.figi,
            uid=metadata.uid or instrument.uid,
            class_code=metadata.class_code or instrument.class_code,
            lot_size=max(1, int(metadata.lot_size or instrument.lot_size or 1)),
            tick_size=float(metadata.tick_size or instrument.tick_size or 0.01),
            currency=metadata.currency or instrument.currency,
            initial_margin_buy=float(
                metadata.initial_margin_buy or instrument.initial_margin_buy or 0.0
            ),
            initial_margin_sell=float(
                metadata.initial_margin_sell or instrument.initial_margin_sell or 0.0
            ),
            tick_value=float(metadata.tick_value or instrument.tick_value or 0.0),
        )

    @staticmethod
    def _timestamp_series(frame):
        for key in ("time", "timestamp", "utc_ts", "timestamp_utc", "datetime"):
            if key in frame.columns:
                return frame[key]
        if getattr(frame.index, "name", "") in {"time", "timestamp", "utc_ts", "timestamp_utc", "datetime"}:
            return frame.index
        raise ValueError(
            "Unsupported parquet schema: expected a time/timestamp/utc_ts column or index."
        )
