from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .domain import Candle, Instrument, InstrumentType


class OfflineParquetCacheDependencyError(RuntimeError):
    """Raised when parquet cache dependencies are unavailable."""


INSTRUMENT_METADATA_FILENAME = "instrument_metadata.json"


def _imports():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise OfflineParquetCacheDependencyError(
            "Offline parquet cache support requires pandas and pyarrow."
        ) from exc
    return pd


def load_candle_frame(path: Path):
    pd = _imports()
    if not path.exists():
        return _empty_candle_frame(pd)

    frame = pd.read_parquet(path)
    timestamp_series = _timestamp_series(frame)
    normalized = frame.copy()
    normalized["__timestamp"] = pd.to_datetime(timestamp_series, utc=True)
    columns = [column for column in ("open", "high", "low", "close", "volume") if column in normalized.columns]
    normalized = normalized[columns + ["__timestamp"]]
    normalized = normalized.set_index("__timestamp").sort_index()
    normalized.index.name = "time"
    return normalized


def candles_to_frame(candles: list[Candle]):
    pd = _imports()
    if not candles:
        return _empty_candle_frame(pd)

    frame = pd.DataFrame(
        [
            {
                "time": candle.timestamp.astimezone(timezone.utc),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ]
    )
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time").sort_index()
    frame.index.name = "time"
    return frame


def merge_candle_frames(existing, fresh):
    pd = _imports()
    if existing.empty:
        merged = fresh.copy()
    elif fresh.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, fresh], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    merged.index = pd.to_datetime(merged.index, utc=True)
    merged.index.name = "time"
    return merged


def write_candle_frame(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_index().to_parquet(path)


def instrument_metadata_path(base_path: Path) -> Path:
    return base_path / INSTRUMENT_METADATA_FILENAME


def write_instrument_metadata(base_path: Path, instruments: list[Instrument]) -> Path:
    path = instrument_metadata_path(base_path)
    payload = {
        "instruments": {
            instrument.symbol.upper(): _instrument_to_payload(instrument)
            for instrument in sorted(instruments, key=lambda item: item.symbol.upper())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_instrument_metadata(base_path: Path) -> dict[str, Instrument]:
    path = instrument_metadata_path(base_path)
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_instruments = payload.get("instruments", payload) if isinstance(payload, dict) else payload
    if isinstance(raw_instruments, list):
        pairs = [
            (str(item.get("symbol", "")), item)
            for item in raw_instruments
            if isinstance(item, dict)
        ]
    elif isinstance(raw_instruments, dict):
        pairs = [
            (str(symbol), item)
            for symbol, item in raw_instruments.items()
            if isinstance(item, dict)
        ]
    else:
        return {}

    metadata: dict[str, Instrument] = {}
    for symbol, item in pairs:
        instrument = _instrument_from_payload(item, fallback_symbol=symbol)
        if instrument is not None:
            metadata[instrument.symbol.upper()] = instrument
    return metadata


def latest_candle_timestamp(frame) -> datetime | None:
    if frame.empty:
        return None
    latest = frame.index.max()
    return latest.to_pydatetime().astimezone(timezone.utc)


def incremental_fetch_start(
    latest_timestamp: datetime | None,
    *,
    bootstrap_history_days: int,
    overlap_hours: int,
) -> datetime:
    now = datetime.now(timezone.utc)
    if latest_timestamp is None:
        return now - timedelta(days=max(1, bootstrap_history_days))
    return latest_timestamp - timedelta(hours=max(1, overlap_hours))


def _empty_candle_frame(pd):
    frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame.index = pd.DatetimeIndex([], tz="UTC", name="time")
    return frame


def _instrument_to_payload(instrument: Instrument) -> dict[str, object]:
    return {
        "symbol": instrument.symbol.upper(),
        "instrument_type": instrument.instrument_type.value,
        "figi": instrument.figi,
        "uid": instrument.uid,
        "class_code": instrument.class_code,
        "lot_size": max(1, int(instrument.lot_size or 1)),
        "tick_size": float(instrument.tick_size or 0.01),
        "currency": instrument.currency or "rub",
        "initial_margin_buy": float(instrument.initial_margin_buy or 0.0),
        "initial_margin_sell": float(instrument.initial_margin_sell or 0.0),
        "tick_value": float(instrument.tick_value or 0.0),
    }


def _instrument_from_payload(
    payload: dict[str, object],
    *,
    fallback_symbol: str = "",
) -> Instrument | None:
    symbol = str(payload.get("symbol") or fallback_symbol or "").upper()
    if not symbol:
        return None
    instrument_type_value = str(
        payload.get("instrument_type", InstrumentType.STOCK.value) or InstrumentType.STOCK.value
    )
    try:
        instrument_type = InstrumentType(instrument_type_value)
    except ValueError:
        instrument_type = InstrumentType.STOCK
    return Instrument(
        symbol=symbol,
        instrument_type=instrument_type,
        figi=str(payload.get("figi", "") or ""),
        uid=str(payload.get("uid", "") or ""),
        class_code=str(payload.get("class_code", "") or ""),
        lot_size=max(1, int(payload.get("lot_size", 1) or 1)),
        tick_size=float(payload.get("tick_size", 0.01) or 0.01),
        currency=str(payload.get("currency", "rub") or "rub"),
        initial_margin_buy=float(payload.get("initial_margin_buy", 0.0) or 0.0),
        initial_margin_sell=float(payload.get("initial_margin_sell", 0.0) or 0.0),
        tick_value=float(payload.get("tick_value", 0.0) or 0.0),
    )


def _timestamp_series(frame):
    for key in ("time", "timestamp", "utc_ts", "timestamp_utc", "datetime"):
        if key in frame.columns:
            return frame[key]
    if getattr(frame.index, "name", "") in {"time", "timestamp", "utc_ts", "timestamp_utc", "datetime"}:
        return frame.index
    raise ValueError(
        "Unsupported parquet schema: expected a time/timestamp/utc_ts column or index."
    )
