from __future__ import annotations

import logging
import os
import tempfile
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from ..config import AppConfig, read_secret_from_env_or_file
from ..domain import Candle, Instrument, InstrumentType

LOGGER = logging.getLogger(__name__)
_REQUEST_PACE_LOCK = Lock()
_DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.5
_RATE_LIMIT_LOCK_STALE_SECONDS = 30.0


class TBankDependencyError(RuntimeError):
    """Raised when the T-Bank SDK is unavailable."""


def _sdk_imports():
    try:
        from t_tech.invest import CandleInterval, Client, InstrumentType as TInstrumentType
        from t_tech.invest.exceptions import RequestError
        from t_tech.invest.utils import now, quotation_to_decimal
    except ImportError as exc:  # pragma: no cover - depends on external package
        raise TBankDependencyError(
            "T-Bank SDK is not installed. Install requirements-tbank.txt first."
        ) from exc
    return CandleInterval, Client, TInstrumentType, RequestError, now, quotation_to_decimal


def timeframe_to_tbank_interval(timeframe: str):
    candle_interval, _, _, _, _, _ = _sdk_imports()
    mapping = {
        "day": candle_interval.CANDLE_INTERVAL_DAY,
        "hour": candle_interval.CANDLE_INTERVAL_HOUR,
        "30min": candle_interval.CANDLE_INTERVAL_30_MIN,
        "15min": candle_interval.CANDLE_INTERVAL_15_MIN,
        "10min": candle_interval.CANDLE_INTERVAL_10_MIN,
        "5min": candle_interval.CANDLE_INTERVAL_5_MIN,
        "1min": candle_interval.CANDLE_INTERVAL_1_MIN,
    }
    try:
        return mapping[timeframe.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def timeframe_to_duration(timeframe: str) -> timedelta:
    mapping = {
        "day": timedelta(days=1),
        "hour": timedelta(hours=1),
        "30min": timedelta(minutes=30),
        "15min": timedelta(minutes=15),
        "10min": timedelta(minutes=10),
        "5min": timedelta(minutes=5),
        "1min": timedelta(minutes=1),
    }
    try:
        return mapping[timeframe.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def drop_incomplete_trailing_candle(
    candles: list[Candle],
    *,
    timeframe: str,
    as_of: datetime,
) -> list[Candle]:
    if not candles:
        return []
    duration = timeframe_to_duration(timeframe)
    last_candle = candles[-1]
    if last_candle.timestamp + duration > as_of:
        return candles[:-1]
    return candles


class TBankMarketDataProvider:
    def __init__(self, config: AppConfig):
        self.config = config

    def _token(self) -> str:
        return read_secret_from_env_or_file(
            self.config.tbank.token_env,
            self.config.tbank.token_file,
            label="T-Bank invest token",
        )

    @contextmanager
    def _client(self) -> Iterator[object]:
        _, client_cls, _, _, _, _ = _sdk_imports()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with client_cls(self._token(), app_name=self.config.tbank.app_name) as client:
                yield client

    def list_accounts(self) -> list[dict[str, str]]:
        with self._client() as client:
            response = client.users.get_accounts()
            return [
                {
                    "id": account.id,
                    "name": account.name,
                    "status": str(account.status),
                    "type": str(account.type),
                    "access_level": str(account.access_level),
                }
                for account in response.accounts
            ]

    def resolve_universe(self, instruments: list[Instrument]) -> list[Instrument]:
        return [self.resolve_instrument(instrument) for instrument in instruments]

    def resolve_instrument(self, instrument: Instrument) -> Instrument:
        if instrument.uid or instrument.figi:
            return instrument

        _, _, tbank_instrument_type, _, _, quotation_to_decimal = _sdk_imports()
        try:
            from t_tech.invest.utils import money_to_decimal
        except ImportError as exc:  # pragma: no cover - depends on external package
            raise TBankDependencyError(
                "T-Bank SDK is not installed. Install requirements-tbank.txt first."
            ) from exc
        try:
            from t_tech.invest import InstrumentIdType
        except ImportError as exc:  # pragma: no cover - depends on external package
            raise TBankDependencyError(
                "T-Bank SDK is not installed. Install requirements-tbank.txt first."
            ) from exc
        kind_map = {
            InstrumentType.STOCK: tbank_instrument_type.INSTRUMENT_TYPE_SHARE,
            InstrumentType.FUTURE: tbank_instrument_type.INSTRUMENT_TYPE_FUTURES,
        }

        with self._client() as client:
            response = self._call_with_retries(
                lambda: client.instruments.find_instrument(
                    query=instrument.symbol,
                    instrument_kind=kind_map[instrument.instrument_type],
                    api_trade_available_flag=True,
                ),
                context=f"find_instrument:{instrument.symbol}",
            )

        exact_match = None
        for candidate in response.instruments:
            if candidate.ticker.upper() == instrument.symbol.upper():
                exact_match = candidate
                break
        candidate = exact_match or (response.instruments[0] if response.instruments else None)
        if candidate is None:
            raise LookupError(f"Instrument not found for symbol {instrument.symbol}")

        with self._client() as client:
            if instrument.instrument_type == InstrumentType.STOCK:
                full = self._call_with_retries(
                    lambda: client.instruments.share_by(
                        id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_UID,
                        id=candidate.uid,
                    ),
                    context=f"share_by:{instrument.symbol}",
                ).instrument
                margin = None
            else:
                full = self._call_with_retries(
                    lambda: client.instruments.future_by(
                        id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_UID,
                        id=candidate.uid,
                    ),
                    context=f"future_by:{instrument.symbol}",
                ).instrument
                margin = self._call_with_retries(
                    lambda: client.instruments.get_futures_margin(instrument_id=candidate.uid),
                    context=f"get_futures_margin:{instrument.symbol}",
                )

        resolved = Instrument(
            symbol=instrument.symbol,
            instrument_type=instrument.instrument_type,
            figi=candidate.figi,
            uid=candidate.uid,
            class_code=getattr(full, "class_code", getattr(candidate, "class_code", "")),
            lot_size=int(getattr(full, "lot", instrument.lot_size or 1)),
            tick_size=float(quotation_to_decimal(full.min_price_increment)),
            currency=getattr(full, "currency", instrument.currency),
            initial_margin_buy=(
                float(money_to_decimal(margin.initial_margin_on_buy)) if margin is not None else 0.0
            ),
            initial_margin_sell=(
                float(money_to_decimal(margin.initial_margin_on_sell)) if margin is not None else 0.0
            ),
            tick_value=(
                float(quotation_to_decimal(margin.min_price_increment_amount))
                if margin is not None
                else 0.0
            ),
        )
        LOGGER.info(
            "Resolved %s as uid=%s figi=%s lot=%s margin_buy=%.2f margin_sell=%.2f",
            resolved.symbol,
            resolved.uid,
            resolved.figi,
            resolved.lot_size,
            resolved.initial_margin_buy,
            resolved.initial_margin_sell,
        )
        return resolved

    def get_candles(
        self,
        instrument: Instrument,
        *,
        timeframe: str,
        history_days: int,
    ) -> list[Candle]:
        _, _, _, _, now_fn, _ = _sdk_imports()
        to_dt = now_fn()
        from_dt = to_dt - timedelta(days=history_days)
        return self.get_candles_range(
            instrument,
            timeframe=timeframe,
            from_dt=from_dt,
            to_dt=to_dt,
        )

    def get_candles_range(
        self,
        instrument: Instrument,
        *,
        timeframe: str,
        from_dt: datetime,
        to_dt: datetime | None = None,
    ) -> list[Candle]:
        interval = timeframe_to_tbank_interval(timeframe)
        _, _, _, _, now_fn, quotation_to_decimal = _sdk_imports()
        if to_dt is None:
            to_dt = now_fn()

        candle_source = _tbank_candle_source(self.config.data.tbank_candle_source)
        response_items: list[Any] = []
        with self._client() as client:
            for chunk_from, chunk_to in _candle_request_ranges(from_dt, to_dt, timeframe):
                response = self._call_with_retries(
                    lambda chunk_from=chunk_from, chunk_to=chunk_to: client.market_data.get_candles(
                        instrument_id=instrument.instrument_id,
                        from_=chunk_from,
                        to=chunk_to,
                        interval=interval,
                        candle_source_type=candle_source,
                    ).candles,
                    context=f"get_candles:{instrument.symbol}:{chunk_from.isoformat()}",
                )
                response_items.extend(list(response))

        unique_items = {item.time: item for item in response_items}
        candles = [
            Candle(
                timestamp=item.time,
                open=float(quotation_to_decimal(item.open)),
                high=float(quotation_to_decimal(item.high)),
                low=float(quotation_to_decimal(item.low)),
                close=float(quotation_to_decimal(item.close)),
                volume=float(item.volume),
            )
            for item in unique_items.values()
        ]
        candles = sorted(candles, key=lambda candle: candle.timestamp)
        return drop_incomplete_trailing_candle(candles, timeframe=timeframe, as_of=to_dt)

    def load_history(self, instruments: list[Instrument]) -> dict[str, list[Candle]]:
        resolved = self.resolve_universe(instruments)
        return {
            instrument.symbol: self.get_candles(
                instrument,
                timeframe=self.config.data.timeframe,
                history_days=self.config.data.history_days,
            )
            for instrument in resolved
        }

    def load_history_for_timeframe(
        self,
        instruments: list[Instrument],
        timeframe: str,
        *,
        history_days: int | None = None,
    ) -> dict[str, list[Candle]]:
        resolved = self.resolve_universe(instruments)
        requested_days = (
            max(1, int(history_days))
            if history_days is not None
            else _secondary_timeframe_history_days(
                primary_timeframe=self.config.data.timeframe,
                secondary_timeframe=timeframe,
                configured_days=self.config.data.history_days,
            )
        )
        return {
            instrument.symbol: self.get_candles(
                instrument,
                timeframe=timeframe,
                history_days=requested_days,
            )
            for instrument in resolved
        }

    def get_last_prices(self, instruments: list[Instrument]) -> dict[str, float]:
        _, _, _, _, _, quotation_to_decimal = _sdk_imports()
        resolved = self.resolve_universe(instruments)
        with self._client() as client:
            response = self._call_with_retries(
                lambda: client.market_data.get_last_prices(
                    instrument_id=[instrument.instrument_id for instrument in resolved],
                ),
                context="get_last_prices",
            )

        price_map = {}
        for instrument, item in zip(resolved, response.last_prices):
            price_map[instrument.symbol] = float(quotation_to_decimal(item.price))
        return price_map

    def get_order_book_snapshot(
        self,
        instrument: Instrument,
        *,
        depth: int,
        quantity_lots: int = 0,
        direction: str = "",
    ) -> dict[str, object]:
        _, _, _, _, _, quotation_to_decimal = _sdk_imports()
        resolved = self.resolve_instrument(instrument)
        normalized_depth = max(1, min(50, int(depth or 1)))
        with self._client() as client:
            response = self._call_with_retries(
                lambda: client.market_data.get_order_book(
                    instrument_id=resolved.instrument_id,
                    depth=normalized_depth,
                ),
                context=f"get_order_book:{resolved.symbol}",
            )
        return order_book_snapshot_from_response(
            response,
            symbol=resolved.symbol,
            lot_size=resolved.lot_size,
            depth=normalized_depth,
            quantity_lots=quantity_lots,
            direction=direction,
            quotation_to_float=lambda value: float(quotation_to_decimal(value)),
        )

    @staticmethod
    def _call_with_retries(action, *, context: str, attempts: int = 6, wait_seconds: float = 2.0):
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                _pace_tbank_request()
                return action()
            except Exception as exc:  # pragma: no cover - depends on live API failures
                last_error = exc
                if attempt >= attempts:
                    break
                retry_wait = max(wait_seconds, _retry_wait_seconds(exc))
                LOGGER.warning(
                    "T-Bank request failed, retrying %s/%s after %.1fs: %s (%s)",
                    attempt,
                    attempts,
                    retry_wait,
                    context,
                    exc,
                )
                time.sleep(retry_wait)
        assert last_error is not None
        raise last_error


def _candle_request_ranges(from_dt: datetime, to_dt: datetime, timeframe: str) -> list[tuple[datetime, datetime]]:
    if to_dt <= from_dt:
        return []
    step = _candle_request_window(timeframe)
    ranges: list[tuple[datetime, datetime]] = []
    current = from_dt
    while current < to_dt:
        chunk_to = min(current + step, to_dt)
        ranges.append((current, chunk_to))
        current = chunk_to
    return ranges


def _candle_request_window(timeframe: str) -> timedelta:
    normalized = timeframe.lower()
    if normalized in {"1min", "5min", "10min", "15min", "30min"}:
        return timedelta(days=1)
    if normalized == "hour":
        return timedelta(days=7)
    if normalized == "day":
        return timedelta(days=365)
    return timeframe_to_duration(timeframe)


def _secondary_timeframe_history_days(
    *,
    primary_timeframe: str,
    secondary_timeframe: str,
    configured_days: int,
) -> int:
    configured_days = max(1, int(configured_days or 1))
    if timeframe_to_duration(secondary_timeframe) < timeframe_to_duration(primary_timeframe):
        return min(configured_days, 4)
    return configured_days


def _pace_tbank_request() -> None:
    min_interval = _min_request_interval_seconds()
    if min_interval <= 0:
        return
    with _REQUEST_PACE_LOCK:
        _pace_tbank_request_across_processes(min_interval)


def _min_request_interval_seconds() -> float:
    raw_value = os.environ.get("SAMOSBOR_TBANK_MIN_REQUEST_INTERVAL_SEC", "")
    if not raw_value:
        return _DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return _DEFAULT_MIN_REQUEST_INTERVAL_SECONDS


def _pace_tbank_request_across_processes(min_interval: float) -> None:
    state_path = _rate_limit_state_path()
    lock_dir = _rate_limit_lock_dir(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _acquire_rate_limit_lock(lock_dir)
    try:
        now = time.time()
        last_request_at = _read_rate_limit_timestamp(state_path)
        if last_request_at is not None and last_request_at <= now + _RATE_LIMIT_LOCK_STALE_SECONDS:
            wait_seconds = min_interval - (now - last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        state_path.write_text(f"{time.time():.9f}", encoding="ascii")
    finally:
        _release_rate_limit_lock(lock_dir)


def _rate_limit_state_path() -> Path:
    raw_path = os.environ.get("SAMOSBOR_TBANK_RATE_LIMIT_PATH", "")
    if raw_path:
        return Path(raw_path).expanduser()
    return Path(tempfile.gettempdir()) / "samosbor_tbank_rate_limit.state"


def _rate_limit_lock_dir(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.name}.lock")


def _read_rate_limit_timestamp(state_path: Path) -> float | None:
    try:
        raw_value = state_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _acquire_rate_limit_lock(lock_dir: Path) -> None:
    deadline = time.monotonic() + _RATE_LIMIT_LOCK_STALE_SECONDS
    while True:
        try:
            os.mkdir(lock_dir)
            return
        except FileExistsError:
            _remove_stale_rate_limit_lock(lock_dir)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"T-Bank rate-limit lock is busy: {lock_dir}")
            time.sleep(0.05)


def _remove_stale_rate_limit_lock(lock_dir: Path) -> None:
    try:
        age = time.time() - lock_dir.stat().st_mtime
    except OSError:
        return
    if age < _RATE_LIMIT_LOCK_STALE_SECONDS:
        return
    try:
        os.rmdir(lock_dir)
    except OSError:
        return


def _release_rate_limit_lock(lock_dir: Path) -> None:
    try:
        os.rmdir(lock_dir)
    except OSError:
        return


def _retry_wait_seconds(exc: Exception) -> float:
    text = repr(exc)
    if "RESOURCE_EXHAUSTED" not in text:
        return 2.0
    reset = _metadata_value(text, "ratelimit_reset")
    if reset is None:
        return 65.0
    try:
        return min(120.0, max(5.0, float(reset) + 2.0))
    except ValueError:
        return 65.0


def _metadata_value(text: str, name: str) -> str | None:
    marker = f"{name}='"
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = text.find("'", start)
    if end < 0:
        return None
    return text[start:end]


def _tbank_candle_source(value: str):
    normalized = str(value or "").strip().lower().replace("_", "-")
    if not normalized:
        return None
    from t_tech.invest.schemas import CandleSource

    mapping = {
        "unspecified": CandleSource.CANDLE_SOURCE_UNSPECIFIED,
        "exchange": CandleSource.CANDLE_SOURCE_EXCHANGE,
        "dealer-weekend": CandleSource.CANDLE_SOURCE_DEALER_WEEKEND,
        "include-weekend": CandleSource.CANDLE_SOURCE_INCLUDE_WEEKEND,
        "all": CandleSource.CANDLE_SOURCE_INCLUDE_WEEKEND,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported T-Bank candle source: {value}") from exc


def order_book_snapshot_from_response(
    response: Any,
    *,
    symbol: str,
    lot_size: int,
    depth: int,
    quantity_lots: int = 0,
    direction: str = "",
    quotation_to_float,
) -> dict[str, object]:
    bids = _order_levels(response, "bids", quotation_to_float)
    asks = _order_levels(response, "asks", quotation_to_float)
    best_bid = bids[0]["price"] if bids else 0.0
    best_ask = asks[0]["price"] if asks else 0.0
    mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
    spread_abs = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
    spread_bps = spread_abs / mid_price * 10_000 if mid_price > 0 else 0.0

    bid_depth_lots = sum(float(level["quantity_lots"]) for level in bids)
    ask_depth_lots = sum(float(level["quantity_lots"]) for level in asks)
    bid_depth_rub = sum(float(level["price"]) * float(level["quantity_lots"]) * lot_size for level in bids)
    ask_depth_rub = sum(float(level["price"]) * float(level["quantity_lots"]) * lot_size for level in asks)
    total_depth = bid_depth_lots + ask_depth_lots
    imbalance = (bid_depth_lots - ask_depth_lots) / total_depth if total_depth > 0 else 0.0

    requested_lots = max(0, int(quantity_lots or 0))
    normalized_direction = str(direction).strip().lower()
    if normalized_direction == "long":
        entry_depth_lots = ask_depth_lots
        entry_depth_rub = ask_depth_rub
        best_executable_price = best_ask
        side_imbalance = imbalance
        estimated_spread_cost_bps = (best_ask - mid_price) / mid_price * 10_000 if mid_price > 0 else 0.0
    elif normalized_direction == "short":
        entry_depth_lots = bid_depth_lots
        entry_depth_rub = bid_depth_rub
        best_executable_price = best_bid
        side_imbalance = -imbalance
        estimated_spread_cost_bps = (mid_price - best_bid) / mid_price * 10_000 if mid_price > 0 else 0.0
    else:
        entry_depth_lots = 0.0
        entry_depth_rub = 0.0
        best_executable_price = 0.0
        side_imbalance = 0.0
        estimated_spread_cost_bps = 0.0

    entry_liquidity_cover = entry_depth_lots / requested_lots if requested_lots > 0 else 0.0
    timestamp = getattr(response, "orderbook_ts", None)
    return {
        "available": bool(bids and asks),
        "symbol": symbol,
        "depth_requested": int(depth),
        "depth_returned": max(len(bids), len(asks)),
        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else "",
        "best_bid": round(best_bid, 8),
        "best_ask": round(best_ask, 8),
        "mid_price": round(mid_price, 8),
        "spread_abs": round(spread_abs, 8),
        "spread_bps": round(spread_bps, 4),
        "estimated_spread_cost_bps": round(max(0.0, estimated_spread_cost_bps), 4),
        "bid_depth_lots": round(bid_depth_lots, 4),
        "ask_depth_lots": round(ask_depth_lots, 4),
        "bid_depth_rub": round(bid_depth_rub, 2),
        "ask_depth_rub": round(ask_depth_rub, 2),
        "imbalance": round(imbalance, 4),
        "side_imbalance": round(side_imbalance, 4),
        "requested_lots": requested_lots,
        "entry_depth_lots": round(entry_depth_lots, 4),
        "entry_depth_rub": round(entry_depth_rub, 2),
        "entry_liquidity_cover": round(entry_liquidity_cover, 4),
        "best_executable_price": round(best_executable_price, 8),
        "bids": bids,
        "asks": asks,
    }


def _order_levels(response: Any, side: str, quotation_to_float) -> list[dict[str, object]]:
    levels = []
    for item in list(getattr(response, side, []) or []):
        price = float(quotation_to_float(getattr(item, "price")))
        quantity = int(getattr(item, "quantity", 0) or 0)
        if price <= 0 or quantity <= 0:
            continue
        levels.append({"price": round(price, 8), "quantity_lots": quantity})
    return levels
