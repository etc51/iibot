from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .config import AppConfig
from .data.tbank import TBankMarketDataProvider
from .domain import Instrument
from .reporting.writer import write_json_payload
from .runtime_metadata import current_commit_hash


class OrderBookProvider(Protocol):
    def get_order_book_snapshot(
        self,
        instrument: Instrument,
        *,
        depth: int,
        quantity_lots: int = 0,
        direction: str = "",
    ) -> dict[str, object]:
        ...


def microstructure_latest_path(config: AppConfig) -> Path:
    return config.resolve_path(config.execution.state_path).with_name(
        f"{config.runtime_profile_name()}_microstructure_latest.json"
    )


def collect_microstructure_snapshot(
    config: AppConfig,
    *,
    provider: OrderBookProvider | None = None,
    output_dir: str | Path | None = None,
    depth: int | None = None,
    collected_at: datetime | None = None,
) -> dict[str, object]:
    provider = provider or TBankMarketDataProvider(config)
    collected_at = collected_at or datetime.now(timezone.utc)
    normalized_depth = max(1, min(50, int(depth or config.strategy.order_book_depth or 10)))
    output_root = _output_root(config, output_dir)
    day_dir = output_root / collected_at.strftime("%Y%m%d")

    rows: dict[str, dict[str, object]] = {}
    ok_count = 0
    error_count = 0
    for instrument in config.data.instruments:
        row = _collect_one(
            provider,
            instrument,
            depth=normalized_depth,
            collected_at=collected_at,
        )
        if row.get("ok"):
            ok_count += 1
        else:
            error_count += 1
        rows[instrument.symbol] = row
        _append_jsonl(day_dir / f"{instrument.symbol}.jsonl", row)

    payload = {
        "generated_at": collected_at.isoformat(),
        "commit_hash": current_commit_hash(),
        "output_dir": str(output_root),
        "depth": normalized_depth,
        "symbols_total": len(config.data.instruments),
        "symbols_ok": ok_count,
        "symbols_error": error_count,
        "rows": rows,
    }
    write_json_payload(output_root / "latest.json", payload)
    write_json_payload(microstructure_latest_path(config), payload)
    return payload


def run_microstructure_collector(
    config: AppConfig,
    *,
    interval_sec: float,
    output_dir: str | Path | None = None,
    depth: int | None = None,
    cycles: int | None = None,
) -> dict[str, object]:
    logging.getLogger("t_tech.invest.logging").setLevel(logging.WARNING)
    completed = 0
    last_payload: dict[str, object] = {}
    while cycles is None or completed < cycles:
        started = time.monotonic()
        last_payload = collect_microstructure_snapshot(
            config,
            output_dir=output_dir,
            depth=depth,
        )
        completed += 1
        if cycles is not None and completed >= cycles:
            break
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, float(interval_sec) - elapsed))
    return last_payload


def _collect_one(
    provider: OrderBookProvider,
    instrument: Instrument,
    *,
    depth: int,
    collected_at: datetime,
) -> dict[str, object]:
    base: dict[str, object] = {
        "collected_at": collected_at.isoformat(),
        "symbol": instrument.symbol,
        "instrument_type": instrument.instrument_type.value,
        "figi": instrument.figi,
        "uid": instrument.uid,
        "lot_size": instrument.lot_size,
        "ok": False,
        "error": "",
    }
    try:
        snapshot = provider.get_order_book_snapshot(instrument, depth=depth)
    except Exception as exc:  # pragma: no cover - live API behavior
        base["error"] = str(exc)
        base["available"] = False
        return base

    base.update(snapshot)
    base["ok"] = bool(snapshot.get("available"))
    return base


def _output_root(config: AppConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
        return path if path.is_absolute() else config.root_dir / path
    return config.resolve_path(config.reporting.output_dir) / "microstructure"


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
