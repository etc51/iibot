from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from samosbor.config import (
    AppConfig,
    AppSection,
    BacktestSection,
    DataSection,
    ExecutionSection,
    ReportingSection,
    ResearchSection,
    RiskSection,
    StrategySection,
    TBankSection,
)
from samosbor.domain import Instrument, InstrumentType
from samosbor.microstructure_collector import (
    collect_microstructure_snapshot,
    microstructure_latest_path,
)


class FakeOrderBookProvider:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def get_order_book_snapshot(self, instrument, *, depth, quantity_lots=0, direction=""):
        self.calls.append((instrument.symbol, depth))
        return {
            "available": True,
            "symbol": instrument.symbol,
            "depth_requested": depth,
            "depth_returned": 2,
            "timestamp": "2026-07-01T18:00:00+00:00",
            "best_bid": 100.0,
            "best_ask": 100.1,
            "mid_price": 100.05,
            "spread_abs": 0.1,
            "spread_bps": 9.995,
            "bid_depth_lots": 10.0,
            "ask_depth_lots": 11.0,
            "imbalance": -0.0476,
            "bids": [{"price": 100.0, "quantity_lots": 10}],
            "asks": [{"price": 100.1, "quantity_lots": 11}],
        }


def test_collect_microstructure_snapshot_writes_all_configured_instruments(tmp_path: Path):
    config = _config(tmp_path)
    provider = FakeOrderBookProvider()

    payload = collect_microstructure_snapshot(
        config,
        provider=provider,
        depth=10,
        collected_at=datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc),
    )

    assert provider.calls == [("SBER", 10), ("MTLR", 10)]
    assert payload["symbols_total"] == 2
    assert payload["symbols_ok"] == 2
    assert payload["symbols_error"] == 0

    sber_path = tmp_path / "runs" / "microstructure" / "20260701" / "SBER.jsonl"
    mtlr_path = tmp_path / "runs" / "microstructure" / "20260701" / "MTLR.jsonl"
    assert sber_path.exists()
    assert mtlr_path.exists()
    sber_row = json.loads(sber_path.read_text(encoding="utf-8").splitlines()[0])
    assert sber_row["symbol"] == "SBER"
    assert sber_row["ok"] is True
    assert sber_row["spread_bps"] == 9.995

    latest = json.loads(microstructure_latest_path(config).read_text(encoding="utf-8"))
    assert latest["rows"]["MTLR"]["best_bid"] == 100.0


def _config(root: Path) -> AppConfig:
    return AppConfig(
        root_dir=root,
        app=AppSection(),
        tbank=TBankSection(),
        data=DataSection(
            instruments=[
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, uid="sber"),
                Instrument(symbol="MTLR", instrument_type=InstrumentType.STOCK, uid="mtlr"),
            ]
        ),
        strategy=StrategySection(order_book_depth=10),
        risk=RiskSection(),
        execution=ExecutionSection(state_path="state/demo_state.json"),
        backtest=BacktestSection(),
        reporting=ReportingSection(output_dir="runs"),
        research=ResearchSection(),
    )
