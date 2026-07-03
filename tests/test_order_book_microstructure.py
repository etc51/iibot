from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from samosbor.data.tbank import order_book_snapshot_from_response


@dataclass
class _Order:
    price: float
    quantity: int


@dataclass
class _OrderBook:
    bids: list[_Order]
    asks: list[_Order]
    orderbook_ts: datetime


def test_order_book_snapshot_calculates_entry_microstructure_for_long():
    response = _OrderBook(
        bids=[_Order(99.9, 10), _Order(99.8, 20)],
        asks=[_Order(100.1, 5), _Order(100.2, 15)],
        orderbook_ts=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
    )

    snapshot = order_book_snapshot_from_response(
        response,
        symbol="SBER",
        lot_size=10,
        depth=2,
        quantity_lots=10,
        direction="long",
        quotation_to_float=float,
    )

    assert snapshot["available"] is True
    assert snapshot["best_bid"] == 99.9
    assert snapshot["best_ask"] == 100.1
    assert snapshot["spread_bps"] == 20.0
    assert snapshot["bid_depth_lots"] == 30
    assert snapshot["ask_depth_lots"] == 20
    assert snapshot["entry_depth_lots"] == 20
    assert snapshot["entry_liquidity_cover"] == 2.0
    assert snapshot["imbalance"] == 0.2
    assert snapshot["side_imbalance"] == 0.2


def test_order_book_snapshot_calculates_entry_microstructure_for_short():
    response = _OrderBook(
        bids=[_Order(99.9, 10), _Order(99.8, 20)],
        asks=[_Order(100.1, 5), _Order(100.2, 15)],
        orderbook_ts=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
    )

    snapshot = order_book_snapshot_from_response(
        response,
        symbol="SBER",
        lot_size=10,
        depth=2,
        quantity_lots=15,
        direction="short",
        quotation_to_float=float,
    )

    assert snapshot["entry_depth_lots"] == 30
    assert snapshot["entry_liquidity_cover"] == 2.0
    assert snapshot["side_imbalance"] == -0.2
    assert snapshot["best_executable_price"] == 99.9
