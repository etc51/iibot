from __future__ import annotations

from samosbor.minimal_dashboard import (
    _gross_exposure,
    _portfolio_equity,
    _positions_from_state,
    _request_path,
)


def _portfolio() -> dict[str, object]:
    return {
        "cash": 347_359.77,
        "positions": {
            "TRNFP": {
                "instrument": {
                    "symbol": "TRNFP",
                    "instrument_type": "stock",
                    "lot_size": 1,
                },
                "direction": "short",
                "quantity_lots": 35,
                "entry_price": 1301.0796,
                "current_price": 1301.6,
                "stop_price": 1306.8,
                "take_profit": 1288.5,
                "runner_active": True,
                "runner_activation_price": 1288.5,
                "runner_extreme_price": 1284.0,
                "signal_strength": 0.25,
                "opened_at": "2026-07-01T14:30:00+00:00",
            }
        },
    }


def test_open_position_uses_live_price_when_available():
    positions = _positions_from_state(_portfolio(), live_prices={"TRNFP": 1290.0})

    assert positions[0]["current_price"] == 1290.0
    assert positions[0]["state_current_price"] == 1301.6
    assert positions[0]["price_source"] == "live"
    assert positions[0]["quantity_units"] == 35
    assert positions[0]["unrealized_pnl_rub"] == 387.79
    assert positions[0]["runner_status"] == "runner"
    assert positions[0]["runner_extreme_price"] == 1284.0


def test_open_position_falls_back_to_state_price():
    positions = _positions_from_state(_portfolio(), live_prices={})

    assert positions[0]["current_price"] == 1301.6
    assert positions[0]["price_source"] == "state"


def test_dashboard_equity_and_exposure_use_units_and_short_sign():
    positions = _positions_from_state(_portfolio(), live_prices={"TRNFP": 1290.0})

    assert _gross_exposure(positions) == 45_150.0
    assert _portfolio_equity(_portfolio(), positions) == 302_209.77


def test_request_path_ignores_refresh_query_string():
    assert _request_path("/?ts=123") == "/"
    assert _request_path("/api/status?ts=123") == "/api/status"
