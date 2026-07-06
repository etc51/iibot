from __future__ import annotations

from samosbor.minimal_dashboard import (
    _dashboard_trades,
    _display_exit_reason,
    _gross_exposure,
    _portfolio_equity,
    _positions_from_state,
    _request_path,
    _trades_table,
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


def test_zero_lot_position_is_hidden_from_open_positions():
    portfolio = _portfolio()
    portfolio["positions"]["ROSN"] = {
        "instrument": {
            "symbol": "ROSN",
            "instrument_type": "stock",
            "lot_size": 10,
        },
        "direction": "short",
        "quantity_lots": 0,
        "entry_price": 301.6293,
        "current_price": 300.9,
        "stop_price": 305.6446,
        "take_profit": 292.0134,
    }

    positions = _positions_from_state(portfolio, live_prices={"TRNFP": 1290.0, "ROSN": 300.9})

    assert [position["symbol"] for position in positions] == ["TRNFP"]


def test_dashboard_equity_and_exposure_use_units_and_short_sign():
    positions = _positions_from_state(_portfolio(), live_prices={"TRNFP": 1290.0})

    assert _gross_exposure(positions) == 45_150.0
    assert _portfolio_equity(_portfolio(), positions) == 302_209.77


def test_recent_trades_display_positive_stop_loss_as_profit_protection():
    trade = {
        "symbol": "X5",
        "direction": "short",
        "net_pnl": 46.51,
        "reason": "stop-loss",
        "exit_time": "2026-07-06T14:00:00+00:00",
    }

    assert _display_exit_reason(trade) == "profit-protect-stop"
    html = _trades_table([trade])
    assert "profit-protect-stop" in html
    assert "<td>stop-loss</td>" not in html


def test_api_recent_trades_normalizes_positive_stop_loss_reason():
    trade = {
        "symbol": "TRNFP",
        "direction": "short",
        "net_pnl": 1.69,
        "reason": "stop-loss",
        "exit_time": "2026-07-06T14:00:00+00:00",
    }

    rows = _dashboard_trades([trade])

    assert rows[0]["reason"] == "profit-protect-stop"
    assert rows[0]["raw_reason"] == "stop-loss"


def test_request_path_ignores_refresh_query_string():
    assert _request_path("/?ts=123") == "/"
    assert _request_path("/api/status?ts=123") == "/api/status"
