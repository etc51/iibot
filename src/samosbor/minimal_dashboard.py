from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from .autonomy.trade_review import trade_review_path
from .config import load_config
from .data.tbank import TBankMarketDataProvider
from .domain import ExitReason, Instrument, InstrumentType
from .runtime_metadata import current_commit_hash

_LIVE_PRICE_CACHE_TTL_SECONDS = 60.0
_LIVE_PRICE_CACHE_LOCK = Lock()
_LIVE_PRICE_CACHE: dict[str, object] = {
    "key": "",
    "fetched_at": 0.0,
    "prices": {},
}


def build_minimal_dashboard_payload(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    root = config.root_dir
    reporting_dir = config.resolve_path(config.reporting.output_dir)
    state_path = config.resolve_path(config.execution.state_path)
    state = _read_json_file(state_path)
    latest_cycle = _read_latest_json(reporting_dir / "paper", "cycle_summary.json")
    trade_review = _read_json_file(trade_review_path(state_path))
    loop = _paper_loop_status(root)

    portfolio = dict(state.get("portfolio", {}))
    live_prices, market_data_error = _live_price_marks(config, portfolio)
    positions = _positions_from_state(portfolio, live_prices=live_prices)
    trades = _dashboard_trades(list(state.get("trades", []))[-10:])
    closed_trades_count = len(list(state.get("trades", [])))
    events = list(state.get("events", []))[-20:]
    equity = _portfolio_equity(portfolio, positions)
    cash = float(portfolio.get("cash", latest_cycle.get("cash_rub", 0.0)))
    exposure = _gross_exposure(positions)
    daily_target = float(config.research.target_daily_profit_rub)
    realized = float(portfolio.get("realized_pnl", 0.0))
    open_pnl = sum(float(position.get("unrealized_pnl_rub", 0.0)) for position in positions)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit_hash": current_commit_hash(),
        "host": _local_ip(),
        "config": {
            "path": str(Path(config_path).resolve()),
            "mode": config.execution.mode.value,
            "allow_live_trading": config.execution.allow_live_trading,
            "state_path": str(state_path),
            "target_daily_profit_rub": daily_target,
            "runtime_symbols": [instrument.symbol for instrument in config.data.instruments],
        },
        "market_data": {
            "live_prices": len(live_prices),
            "error": market_data_error,
        },
        "loop": loop,
        "account": {
            "equity_rub": round(equity, 2),
            "cash_rub": round(cash, 2),
            "gross_exposure_rub": round(exposure, 2),
            "realized_pnl_rub": round(realized, 2),
            "open_pnl_rub": round(open_pnl, 2),
            "open_positions": len(positions),
            "closed_positions": closed_trades_count,
            "trading_halted": bool(portfolio.get("trading_halted", False)),
        },
        "latest_cycle": latest_cycle,
        "target": {
            "daily_rub": daily_target,
            "progress_pct": round(realized / daily_target * 100, 2) if daily_target else 0.0,
        },
        "positions": positions,
        "recent_trades": trades,
        "recent_events": events,
        "trade_review": {
            "reviewed_trades": int(trade_review.get("reviewed_trades", 0)),
            "summary": dict(trade_review.get("summary", {})),
            "mistake_breakdown": dict(trade_review.get("mistake_breakdown", {})),
            "recommendations": list(trade_review.get("recommendations", []))[:5],
            "config_patch_candidates": dict(trade_review.get("config_patch_candidates", {})),
        },
        "paths": {
            "paper_loop_log": str(root / "runs" / "runtime" / "paper-loop.log"),
            "latest_cycle_dir": str(_latest_timestamped_dir(reporting_dir / "paper") or ""),
        },
        "log_tail": _read_tail(root / "runs" / "runtime" / "paper-loop.log", lines=30),
    }


def render_minimal_dashboard_html(payload: dict[str, object]) -> str:
    config = dict(payload["config"])
    account = dict(payload["account"])
    loop = dict(payload["loop"])
    latest_cycle = dict(payload["latest_cycle"])
    target = dict(payload["target"])
    trade_review = dict(payload["trade_review"])
    market_data = dict(payload.get("market_data", {}))
    status_tone = "good" if loop.get("running") and not account.get("trading_halted") else "warn"
    live_tone = "good" if not config.get("allow_live_trading") else "bad"
    market_tone = "good" if int(market_data.get("live_prices", 0)) > 0 else "warn"

    positions_html = _positions_table(list(payload["positions"]))
    trades_html = _trades_table(list(payload["recent_trades"]))
    mistakes_html = _kv_table(dict(trade_review.get("mistake_breakdown", {})), "No mistakes yet")
    recommendations_html = _recommendations(list(trade_review.get("recommendations", [])))
    log_html = "\n".join(_escape(line) for line in list(payload["log_tail"]))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>MOEX AI Trader</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --good: #0f766e;
      --bad: #b91c1c;
      --warn: #a16207;
      --info: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; font-weight: 700; }}
    h2 {{ margin: 0 0 12px; font-size: 15px; font-weight: 700; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .wide {{ grid-column: span 2; }}
    .full {{ grid-column: 1 / -1; }}
    section, .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-size: 22px; line-height: 1.2; font-weight: 700; margin-top: 6px; overflow-wrap: anywhere; }}
    .metric .note {{ color: var(--muted); margin-top: 4px; font-size: 12px; }}
    .bad {{ color: var(--bad); }}
    .good {{ color: var(--good); }}
    .warn {{ color: var(--warn); }}
    .info {{ color: var(--info); }}
    .pillbar {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; background: #fff; color: var(--muted); font-size: 12px; }}
    .pill.good {{ border-color: #99f6e4; background: #f0fdfa; color: var(--good); }}
    .pill.bad {{ border-color: #fecaca; background: #fef2f2; color: var(--bad); }}
    .pill.warn {{ border-color: #fde68a; background: #fffbeb; color: var(--warn); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    tr:last-child td {{ border-bottom: 0; }}
    pre {{ margin: 0; max-height: 260px; overflow: auto; font: 12px/1.45 Consolas, monospace; color: #374151; white-space: pre-wrap; }}
    .mono {{ font-family: Consolas, ui-monospace, monospace; }}
    .empty {{ color: var(--muted); margin: 0; }}
    .stack {{ display: grid; gap: 12px; }}
    @media (max-width: 900px) {{
      main {{ padding: 14px; }}
      header {{ display: grid; }}
      .grid {{ grid-template-columns: 1fr; }}
      .wide {{ grid-column: auto; }}
      .pillbar {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>MOEX AI Trader</h1>
      <div class="sub">Generated {_escape(_fmt_time(str(payload["generated_at"])))} | commit {_escape(str(payload.get("commit_hash", "unknown"))[:12])} | auto refresh 5s</div>
    </div>
    <div class="pillbar">
      <span class="pill {status_tone}">loop {_escape(str(loop.get("status", "unknown")))}</span>
      <span class="pill {market_tone}">prices live={_escape(str(market_data.get("live_prices", 0)))}</span>
      <span class="pill {live_tone}">{_escape(str(config.get("mode")))} / live={str(config.get("allow_live_trading")).lower()}</span>
      <span class="pill">PID {_escape(str(loop.get("pid", "-")))}</span>
    </div>
  </header>

  <div class="grid">
    {_metric("Equity", _money(account.get("equity_rub", 0)), _fmt_time(str(latest_cycle.get("timestamp", ""))))}
    {_metric("Open / Closed", f"{account.get('open_positions', 0)} / {account.get('closed_positions', 0)}", "positions count")}
    {_metric("Open PnL", _money(account.get("open_pnl_rub", 0)), "unrealized", _tone(float(account.get("open_pnl_rub", 0))))}
    {_metric("Realized PnL", _money(account.get("realized_pnl_rub", 0)), f"target {target.get('progress_pct', 0)}%", _tone(float(account.get("realized_pnl_rub", 0))))}

    <section class="wide">
      <h2>Open Positions</h2>
      {positions_html}
    </section>

    <section class="wide">
      <h2>Latest Cycle</h2>
      {_cycle_table(latest_cycle)}
    </section>

    <section class="wide">
      <h2>Trade Review</h2>
      <div class="sub">Reviewed trades: {_escape(str(trade_review.get("reviewed_trades", 0)))}</div>
      <div style="height:10px"></div>
      {mistakes_html}
      <div style="height:10px"></div>
      {recommendations_html}
    </section>

    <section class="wide">
      <h2>Recent Trades</h2>
      {trades_html}
    </section>

    <section class="full">
      <h2>Runtime Log</h2>
      <pre>{log_html}</pre>
    </section>
  </div>
</main>
<script>
  window.setTimeout(function () {{
    var next = new URL(window.location.href);
    next.searchParams.set("ts", Date.now().toString());
    window.location.replace(next.toString());
  }}, 5000);
</script>
</body>
</html>"""


def serve_minimal_dashboard(config_path: str | Path, *, host: str, port: int) -> None:
    config_path = str(Path(config_path).resolve())
    server = ThreadingHTTPServer((host, port), _handler_factory(config_path))
    print(f"Minimal dashboard listening on http://{host}:{port}")
    server.serve_forever()


def _handler_factory(config_path: str):
    class MinimalDashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = _request_path(self.path)
            if route == "/health":
                self._send("ok\n", "text/plain; charset=utf-8")
                return
            payload = build_minimal_dashboard_payload(config_path)
            if route == "/api/status":
                self._send(json.dumps(payload, ensure_ascii=False, indent=2), "application/json; charset=utf-8")
                return
            if route != "/":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send(render_minimal_dashboard_html(payload), "text/html; charset=utf-8")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send(self, body: str, content_type: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return MinimalDashboardHandler


def _request_path(target: str) -> str:
    return urlparse(target).path or "/"


def _positions_from_state(
    portfolio: dict[str, object],
    *,
    live_prices: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    result = []
    live_prices = live_prices or {}
    positions = dict(portfolio.get("positions", {}))
    for symbol, raw_position in sorted(positions.items()):
        position = dict(raw_position)
        instrument = dict(position.get("instrument", {}))
        lot_size = max(1, int(instrument.get("lot_size", 1)))
        quantity_lots = int(position.get("quantity_lots", 0))
        if quantity_lots <= 0:
            continue
        units = quantity_lots * lot_size
        entry = float(position.get("entry_price", 0.0))
        state_current = float(position.get("current_price", entry))
        live_current = float(live_prices.get(symbol, 0.0) or 0.0)
        current = live_current if live_current > 0 else state_current
        price_source = "live" if live_current > 0 else "state"
        direction = str(position.get("direction", ""))
        if direction == "short":
            pnl = (entry - current) * units
        else:
            pnl = (current - entry) * units
        result.append(
            {
                "symbol": symbol,
                "direction": direction,
                "quantity_lots": quantity_lots,
                "quantity_units": units,
                "instrument_type": str(instrument.get("instrument_type", "stock")),
                "entry_price": entry,
                "current_price": current,
                "state_current_price": state_current,
                "price_source": price_source,
                "stop_price": float(position.get("stop_price", 0.0)),
                "take_profit": float(position.get("take_profit", 0.0)),
                "runner_active": bool(position.get("runner_active", False)),
                "runner_status": "runner" if bool(position.get("runner_active", False)) else "fixed-tp",
                "runner_activation_price": float(position.get("runner_activation_price", 0.0)),
                "runner_extreme_price": float(position.get("runner_extreme_price", 0.0)),
                "unrealized_pnl_rub": round(pnl, 2),
                "signal_strength": float(position.get("signal_strength", 0.0)),
                "opened_at": str(position.get("opened_at", "")),
            }
        )
    return result


def _positions_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return '<p class="empty">No open positions.</p>'
    body = "".join(
        "<tr>"
        f"<td>{_escape(str(row['symbol']))}</td>"
        f"<td>{_escape(str(row['direction']))}</td>"
        f"<td>{int(row['quantity_lots'])}</td>"
        f"<td>{float(row['entry_price']):.4f}</td>"
        f"<td>{float(row['current_price']):.4f}</td>"
        f"<td>{_escape(str(row['price_source']))}</td>"
        f"<td class=\"{_tone(float(row['unrealized_pnl_rub']))}\">{_money(row['unrealized_pnl_rub'])}</td>"
        f"<td>{float(row['stop_price']):.4f}</td>"
        f"<td>{float(row['take_profit']):.4f}</td>"
        f"<td>{_escape(str(row['runner_status']))}</td>"
        f"<td>{float(row['signal_strength']):.2f}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><tr><th>Symbol</th><th>Dir</th><th>Lots</th><th>Entry</th><th>Now</th><th>Src</th>"
        "<th>PnL</th><th>Stop</th><th>Take</th><th>Runner</th><th>Signal</th></tr>"
        f"{body}</table>"
    )


def _trades_table(rows: list[object]) -> str:
    if not rows:
        return '<p class="empty">No closed trades yet.</p>'
    body = ""
    for item in reversed(rows):
        trade = dict(item)
        pnl = float(trade.get("net_pnl", 0.0))
        reason = _display_exit_reason(trade, pnl=pnl)
        body += (
            "<tr>"
            f"<td>{_escape(str(trade.get('symbol', '')))}</td>"
            f"<td>{_escape(str(trade.get('direction', '')))}</td>"
            f"<td class=\"{_tone(pnl)}\">{_money(pnl)}</td>"
            f"<td>{_escape(reason)}</td>"
            f"<td>{_escape(_fmt_time(str(trade.get('exit_time', ''))))}</td>"
            "</tr>"
        )
    return "<table><tr><th>Symbol</th><th>Dir</th><th>Net</th><th>Exit</th><th>Time</th></tr>" + body + "</table>"


def _dashboard_trades(rows: list[object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in rows:
        trade = dict(item)
        raw_reason = str(trade.get("reason", ""))
        pnl = float(trade.get("net_pnl", 0.0))
        display_reason = _display_exit_reason(trade, pnl=pnl)
        if display_reason != raw_reason:
            trade["raw_reason"] = raw_reason
            trade["reason"] = display_reason
        result.append(trade)
    return result


def _display_exit_reason(trade: dict[str, object], *, pnl: float | None = None) -> str:
    reason = str(trade.get("reason", ""))
    if reason != ExitReason.STOP_LOSS.value:
        return reason
    net_pnl = float(trade.get("net_pnl", 0.0) if pnl is None else pnl)
    if net_pnl > 0:
        return ExitReason.PROFIT_PROTECT_STOP.value
    if net_pnl == 0:
        return ExitReason.BREAKEVEN_STOP.value
    return reason


def _cycle_table(cycle: dict[str, object]) -> str:
    if not cycle:
        return '<p class="empty">No cycle yet.</p>'
    keys = [
        "equity_rub",
        "gross_exposure_rub",
        "signals_total",
        "signals_approved",
        "signals_rejected",
        "open_positions",
        "trading_halted",
    ]
    rows = "".join(
        f"<tr><th>{_escape(key)}</th><td>{_escape(str(cycle.get(key, '-')))}</td></tr>"
        for key in keys
    )
    return f"<table>{rows}</table>"


def _recommendations(items: list[object]) -> str:
    if not items:
        return '<p class="empty">No recommendations.</p>'
    rows = ""
    for item in items:
        rec = dict(item)
        rows += (
            "<tr>"
            f"<td>{_escape(str(rec.get('action', '')))}</td>"
            f"<td>{_escape(str(rec.get('confidence', '')))}</td>"
            f"<td>{_escape(str(rec.get('reason', '')))}</td>"
            "</tr>"
        )
    return "<table><tr><th>Action</th><th>Conf</th><th>Reason</th></tr>" + rows + "</table>"


def _kv_table(values: dict[str, object], empty: str) -> str:
    if not values:
        return f'<p class="empty">{_escape(empty)}</p>'
    rows = "".join(f"<tr><th>{_escape(str(k))}</th><td>{_escape(str(v))}</td></tr>" for k, v in values.items())
    return f"<table>{rows}</table>"


def _metric(label: str, value: str, note: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        '<div class="metric">'
        f'<div class="label">{_escape(label)}</div>'
        f'<div class="value{tone_class}">{_escape(value)}</div>'
        f'<div class="note">{_escape(note)}</div>'
        "</div>"
    )


def _paper_loop_status(root: Path) -> dict[str, object]:
    pid_path = root / "runs" / "runtime" / "paper-loop.pid"
    pid = _read_pid(pid_path)
    running = _pid_running(pid) if pid else False
    return {
        "pid": pid,
        "running": running,
        "status": "running" if running else "stopped",
        "pid_path": str(pid_path),
    }


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8-sig").strip().lstrip("\ufeff"))
    except (OSError, ValueError):
        return None


def _pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_latest_json(root: Path, filename: str) -> dict[str, object]:
    latest = _latest_timestamped_dir(root)
    if latest is None:
        return {}
    return _read_json_file(latest / filename)


def _latest_timestamped_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def _read_json_file(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_tail(path: Path, *, lines: int) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return content[-lines:]


def _portfolio_equity(
    portfolio: dict[str, object],
    positions: list[dict[str, object]] | None = None,
) -> float:
    cash = float(portfolio.get("cash", 0.0))
    positions = positions if positions is not None else _positions_from_state(portfolio)
    value = 0.0
    for row in positions:
        current = float(row.get("current_price", 0.0))
        units = int(row.get("quantity_units", row.get("quantity_lots", 0)))
        direction = str(row.get("direction", ""))
        instrument_type = str(row.get("instrument_type", "stock"))
        if instrument_type == "future":
            value += float(row.get("unrealized_pnl_rub", 0.0))
        elif direction == "short":
            value -= current * units
        else:
            value += current * units
    return cash + value


def _gross_exposure(positions: list[dict[str, object]]) -> float:
    return sum(
        abs(float(row.get("current_price", 0.0)) * int(row.get("quantity_units", row.get("quantity_lots", 0))))
        for row in positions
    )


def _live_price_marks(config, portfolio: dict[str, object]) -> tuple[dict[str, float], str]:
    instruments = _position_instruments(portfolio)
    if not instruments:
        return {}, ""
    cache_key = _live_price_cache_key(instruments)
    now = time.monotonic()
    with _LIVE_PRICE_CACHE_LOCK:
        cached_prices = dict(_LIVE_PRICE_CACHE.get("prices", {}))
        cached_at = float(_LIVE_PRICE_CACHE.get("fetched_at", 0.0) or 0.0)
        if (
            _LIVE_PRICE_CACHE.get("key") == cache_key
            and cached_prices
            and now - cached_at <= _LIVE_PRICE_CACHE_TTL_SECONDS
        ):
            return cached_prices, ""
    try:
        prices = TBankMarketDataProvider(config).get_last_prices(instruments)
    except Exception as exc:  # pragma: no cover - depends on live broker API
        with _LIVE_PRICE_CACHE_LOCK:
            cached_prices = dict(_LIVE_PRICE_CACHE.get("prices", {}))
            if _LIVE_PRICE_CACHE.get("key") == cache_key and cached_prices:
                return cached_prices, f"{type(exc).__name__}: {exc}"
        return {}, f"{type(exc).__name__}: {exc}"
    live_prices = {symbol: price for symbol, price in prices.items() if price > 0}
    with _LIVE_PRICE_CACHE_LOCK:
        _LIVE_PRICE_CACHE.update(
            {
                "key": cache_key,
                "fetched_at": now,
                "prices": dict(live_prices),
            }
        )
    return live_prices, ""


def _live_price_cache_key(instruments: list[Instrument]) -> str:
    return "|".join(
        f"{instrument.symbol}:{instrument.figi}:{instrument.uid}:{instrument.lot_size}"
        for instrument in instruments
    )


def _position_instruments(portfolio: dict[str, object]) -> list[Instrument]:
    result: list[Instrument] = []
    positions = dict(portfolio.get("positions", {}))
    for symbol, raw_position in sorted(positions.items()):
        position = dict(raw_position)
        if int(position.get("quantity_lots", 0)) <= 0:
            continue
        raw_instrument = dict(position.get("instrument", {}))
        try:
            instrument_type = InstrumentType(str(raw_instrument.get("instrument_type", "stock")))
            result.append(
                Instrument(
                    symbol=str(raw_instrument.get("symbol", symbol)),
                    instrument_type=instrument_type,
                    figi=str(raw_instrument.get("figi", "")),
                    uid=str(raw_instrument.get("uid", "")),
                    class_code=str(raw_instrument.get("class_code", "")),
                    lot_size=max(1, int(raw_instrument.get("lot_size", 1) or 1)),
                    tick_size=float(raw_instrument.get("tick_size", 0.01) or 0.01),
                    currency=str(raw_instrument.get("currency", "rub")),
                    initial_margin_buy=float(raw_instrument.get("initial_margin_buy", 0.0) or 0.0),
                    initial_margin_sell=float(raw_instrument.get("initial_margin_sell", 0.0) or 0.0),
                    tick_value=float(raw_instrument.get("tick_value", 0.0) or 0.0),
                )
            )
        except (TypeError, ValueError):
            continue
    return result


def _local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _money(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return f"{number:,.2f} RUB".replace(",", " ")


def _tone(value: float) -> str:
    if value > 0:
        return "good"
    if value < 0:
        return "bad"
    return ""


def _fmt_time(value: str) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MOEX AI Trader dashboard")
    parser.add_argument("--config", default="configs/server_tbank_stocks_intraday_300k_focused.toml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve_minimal_dashboard(args.config, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
