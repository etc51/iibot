from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .autonomy.signal_feedback import signal_feedback_path
from .config import StrategySection, load_config


def build_dashboard_payload(
    config_path: str | Path,
    *,
    effective_config_path: str | Path | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    reporting_dir = config.resolve_path(config.reporting.output_dir)
    autotune_dir = config.autotune_dir()
    state_path = config.resolve_path(config.execution.state_path)
    feedback_path = signal_feedback_path(state_path)
    configured_symbols = _normalize_symbol_list([instrument.symbol for instrument in config.data.instruments])
    runtime_symbols = set(configured_symbols)
    runtime_hours = {int(hour) for hour in config.strategy.allowed_entry_hours}

    paper_cycle = _read_latest_json(reporting_dir / "paper", "cycle_summary.json")
    paper_report = _read_latest_json(reporting_dir / "paper-reports", "summary.json")
    nightly = _read_latest_json(autotune_dir / "nightly-autonomy", "nightly_autonomy.json")
    effective_runtime = _read_latest_json(
        autotune_dir / "effective-config",
        "effective_config.json",
    )
    active_config = _load_display_config(
        config_path,
        effective_config_path or effective_runtime.get("effective_config_path"),
    )
    entry_symbols = _read_latest_compatible_json(
        autotune_dir / "entry-symbols",
        "symbol_restrictions.json",
        lambda payload: _entry_symbols_payload_matches_runtime(payload, runtime_symbols),
    )
    if not entry_symbols:
        entry_symbols = _sanitize_entry_symbols_payload(
            _read_latest_json(autotune_dir / "entry-symbols", "symbol_restrictions.json"),
            runtime_symbols,
        )
    entry_schedule = _read_latest_compatible_json(
        autotune_dir / "entry-schedule",
        "schedule_tuning.json",
        lambda payload: _entry_schedule_payload_matches_runtime(payload, runtime_hours),
    )
    if not entry_schedule:
        entry_schedule = _sanitize_entry_schedule_payload(
            _read_latest_json(autotune_dir / "entry-schedule", "schedule_tuning.json"),
            runtime_hours,
        )
    entry_quality = _read_latest_json(
        autotune_dir / "entry-quality",
        "entry_quality_tuning.json",
    )
    state_payload = _read_json_file(state_path)
    feedback_payload = _read_json_file(feedback_path)

    portfolio = dict(state_payload.get("portfolio", {}))
    positions = _positions_from_state(
        portfolio.get("positions", {}),
        state_payload.get("events", []),
        active_config.strategy,
    )
    latest_summary = paper_cycle if paper_cycle else {}
    report_summary = paper_report.get("summary", {}) if paper_report else {}
    report_portfolio = paper_report.get("portfolio", {}) if paper_report else {}
    overrides = dict(effective_runtime.get("applied_strategy_overrides", {})) if effective_runtime else {}
    runtime_universe = _read_latest_compatible_json(
        autotune_dir / "universe-selection",
        "universe_selection.json",
        lambda payload: _runtime_universe_payload_matches_runtime(payload, runtime_symbols),
    )
    if not runtime_universe:
        runtime_universe = _sanitize_runtime_universe_payload(
            _read_latest_json(autotune_dir / "universe-selection", "universe_selection.json"),
            runtime_symbols,
        )
    active_allowed_symbols_override = _normalize_symbol_list(overrides.get("allowed_symbols", []))
    active_config_allowed_symbols = _normalize_symbol_list(active_config.strategy.allowed_symbols)
    active_effective_symbols = (
        active_allowed_symbols_override or active_config_allowed_symbols or list(configured_symbols)
    )
    runtime_universe_source = _effective_runtime_source(effective_runtime, "universe-selection")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "config_path": str(Path(config_path).resolve()),
            "effective_config_path": (
                str(Path(effective_config_path).resolve())
                if effective_config_path is not None
                else str(effective_runtime.get("effective_config_path", ""))
            ),
            "paper_only_mode": config.execution.mode.value,
            "allow_live_trading": config.execution.allow_live_trading,
            "account_name": config.tbank.account_name,
            "instruments": list(configured_symbols),
            "state_path": str(state_path),
            "feedback_path": str(feedback_path),
            "reporting_dir": str(reporting_dir),
            "autotune_dir": str(autotune_dir),
        },
        "runtime": {
            "portfolio_state": {
                "cash_rub": round(float(portfolio.get("cash", 0.0)), 2),
                "realized_pnl_rub": round(float(portfolio.get("realized_pnl", 0.0)), 2),
                "peak_equity_rub": round(float(portfolio.get("peak_equity", 0.0)), 2),
                "open_positions": len(positions),
                "trading_halted": bool(portfolio.get("trading_halted", False)),
            },
            "latest_cycle": latest_summary,
            "latest_report_summary": report_summary,
            "latest_report_portfolio": report_portfolio,
            "positions": positions,
            "strategy_state": {
                "style": active_config.strategy.style,
                "atr_stop_multiple": float(active_config.strategy.atr_stop_multiple),
                "reward_to_risk": float(active_config.strategy.reward_to_risk),
                "breakeven_trigger_pct": float(active_config.strategy.breakeven_trigger_pct),
                "trailing_profit_trigger_rub": float(active_config.strategy.trailing_profit_trigger_rub),
                "trailing_profit_lock_ratio": float(active_config.strategy.trailing_profit_lock_ratio),
            },
            "signal_feedback": {
                "pending_signals": len(feedback_payload.get("pending", [])),
                "resolved_signals": len(feedback_payload.get("resolved", [])),
            },
        },
        "autonomy": {
            "effective_runtime": {
                "output_dir": str(effective_runtime.get("output_dir", "")),
                "applied_strategy_overrides": overrides,
                "rollback_guardrail": dict(effective_runtime.get("rollback_guardrail", {})),
                "sources": list(effective_runtime.get("sources", [])),
            },
            "entry_symbols": {
                "changed": bool(entry_symbols.get("changed", False)),
                "reason": str(entry_symbols.get("reason", "")),
                "evidence_source": str(entry_symbols.get("evidence_source", "")),
                "evidence_counts": dict(entry_symbols.get("evidence_counts", {})),
                "proposed_blocked_symbols": list(entry_symbols.get("proposed_blocked_symbols", [])),
                "proposed_blocked_long_symbols": list(
                    entry_symbols.get("proposed_blocked_long_symbols", [])
                ),
                "proposed_blocked_short_symbols": list(
                    entry_symbols.get("proposed_blocked_short_symbols", [])
                ),
                "symbol_direction_breakdown": list(
                    entry_symbols.get("symbol_direction_breakdown", [])
                ),
            },
            "entry_schedule": {
                "changed": bool(entry_schedule.get("changed", False)),
                "reason": str(entry_schedule.get("reason", "")),
                "evidence_source": str(entry_schedule.get("evidence_source", "")),
                "evidence_counts": dict(entry_schedule.get("evidence_counts", {})),
                "proposed_hours": list(entry_schedule.get("proposed_hours", [])),
            },
            "entry_quality": {
                "changed": bool(entry_quality.get("changed", False)),
                "reason": str(entry_quality.get("reason", "")),
                "evidence_source": str(entry_quality.get("evidence_source", "")),
                "evidence_counts": dict(entry_quality.get("evidence_counts", {})),
                "current_min_signal_strength": float(
                    entry_quality.get("current_min_signal_strength", 0.0)
                ),
                "recommended_min_signal_strength": float(
                    entry_quality.get("recommended_min_signal_strength", 0.0)
                ),
                "lookback": dict(entry_quality.get("lookback", {})),
                "baseline_summary": dict(entry_quality.get("baseline_summary", {})),
                "recommended_summary": dict(entry_quality.get("recommended_summary", {})),
            },
            "nightly_autonomy": {
                "output_dir": str(nightly.get("output_dir", "")),
                "steps_executed": list(nightly.get("steps_executed", [])),
                "timestamp": str(nightly.get("timestamp", "")),
            },
            "runtime_universe": {
                "changed": bool(runtime_universe.get("changed", False)),
                "reason": str(runtime_universe.get("reason", "")),
                "guardrails": dict(runtime_universe.get("guardrails", {})),
                "configured_symbols": list(configured_symbols),
                "current_allowed_symbols": list(runtime_universe.get("current_allowed_symbols", [])),
                "current_effective_symbols": list(
                    runtime_universe.get("current_effective_symbols", [])
                )
                or list(runtime_universe.get("current_allowed_symbols", []))
                or list(configured_symbols),
                "active_allowed_symbols_override": list(active_allowed_symbols_override),
                "active_effective_symbols": list(active_effective_symbols),
                "optimizer_best_symbols": list(runtime_universe.get("optimizer_best_symbols", [])),
                "walk_forward_latest_symbols": list(
                    runtime_universe.get("walk_forward_latest_symbols", [])
                ),
                "consensus_symbols": list(runtime_universe.get("consensus_symbols", [])),
                "proposed_allowed_symbols": list(runtime_universe.get("proposed_allowed_symbols", [])),
                "proposed_effective_symbols": list(runtime_universe.get("proposed_effective_symbols", [])),
                "optimizer_summary": dict(runtime_universe.get("optimizer_summary", {})),
                "walk_forward_summary": dict(runtime_universe.get("walk_forward_summary", {})),
                "activation": dict(runtime_universe_source.get("activation", {})),
                "selected_values": dict(runtime_universe_source.get("selected_values", {})),
            },
        },
    }


def render_dashboard_html(payload: dict[str, object]) -> str:
    config = dict(payload["config"])
    runtime = dict(payload["runtime"])
    autonomy = dict(payload["autonomy"])
    portfolio_state = dict(runtime["portfolio_state"])
    latest_cycle = dict(runtime["latest_cycle"])
    latest_report_summary = dict(runtime["latest_report_summary"])
    latest_report_portfolio = dict(runtime["latest_report_portfolio"])
    strategy_state = dict(runtime["strategy_state"])
    effective_runtime = dict(autonomy["effective_runtime"])
    entry_symbols = dict(autonomy["entry_symbols"])
    entry_schedule = dict(autonomy["entry_schedule"])
    entry_quality = dict(autonomy["entry_quality"])
    nightly_autonomy = dict(autonomy["nightly_autonomy"])
    runtime_universe = dict(autonomy["runtime_universe"])
    signal_rejection_reason_breakdown = dict(
        latest_cycle.get("signal_rejection_reason_breakdown", {})
    )
    entry_quality_lookback = dict(entry_quality.get("lookback", {}))
    entry_quality_baseline = dict(entry_quality.get("baseline_summary", {}))
    entry_quality_recommended = dict(entry_quality.get("recommended_summary", {}))

    positions_html = _render_positions(list(runtime["positions"]))
    overrides_html = _render_key_values(dict(effective_runtime["applied_strategy_overrides"]))
    sources_html = _render_sources(list(effective_runtime["sources"]))
    direction_rows_html = _render_direction_rows(list(entry_symbols["symbol_direction_breakdown"]))
    status_badges_html = "".join(
        [
            _render_badge(f"mode: {config['paper_only_mode']}", tone="info"),
            _render_badge("paper only", tone="good"),
            _render_badge(
                "live blocked" if not bool(config["allow_live_trading"]) else "live enabled",
                tone="good" if not bool(config["allow_live_trading"]) else "bad",
            ),
            _render_badge(
                "trading active"
                if not bool(latest_cycle.get("trading_halted", portfolio_state.get("trading_halted", False)))
                else "trading halted",
                tone="good"
                if not bool(latest_cycle.get("trading_halted", portfolio_state.get("trading_halted", False)))
                else "warn",
            ),
        ]
    )
    runtime_meta_badges_html = "".join(
        [
            _render_badge(
                f"resolved feedback: {int(runtime['signal_feedback']['resolved_signals'])}",
                tone="info",
            ),
            _render_badge(
                f"pending feedback: {int(runtime['signal_feedback']['pending_signals'])}",
                tone="warn" if int(runtime["signal_feedback"]["pending_signals"]) else "neutral",
            ),
            _render_badge(
                f"active overrides: {len(dict(effective_runtime['applied_strategy_overrides']))}",
                tone="info",
            ),
            _render_badge(
                f"allowed hours: {len(list(entry_schedule['proposed_hours']))}",
                tone="neutral",
            ),
            _render_badge(
                f"effective symbols: {len(list(runtime_universe['active_effective_symbols']))}",
                tone="info",
            ),
            _render_badge(
                f"signals rejected: {int(latest_cycle.get('signals_rejected', 0))}",
                tone="warn" if int(latest_cycle.get("signals_rejected", 0)) else "neutral",
            ),
        ]
    )
    restriction_badges_html = "".join(
        [
            _render_badge(
                f"blocked symbols: {', '.join(entry_symbols['proposed_blocked_symbols']) or '-'}",
                tone="bad" if entry_symbols["proposed_blocked_symbols"] else "neutral",
            ),
            _render_badge(
                f"blocked longs: {', '.join(entry_symbols['proposed_blocked_long_symbols']) or '-'}",
                tone="warn" if entry_symbols["proposed_blocked_long_symbols"] else "neutral",
            ),
            _render_badge(
                f"blocked shorts: {', '.join(entry_symbols['proposed_blocked_short_symbols']) or '-'}",
                tone="warn" if entry_symbols["proposed_blocked_short_symbols"] else "neutral",
            ),
        ]
    )
    evidence_badges_html = "".join(
        [
            _render_badge(
                f"symbols: {entry_symbols['evidence_source'] or '-'} "
                f"[{_render_counts_summary(dict(entry_symbols.get('evidence_counts', {})))}]",
                tone="info",
            ),
            _render_badge(
                f"schedule: {entry_schedule['evidence_source'] or '-'} "
                f"[{_render_counts_summary(dict(entry_schedule.get('evidence_counts', {})))}]",
                tone="info",
            ),
            _render_badge(
                f"quality: {entry_quality['evidence_source'] or '-'} "
                f"[{_render_counts_summary(dict(entry_quality.get('evidence_counts', {})))}]",
                tone="info",
            ),
        ]
    )
    nightly_steps_html = _render_chips(list(nightly_autonomy["steps_executed"]), tone="info")
    instrument_chips_html = _render_chips(list(config["instruments"]), tone="neutral")
    active_runtime_chips_html = _render_chips(list(runtime_universe["active_effective_symbols"]), tone="info")
    universe_candidate_chips_html = _render_chips(
        list(runtime_universe["proposed_effective_symbols"]),
        tone="neutral",
    )
    consensus_chips_html = _render_chips(list(runtime_universe["consensus_symbols"]), tone="neutral")
    universe_status_badges_html = "".join(
        [
            _render_badge(
                f"configured: {len(list(runtime_universe['configured_symbols']))}",
                tone="neutral",
            ),
            _render_badge(
                f"active effective: {len(list(runtime_universe['active_effective_symbols']))}",
                tone="info",
            ),
            _render_badge(
                "allowed_symbols override active"
                if runtime_universe["active_allowed_symbols_override"]
                else "full configured universe active",
                tone="warn" if runtime_universe["active_allowed_symbols_override"] else "good",
            ),
            _render_badge(
                "nightly universe candidate changed"
                if bool(runtime_universe["changed"])
                else "nightly universe unchanged",
                tone="info" if bool(runtime_universe["changed"]) else "neutral",
            ),
        ]
    )

    equity_rub = float(latest_cycle.get("equity_rub", portfolio_state.get("peak_equity_rub", 0.0)))
    cash_rub = float(latest_cycle.get("cash_rub", portfolio_state.get("cash_rub", 0.0)))
    exposure_rub = float(latest_cycle.get("gross_exposure_rub", 0.0))
    realized_pnl_rub = float(portfolio_state.get("realized_pnl_rub", 0.0))
    daily_pnl_rub = float(latest_report_summary.get("net_pnl_rub", 0.0))
    open_positions = int(latest_cycle.get("open_positions", portfolio_state.get("open_positions", 0)))
    resolved_feedback = int(runtime["signal_feedback"]["resolved_signals"])
    active_override_count = len(dict(effective_runtime["applied_strategy_overrides"]))
    active_restriction_count = (
        len(entry_symbols["proposed_blocked_symbols"])
        + len(entry_symbols["proposed_blocked_long_symbols"])
        + len(entry_symbols["proposed_blocked_short_symbols"])
    )
    signal_diagnostics_html = _render_named_value_table(
        signal_rejection_reason_breakdown,
        key_label="Reason",
        value_label="Count",
        empty_message="No rejected signals recorded in the latest cycle.",
    )
    kpi_cards_html = "".join(
        [
            _render_stat_card("Equity", _fmt_money(equity_rub), _fmt_timestamp(str(latest_cycle.get("timestamp", "")))),
            _render_stat_card("Cash", _fmt_money(cash_rub), "Свободные paper-средства"),
            _render_stat_card("Exposure", _fmt_money(exposure_rub), "Текущее gross exposure"),
            _render_stat_card(
                "Realized PnL",
                _fmt_money(realized_pnl_rub),
                "Реализованный результат",
                tone=_number_tone(realized_pnl_rub),
            ),
            _render_stat_card(
                "Day Net",
                _fmt_money(daily_pnl_rub),
                f"Закрытых сделок: {int(latest_report_summary.get('trades', 0))}",
                tone=_number_tone(daily_pnl_rub),
            ),
            _render_stat_card("Open Positions", str(open_positions), "Активные позиции сейчас"),
            _render_stat_card("Resolved Signals", str(resolved_feedback), "Накопленная обратная связь"),
            _render_stat_card(
                "Restrictions",
                str(active_restriction_count),
                f"Overrides: {active_override_count}",
                tone="warn" if active_restriction_count else "neutral",
            ),
        ]
    )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Samosbor Paper Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #071219;
      --panel: #0d1f28;
      --panel-2: #122b36;
      --line: #234758;
      --line-soft: rgba(35, 71, 88, 0.54);
      --text: #edf6fb;
      --muted: #8fa8b7;
      --good: #55d58f;
      --bad: #ff7676;
      --warn: #ffcc66;
      --accent: #4ad2ff;
      --accent-2: #89b4ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(74, 210, 255, 0.14), transparent 24%),
        radial-gradient(circle at top left, rgba(137, 180, 255, 0.1), transparent 28%),
        linear-gradient(180deg, #061018 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 22px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-end;
      position: sticky;
      top: 0;
      z-index: 5;
      margin: -22px -22px 18px;
      padding: 18px 22px 16px;
      border-bottom: 1px solid var(--line-soft);
      background: rgba(6, 16, 24, 0.88);
      backdrop-filter: blur(14px);
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      letter-spacing: 0.01em;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    .hero {{
      display: grid;
      gap: 10px;
    }}
    .hero-sub {{
      max-width: 780px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .status-box {{
      min-width: 320px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(13, 31, 40, 0.92), rgba(18, 43, 54, 0.8));
      box-shadow: 0 16px 42px rgba(0, 0, 0, 0.24);
    }}
    .status-kicker {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .status-line {{
      margin-top: 10px;
      font-size: 15px;
      font-weight: 600;
    }}
    .status-meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.28fr 0.92fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card, .panel {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(13, 31, 40, 0.92), rgba(18, 43, 54, 0.82));
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
    }}
    .card {{
      padding: 16px;
      min-height: 108px;
    }}
    .panel {{
      padding: 16px;
    }}
    .kpi-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .kpi-value {{
      font-size: 28px;
      margin-top: 12px;
      font-weight: 700;
      line-height: 1.15;
    }}
    .kpi-note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .info {{ color: var(--accent); }}
    .mono {{
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .cell-note {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line-soft);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .chip {{
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(7, 18, 25, 0.62);
      font-size: 13px;
    }}
    .chip.info {{
      color: var(--accent);
      border-color: rgba(74, 210, 255, 0.32);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(7, 18, 25, 0.74);
      font-size: 13px;
      line-height: 1;
      color: var(--text);
    }}
    .badge.good {{
      border-color: rgba(85, 213, 143, 0.34);
      color: var(--good);
    }}
    .badge.bad {{
      border-color: rgba(255, 118, 118, 0.34);
      color: var(--bad);
    }}
    .badge.warn {{
      border-color: rgba(255, 204, 102, 0.34);
      color: var(--warn);
    }}
    .badge.info {{
      border-color: rgba(74, 210, 255, 0.34);
      color: var(--accent);
    }}
    .stack {{
      display: grid;
      gap: 12px;
    }}
    .section-gap {{
      margin-top: 14px;
    }}
    .footer {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    a {{ color: var(--accent); }}
    @media (max-width: 1280px) {{
      .cards {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 1050px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .status-box {{
        min-width: 0;
        width: 100%;
      }}
    }}
    @media (max-width: 760px) {{
      .wrap {{
        padding: 14px;
      }}
      .topbar {{
        margin: -14px -14px 16px;
        padding: 14px;
      }}
      .cards {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .kpi-value {{
        font-size: 22px;
      }}
      h1 {{
        font-size: 24px;
      }}
      table {{
        font-size: 13px;
      }}
    }}
  </style>
  <script>
    setTimeout(() => window.location.reload(), 15000);
  </script>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="hero">
        <div>
          <h1>Samosbor Paper Dashboard</h1>
          <p class="hero-sub">Операционная панель для текущего samosbor runtime: позиции, ограничения, автотюнинг и доказательная обратная связь в одном экране.</p>
        </div>
        <div class="badge-row">{status_badges_html}</div>
      </div>
      <div class="status-box">
        <div class="status-kicker">Runtime Snapshot</div>
        <div class="status-line">{_escape(str(config["account_name"]))} | {_escape(str(config["paper_only_mode"]))}</div>
        <div class="status-meta">
          Последний цикл: <span class="mono">{_escape(_fmt_timestamp(str(latest_cycle.get("timestamp", ""))))}</span><br/>
          Ночной цикл: <span class="mono">{_escape(_fmt_timestamp(str(nightly_autonomy.get("timestamp", ""))))}</span><br/>
          Автообновление каждые 15 секунд
        </div>
      </div>
    </header>

    <div class="cards">{kpi_cards_html}</div>

    <div class="layout">
      <section class="panel">
        <h2>Open Positions</h2>
        <div class="table-wrap">{positions_html}</div>
      </section>
      <section class="panel">
        <h2>Trade Posture</h2>
        <div class="stack">
          <div>
            <h3>Runtime Flags</h3>
            <div class="badge-row">{runtime_meta_badges_html}</div>
          </div>
          <div>
            <h3>Signal Diagnostics</h3>
            <div class="table-wrap">
              <table>
                <tr><th>Total signals</th><td>{int(latest_cycle.get("signals_total", 0))}</td></tr>
                <tr><th>Approved</th><td>{int(latest_cycle.get("signals_approved", 0))}</td></tr>
                <tr><th>Rejected</th><td>{int(latest_cycle.get("signals_rejected", 0))}</td></tr>
              </table>
            </div>
            <div class="table-wrap">{signal_diagnostics_html}</div>
          </div>
          <div>
            <h3>Universe</h3>
            <div class="chips">{instrument_chips_html}</div>
          </div>
          <div class="table-wrap">
            <table>
              <tr><th>Mode</th><td>{_escape(str(config["paper_only_mode"]))}</td></tr>
              <tr><th>Live allowed</th><td>{_bool_text(bool(config["allow_live_trading"]))}</td></tr>
              <tr><th>Style</th><td>{_escape(str(strategy_state.get("style", "-")))}</td></tr>
              <tr><th>Reward / risk</th><td>{float(strategy_state.get("reward_to_risk", 0.0)):.2f}</td></tr>
              <tr><th>ATR stop x</th><td>{float(strategy_state.get("atr_stop_multiple", 0.0)):.2f}</td></tr>
              <tr><th>Break-even trigger</th><td>{float(strategy_state.get("breakeven_trigger_pct", 0.0)):.2f}%</td></tr>
              <tr><th>Trailing trigger</th><td>{_fmt_money(float(strategy_state.get("trailing_profit_trigger_rub", 0.0)))}</td></tr>
              <tr><th>Trailing lock</th><td>{float(strategy_state.get("trailing_profit_lock_ratio", 0.0)):.0%}</td></tr>
              <tr><th>Trading halted</th><td>{_bool_text(bool(latest_cycle.get("trading_halted", portfolio_state.get("trading_halted", False))))}</td></tr>
              <tr><th>Peak equity</th><td>{_fmt_money(float(portfolio_state.get("peak_equity_rub", 0.0)))}</td></tr>
              <tr><th>Last cycle</th><td class="mono">{_escape(_fmt_timestamp(str(latest_cycle.get("timestamp", ""))))}</td></tr>
              <tr><th>Nightly cycle</th><td class="mono">{_escape(_fmt_timestamp(str(nightly_autonomy.get("timestamp", ""))))}</td></tr>
            </table>
          </div>
        </div>
      </section>
    </div>

    <div class="layout">
      <section class="panel">
        <h2>Daily Summary</h2>
        <div class="table-wrap">
          <table>
            <tr><th>Trades</th><td>{int(latest_report_summary.get("trades", 0))}</td></tr>
            <tr><th>Net PnL</th><td class="{_number_tone(float(latest_report_summary.get("net_pnl_rub", 0.0)))}">{_fmt_money(float(latest_report_summary.get("net_pnl_rub", 0.0)))}</td></tr>
            <tr><th>Win rate</th><td>{float(latest_report_summary.get("win_rate_pct", 0.0)):.3f}%</td></tr>
            <tr><th>Profit factor</th><td>{float(latest_report_summary.get("profit_factor", 0.0)):.3f}</td></tr>
            <tr><th>Expectancy</th><td class="{_number_tone(float(latest_report_summary.get("expectancy_rub", 0.0)))}">{_fmt_money(float(latest_report_summary.get("expectancy_rub", 0.0)))}</td></tr>
            <tr><th>Report open positions</th><td>{int(latest_report_portfolio.get("open_positions", 0))}</td></tr>
          </table>
        </div>
      </section>
      <section class="panel">
        <h2>Autonomy Cycle</h2>
        <div class="stack">
          <div>
            <h3>Pipeline Steps</h3>
            <div class="chips">{nightly_steps_html}</div>
          </div>
          <div>
            <h3>Evidence Sources</h3>
            <div class="badge-row">{evidence_badges_html}</div>
          </div>
          <div class="footer">
            Rollback guardrail: {_escape(str(effective_runtime["rollback_guardrail"].get("reason", "")))}<br/>
            Nightly output: <span class="mono">{_escape(str(nightly_autonomy.get("output_dir", "")) or "-")}</span>
          </div>
        </div>
      </section>
    </div>

    <div class="layout">
      <section class="panel">
        <h2>Entry Restrictions</h2>
        <div class="stack">
          <div>
            <h3>Directional Blocks</h3>
            <div class="badge-row">{restriction_badges_html}</div>
          </div>
          <div>
            <h3>Allowed Entry Hours</h3>
            <div class="chips">{_render_chips(entry_schedule.get("proposed_hours", []), tone="info")}</div>
          </div>
          <div class="table-wrap">
            <table>
              <tr><th>Symbol tune</th><td>{_bool_text(bool(entry_symbols["changed"]))}</td></tr>
              <tr><th>Schedule tune</th><td>{_bool_text(bool(entry_schedule["changed"]))}</td></tr>
              <tr><th>Quality tune</th><td>{_bool_text(bool(entry_quality["changed"]))}</td></tr>
              <tr><th>Current min signal strength</th><td>{float(entry_quality["current_min_signal_strength"]):.3f}</td></tr>
              <tr><th>Min signal strength</th><td>{float(entry_quality["recommended_min_signal_strength"]):.3f}</td></tr>
              <tr><th>Lookback eligible trades</th><td>{int(entry_quality_lookback.get("eligible_trades", 0))} / {int(entry_quality_lookback.get("requested_trades", 0))} (min {int(entry_quality_lookback.get("min_trades", 0))})</td></tr>
              <tr><th>Baseline expectancy</th><td>{_fmt_money(float(entry_quality_baseline.get("expectancy_rub", 0.0)))} ({int(entry_quality_baseline.get("trades", 0))} trades)</td></tr>
              <tr><th>Recommended expectancy</th><td>{_fmt_money(float(entry_quality_recommended.get("expectancy_rub", 0.0)))} ({int(entry_quality_recommended.get("trades", 0))} trades)</td></tr>
              <tr><th>Baseline profit factor</th><td>{float(entry_quality_baseline.get("profit_factor", 0.0)):.3f}</td></tr>
              <tr><th>Recommended profit factor</th><td>{float(entry_quality_recommended.get("profit_factor", 0.0)):.3f}</td></tr>
            </table>
          </div>
          <div class="footer">
            Symbols: {_escape(str(entry_symbols["reason"]))}<br/>
            Schedule: {_escape(str(entry_schedule["reason"]))}<br/>
            Quality: {_escape(str(entry_quality["reason"]))}
          </div>
        </div>
      </section>
      <section class="panel">
        <h2>Active Overrides</h2>
        <div class="table-wrap">{overrides_html}</div>
        <div class="footer">
          Effective config: <span class="mono">{_escape(str(config["effective_config_path"]))}</span>
        </div>
      </section>
    </div>

    <section class="panel section-gap">
      <h2>Effective Runtime Universe</h2>
      <div class="stack">
        <div>
          <h3>Status</h3>
          <div class="badge-row">{universe_status_badges_html}</div>
        </div>
        <div>
          <h3>Configured Symbols</h3>
          <div class="chips">{instrument_chips_html}</div>
        </div>
        <div>
          <h3>Active Effective Symbols</h3>
          <div class="chips">{active_runtime_chips_html}</div>
        </div>
        <div>
          <h3>Nightly Consensus</h3>
          <div class="chips">{consensus_chips_html}</div>
        </div>
        <div>
          <h3>Nightly Proposed Effective Set</h3>
          <div class="chips">{universe_candidate_chips_html}</div>
        </div>
        <div class="table-wrap">
          <table>
            <tr><th>Active allowed override</th><td class="mono">{_escape(_render_inline_list(list(runtime_universe["active_allowed_symbols_override"])))}</td></tr>
            <tr><th>Nightly current allowed</th><td class="mono">{_escape(_render_inline_list(list(runtime_universe["current_allowed_symbols"])))}</td></tr>
            <tr><th>Nightly current effective</th><td class="mono">{_escape(_render_inline_list(list(runtime_universe["current_effective_symbols"])))}</td></tr>
            <tr><th>Nightly proposed allowed</th><td class="mono">{_escape(_render_inline_list(list(runtime_universe["proposed_allowed_symbols"])))}</td></tr>
            <tr><th>Optimizer best</th><td class="mono">{_escape(_render_inline_list(list(runtime_universe["optimizer_best_symbols"])))}</td></tr>
            <tr><th>Walk-forward latest</th><td class="mono">{_escape(_render_inline_list(list(runtime_universe["walk_forward_latest_symbols"])))}</td></tr>
            <tr><th>Source activation</th><td>{_escape(str(runtime_universe["activation"].get("reason", "")) or "-")}</td></tr>
            <tr><th>Guardrails</th><td class="mono">min_effective_symbols={_escape(str(runtime_universe["guardrails"].get("min_effective_symbols", "-")))} | coverage={_escape(str(runtime_universe["guardrails"].get("min_configured_coverage_ratio", "-")))} | wf_positive={_escape(str(runtime_universe["walk_forward_summary"].get("probability_positive_pct", "-")))} | wf_trades={_escape(str(runtime_universe["walk_forward_summary"].get("latest_fold_test_trades", "-")))}</td></tr>
          </table>
        </div>
        <div class="footer">
          Nightly decision: {_escape(str(runtime_universe["reason"]))}<br/>
          Selected override payload: <span class="mono">{_escape(json.dumps(runtime_universe["selected_values"], ensure_ascii=False) or "-")}</span>
        </div>
      </div>
    </section>

    <section class="panel section-gap">
      <h2>Directional Breakdown</h2>
      <div class="table-wrap">{direction_rows_html}</div>
      <div class="footer">{_escape(str(entry_symbols["reason"]))} | source={_escape(str(entry_symbols["evidence_source"]))}</div>
    </section>

    <section class="panel section-gap">
      <h2>Active Sources</h2>
      <div class="table-wrap">{sources_html}</div>
    </section>

    <section class="panel section-gap">
      <h2>Paths & API</h2>
      <div class="footer">
        Config: <span class="mono">{_escape(str(config["config_path"]))}</span><br/>
        State: <span class="mono">{_escape(str(config["state_path"]))}</span><br/>
        Feedback: <span class="mono">{_escape(str(config["feedback_path"]))}</span><br/>
        Reporting: <span class="mono">{_escape(str(config["reporting_dir"]))}</span><br/>
        JSON API: <a href="/api/payload">/api/payload</a> | Health: <a href="/health">/health</a>
      </div>
    </section>
  </div>
</body>
</html>
"""


def serve_dashboard(
    config_path: str | Path,
    *,
    effective_config_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8790,
) -> None:
    config_path = str(Path(config_path).resolve())
    effective_path = str(Path(effective_config_path).resolve()) if effective_config_path else None
    server = ThreadingHTTPServer(
        (host, port),
        _handler_factory(config_path, effective_path),
    )
    print(f"Samosbor dashboard listening on http://{host}:{port}")
    server.serve_forever()


def _handler_factory(config_path: str, effective_config_path: str | None):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_response("ok\n", content_type="text/plain; charset=utf-8")
                return

            payload = build_dashboard_payload(
                config_path,
                effective_config_path=effective_config_path,
            )
            if self.path == "/api/payload":
                self._send_response(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    content_type="application/json; charset=utf-8",
                )
                return
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send_response(
                render_dashboard_html(payload),
                content_type="text/html; charset=utf-8",
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send_response(self, body: str, *, content_type: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return DashboardHandler


def _read_latest_json(root: Path, filename: str) -> dict[str, object]:
    latest_dir = _latest_timestamped_dir(root)
    if latest_dir is None:
        return {}
    return _read_json_file(latest_dir / filename)


def _read_latest_compatible_json(
    root: Path,
    filename: str,
    matcher: Callable[[dict[str, object]], bool],
) -> dict[str, object]:
    for candidate in reversed(_timestamped_dirs(root)):
        payload = _read_json_file(candidate / filename)
        if payload and matcher(payload):
            if filename == "symbol_restrictions.json":
                return _sanitize_entry_symbols_payload(payload, set())
            if filename == "schedule_tuning.json":
                return _sanitize_entry_schedule_payload(payload, set())
            return payload
    return {}


def _latest_timestamped_dir(root: Path) -> Path | None:
    candidates = _timestamped_dirs(root)
    if not candidates:
        return None
    return candidates[-1]


def _timestamped_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def _read_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_symbols_payload_matches_runtime(
    payload: dict[str, object],
    allowed_symbols: set[str],
) -> bool:
    if not payload or not allowed_symbols:
        return bool(payload)
    referenced = set(_normalize_symbol_list(payload.get("proposed_blocked_symbols", [])))
    referenced.update(_normalize_symbol_list(payload.get("proposed_blocked_long_symbols", [])))
    referenced.update(_normalize_symbol_list(payload.get("proposed_blocked_short_symbols", [])))
    for row in payload.get("symbol_direction_breakdown", []):
        if isinstance(row, dict):
            symbol = str(row.get("symbol", "")).strip().upper()
            if symbol:
                referenced.add(symbol)
    return not referenced or referenced.issubset(allowed_symbols)


def _sanitize_entry_symbols_payload(
    payload: dict[str, object],
    allowed_symbols: set[str],
) -> dict[str, object]:
    if not payload:
        return {
            "changed": False,
            "reason": "",
            "evidence_source": "",
            "evidence_counts": {},
            "proposed_blocked_symbols": [],
            "proposed_blocked_long_symbols": [],
            "proposed_blocked_short_symbols": [],
            "symbol_direction_breakdown": [],
        }
    blocked_symbols = [
        symbol
        for symbol in _normalize_symbol_list(payload.get("proposed_blocked_symbols", []))
        if not allowed_symbols or symbol in allowed_symbols
    ]
    blocked_long_symbols = [
        symbol
        for symbol in _normalize_symbol_list(payload.get("proposed_blocked_long_symbols", []))
        if not allowed_symbols or symbol in allowed_symbols
    ]
    blocked_short_symbols = [
        symbol
        for symbol in _normalize_symbol_list(payload.get("proposed_blocked_short_symbols", []))
        if not allowed_symbols or symbol in allowed_symbols
    ]
    direction_rows: list[dict[str, object]] = []
    for row in payload.get("symbol_direction_breakdown", []):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        if allowed_symbols and symbol not in allowed_symbols:
            continue
        direction_rows.append({**row, "symbol": symbol})
    has_restrictions = bool(
        blocked_symbols or blocked_long_symbols or blocked_short_symbols or direction_rows
    )
    reason = str(payload.get("reason", ""))
    evidence_source = str(payload.get("evidence_source", ""))
    evidence_counts = dict(payload.get("evidence_counts", {}))
    if not has_restrictions and allowed_symbols and not _entry_symbols_payload_matches_runtime(payload, allowed_symbols):
        reason = "latest restrictions artifact is not compatible with current runtime universe"
        evidence_source = ""
        evidence_counts = {}
    return {
        "changed": bool(payload.get("changed", False)) and has_restrictions,
        "reason": reason,
        "evidence_source": evidence_source,
        "evidence_counts": evidence_counts,
        "proposed_blocked_symbols": blocked_symbols,
        "proposed_blocked_long_symbols": blocked_long_symbols,
        "proposed_blocked_short_symbols": blocked_short_symbols,
        "symbol_direction_breakdown": direction_rows,
    }


def _entry_schedule_payload_matches_runtime(
    payload: dict[str, object],
    allowed_hours: set[int],
) -> bool:
    if not payload or not allowed_hours:
        return bool(payload)
    proposed_hours = _normalize_hour_list(payload.get("proposed_hours", []))
    return not proposed_hours or set(proposed_hours).issubset(allowed_hours)


def _sanitize_entry_schedule_payload(
    payload: dict[str, object],
    allowed_hours: set[int],
) -> dict[str, object]:
    if not payload:
        return {
            "changed": False,
            "reason": "",
            "evidence_source": "",
            "evidence_counts": {},
            "proposed_hours": [],
        }
    proposed_hours = [
        hour for hour in _normalize_hour_list(payload.get("proposed_hours", [])) if not allowed_hours or hour in allowed_hours
    ]
    reason = str(payload.get("reason", ""))
    evidence_source = str(payload.get("evidence_source", ""))
    evidence_counts = dict(payload.get("evidence_counts", {}))
    if not proposed_hours and allowed_hours and not _entry_schedule_payload_matches_runtime(payload, allowed_hours):
        reason = "latest schedule artifact is not compatible with current runtime hours"
        evidence_source = ""
        evidence_counts = {}
    return {
        "changed": bool(payload.get("changed", False)) and bool(proposed_hours),
        "reason": reason,
        "evidence_source": evidence_source,
        "evidence_counts": evidence_counts,
        "proposed_hours": proposed_hours,
    }


def _runtime_universe_payload_matches_runtime(
    payload: dict[str, object],
    allowed_symbols: set[str],
) -> bool:
    if not payload or not allowed_symbols:
        return bool(payload)
    configured = set(_normalize_symbol_list(payload.get("configured_symbols", [])))
    if configured and configured != allowed_symbols:
        return False
    referenced: set[str] = set()
    for key in (
        "current_allowed_symbols",
        "current_effective_symbols",
        "optimizer_best_symbols",
        "walk_forward_latest_symbols",
        "consensus_symbols",
        "proposed_allowed_symbols",
        "proposed_effective_symbols",
        "additions",
        "removals",
    ):
        referenced.update(_normalize_symbol_list(payload.get(key, [])))
    return not referenced or referenced.issubset(allowed_symbols)


def _sanitize_runtime_universe_payload(
    payload: dict[str, object],
    allowed_symbols: set[str],
) -> dict[str, object]:
    if not payload:
        return {
            "changed": False,
            "reason": "",
            "guardrails": {},
            "current_allowed_symbols": [],
            "current_effective_symbols": [],
            "optimizer_best_symbols": [],
            "walk_forward_latest_symbols": [],
            "consensus_symbols": [],
            "proposed_allowed_symbols": [],
            "proposed_effective_symbols": [],
            "optimizer_summary": {},
            "walk_forward_summary": {},
        }

    def filtered(key: str) -> list[str]:
        return [
            symbol
            for symbol in _normalize_symbol_list(payload.get(key, []))
            if not allowed_symbols or symbol in allowed_symbols
        ]

    current_allowed_symbols = filtered("current_allowed_symbols")
    current_effective_symbols = filtered("current_effective_symbols")
    optimizer_best_symbols = filtered("optimizer_best_symbols")
    walk_forward_latest_symbols = filtered("walk_forward_latest_symbols")
    consensus_symbols = filtered("consensus_symbols")
    proposed_allowed_symbols = filtered("proposed_allowed_symbols")
    proposed_effective_symbols = filtered("proposed_effective_symbols")
    reason = str(payload.get("reason", ""))
    if (
        not any(
            [
                current_allowed_symbols,
                current_effective_symbols,
                optimizer_best_symbols,
                walk_forward_latest_symbols,
                consensus_symbols,
                proposed_allowed_symbols,
                proposed_effective_symbols,
            ]
        )
        and allowed_symbols
        and not _runtime_universe_payload_matches_runtime(payload, allowed_symbols)
    ):
        reason = "latest runtime universe artifact is not compatible with current runtime universe"
    return {
        "changed": bool(payload.get("changed", False)) and bool(proposed_effective_symbols),
        "reason": reason,
        "guardrails": dict(payload.get("guardrails", {})),
        "current_allowed_symbols": current_allowed_symbols,
        "current_effective_symbols": current_effective_symbols,
        "optimizer_best_symbols": optimizer_best_symbols,
        "walk_forward_latest_symbols": walk_forward_latest_symbols,
        "consensus_symbols": consensus_symbols,
        "proposed_allowed_symbols": proposed_allowed_symbols,
        "proposed_effective_symbols": proposed_effective_symbols,
        "optimizer_summary": dict(payload.get("optimizer_summary", {})),
        "walk_forward_summary": dict(payload.get("walk_forward_summary", {})),
    }


def _normalize_symbol_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = str(value).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def _normalize_hour_list(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        hour = int(value)
        if hour in seen:
            continue
        seen.add(hour)
        result.append(hour)
    return result


def _effective_runtime_source(
    payload: dict[str, object],
    source_name: str,
) -> dict[str, object]:
    for source in payload.get("sources", []):
        if isinstance(source, dict) and str(source.get("source", "")) == source_name:
            return source
    return {}


def _load_display_config(
    config_path: str | Path,
    effective_config_path: str | Path | None,
):
    candidates: list[Path] = []
    if effective_config_path:
        candidates.append(Path(effective_config_path).resolve())
    candidates.append(Path(config_path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return load_config(candidate)
    return load_config(config_path)


def _positions_from_state(
    positions_payload: dict[str, Any],
    events_payload: object,
    strategy: StrategySection,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol, position in sorted(positions_payload.items()):
        direction = str(position.get("direction", ""))
        quantity_lots = int(position.get("quantity_lots", 0))
        entry_price = float(position.get("entry_price", 0.0))
        current_price = float(position.get("current_price", 0.0))
        stop_price = float(position.get("stop_price", 0.0))
        take_profit = float(position.get("take_profit", 0.0))
        instrument = position.get("instrument", {})
        lot_size = int(instrument.get("lot_size", 1)) if isinstance(instrument, dict) else 1
        quantity_units = max(0, quantity_lots * max(1, lot_size))
        unrealized_pnl_rub = _position_unrealized_pnl(
            direction,
            entry_price,
            current_price,
            quantity_units,
        )
        stop_pct = _position_stop_pct(direction, entry_price, stop_price)
        take_pct = _position_take_pct(direction, entry_price, take_profit)
        reward_risk_ratio = (take_pct / stop_pct) if stop_pct > 0 else 0.0
        trailing = _position_trailing_snapshot(
            symbol,
            position,
            events_payload,
            strategy,
        )
        rows.append(
            {
                "symbol": symbol,
                "direction": direction,
                "quantity_lots": quantity_lots,
                "entry_price": entry_price,
                "current_price": current_price,
                "stop_price": stop_price,
                "take_profit": take_profit,
                "margin_requirement": float(position.get("margin_requirement", 0.0)),
                "signal_strength": float(position.get("signal_strength", 0.0)),
                "opened_at": str(position.get("opened_at", "")),
                "updated_at": str(position.get("updated_at", "")),
                "lot_size": lot_size,
                "quantity_units": quantity_units,
                "unrealized_pnl_rub": unrealized_pnl_rub,
                "stop_pct": stop_pct,
                "take_pct": take_pct,
                "reward_risk_ratio": reward_risk_ratio,
                **trailing,
            }
        )
    return rows


def _render_positions(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p>Нет открытых paper-позиций.</p>"
    body = "".join(
        [
            "<tr>"
            f"<td>{_escape(str(row['symbol']))}</td>"
            f"<td>{_escape(str(row['direction']))}</td>"
            f"<td>{int(row['quantity_lots'])}</td>"
            f"<td>{float(row['entry_price']):.6f}</td>"
            f"<td>{float(row['current_price']):.6f}</td>"
            f"<td class=\"{_escape(_number_tone(float(row['unrealized_pnl_rub'])))}\">{_fmt_money(float(row['unrealized_pnl_rub']))}</td>"
            f"<td>{_render_price_metric(float(row['stop_price']), float(row['stop_pct']))}</td>"
            f"<td>{_render_price_metric(float(row['take_profit']), float(row['take_pct']))}</td>"
            f"<td>{float(row['reward_risk_ratio']):.2f}</td>"
            f"<td>{_render_trailing_cell(row)}</td>"
            f"<td>{_fmt_money(float(row['margin_requirement']))}</td>"
            "</tr>"
            for row in rows
        ]
    )
    return (
        "<table><tr><th>Symbol</th><th>Dir</th><th>Lots</th><th>Entry</th><th>Current</th>"
        "<th>PnL</th><th>Stop</th><th>Take</th><th>R/R</th><th>Trailing</th><th>Margin</th></tr>"
        f"{body}</table>"
    )


def _render_direction_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p>Нет direction-level статистики.</p>"
    body = "".join(
        [
            "<tr>"
            f"<td>{_escape(str(row.get('symbol', '')))}</td>"
            f"<td>{_escape(str(row.get('direction', '')))}</td>"
            f"<td>{int(row.get('trades', 0))}</td>"
            f"<td>{float(row.get('win_rate_pct', 0.0)):.3f}%</td>"
            f"<td>{_fmt_money(float(row.get('net_pnl_rub', 0.0)))}</td>"
            f"<td>{float(row.get('profit_factor', 0.0)):.3f}</td>"
            f"<td>{_fmt_money(float(row.get('expectancy_rub', 0.0)))}</td>"
            "</tr>"
            for row in rows
        ]
    )
    return (
        "<table><tr><th>Symbol</th><th>Direction</th><th>Trades</th><th>Win rate</th>"
        "<th>Net PnL</th><th>PF</th><th>Expectancy</th></tr>"
        f"{body}</table>"
    )


def _render_sources(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p>Нет active autotune sources.</p>"
    body = "".join(
        [
            "<tr>"
            f"<td>{_escape(str(row.get('source', '')))}</td>"
            f"<td>{_bool_text(bool(row.get('changed', False)))}</td>"
            f"<td>{_escape(json.dumps(row.get('selected_values', {}), ensure_ascii=False))}</td>"
            f"<td>{_escape(str(row.get('activation', {}).get('reason', '')))}</td>"
            "</tr>"
            for row in rows
        ]
    )
    return (
        "<table><tr><th>Source</th><th>Changed</th><th>Selected values</th><th>Activation</th></tr>"
        f"{body}</table>"
    )


def _render_key_values(values: dict[str, object]) -> str:
    if not values:
        return "<p>Активных overrides сейчас нет.</p>"
    body = "".join(
        [
            f"<tr><th>{_escape(str(key))}</th><td class=\"mono\">{_escape(json.dumps(value, ensure_ascii=False))}</td></tr>"
            for key, value in values.items()
        ]
    )
    return f"<table>{body}</table>"


def _render_named_value_table(
    values: dict[str, object],
    *,
    key_label: str,
    value_label: str,
    empty_message: str,
) -> str:
    if not values:
        return f"<p>{_escape(empty_message)}</p>"
    body = "".join(
        [
            f"<tr><th>{_escape(str(key))}</th><td>{_escape(str(value))}</td></tr>"
            for key, value in values.items()
        ]
    )
    return (
        f"<table><tr><th>{_escape(key_label)}</th><th>{_escape(value_label)}</th></tr>"
        f"{body}</table>"
    )


def _render_chips(values: list[object], *, tone: str = "neutral") -> str:
    chip_class = f"chip {_escape(tone)}" if tone != "neutral" else "chip"
    if not values:
        return f"<span class=\"{chip_class}\">-</span>"
    return "".join(f"<span class=\"{chip_class}\">{_escape(str(value))}</span>" for value in values)


def _render_badge(text: str, *, tone: str = "neutral") -> str:
    return f"<span class=\"badge {_escape(tone)}\">{_escape(text)}</span>"


def _render_stat_card(label: str, value: str, note: str, *, tone: str = "neutral") -> str:
    value_class = f" {tone}" if tone != "neutral" else ""
    return (
        "<div class=\"card\">"
        f"<div class=\"kpi-label\">{_escape(label)}</div>"
        f"<div class=\"kpi-value{value_class}\">{_escape(value)}</div>"
        f"<div class=\"kpi-note\">{_escape(note)}</div>"
        "</div>"
    )


def _number_tone(value: float) -> str:
    if value > 0:
        return "good"
    if value < 0:
        return "bad"
    return "neutral"


def _fmt_timestamp(value: str) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    zone = parsed.strftime("%z")
    zone_suffix = f" UTC{zone[:3]}:{zone[3:]}" if zone else ""
    return parsed.strftime("%Y-%m-%d %H:%M:%S") + zone_suffix


def _fmt_money(value: float) -> str:
    return f"{value:,.2f} RUB".replace(",", " ")


def _bool_text(value: bool) -> str:
    return "да" if value else "нет"


def _render_inline_list(values: list[object]) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(items) if items else "-"


def _render_counts_summary(payload: dict[str, object]) -> str:
    if not payload:
        return "-"
    parts: list[str] = []
    if "combined_trades" in payload:
        parts.append(f"combined={payload['combined_trades']}")
    if "closed_trades" in payload:
        parts.append(f"closed={payload['closed_trades']}")
    if "deduplicated_feedback_trades" in payload:
        parts.append(f"feedback={payload['deduplicated_feedback_trades']}")
    if payload.get("duplicate_feedback_trades"):
        parts.append(f"dupes={payload['duplicate_feedback_trades']}")
    return ", ".join(parts) if parts else "-"


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _position_unrealized_pnl(
    direction: str,
    entry_price: float,
    current_price: float,
    quantity_units: int,
) -> float:
    if direction == "short":
        return (entry_price - current_price) * quantity_units
    return (current_price - entry_price) * quantity_units


def _position_stop_pct(direction: str, entry_price: float, stop_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    if direction == "short":
        return max(0.0, (stop_price - entry_price) / entry_price * 100.0)
    return max(0.0, (entry_price - stop_price) / entry_price * 100.0)


def _position_take_pct(direction: str, entry_price: float, take_profit: float) -> float:
    if entry_price <= 0:
        return 0.0
    if direction == "short":
        return max(0.0, (entry_price - take_profit) / entry_price * 100.0)
    return max(0.0, (take_profit - entry_price) / entry_price * 100.0)


def _position_trailing_snapshot(
    symbol: str,
    position: dict[str, Any],
    events_payload: object,
    strategy: StrategySection,
) -> dict[str, object]:
    direction = str(position.get("direction", ""))
    quantity_lots = int(position.get("quantity_lots", 0))
    entry_price = float(position.get("entry_price", 0.0))
    current_price = float(position.get("current_price", 0.0))
    stop_price = float(position.get("stop_price", 0.0))
    take_profit = float(position.get("take_profit", 0.0))
    instrument = position.get("instrument", {})
    lot_size = int(instrument.get("lot_size", 1)) if isinstance(instrument, dict) else 1
    quantity_units = max(0, quantity_lots * max(1, lot_size))
    breakeven_trigger_pct = max(0.0, float(strategy.breakeven_trigger_pct))
    trigger_rub = max(0.0, float(strategy.trailing_profit_trigger_rub))
    lock_ratio = max(0.0, min(1.0, float(strategy.trailing_profit_lock_ratio)))
    if bool(position.get("runner_active", False)):
        if direction == "short":
            protected_profit_rub = max(0.0, (entry_price - stop_price) * quantity_units)
        else:
            protected_profit_rub = max(0.0, (stop_price - entry_price) * quantity_units)
        return {
            "trailing_status": "runner",
            "trailing_mode": "take-profit-runner",
            "trailing_breakeven_trigger_pct": breakeven_trigger_pct,
            "trailing_trigger_rub": trigger_rub,
            "trailing_lock_ratio": max(0.0, min(1.0, float(strategy.runner_profit_lock_ratio))),
            "trailing_trigger_price": take_profit,
            "trailing_first_lock_price": float(position.get("runner_activation_price", take_profit)),
            "trailing_remaining_rub": 0.0,
            "trailing_protected_profit_rub": protected_profit_rub,
            "trailing_activated_at": str(position.get("runner_activated_at", "")),
            "runner_extreme_price": float(position.get("runner_extreme_price", 0.0)),
        }
    if quantity_units <= 0:
        return {
            "trailing_status": "disabled",
            "trailing_mode": "",
            "trailing_breakeven_trigger_pct": breakeven_trigger_pct,
            "trailing_trigger_rub": trigger_rub,
            "trailing_lock_ratio": lock_ratio,
            "trailing_trigger_price": 0.0,
            "trailing_first_lock_price": 0.0,
            "trailing_remaining_rub": 0.0,
            "trailing_protected_profit_rub": 0.0,
            "trailing_activated_at": "",
        }

    unrealized_pnl_rub = _position_unrealized_pnl(direction, entry_price, current_price, quantity_units)
    if breakeven_trigger_pct > 0:
        trigger_move = entry_price * breakeven_trigger_pct / 100.0
        trigger_rub = entry_price * quantity_units * breakeven_trigger_pct / 100.0
        if direction == "short":
            trigger_price = entry_price - trigger_move
            first_lock_price = entry_price
            protected_profit_rub = max(0.0, (entry_price - stop_price) * quantity_units)
        else:
            trigger_price = entry_price + trigger_move
            first_lock_price = entry_price
            protected_profit_rub = max(0.0, (stop_price - entry_price) * quantity_units)
    else:
        if trigger_rub <= 0 or lock_ratio <= 0:
            return {
                "trailing_status": "disabled",
                "trailing_mode": "",
                "trailing_breakeven_trigger_pct": breakeven_trigger_pct,
                "trailing_trigger_rub": trigger_rub,
                "trailing_lock_ratio": lock_ratio,
                "trailing_trigger_price": 0.0,
                "trailing_first_lock_price": 0.0,
                "trailing_remaining_rub": 0.0,
                "trailing_protected_profit_rub": 0.0,
                "trailing_activated_at": "",
            }
        trigger_move = trigger_rub / quantity_units
        first_lock_move = trigger_rub * lock_ratio / quantity_units
        if direction == "short":
            trigger_price = entry_price - trigger_move
            first_lock_price = entry_price - first_lock_move
            protected_profit_rub = max(0.0, (entry_price - stop_price) * quantity_units)
        else:
            trigger_price = entry_price + trigger_move
            first_lock_price = entry_price + first_lock_move
            protected_profit_rub = max(0.0, (stop_price - entry_price) * quantity_units)

    trailing_event = _latest_trailing_event(
        symbol,
        str(position.get("opened_at", "")),
        events_payload,
    )
    return {
        "trailing_status": "active" if trailing_event or protected_profit_rub > 0 else "arming",
        "trailing_mode": "breakeven-pct" if breakeven_trigger_pct > 0 else "rub-trigger",
        "trailing_breakeven_trigger_pct": breakeven_trigger_pct,
        "trailing_trigger_rub": trigger_rub,
        "trailing_lock_ratio": lock_ratio,
        "trailing_trigger_price": trigger_price,
        "trailing_first_lock_price": first_lock_price,
        "trailing_remaining_rub": max(0.0, trigger_rub - unrealized_pnl_rub),
        "trailing_protected_profit_rub": protected_profit_rub,
        "trailing_activated_at": str(trailing_event.get("timestamp", "")) if trailing_event else "",
    }


def _latest_trailing_event(
    symbol: str,
    opened_at: str,
    events_payload: object,
) -> dict[str, object]:
    opened_timestamp = _parse_timestamp(opened_at)
    if not isinstance(events_payload, list):
        return {}
    for item in reversed(events_payload):
        if not isinstance(item, dict):
            continue
        if str(item.get("symbol", "")).strip().upper() != symbol.strip().upper():
            continue
        if str(item.get("action", "")) != "protect":
            continue
        if str(item.get("reason", "")) != "trailing-profit-protection":
            continue
        event_timestamp = _parse_timestamp(str(item.get("timestamp", "")))
        if opened_timestamp is not None and event_timestamp is not None and event_timestamp < opened_timestamp:
            continue
        return item
    return {}


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _render_price_metric(price: float, pct: float) -> str:
    return f"{price:.6f}<div class=\"cell-note mono\">{pct:.3f}%</div>"


def _render_trailing_cell(row: dict[str, object]) -> str:
    status = str(row.get("trailing_status", "disabled"))
    if status == "disabled":
        return "<span class=\"cell-note\">off</span>"
    if status == "runner":
        activated_at = str(row.get("trailing_activated_at", ""))
        activation_price = float(row.get("trailing_first_lock_price", 0.0))
        extreme_price = float(row.get("runner_extreme_price", 0.0))
        protected_profit_rub = float(row.get("trailing_protected_profit_rub", 0.0))
        state_line = f"runner since {_fmt_timestamp(activated_at)}" if activated_at else "runner active"
        return (
            f"<div class=\"cell-note\">{_escape(state_line)}</div>"
            f"<div class=\"cell-note mono\">activation @ {activation_price:.6f}</div>"
            f"<div class=\"cell-note mono\">extreme @ {extreme_price:.6f}</div>"
            f"<div class=\"cell-note mono\">protects {_escape(_fmt_money(protected_profit_rub))}</div>"
        )

    arm_price = float(row.get("trailing_trigger_price", 0.0))
    first_lock_price = float(row.get("trailing_first_lock_price", 0.0))
    trigger_rub = float(row.get("trailing_trigger_rub", 0.0))
    lock_ratio = float(row.get("trailing_lock_ratio", 0.0))
    protected_profit_rub = float(row.get("trailing_protected_profit_rub", 0.0))
    activated_at = str(row.get("trailing_activated_at", ""))
    breakeven_trigger_pct = float(row.get("trailing_breakeven_trigger_pct", 0.0))
    if activated_at:
        state_line = f"active since {_fmt_timestamp(activated_at)}"
    else:
        state_line = f"arming, left {_fmt_money(float(row.get('trailing_remaining_rub', 0.0)))}"
    mode_line = (
        f"breakeven {breakeven_trigger_pct:.2f}%"
        if breakeven_trigger_pct > 0
        else f"trigger {trigger_rub:.2f} RUB"
    )
    return (
        f"<div class=\"cell-note\">{_escape(state_line)}</div>"
        f"<div class=\"cell-note mono\">arm @ {arm_price:.6f}</div>"
        f"<div class=\"cell-note mono\">first lock @ {first_lock_price:.6f}</div>"
        f"<div class=\"cell-note mono\">{mode_line} / lock {lock_ratio:.0%}</div>"
        f"<div class=\"cell-note mono\">protects {_escape(_fmt_money(protected_profit_rub))}</div>"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Samosbor paper dashboard")
    parser.add_argument("--config", required=True, help="Path to the runtime config")
    parser.add_argument(
        "--effective-config",
        help="Optional effective runtime config path for display purposes",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8790, help="Bind port")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve_dashboard(
        args.config,
        effective_config_path=args.effective_config,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
