from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..config import AppConfig

_SECTION_PATTERN = re.compile(r"^\s*\[.+\]\s*$")
_KEY_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_STRATEGY_OVERRIDE_ORDER = [
    "style",
    "fast_window",
    "slow_window",
    "require_breakout",
    "opening_range_bars",
    "rel_volume_threshold",
    "atr_stop_multiple",
    "reward_to_risk",
    "breakeven_trigger_pct",
    "trailing_profit_trigger_rub",
    "trailing_profit_lock_ratio",
    "min_signal_strength",
    "min_trend_strength",
    "adx_min",
    "rsi_long_max",
    "rsi_short_min",
    "allowed_entry_hours",
    "allowed_symbols",
    "blocked_symbols",
    "blocked_long_symbols",
    "blocked_short_symbols",
]
_REQUIRED_CONFIRMATIONS = 2
_MIN_RUNTIME_SYMBOLS_FOR_ENTRY_RESTRICTIONS = 8
_MAX_ENTRY_RESTRICTED_SYMBOL_SHARE = 0.25
_MIN_ALLOWED_RUNTIME_SYMBOLS = 5
_MIN_ALLOWED_RUNTIME_COVERAGE_RATIO = 0.5
_MIN_ACTIVE_ENTRY_HOURS = 6
_MIN_ACTIVE_ENTRY_HOUR_COVERAGE_RATIO = 0.75


def default_effective_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.name.endswith(".effective.toml"):
        return path
    suffix = path.suffix or ".toml"
    return path.with_name(f"{path.stem}.effective{suffix}")


def base_strategy_values(config: AppConfig) -> dict[str, object]:
    return {
        "style": config.strategy.style,
        "fast_window": config.strategy.fast_window,
        "slow_window": config.strategy.slow_window,
        "require_breakout": config.strategy.require_breakout,
        "opening_range_bars": config.strategy.opening_range_bars,
        "rel_volume_threshold": config.strategy.rel_volume_threshold,
        "atr_stop_multiple": config.strategy.atr_stop_multiple,
        "reward_to_risk": config.strategy.reward_to_risk,
        "breakeven_trigger_pct": config.strategy.breakeven_trigger_pct,
        "trailing_profit_trigger_rub": config.strategy.trailing_profit_trigger_rub,
        "trailing_profit_lock_ratio": config.strategy.trailing_profit_lock_ratio,
        "min_signal_strength": config.strategy.min_signal_strength,
        "min_trend_strength": config.strategy.min_trend_strength,
        "adx_min": config.strategy.adx_min,
        "rsi_long_max": config.strategy.rsi_long_max,
        "rsi_short_min": config.strategy.rsi_short_min,
        "allowed_entry_hours": list(config.strategy.allowed_entry_hours),
        "allowed_symbols": list(config.strategy.allowed_symbols),
        "blocked_symbols": list(config.strategy.blocked_symbols),
        "blocked_long_symbols": list(config.strategy.blocked_long_symbols),
        "blocked_short_symbols": list(config.strategy.blocked_short_symbols),
    }


def build_effective_strategy_overrides(
    config: AppConfig,
    *,
    source_summaries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    summaries = source_summaries
    if summaries is None:
        autotune_dir = config.autotune_dir()
        summaries = summarize_effective_config_sources(autotune_dir)
    summaries = align_effective_config_sources(config, summaries)
    overrides: dict[str, object] = {}
    for item in summaries:
        overrides.update(item["selected_values"])
    return overrides


def align_effective_config_sources(
    config: AppConfig,
    source_summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    base_values = base_strategy_values(config)
    configured_symbols = {
        instrument.symbol.strip().upper()
        for instrument in config.data.instruments
        if instrument.symbol.strip()
    }
    base_hours = [int(value) for value in config.strategy.allowed_entry_hours]
    aligned: list[dict[str, object]] = []

    for source in source_summaries:
        source_name = str(source.get("source", ""))
        candidate_allowed_hours = [] if source_name == "entry-schedule" else base_hours
        current_values = _normalize_override_values(
            source.get("current_values", {}),
            configured_symbols=configured_symbols,
            allowed_hours=[],
        )
        candidate_values = _normalize_override_values(
            source.get("candidate_values", {}),
            configured_symbols=configured_symbols,
            allowed_hours=candidate_allowed_hours,
        )
        selected_values = _normalize_override_values(
            source.get("selected_values", {}),
            configured_symbols=configured_symbols,
            allowed_hours=candidate_allowed_hours,
        )
        activation = dict(source.get("activation", {}))

        if current_values and not _values_match_base(base_values, current_values):
            selected_values = {}
            activation.update(
                {
                    "confirmed": False,
                    "pending_activation": False,
                    "reason": "latest tuning payload does not match the current base runtime config",
                }
            )
        elif selected_values:
            selected_values = {
                key: value for key, value in selected_values.items() if base_values.get(key) != value
            }
            if not selected_values:
                activation.update(
                    {
                        "confirmed": False,
                        "pending_activation": False,
                        "reason": "latest tuning payload is not compatible with the current runtime universe",
                    }
                )
            else:
                guardrail_reason = _source_guardrail_reason(
                    source_name=source_name,
                    selected_values=selected_values,
                    configured_symbols=configured_symbols,
                    base_hours=base_hours,
                )
                if guardrail_reason:
                    selected_values = {}
                    activation.update(
                        {
                            "confirmed": False,
                            "pending_activation": False,
                            "reason": guardrail_reason,
                        }
                    )

        aligned.append(
            {
                **source,
                "current_values": current_values,
                "candidate_values": candidate_values,
                "selected_values": selected_values,
                "activation": activation,
            }
        )
    return aligned


def summarize_effective_config_sources(
    autotune_dir: Path,
    *,
    required_confirmations: int = _REQUIRED_CONFIRMATIONS,
) -> list[dict[str, object]]:
    return [
        _build_source_summary(
            autotune_dir=autotune_dir,
            source_name="strategy",
            json_name="strategy_tuning.json",
            current_value_builder=_strategy_current_values,
            candidate_value_builder=_strategy_candidate_values,
            required_confirmations=required_confirmations,
        ),
        _build_source_summary(
            autotune_dir=autotune_dir,
            source_name="exits",
            json_name="exit_tuning.json",
            current_value_builder=_exit_current_values,
            candidate_value_builder=_exit_candidate_values,
            required_confirmations=required_confirmations,
        ),
        _build_source_summary(
            autotune_dir=autotune_dir,
            source_name="entry-schedule",
            json_name="schedule_tuning.json",
            current_value_builder=_entry_schedule_current_values,
            candidate_value_builder=_entry_schedule_candidate_values,
            required_confirmations=required_confirmations,
        ),
        _build_source_summary(
            autotune_dir=autotune_dir,
            source_name="entry-quality",
            json_name="entry_quality_tuning.json",
            current_value_builder=_entry_quality_current_values,
            candidate_value_builder=_entry_quality_candidate_values,
            required_confirmations=required_confirmations,
        ),
        _build_source_summary(
            autotune_dir=autotune_dir,
            source_name="universe-selection",
            json_name="universe_selection.json",
            current_value_builder=_universe_selection_current_values,
            candidate_value_builder=_universe_selection_candidate_values,
            required_confirmations=required_confirmations,
        ),
        _build_source_summary(
            autotune_dir=autotune_dir,
            source_name="entry-symbols",
            json_name="symbol_restrictions.json",
            current_value_builder=_entry_symbols_current_values,
            candidate_value_builder=_entry_symbols_candidate_values,
            required_confirmations=required_confirmations,
        ),
    ]


def build_effective_config_guardrail_payload(
    *,
    base_values: dict[str, object],
    source_summaries: list[dict[str, object]],
    paper_report: dict[str, object],
    guardrail_days: int = 3,
    min_recent_trades: int = 6,
) -> dict[str, object]:
    active_sources: list[str] = []
    active_override_keys: list[str] = []
    for source in source_summaries:
        changed_keys = [
            key
            for key, value in source.get("selected_values", {}).items()
            if base_values.get(key) != value
        ]
        if changed_keys:
            active_sources.append(str(source.get("source", "")))
            active_override_keys.extend(changed_keys)

    summary = dict(paper_report.get("summary", {}))
    comparison = dict(paper_report.get("comparison_to_previous_window", {}))
    previous_summary = dict(comparison.get("summary", {}))
    delta = dict(comparison.get("delta", {}))
    portfolio = dict(paper_report.get("portfolio", {}))

    trades = int(summary.get("trades", 0))
    net_pnl = float(summary.get("net_pnl_rub", 0.0))
    expectancy = float(summary.get("expectancy_rub", 0.0))
    profit_factor = float(summary.get("profit_factor", 0.0))
    previous_trades = int(previous_summary.get("trades", 0))
    delta_net_pnl = float(delta.get("net_pnl_rub", 0.0))
    trading_halted = bool(portfolio.get("trading_halted", False))

    enough_trades = trades >= min_recent_trades
    recent_window_negative = net_pnl < 0 and expectancy < 0 and profit_factor < 1.0
    deterioration_confirmed = delta_net_pnl < 0 or previous_trades == 0
    rollback_to_base = bool(active_sources) and (
        trading_halted or (enough_trades and recent_window_negative and deterioration_confirmed)
    )

    if rollback_to_base and trading_halted:
        reason = "portfolio drawdown halt is active while autotune overrides are applied"
    elif rollback_to_base:
        reason = "recent paper window deteriorated while autotune overrides were active"
    elif not active_sources:
        reason = "no active overrides versus the base runtime config"
    elif not enough_trades:
        reason = f"need at least {min_recent_trades} recent trades before rollback guardrail can judge overrides"
    elif not recent_window_negative:
        reason = "recent paper window does not show a broad enough deterioration"
    else:
        reason = "recent paper window is weak but not worse than the previous comparison window"

    return {
        "rollback_to_base": rollback_to_base,
        "reason": reason,
        "guardrails": {
            "days": guardrail_days,
            "min_recent_trades": min_recent_trades,
            "require_negative_net_pnl": True,
            "require_negative_expectancy": True,
            "require_profit_factor_below_one": True,
            "require_worse_than_previous_window": True,
            "always_rollback_if_trading_halted": True,
        },
        "active_sources": active_sources,
        "active_override_keys": sorted(set(active_override_keys)),
        "recent_summary": summary,
        "previous_summary": previous_summary,
        "comparison_delta": delta,
        "trading_halted": trading_halted,
    }


def write_effective_config(
    source_config_path: str | Path,
    output_path: str | Path,
    *,
    strategy_overrides: dict[str, object],
) -> None:
    source_path = Path(source_config_path).resolve()
    target_path = Path(output_path).resolve()
    rendered = _apply_strategy_overrides(
        source_path.read_text(encoding="utf-8"),
        strategy_overrides,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(rendered, encoding="utf-8")


def _build_source_summary(
    *,
    autotune_dir: Path,
    source_name: str,
    json_name: str,
    current_value_builder,
    candidate_value_builder,
    required_confirmations: int,
) -> dict[str, object]:
    payload_paths = _payload_history_paths(autotune_dir / source_name, json_name)
    payload_path = payload_paths[-1] if payload_paths else None
    if payload_path is None:
        return {
            "source": source_name,
            "artifact_path": "",
            "changed": False,
            "current_values": {},
            "candidate_values": {},
            "selected_values": {},
            "reason": "no tuning artifacts found",
            "activation": {
                "required_confirmations": required_confirmations,
                "confirmation_count": 0,
                "confirmed": False,
                "pending_activation": False,
                "reason": "no tuning artifacts found",
            },
        }

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    changed = bool(payload.get("changed", False))
    current_values = current_value_builder(payload)
    candidate_values = candidate_value_builder(payload)
    activation = {
        "required_confirmations": required_confirmations,
        "confirmation_count": 0,
        "confirmed": False,
        "pending_activation": False,
        "reason": "latest tuning run produced no active runtime override",
    }
    selected_values: dict[str, object] = {}

    if changed and candidate_values:
        confirmation_count = _candidate_confirmation_count(
            payload_paths,
            candidate_value_builder=candidate_value_builder,
            target_values=candidate_values,
        )
        confirmed = confirmation_count >= max(1, required_confirmations)
        activation = {
            "required_confirmations": required_confirmations,
            "confirmation_count": confirmation_count,
            "confirmed": confirmed,
            "pending_activation": not confirmed,
            "reason": (
                f"candidate confirmed across {confirmation_count} consecutive tuning runs"
                if confirmed
                else f"candidate is waiting for {required_confirmations} consecutive confirmations"
            ),
        }
        selected_values = candidate_values if confirmed else {}

    return {
        "source": source_name,
        "artifact_path": str(payload_path),
        "changed": changed,
        "current_values": current_values,
        "candidate_values": candidate_values,
        "selected_values": selected_values,
        "reason": str(payload.get("reason", "latest tuning payload loaded")),
        "activation": activation,
    }


def _payload_history_paths(source_dir: Path, json_name: str) -> list[Path]:
    if not source_dir.exists():
        return []
    return sorted(
        [
            path / json_name
            for path in source_dir.iterdir()
            if path.is_dir() and (path / json_name).exists()
        ],
        key=lambda path: path.parent.name,
    )


def _candidate_confirmation_count(
    payload_paths: list[Path],
    *,
    candidate_value_builder,
    target_values: dict[str, object],
) -> int:
    count = 0
    for payload_path in reversed(payload_paths):
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not payload.get("changed", False):
            break
        if candidate_value_builder(payload) != target_values:
            break
        count += 1
    return count


def _values_match_base(
    base_values: dict[str, object],
    current_values: dict[str, object],
) -> bool:
    for key, value in current_values.items():
        if base_values.get(key) != value:
            return False
    return True


def _normalize_override_values(
    values: dict[str, object],
    *,
    configured_symbols: set[str],
    allowed_hours: list[int],
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in values.items():
        if key in {
            "allowed_symbols",
            "blocked_symbols",
            "blocked_long_symbols",
            "blocked_short_symbols",
        } and isinstance(value, list):
            normalized[key] = _normalize_symbol_list(value, configured_symbols)
        elif key == "allowed_entry_hours" and isinstance(value, list):
            normalized[key] = _normalize_hour_list(value, allowed_hours)
        else:
            normalized[key] = value
    return normalized


def _normalize_symbol_list(values: list[object], configured_symbols: set[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        if configured_symbols and symbol not in configured_symbols:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _normalize_hour_list(values: list[object], allowed_hours: list[int]) -> list[int]:
    allowed = {int(value) for value in allowed_hours}
    normalized: list[int] = []
    seen: set[int] = set()
    for raw in values:
        hour = int(raw)
        if hour in seen:
            continue
        if hour < 0 or hour > 23:
            continue
        if allowed and hour not in allowed:
            continue
        seen.add(hour)
        normalized.append(hour)
    return normalized


def _source_guardrail_reason(
    *,
    source_name: str,
    selected_values: dict[str, object],
    configured_symbols: set[str],
    base_hours: list[int],
) -> str:
    if source_name == "entry-symbols":
        configured_count = len(configured_symbols)
        if configured_count < _MIN_RUNTIME_SYMBOLS_FOR_ENTRY_RESTRICTIONS:
            return (
                "entry symbol restrictions are disabled because the runtime universe is too small "
                f"({configured_count} symbols)"
            )
        restricted_symbols = set()
        for key in ("blocked_symbols", "blocked_long_symbols", "blocked_short_symbols"):
            restricted_symbols.update(selected_values.get(key, []))
        if restricted_symbols:
            restricted_share = len(restricted_symbols) / configured_count
            if restricted_share > _MAX_ENTRY_RESTRICTED_SYMBOL_SHARE:
                return (
                    "entry symbol restrictions would over-concentrate the runtime universe "
                    f"({len(restricted_symbols)}/{configured_count} symbols)"
                )
        return ""

    if source_name == "entry-schedule":
        candidate_hours = selected_values.get("allowed_entry_hours", [])
        if not isinstance(candidate_hours, list):
            return ""
        minimum_hours = (
            min(
                len(base_hours),
                max(
                    _MIN_ACTIVE_ENTRY_HOURS,
                    math.ceil(len(base_hours) * _MIN_ACTIVE_ENTRY_HOUR_COVERAGE_RATIO),
                ),
            )
            if base_hours
            else _MIN_ACTIVE_ENTRY_HOURS
        )
        if base_hours and len(candidate_hours) < minimum_hours:
            return (
                "entry schedule would over-narrow the runtime window "
                f"({len(candidate_hours)}/{len(base_hours)} hours)"
            )
        return ""

    if source_name == "universe-selection":
        allowed_symbols = selected_values.get("allowed_symbols", [])
        if not isinstance(allowed_symbols, list):
            return ""
        configured_count = len(configured_symbols)
        minimum_symbols = (
            min(
                configured_count,
                max(
                    _MIN_ALLOWED_RUNTIME_SYMBOLS,
                    math.ceil(configured_count * _MIN_ALLOWED_RUNTIME_COVERAGE_RATIO),
                ),
            )
            if configured_count
            else _MIN_ALLOWED_RUNTIME_SYMBOLS
        )
        if configured_count and len(allowed_symbols) < minimum_symbols:
            return (
                "runtime universe override would over-concentrate the paper bot "
                f"({len(allowed_symbols)}/{configured_count} symbols)"
            )
    return ""


def _strategy_current_values(payload: dict[str, object]) -> dict[str, object]:
    source = payload.get("current_strategy", {})
    return {
        key: source[key]
        for key in (
            "style",
            "fast_window",
            "slow_window",
            "require_breakout",
            "opening_range_bars",
            "rel_volume_threshold",
            "breakeven_trigger_pct",
            "trailing_profit_trigger_rub",
            "trailing_profit_lock_ratio",
            "min_trend_strength",
            "adx_min",
            "rsi_long_max",
            "rsi_short_min",
        )
        if key in source
    }


def _strategy_candidate_values(payload: dict[str, object]) -> dict[str, object]:
    source = payload.get("candidate_strategy", {})
    return {
        key: source[key]
        for key in (
            "style",
            "fast_window",
            "slow_window",
            "require_breakout",
            "opening_range_bars",
            "rel_volume_threshold",
            "breakeven_trigger_pct",
            "trailing_profit_trigger_rub",
            "trailing_profit_lock_ratio",
            "min_trend_strength",
            "adx_min",
            "rsi_long_max",
            "rsi_short_min",
        )
        if key in source
    }


def _exit_current_values(payload: dict[str, object]) -> dict[str, object]:
    source = payload.get("current_exit_settings", {})
    return {
        key: source[key]
        for key in (
            "atr_stop_multiple",
            "reward_to_risk",
            "breakeven_trigger_pct",
            "trailing_profit_trigger_rub",
            "trailing_profit_lock_ratio",
        )
        if key in source
    }


def _exit_candidate_values(payload: dict[str, object]) -> dict[str, object]:
    source = payload.get("candidate_exit_settings", {})
    return {
        key: source[key]
        for key in (
            "atr_stop_multiple",
            "reward_to_risk",
            "breakeven_trigger_pct",
            "trailing_profit_trigger_rub",
            "trailing_profit_lock_ratio",
        )
        if key in source
    }


def _entry_schedule_current_values(payload: dict[str, object]) -> dict[str, object]:
    return {"allowed_entry_hours": [int(value) for value in payload.get("current_hours", [])]}


def _entry_schedule_candidate_values(payload: dict[str, object]) -> dict[str, object]:
    return {"allowed_entry_hours": [int(value) for value in payload.get("proposed_hours", [])]}


def _entry_quality_current_values(payload: dict[str, object]) -> dict[str, object]:
    return {"min_signal_strength": float(payload.get("current_min_signal_strength", 0.0))}


def _entry_quality_candidate_values(payload: dict[str, object]) -> dict[str, object]:
    return {"min_signal_strength": float(payload.get("recommended_min_signal_strength", 0.0))}


def _entry_symbols_current_values(payload: dict[str, object]) -> dict[str, object]:
    return {
        "blocked_symbols": [
            str(value).strip().upper() for value in payload.get("current_blocked_symbols", [])
        ],
        "blocked_long_symbols": [
            str(value).strip().upper() for value in payload.get("current_blocked_long_symbols", [])
        ],
        "blocked_short_symbols": [
            str(value).strip().upper() for value in payload.get("current_blocked_short_symbols", [])
        ],
    }


def _universe_selection_current_values(payload: dict[str, object]) -> dict[str, object]:
    return {
        "allowed_symbols": [
            str(value).strip().upper() for value in payload.get("current_allowed_symbols", [])
        ],
    }


def _universe_selection_candidate_values(payload: dict[str, object]) -> dict[str, object]:
    return {
        "allowed_symbols": [
            str(value).strip().upper() for value in payload.get("proposed_allowed_symbols", [])
        ],
    }


def _entry_symbols_candidate_values(payload: dict[str, object]) -> dict[str, object]:
    return {
        "blocked_symbols": [
            str(value).strip().upper() for value in payload.get("proposed_blocked_symbols", [])
        ],
        "blocked_long_symbols": [
            str(value).strip().upper() for value in payload.get("proposed_blocked_long_symbols", [])
        ],
        "blocked_short_symbols": [
            str(value).strip().upper() for value in payload.get("proposed_blocked_short_symbols", [])
        ],
    }


def _strategy_values(payload: dict[str, object]) -> dict[str, object]:
    source = payload.get("candidate_strategy", {}) if payload.get("changed") else payload.get("current_strategy", {})
    return {
        key: source[key]
        for key in (
            "style",
            "fast_window",
            "slow_window",
            "require_breakout",
            "opening_range_bars",
            "rel_volume_threshold",
            "breakeven_trigger_pct",
            "trailing_profit_trigger_rub",
            "trailing_profit_lock_ratio",
            "min_trend_strength",
            "adx_min",
            "rsi_long_max",
            "rsi_short_min",
        )
        if key in source
    }


def _exit_values(payload: dict[str, object]) -> dict[str, object]:
    source = payload.get("candidate_exit_settings", {}) if payload.get("changed") else payload.get("current_exit_settings", {})
    return {
        key: source[key]
        for key in (
            "atr_stop_multiple",
            "reward_to_risk",
            "breakeven_trigger_pct",
            "trailing_profit_trigger_rub",
            "trailing_profit_lock_ratio",
        )
        if key in source
    }


def _entry_schedule_values(payload: dict[str, object]) -> dict[str, object]:
    key = "proposed_hours" if payload.get("changed") else "current_hours"
    return {"allowed_entry_hours": [int(value) for value in payload.get(key, [])]}


def _entry_quality_values(payload: dict[str, object]) -> dict[str, object]:
    key = "recommended_min_signal_strength" if payload.get("changed") else "current_min_signal_strength"
    return {"min_signal_strength": float(payload.get(key, 0.0))}


def _apply_strategy_overrides(base_text: str, overrides: dict[str, object]) -> str:
    if not overrides:
        return base_text

    lines = base_text.splitlines()
    strategy_start = None
    strategy_end = len(lines)
    for index, line in enumerate(lines):
        if line.strip() == "[strategy]":
            strategy_start = index
            continue
        if strategy_start is not None and index > strategy_start and _SECTION_PATTERN.match(line.strip()):
            strategy_end = index
            break

    if strategy_start is None:
        raise ValueError("strategy section not found in config")

    line_indexes: dict[str, int] = {}
    for index in range(strategy_start + 1, strategy_end):
        match = _KEY_PATTERN.match(lines[index])
        if match:
            line_indexes[match.group(1)] = index

    for key in _STRATEGY_OVERRIDE_ORDER:
        if key not in overrides:
            continue
        rendered_line = f"{key} = {_render_toml_value(overrides[key])}"
        if key in line_indexes:
            lines[line_indexes[key]] = rendered_line
            continue
        lines.insert(strategy_end, rendered_line)
        line_indexes[key] = strategy_end
        strategy_end += 1

    return "\n".join(lines) + "\n"


def _render_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_render_toml_value(item) for item in value) + "]"
    return str(value)
