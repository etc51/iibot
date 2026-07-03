from __future__ import annotations

import tomllib

from .config import ResearchSection, StrategySection

_DEFAULT_RESEARCH = ResearchSection()
_DEFAULT_STRATEGY = StrategySection()

_AUTONOMY_RESEARCH_ARRAY_LIMITS: dict[str, tuple[str, int]] = {
    "strategy_styles": ("style", 4),
    "fast_windows": ("fast_window", 2),
    "slow_windows": ("slow_window", 2),
    "require_breakout_values": ("require_breakout", 2),
    "opening_range_bars_values": ("opening_range_bars", 2),
    "rel_volume_threshold_values": ("rel_volume_threshold", 2),
    "atr_stop_multipliers": ("atr_stop_multiple", 1),
    "reward_to_risk_values": ("reward_to_risk", 2),
    "breakeven_trigger_pct_values": ("breakeven_trigger_pct", 2),
    "trailing_profit_trigger_rub_values": ("trailing_profit_trigger_rub", 2),
    "trailing_profit_lock_ratio_values": ("trailing_profit_lock_ratio", 2),
    "trend_strength_values": ("min_trend_strength", 2),
    "adx_min_values": ("adx_min", 2),
    "rsi_long_max_values": ("rsi_long_max", 2),
    "rsi_short_min_values": ("rsi_short_min", 2),
}
DAILY_AUTONOMY_RESEARCH_ARRAY_LIMITS: dict[str, tuple[str, int]] = {
    "strategy_styles": ("style", 4),
    "fast_windows": ("fast_window", 2),
    "slow_windows": ("slow_window", 2),
    "require_breakout_values": ("require_breakout", 1),
    "opening_range_bars_values": ("opening_range_bars", 1),
    "rel_volume_threshold_values": ("rel_volume_threshold", 1),
    "atr_stop_multipliers": ("atr_stop_multiple", 1),
    "reward_to_risk_values": ("reward_to_risk", 2),
    "breakeven_trigger_pct_values": ("breakeven_trigger_pct", 1),
    "trailing_profit_trigger_rub_values": ("trailing_profit_trigger_rub", 1),
    "trailing_profit_lock_ratio_values": ("trailing_profit_lock_ratio", 1),
    "trend_strength_values": ("min_trend_strength", 1),
    "adx_min_values": ("adx_min", 1),
    "rsi_long_max_values": ("rsi_long_max", 1),
    "rsi_short_min_values": ("rsi_short_min", 1),
}
_AUTONOMY_TOP_N_CAP = 8


def build_offline_autonomy_config_text(
    source_text: str,
    *,
    parquet_dir_path: str,
    research_array_limits: dict[str, tuple[str, int]] | None = None,
) -> str:
    source_text = source_text.removeprefix("\ufeff")
    autonomy_research_overrides = _build_autonomy_research_overrides(
        source_text,
        research_array_limits=research_array_limits or _AUTONOMY_RESEARCH_ARRAY_LIMITS,
    )
    lines = source_text.splitlines()
    output: list[str] = []
    in_data_section = False
    in_research_section = False
    saw_data_section = False
    saw_research_section = False
    wrote_source = False
    wrote_parquet_dir = False
    wrote_research_keys: set[str] = set()

    def flush_data_defaults() -> None:
        nonlocal wrote_source, wrote_parquet_dir
        if not wrote_source:
            output.append('source = "parquet-directory"')
            wrote_source = True
        if not wrote_parquet_dir:
            output.append(f'parquet_dir_path = "{parquet_dir_path}"')
            wrote_parquet_dir = True

    def flush_research_overrides() -> None:
        for key, value in autonomy_research_overrides.items():
            if key in wrote_research_keys:
                continue
            output.append(f"{key} = {_render_toml_value(value)}")
            wrote_research_keys.add(key)

    for line in lines:
        stripped = line.strip()
        is_section_header = stripped.startswith("[") and stripped.endswith("]")
        if is_section_header:
            if in_data_section:
                flush_data_defaults()
            if in_research_section:
                flush_research_overrides()
            in_data_section = stripped == "[data]"
            in_research_section = stripped == "[research]"
            if in_data_section:
                saw_data_section = True
                wrote_source = False
                wrote_parquet_dir = False
            if in_research_section:
                saw_research_section = True
                wrote_research_keys = set()
            output.append(line)
            continue

        if in_data_section:
            if stripped.startswith("source ="):
                output.append('source = "parquet-directory"')
                wrote_source = True
                continue
            if stripped.startswith("parquet_dir_path ="):
                output.append(f'parquet_dir_path = "{parquet_dir_path}"')
                wrote_parquet_dir = True
                continue
            if stripped.startswith("csv_path =") or stripped.startswith("local_data_pack_path ="):
                continue

        if in_research_section:
            key = _toml_key_for_line(stripped)
            if key in autonomy_research_overrides:
                output.append(f"{key} = {_render_toml_value(autonomy_research_overrides[key])}")
                wrote_research_keys.add(key)
                continue

        output.append(line)

    if in_data_section:
        flush_data_defaults()
    if in_research_section:
        flush_research_overrides()

    if not saw_data_section:
        raise ValueError("Config text is missing a [data] section.")
    if not saw_research_section:
        if output and output[-1].strip():
            output.append("")
        output.append("[research]")
        for key, value in autonomy_research_overrides.items():
            output.append(f"{key} = {_render_toml_value(value)}")

    return "\n".join(output).rstrip() + "\n"


def _build_autonomy_research_overrides(
    source_text: str,
    *,
    research_array_limits: dict[str, tuple[str, int]],
) -> dict[str, object]:
    payload = tomllib.loads(source_text)
    strategy = payload.get("strategy", {})
    research = payload.get("research", {})
    data = payload.get("data", {})
    data_source = str(data.get("source", "tbank")).strip().lower()
    instrument_count = len(data.get("instruments", []))
    source_subset_min = int(research.get("subset_min_size", _DEFAULT_RESEARCH.subset_min_size))
    source_subset_max = int(research.get("subset_max_size", _DEFAULT_RESEARCH.subset_max_size))
    if instrument_count > 0 and not _preserve_source_subset_search(data_source):
        desired_subset_min = instrument_count
        desired_subset_max = instrument_count
    else:
        if instrument_count > 0:
            desired_subset_max = min(max(1, source_subset_max), instrument_count)
        else:
            desired_subset_max = max(1, source_subset_max)
        desired_subset_min = min(max(1, source_subset_min), desired_subset_max)

    overrides: dict[str, object] = {}
    for research_key, (strategy_field, limit) in research_array_limits.items():
        preferred_value = strategy.get(strategy_field, getattr(_DEFAULT_STRATEGY, strategy_field))
        overrides[research_key] = _select_autonomy_values(
            research.get(research_key),
            preferred=preferred_value,
            limit=limit,
        )

    overrides["subset_min_size"] = desired_subset_min
    overrides["subset_max_size"] = desired_subset_max
    overrides["top_n"] = max(1, min(int(research.get("top_n", _DEFAULT_RESEARCH.top_n)), _AUTONOMY_TOP_N_CAP))
    return overrides


def _select_autonomy_values(source_values: object, *, preferred: object, limit: int) -> list[object]:
    if isinstance(source_values, list) and source_values:
        candidates = list(source_values)
    elif source_values is None:
        candidates = []
    else:
        candidates = [source_values]

    ordered: list[object] = []
    if preferred is not None:
        ordered.append(preferred)
    ordered.extend(candidates)

    selected: list[object] = []
    seen: set[tuple[str, object]] = set()
    for value in ordered:
        identity = _value_identity(value)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(value)
        if len(selected) >= limit:
            break
    return selected


def _preserve_source_subset_search(data_source: str) -> bool:
    return data_source in {"csv", "parquet-directory", "moex-data-pack"}


def _toml_key_for_line(line: str) -> str | None:
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, _separator, _value = line.partition("=")
    return key.strip()


def _render_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_render_toml_value(item) for item in value) + "]"
    return repr(value)


def _value_identity(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("str", value.strip().lower())
    return (type(value).__name__, value)
