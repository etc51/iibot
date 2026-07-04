from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .domain import Instrument, InstrumentType, TradeMode


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class AppSection:
    name: str = "samosbor"
    timezone: str = "Europe/Moscow"


@dataclass(frozen=True)
class TBankSection:
    token_env: str = "TBANK_INVEST_TOKEN"
    token_file: str = ""
    sandbox_token_env: str = "TBANK_SANDBOX_TOKEN"
    sandbox_token_file: str = ""
    account_id_env: str = "TBANK_ACCOUNT_ID"
    account_name: str = "Фьючерсы"
    app_name: str = "samosbor"
    ssl_verify_env: str = "SSL_TBANK_VERIFY"


@dataclass(frozen=True)
class DataSection:
    source: str = "tbank"
    timeframe: str = "hour"
    history_days: int = 120
    tbank_candle_source: str = "include-weekend"
    csv_path: str = ""
    parquet_dir_path: str = ""
    local_data_pack_path: str = ""
    instruments: list[Instrument] = field(default_factory=list)


@dataclass(frozen=True)
class StrategySection:
    style: str = "sma_breakout"
    fast_window: int = 20
    slow_window: int = 50
    atr_window: int = 14
    volume_window: int = 20
    breakout_window: int = 20
    opening_range_bars: int = 2
    rel_volume_threshold: float = 1.1
    require_breakout: bool = True
    atr_stop_multiple: float = 2.0
    reward_to_risk: float = 2.0
    breakeven_trigger_pct: float = 0.0
    trailing_profit_trigger_rub: float = 0.0
    trailing_profit_lock_ratio: float = 0.0
    take_profit_activates_runner: bool = False
    runner_breakeven_buffer_bps: float = 10.0
    runner_trailing_atr_multiple: float = 1.3
    runner_profit_lock_ratio: float = 0.35
    min_signal_strength: float = 0.0
    min_trend_strength: float = 0.004
    min_liquidity_rub: float = 50_000_000
    allow_shorts: bool = True
    adx_window: int = 14
    adx_min: float = 20.0
    rsi_window: int = 14
    rsi_long_min: float = 50.0
    rsi_long_max: float = 75.0
    rsi_short_min: float = 25.0
    rsi_short_max: float = 50.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    order_book_depth: int = 10
    require_order_book: bool = False
    max_entry_spread_bps: float = 0.0
    min_entry_liquidity_cover: float = 0.0
    min_entry_book_imbalance: float = -1.0
    use_market_context: bool = False
    market_context_fast_window: int = 20
    market_context_slow_window: int = 50
    market_context_return_window: int = 4
    market_context_min_symbols: int = 8
    market_context_max_score: float = 0.25
    market_context_block_threshold: float = 0.15
    schedule_timezone: str = "Europe/Moscow"
    allowed_entry_hours: list[int] = field(default_factory=list)
    allowed_entry_weekdays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    allowed_symbols: list[str] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    blocked_long_symbols: list[str] = field(default_factory=list)
    blocked_short_symbols: list[str] = field(default_factory=list)
    forced_flat_hours: list[int] = field(default_factory=list)
    forced_flat_weekdays: list[int] = field(default_factory=list)
    entry_confirmation_timeframe: str = ""
    entry_confirmation_min_bars: int = 3
    entry_confirmation_max_adverse_ret: float = 0.005


@dataclass(frozen=True)
class RiskSection:
    max_risk_per_trade: float = 0.01
    max_gross_exposure: float = 1.25
    max_drawdown: float = 0.12
    cash_reserve_ratio: float = 0.15
    max_positions: int = 6
    max_position_exposure_ratio: float = 1.0
    kelly_lookback_trades: int = 20
    min_trades_for_kelly: int = 8


@dataclass(frozen=True)
class ExecutionSection:
    mode: TradeMode = TradeMode.LOCAL_PAPER
    slippage_bps: float = 5.0
    commission_bps: float = 4.0
    state_path: str = "state/paper_state.json"
    allow_live_trading: bool = False


@dataclass(frozen=True)
class BacktestSection:
    initial_cash: float = 1_000_000.0
    warmup_bars: int = 60


@dataclass(frozen=True)
class ReportingSection:
    output_dir: str = "runs"
    write_csv: bool = True


@dataclass(frozen=True)
class ResearchSection:
    strategy_styles: list[str] = field(default_factory=lambda: ["sma_breakout"])
    fast_windows: list[int] = field(default_factory=lambda: [10, 15, 20])
    slow_windows: list[int] = field(default_factory=lambda: [30, 40, 50])
    require_breakout_values: list[bool] = field(default_factory=lambda: [True])
    opening_range_bars_values: list[int] = field(default_factory=lambda: [2, 3])
    rel_volume_threshold_values: list[float] = field(default_factory=lambda: [1.0, 1.15, 1.3])
    atr_stop_multipliers: list[float] = field(default_factory=lambda: [1.5, 2.0])
    reward_to_risk_values: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    breakeven_trigger_pct_values: list[float] = field(default_factory=lambda: [0.0])
    trailing_profit_trigger_rub_values: list[float] = field(default_factory=lambda: [0.0])
    trailing_profit_lock_ratio_values: list[float] = field(default_factory=lambda: [0.0])
    trend_strength_values: list[float] = field(default_factory=lambda: [0.004, 0.006])
    adx_min_values: list[float] = field(default_factory=lambda: [20.0])
    rsi_long_max_values: list[float] = field(default_factory=lambda: [70.0, 75.0])
    rsi_short_min_values: list[float] = field(default_factory=lambda: [25.0, 30.0])
    subset_min_size: int = 1
    subset_max_size: int = 3
    top_n: int = 10
    min_trades: int = 6
    walk_forward_train_months: int = 6
    walk_forward_test_months: int = 1
    walk_forward_step_months: int = 1
    monte_carlo_iterations: int = 1000
    monte_carlo_horizon_months: int = 12
    trading_days_per_month: int = 20
    target_daily_profit_rub: float = 0.0
    target_monthly_return_pct: float = 5.0
    target_monthly_profit_rub: float = 0.0
    random_seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    app: AppSection
    tbank: TBankSection
    data: DataSection
    strategy: StrategySection
    risk: RiskSection
    execution: ExecutionSection
    backtest: BacktestSection
    reporting: ReportingSection
    research: ResearchSection

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def runtime_profile_name(self) -> str:
        stem = Path(self.execution.state_path).stem.strip()
        if not stem:
            return "paper"
        if stem.endswith("_state"):
            stem = stem[: -len("_state")]
        elif stem.endswith("-state"):
            stem = stem[: -len("-state")]
        return stem or "paper"

    def autotune_dir(self) -> Path:
        return self.resolve_path(self.configured_autotune_dir())

    def configured_autotune_dir(self) -> str:
        return str(Path(self.reporting.output_dir) / "autotune" / self.runtime_profile_name())


def _parse_instrument(payload: dict[str, Any]) -> Instrument:
    return Instrument(
        symbol=payload["symbol"].strip().upper(),
        instrument_type=InstrumentType(payload["instrument_type"]),
        figi=payload.get("figi", ""),
        uid=payload.get("uid", ""),
        class_code=payload.get("class_code", ""),
        lot_size=int(payload.get("lot_size", 1)),
        tick_size=float(payload.get("tick_size", 0.01)),
        currency=payload.get("currency", "rub"),
        initial_margin_buy=float(payload.get("initial_margin_buy", 0.0)),
        initial_margin_sell=float(payload.get("initial_margin_sell", 0.0)),
        tick_value=float(payload.get("tick_value", 0.0)),
    )


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).resolve()
    root_dir = config_path.parent.parent
    load_dotenv(root_dir / ".env")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))

    app = AppSection(**raw.get("app", {}))
    tbank = TBankSection(**raw.get("tbank", {}))

    data_raw = raw.get("data", {})
    instruments = [_parse_instrument(item) for item in data_raw.get("instruments", [])]
    data = DataSection(
        source=data_raw.get("source", "tbank"),
        timeframe=data_raw.get("timeframe", "hour"),
        history_days=int(data_raw.get("history_days", 120)),
        tbank_candle_source=str(data_raw.get("tbank_candle_source", "include-weekend")),
        csv_path=data_raw.get("csv_path", ""),
        parquet_dir_path=data_raw.get("parquet_dir_path", ""),
        local_data_pack_path=data_raw.get("local_data_pack_path", ""),
        instruments=instruments,
    )

    strategy = StrategySection(**raw.get("strategy", {}))
    risk = RiskSection(**raw.get("risk", {}))

    execution_raw = raw.get("execution", {})
    execution = ExecutionSection(
        mode=TradeMode(execution_raw.get("mode", TradeMode.LOCAL_PAPER.value)),
        slippage_bps=float(execution_raw.get("slippage_bps", 5.0)),
        commission_bps=float(execution_raw.get("commission_bps", 4.0)),
        state_path=execution_raw.get("state_path", "state/paper_state.json"),
        allow_live_trading=bool(execution_raw.get("allow_live_trading", False)),
    )

    backtest = BacktestSection(**raw.get("backtest", {}))
    reporting = ReportingSection(**raw.get("reporting", {}))
    research = ResearchSection(**raw.get("research", {}))

    return AppConfig(
        root_dir=root_dir,
        app=app,
        tbank=tbank,
        data=data,
        strategy=strategy,
        risk=risk,
        execution=execution,
        backtest=backtest,
        reporting=reporting,
        research=research,
    )


def read_secret_from_env_or_file(env_name: str, file_path: str, *, label: str) -> str:
    value = os.environ.get(env_name, "")
    if value.strip():
        return _normalize_secret_value(value)

    if file_path:
        path = Path(os.path.expandvars(os.path.expanduser(file_path)))
        if not path.exists():
            raise RuntimeError(f"{label} file does not exist: {path}")
        return _normalize_secret_value(path.read_text(encoding="utf-8-sig"))

    raise RuntimeError(f"{label} is missing. Set {env_name} or configure a token file.")


def _normalize_secret_value(raw: str) -> str:
    text = raw.strip().strip("\"'")
    if not text:
        raise RuntimeError("Secret value is empty.")

    bearer_match = re.search(r"Bearer\s+([A-Za-z0-9._=-]{20,})", text, flags=re.IGNORECASE)
    if bearer_match:
        return bearer_match.group(1)

    for raw_line in text.splitlines():
        line = raw_line.strip().strip("\"'")
        if not line:
            continue
        if "=" in line:
            line = line.split("=", 1)[1].strip().strip("\"'")
        token_match = re.search(r"([A-Za-z0-9._=-]{20,})", line)
        if token_match:
            return token_match.group(1)

    if re.fullmatch(r"[A-Za-z0-9._=-]{20,}", text):
        return text
    raise RuntimeError("Secret file does not contain a valid-looking token.")
