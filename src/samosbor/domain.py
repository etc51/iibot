from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Any


class InstrumentType(str, Enum):
    STOCK = "stock"
    FUTURE = "future"


class TradeMode(str, Enum):
    LOCAL_PAPER = "local-paper"
    TBANK_SANDBOX = "tbank-sandbox"
    LIVE = "live"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    STOP_LOSS = "stop-loss"
    BREAKEVEN_STOP = "breakeven-stop"
    PROFIT_PROTECT_STOP = "profit-protect-stop"
    TAKE_PROFIT = "take-profit"
    SIGNAL_FLIP = "signal-flip"
    SESSION_FLAT = "session-flat"
    RISK_HALT = "risk-halt"
    END_OF_TEST = "end-of-test"
    MANUAL = "manual"
    SHORT_ONLY_POLICY_FLATTEN_LONG = "short_only_policy_flatten_long"
    SHORT_ONLY_EARLY_LOSS_GUARD = "short_only_early_loss_guard"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    instrument_type: InstrumentType
    figi: str = ""
    uid: str = ""
    class_code: str = ""
    lot_size: int = 1
    tick_size: float = 0.01
    currency: str = "rub"
    initial_margin_buy: float = 0.0
    initial_margin_sell: float = 0.0
    tick_value: float = 0.0

    @property
    def instrument_id(self) -> str:
        return self.uid or self.figi or self.symbol


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    instrument: Instrument
    direction: SignalDirection
    strength: float
    entry_price: float
    stop_price: float
    take_profit: float
    reason: str
    context_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    quantity_lots: int = 0
    risk_budget_rub: float = 0.0
    estimated_notional_rub: float = 0.0


@dataclass
class Position:
    instrument: Instrument
    direction: SignalDirection
    quantity_lots: int
    entry_price: float
    entry_commission: float
    margin_requirement: float
    current_price: float
    stop_price: float
    take_profit: float
    opened_at: datetime
    updated_at: datetime
    signal_strength: float = 0.0
    entry_reason: str = ""
    entry_context_score: float = 0.0
    entry_metadata: dict[str, Any] = field(default_factory=dict)
    initial_stop_price: float = 0.0
    initial_take_profit: float = 0.0
    runner_active: bool = False
    runner_activated_at: datetime | None = None
    runner_activation_price: float = 0.0
    runner_extreme_price: float = 0.0
    mfe_price: float = 0.0
    mfe_pnl: float = 0.0
    mae_price: float = 0.0
    mae_pnl: float = 0.0

    @property
    def quantity_units(self) -> int:
        return self.quantity_lots * self.instrument.lot_size

    @property
    def signed_units(self) -> int:
        sign = 1 if self.direction == SignalDirection.LONG else -1
        return sign * self.quantity_units

    def notional(self, price: float | None = None) -> float:
        mark = self.current_price if price is None else price
        return abs(self.quantity_units * mark)

    def market_value(self, price: float | None = None) -> float:
        mark = self.current_price if price is None else price
        if self.instrument.instrument_type == InstrumentType.FUTURE:
            return self.unrealized_pnl(mark)
        return self.signed_units * mark

    def unrealized_pnl(self, price: float | None = None) -> float:
        mark = self.current_price if price is None else price
        if self.direction == SignalDirection.LONG:
            return (mark - self.entry_price) * self.quantity_units
        return (self.entry_price - mark) * self.quantity_units

    def record_price_extremes(
        self,
        *,
        price: float | None = None,
        low_price: float | None = None,
        high_price: float | None = None,
    ) -> None:
        prices: list[float] = []
        for candidate in (price, low_price, high_price):
            normalized = _positive_finite_float(candidate)
            if normalized is not None:
                prices.append(normalized)
        if not prices:
            return

        if self.mfe_price <= 0:
            self.mfe_price = self.entry_price
        if self.mae_price <= 0:
            self.mae_price = self.entry_price

        for candidate in prices:
            pnl = self.unrealized_pnl(candidate)
            if pnl > self.mfe_pnl:
                self.mfe_pnl = pnl
                self.mfe_price = candidate
            if pnl < self.mae_pnl:
                self.mae_pnl = pnl
                self.mae_price = candidate

    def refresh_excursion_pnl(self) -> None:
        if self.mfe_price <= 0:
            self.mfe_price = self.entry_price
        if self.mae_price <= 0:
            self.mae_price = self.entry_price
        self.mfe_pnl = max(0.0, self.unrealized_pnl(self.mfe_price))
        self.mae_pnl = min(0.0, self.unrealized_pnl(self.mae_price))

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": {
                "symbol": self.instrument.symbol,
                "instrument_type": self.instrument.instrument_type.value,
                "figi": self.instrument.figi,
                "uid": self.instrument.uid,
                "class_code": self.instrument.class_code,
                "lot_size": self.instrument.lot_size,
                "tick_size": self.instrument.tick_size,
                "currency": self.instrument.currency,
                "initial_margin_buy": self.instrument.initial_margin_buy,
                "initial_margin_sell": self.instrument.initial_margin_sell,
                "tick_value": self.instrument.tick_value,
            },
            "direction": self.direction.value,
            "quantity_lots": self.quantity_lots,
            "entry_price": self.entry_price,
            "entry_commission": self.entry_commission,
            "margin_requirement": self.margin_requirement,
            "current_price": self.current_price,
            "stop_price": self.stop_price,
            "take_profit": self.take_profit,
            "opened_at": self.opened_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "signal_strength": self.signal_strength,
            "entry_reason": self.entry_reason,
            "entry_context_score": self.entry_context_score,
            "entry_metadata": self.entry_metadata,
            "initial_stop_price": self.initial_stop_price,
            "initial_take_profit": self.initial_take_profit,
            "runner_active": self.runner_active,
            "runner_activated_at": (
                self.runner_activated_at.isoformat() if self.runner_activated_at is not None else None
            ),
            "runner_activation_price": self.runner_activation_price,
            "runner_extreme_price": self.runner_extreme_price,
            "mfe_price": self.mfe_price,
            "mfe_pnl": self.mfe_pnl,
            "mae_price": self.mae_price,
            "mae_pnl": self.mae_pnl,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Position":
        instrument_payload = payload["instrument"]
        instrument = Instrument(
            symbol=instrument_payload["symbol"],
            instrument_type=InstrumentType(instrument_payload["instrument_type"]),
            figi=instrument_payload.get("figi", ""),
            uid=instrument_payload.get("uid", ""),
            class_code=instrument_payload.get("class_code", ""),
            lot_size=int(instrument_payload.get("lot_size", 1)),
            tick_size=float(instrument_payload.get("tick_size", 0.01)),
            currency=instrument_payload.get("currency", "rub"),
            initial_margin_buy=float(instrument_payload.get("initial_margin_buy", 0.0)),
            initial_margin_sell=float(instrument_payload.get("initial_margin_sell", 0.0)),
            tick_value=float(instrument_payload.get("tick_value", 0.0)),
        )
        return cls(
            instrument=instrument,
            direction=SignalDirection(payload["direction"]),
            quantity_lots=int(payload["quantity_lots"]),
            entry_price=float(payload["entry_price"]),
            entry_commission=float(payload.get("entry_commission", 0.0)),
            margin_requirement=float(payload.get("margin_requirement", 0.0)),
            current_price=float(payload["current_price"]),
            stop_price=float(payload["stop_price"]),
            take_profit=float(payload["take_profit"]),
            opened_at=datetime.fromisoformat(payload["opened_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            signal_strength=float(payload.get("signal_strength", 0.0)),
            entry_reason=str(payload.get("entry_reason", "")),
            entry_context_score=float(payload.get("entry_context_score", 0.0)),
            entry_metadata=dict(payload.get("entry_metadata", {})),
            initial_stop_price=float(payload.get("initial_stop_price", payload["stop_price"])),
            initial_take_profit=float(payload.get("initial_take_profit", payload["take_profit"])),
            runner_active=bool(payload.get("runner_active", False)),
            runner_activated_at=(
                datetime.fromisoformat(payload["runner_activated_at"])
                if payload.get("runner_activated_at")
                else None
            ),
            runner_activation_price=float(payload.get("runner_activation_price", 0.0)),
            runner_extreme_price=float(payload.get("runner_extreme_price", 0.0)),
            mfe_price=float(payload.get("mfe_price", payload["entry_price"])),
            mfe_pnl=float(payload.get("mfe_pnl", 0.0)),
            mae_price=float(payload.get("mae_price", payload["entry_price"])),
            mae_pnl=float(payload.get("mae_pnl", 0.0)),
        )


@dataclass
class PortfolioState:
    cash: float
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    trading_halted: bool = False
    positions: dict[str, Position] = field(default_factory=dict)

    def equity(self, marks: dict[str, float]) -> float:
        position_value = 0.0
        for symbol, position in self.positions.items():
            position_value += position.market_value(marks.get(symbol, position.current_price))
        return self.cash + position_value

    def gross_exposure(self, marks: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            total += position.notional(marks.get(symbol, position.current_price))
        return total

    def margin_reserved(self) -> float:
        return sum(position.margin_requirement for position in self.positions.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "peak_equity": self.peak_equity,
            "trading_halted": self.trading_halted,
            "positions": {
                symbol: position.to_dict() for symbol, position in self.positions.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioState":
        positions = {}
        for symbol, position_payload in payload.get("positions", {}).items():
            position = Position.from_dict(position_payload)
            if position.quantity_lots <= 0:
                continue
            positions[symbol] = position
        return cls(
            cash=float(payload.get("cash", 0.0)),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            peak_equity=float(payload.get("peak_equity", 0.0)),
            trading_halted=bool(payload.get("trading_halted", False)),
            positions=positions,
        )


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    direction: SignalDirection
    quantity_lots: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    reason: str
    signal_strength: float = 0.0
    entry_reason: str = ""
    entry_context_score: float = 0.0
    entry_metadata: dict[str, Any] = field(default_factory=dict)
    initial_stop_price: float = 0.0
    initial_take_profit: float = 0.0


def _positive_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    cash: float
    gross_exposure: float


@dataclass
class BacktestResult:
    portfolio: PortfolioState
    trades: list[TradeRecord]
    equity_curve: list[EquityPoint]
    events: list[dict[str, Any]]
