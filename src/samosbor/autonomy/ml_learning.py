from __future__ import annotations

import math
import re
from statistics import median
from datetime import datetime
from zoneinfo import ZoneInfo

from ..domain import Candle, Signal, SignalDirection, TradeRecord


LOW_QUALITY_PROBABILITY_THRESHOLD = 0.40
MIN_LEARNING_SAMPLES = 50
LOW_QUALITY_TAG = "low-quality-learning"
NEGATIVE_EXPECTANCY_TAG = "negative-expectancy-learning"
COMMISSION_EDGE_TAG = "commission-edge-learning"
CONFIRMATION_AFTER_IMPULSE_TAG = "confirmation-after-impulse-learning"
SHORT_AFTER_EXHAUSTION_TAG = "short-after-exhaustion-learning"
LATE_REENTRY_TAG = "late-reentry-learning"

_CATEGORICAL_FEATURES = ["symbol", "direction"]
_NUMERIC_FEATURES = [
    "hour_msk",
    "weekday",
    "signal_strength",
    "entry_price",
    "stop_pct",
    "take_pct",
    "reward_risk",
    "context_score",
    "trend_strength",
    "turnover",
    "volatility",
    "entry_candle_range_pct",
    "entry_candle_body_pct",
    "entry_candle_ret1",
    "entry_candle_ret4",
    "entry_candle_range_ratio",
    "micro_spread_bps",
    "micro_liquidity_cover",
    "micro_side_imbalance",
    "slippage_bps",
    "commission_bps",
]
_FEATURES = _CATEGORICAL_FEATURES + _NUMERIC_FEATURES


def build_post_close_learning_analysis(trade: TradeRecord) -> dict[str, object]:
    outcome = "profit" if trade.net_pnl > 0 else "error" if trade.net_pnl < 0 else "flat"
    planned_risk = _planned_risk_rub(trade)
    realized_r = trade.net_pnl / planned_risk if planned_risk and planned_risk > 0 else None
    entry_learning = trade.entry_metadata.get("ml_learning", {})
    if not isinstance(entry_learning, dict):
        entry_learning = {}

    ml_available = bool(entry_learning.get("available"))
    ml_probability = _optional_float(entry_learning.get("probability_profit"))
    ml_expected_position = _optional_float(entry_learning.get("expected_pnl_position_rub"))
    learning_tags = entry_learning.get("learning_tags", [])
    if not isinstance(learning_tags, list):
        learning_tags = []

    ml_entry_bias = "unavailable"
    ml_verdict = "not_available"
    if ml_available:
        warns_risk = (
            LOW_QUALITY_TAG in learning_tags
            or NEGATIVE_EXPECTANCY_TAG in learning_tags
            or COMMISSION_EDGE_TAG in learning_tags
            or (ml_probability is not None and ml_probability < LOW_QUALITY_PROBABILITY_THRESHOLD)
            or (ml_expected_position is not None and ml_expected_position < 0)
        )
        predicts_profit = (
            ml_probability is not None
            and ml_probability >= 0.5
            and (ml_expected_position is None or ml_expected_position >= 0)
        )
        if predicts_profit:
            ml_entry_bias = "positive"
        elif warns_risk:
            ml_entry_bias = "risk"
        else:
            ml_entry_bias = "mixed"

        if outcome == "profit" and ml_entry_bias == "positive":
            ml_verdict = "ml_confirmed_win"
        elif outcome == "error" and ml_entry_bias == "risk":
            ml_verdict = "ml_warned_loss"
        elif outcome == "profit" and ml_entry_bias == "risk":
            ml_verdict = "ml_false_warning"
        elif outcome == "error" and ml_entry_bias == "positive":
            ml_verdict = "ml_missed_loss"
        else:
            ml_verdict = "ml_mixed_result"

    tags = [f"{outcome}-trade"]
    if trade.reason:
        tags.append(f"exit-{trade.reason}")
    if realized_r is not None and realized_r <= -0.9:
        tags.append("full-risk-error")
    if ml_verdict != "not_available":
        tags.append(ml_verdict)
    for learning_tag in (
        CONFIRMATION_AFTER_IMPULSE_TAG,
        SHORT_AFTER_EXHAUSTION_TAG,
        LATE_REENTRY_TAG,
        COMMISSION_EDGE_TAG,
    ):
        if learning_tag in learning_tags:
            tags.append(learning_tag)

    return {
        "available": True,
        "stage": "post_close",
        "outcome": outcome,
        "is_error": outcome == "error",
        "net_pnl_rub": round(float(trade.net_pnl), 2),
        "gross_pnl_rub": round(float(trade.gross_pnl), 2),
        "exit_reason": trade.reason,
        "planned_risk_rub": round(planned_risk, 2) if planned_risk is not None else None,
        "realized_r": round(realized_r, 3) if realized_r is not None else None,
        "ml_available": ml_available,
        "ml_probability_profit": round(ml_probability, 4) if ml_probability is not None else None,
        "ml_expected_pnl_position_rub": (
            round(ml_expected_position, 2) if ml_expected_position is not None else None
        ),
        "ml_entry_bias": ml_entry_bias,
        "ml_verdict": ml_verdict,
        "tags": tags,
        "summary": _post_close_summary(outcome, trade.reason, realized_r, ml_verdict),
    }


def build_entry_candle_context(candles: list[Candle], direction: str) -> dict[str, object]:
    if not candles:
        return {"available": False, "reason": "no candles"}

    latest = candles[-1]
    previous = candles[-2] if len(candles) >= 2 else latest
    previous_4 = candles[-5] if len(candles) >= 5 else candles[0]
    close = latest.close
    if close <= 0:
        return {"available": False, "reason": "bad close"}

    normalized_direction = direction.strip().lower()
    range_pct = (latest.high - latest.low) / close
    body_pct = abs(latest.close - latest.open) / close
    ret1 = (latest.close / previous.close - 1.0) if previous.close > 0 else 0.0
    ret4 = (latest.close / previous_4.close - 1.0) if previous_4.close > 0 else 0.0
    close_position = (
        (latest.close - latest.low) / (latest.high - latest.low)
        if latest.high > latest.low
        else 0.5
    )
    if normalized_direction == "short":
        direction_confirmed_by_close = latest.close < previous.low
        reversal_against_direction = latest.close > latest.open or ret1 > 0
    elif normalized_direction == "long":
        direction_confirmed_by_close = latest.close > previous.high
        reversal_against_direction = latest.close < latest.open or ret1 < 0
    else:
        direction_confirmed_by_close = False
        reversal_against_direction = False
    recent_ranges = [
        (candle.high - candle.low) / candle.close
        for candle in candles[-21:-1]
        if candle.close > 0 and candle.high >= candle.low
    ]
    median_range_pct = median(recent_ranges) if recent_ranges else range_pct
    range_ratio = range_pct / median_range_pct if median_range_pct > 0 else 1.0

    is_large_range = range_pct >= 0.008 or range_ratio >= 1.5
    needs_confirmation = False
    if normalized_direction == "short":
        needs_confirmation = is_large_range and ret1 <= -0.004 and ret4 >= 0.001
    elif normalized_direction == "long":
        needs_confirmation = is_large_range and ret1 >= 0.004 and ret4 <= -0.001

    return {
        "available": True,
        "range_pct": round(range_pct, 6),
        "body_pct": round(body_pct, 6),
        "close_position": round(close_position, 6),
        "ret1": round(ret1, 6),
        "ret4": round(ret4, 6),
        "median_range_pct": round(median_range_pct, 6),
        "range_ratio": round(range_ratio, 4),
        "needs_confirmation_after_impulse": needs_confirmation,
        "direction_confirmed_by_close": direction_confirmed_by_close,
        "reversal_against_direction": reversal_against_direction,
    }


def build_setup_learning_tags(
    signal: Signal,
    recent_trades: list[TradeRecord],
    *,
    timestamp: datetime,
    timezone_name: str,
) -> list[str]:
    tags: list[str] = []
    entry_candle = signal.metadata.get("entry_candle", {})
    if not isinstance(entry_candle, dict):
        entry_candle = {}
    rsi_value = indicator_from_reason(signal.reason, "rsi")
    ret1 = _finite(entry_candle.get("ret1", 0.0))
    range_ratio = _finite(entry_candle.get("range_ratio", 0.0))
    trend_strength = _finite(signal.metadata.get("trend_strength", 0.0))

    if signal.direction == SignalDirection.SHORT:
        rsi_exhausted = rsi_value is not None and rsi_value < 33.0 and (
            ret1 <= -0.007 or range_ratio > 3.0
        )
        reversal_exhausted = ret1 > 0.0 and rsi_value is not None and rsi_value < 33.0 and (
            range_ratio >= 2.0 or trend_strength < 0.005
        )
        if rsi_exhausted or reversal_exhausted:
            tags.append(SHORT_AFTER_EXHAUSTION_TAG)

    if _is_late_reentry(signal, recent_trades, timestamp=timestamp, timezone_name=timezone_name):
        tags.append(LATE_REENTRY_TAG)

    if entry_candle.get("needs_confirmation_after_impulse"):
        tags.append(CONFIRMATION_AFTER_IMPULSE_TAG)
    return tags


def indicator_from_reason(reason: str, name: str) -> float | None:
    match = re.search(rf"\b{re.escape(name)}=([-+]?\d+(?:\.\d+)?)", reason, flags=re.IGNORECASE)
    if not match:
        return None
    return _optional_float(match.group(1))


def assess_signal_learning(
    signal: Signal,
    feedback_payload: dict[str, list[dict[str, object]]],
    *,
    timestamp: datetime,
    quantity_lots: int,
    timezone_name: str,
    slippage_bps: float,
    commission_bps: float,
    min_samples: int = MIN_LEARNING_SAMPLES,
    low_quality_probability_threshold: float = LOW_QUALITY_PROBABILITY_THRESHOLD,
) -> dict[str, object]:
    rows = [
        row
        for item in feedback_payload.get("resolved", [])
        if (row := _row_from_feedback_item(item, timezone_name=timezone_name)) is not None
    ]
    if len(rows) < min_samples:
        return _unavailable("insufficient resolved signal feedback", rows, signal=signal)

    wins = [int(row["win"]) for row in rows]
    if sum(wins) == 0 or sum(wins) == len(wins):
        return _unavailable("feedback has only one outcome class", rows, signal=signal)

    sample = _row_from_signal(
        signal,
        timestamp=timestamp,
        quantity_lots=quantity_lots,
        timezone_name=timezone_name,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
    )

    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:  # pragma: no cover - optional ML dependency
        return _unavailable(f"ml dependencies unavailable: {exc}", rows, signal=signal)

    frame = pd.DataFrame(rows)
    x_train = frame[_FEATURES]
    y_train = frame["win"].astype(int)
    pnl_train = frame["net_pnl"].astype(float)

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), _CATEGORICAL_FEATURES),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                _NUMERIC_FEATURES,
            ),
        ]
    )
    classifier = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=17,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    regressor = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=19,
                    min_samples_leaf=5,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    regressor.fit(x_train, pnl_train)

    x_sample = pd.DataFrame([sample])[_FEATURES]
    classes = list(classifier.named_steps["model"].classes_)
    positive_index = classes.index(1)
    probability_profit = float(classifier.predict_proba(x_sample)[0][positive_index])
    expected_pnl_per_lot = float(regressor.predict(x_sample)[0])
    quantity = max(1, int(quantity_lots))
    expected_pnl_position = expected_pnl_per_lot * quantity
    commission_round_rub = _round_turnover_commission_rub(signal, quantity, commission_bps)
    required_net_edge_rub = 2.0 * commission_round_rub

    tags: list[str] = []
    if probability_profit < low_quality_probability_threshold:
        tags.append(LOW_QUALITY_TAG)
    if expected_pnl_per_lot < 0:
        tags.append(NEGATIVE_EXPECTANCY_TAG)
    if expected_pnl_position <= required_net_edge_rub:
        tags.append(COMMISSION_EDGE_TAG)
    if _needs_confirmation_after_impulse(signal):
        tags.append(CONFIRMATION_AFTER_IMPULSE_TAG)
    for tag in signal.metadata.get("setup_learning_tags", []):
        if isinstance(tag, str) and tag not in tags:
            tags.append(tag)
    blocks_entry = (
        LOW_QUALITY_TAG in tags
        or NEGATIVE_EXPECTANCY_TAG in tags
        or COMMISSION_EDGE_TAG in tags
    )

    return {
        "available": True,
        "action": "block_entry" if blocks_entry else "allow_entry",
        "blocks_entry": blocks_entry,
        "model": "random_forest_signal_feedback",
        "resolved_samples": len(feedback_payload.get("resolved", [])),
        "usable_samples": len(rows),
        "training_win_rate_pct": round(sum(wins) / len(wins) * 100.0, 3),
        "training_expectancy_per_lot_rub": round(sum(float(row["net_pnl"]) for row in rows) / len(rows), 4),
        "probability_profit": round(probability_profit, 4),
        "expected_pnl_per_lot_rub": round(expected_pnl_per_lot, 4),
        "expected_pnl_position_rub": round(expected_pnl_position, 2),
        "round_turnover_commission_rub": round(commission_round_rub, 2),
        "required_net_edge_rub": round(required_net_edge_rub, 2),
        "low_quality_probability_threshold": low_quality_probability_threshold,
        "learning_tags": tags,
    }


def _unavailable(reason: str, rows: list[dict[str, object]], *, signal: Signal | None = None) -> dict[str, object]:
    wins = sum(int(row.get("win", 0)) for row in rows)
    tags: list[str] = []
    if signal is not None:
        tags = [
            str(tag)
            for tag in signal.metadata.get("setup_learning_tags", [])
            if str(tag)
        ]
    return {
        "available": False,
        "action": "observe_only",
        "blocks_entry": False,
        "reason": reason,
        "usable_samples": len(rows),
        "training_win_rate_pct": round(wins / len(rows) * 100.0, 3) if rows else 0.0,
        "learning_tags": tags,
    }


def _row_from_feedback_item(
    item: dict[str, object],
    *,
    timezone_name: str,
) -> dict[str, object] | None:
    try:
        created_at = datetime.fromisoformat(str(item["created_at"]))
        entry_price = float(item["entry_price"])
        stop_price = float(item["stop_price"])
        take_profit = float(item["take_profit"])
        net_pnl = float(item["net_pnl"])
    except (KeyError, TypeError, ValueError):
        return None
    if entry_price <= 0 or not math.isfinite(net_pnl):
        return None

    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return _feature_row(
        symbol=str(item.get("symbol", "")),
        direction=str(item.get("direction", "")),
        timestamp=created_at,
        timezone_name=timezone_name,
        signal_strength=float(item.get("signal_strength", 0.0) or 0.0),
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit=take_profit,
        context_score=float(item.get("context_score", metadata.get("context_score", 0.0)) or 0.0),
        metadata=metadata,
        slippage_bps=float(item.get("slippage_bps", 0.0) or 0.0),
        commission_bps=float(item.get("commission_bps", 0.0) or 0.0),
        net_pnl=net_pnl,
    )


def _row_from_signal(
    signal: Signal,
    *,
    timestamp: datetime,
    quantity_lots: int,
    timezone_name: str,
    slippage_bps: float,
    commission_bps: float,
) -> dict[str, object]:
    row = _feature_row(
        symbol=signal.instrument.symbol,
        direction=signal.direction.value,
        timestamp=timestamp,
        timezone_name=timezone_name,
        signal_strength=signal.strength,
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
        take_profit=signal.take_profit,
        context_score=signal.context_score,
        metadata=signal.metadata,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        net_pnl=0.0,
    )
    row["quantity_lots"] = max(1, int(quantity_lots))
    return row


def _feature_row(
    *,
    symbol: str,
    direction: str,
    timestamp: datetime,
    timezone_name: str,
    signal_strength: float,
    entry_price: float,
    stop_price: float,
    take_profit: float,
    context_score: float,
    metadata: dict[str, object],
    slippage_bps: float,
    commission_bps: float,
    net_pnl: float,
) -> dict[str, object]:
    localized = timestamp.astimezone(ZoneInfo(timezone_name))
    stop_pct = abs(stop_price - entry_price) / entry_price if entry_price > 0 else 0.0
    take_pct = abs(take_profit - entry_price) / entry_price if entry_price > 0 else 0.0
    entry_candle = metadata.get("entry_candle", {})
    if not isinstance(entry_candle, dict):
        entry_candle = {}
    microstructure = metadata.get("microstructure", {})
    if not isinstance(microstructure, dict):
        microstructure = {}
    return {
        "symbol": symbol.strip().upper(),
        "direction": direction.strip().lower(),
        "hour_msk": localized.hour,
        "weekday": localized.weekday(),
        "signal_strength": _finite(signal_strength),
        "entry_price": _finite(entry_price),
        "stop_pct": _finite(stop_pct),
        "take_pct": _finite(take_pct),
        "reward_risk": _finite(take_pct / stop_pct if stop_pct > 0 else 0.0),
        "context_score": _finite(context_score),
        "trend_strength": _finite(metadata.get("trend_strength", 0.0)),
        "turnover": _finite(metadata.get("turnover", 0.0)),
        "volatility": _finite(metadata.get("volatility", 0.0)),
        "entry_candle_range_pct": _finite(entry_candle.get("range_pct", 0.0)),
        "entry_candle_body_pct": _finite(entry_candle.get("body_pct", 0.0)),
        "entry_candle_ret1": _finite(entry_candle.get("ret1", 0.0)),
        "entry_candle_ret4": _finite(entry_candle.get("ret4", 0.0)),
        "entry_candle_range_ratio": _finite(entry_candle.get("range_ratio", 0.0)),
        "micro_spread_bps": _finite(microstructure.get("spread_bps", 0.0)),
        "micro_liquidity_cover": _finite(microstructure.get("entry_liquidity_cover", 0.0)),
        "micro_side_imbalance": _finite(microstructure.get("side_imbalance", 0.0)),
        "slippage_bps": _finite(slippage_bps),
        "commission_bps": _finite(commission_bps),
        "net_pnl": _finite(net_pnl),
        "win": 1 if net_pnl > 0 else 0,
    }


def _needs_confirmation_after_impulse(signal: Signal) -> bool:
    entry_candle = signal.metadata.get("entry_candle", {})
    if not isinstance(entry_candle, dict):
        return False
    return bool(entry_candle.get("needs_confirmation_after_impulse"))


def _round_turnover_commission_rub(
    signal: Signal,
    quantity_lots: int,
    commission_bps: float,
) -> float:
    notional = abs(signal.entry_price * max(1, int(signal.instrument.lot_size)) * max(1, quantity_lots))
    return 2.0 * notional * max(0.0, float(commission_bps)) / 10_000


def _is_late_reentry(
    signal: Signal,
    recent_trades: list[TradeRecord],
    *,
    timestamp: datetime,
    timezone_name: str,
) -> bool:
    timezone_info = ZoneInfo(timezone_name)
    local_timestamp = timestamp.astimezone(timezone_info)
    for trade in reversed(recent_trades[-50:]):
        if trade.symbol != signal.instrument.symbol:
            continue
        if trade.direction != signal.direction:
            continue
        if trade.exit_time.astimezone(timezone_info).date() != local_timestamp.date():
            continue
        if trade.exit_time > timestamp:
            continue
        if trade.net_pnl > 0 or trade.reason == "take-profit":
            return True
    return False


def _planned_risk_rub(trade: TradeRecord) -> float | None:
    initial_stop = _optional_float(trade.initial_stop_price)
    if initial_stop is None or initial_stop <= 0:
        return None
    distance = abs(float(trade.entry_price) - initial_stop)
    if distance <= 0:
        return None
    units = _infer_units(trade)
    return distance * units


def _infer_units(trade: TradeRecord) -> float:
    if trade.direction.value == "long":
        price_delta = trade.exit_price - trade.entry_price
    else:
        price_delta = trade.entry_price - trade.exit_price
    if abs(price_delta) > 1e-12 and abs(trade.gross_pnl) > 1e-12:
        return abs(trade.gross_pnl / price_delta)
    return max(1, int(trade.quantity_lots))


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _post_close_summary(
    outcome: str,
    exit_reason: str,
    realized_r: float | None,
    ml_verdict: str,
) -> str:
    r_text = "n/a" if realized_r is None else f"{realized_r:.2f}R"
    if outcome == "profit":
        base = f"profit via {exit_reason}, {r_text}"
    elif outcome == "error":
        base = f"error via {exit_reason}, {r_text}"
    else:
        base = f"flat via {exit_reason}, {r_text}"
    if ml_verdict == "not_available":
        return base
    return f"{base}; {ml_verdict}"


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0
