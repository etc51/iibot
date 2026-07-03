from __future__ import annotations

import os

from .domain import TradeMode


class LiveTradingDisabledError(RuntimeError):
    """Raised when code tries to leave paper-only mode."""


def assert_paper_only_mode(
    mode: TradeMode,
    *,
    allow_live_trading: bool,
    live_flag: bool = False,
) -> None:
    if mode != TradeMode.LIVE:
        return

    env_unlock = os.environ.get("SAMOSBOR_ENABLE_LIVE_TRADING", "")
    if not allow_live_trading or not live_flag or env_unlock != "YES_I_UNDERSTAND":
        raise LiveTradingDisabledError(
            "Live trading is blocked. Explicit unlock is required, "
            "and the current prototype intentionally ships in paper-only mode."
        )

    raise LiveTradingDisabledError(
        "Live trading remains intentionally unimplemented in this prototype."
    )
