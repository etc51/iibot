from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from decimal import Decimal
from typing import Iterator
from uuid import uuid4

from ..config import AppConfig, read_secret_from_env_or_file
from ..domain import Signal, SignalDirection


def _sandbox_sdk_imports():
    try:
        from t_tech.invest import OrderDirection, OrderType
        from t_tech.invest.sandbox.client import SandboxClient
        from t_tech.invest.utils import decimal_to_money
    except ImportError as exc:  # pragma: no cover - depends on external package
        raise RuntimeError(
            "T-Bank sandbox SDK is unavailable. Install requirements-tbank.txt first."
        ) from exc
    return OrderDirection, OrderType, SandboxClient, decimal_to_money


class TBankSandboxExecutor:
    def __init__(self, config: AppConfig):
        self.config = config

    def _token(self) -> str:
        return read_secret_from_env_or_file(
            self.config.tbank.sandbox_token_env,
            self.config.tbank.sandbox_token_file,
            label="T-Bank sandbox token",
        )

    @contextmanager
    def _client(self) -> Iterator[object]:
        _, _, sandbox_client, _ = _sandbox_sdk_imports()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sandbox_client(self._token()) as client:
                yield client

    def ensure_account(self) -> str:
        desired_name = self.config.tbank.account_name
        explicit_id = os.environ.get(self.config.tbank.account_id_env, "")
        with self._client() as client:
            for account in client.users.get_accounts().accounts:
                if explicit_id and account.id == explicit_id:
                    return account.id
                if account.name == desired_name:
                    return account.id
            response = client.sandbox.open_sandbox_account(name=desired_name)
            return response.account_id

    def fund_account(self, amount_rub: float) -> None:
        _, _, _, decimal_to_money = _sandbox_sdk_imports()
        account_id = self.ensure_account()
        with self._client() as client:
            client.sandbox.sandbox_pay_in(
                account_id=account_id,
                amount=decimal_to_money(Decimal(str(amount_rub)), "rub"),
            )

    def submit_market_order(self, signal: Signal, quantity_lots: int) -> dict[str, str]:
        order_direction, order_type, _, _ = _sandbox_sdk_imports()
        account_id = self.ensure_account()
        direction = (
            order_direction.ORDER_DIRECTION_BUY
            if signal.direction == SignalDirection.LONG
            else order_direction.ORDER_DIRECTION_SELL
        )
        with self._client() as client:
            response = client.orders.post_order(
                instrument_id=signal.instrument.instrument_id,
                quantity=quantity_lots,
                direction=direction,
                account_id=account_id,
                order_type=order_type.ORDER_TYPE_MARKET,
                order_id=str(uuid4()),
            )
        return {
            "order_id": response.order_id,
            "execution_report_status": str(response.execution_report_status),
            "message": response.message,
        }
