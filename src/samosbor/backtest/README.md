# src/samosbor/backtest

## Purpose

Historical backtest engine for evaluating strategies on candle data.

## What is here

- `engine.py` runs strategy signals through risk sizing, fills, exits, slippage, commissions, and portfolio accounting.

## Rules

- Keep the backtest deterministic for a fixed candle set and config.
- Do not call live APIs from backtest code.
- When changing fill, stop, take-profit, or commission behavior, update tests and compare report metrics.

## Search hints

- Use `rg "BacktestEngine|stop|take_profit|commission|slippage" src/samosbor/backtest tests`.
- Use `rg "run_with_instruments|compute_summary" src/samosbor`.
