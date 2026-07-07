# src/samosbor/risk

## Purpose

Risk approval, position sizing, drawdown checks, and stop/trailing-stop helpers.

## What is here

- `manager.py` implements risk checks for entries, sizing, trailing stops, runner stops, and halt state.

## Rules

- Treat this as safety-critical code.
- Any change to sizing, max risk, halt logic, or stop placement needs focused tests.
- Keep short and long PnL/stop signs explicit; do not rely on implicit direction math.

## Search hints

- Use `rg "RiskManager|approve|drawdown|trailing_stop|runner" src/samosbor/risk tests`.
- Use `rg "max_risk|max_positions|halt|stop_price" src/samosbor`.
