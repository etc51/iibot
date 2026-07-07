# src/samosbor/execution

## Purpose

Execution backends for paper broker state and T-Bank sandbox order flow.

## What is here

- `paper.py` implements local paper portfolio state, position opens/closes, events, and persistence.
- `sandbox.py` contains T-Bank sandbox executor integration.

## Rules

- Keep local-paper behavior auditable through events and state JSON.
- Do not enable real live trading here without explicit safety work across config, CLI, and tests.
- Position quantity, lot size, PnL sign, stop, and take-profit behavior must be covered by tests.

## Search hints

- Use `rg "LocalPaperBroker|open_position|close_position|mark_to_market" src/samosbor/execution tests`.
- Use `rg "ExitReason|quantity_lots|lot_size|unrealized_pnl" src/samosbor`.
