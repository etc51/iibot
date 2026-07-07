# src/samosbor/analysis

## Purpose

Shared market-analysis helpers used by strategies, reviews, and runtime diagnostics.

## What is here

- `context.py` builds reusable market context.
- `indicators.py` contains indicator calculations such as ATR and related series helpers.

## Rules

- Keep indicator functions deterministic and side-effect free.
- Avoid broker/API calls here; this package should operate on passed candles/data only.
- Add tests when changing formulas or window behavior.

## Search hints

- Use `rg "atr|indicator|context" src/samosbor/analysis tests`.
- Use `rg "from \\.analysis|from .*analysis" src/samosbor`.
