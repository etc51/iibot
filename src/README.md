# src

## Purpose

Python package source root.

## What is here

- `samosbor/` is the application package used by CLI entrypoints.
- `moex_ai_trader.egg-info/` may appear after editable installs and is generated packaging metadata.

## Rules

- Put application code under `src/samosbor/`.
- Do not edit generated `*.egg-info` metadata manually.
- Keep package imports relative to `samosbor` and covered by tests.

## Search hints

- Use `rg "class TradingOrchestrator|def main|build_.*payload" src/samosbor`.
- Use `rg --files src/samosbor` for a compact source inventory.
