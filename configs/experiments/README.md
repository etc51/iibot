# configs/experiments

## Purpose

Experimental TOML profiles for comparing shorter timeframes, quicker research windows, and focused symbol universes.

## What is here

- `*_15min*.toml` profiles test 15-minute candle variants.
- `*_30min*.toml` profiles test 30-minute candle variants.
- `*_quick.toml` profiles are reduced runs for faster iteration.
- `*_240d.toml` profiles use longer historical windows.

## Rules

- Keep experiments reproducible: document material parameter changes in the file name or nearby comments.
- Do not point experiments at live trading state.
- Promote only evidence-backed settings into the main `configs/` profiles.

## Search hints

- Use `rg "timeframe|history_days|subset|instruments" configs/experiments`.
- Compare experiment profiles with `git diff -- configs/experiments/<file>`.
