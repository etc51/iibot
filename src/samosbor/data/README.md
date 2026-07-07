# src/samosbor/data

## Purpose

Market-data providers and loaders for CSV, local parquet/MOEX data packs, and T-Bank API.

## What is here

- `csv_provider.py` loads simple CSV candles.
- `parquet_directory.py` reads local parquet candle archives.
- `moex_data_pack.py` handles MOEX/offline data packs.
- `tbank.py` integrates with T-Bank Invest market-data API and rate-limit handling.

## Rules

- Keep API-specific logic inside provider modules.
- Do not leak tokens or credential values into logs, fixtures, or reports.
- Respect T-Bank rate limits; avoid concurrent heavy loops unless explicitly coordinated.
- Tests should use fake providers or local fixtures, not live network calls.

## Search hints

- Use `rg "RESOURCE_EXHAUSTED|rate-limit|GetCandles|OrderBook" src/samosbor/data tests`.
- Use `rg "load_history|resolve_universe|get_order_book_snapshot" src/samosbor`.
