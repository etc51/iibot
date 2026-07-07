# src/samosbor/reporting

## Purpose

Report writers and metric builders for backtests, paper reports, research output, and runtime JSON artifacts.

## What is here

- `metrics.py` computes summary statistics.
- `paper_report.py` builds and writes paper trading reports.
- `research_writer.py` writes optimizer, Monte Carlo, and walk-forward reports.
- `writer.py` contains shared JSON/CSV/report writing helpers.

## Rules

- Runtime JSON reports should include `commit_hash` through runtime metadata helpers.
- Keep CSV schemas stable unless tests and consumers are updated.
- Do not perform trading decisions in reporting code; only summarize and serialize.

## Search hints

- Use `rg "write_json_payload|with_runtime_metadata|summary.json|commit_hash" src/samosbor/reporting src/samosbor`.
- Use `rg "profit_factor|win_rate|expectancy|drawdown" src/samosbor/reporting tests`.
