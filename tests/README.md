# tests

## Purpose

Unit and regression tests for the trading system, runtime reports, dashboards, data providers, and safety logic.

## What is here

- `test_paper_cycle.py` covers paper-cycle decisions and event/report outputs.
- `test_trade_review.py`, `test_daily_review.py`, and autonomy tests cover learning/tuning payloads.
- `test_dashboard.py` and `test_minimal_dashboard.py` cover dashboard payload and rendering behavior.
- Provider tests cover CSV, parquet, MOEX data packs, and T-Bank helper logic.
- Risk, backtest, strategy, and execution tests cover core trading mechanics.

## Rules

- Prefer fake providers and deterministic fixtures over live network calls.
- Add focused regression tests for every bug fixed in runtime behavior.
- Keep tests close to the module or workflow they protect.
- Run `python -m pytest -q` before committing broad changes.

## Search hints

- Use `rg "commit_hash|shadow_only|wait_pullback|RESOURCE_EXHAUSTED|stop-loss" tests`.
- Use `rg "class .*Test|def test_" tests/<file>.py` to locate coverage quickly.
