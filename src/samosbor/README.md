# src/samosbor

## Purpose

Main Python package for the MOEX AI Trader / samosbor paper-trading system.

## What is here

- `cli.py` is the command-line entrypoint.
- `orchestrator.py` coordinates data, strategy, risk, broker state, reports, and autonomy jobs.
- `config.py` defines typed TOML configuration.
- `domain.py` contains shared domain objects.
- `dashboard.py` and `minimal_dashboard.py` serve dashboard payloads and HTML.
- `microstructure_collector.py` collects order-book snapshots.
- `runtime_metadata.py` adds the current git commit hash to runtime reports.
- Subpackages split analysis, autonomy, backtest, data, execution, reporting, research, risk, and strategy logic.

## Rules

- Preserve paper-only safety boundaries unless changing safety code and tests deliberately.
- Keep T-Bank access behind provider/executor modules.
- Add or update tests for behavior that changes runtime reports, trading decisions, or risk controls.
- Runtime report payloads should include `commit_hash`.

## Search hints

- Use `rg "paper-cycle|trade-review|daily-review|refresh-effective-config" src/samosbor`.
- Use `rg "commit_hash|with_runtime_metadata|current_commit_hash" src/samosbor tests`.
