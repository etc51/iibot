# scripts/server

## Purpose

Linux server scripts used by systemd units and manual operations on `/opt/samosbor`.

## What is here

- `install-server.sh` installs or updates server-side wiring.
- `run-paper-cycle.sh` refreshes effective config and runs one paper cycle.
- `run-dashboard.sh` starts the dashboard.
- `run-microstructure-collector.sh` starts order-book collection.
- `run-daily-review.sh` runs daily review/autonomy reports.
- `update-from-github.sh` pulls the latest GitHub commit.
- `update-offline-parquet-cache.py` and `build-offline-autonomy-config.py` support offline data/config maintenance.

## Rules

- Keep `flock` protection around jobs that must not overlap.
- Do not start extra T-Bank-heavy loops while paper-cycle is running unless quota impact is intentional.
- Keep base/effective config paths aligned with `configs/server_tbank_stocks_intraday_300k_focused*.toml`.
- After script changes, deploy and verify the matching systemd unit.

## Search hints

- Use `rg "flock|EFFECTIVE_CONFIG|BASE_CONFIG|collect-microstructure" scripts/server`.
- Use `journalctl -u samosbor-paper-cycle.service -n 100 --no-pager` on the server for cycle diagnostics.
