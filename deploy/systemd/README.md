# deploy/systemd

## Purpose

Systemd service and timer definitions for the server runtime.

## What is here

- `samosbor-paper-cycle.*` runs recurring paper-trading cycles.
- `samosbor-dashboard.service` serves the external dashboard.
- `samosbor-microstructure.service` collects order-book snapshots.
- `samosbor-daily-review.*` runs daily review and tuning jobs.
- `samosbor-updater.*` pulls new GitHub commits on the server.

## Rules

- Keep one responsibility per unit.
- Avoid overlapping T-Bank-heavy jobs; paper-cycle and microstructure both use market-data API quota.
- When a unit changes, reload systemd and restart only the affected services.
- Keep paths aligned with `/opt/samosbor` unless the server install path changes intentionally.

## Search hints

- Use `rg "paper-cycle|microstructure|dashboard" deploy/systemd scripts/server`.
- Use `systemctl status <unit> --no-pager -n 50` on the server for live diagnostics.
