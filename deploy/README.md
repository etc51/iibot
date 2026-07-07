# deploy

## Purpose

Deployment assets for installing the project as long-running services on the external server.

## What is here

- `systemd/` contains service and timer units for paper-cycle, dashboard, microstructure, daily review, and updater jobs.

## Rules

- Keep deployment files deterministic and shell-friendly.
- Prefer changing scripts under `scripts/server/` before changing systemd units, unless the scheduling or service boundary itself changes.
- After deployment changes, verify `systemctl list-units 'samosbor*' --all --no-pager` on the server.

## Search hints

- Use `rg "samosbor-" deploy scripts/server`.
- Use `rg "ExecStart|OnCalendar|WorkingDirectory" deploy/systemd`.
