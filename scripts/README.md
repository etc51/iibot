# scripts

## Purpose

Operational scripts for server and Windows runtime management.

## What is here

- `server/` contains Linux scripts used by systemd on `/opt/samosbor`.
- `windows/` contains PowerShell helpers for local Windows operation.

## Rules

- Keep scripts idempotent where possible.
- Do not bake credentials into scripts.
- Prefer invoking the package CLI (`python -m samosbor.cli`) over duplicating application logic in shell.
- When changing a script used by systemd, verify the matching unit in `deploy/systemd/`.

## Search hints

- Use `rg "python -m samosbor.cli|systemctl|flock|config" scripts deploy`.
- Use `rg "paper-cycle|dashboard|microstructure|daily-review" scripts`.
