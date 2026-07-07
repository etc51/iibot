# scripts/windows

## Purpose

PowerShell helpers for local Windows operation and quick status checks.

## What is here

- `run-min-dashboard.ps1` starts the local minimal dashboard.
- `run-paper-loop.ps1` runs local paper-cycle loops.
- `start-server.ps1` and `status-server.ps1` help manage local runtime processes.
- `install-server-task.ps1` wires Windows scheduled task style execution.

## Rules

- Keep PowerShell compatible with the user's default Windows shell.
- Do not hard-code tokens or account secrets.
- Prefer explicit config paths and visible status output.
- Avoid destructive file operations in these scripts.

## Search hints

- Use `rg "python|samosbor|config|Start-Process" scripts/windows`.
- Use `Get-Content scripts/windows/<file>.ps1` when checking local runtime behavior.
