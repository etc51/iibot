# configs

## Purpose

Configuration profiles for local research, server paper runtime, T-Bank runtime, and candidate strategy experiments.

## What is here

- `paper.toml` is the baseline local paper profile.
- `server_tbank_stocks_intraday_300k_focused.toml` is the main server paper-runtime profile.
- `*.effective.toml` files are derived runtime configs built from base configs plus latest autotune artifacts.
- `local_pack_*.toml` files are offline/local research profiles.
- `experiments/` contains narrower experimental profiles and time-frame variants.

## Rules

- Do not commit secrets, account tokens, or raw credential values here.
- Treat base configs as the source of truth; regenerate effective configs with `refresh-effective-config` when needed.
- Keep `execution.mode` in safe paper/sandbox modes unless an explicit safety review changes code and config together.
- When changing risk, universe, or T-Bank settings, run the relevant tests and check the external dashboard after deploy.

## Search hints

- Use `rg "server_tbank_stocks_intraday_300k_focused" configs src tests` for the active runtime profile.
- Use `rg "allow_live_trading|mode|state_path|output_dir" configs` for safety/runtime settings.
