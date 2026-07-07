# src/samosbor/autonomy

## Purpose

Autonomous learning, review, tuning, and policy modules used to improve paper trading from evidence.

## What is here

- `trade_review.py`, `daily_review.py`, and `signal_feedback.py` build runtime learning evidence.
- `regime_policy.py`, `market_regime.py`, and `pending_entries.py` control entry decisions and wait/pullback behavior.
- `ml_learning.py`, `entry_confirmation.py`, and `entry_quality_tuning.py` score entry quality.
- `entry_schedule.py`, `entry_symbols.py`, `strategy_tuning.py`, `exit_tuning.py`, and `universe_selection.py` generate autotune recommendations.
- `effective_config.py` merges base config with safe runtime overrides.
- `runner.py` contains runner/trailing-profit helpers.

## Rules

- Treat this package as high blast radius: policy changes need focused tests and runtime report checks.
- Keep recommendations evidence-backed; avoid writing config overrides without guardrails.
- Do not hide rejected signals: shadow, wait, and block reasons must remain visible in reports.
- Keep report payloads stable and include `commit_hash`.

## Search hints

- Use `rg "shadow_only|wait_pullback|regime_policy|policy_decision" src/samosbor/autonomy tests`.
- Use `rg "build_.*payload|write_.*tuning|commit_hash" src/samosbor/autonomy`.
