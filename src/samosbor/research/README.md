# src/samosbor/research

## Purpose

Research engines for optimization, Monte Carlo robustness checks, targets, and walk-forward validation.

## What is here

- `optimizer.py` searches strategy parameter and symbol candidates.
- `walk_forward.py` validates rolling train/test windows.
- `monte_carlo.py` simulates robustness from trade distributions.
- `targets.py` normalizes daily/monthly target metrics.

## Rules

- Keep research code deterministic when a seed is configured.
- Do not use future data in train/test splits.
- Promote candidates into runtime only through guardrailed tuning/effective-config flow.

## Search hints

- Use `rg "ParameterOptimizer|WalkForwardValidator|MonteCarloSimulator" src/samosbor/research tests`.
- Use `rg "target_daily_profit|monthly_return|positive_probability" src/samosbor`.
