# src/samosbor/strategy

## Purpose

Trading strategy implementations that produce signals from prepared market history.

## What is here

- `trend_following.py` implements the main trend-following/TA signal generator used by paper runtime and research.

## Rules

- Strategies should emit signals and metadata, not place trades directly.
- Keep signal metadata useful for policy, learning, and report attribution.
- Add tests when changing entry thresholds, breakout logic, market context, or metadata.

## Search hints

- Use `rg "TrendFollowingStrategy|generate_signal|prepare_history|metadata" src/samosbor/strategy tests`.
- Use `rg "min_signal_strength|adx|rsi|breakout|trend" src/samosbor`.
