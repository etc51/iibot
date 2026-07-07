# docs

## Purpose

Human-readable project notes, architecture writeups, and generated investigation reports.

## What is here

- `architecture.md` describes system design and runtime flow.
- Ad-hoc project reports may be created here during investigations.

## Rules

- Keep durable architecture decisions in committed Markdown.
- Do not commit secrets, access tokens, account identifiers, or raw broker credentials.
- If a report is temporary or machine-local, leave it untracked unless the user explicitly asks to preserve it.
- Prefer concrete numbers, file paths, timestamps, and commit hashes in runtime reports.

## Search hints

- Use `rg "policy|risk|dashboard|T-Bank|commit_hash" docs src tests`.
- Use `git log -- docs` when reconstructing why a design decision changed.
