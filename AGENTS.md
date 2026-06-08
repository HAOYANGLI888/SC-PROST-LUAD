# AGENTS.md

This repository is developed stage by stage. Keep changes small, runnable, tested, and auditable.

## Project Rules

- Use Python as the primary language.
- Keep all paths Windows-compatible by using `pathlib.Path` and relative paths from the project root.
- Do not make Linux-only bash scripts the only entry point. Prefer Python CLIs and PowerShell-compatible commands.
- Store raw data under `data/raw/`, processed data under `data/processed/`, metadata under `data/metadata/`, and generated outputs under `outputs/`.
- Save figures to `outputs/figures/`, tables to `outputs/tables/`, checkpoints to `outputs/checkpoints/`, and logs to `outputs/logs/`.
- Do not commit large data files, model weights, private credentials, or API tokens.
- Every stage must generate `outputs/audit/<stage>/audit_report.md`.
- Every main script must expose `--help`.
- Each stage needs a smoke test that can run on a tiny local example or on metadata-only checks.

## Stage Discipline

Do not jump ahead. If working on Stage N, avoid implementing Stage N+1 except for harmless documentation of the planned entry point. New code should remain modular under `src/scprost_luad/`, with thin scripts in `scripts/`.

## Reproducibility

- Put parameters in YAML under `configs/`.
- Record input files, output files, commands, potential issues, and next steps in the audit report.
- Prefer deterministic seeds where practical.
- Keep generated files in `outputs/` unless they are small metadata templates.
