# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

# openSeisVelocity — project rules for agents

The full design is in `PLAN.md`. Read the sections relevant to your task before writing code.

## Who does what

- The **orchestrator** (Opus) assigns one bead per subagent, reviews the work, closes the bead, and makes all git commits and pushes.
- A **subagent**:
  1. Starts with `bd show <id>` and `bd update <id> --claim`.
  2. Changes only the files its task needs. Ask (in its final report) rather than editing modules another task owns; small additive edits to shared files (e.g. adding a dependency to `pyproject.toml`, registering a CLI subcommand) are fine.
  3. Runs the quality gates below until they pass.
  4. Records a summary with `bd note <id> "..."` (what was built, files, how to test, any limitations or follow-ups).
  5. **Does not** run `git commit`, `git push`, `bd close` or `bd dolt push`.
  6. If it finds extra work, files it: `bd create "..." -t task --deps discovered-from:<id> -l agent:sonnet`.

## Build & Test

Development environment (until the installer, task 1.6, replaces it):

```bash
uv venv .venv --python 3.11          # once
uv pip install -e ".[dev]"           # after pyproject changes
.venv/bin/pytest -q                   # unit tests (must pass)
.venv/bin/pytest -q -m integration    # tests that need built binaries / external data
.venv/bin/ruff check src tests        # lint (must be clean)
.venv/bin/ruff format src tests       # format
```

- Fortran compiler for development: system `gfortran` 11 (`/usr/bin/gfortran`). Builds go in `build/` (git-ignored); never commit binaries.
- Use `--` style non-interactive commands only (`cp -f`, `rm -f`, …).

## Architecture

- Python package `openseisvelocity`, import name `osv`, `src/` layout (`src/osv/...`), console script `osv` (typer).
- Subpackages: `config`, `io` (file formats), `data` (Stage 1), `velest` (Stage 2), `model3d` (Stage 3), `fmtomo` (Stage 4), `analysis` (Stage 5), `resolution` (Stage 6), `validation` (Stage 7), `plotting`, `report`; plus `osv.tools` (running executables), `osv.coords`, `osv.stages` (stage framework).
- Vendored Fortran: `fortran/velest`, `fortran/fmtomo`, `fortran/fmtomo_extras`. Patches are minimal and each one is listed in `fortran/PATCHES.md`.
- Every stage runs in its own numbered directory under `project.workdir` and writes `manifest.json`, `figures/` and `report.html`.

## Conventions

- Python 3.11, full type hints, numpy-style docstrings on public functions, `from __future__ import annotations`.
- Data tables are `pandas.DataFrame`s (stored as Parquet); gridded models are `xarray.Dataset`s with coords `lat`, `lon`, `depth` (km, positive down).
- **Coordinates:** store true signed latitude/longitude (degrees, east positive) and depth in km positive down; elevation in m positive up. Only `osv.io.*` writers convert to program-specific conventions (e.g. VELEST longitude sign). Never hand-roll conversions elsewhere—use `osv.coords`.
- **No hard-coded paths or magic numbers**: anything a user might change comes from the config (`osv.config`). Defaults live in the config schema.
- Each executable run goes in its own directory; never edit shared control files in place.
- Use `logging` (logger per module), not `print`, except in CLI output.
- Raise specific exceptions with messages that say what to fix.
- Figures: use helpers in `osv.plotting` (consistent style, saved via the figure registry with a caption). Every stage provides diagnostic figures and QC checks (see PLAN.md section 5).
- Tests in `tests/` mirror the package layout (`tests/io/test_velest.py`, …). Small fixtures go in `tests/data/` (keep each < 200 kB). Tests needing compiled binaries or external data are marked `@pytest.mark.integration` and skip cleanly when unavailable.

## Test data

- QuakeML catalogue: `manual_only_catalog.xml` (repo root). Known quirks are listed in PLAN.md section 8 — code must handle them, not the data be edited.
- StationXML: `/home/steve/seismic/Station_Files/*_STATIONXML.xml` (outside the repo; override with env var `OSV_STATIONXML_DIR`).
- Legacy VELEST examples: `/home/steve/seismic/velest/MonteCarlo`, `/home/steve/seismic/velest/MC`.
- FMTOMO examples and manuals: `fortran/fmtomo/example1`, `/home/steve/seismic/FMTOMO/docs/*.pdf`.
