# Developer guide

## Architecture

`osv` is a Python package (`src/osv/`, console script `osv`) that wraps the Fortran programs
VELEST and FMTOMO (vendored under `fortran/`; every patch is listed in `fortran/PATCHES.md`).

| Subpackage | Role |
|---|---|
| `osv.config` | Pydantic schema (`schema.py`), loader (path/glob/env expansion), `osv init` template generator |
| `osv.io` | File formats; the only place that converts to program conventions (VELEST longitude sign, FMTOMO radians) |
| `osv.coords` | Coordinate conversions (local projection, VELEST short-distance conversion) |
| `osv.data` | Stage 1 (prep) |
| `osv.velest` | Stage 2 (Monte Carlo runs and analysis) |
| `osv.model3d` | Stage 3 (starting model, user grids, geology, wells) |
| `osv.fmtomo` | Stage 4 (trade-off, inversion) |
| `osv.analysis`, `osv.resolution`, `osv.validation` | Stages 5, 6, 7 |
| `osv.plotting`, `osv.report` | Figure helpers and registry; stage and project HTML reports |
| `osv.tools` | Locating and running executables, each run in its own directory |
| `osv.stages` | Stage framework |
| `osv.doctor`, `osv.cli` | Installation checks and the typer CLI |

Principles (see `CLAUDE.md` and `PLAN.md`): data tables are Parquet-backed DataFrames; gridded
models are xarray Datasets with `lat`, `lon`, `depth` (km, positive down); coordinates are true
signed values everywhere except inside `osv.io` writers ([conventions](conventions.md)); nothing a
user might change is hard-coded, it lives in the config schema; every executable run gets its
own directory.

## Stage framework

A stage is a subclass of `osv.stages.Stage` registered with `@register`. The framework
(`src/osv/stages.py`) provides:

* numbered directories `<workdir>/NN_name/` (the order is `STAGE_ORDER`);
* `StageContext` with `ctx.config`, `ctx.path(...)`, `ctx.figures`, `ctx.logger`, `ctx.n_jobs`,
  `ctx.upstream(name)` and `ctx.add_check(...)`;
* a `manifest.json` per stage (status, run id, timings, config-section hash, input hashes,
  upstream run ids, tool hashes, outputs, QC checks, summary, error);
* staleness: a stage is skipped when complete and when the hash of its `config_sections`, the
  sha256 of its declared `inputs`, its upstream run ids, its tool hashes and its declared
  `outputs` are unchanged;
* `--from`, `--to`, `--force`, and `osv plot` (calls `Stage.plot`).

## Adding a stage

1. Add the config section to `osv/config/schema.py` with defaults and descriptions (use `F(...)`;
   every key needs a description). Regenerate the reference:
   `python scripts/gen_config_reference.py`.
2. Write the stage class:

    ```python
    from osv.stages import Stage, StageContext, register

    @register
    class MyStage(Stage):
        name = "my_stage"
        requires = ("prep",)
        config_sections = ("my_section",)
        outputs = ("result.parquet",)

        def run(self, ctx: StageContext) -> None:
            prep = ctx.upstream("prep")
            ...
            ctx.add_check({"name": "n_rows", "status": "pass", "message": "..."})

        def plot(self, ctx: StageContext) -> None:
            ...
    ```

3. Add the name to `STAGE_ORDER`, its dependencies to `DEFAULT_CONFIG_SECTIONS` and its module to
   `STAGE_MODULES` in `osv/stages.py`; the CLI
   command is bound in `osv/cli.py`.
4. Provide diagnostic figures (through the figure registry with captions), QC checks that say what
   to fix, and summary numbers in the manifest; the report picks them up.
5. Write tests under `tests/<subpackage>/` and a `docs/stageN_*.md` page (add it to `mkdocs.yml`).

## Tests

```bash
.venv/bin/pytest -q                    # unit tests (must pass)
.venv/bin/pytest -q -m integration     # need built binaries or external data
.venv/bin/ruff check src tests scripts # lint
```

`tests/` mirrors the package layout. Small fixtures live in `tests/data/` (each below 200 kB).
Tests that need compiled `velest`/`fm3d` or data outside the repository (Station_Files, legacy
VELEST examples) are marked `@pytest.mark.integration` and skip cleanly when unavailable.
`tests/test_docs.py` checks that `docs/config_reference.md` matches the schema.

## Documentation

```bash
uv pip install -e ".[docs]"
python scripts/gen_config_reference.py   # after any schema change
mkdocs serve                             # live preview
mkdocs build --strict                    # what CI runs; fails on broken links
```

`docs/config_reference.md` is generated; never edit it by hand.

## Beads workflow

Work is tracked with `bd` (beads), not markdown TODO lists.

```bash
bd ready                    # available work
bd show <id>
bd update <id> --claim
bd note <id> "what was built, files, how to test, limitations"
bd create "..." -t task --deps discovered-from:<id>
```

The orchestrator assigns one bead per subagent, reviews the work, closes beads and makes all git
commits and pushes. A subagent claims its bead, changes only the files it needs, runs the quality
gates, records a `bd note`, and does not commit, push or close. See `CLAUDE.md`.

## Continuous integration

`.github/workflows/ci.yml` runs: lint (`ruff check`, advisory `ruff format --check`); unit tests
(builds the Fortran code with `scripts/build_fortran.sh`, `osv doctor --quick`,
`pytest -m "not integration"`); integration tests (`pytest -m integration`); an `install.sh
--dry-run` (a full conda install on manual dispatch); and a Docker image build with
`osv doctor --quick`. Documentation is built with `mkdocs build --strict`.
