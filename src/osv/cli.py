"""Command line interface for openSeisVelocity (`osv`).

Each command is a separate function so later tasks can replace one at a time.
"""

from __future__ import annotations

from pathlib import Path

import typer

from osv import __version__
from osv import stages as _stages
from osv.config import ConfigError, OSVConfig, dump_config, load_config, write_template

app = typer.Typer(
    name="osv",
    help="openSeisVelocity: seismic velocity inversion with VELEST and FMTOMO.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect the project configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


def _not_implemented(name: str) -> None:
    typer.echo(f"osv {name}: not implemented yet", err=True)
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """openSeisVelocity command line."""


@app.command("version")
def version_cmd() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("init")
def init_cmd(
    directory: Path = typer.Argument(..., help="Project directory to create."),
    name: str | None = typer.Option(None, "--name", help="Project name (default: directory name)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config.yaml."),
) -> None:
    """Create a new project directory with a config.yaml template."""
    try:
        target = write_template(directory, force=force, project_name=name)
    except FileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Wrote {target}")


@app.command("doctor")
def doctor_cmd(
    quick: bool = typer.Option(False, "--quick", help="Skip the smoke tests."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    gridloc: bool = typer.Option(
        False, "--gridloc", help="Also run the gridloc smoke test (about 35 s)."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Project config: also check workdir and data size."
    ),
) -> None:
    """Check the installation and external tools, and run smoke tests."""
    from osv import doctor

    checks = doctor.run_doctor(quick=quick, config=config, gridloc=gridloc)
    if json_out:
        typer.echo(doctor.to_json(checks, quick=quick))
    else:
        typer.echo(doctor.render_text(checks))
    raise typer.Exit(code=doctor.exit_code(checks))


def _load_cfg(config: Path) -> OSVConfig:
    try:
        return load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None


CONFIG_OPT = typer.Option(Path("config.yaml"), "--config", "-c", help="Config file.")
FORCE_OPT = typer.Option(False, "--force", help="Re-run even if up to date.")
FROM_OPT = typer.Option(None, "--from", help="First stage to run.")
TO_OPT = typer.Option(None, "--to", help="Last stage to run.")


def _execute(
    config: Path, *, from_: str | None, to: str | None, force: bool, only: str | None = None
) -> None:
    """Run the pipeline, reporting results/errors in CLI style."""
    cfg = _load_cfg(config)
    try:
        result = _stages.run_pipeline(cfg, from_=from_, to=to, force=force, only=only)
    except _stages.StageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for name, state in result.items():
        typer.echo(f"{name}: {state}")
    if not result:
        typer.echo("No stages selected.")


def bind_stage_command(name: str, help_text: str) -> None:
    """Register ``osv <name>`` running that stage (placeholder until the stage exists).

    A later task only has to register its `osv.stages.Stage` subclass (and list
    its module in `osv.stages.STAGE_MODULES`); this command then runs it with
    the shared ``--config/--force/--from/--to`` options.
    """
    stage_name = _stages.normalise_name(name)

    def command(
        config: Path = CONFIG_OPT,
        force: bool = FORCE_OPT,
        from_: str | None = FROM_OPT,
        to: str | None = TO_OPT,
    ) -> None:
        _stages.discover()
        if stage_name not in {c.name for c in _stages.registered_stages()}:
            _not_implemented(name)
        _execute(config, from_=from_, to=to, force=force, only=stage_name)

    command.__doc__ = help_text
    app.command(name)(command)


for _name, _help in [
    ("prep", "Stage 1: prepare and QC catalogue and station data."),
    ("velest-mc", "Stage 2: run VELEST Monte Carlo ensemble."),
    ("velest-analyse", "Stage 2: analyse the VELEST ensemble."),
    ("build-model", "Stage 3: build the FMTOMO starting model."),
    ("tradeoff", "Stage 4: run damping/smoothing trade-off sweeps."),
    ("invert", "Stage 4: run the FMTOMO inversion."),
    ("analyse", "Stage 5: analyse inversion results."),
    ("resolution", "Stage 6: run resolution tests."),
    ("validation", "Stage 7: validate against 1-D relocation and bootstrap."),
    ("report", "Build the project report."),
]:
    bind_stage_command(_name, _help)


@app.command("run")
def run_cmd(
    config: Path = CONFIG_OPT,
    force: bool = FORCE_OPT,
    from_: str | None = FROM_OPT,
    to: str | None = TO_OPT,
) -> None:
    """Run all stages (skipping those that are up to date)."""
    _stages.discover()
    _execute(config, from_=from_, to=to, force=force)


@app.command("plot")
def plot_cmd(
    stage: str = typer.Argument(..., help="Stage name, e.g. prep or velest-mc."),
    config: Path = CONFIG_OPT,
) -> None:
    """Regenerate figures for a stage without re-running it."""
    _stages.discover()
    cfg = _load_cfg(config)
    try:
        _stages.plot_only(cfg, stage)
    except _stages.StageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"{_stages.normalise_name(stage)}: figures regenerated")


@config_app.command("show")
def config_show_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Config file."),
) -> None:
    """Print the settings in use (resolved config as YAML)."""
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(dump_config(cfg), nl=False)
