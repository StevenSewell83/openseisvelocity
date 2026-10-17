"""Command line interface for openSeisVelocity (`osv`).

Each command is a separate function so later tasks can replace one at a time.
"""

from __future__ import annotations

import typer

from osv import __version__

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
def init_cmd() -> None:
    """Create a new project directory with a config.yaml template."""
    _not_implemented("init")


@app.command("doctor")
def doctor_cmd() -> None:
    """Check the installation and external tools."""
    _not_implemented("doctor")


@app.command("prep")
def prep_cmd() -> None:
    """Stage 1: prepare and QC catalogue and station data."""
    _not_implemented("prep")


@app.command("velest-mc")
def velest_mc_cmd() -> None:
    """Stage 2: run VELEST Monte Carlo ensemble."""
    _not_implemented("velest-mc")


@app.command("velest-analyse")
def velest_analyse_cmd() -> None:
    """Stage 2: analyse the VELEST ensemble."""
    _not_implemented("velest-analyse")


@app.command("build-model")
def build_model_cmd() -> None:
    """Stage 3: build the FMTOMO starting model."""
    _not_implemented("build-model")


@app.command("tradeoff")
def tradeoff_cmd() -> None:
    """Stage 4: run damping/smoothing trade-off sweeps."""
    _not_implemented("tradeoff")


@app.command("invert")
def invert_cmd() -> None:
    """Stage 4: run the FMTOMO inversion."""
    _not_implemented("invert")


@app.command("analyse")
def analyse_cmd() -> None:
    """Stage 5: analyse inversion results."""
    _not_implemented("analyse")


@app.command("resolution")
def resolution_cmd() -> None:
    """Stage 6: run resolution tests."""
    _not_implemented("resolution")


@app.command("report")
def report_cmd() -> None:
    """Build the project report."""
    _not_implemented("report")


@app.command("run")
def run_cmd() -> None:
    """Run all stages."""
    _not_implemented("run")


@app.command("plot")
def plot_cmd() -> None:
    """Regenerate figures for a stage without re-running it."""
    _not_implemented("plot")


@config_app.command("show")
def config_show_cmd() -> None:
    """Print the settings in use."""
    _not_implemented("config show")
