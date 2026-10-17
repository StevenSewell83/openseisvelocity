from __future__ import annotations

import pytest
from typer.testing import CliRunner

from osv import __version__
from osv.cli import app

runner = CliRunner()

COMMANDS = [
    "init", "doctor", "prep", "velest-mc", "velest-analyse", "build-model", "tradeoff",
    "invert", "analyse", "resolution", "validation", "report", "run", "plot", "config", "version",
]  # fmt: skip


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_version_flag_and_command() -> None:
    assert __version__ in runner.invoke(app, ["--version"]).output
    assert __version__ in runner.invoke(app, ["version"]).output


def _placeholders() -> list[list[str]]:
    """Stage commands whose stage class is not registered yet (still placeholders)."""
    from osv import stages

    stages.discover()
    done = {stages.normalise_name(c.name) for c in stages.registered_stages()}
    non_stage = {"init", "doctor", "run", "plot", "config", "version"}
    return [
        [c]
        for c in COMMANDS
        if c not in non_stage and stages.normalise_name(c) not in done
    ]


@pytest.mark.parametrize("args", _placeholders())
def test_placeholders_exit_1(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "not implemented yet" in result.output
