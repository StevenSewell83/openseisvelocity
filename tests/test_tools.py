from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from osv import tools
from osv.tools import (
    ToolNotFoundError,
    ToolRunError,
    ToolTimeoutError,
    available_tools,
    run_tool,
    tool_versions,
    which_tool,
)


def make_exe(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(0o755)
    return p


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for v in ("OSV_TOOLS_DIR", "CONDA_PREFIX", "OSV_EXTRAS_PREFIX"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(tools, "REPO_BUILD_BIN", tmp_path / "nonexistent_build")
    # an install.sh --venv environment has <sys.prefix>/bin/osv-tools: hide it from the tests
    monkeypatch.setattr("sys.prefix", str(tmp_path / "no_prefix"))
    monkeypatch.setenv("PATH", str(tmp_path / "emptypath") + os.pathsep + "/usr/bin:/bin")


def test_groups():
    assert tools.tool_group("velest") == "velest"
    assert tools.tool_group("fm3d") == "fmtomo"
    assert tools.tool_group("tomo3di") == "extras"
    assert len(tools.TOOL_GROUPS) == 27


def test_missing_message(tmp_path):
    with pytest.raises(ToolNotFoundError) as e:
        which_tool("velest")
    assert "install.sh" in str(e.value) and "OSV_TOOLS_DIR" in str(e.value)
    assert available_tools()["velest"] is None


def test_resolution_order(tmp_path, monkeypatch, caplog):
    path_dir = tmp_path / "emptypath"
    make_exe(path_dir, "velest", "exit 0")
    with caplog.at_level(logging.WARNING):
        assert which_tool("velest") == path_dir / "velest"
    assert "PATH" in caplog.text

    build = make_exe(tmp_path / "build", "velest", "exit 0")
    monkeypatch.setattr(tools, "REPO_BUILD_BIN", tmp_path / "build")
    assert which_tool("velest") == build

    conda = make_exe(tmp_path / "conda" / "bin" / "osv-tools", "velest", "exit 0")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "conda"))
    assert which_tool("velest") == conda

    envd = make_exe(tmp_path / "envd", "velest", "exit 0")
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "envd"))
    assert which_tool("velest") == envd

    explicit = make_exe(tmp_path / "explicit", "myv", "exit 0")
    assert which_tool("velest", overrides={"velest": explicit}) == explicit
    with pytest.raises(ToolNotFoundError):
        which_tool("velest", overrides={"velest": tmp_path / "nope"})


def test_name_map_and_prefix(tmp_path, monkeypatch):
    d = make_exe(tmp_path / "t", "osvx_tomo3di", "exit 0").parent
    monkeypatch.setenv("OSV_TOOLS_DIR", str(d))
    assert which_tool("tomo3di", name_map={"tomo3di": "osvx_tomo3di"}).name == "osvx_tomo3di"
    monkeypatch.setenv("OSV_EXTRAS_PREFIX", "osvx_")
    assert which_tool("tomo3di").name == "osvx_tomo3di"
    with pytest.raises(ToolNotFoundError):
        which_tool("fm3d")


def test_run_logs_cwd_args_stdin(tmp_path, monkeypatch):
    make_exe(
        tmp_path / "bin",
        "velest",
        'pwd; echo "args: $@"; cat; echo "e" >&2; echo "$FOO"; cat control.txt',
    )
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "bin"))
    run = tmp_path / "run"
    run.mkdir()
    (run / "control.txt").write_text("ctl\n")
    r = run_tool("velest", run, args=["a", "b"], stdin="hello\n", env={"FOO": "bar"})
    assert r.ok and r.returncode == 0 and r.wall_time >= 0
    out = (run / "velest.stdout.log").read_text()
    assert str(run) in out and "args: a b" in out and "hello" in out
    assert "bar" in out and "ctl" in out
    assert (run / "velest.stderr.log").read_text() == "e\n"
    r2 = run_tool("velest", run, log_prefix="second")
    assert r2.stdout_log == run / "second.stdout.log" and r2.stdout_log.exists()


def test_nonzero_exit(tmp_path, monkeypatch):
    make_exe(tmp_path / "bin", "fm3d", "echo boom >&2; exit 3")
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "bin"))
    with pytest.raises(ToolRunError) as e:
        run_tool("fm3d", tmp_path / "r")
    assert "return code 3" in str(e.value) and "boom" in str(e.value)
    assert "fm3d.stderr.log" in str(e.value)
    assert e.value.result.returncode == 3
    r = run_tool("fm3d", tmp_path / "r", check=False)
    assert r.returncode == 3 and not r.ok


def test_timeout_kills_group(tmp_path, monkeypatch):
    marker = tmp_path / "child_alive"
    make_exe(
        tmp_path / "bin",
        "fm3d",
        f"(sleep 2; touch {marker}) &\nsleep 30",
    )
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "bin"))
    t0 = time.monotonic()
    with pytest.raises(ToolTimeoutError):
        run_tool("fm3d", tmp_path / "r", timeout=0.5)
    assert time.monotonic() - t0 < 10
    time.sleep(2.5)
    assert not marker.exists()  # background child was killed too


@pytest.mark.parametrize(
    "line",
    ["At line 3 Fortran runtime error: bad", "Segmentation fault", "STOP 2", "STOP input missing"],
)
def test_stdout_error_detection(tmp_path, monkeypatch, line):
    make_exe(tmp_path / "bin", "velest", f"echo '{line}'; exit 0")
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "bin"))
    with pytest.raises(ToolRunError) as e:
        run_tool("velest", tmp_path / "r")
    assert "despite exit code 0" in str(e.value)
    r = run_tool("velest", tmp_path / "r", check=False)
    assert r.detected_errors


def test_plain_stop_ok_and_custom_patterns(tmp_path, monkeypatch):
    make_exe(tmp_path / "bin", "velest", "echo STOP; echo 'WARNING bad thing'")
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "bin"))
    assert run_tool("velest", tmp_path / "r").ok
    with pytest.raises(ToolRunError):
        run_tool("velest", tmp_path / "r", error_patterns=[r"WARNING bad"])
    monkeypatch.setitem(tools.TOOL_ERROR_PATTERNS, "velest", (r"WARNING bad",))
    with pytest.raises(ToolRunError):
        run_tool("velest", tmp_path / "r")


def test_tool_versions(tmp_path, monkeypatch):
    p = make_exe(tmp_path / "bin", "velest", "exit 0")
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "bin"))
    v = tool_versions()
    assert list(v) == ["velest"]
    assert v["velest"]["path"] == str(p) and len(str(v["velest"]["sha256"])) == 64
    assert v["velest"]["mtime"] == p.stat().st_mtime


def test_venv_prefix_search(tmp_path, monkeypatch):
    """Tools installed by ``install.sh --venv`` are found without activating the venv."""
    exe = make_exe(tmp_path / "venv" / "bin" / "osv-tools", "velest", "exit 0")
    monkeypatch.setattr("sys.prefix", str(tmp_path / "venv"))
    assert which_tool("velest") == exe
