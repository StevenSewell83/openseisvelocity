"""Tests for ``osv doctor`` (unit tests with monkeypatched tools, plus an integration run)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from osv import doctor, tools
from osv.cli import app

runner = CliRunner()
REPO = Path(__file__).resolve().parents[1]


def _by_name(checks: list[doctor.Check]) -> dict[str, doctor.Check]:
    return {c.name: c for c in checks}


def _make_exe(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No tools anywhere: empty PATH, no env dirs, nonexistent repo build dir."""
    for var in (tools.ENV_TOOLS_DIR, "CONDA_PREFIX", tools.ENV_EXTRAS_PREFIX):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tools, "REPO_BUILD_BIN", tmp_path / "nobuild" / "bin")
    monkeypatch.setattr("sys.prefix", str(tmp_path / "prefix"))
    path_dir = tmp_path / "pathdir"
    path_dir.mkdir()
    monkeypatch.setenv("PATH", str(path_dir))
    return path_dir


def test_missing_tool_fails_with_hint(clean_env: Path) -> None:
    res = _by_name(doctor.check_tools())
    c = res["tool:velest"]
    assert c.status == doctor.FAIL
    assert "install.sh" in c.fix and tools.ENV_TOOLS_DIR in c.fix
    # Extras are optional: warn only
    assert res["tool:gridloc"].status == doctor.WARN
    assert doctor.exit_code(res.values()) == 1


def test_tool_on_path_only_warns(clean_env: Path) -> None:
    _make_exe(clean_env / "velest")
    c = _by_name(doctor.check_tools())["tool:velest"]
    assert c.status == doctor.WARN
    assert "PATH" in c.message and c.fix


def test_tool_in_tools_dir_ok(clean_env: Path, tmp_path: Path, monkeypatch) -> None:
    d = tmp_path / "tools"
    d.mkdir()
    _make_exe(d / "velest")
    monkeypatch.setenv(tools.ENV_TOOLS_DIR, str(d))
    c = _by_name(doctor.check_tools())["tool:velest"]
    assert c.status == doctor.OK and c.details["origin"] == "tools-dir"


def test_renamed_binary_fails(clean_env: Path, tmp_path: Path, monkeypatch) -> None:
    d = tmp_path / "tools"
    d.mkdir()
    _make_exe(d / "velest.bak")
    monkeypatch.setenv(tools.ENV_TOOLS_DIR, str(d))
    assert _by_name(doctor.check_tools())["tool:velest"].status == doctor.FAIL


def test_parse_velest_inc() -> None:
    text = (
        "      parameter(ieq=30)      ! number of earthquakes\n"
        "c      parameter(ieq=3000)\n"
        "      parameter(inshot=0,\n"
        "     &          itotmodels=2,\n"
        "     &          inltot=10,\n"
        "     &          ist=60,\n"
        "     &          maxobsperevent=180)\n"
    )
    assert doctor.parse_velest_inc(text) == {
        "ieq": 30,
        "inshot": 0,
        "itotmodels": 2,
        "inltot": 10,
        "ist": 60,
        "maxobs": 180,
    }


def test_limits_from_build_info(clean_env: Path, tmp_path: Path, monkeypatch) -> None:
    bindir = tmp_path / "p" / "bin"
    share = tmp_path / "p" / "share" / "osv-tools"
    bindir.mkdir(parents=True)
    share.mkdir(parents=True)
    (share / "build_info.json").write_text(json.dumps({"velest_limits": {"ieq": 7, "ist": 9}}))
    lim, src = doctor.velest_limits(bindir / "velest")
    assert lim == {"ieq": 7, "ist": 9} and "build_info" in src


def test_limits_from_inc_fallback(clean_env: Path, tmp_path: Path, monkeypatch) -> None:
    build = tmp_path / "b"
    (build / "fortran" / "velest_src").mkdir(parents=True)
    (build / "fortran" / "velest_src" / "velest_common.inc").write_text(
        "      parameter(ieq=11)\n      parameter(ist=22,\n     &  inltot=3)\n"
    )
    monkeypatch.setattr(tools, "REPO_BUILD_BIN", build / "bin")
    lim, _ = doctor.velest_limits(None)
    assert lim["ieq"] == 11 and lim["ist"] == 22 and lim["inltot"] == 3


def test_project_limits_compared(tmp_path: Path, monkeypatch) -> None:
    from osv.stages import stage_dirname

    stage = tmp_path / stage_dirname("prep")
    stage.mkdir()
    pd.DataFrame({"event_id": range(50)}).to_parquet(stage / "events.parquet")
    pd.DataFrame({"station": range(5)}).to_parquet(stage / "stations.parquet")
    pd.DataFrame({"event_id": [0, 0, 1]}).to_parquet(stage / "picks.parquet")
    monkeypatch.setattr(
        doctor, "velest_limits", lambda *_a, **_k: ({"ieq": 40, "ist": 300, "maxobs": 180}, "x")
    )
    monkeypatch.setattr(tools, "which_tool", lambda *_a, **_k: Path("/x/velest"))

    class P:
        workdir = tmp_path

    class Cfg:
        project = P()

    res = _by_name(doctor.check_project_limits(Cfg()))
    assert res["project:ieq"].status == doctor.FAIL and "--velest-ieq" in res["project:ieq"].fix
    assert res["project:ist"].status == doctor.OK


def test_system_checks(tmp_path: Path, monkeypatch) -> None:
    res = _by_name(doctor.check_system(tmp_path / "new" / "wd", 2))
    assert res["workdir:write"].status == doctor.OK
    assert "n_jobs=2" in res["cpu"].message
    monkeypatch.setattr(doctor, "_free_gb", lambda p: 1.0)
    res = _by_name(doctor.check_system(tmp_path, -1))
    assert res["disk:tmp"].status == doctor.WARN and res["disk:workdir"].status == doctor.WARN
    too_many = (os.cpu_count() or 1) + 5
    assert _by_name(doctor.check_system(tmp_path, too_many))["cpu"].status == doctor.WARN


def _fake_checks(fail: bool) -> list[doctor.Check]:
    out = [
        doctor.Check("a", doctor.OK, "fine"),
        doctor.Check("b", doctor.WARN, "hmm", "do x"),
    ]
    if fail:
        out.append(doctor.Check("c", doctor.FAIL, "bad", "do y", seconds=0.5))
    return out


@pytest.mark.parametrize("fail", [False, True])
def test_cli_exit_codes_and_json(monkeypatch, fail: bool) -> None:
    monkeypatch.setattr(doctor, "run_doctor", lambda **_k: _fake_checks(fail))
    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == (1 if fail else 0)
    data = json.loads(res.stdout)
    assert set(data) == {"osv_version", "python", "quick", "ok", "summary", "checks"}
    assert data["ok"] is (not fail)
    assert data["summary"]["warn"] == 1
    for c in data["checks"]:
        assert set(c) == {"name", "status", "message", "fix", "group", "seconds", "details"}
        assert c["status"] in {"ok", "warn", "fail"}
    text = runner.invoke(app, ["doctor"]).stdout
    assert "do x" in text and ("do y" in text) is fail


def test_plain_text_render() -> None:
    txt = doctor.render_text(_fake_checks(True), use_rich=False)
    assert "FAIL" in txt and "fix: do y" in txt and "1 failures" in txt


def test_quick_skips_smoke(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "run_smoke_tests", lambda **_k: pytest.fail("ran smoke"))
    checks = doctor.run_doctor(quick=True)
    assert not any(c.group == "smoke" for c in checks)


def test_bad_config_reported(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("project: [not a mapping\n")
    checks = doctor.run_doctor(quick=True, config=bad)
    assert _by_name(checks)["config"].status == doctor.FAIL


@pytest.mark.integration
def test_doctor_full_on_repo_build() -> None:
    if (
        not (REPO / "build" / "bin" / "velest").exists()
        or not (REPO / "build" / "bin" / "fm3d").exists()
    ):
        pytest.skip("repo build (build/bin) missing")
    if os.environ.get(tools.ENV_TOOLS_DIR):
        pytest.skip("OSV_TOOLS_DIR overrides the repo build")
    checks = _by_name(doctor.run_doctor())
    failed = {n: c.message for n, c in checks.items() if c.status == doctor.FAIL}
    assert not failed, failed
    assert checks["smoke:velest"].status == doctor.OK and checks["smoke:velest"].seconds is not None
    assert checks["smoke:fm3d"].status == doctor.OK
    assert checks["build:velest_limits"].details.get("ieq")
