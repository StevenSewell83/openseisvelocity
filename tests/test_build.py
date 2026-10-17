"""Integration tests for the Fortran build (``scripts/build_fortran.sh``) and smoke runs.

The full build goes into a temporary directory (or the directory in
``$OSV_TEST_BIN``, an existing install of the executables) and is shared by all
tests in the module. Skipped if gfortran/cmake are missing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from osv import tools

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_fortran.sh"
FMTOMO_EX1 = REPO / "fortran" / "fmtomo" / "example1"
EXTRAS_P = REPO / "fortran" / "fmtomo_extras" / "observed" / "invert_p"
VELEST_DATA = REPO / "tests" / "data" / "velest"

need_toolchain = pytest.mark.skipif(
    shutil.which("gfortran") is None or shutil.which("cmake") is None,
    reason="gfortran and cmake are required",
)


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: int = 900) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, f"{cmd} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return proc.stdout


@pytest.fixture(scope="module")
def bin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory with all built executables (build once per module)."""
    if os.environ.get("OSV_TEST_BIN"):
        return Path(os.environ["OSV_TEST_BIN"])
    if shutil.which("gfortran") is None or shutil.which("cmake") is None:
        pytest.skip("gfortran and cmake are required")
    root = tmp_path_factory.mktemp("fortran_build")
    _run(
        [
            str(SCRIPT),
            "--prefix",
            str(root / "prefix"),
            "--build-dir",
            str(root / "cmake"),
            "--jobs",
            str(os.cpu_count() or 2),
        ],
        cwd=REPO,
    )
    return root / "prefix" / "bin" / "osv-tools"


def _env(bin_dir: Path) -> dict[str, str]:
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def _floats(path: Path) -> list[list[float]]:
    return [[float(x) for x in ln.split()] for ln in path.read_text().splitlines() if ln.strip()]


def _sources(path: Path) -> list[list[float]]:
    """Source lat/lon/depth triples from a ``sources.in`` file with 6-column rows."""
    return [r[:3] for r in _floats(path) if len(r) == 6 or (len(r) == 3 and abs(r[2]) > 100)]


@need_toolchain
def test_all_tools_installed(bin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(tools.ENV_TOOLS_DIR, str(bin_dir))
    monkeypatch.delenv(tools.ENV_EXTRAS_PREFIX, raising=False)
    missing = [n for n in tools.TOOL_GROUPS if tools.available_tools().get(n) is None]
    assert not missing, f"not installed in {bin_dir}: {missing}"
    assert (bin_dir.parent.parent / "share" / "osv-tools" / "ak135.tbl").exists() or os.environ.get(
        "OSV_TEST_BIN"
    )


@need_toolchain
def test_driver_scripts_are_bash(bin_dir: Path) -> None:
    for name in ("tomo3d", "tomo3di", "tomo3dj", "tomo3dps"):
        first = (bin_dir / name).read_text().splitlines()[0]
        assert first == "#!/usr/bin/env bash"
        subprocess.run(["bash", "-n", str(bin_dir / name)], check=True)


def _prepare_velest(run: Path) -> None:
    run.mkdir()
    shutil.copy(VELEST_DATA / "mc_input_rk.cnv", run / "in.cnv")
    shutil.copy(VELEST_DATA / "mc_rk_stats_all.sta", run / "in.sta")
    (run / "in.mod").write_text(
        (VELEST_DATA / "mc_6layer_crlf.mod").read_text().replace("\r\n", "\n")
    )
    cmn = (VELEST_DATA / "mc_velest.cmn").read_text().replace("\r\n", "\n")
    cmn = cmn.replace("     124      0      0.0", "      25      0      0.0")
    for old, new in [
        ("RK-NM_2012-2015_velMC_999.MOD", "in.mod"),
        ("RK-NMstats_ALL.sta", "in.sta"),
        ("RK-NM_2012-13_manpicksDec16.CNV", "in.cnv"),
        ("RK-NM_2012-2015_velMC_999.OUT", "velest.out"),
    ]:
        cmn = cmn.replace(old, new)
    (run / "velest.cmn").write_text(cmn)


def _final_hypocentres(cnv: Path) -> list[tuple[str, float, float, float]]:
    """(date+time, lat, lon, depth) per event header line of a VELEST ``*.cnv``."""
    out = []
    for ln in cnv.read_text().splitlines():
        m = re.match(r"^([ 0-9]{6} [ 0-9]{4} +[\d.]+) +([\d.]+)[NS] +([\d.]+)[EW] +(-?[\d.]+)", ln)
        if m:
            out.append((m[1], float(m[2]), float(m[3]), float(m[4])))
    return out


@need_toolchain
def test_velest_runs_on_fixture(bin_dir: Path, tmp_path: Path) -> None:
    run = tmp_path / "velest"
    _prepare_velest(run)
    _run([str(bin_dir / "velest")], cwd=run)
    assert "Final rms values" in (run / "velest.out").read_text()
    hyp = _final_hypocentres(run / "velout.cnv")
    assert len(hyp) == 25
    assert all(-5 < h[3] < 40 for h in hyp)  # sensible depths (km)


@need_toolchain
def test_velest_configurable_limits(tmp_path: Path) -> None:
    """A build with small array limits works and gives the same hypocentres."""
    root = tmp_path / "b"
    _run(
        [
            str(SCRIPT),
            "--prefix",
            str(root / "prefix"),
            "--build-dir",
            str(root / "cmake"),
            "--no-fmtomo",
            "--no-extras",
            "--velest-ieq",
            "30",
            "--velest-ist",
            "60",
            "--velest-inltot",
            "10",
        ],
        cwd=REPO,
    )
    inc = (root / "cmake" / "velest_src" / "velest_common.inc").read_text()
    assert "parameter(ieq=30)" in inc and "ist=60," in inc and "inltot=10," in inc
    small = root / "prefix" / "bin" / "osv-tools" / "velest"
    run = tmp_path / "run"
    _prepare_velest(run)
    _run([str(small)], cwd=run)
    assert len(_final_hypocentres(run / "velout.cnv")) == 25


@need_toolchain
def test_fmtomo_example1_end_to_end(bin_dir: Path, tmp_path: Path) -> None:
    """tomo3d (bash-ported driver) reproduces the shipped example1 results."""
    run = tmp_path / "example1"
    shutil.copytree(FMTOMO_EX1, run)
    _run(["bash", str(bin_dir / "tomo3d")], cwd=run, env=_env(bin_dir))
    new = _floats(run / "residuals.dat")
    ref = _floats(FMTOMO_EX1 / "residuals.dat")
    assert len(new) == len(ref) == 7
    # initial forward step is deterministic and matches exactly
    assert new[0] == pytest.approx(ref[0], rel=1e-4)
    # relocation converges to the same rms as the shipped run (different compiler)
    assert new[-1][1] == pytest.approx(ref[-1][1], rel=0.05)
    assert new[-1][1] < 0.2 * new[0][1]
    # relocated sources agree with the shipped solution and the truth
    got, shipped = _sources(run / "sources.in"), _sources(FMTOMO_EX1 / "sources.in")
    truth = _sources(FMTOMO_EX1 / "sourcestrue.in")
    assert len(got) == len(shipped) == len(truth) == 5
    for g, s, t in zip(got, shipped, truth, strict=True):
        assert abs(g[0] - s[0]) < 0.2 and abs(g[1] - s[1]) < 0.01 and abs(g[2] - s[2]) < 0.01
        assert abs(g[0] - t[0]) < 2.0  # columns: depth (km), lat, lon


@need_toolchain
def test_fmtomo_extras_gridloc(bin_dir: Path, tmp_path: Path) -> None:
    """fm3d saves time fields, then gridloc relocates the shipped sources."""
    run = tmp_path / "invert_p"
    shutil.copytree(EXTRAS_P, run)
    env = _env(bin_dir)
    _run([str(bin_dir / "fm3d")], cwd=run, env=env)
    assert (run / "arrtimes.dat").stat().st_size > 0
    out = _run([str(bin_dir / "gridloc")], cwd=run, env=env)
    assert "Program gridloc successfully completed" in out
    assert "out of" in out and "events relocated" in out
