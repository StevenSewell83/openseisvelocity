"""P+S, joint and Vp/Vs inversions (``osv.fmtomo.joint``): unit tests and driver equivalence.

The integration tests run the shipped Extras scripts (``tomo3dj``, ``cp_script`` +
``tomo3dps``) on the Extras ``observed`` example and compare the files of the Python loops
byte for byte.  The installed ``tomo3dj`` writes ``inviter.in`` from an unset ``COUNT`` before
the first ``invert3d``, so the reference uses a copy with ``COUNT=ITER`` added.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from osv.config.schema import FmtomoConfig
from osv.fmtomo.controls import ModelInfo, build_controls
from osv.fmtomo.joint import (
    InversionError,
    prepare_vpvs_dir,
    run_joint_inversion,
    run_ps_inversions,
    run_vpvs_inversion,
    transfer_hypocentres,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "fortran" / "fmtomo_extras" / "observed"
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))
WORK = Path(os.environ.get("OSV_TEST_WORKDIR", REPO / "build" / "test-runs" / "osv_lg55"))


def _controls(phases, relocate="nonlinear", **kw):
    cfg = FmtomoConfig(
        package="extras",
        relocate=relocate,
        phases=phases,
        invert_velocity=True,
        iterations=1,
        **kw,
    )
    return build_controls(cfg, ModelInfo(n_layers=1))


@pytest.fixture
def workdir():
    """Unique scratch directory under build/ (concurrent pytest runs never share one)."""
    WORK.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="run-", dir=WORK))
    yield d
    shutil.rmtree(d, ignore_errors=True)
    for p in (WORK, WORK.parent):
        try:
            p.rmdir()
        except OSError:
            pass


def _need(*exes: str) -> None:
    for e in exes:
        if not (BIN / e).exists():
            pytest.skip(f"{BIN / e} not built")
    if not EXAMPLE.is_dir():
        pytest.skip("Extras example not available")


@pytest.fixture
def tools_env(monkeypatch):
    monkeypatch.setenv("OSV_TOOLS_DIR", str(BIN))
    return {
        **os.environ,
        "OSV_TOOLS_DIR": str(BIN),
        "PATH": f"{BIN}{os.pathsep}{os.environ['PATH']}",
    }


def _example_copy(dst: Path, tomo3dj: str = "1\n0\n0\n") -> Path:
    shutil.copytree(EXAMPLE, dst)
    (dst / "tomo3dj.in").write_text(tomo3dj)
    for d in ("invert_p", "invert_s"):
        (dst / d / "tomo3di.in").write_text("1\n0\n0\n")
        f = dst / d / "invert3d.in"
        f.write_text(f.read_text().replace("rtimesnec.dat", "rtimes.dat   "))
    return dst


# ---------------------------------------------------------------------------------------
# unit tests (no binaries)
# ---------------------------------------------------------------------------------------
def _write_phase(d: Path, events: list[tuple[float, float, float]], ref_t: float, cur_t: float):
    """One station, one pick per event; ``receivers.in`` current = ``events`` (depth, lat, lon)."""
    d.mkdir(parents=True)
    n = len(events)
    (d / "src.dat").write_text(f"1\n1 {n}\n" + "".join(f"{i + 1} {ref_t} 0.1\n" for i in range(n)))
    ref = f"{n}\n" + "".join(f"{e[0]} {e[1]} {e[2]}\n1\n1\n1\n" for e in events)
    (d / "receiversref.in").write_text(ref)
    shifted = [(e[0] + 1.0, e[1] + 0.1, e[2] + 0.2) for e in events]
    (d / "receivers.in").write_text(
        f"{n}\n" + "".join(f"{e[0]} {e[1]} {e[2]}\n1\n1\n1\n" for e in shifted)
    )
    ot = lambda t: f"{n}\n" + "".join(f"{i + 1} 1 1 0 {t} 0.1\n" for i in range(n))  # noqa: E731
    (d / "otimesref.dat").write_text(ot(ref_t))
    (d / "otimes.dat").write_text(ot(cur_t))


def test_transfer_hypocentres(tmp_path):
    ev = [(5.0, -31.0, 138.0), (6.0, -32.0, 139.0)]
    _write_phase(tmp_path / "invert_p", ev, 10.0, 9.5)
    # S: events 2 and 3 (3 is S-only); the association file marks S pick 1 = event 2
    s = tmp_path / "invert_s"
    _write_phase(s, [ev[1], (7.0, -33.0, 140.0)], 20.0, 20.0)
    (tmp_path / "src.dat").write_text("2\n1 2\n1 10.0 0.1\n2 10.0 0.1\n2 1\n2 20.0 0.1\n")
    # records: P station (2 picks), S station (1 pick: event 2) -> S file must then hold 1 pick
    (s / "src.dat").write_text("1\n1 1\n2 20.0 0.1\n")
    (s / "receiversref.in").write_text("1\n6.0 -32.0 139.0\n1\n1\n1\n")
    (s / "otimesref.dat").write_text("1\n1 1 1 0 20.0 0.1\n")
    n = transfer_hypocentres(tmp_path, tmp_path / "invert_p", s)
    assert n == 1
    lines = (s / "receiversref.in").read_text().split("\n")
    assert [float(x) for x in lines[1].split()] == [7.0, -31.9, 139.2]
    assert float(((s / "otimesref.dat").read_text().split("\n")[1]).split()[4]) == pytest.approx(
        19.5
    )
    assert (s / "receiversref.in.orig").exists()


def test_joint_needs_joint_controls(tmp_path):
    with pytest.raises(InversionError, match="joint"):
        run_joint_inversion(tmp_path, _controls(["P"], "nonlinear"))


def test_joint_missing_files(tmp_path):
    with pytest.raises(InversionError, match="missing"):
        run_joint_inversion(tmp_path, _controls(["P", "S"]))


def test_ps_bad_mode(tmp_path):
    with pytest.raises(ValueError, match="relocation"):
        run_ps_inversions(tmp_path / "a", tmp_path / "b", _controls(["P", "S"]), relocation="x")


def test_prepare_vpvs_dir_missing(tmp_path):
    with pytest.raises(InversionError, match="missing"):
        prepare_vpvs_dir(tmp_path)


# ---------------------------------------------------------------------------------------
# equivalence with the installed drivers
# ---------------------------------------------------------------------------------------
def _src_tokens(path: Path) -> list[tuple[str, float]]:
    # the shipped file was written with another compiler: compare ids, values to REAL*4 digits
    return [(t, 0.0) if t.isdigit() else ("", round(float(t), 5)) for t in path.read_text().split()]


def _same(a: Path, b: Path, names) -> None:
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


@pytest.mark.integration
def test_joint_equivalent_to_tomo3dj(workdir, tools_env):
    """Extras observed example (29 + 28 stations, 411 events), 1 iteration of tomo3dj."""
    _need("tomo3dj", "fm3d", "frechgen", "invert3d", "residuals", "gridlocj", "jointimes")
    ref, mine = _example_copy(workdir / "tomo3dj"), _example_copy(workdir / "osv")
    script = (BIN / "tomo3dj").read_text().replace("   $rel\n", "   COUNT=ITER\n   $rel\n", 1)
    fixed = workdir / "tomo3dj_fixed"
    fixed.write_text(script)
    fixed.chmod(0o755)
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(fixed)], cwd=ref, env=tools_env, capture_output=True, text=True, check=False
    )
    t_ref = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr[-2000:]
    src_ref = _src_tokens(ref / "src.dat")
    (mine / "src.dat").unlink()  # let run_joint_inversion run jointimes
    t0 = time.perf_counter()
    res = run_joint_inversion(mine, _controls(["P", "S"]), 1, n_jobs=min(12, os.cpu_count() or 4))
    t_mine = time.perf_counter() - t0
    print(f"tomo3dj driver {t_ref:.1f} s, run_joint_inversion {t_mine:.1f} s")
    assert _src_tokens(mine / "src.dat") == src_ref  # jointimes reproduces the shipped file
    for d in ("invert_p", "invert_s"):
        _same(
            ref / d,
            mine / d,
            (
                "residuals.dat",
                "vgrids.in",
                "receivers.in",
                "otimes.dat",
                "arrivals.dat",
                "inviter.in",
            ),
        )
    assert list(res.iterations["iter"]) == [0, 1]
    assert res.success
    assert (res.iterations["rms_reloc_p_ms"].iloc[1]) < res.iterations["rms_p_ms"].iloc[0]
    assert res.iterations["rms_p_ms"].iloc[-1] < res.iterations["rms_p_ms"].iloc[0]
    assert len(res.p.hypocentres) > 0 and len(res.s.hypocentres) > 0
    assert (mine / "invert_p" / "iterations" / "01" / "summary.json").exists()


@pytest.mark.integration
def test_vpvs_equivalent_to_tomo3dps(workdir, tools_env):
    """cp_script + tomo3dps vs run_vpvs_inversion (1 iteration) on the Extras example."""
    _need("tomo3dps", "cp_script", "fm3d", "mkvpvs", "frvpvs", "vpvsinv", "vpvsresids")
    ref, mine = _example_copy(workdir / "ref"), _example_copy(workdir / "osv")
    for d in (ref, mine):
        (d / "vpvs" / "tomo3dps.in").write_text("1\n0\n0\n")
    # the S rays of the current S model (the shipped example has no rays.dat)
    subprocess.run(["fm3d"], cwd=ref / "invert_s", env=tools_env, capture_output=True, check=True)
    for cmd in ("cp_script", "tomo3dps"):
        r = subprocess.run(
            [cmd], cwd=ref / "vpvs", env=tools_env, capture_output=True, text=True, check=False
        )
        assert r.returncode == 0, r.stderr[-2000:]
    ctl = _controls(["vpvs"], "none")
    res = run_vpvs_inversion(mine, ctl, 1, forward=True, n_jobs=min(12, os.cpu_count() or 4))
    _same(
        ref / "vpvs",
        mine / "vpvs",
        ("vpvsresids.out", "vgrids.in", "vgridvpvs.in", "mtimes.dat", "frechet.dat", "inviter.in"),
    )
    assert res.success and list(res.iterations["iter"]) == [0, 1]
    assert res.final_rms_s < res.initial_rms_s
    assert res.iterations["vpvs_mean"].iloc[0] == pytest.approx(
        res.iterations["vpvs_min"].iloc[0]
    )  # constant start
    assert res.ratio_of_vp_vs_model is not None and res.ratio_of_vp_vs_model.exists()


@pytest.mark.integration
def test_ps_p_fixed_for_s(workdir, tools_env):
    """P relocates, the S run keeps the transferred P hypocentres (1 iteration)."""
    _need("fm3d", "frechgen", "invert3d", "residuals", "gridloc", "jointimes")
    root = _example_copy(workdir / "ps")
    res = run_ps_inversions(
        root / "invert_p",
        root / "invert_s",
        _controls(["P", "S"]),
        1,
        n_jobs=min(12, os.cpu_count() or 4),
        relocation="p_fixed_for_s",
    )
    assert res.success and res.transferred_events > 0
    assert res.p.iterations["mean_hypo_shift_km"].iloc[1] > 0  # P relocated
    assert (res.s.iterations["mean_hypo_shift_km"] == 0).all()  # S hypocentres fixed
    s = root / "invert_s"
    assert (s / "receivers.in").read_text().split() == (s / "receiversref.in").read_text().split()
