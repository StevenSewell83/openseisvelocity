"""Inversion loop (``osv.fmtomo.invert``): control flow with dummy tools, equivalence to tomo3d."""

from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

from osv.config.schema import FmtomoConfig
from osv.fmtomo import invert as inv
from osv.fmtomo.controls import ModelInfo, build_controls
from osv.fmtomo.forward import Fm3dResult
from osv.fmtomo.invert import InversionError, run_inversion
from osv.fmtomo.rundir import assemble_run_dir
from osv.io.fmtomo import (
    read_interfaces,
    read_observed,
    read_propgrid,
    read_receivers,
    read_sources,
    read_vgrids,
)
from osv.io.fmtomo.times import write_arrivals

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "tests" / "data" / "fmtomo" / "example1"
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))
WORK = Path(os.environ.get("OSV_TEST_WORKDIR", REPO / "build" / "test-runs" / "osv_lg52"))


def _ex1() -> dict:
    return {
        "grids": read_vgrids(E1 / "vgrids.in"),
        "interfaces": read_interfaces(E1 / "interfaces.in"),
        "propgrid": read_propgrid(E1 / "propgrid.in"),
        "sources": read_sources(E1 / "sourcesref.in"),
        "receivers": read_receivers(E1 / "receivers.in"),
        "otimes": read_observed(E1 / "otimes.dat"),
    }


def _cfg(**kw) -> FmtomoConfig:
    base = {
        "package": "base",
        "relocate": "linear",
        "damping": 1.0,
        "smoothing": 0.0,
        "invert_velocity": False,
        "iterations": 2,
    }
    base.update(kw)
    return FmtomoConfig(**base)


def _assemble(root: Path, cfg: FmtomoConfig):
    ex = _ex1()
    ctl = build_controls(cfg, ModelInfo.from_model(ex["grids"]))
    assemble_run_dir(
        root,
        ex["grids"],
        ex["interfaces"],
        ex["propgrid"],
        ex["sources"],
        ex["receivers"],
        ex["otimes"],
        ctl,
    )
    return ctl


# ---------------------------------------------------------------------------------------
# dummy tools
# ---------------------------------------------------------------------------------------
def _script(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """Dummy frechgen/invert3d/residuals/scorrect; ``rms`` file holds the residual sequence."""
    d = tmp_path / "tools"
    d.mkdir()
    _script(d / "frechgen", "echo 1 > frechet.in\n")
    _script(
        d / "invert3d",
        'if [ -f FAIL_INV ]; then echo "Fortran runtime error: boom"; exit 2; fi\n'
        'if [ -f NAN_MODEL ]; then sed -i "100 s/^ *[^ ]*/ nan/" vgrids.in; fi\n'
        'echo " Due to redundancy, the subspace dimension"\n'
        'echo " is being reduced from           20  to           15"\n'
        'echo " PROGRAM invert successfully completed!!"\n',
    )
    _script(d / "scorrect", "echo scorrect >> scorrect.calls\n")
    _script(
        d / "residuals",
        "n=$(cat resid.count 2>/dev/null || echo 0)\n"
        'v=$(sed -n "$((n+1))p" ../rms_seq)\n'
        "echo $((n+1)) > resid.count\n"
        'printf "%10.2f  %10.5f  %10.5f\\n" $v 0.01 1.0\n',
    )
    monkeypatch.setenv("OSV_TOOLS_DIR", str(d))
    return d


@pytest.fixture
def fake_forward(monkeypatch):
    calls = []

    def fake(inputs, sources, receivers, n_jobs, mem=None, workdir=None, **kw):
        calls.append((Path(inputs), len(sources), n_jobs))
        w = Path(workdir)
        w.mkdir(parents=True, exist_ok=True)
        obs = read_observed(Path(inputs) / "otimes.dat")
        arr = obs.assign(subray=0, diffracted=False, headwave=False)
        write_arrivals(w / "arrivals.dat", arr)
        return Fm3dResult(w, w / "arrivals.dat", None, None, n_jobs, [[1]], 0.0)

    monkeypatch.setattr(inv, "run_fm3d_parallel", fake)
    return calls


def _setup(tmp_path, rms, cfg=None):
    run = tmp_path / "run" / "d"
    ctl = _assemble(run, cfg or _cfg())
    (tmp_path / "run" / "rms_seq").write_text("\n".join(str(v) for v in rms) + "\n")
    return run, ctl


def test_loop_order_and_capture(tmp_path, tools, fake_forward):
    run, ctl = _setup(tmp_path, [300, 200, 100])
    res = run_inversion(run, ctl, 2, n_jobs=3)
    assert res.success
    assert list(res.iterations["iter"]) == [0, 1, 2]
    assert list(res.iterations["rms_ms"]) == [300.0, 200.0, 100.0]
    assert (run / "residuals.dat").read_text() == "300.00 0.01000 1.00000\n" * 0 + (
        "300.00 0.01000 1.00000\n200.00 0.01000 1.00000\n100.00 0.01000 1.00000\n"
    )
    assert len(fake_forward) == 3 and fake_forward[0][2] == 3
    assert (run / "inviter.in").read_text().split() == ["3"]
    assert (run / "rtimes.dat").exists() and (run / "mtimes.dat").exists()
    assert len((run / "scorrect.calls").read_text().split()) == 2
    for k in range(3):
        d = run / "iterations" / f"{k:02d}"
        assert (d / "summary.json").exists() and (d / "vgrids.in").exists()
        assert (d / "sources.in").exists()
    assert (run / "iterations" / "01" / "invert3d.log").exists()
    assert res.iterations["subspace_dim"].iloc[1] == 15
    assert res.iterations["n_lost"].iloc[0] == 0
    assert set(res.hypocentres["iter"]) == {0, 1, 2}
    assert res.final_model == run / "vgrids.in"
    assert res.flags == []


def test_keep_iterations_false(tmp_path, tools, fake_forward):
    run, ctl = _setup(tmp_path, [300, 200])
    run_inversion(run, ctl, 1, 1, keep_iterations=False)
    d = run / "iterations" / "01"
    assert (d / "summary.json").exists() and not (d / "vgrids.in").exists()


def test_residual_increase_flagged(tmp_path, tools, fake_forward):
    run, ctl = _setup(tmp_path, [100, 150])
    res = run_inversion(run, ctl, 1, 1)
    assert "residual_increase" in res.iterations["flags"].iloc[1]
    assert any("residual_increase" in f for f in res.flags)


def test_invert3d_failure(tmp_path, tools, fake_forward):
    run, ctl = _setup(tmp_path, [100, 90])
    (run / "FAIL_INV").write_text("")
    with pytest.raises(InversionError, match="invert3d failed") as ei:
        run_inversion(run, ctl, 1, 1)
    part = ei.value.result
    assert part is not None and not part.success and len(part.iterations) == 1
    res = run_inversion(run, ctl, 1, 1, stop_on_error=False)
    assert not res.success and res.flags[-1].startswith("error")


def test_forward_failure(tmp_path, tools, monkeypatch):
    def boom(*a, **k):
        raise inv.Fm3dError("fm3d exploded")

    monkeypatch.setattr(inv, "run_fm3d_parallel", boom)
    run, ctl = _setup(tmp_path, [100])
    with pytest.raises(InversionError, match="fm3d exploded"):
        run_inversion(run, ctl, 1, 1)


def test_nan_model_stops(tmp_path, tools, fake_forward):
    run, ctl = _setup(tmp_path, [100, 90])
    (run / "NAN_MODEL").write_text("")
    with pytest.raises(InversionError, match="NaN"):
        run_inversion(run, ctl, 1, 1)


def test_nonfinite_misfit(tmp_path, tools, fake_forward):
    run, ctl = _setup(tmp_path, ["NaN"])
    with pytest.raises(InversionError, match="non-finite"):
        run_inversion(run, ctl, 1, 1)


def test_missing_run_dir_files(tmp_path):
    with pytest.raises(InversionError, match="assemble"):
        run_inversion(tmp_path, build_controls(_cfg()), 1, 1)


def test_joint_not_supported(tmp_path):
    ctl = build_controls(
        _cfg(package="extras", relocate="nonlinear", invert_velocity=True), phases=["P", "S"]
    )
    with pytest.raises(NotImplementedError):
        run_inversion(tmp_path, ctl, 1, 1)


# ---------------------------------------------------------------------------------------
# equivalence with the installed driver (real programs)
# ---------------------------------------------------------------------------------------
@pytest.fixture
def workdir():
    """Scratch directory under the (git-ignored) build/ tree, removed after the test."""
    import shutil

    WORK.mkdir(parents=True, exist_ok=True)
    yield WORK
    shutil.rmtree(WORK, ignore_errors=True)
    try:
        WORK.parent.rmdir()  # only if empty
    except OSError:
        pass


def _need(*exes: str) -> None:
    for e in exes:
        if not (BIN / e).exists():
            pytest.skip(f"{BIN / e} not built")


@pytest.mark.integration
@pytest.mark.parametrize("invert_velocity", [False, True])
def test_equivalent_to_tomo3d_driver(invert_velocity, workdir):
    _need("tomo3d", "fm3d", "frechgen", "invert3d", "residuals", "scorrect")
    cfg = _cfg(invert_velocity=invert_velocity, subspace_dimension=15)
    base = workdir / f"equiv_{int(invert_velocity)}"
    import shutil

    shutil.rmtree(base, ignore_errors=True)
    ref, mine = base / "tomo3d", base / "osv"
    _assemble(ref, cfg)
    ctl = _assemble(mine, cfg)
    env = {**os.environ, "PATH": f"{BIN}{os.pathsep}{os.environ['PATH']}"}
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(BIN / "tomo3d")], cwd=ref, env=env, capture_output=True, text=True, check=False
    )
    t_ref = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr[-2000:]
    t0 = time.perf_counter()
    res = run_inversion(mine, ctl, 2, n_jobs=3)
    t_mine = time.perf_counter() - t0
    print(f"tomo3d driver {t_ref:.1f} s, run_inversion {t_mine:.1f} s")
    assert (mine / "residuals.dat").read_text() == (ref / "residuals.dat").read_text()
    for name in ("vgrids.in", "sources.in", "stimes.dat", "inviter.in", "frechet.in"):
        a, b = (ref / name).read_text().split(), (mine / name).read_text().split()
        assert len(a) == len(b), name
        diff = max((abs(float(x) - float(y)) for x, y in zip(a, b, strict=True)), default=0.0)
        assert diff < 1e-5, (name, diff)
    assert (ref / "arrivals.dat").read_text() == (mine / "arrivals.dat").read_text()
    assert list(res.iterations["iter"]) == [0, 1, 2]
    assert res.iterations["rms_ms"].is_monotonic_decreasing
    assert isinstance(res.hypocentres, pd.DataFrame) and len(res.hypocentres) == 3 * 5
    if invert_velocity:
        assert res.iterations["model_change_rms"].iloc[1] > 0


@pytest.mark.integration
def test_nonlinear_equivalent_to_tomo3di_driver(workdir):
    """Extras observed example (29 stations, 411 events): gridloc/fm3d/invert3d loop, 1 iteration.

    The installed ``tomo3di`` writes ``inviter.in`` from an unset ``COUNT`` before ``invert3d``
    (crashes in the bash port), so the reference run uses a copy with ``COUNT=ITER`` added.
    """
    import shutil

    src = REPO / "fortran" / "fmtomo_extras" / "observed" / "invert_p"
    _need("tomo3di", "fm3d", "frechgen", "invert3d", "residuals", "gridloc")
    if not src.is_dir():
        pytest.skip("Extras example not available")
    base = workdir / "equiv_nl"
    shutil.rmtree(base, ignore_errors=True)
    ref, mine = base / "tomo3di", base / "osv"
    for d in (ref, mine):
        shutil.copytree(src, d)
        (d / "tomo3di.in").write_text("1\n0\n0\n")
        txt = (d / "invert3d.in").read_text().replace("rtimesnec.dat", "rtimes.dat   ")
        (d / "invert3d.in").write_text(txt)
    script = (BIN / "tomo3di").read_text().replace("   $rel\n", "   COUNT=ITER\n   $rel\n", 1)
    fixed = base / "tomo3di_fixed"
    fixed.write_text(script)
    fixed.chmod(0o755)
    env = {**os.environ, "PATH": f"{BIN}{os.pathsep}{os.environ['PATH']}"}
    t0 = time.perf_counter()
    r = subprocess.run([str(fixed)], cwd=ref, env=env, capture_output=True, text=True, check=False)
    t_ref = time.perf_counter() - t0
    assert r.returncode == 0
    cfg = _cfg(package="extras", relocate="nonlinear", invert_velocity=True, iterations=1)
    ctl = build_controls(cfg, ModelInfo(n_layers=1))
    t0 = time.perf_counter()
    res = run_inversion(mine, ctl, 1, n_jobs=os.cpu_count() or 4)
    t_mine = time.perf_counter() - t0
    print(f"tomo3di driver {t_ref:.1f} s, run_inversion {t_mine:.1f} s")
    assert (mine / "residuals.dat").read_text() == (ref / "residuals.dat").read_text()
    for name in ("vgrids.in", "receivers.in", "otimes.dat", "arrivals.dat"):
        assert (mine / name).read_bytes() == (ref / name).read_bytes(), name
    assert list(res.iterations["iter"]) == [0, 1]
    assert res.iterations["rms_reloc_ms"].iloc[1] < res.iterations["rms_ms"].iloc[0]
    assert len(res.hypocentres) > 0
