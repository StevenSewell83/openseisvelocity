"""End-to-end checks: files written by osv.io.fmtomo are accepted by the real Fortran programs.

Marked ``integration``; skipped when the FMTOMO binaries are not built (``build/bin`` of the
repository or the directory in ``OSV_FMTOMO_BIN``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.io.fmtomo import controls as c
from osv.io.fmtomo import datagen as d
from osv.io.fmtomo import frechet as fr
from osv.io.fmtomo import grid3dg as g3
from osv.io.fmtomo import grids, rays, sources, times

REPO = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))
E1 = REPO / "tests" / "data" / "fmtomo" / "example1"

pytestmark = pytest.mark.integration


def _run(name: str, cwd: Path, timeout: int = 100) -> subprocess.CompletedProcess:
    exe = BIN / name
    if not exe.exists():
        pytest.skip(f"{exe} not built")
    return subprocess.run(
        [str(exe)], cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


def test_grid3dg_accepts_written_control_file_and_reproduces_example(tmp_path):
    orig, mine = tmp_path / "orig", tmp_path / "mine"
    orig.mkdir()
    mine.mkdir()
    shutil.copy(E1 / "mkmodel" / "grid3dg.in", orig / "grid3dg.in")
    g3.Grid3dg.read(E1 / "mkmodel" / "grid3dg.in").write(mine / "grid3dg.in")
    for d_ in (orig, mine):
        _run("grid3dg", d_)
    for name in ("vgrids.in", "interfaces.in", "propgrid.in"):
        assert (mine / name).read_text() == (orig / name).read_text(), name
    # the generated model is the one shipped as example 1 (the seeded random noise differs by
    # ~1e-6 between compilers, the header text by the compiler's list-directed formatting)
    ref = grids.read_vgrids(E1 / "vgrids.in")
    new = grids.read_vgrids(mine / "vgrids.in")
    for (_, _, a), (_, _, b) in zip(ref, new, strict=True):
        np.testing.assert_allclose(a["velocity"].values, b["velocity"].values, atol=1e-4)
        assert a.attrs["fmtomo_r0_km"] == pytest.approx(b.attrs["fmtomo_r0_km"], rel=1e-12)
    grids.validate_model(
        new,
        grids.read_interfaces(mine / "interfaces.in"),
        grids.read_propgrid(mine / "propgrid.in"),
    )


def test_fm3d_runs_on_osv_written_inputs_and_reproduces_arrivals(tmp_path):
    vg = grids.read_vgrids(E1 / "vgrids.in")
    itf = grids.read_interfaces(E1 / "interfaces.in")
    pg = grids.read_propgrid(E1 / "propgrid.in")
    src = sources.read_sources(E1 / "sources.in")
    rec = sources.read_receivers(E1 / "receivers.in")
    grids.write_vgrids(tmp_path / "vgrids.in", vg)
    grids.write_interfaces(tmp_path / "interfaces.in", itf)
    grids.write_propgrid(tmp_path / "propgrid.in", pg)
    sources.write_sources(tmp_path / "sources.in", src)
    sources.write_receivers(tmp_path / "receivers.in", rec)
    c.ModeSet(save_rays_mode=True).write(tmp_path / "mode_set.in")
    c.FrechetSpec(True, [], [], [1, 2, 3, 4, 5]).write(tmp_path / "frechet.in")
    res = _run("fm3d", tmp_path)
    assert res.returncode == 0, res.stderr[-500:]
    arr = times.read_arrivals(tmp_path / "arrivals.dat")
    ref = times.read_arrivals(E1 / "arrivals.dat")
    pd.testing.assert_frame_equal(arr, ref)
    rp = rays.read_rays(tmp_path / "rays.dat")
    assert len(rp.rays) == 500 and int(rp.rays["valid"].sum()) == int(ref["valid"].sum())
    f = fr.read_frechet(tmp_path / "frechet.dat")
    assert f.matrix.shape == (500, 20)
    assert list(f.rays["n_deriv"] == 0) == list(~ref["valid"])
    # a valid ray starts at its source and ends at its receiver
    first_valid = rp.rays.index[rp.rays["valid"]][0]
    pts = rp.points[rp.points["ray"] == first_valid]
    s = src[int(rp.rays.loc[first_valid, "source"]) - 1]
    r = rec.positions.loc[int(rp.rays.loc[first_valid, "receiver"])]
    assert (pts.iloc[0].depth_km, pts.iloc[0].lat, pts.iloc[0].lon) == pytest.approx(
        (s.depth_km, s.lat, s.lon), abs=1e-5
    )
    assert (pts.iloc[-1].depth_km, pts.iloc[-1].lat, pts.iloc[-1].lon) == pytest.approx(
        (r.depth_km, r.lat, r.lon), abs=1e-5
    )


def test_frechgen_and_residuals_accept_written_control_files(tmp_path):
    for name in ("sourcesref.in", "otimes.dat", "rtimes.dat", "sources.in", "receivers.in"):
        shutil.copy(E1 / name, tmp_path / name)
    shutil.copy(E1 / "arrivals.dat", tmp_path / "mtimes.dat")
    shutil.copy(E1 / "vgrids.in", tmp_path / "vgridsref.in")
    c.Invert3d.read(E1 / "invert3d.in").write(tmp_path / "invert3d.in")
    c.Frechgen.read(E1 / "frechgen.in").write(tmp_path / "frechgen.in")
    c.Residuals.read(E1 / "residuals.in").write(tmp_path / "residuals.in")
    res = _run("frechgen", tmp_path)
    assert res.returncode == 0, res.stderr[-300:]
    assert c.FrechetSpec.read(tmp_path / "frechet.in") == c.FrechetSpec.read(E1 / "frechet.in")
    res = _run("residuals", tmp_path)
    assert res.returncode == 0, res.stderr[-300:]
    df = times.parse_residuals(res.stdout)
    assert len(df) == 1 and df.loc[0, "rms_ms"] > 0


def test_moddata_accepts_written_control_file(tmp_path):
    mk = E1 / "mkmodel"
    for name in ("sourcesle.in", "arraygen.out"):
        shutil.copy(mk / name, tmp_path / name)
    d.Moddata.read(mk / "moddata.in").write(tmp_path / "moddata.in")
    res = _run("moddata", tmp_path)
    assert res.returncode == 0, res.stderr[-300:]
    src = sources.read_sources(tmp_path / "sources.in")
    assert len(src) == 5 and src[0].uncertainty == (2.0, 2.0, 2.0)
    assert src[0].paths[0] == sources.PathSpec.reflected(2)
    rec = sources.read_receivers(tmp_path / "receivers.in")
    assert len(rec.positions) == 100 and len(rec.rays) == 500
    assert list(times.read_source_ids(tmp_path / "sourcederivs.in")) == [1, 2, 3, 4, 5]
    ref = sources.read_receivers(E1 / "receivers.in")
    pd.testing.assert_frame_equal(rec.positions, ref.positions, atol=1e-4)


def test_arraygen_accepts_written_control_file(tmp_path):
    d.Arraygen.read(E1 / "mkmodel" / "arraygen.in").write(tmp_path / "arraygen.in")
    res = _run("arraygen", tmp_path)
    assert res.returncode == 0
    pts = d.parse_input_receivers((tmp_path / "arraygen.out").read_text())
    want = d.Arraygen.read(E1 / "mkmodel" / "arraygen.in").points()
    np.testing.assert_allclose(
        pts[["lat", "lon", "depth_km"]].to_numpy(), want.to_numpy(), atol=1e-4
    )


def test_invert3d_outputs_are_readable(tmp_path):
    """One real invert3d step (sources only): its list-directed outputs parse with our readers."""
    for name in (
        "sourcesref.in",
        "sources.in",
        "stimes.dat",
        "receivers.in",
        "otimes.dat",
        "rtimes.dat",
        "frechet.in",
        "frechet.dat",
        "inviter.in",
        "propgrid.in",
        "interfaces.in",
        "vgrids.in",
    ):
        shutil.copy(E1 / name, tmp_path / name)
    shutil.copy(E1 / "arrivals.dat", tmp_path / "mtimes.dat")
    shutil.copy(E1 / "vgrids.in", tmp_path / "vgridsref.in")
    shutil.copy(E1 / "interfaces.in", tmp_path / "interfacesref.in")
    c.Invert3d.read(E1 / "invert3d.in").write(tmp_path / "invert3d.in")
    before = sources.read_sources(tmp_path / "sources.in")
    res = _run("invert3d", tmp_path)
    assert res.returncode == 0, res.stderr[-300:]
    after = sources.read_sources(tmp_path / "sources.in")
    assert len(after) == 5 and after[0].paths == before[0].paths
    assert (after[0].depth_km, after[0].lat) != (before[0].depth_km, before[0].lat)  # relocated
    st = times.read_stimes(tmp_path / "stimes.dat")
    assert st.shape == (5,)
    # with only the source inversion switched on invert3d leaves the model files untouched
    vg = grids.read_vgrids(tmp_path / "vgrids.in")
    assert vg.get(1)["velocity"].shape == (14, 14, 14)
    assert grids.read_interfaces(tmp_path / "interfaces.in").n_interfaces == 3
