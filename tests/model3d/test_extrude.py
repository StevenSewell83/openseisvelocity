"""Tests for osv.model3d.extrude (1-D model extrusion)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import ExtrusionUncertaintyConfig, GridConfig, Model3DConfig
from osv.coords import spherical_distance_km
from osv.io.fmtomo import grids as fg
from osv.model3d.extrude import (
    ExtrusionError,
    effective_profile,
    extrude_1d,
    extrusion_to_vgrids,
    profile_1d,
)
from osv.model3d.grid import define_grid

REPO = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))
RUN01 = REPO / "projects" / "rk-nm-2017" / "runs" / "01_prep"
RUN03 = REPO / "projects" / "rk-nm-2017" / "runs" / "03_velest_analyse"

MODEL = pd.DataFrame(
    {
        "depth_top_km": [-0.6, 0.2, 1.0, 3.0],
        "vp": [1.0, 3.0, 4.0, 6.0],
        "vpvs": [1.7] * 4,
        "vp_p05": [0.8, 2.6, 3.8, 5.0],
        "vp_p95": [1.2, 3.4, 4.2, 6.4],
    }
)


@pytest.fixture
def grid():
    lat0, lon0 = -38.5, 176.2
    la, lo = np.meshgrid(np.linspace(0, 0.1, 3), np.linspace(0, 0.12, 3), indexing="ij")
    st = pd.DataFrame(
        {
            "lat": lat0 + la.ravel(),
            "lon": lon0 + lo.ravel(),
            "elevation_m": np.linspace(300.0, 600.0, la.size),
        }
    )
    cfg = Model3DConfig(
        grid=GridConfig(dx_km=1.0, dy_km=1.0, dz_km=0.5, pad_km=2.0, zmax_km=6.0),
        propgrid_refine=2,
    )
    return define_grid(st, None, cfg)


def test_layered_is_step_function(grid):
    ds = extrude_1d(MODEL, grid, mode="layered", vpvs=1.7)
    z = grid.depth_full
    prof = ds["vp"].isel(lat=3, lon=3).to_numpy()
    for zi, vi in zip(z, prof, strict=True):
        layer = np.searchsorted(MODEL.depth_top_km, zi + 1e-9, side="right") - 1
        assert vi == MODEL.vp[max(layer, 0)]
    assert prof[0] == 1.0  # above the first top: first layer extended
    assert prof[-1] == 6.0
    assert ds["vs"].isel(lat=1, lon=1, depth=5).item() == pytest.approx(
        ds["vp"].isel(lat=1, lon=1, depth=5).item() / 1.7
    )
    assert ds["vp"].dims == ("depth", "lat", "lon")


def test_layered_node_on_top_belongs_to_layer_below():
    v = profile_1d(np.array([0.2]), np.array([-0.6, 0.2]), np.array([1.0, 3.0]), mode="layered")
    assert v[0] == 3.0


def test_gradient_interpolates_mid_depths(grid):
    ds = extrude_1d(MODEL, grid, mode="gradient")
    mids = [-0.2, 0.6, 2.0, 3.0 + 0.5 * 2.0]  # last layer borrows previous thickness
    prof = ds["vp"].isel(lat=2, lon=2).to_numpy()
    assert np.allclose(prof, np.interp(grid.depth_full, mids, MODEL.vp))
    assert prof[0] == 1.0 and prof[-1] == 6.0  # constant extension at both ends
    assert np.all(np.diff(prof) >= -1e-12)  # monotonic for a monotonic model
    # the vs column is derived from the vpvs column
    assert np.allclose(ds["vs"], ds["vp"] / 1.7)


def test_vs_sources():
    m = MODEL.drop(columns="vpvs")
    with pytest.raises(ExtrusionError, match="S model"):
        extrude_1d(m, _tiny(), vpvs=None)
    m2 = m.assign(vs=m.vp / 2.0)
    g = _tiny()
    assert np.allclose(extrude_1d(m2, g)["vs"], extrude_1d(m2, g)["vp"] / 2.0)
    assert np.allclose(extrude_1d(m2, g, vpvs=1.5)["vs"], extrude_1d(m2, g)["vp"] / 1.5)


def _tiny():
    st = pd.DataFrame(
        {"lat": [-38.5, -38.45], "lon": [176.2, 176.25], "elevation_m": [300.0, 500.0]}
    )
    cfg = Model3DConfig(
        grid=GridConfig(dx_km=1.0, dy_km=1.0, dz_km=0.5, pad_km=2.0, zmax_km=4.0),
        propgrid_refine=2,
    )
    return define_grid(st, None, cfg)


def test_smoothing_layered():
    z = np.arange(0, 10, 0.5)
    tops, vals = np.array([0.0, 5.0]), np.array([2.0, 4.0])
    sharp = profile_1d(z, tops, vals, mode="layered")
    for sm in (0.25, 0.5, 1.0):
        s = profile_1d(z, tops, vals, mode="layered", smoothing_km=sm)
        assert s.min() >= 2.0 - 1e-9 and s.max() <= 4.0 + 1e-9  # no overshoot
        assert np.all(np.diff(s) >= -1e-12)
        assert s[np.searchsorted(z, 5.0)] == pytest.approx(3.0, abs=0.05)  # step centre
        # edge values untouched far from the step
        assert s[0] == pytest.approx(2.0) and s[-1] == pytest.approx(4.0)
    # wider kernel, wider transition
    w = [
        np.ptp(
            np.flatnonzero(
                (profile_1d(z, tops, vals, mode="layered", smoothing_km=sm) > 2.2)
                & (profile_1d(z, tops, vals, mode="layered", smoothing_km=sm) < 3.8)
            )
        )
        for sm in (0.25, 1.0)
    ]
    assert w[1] > w[0]
    assert sharp[np.searchsorted(z, 5.0)] == 4.0


def test_fm3d_bspline_has_no_ringing():
    """The approximating B-spline of interpolate_velocity smooths a step but never overshoots."""
    x = np.arange(0, 10, 0.5)
    v = profile_1d(x, np.array([0.0, 5.0]), np.array([2.0, 4.0]), mode="layered")
    z = np.linspace(1.0, 8.5, 1501)
    e = effective_profile(x, v, z)
    assert np.nanmax(e) <= 4.0 + 1e-12 and np.nanmin(e) >= 2.0 - 1e-12
    # at a node the spline gives (v[i-1] + 4 v[i] + v[i+1]) / 6
    i = 6
    assert effective_profile(x, v, np.array([x[i]]))[0] == pytest.approx(
        (v[i - 1] + 4 * v[i] + v[i + 1]) / 6
    )
    assert np.isnan(effective_profile(x, v, np.array([x[0]]))[0])
    # a smooth (linear) profile is reproduced exactly
    lin = 1.0 + 0.3 * x
    assert np.allclose(effective_profile(x, lin, z), 1.0 + 0.3 * z)


def test_uncertainty_modes(grid):
    ds = extrude_1d(MODEL, grid, mode="layered", vpvs=1.7)
    assert np.all(ds["vp_unc"] == 0.3) and np.all(ds["vs_unc"] == 0.3)
    cfg = ExtrusionUncertaintyConfig(mode="spread", min_km_s=0.15, max_km_s=1.0)
    ds = extrude_1d(MODEL, grid, mode="layered", vpvs=1.7, uncertainty=cfg)
    u = ds["vp_unc"].isel(lat=0, lon=0).to_numpy()
    assert u[0] == pytest.approx(0.2) and u[-1] == pytest.approx(0.7)
    assert np.all(ds["vs_unc"] >= 0.15)
    with pytest.raises(ExtrusionError, match="p05"):
        extrude_1d(MODEL.drop(columns=["vp_p05"]), grid, uncertainty=cfg)


def test_vgrids_roundtrip_with_uncertainty(grid, tmp_path):
    ds = extrude_1d(MODEL, grid, mode="layered", vpvs=1.7)
    vg = extrusion_to_vgrids(grid, ds)
    assert vg.n_types == 2
    p = tmp_path / "vgridsref.in"
    fg.write_vgrids(p, vg)
    back = fg.read_vgrids(p)
    vp = back.get(1, 1)
    assert float(vp["uncertainty"].mean()) == pytest.approx(0.3)
    # profile through the written file matches the 1-D model at node depths
    col = vp["velocity"].isel(lat=3, lon=3).to_numpy()
    assert np.allclose(col, ds["vp"].isel(lat=3, lon=3).to_numpy(), atol=1e-7)


def test_bad_models(grid):
    with pytest.raises(ExtrusionError, match="depth_top_km"):
        extrude_1d(pd.DataFrame({"vp": [1.0]}), grid)
    with pytest.raises(ExtrusionError, match="positive"):
        extrude_1d(MODEL.assign(vp=[1, -3, 4, 6]), grid, vpvs=1.7)
    with pytest.raises(ExtrusionError, match="mode"):
        extrude_1d(MODEL, grid, mode="cubic", vpvs=1.7)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_fm3d_real_project_extrusion_and_surface(tmp_path, capsys):
    from osv.io.fmtomo import controls as c
    from osv.io.fmtomo import sources as so
    from osv.io.fmtomo import times
    from osv.model3d.interfaces import build_interfaces, place_sources
    from osv.tools import run_tool

    if not (BIN / "fm3d").exists():
        pytest.skip(f"{BIN / 'fm3d'} not built")
    if not (RUN01 / "stations.parquet").exists() or not (RUN03 / "final_model.csv").exists():
        pytest.skip("project runs 01_prep / 03_velest_analyse not available")
    st = pd.read_parquet(RUN01 / "stations.parquet")
    ev = pd.read_parquet(RUN01 / "events.parquet")
    model = pd.read_csv(RUN03 / "final_model.csv")
    cfg = Model3DConfig(propgrid_refine=2)
    g = define_grid(st, ev, cfg)
    findings: list[str] = []
    ds = extrude_1d(
        model, g, mode="layered", smoothing_km=0.25, uncertainty=cfg.extrusion.uncertainty
    )
    vg = extrusion_to_vgrids(g, ds)
    itf = build_interfaces(g, st, cfg, findings=findings)
    ev2 = place_sources(itf, g, ev, findings=findings)
    fg.validate_model(vg, itf, g.propgrid)
    fg.write_vgrids(tmp_path / "vgrids.in", vg)
    fg.write_interfaces(tmp_path / "interfaces.in", itf)
    fg.write_propgrid(tmp_path / "propgrid.in", g.propgrid)

    sel = ev2.sort_values("depth_km_bsl").iloc[[0, len(ev2) // 2, -1]]
    sts = st[st["used"]].iloc[::12].head(5)
    sources = [
        so.Source(float(r.depth_km_bsl), float(r.lat), float(r.lon), [so.PathSpec.direct()])
        for r in sel.itertuples()
    ]
    so.write_sources(tmp_path / "sources.in", sources)
    pos = pd.DataFrame(
        {
            "depth_km": -sts["sensor_elevation_m"].to_numpy() / 1000.0,
            "lat": sts["lat"].to_numpy(),
            "lon": sts["lon"].to_numpy(),
        },
        index=pd.RangeIndex(1, len(sts) + 1),
    )
    rays = pd.DataFrame(
        [(r, s, 1) for r in pos.index for s in range(1, len(sources) + 1)],
        columns=["receiver", "source", "path"],
    )
    rec = so.Receivers(pos, rays)
    so.validate_receivers(rec, sources)
    so.write_receivers(tmp_path / "receivers.in", rec)
    c.ModeSet().write(tmp_path / "mode_set.in")
    c.FrechetSpec(False, [], [], []).write(tmp_path / "frechet.in")

    t0 = time.time()
    run_tool("fm3d", tmp_path, overrides={"fm3d": BIN / "fm3d"}, timeout=1800)
    rt = time.time() - t0
    arr = times.read_arrivals(tmp_path / "arrivals.dat")
    assert len(arr) == len(rays) and arr["valid"].all()
    for row in arr.itertuples():
        s, r = sources[row.source - 1], pos.loc[row.receiver]
        epi = float(spherical_distance_km(s.lat, s.lon, r.lat, r.lon))
        dist = float(np.hypot(epi, s.depth_km - r.depth_km))
        # travel time bounded by the fastest and slowest velocities along the path
        vmin, vmax = float(ds["vp"].min()), float(ds["vp"].max())
        assert dist / vmax * 0.95 < row.time < dist / vmin * 1.05
    with capsys.disabled():
        print(f"\nfm3d runtime {rt:.1f} s; grid {g.summary()}")
        print("\n".join(findings))
