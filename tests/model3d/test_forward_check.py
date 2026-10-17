"""Tests for osv.model3d.forward_check (fm3d on the starting model, residual summary)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import Model3DConfig
from osv.coords import spherical_distance_km
from osv.model3d.builder import StartingModel
from osv.model3d.forward_check import (
    VelestReference,
    forward_check,
    phase_vgrids,
    residual_stats,
)
from osv.model3d.grid import define_grid
from osv.model3d.sources import build_fmtomo_data
from osv.tools import ToolNotFoundError, which_tool

T0 = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
VP, VS = 5.0, 3.0


def _have_fm3d() -> bool:
    try:
        which_tool("fm3d")
    except ToolNotFoundError:
        return False
    return True


def test_residual_stats():
    r = np.array([1.0, -1.0, 2.0, np.nan, -2.0])
    ev = np.array(["a", "a", "b", "b", "b"])
    s = residual_stats(r, ev)
    assert s["n"] == 4 and s["mean_s"] == pytest.approx(0.0)
    assert s["rms_s"] == pytest.approx(np.sqrt(2.5))
    assert s["mad_s"] == pytest.approx(1.5)
    assert s["rms_demeaned_s"] == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4 - 0))
    assert residual_stats(np.array([np.nan]))["n"] == 0


@pytest.fixture(scope="module")
def setup():
    stations = pd.DataFrame(
        {
            "alias": ["A", "B", "C", "D"],
            "lat": [-38.55, -38.45, -38.50, -38.53],
            "lon": [176.15, 176.25, 176.30, 176.18],
            "elevation_m": [400.0, 300.0, 350.0, 380.0],
            "sensor_elevation_m": [400.0, 300.0, 350.0, 380.0],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(4)],
            "time": [T0 + pd.Timedelta(hours=i) for i in range(4)],
            "lat": [-38.50, -38.51, -38.52, -38.49],
            "lon": [176.20, 176.21, 176.19, 176.24],
            "depth_km_bsl": [2.0, 3.0, 2.5, 3.5],
            "split": "calibration",
        }
    )
    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 0.5, "pad_km": 2.0, "zmax_km": 6.0},
        propgrid_refine=2,
    )
    grid = define_grid(stations, events, cfg)
    rows = []
    for ev in events.itertuples():
        for st in stations.itertuples():
            epi = float(spherical_distance_km(ev.lat, ev.lon, st.lat, st.lon))
            dist = np.hypot(epi, ev.depth_km_bsl + st.sensor_elevation_m / 1000)
            for ph, v in (("P", VP), ("S", VS)):
                rows.append(
                    {
                        "event_id": ev.event_id,
                        "pick_id": f"{ev.event_id}{st.alias}{ph}",
                        "phase": ph,
                        "time": ev.time + pd.Timedelta(seconds=dist / v),
                        "alias": st.alias,
                        "sigma_s": 0.03,
                    }
                )
    picks = pd.DataFrame(rows)
    p = build_fmtomo_data(
        picks, events, stations, cfg, grid=grid, hypocentres="catalog", subset="all"
    )
    s = build_fmtomo_data(
        picks,
        events,
        stations,
        cfg,
        grid=grid,
        hypocentres="catalog",
        subset="all",
        phases=("S",),
        layout="separate",
    )
    return grid, p, s, picks, events, stations, cfg


def _model(grid, vp_scale=None):
    ds = grid.template().drop_vars("velocity")
    shape = grid.shape_full
    vp = np.full(shape, VP)
    vs = np.full(shape, VS)
    for name, arr in (("vp_bg", vp), ("vs_bg", vs)):
        ds[name] = (("depth", "lat", "lon"), arr.copy())
    if vp_scale is not None:
        vp = vp.copy()
        vp[:, : shape[1] // 2, :] *= vp_scale
        vs = vs.copy()
        vs[:, : shape[1] // 2, :] *= vp_scale
    for name, arr in (("vp", vp), ("vs", vs), ("vp_unc", vp * 0.06), ("vs_unc", vs * 0.06)):
        ds[name] = (("depth", "lat", "lon"), arr)
    return StartingModel(grid, ds)


def test_phase_vgrids_types(setup):
    grid, p, s, *_ = setup
    m = _model(grid)
    assert len(list(phase_vgrids(m.dataset, grid, p))) == 1
    vs_grid = list(phase_vgrids(m.dataset, grid, s))
    assert len(vs_grid) == 1
    assert float(vs_grid[0][2]["velocity"].mean()) == pytest.approx(VS)


@pytest.mark.integration
@pytest.mark.skipif(not _have_fm3d(), reason="no fm3d")
def test_synthetic_1d_data_gives_zero_residuals(setup, tmp_path):
    grid, p, s, *_rest = setup
    itf = grid.flat_interfaces()
    fc = forward_check(_model(grid), grid, itf, [p, s], tmp_path, 2)
    assert not fc.summary["model_differs_from_background"]
    for ph in ("P", "S"):
        st = fc.summary["phases"][ph]["start"]
        assert st["n"] == 16
        assert st["rms_s"] < 0.05, st
        assert abs(st["mean_s"]) < 0.05
        assert fc.summary["phases"][ph]["verdict"] == "not_applicable"
    assert {"res_start", "dist_km", "event_id", "station"} <= set(fc.residuals.columns)
    assert len(fc.by_station) == 8 and len(fc.by_event) == 8
    assert fc.residuals["dist_km"].between(1, 30).all()
    assert (tmp_path / "start_P" / "arrivals.dat").exists()
    assert not (tmp_path / "start_P" / "vgrids.in").exists()


@pytest.mark.integration
@pytest.mark.skipif(not _have_fm3d(), reason="no fm3d")
def test_user_model_worse_than_background(setup, tmp_path):
    grid, p, *_ = setup
    itf = grid.flat_interfaces()
    ref = VelestReference(
        rms_s=0.05,
        events=pd.DataFrame({"event_id": [f"e{i}" for i in range(4)], "rms": [0.04] * 4}),
        delays=pd.DataFrame({"alias": list("ABCD"), "ptcor": [0.1, -0.1, 0.0, 0.05]}),
    )
    fc = forward_check(_model(grid, vp_scale=1.25), grid, itf, p, tmp_path, 2, velest=ref)
    ph = fc.summary["phases"]["P"]
    assert fc.summary["background_run"]
    assert ph["background"]["rms_s"] < 0.05 < ph["start"]["rms_s"]
    assert ph["verdict"] == "worse" and ph["rms_ratio"] > 2
    codes = {f.code for f in fc.findings}
    assert "user_model_fits_worse" in codes
    assert "velest" in fc.summary and fc.summary["velest"]["velest_rms_s"] == 0.05
    assert fc.to_json()["findings"]
