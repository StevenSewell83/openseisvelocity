"""Tests for osv.resolution.models (checkerboard, spikes, anomaly bodies, input models)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import AnomalyBodyConfig, Model3DConfig, ResolutionConfig
from osv.coords import LocalProjection
from osv.model3d.grid import define_grid
from osv.resolution.models import (
    UnderSamplingWarning,
    anomaly_bodies,
    checkerboard,
    grid_xyz,
    input_model,
    pattern_summary,
    spikes,
    write_vgrids,
)

LAT0, LON0 = -38.5, 176.2
GRID_CFG = {"dx_km": 0.5, "dy_km": 0.5, "dz_km": 0.5, "pad_km": 2.0, "zmax_km": 6.0}


@pytest.fixture(scope="module")
def grid():
    la, lo = np.meshgrid(np.linspace(0, 0.1, 3), np.linspace(0, 0.12, 3), indexing="ij")
    st = pd.DataFrame({"lat": LAT0 + la.ravel(), "lon": LON0 + lo.ravel(), "elevation_m": 300.0})
    return define_grid(st, None, Model3DConfig(grid=GRID_CFG, propgrid_refine=2))


def test_checkerboard_sign_periodicity_and_vpvs(grid):
    cell = 2.0
    ds = checkerboard(grid, 5.0, cell, 6.0, shape="sin", phase=(0.1, 0.1, 0.1))
    x, y, z = grid_xyz(grid)
    k = np.pi / cell
    expect = 6.0 * np.sin(k * (x - 0.1)) * np.sin(k * (y - 0.1)) * np.sin(k * (z - 0.1))
    np.testing.assert_allclose(ds["dvp_pct"].values, expect, atol=1e-9)
    assert pattern_summary(ds, grid)["peak_abs_pct"] <= 6.0 + 1e-9
    # Vp/Vs preserved
    ratio = ds["vp"] / ds["vs"]
    np.testing.assert_allclose(ratio.values, 1.73, rtol=1e-12)
    np.testing.assert_allclose(ds["vp"].values, 5.0 * (1 + ds["dvp_pct"].values / 100))
    # alternation: moving by one cell along x flips sign; by two cells repeats (in km)
    p = ds["dvp_pct"].values
    i = np.argmax(np.abs(p))
    peak = np.unravel_index(i, p.shape)
    px, py, pz = (a[peak] for a in (x, y, z))
    f = lambda xx: 6.0 * np.sin(k * (xx - 0.1)) * np.sin(k * (py - 0.1)) * np.sin(k * (pz - 0.1))  # noqa: E731
    assert f(px + cell) == pytest.approx(-f(px), abs=1e-9)
    assert f(px + 2 * cell) == pytest.approx(f(px), abs=1e-9)


def test_checkerboard_block_and_2d(grid):
    ds = checkerboard(grid, 5.0, 2.0, 5.0, shape="block", depth_cells=False)
    v = np.unique(np.round(ds["dvp_pct"].values, 9))
    assert set(v) == {-5.0, 5.0}
    # constant in depth
    assert np.allclose(ds["dvp_pct"].values, ds["dvp_pct"].values[:1])
    # 3-D block pattern alternates in depth
    d3 = checkerboard(grid, 5.0, 2.0, 5.0, shape="block")["dvp_pct"].values
    assert not np.allclose(d3, d3[:1])


def test_half_cell_offset_is_shifted_pattern(grid):
    cell = 2.0
    a = checkerboard(grid, 5.0, cell, 5.0, offset_half_cell=True)["dvp_pct"].values
    b = checkerboard(grid, 5.0, cell, 5.0, phase=(1.0, 1.0, 1.0))["dvp_pct"].values
    np.testing.assert_allclose(a, b, atol=1e-9)
    c = checkerboard(grid, 5.0, cell, 5.0)["dvp_pct"].values
    assert not np.allclose(a, c)
    # 2-D: offset leaves depth untouched
    a2 = checkerboard(grid, 5.0, cell, 5.0, offset_half_cell=True, depth_cells=False)
    b2 = checkerboard(grid, 5.0, cell, 5.0, depth_cells=False, phase=(1.0, 1.0, 0.0))
    np.testing.assert_allclose(a2["dvp_pct"].values, b2["dvp_pct"].values, atol=1e-9)


def test_undersampling_warning(grid):
    with pytest.warns(UnderSamplingWarning, match="under-sampled"):
        ds = checkerboard(grid, 5.0, 0.9, 5.0)
    assert "under-sampled" in ds.attrs["warnings"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checkerboard(grid, 5.0, 1.0, 5.0)  # exactly 2 x spacing: fine


def test_taper_outside_region(grid):
    reg = pd.DataFrame({"lat": [LAT0 + 0.04, LAT0 + 0.06], "lon": [LON0 + 0.05, LON0 + 0.07]})
    full = checkerboard(grid, 5.0, 2.0, 5.0)["dvp_pct"].values
    t = checkerboard(grid, 5.0, 2.0, 5.0, taper_to_zero_outside=reg, taper_km=1.0)["dvp_pct"].values
    assert np.abs(t).max() <= np.abs(full).max() + 1e-12
    assert np.abs(t[:, 0, 0]).max() == 0.0  # grid corner is far outside
    assert np.abs(t).max() > 0


def test_spike_peak_location_and_amplitude(grid):
    spec = {"x_km": 0.5, "y_km": -0.5, "z_km": 2.0, "radius_km": 1.0, "pct": -8.0}
    ds = spikes(grid, 5.0, [spec])
    p = ds["dvp_pct"].values
    x, y, z = grid_xyz(grid)
    i = np.unravel_index(np.argmax(np.abs(p)), p.shape)
    # peak at the node closest to the requested centre, within half a node
    assert abs(x[i] - 0.5) <= 0.3 and abs(y[i] - -0.5) <= 0.3 and abs(z[i] - 2.0) <= 0.3
    r2 = (x[i] - 0.5) ** 2 + (y[i] + 0.5) ** 2 + (z[i] - 2.0) ** 2
    assert p[i] == pytest.approx(-8.0 * np.exp(-r2 / (2 * 0.25)))
    assert p.min() >= -8.0 - 1e-9 and p.max() <= 0.0
    # sphere: uniform within the radius
    s = spikes(grid, 5.0, [{**spec, "shape": "sphere"}])["dvp_pct"].values
    r = np.sqrt((x - 0.5) ** 2 + (y + 0.5) ** 2 + (z - 2.0) ** 2)
    assert np.all(s[r <= 1.0] == -8.0) and np.all(s[r > 1.0] == 0.0)
    # lat/lon/depth form agrees with the km form
    proj = LocalProjection(grid.centre_lat, grid.centre_lon)
    la, lo = proj.inverse(0.5, -0.5)
    s2 = spikes(grid, 5.0, [{"lat": la, "lon": lo, "depth": 2.0, "radius_km": 1.0, "pct": -8.0}])
    np.testing.assert_allclose(s2["dvp_pct"].values, ds["dvp_pct"].values, atol=1e-6)


def test_vs_scale_options(grid):
    ds = spikes(
        grid, 5.0, [{"x_km": 0, "y_km": 0, "z_km": 2, "radius_km": 1, "pct": 5}], vs_scale=0
    )
    np.testing.assert_allclose(ds["vs"].values, 5.0 / 1.73)
    assert ds["vp"].max() > 5.0


def test_slab_geometry(grid):
    # slab striking north (0 deg), dipping 45 deg to the east, through (0, 0, 3)
    body = AnomalyBodyConfig(
        kind="slab", x_km=0, y_km=0, z_km=3.0, pct=6, strike_deg=0, dip_deg=45,
        length_km=100, width_km=100, thickness_km=0.6,
    )  # fmt: skip
    ds = anomaly_bodies(grid, 5.0, [body])
    x, y, z = grid_xyz(grid)
    # plane z - 3 = x  (down to the east); distance to plane = |z-3-x|/sqrt(2)
    inside = np.abs(z - 3.0 - x) / np.sqrt(2) <= 0.3
    p = ds["dvp_pct"].values
    assert inside.sum() > 10
    assert np.all(p[inside] == 6.0) and np.all(p[~inside] == 0.0)


def test_block_and_cylinder_and_edge(grid):
    blk = {"kind": "block", "x_km": 0, "y_km": 0, "z_km": 3, "pct": 4,
           "length_km": 2, "width_km": 1, "height_km": 1, "strike_deg": 90}  # fmt: skip
    p = anomaly_bodies(grid, 5.0, [blk])["dvp_pct"].values
    x, y, z = grid_xyz(grid)
    # strike 90 = east: length along x
    ins = (np.abs(x) <= 1) & (np.abs(y) <= 0.5) & (np.abs(z - 3) <= 0.5)
    assert np.all(p[ins] == 4.0) and np.all(p[~ins] == 0.0)
    cyl = {"kind": "cylinder", "x_km": 0, "y_km": 0, "z_km": 3, "pct": -3,
           "radius_km": 1.0, "length_km": 2.0, "dip_deg": 90}  # fmt: skip
    p = anomaly_bodies(grid, 5.0, [cyl])["dvp_pct"].values
    ins = (np.hypot(x, y) <= 1.0) & (np.abs(z - 3) <= 1.0)
    assert np.all(p[ins] == -3.0) and np.all(p[~ins] == 0.0)
    smooth = anomaly_bodies(grid, 5.0, [{**cyl, "edge_km": 1.0}])["dvp_pct"].values
    assert np.any((smooth < 0) & (smooth > -3.0))
    assert smooth.min() >= -3.0 - 1e-9
    with pytest.raises(ValueError):
        anomaly_bodies(grid, 5.0, [{**cyl, "kind": "torus"}])


def test_input_models(grid, tmp_path):
    bg = 5.0
    cb = checkerboard(grid, bg, 2.0, 5.0)
    # final_3d from dataset (identical axes) and from NetCDF, on a different grid by interp
    ds = input_model("final_3d", grid, bg, dataset=cb)
    np.testing.assert_allclose(ds["vp"].values, cb["vp"].values)
    np.testing.assert_allclose(ds["dvp_pct"].values, cb["dvp_pct"].values, atol=1e-9)
    nc = tmp_path / "final.nc"
    coarse = cb[["vp", "vs"]].isel(depth=slice(None, None, 2))
    coarse.to_netcdf(nc)
    ds2 = input_model("final_3d", grid, bg, path=nc)
    assert ds2["vp"].shape == grid.shape_full
    # file kind
    f = input_model("file", grid, bg, path=nc)
    assert np.isfinite(f["vp"]).all()
    # no background: relative to laterally averaged model
    nb = input_model("final_3d", grid, dataset=cb)
    assert abs(nb["dvp_pct"].mean()) < 1.0
    # anomaly kind
    an = input_model(
        "anomaly", grid, bg, bodies=[{"kind": "block", "x_km": 0, "y_km": 0, "z_km": 3, "pct": 3}]
    )
    assert an["dvp_pct"].max() == 3.0
    # user_model via builder
    m3 = Model3DConfig(grid=GRID_CFG, propgrid_refine=2, source="velest_1d")
    m1d = pd.DataFrame({"depth_top_km": [-0.6, 3.0], "vp": [3.0, 6.0], "vpvs": [1.7, 1.7]})
    um = input_model("user_model", grid, cfg=m3, model_1d=m1d)
    assert um["vp"].shape == grid.shape_full
    with pytest.raises(ValueError):
        input_model("file", grid, bg)


def test_summary_and_vgrids(grid):
    ds = checkerboard(grid, 5.0, 2.0, 5.0)
    s = pattern_summary(ds, grid)
    assert s["peak_abs_pct"] <= 5.0 + 1e-9
    assert abs(s["mean_pct"]) < 1.0
    assert s["vpvs_min"] == pytest.approx(1.73) and s["vpvs_max"] == pytest.approx(1.73)
    vg = write_vgrids(grid, ds, uncertainty=0.1)
    assert type(vg).__name__ == "VelocityGrids"


def test_config_schema_entries():
    r = ResolutionConfig(
        input_models=["anomaly", "file"],
        anomalies=[{"kind": "slab", "x_km": 0, "y_km": 0, "z_km": 3, "pct": 5, "dip_deg": 60}],
        checkerboard={"shape": "block", "depth_cells": False},
        spikes=[{"x_km": 0, "y_km": 0, "z_km": 2, "radius_km": 1, "pct": -5, "shape": "sphere"}],
    )
    assert r.anomalies[0].dip_deg == 60 and r.checkerboard.shape == "block"
