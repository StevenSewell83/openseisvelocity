"""Tests for osv.model3d.user_grid (user-supplied velocity grid import)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer

from osv.config.schema import Model3DConfig
from osv.model3d.grid import define_grid
from osv.model3d.user_grid import UserGridError, load_user_grid

NZTM = "EPSG:2193"
LAT0, LON0 = -38.55, 176.19


def vel(lat, lon, depth):
    """Smooth (linear) synthetic Vp so that trilinear/Delaunay interpolation is exact."""
    return 2.0 + 0.5 * depth + 3.0 * (lat - LAT0) + 2.0 * (lon - LON0)


@pytest.fixture(scope="module")
def grid():
    la, lo = np.meshgrid(np.linspace(0, 0.1, 3), np.linspace(0, 0.12, 3), indexing="ij")
    st = pd.DataFrame(
        {"lat": LAT0 + la.ravel(), "lon": LON0 + lo.ravel(), "elevation_m": 300.0, "used": True}
    )
    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 0.5, "pad_km": 2.0, "zmax_km": 5.0},
        propgrid_refine=2,
    )
    return define_grid(st, None, cfg)


def _expected(grid):
    zz, la, lo = np.meshgrid(grid.depth_full, grid.lat_full, grid.lon_full, indexing="ij")
    return vel(la, lo, zz)


def test_netcdf_geographic_roundtrip(grid, tmp_path):
    lat = np.linspace(grid.lat_full[0] - 0.05, grid.lat_full[-1] + 0.05, 17)
    lon = np.linspace(grid.lon_full[0] - 0.05, grid.lon_full[-1] + 0.05, 19)
    dep = np.linspace(grid.depth_full[0] - 1, grid.depth_full[-1] + 1, 13)
    zz, la, lo = np.meshgrid(dep, lat, lon, indexing="ij")
    ds = xr.Dataset(
        {
            "vp": (("depth", "lat", "lon"), vel(la, lo, zz)),
            "vs": (("depth", "lat", "lon"), vel(la, lo, zz) / 1.75),
        },
        coords={"depth": dep, "lat": lat, "lon": lon},
    )
    p = tmp_path / "m.nc"
    ds.to_netcdf(p)
    out, cov = load_user_grid(p, grid)
    assert out.vp.shape == grid.shape_full
    assert cov.all()
    np.testing.assert_allclose(out.vp.values, _expected(grid), atol=1e-9)
    np.testing.assert_allclose(out.vs.values, _expected(grid) / 1.75, atol=1e-9)


def test_netcdf_nztm_elevation_metres_partial_coverage(grid, tmp_path):
    tr = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
    tr_inv = Transformer.from_crs(NZTM, "EPSG:4326", always_xy=True)
    lo2, la2 = np.meshgrid(grid.lon_full, grid.lat_full)
    e, n = tr.transform(lo2, la2)
    x = np.arange(e.min() - 1000, e.max() + 1000, 250.0)
    y = np.arange(n.min() - 1000, n.max() + 1000, 250.0)
    # cover only the western half
    x = x[x < e.mean()]
    z_elev = np.linspace(1000.0, -6000.0, 29)  # descending elevation, m
    zz, yy, xx = np.meshgrid(z_elev, y, x, indexing="ij")
    lon, lat = tr_inv.transform(xx, yy)
    v = vel(lat, lon, -zz / 1000.0)
    ds = xr.Dataset(
        {"vp": (("z", "y", "x"), v)}, coords={"z": z_elev, "y": y, "x": x}, attrs={"crs": NZTM}
    )
    p = tmp_path / "m.nc"
    ds.to_netcdf(p)
    out, cov = load_user_grid(p, grid, z_convention="elevation", units="m")
    assert 0.2 < cov.mean() < 0.8
    assert cov.values[:, :, 0].all() and not cov.values[:, :, -1].any()
    exp = _expected(grid)
    # the dip of the field in NZTM -> lat/lon is not exactly linear: allow small tolerance
    np.testing.assert_allclose(out.vp.values[cov.values], exp[cov.values], atol=2e-3)
    assert "vs" not in out


def _csv_points(grid, regular: bool):
    tr = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
    lo2, la2 = np.meshgrid(grid.lon_full, grid.lat_full)
    e, n = tr.transform(lo2, la2)
    if regular:
        xs = np.arange(e.min() - 500, e.max() + 500, 400.0)
        ys = np.arange(n.min() - 500, n.max() + 500, 400.0)
        zs = np.arange(grid.depth_full[0] - 0.5, grid.depth_full[-1] + 0.6, 0.5)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        return xx.ravel(), yy.ravel(), zz.ravel()
    rng = np.random.default_rng(1)
    k = 6000
    xx = rng.uniform(e.min() - 500, e.max() + 500, k)
    yy = rng.uniform(n.min() - 500, n.max() + 500, k)
    zz = rng.uniform(grid.depth_full[0] - 0.5, grid.depth_full[-1] + 0.5, k)
    return xx, yy, zz


@pytest.mark.parametrize("regular", [True, False])
def test_csv_roundtrip_nztm(grid, tmp_path, regular):
    tr_inv = Transformer.from_crs(NZTM, "EPSG:4326", always_xy=True)
    xx, yy, zz = _csv_points(grid, regular)
    lon, lat = tr_inv.transform(xx, yy)
    v = vel(lat, lon, zz)
    df = pd.DataFrame({"E": xx, "N": yy, "Z_m": zz * 1000.0, "Vp": v * 1000.0})  # m/s
    p = tmp_path / "m.csv"
    df.to_csv(p, index=False)
    out, cov = load_user_grid(
        p, grid, crs=NZTM, columns={"x": "E", "y": "N", "z": "Z_m", "vp": "Vp"}, units="m"
    )
    exp = _expected(grid)
    frac = cov.mean()
    assert frac > (0.99 if regular else 0.6)
    tol = 3e-3 if regular else 5e-2
    np.testing.assert_allclose(out.vp.values[cov.values], exp[cov.values], atol=tol)


def test_csv_scattered_max_distance_mask(grid, tmp_path):
    # points only in the western third of the model: east must be masked out
    rng = np.random.default_rng(2)
    k = 4000
    lon = rng.uniform(grid.lon_full[0] - 0.02, grid.lon_full[0] + 0.3 * np.ptp(grid.lon_full), k)
    lat = rng.uniform(grid.lat_full[0] - 0.02, grid.lat_full[-1] + 0.02, k)
    z = rng.uniform(grid.depth_full[0] - 0.5, grid.depth_full[-1] + 0.5, k)
    df = pd.DataFrame(
        {"x": lon, "y": lat, "z": z, "vp": vel(lat, lon, z), "vs": vel(lat, lon, z) / 1.7}
    )
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    out, cov = load_user_grid(p, grid, max_distance_km=1.0)
    assert cov.values[:, :, 0].mean() > 0.9
    assert not cov.values[:, :, -1].any()
    assert np.isnan(out.vp.values[~cov.values]).all()
    np.testing.assert_allclose(
        out.vs.values[cov.values], out.vp.values[cov.values] / 1.7, atol=1e-6
    )


def test_csv_errors(grid, tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(UserGridError, match="missing column"):
        load_user_grid(p, grid)
    # entirely outside the model
    xx, yy, zz = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij")
    df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "z": zz.ravel(), "vp": 3.0})
    p2 = tmp_path / "far.csv"
    df.to_csv(p2, index=False)
    with pytest.raises(UserGridError, match="does not overlap"):
        load_user_grid(p2, grid)
    with pytest.raises(FileNotFoundError):
        load_user_grid(tmp_path / "none.nc", grid)
