"""Tests for osv.model3d.interfaces (surface and base interfaces)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.config.schema import GridConfig, Model3DConfig
from osv.io.fmtomo import grids as fg
from osv.model3d.grid import define_grid
from osv.model3d.interfaces import (
    InterfaceError,
    build_interfaces,
    interface_depth_at,
    place_sources,
)

LAT0, LON0 = -38.5, 176.2


def _cfg(**kw) -> Model3DConfig:
    return Model3DConfig(
        grid=GridConfig(dx_km=1.0, dy_km=1.0, dz_km=0.5, pad_km=3.0, zmax_km=6.0),
        propgrid_refine=2,
        **kw,
    )


@pytest.fixture
def stations() -> pd.DataFrame:
    la, lo = np.meshgrid(np.linspace(0, 0.12, 5), np.linspace(0, 0.15, 5), indexing="ij")
    elev = 300.0 + 2000.0 * la.ravel() + 1000.0 * lo.ravel()  # tilted plane + spike below
    st = pd.DataFrame(
        {
            "sta": [f"S{i:02d}" for i in range(la.size)],
            "lat": LAT0 + la.ravel(),
            "lon": LON0 + lo.ravel(),
            "elevation_m": elev,
            "sensor_depth_m": 0.0,
            "used": True,
        }
    )
    st["elevation_m"] = st["elevation_m"].round()
    st.loc[12, "elevation_m"] += 150.0  # a summit: smoothing will cut it
    st["sensor_elevation_m"] = st["elevation_m"] - st["sensor_depth_m"]
    return st


def _surface(itf) -> np.ndarray:
    return itf.data["interface_depth"].isel(interface=0).to_numpy()


def test_station_elevations_surface_above_receivers(stations):
    cfg = _cfg(surface="station_elevations", surface_smoothing_km=2.0)
    g = define_grid(stations, None, cfg)
    fnd: list[str] = []
    itf = build_interfaces(g, stations, cfg, findings=fnd)
    assert itf.n_interfaces == 2
    assert itf.data["interface_depth"].shape == (2, g.lat_full.size, g.lon_full.size)
    lo, hi = g.surface_limits_km
    d = itf.data["interface_depth"].to_numpy()
    assert d.min() >= lo - 1e-9 and d[1].max() == pytest.approx(hi)
    s = interface_depth_at(itf, g, stations.lat, stations.lon)
    rdep = -stations["sensor_elevation_m"].to_numpy() / 1000.0
    assert np.all(s <= rdep + 1e-9)  # receivers on/below the surface seen by fm3d
    # the summit was lifted locally and reported
    assert any("lifted locally" in f for f in fnd)
    assert any("surface (station_elevations)" in f for f in fnd)
    assert "lifted" in itf.data.attrs["findings"]
    fg.validate_model(g.to_vgrids(3.0), itf, g.propgrid)


def test_no_smoothing_reproduces_linear_surface_at_nodes(stations):
    cfg = _cfg(surface="station_elevations", surface_smoothing_km=0.0)
    flat_stations = stations.copy()
    flat_stations["elevation_m"] = 400.0
    flat_stations["sensor_elevation_m"] = 400.0
    g = define_grid(flat_stations, None, cfg)
    itf = build_interfaces(g, flat_stations, cfg)
    assert np.allclose(_surface(itf), -0.4, atol=5e-3)  # lifted by the 1 m margin


def test_borehole_sensor_below_surface_not_lifted(stations):
    st = stations.copy()
    st["sensor_depth_m"] = 500.0
    st["sensor_elevation_m"] = st["elevation_m"] - 500.0
    cfg = _cfg(surface="station_elevations", surface_smoothing_km=0.0)
    g = define_grid(st, None, cfg)
    fnd: list[str] = []
    build_interfaces(g, st, cfg, findings=fnd)
    assert not any("lifted" in f for f in fnd)  # surface from elevation_m, sensors deeper


def test_flat_mode_and_lifting_report(stations):
    cfg = _cfg(surface="flat", surface_elevation_m=800.0)
    g = define_grid(stations, None, cfg)
    itf = build_interfaces(g, stations, cfg)
    assert np.allclose(_surface(itf), -0.8, atol=2e-3)
    # flat surface below the stations: lifted locally, reported
    cfg2 = _cfg(surface="flat", surface_elevation_m=0.0)
    g2 = define_grid(stations, None, cfg2)
    fnd: list[str] = []
    itf2 = build_interfaces(g2, stations, cfg2, findings=fnd)
    assert any("lifted locally" in f for f in fnd)
    s = interface_depth_at(itf2, g2, stations.lat, stations.lon)
    assert np.all(s <= -stations.sensor_elevation_m / 1000 + 1e-9)
    assert _surface(itf2).max() == pytest.approx(0.0)  # elsewhere stays flat at sea level


def test_dem_dataarray(stations):
    cfg = _cfg(surface="dem", surface_smoothing_km=0.5)
    g = define_grid(stations, None, cfg)
    lat = np.linspace(g.lat_full[0] - 0.05, g.lat_full[-1] + 0.05, 200)
    lon = np.linspace(g.lon_full[0] - 0.05, g.lon_full[-1] + 0.05, 200)
    dem = xr.DataArray(
        800.0 + 0.0 * lat[:, None] + 0.0 * lon[None, :], coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )  # fmt: skip
    itf = build_interfaces(g, stations, cfg, dem=dem)
    assert np.allclose(_surface(itf), -0.8, atol=1e-6)
    with pytest.raises(InterfaceError, match="needs a DEM"):
        build_interfaces(g, stations, cfg)
    # DEM with a void: filled by the nearest value and reported
    dem_void = dem.copy()
    dem_void[:, :80] = np.nan
    fnd: list[str] = []
    itf = build_interfaces(g, stations, cfg, dem=dem_void, findings=fnd)
    assert any("nodata" in f for f in fnd)
    assert np.isfinite(_surface(itf)).all()


def test_dem_geotiff(stations, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    cfg = _cfg(surface="dem", surface_smoothing_km=0.5)
    g = define_grid(stations, None, cfg)
    n = 120
    west, north = g.lon_full[0] - 0.1, g.lat_full[-1] + 0.1
    res = (g.lon_full[-1] - g.lon_full[0] + 0.2) / n
    arr = np.fromfunction(lambda r, c: 700.0 + 2.0 * c, (n, n), dtype="float32")
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
        transform=from_origin(west, north, res, res), nodata=-9999.0,
    ) as dst:  # fmt: skip
        dst.write(arr, 1)
    itf = build_interfaces(g, stations, cfg, dem=path)
    # elevation increases with longitude: the surface gets shallower to the east
    s = _surface(itf)
    assert (s[:, 0] > s[:, -1]).all()
    col = (g.lon_full[g.lon_full.size // 2] - west) / res - 0.5
    expect = (700.0 + 2.0 * col) / 1000.0
    mid = s[s.shape[0] // 2, g.lon_full.size // 2]
    assert -mid == pytest.approx(expect, abs=0.02)
    with pytest.raises(InterfaceError, match="not found"):
        build_interfaces(g, stations, cfg, dem=tmp_path / "missing.tif")


def test_surface_clipped_to_propgrid_top(stations):
    cfg = _cfg(surface="flat", surface_elevation_m=5000.0)
    g = define_grid(stations, None, cfg)
    fnd: list[str] = []
    itf = build_interfaces(g, stations, cfg, findings=fnd)
    assert _surface(itf).min() == pytest.approx(g.surface_limits_km[0])
    assert any("clipped" in f for f in fnd)


def test_base_validation(stations):
    cfg = _cfg(surface="flat", surface_elevation_m=500.0)
    g = define_grid(stations, None, cfg)
    with pytest.raises(InterfaceError, match="base depth"):
        build_interfaces(g, stations, cfg, base_depth_km=g.bottom_depth_km + 5.0)
    itf = build_interfaces(g, stations, cfg, base_depth_km=5.0)
    assert itf.data["interface_depth"].isel(interface=1).max() == pytest.approx(5.0)


def test_place_sources_moves_events_below_surface(stations):
    cfg = _cfg(surface="flat", surface_elevation_m=800.0)
    g = define_grid(stations, None, cfg)
    itf = build_interfaces(g, stations, cfg)
    ev = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "lat": [LAT0 + 0.05] * 3,
            "lon": [LON0 + 0.05] * 3,
            "depth_km_bsl": [-0.9, 0.0, g.bottom_depth_km + 0.2],
        }
    )
    fnd: list[str] = []
    out = place_sources(itf, g, ev, margin_km=0.01, findings=fnd)
    assert out.depth_km_bsl[0] == pytest.approx(-0.8 + 0.01)
    assert out.depth_km_bsl[1] == 0.0 and out.moved_km[1] == 0.0
    assert out.moved_km[0] == pytest.approx(0.11)
    assert any("above the surface moved down" in f and "a" in f for f in fnd)
    assert any("below the base" in f for f in fnd)
    assert ev.depth_km_bsl[0] == -0.9  # input untouched
