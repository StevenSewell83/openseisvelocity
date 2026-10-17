"""Tests for osv.model3d.wells (synthetic LAS files)."""

from __future__ import annotations

from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import pytest

from osv.config.schema import WellsConfig
from osv.model3d.geology import build_geology_model
from osv.model3d.grid import define_grid
from osv.model3d.wells import (
    WellsError,
    backus_average,
    build_geology_wells_model,
    build_wells_model,
    interpolate_lateral,
    load_wells,
    minimum_curvature,
    read_las_curves,
    read_wells_table,
    slowness_to_velocity,
    wells_horizons,
)

LAT0, LON0 = -38.55, 176.19


@pytest.fixture(scope="module")
def grid():
    la, lo = np.meshgrid(np.linspace(0, 0.1, 3), np.linspace(0, 0.12, 3), indexing="ij")
    st = pd.DataFrame(
        {"lat": LAT0 + la.ravel(), "lon": LON0 + lo.ravel(), "elevation_m": 300.0, "used": True}
    )
    from osv.config.schema import Model3DConfig

    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 0.5, "pad_km": 2.0, "zmax_km": 5.0},
        propgrid_refine=2,
    )
    return define_grid(st, None, cfg)


def vp_true(z_km):
    return 2.0 + 0.5 * np.asarray(z_km)


def write_las(
    path: Path,
    md,
    dt=None,
    *,
    dt_unit="US/F",
    dts=None,
    rhob=None,
    well="W",
    depth_unit="m",
) -> Path:
    las = lasio.LASFile()
    las.well["WELL"].value = well
    las.append_curve("DEPT", np.asarray(md), unit=depth_unit)
    if dt is not None:
        las.append_curve("DT", np.asarray(dt), unit=dt_unit)
    if dts is not None:
        las.append_curve("DTS", np.asarray(dts), unit=dt_unit)
    if rhob is not None:
        las.append_curve("RHOB", np.asarray(rhob), unit="g/cc")
    with open(path, "w") as f:
        las.write(f, version=2.0)
    return path


def linear_log(md_m, kb=0.0, unit="us/ft"):
    z = (np.asarray(md_m) - kb) / 1000.0
    v = vp_true(z)
    return 304.8 / v if unit == "us/ft" else 1000.0 / v


def make_table(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "wells.csv", index=False)
    return tmp_path / "wells.csv"


# ---------------------------------------------------------------------------------------
def test_slowness_conversion():
    assert slowness_to_velocity(np.array([100.0]), "us/ft")[0] == pytest.approx(3.048)
    assert slowness_to_velocity(np.array([250.0]), "us/m")[0] == pytest.approx(4.0)
    with pytest.raises(WellsError):
        slowness_to_velocity(np.array([1.0]), "s/m")


def test_read_las_nulls_and_feet(tmp_path):
    md_ft = np.arange(0, 1000, 5.0)
    dt = np.full(md_ft.size, 100.0)
    dt[3:6] = -999.25
    p = write_las(tmp_path / "a.las", md_ft, dt, depth_unit="ft")
    d, data, units, _ = read_las_curves(p, {"dt": "DT", "dts": "DTS"})
    assert d[1] == pytest.approx(5 * 0.3048)
    assert np.isnan(data["dt"][3:6]).all()
    assert "dts" not in data
    assert units["dt"].lower() == "us/f"


def test_missing_las(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_las_curves(tmp_path / "none.las", {"dt": "DT"})


def test_vertical_well_usft_nulls_spikes(tmp_path, grid):
    md = np.arange(0.0, 3000.0, 1.0)
    dt = linear_log(md, kb=50.0)
    dt[1000:1005] = -999.25
    dt[500] = 60.0  # spike
    dt[2000] = 600.0  # spike
    write_las(tmp_path / "w1.las", md, dt)
    tab = make_table(
        tmp_path,
        [{"name": "W1", "las_file": "w1.las", "lon": LON0 + 0.05, "lat": LAT0 + 0.05,
          "elevation_m": 50.0}],
    )  # fmt: skip
    cfg = WellsConfig(las_dir=tmp_path, table=tab, smoothing_m=0.0, shear_curve=None)
    wells, findings = load_wells(cfg, grid)
    w = wells[0]
    # KB 50 m above sea level: depth bsl = (md - 50) / 1000
    assert w.depth_km[0] == pytest.approx(-0.05)
    np.testing.assert_allclose(w.vp, vp_true(w.depth_km), atol=1e-3)
    assert w.vp.size == md.size - 5
    assert any(f.code == "despiked" for f in findings)
    assert any(f.code == "gardner" for f in findings)


def test_us_per_m_units_auto_and_forced(tmp_path, grid):
    md = np.arange(0.0, 2000.0, 2.0)
    write_las(tmp_path / "m.las", md, linear_log(md, unit="us/m"), dt_unit="US/M")
    write_las(tmp_path / "n.las", md, linear_log(md, unit="us/m"), dt_unit="")
    tab = make_table(
        tmp_path,
        [
            {"name": "A", "las_file": "m.las", "lon": LON0 + 0.05, "lat": LAT0 + 0.05},
            {"name": "B", "las_file": "n.las", "lon": LON0 + 0.06, "lat": LAT0 + 0.05},
        ],
    )
    cfg = WellsConfig(las_dir=tmp_path, table=tab, smoothing_m=0.0)
    wells, _ = load_wells(cfg, grid)
    for w in wells:  # unit string, then median heuristic
        np.testing.assert_allclose(w.vp, vp_true(w.depth_km), atol=1e-3)
    cfg = WellsConfig(las_dir=tmp_path, table=tab, smoothing_m=0.0, units="us/m")
    wells, _ = load_wells(cfg, grid)
    np.testing.assert_allclose(wells[1].vp, vp_true(wells[1].depth_km), atol=1e-3)


def test_missing_curve_and_short_log(tmp_path, grid):
    md = np.arange(0.0, 100.0, 1.0)
    write_las(tmp_path / "nocurve.las", md, None)
    write_las(tmp_path / "ok.las", md, linear_log(md))
    tab = make_table(
        tmp_path,
        [
            {"name": "A", "las_file": "nocurve.las", "lon": LON0, "lat": LAT0},
            {"name": "B", "las_file": "ok.las", "lon": LON0 + 0.05, "lat": LAT0 + 0.05},
            {"name": "C", "las_file": "ok.las", "lon": LON0 + 5, "lat": LAT0 + 5},
        ],
    )
    wells, findings = load_wells(WellsConfig(las_dir=tmp_path, table=tab), grid)
    codes = [f.code for f in findings]
    assert [w.name for w in wells] == ["B", "C"]
    assert "missing_curve" in codes and "short_log" in codes
    assert "well_outside_grid" in codes


def test_wells_table_errors(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"name": ["a"], "las_file": ["a.las"]}).to_csv(p, index=False)
    with pytest.raises(WellsError, match="x/y"):
        read_wells_table(p)


def test_crs_table_and_header_fallback(tmp_path, grid):
    from pyproj import Transformer

    e, n = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True).transform(
        LON0 + 0.05, LAT0 + 0.05
    )
    md = np.arange(0.0, 1000.0, 1.0)
    write_las(tmp_path / "w.las", md, linear_log(md))
    tab = make_table(tmp_path, [{"name": "W", "las_file": "w.las", "x": e, "y": n}])
    cfg = WellsConfig(las_dir=tmp_path, table=tab, crs="EPSG:2193")
    wells, _ = load_wells(cfg, grid)
    assert wells[0].lon == pytest.approx(LON0 + 0.05, abs=1e-6)
    assert wells[0].lat == pytest.approx(LAT0 + 0.05, abs=1e-6)
    # no table and no header coordinates -> clear error
    with pytest.raises(WellsError, match="coordinates"):
        load_wells(WellsConfig(las_dir=tmp_path), grid)


def test_minimum_curvature_straight_and_arc():
    md = np.array([0.0, 1000.0])
    tvd, n, e = minimum_curvature(md, [30.0, 30.0], [90.0, 90.0], np.array([500.0, 1000.0]))
    assert tvd[1] == pytest.approx(1000 * np.cos(np.radians(30)), rel=1e-6)
    assert e[1] == pytest.approx(500.0, rel=1e-6)
    assert n[1] == pytest.approx(0.0, abs=1e-6)
    # vertical to 90 degrees build over a quarter circle of radius R: tvd = e = R
    r = 500.0
    md = np.linspace(0, np.pi / 2 * r, 31)
    tvd, n, e = minimum_curvature(md, np.degrees(md / r), np.full(31, 0.0), md[-1:])
    assert tvd[0] == pytest.approx(r, rel=1e-3) and n[0] == pytest.approx(r, rel=1e-3)
    # beyond the survey end the last direction is extended
    tvd, _, e = minimum_curvature(np.array([0.0, 100.0]), [0, 0], [0, 0], np.array([250.0]))
    assert tvd[0] == pytest.approx(250.0)


def test_deviated_well(tmp_path, grid):
    md = np.arange(0.0, 3000.0, 2.0)
    write_las(tmp_path / "d.las", md, np.full(md.size, 100.0))
    pd.DataFrame({"MD": [0, 3000.0], "Inclination": [30, 30.0], "Azimuth": [90, 90.0]}).to_csv(
        tmp_path / "dev.csv", index=False
    )
    tab = make_table(
        tmp_path,
        [{"name": "D", "las_file": "d.las", "lon": LON0 + 0.05, "lat": LAT0 + 0.05,
          "elevation_m": 0.0, "deviation_file": "dev.csv"}],
    )  # fmt: skip
    wells, _ = load_wells(WellsConfig(las_dir=tmp_path, table=tab, smoothing_m=0), grid)
    w = wells[0]
    assert w.depth_km[-1] == pytest.approx(md[-1] * np.cos(np.radians(30)) / 1e3, rel=1e-6)
    assert w.east_km[-1] == pytest.approx(md[-1] * 0.5 / 1e3, rel=1e-6)
    assert w.thick_km.sum() == pytest.approx(w.depth_km[-1] - w.depth_km[0], rel=1e-2)


def test_depth_reference_msl(tmp_path, grid):
    md = np.arange(0.0, 500.0, 1.0)
    write_las(tmp_path / "w.las", md, linear_log(md))
    tab = make_table(
        tmp_path,
        [{"name": "W", "las_file": "w.las", "lon": LON0 + 0.05, "lat": LAT0 + 0.05,
          "elevation_m": 300.0}],
    )  # fmt: skip
    kb, _ = load_wells(WellsConfig(las_dir=tmp_path, table=tab), grid)
    msl, _ = load_wells(WellsConfig(las_dir=tmp_path, table=tab, depth_reference="MSL"), grid)
    assert kb[0].depth_km[0] == pytest.approx(-0.3)
    assert msl[0].depth_km[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------
def test_backus_vs_average():
    vp = np.array([2.0, 4.0])
    vs = vp / 1.7
    rho = np.array([2.0, 2.0])
    w = np.array([1.0, 1.0])
    b = backus_average(vp, vs, rho, w)
    # constant density: Backus Vp = sqrt(harmonic mean of vp^2)
    assert b[0] == pytest.approx(np.sqrt(2.0 / (1 / 4 + 1 / 16)))
    assert b[2] == pytest.approx(2.0)
    avg = 1.0 / np.mean(1.0 / vp)  # slowness average
    assert b[0] != pytest.approx(avg, rel=1e-3)
    # homogeneous block is unchanged
    h = backus_average(np.full(3, 3.0), np.full(3, 1.7), np.full(3, 2.3), np.ones(3))
    assert h == pytest.approx((3.0, 1.7, 2.3))


def test_upscale_methods_on_layered_log(tmp_path, grid):
    md = np.arange(0.0, 3000.0, 1.0)
    v = np.where((md // 10) % 2 == 0, 2.0, 4.0)  # 10 m layers
    write_las(tmp_path / "w.las", md, 304.8 / v, rhob=np.full(md.size, 2.2))
    tab = make_table(tmp_path, [{"name": "W", "las_file": "w.las", "lon": LON0 + 0.05,
                                 "lat": LAT0 + 0.05}])  # fmt: skip
    res = {}
    for m in ("backus", "average", "none"):
        cfg = WellsConfig(
            las_dir=tmp_path, table=tab, smoothing_m=0, despike_window_m=0, upscale=m,
            max_distance_km=100,
        )  # fmt: skip
        res[m] = build_wells_model(cfg, grid).profiles.set_index("depth_km")["vp"]
    mid = res["backus"].index[(res["backus"].index > 1.0) & (res["backus"].index < 2.5)]
    assert np.allclose(res["average"][mid], 1 / np.mean([0.5, 0.25]), rtol=0.05)
    assert np.allclose(res["backus"][mid], np.sqrt(2 / (1 / 4 + 1 / 16)), rtol=0.05)
    assert (res["backus"][mid] < res["average"][mid]).all()  # harmonic mean of v^2


# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pts():
    x, y = np.meshgrid(np.linspace(-8, 8, 4), np.linspace(-8, 8, 4))
    return x.ravel(), y.ravel()


@pytest.mark.parametrize("method", ["kriging", "idw", "nearest"])
def test_interp_exact_at_wells(pts, method):
    px, py = pts
    v = 3.0 + 0.1 * px + 0.05 * py
    out, d = interpolate_lateral(px[:, None], py[:, None], px, py, v, method=method)
    np.testing.assert_allclose(out[:, 0], v, atol=1e-6)
    assert d.max() < 1e-6


def test_kriging_recovers_linear_field(pts):
    px, py = pts
    f = lambda x, y: 3.0 + 0.1 * x + 0.05 * y  # noqa: E731
    xn, yn = np.meshgrid(np.linspace(-6, 6, 13), np.linspace(-6, 6, 13))
    vg = {"model": "spherical", "range_km": 60.0, "sill": 1.0, "nugget": 0.0}
    k, _ = interpolate_lateral(xn, yn, px, py, f(px, py), method="kriging", variogram=vg)
    assert np.abs(k - f(xn, yn)).max() < 0.05  # a few % of the 1.5 km/s range
    i, _ = interpolate_lateral(xn, yn, px, py, f(px, py), method="idw", power=2.0)
    assert np.abs(i - f(xn, yn)).max() < 0.25
    assert np.abs(k - f(xn, yn)).mean() < np.abs(i - f(xn, yn)).mean()
    for model in ("exponential", "gaussian"):
        vg2 = dict(vg, model=model)
        k2, _ = interpolate_lateral(xn, yn, px, py, f(px, py), method="kriging", variogram=vg2)
        assert np.abs(k2 - f(xn, yn)).max() < 0.15


def test_nearest_and_single_well():
    xn, yn = np.meshgrid(np.linspace(0, 10, 5), np.zeros(1))
    out, d = interpolate_lateral(xn, yn, np.array([0.0, 10.0]), np.array([0.0, 0.0]),
                                 np.array([1.0, 2.0]), method="nearest")  # fmt: skip
    assert out[0].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    one, _ = interpolate_lateral(xn, yn, np.array([3.0]), np.array([0.0]), np.array([7.0]))
    assert (one == 7.0).all()
    with pytest.raises(WellsError):
        interpolate_lateral(xn, yn, np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                            np.array([1.0, 2.0]), method="cubic")  # fmt: skip


def test_build_wells_model_laterally_varying(tmp_path, grid):
    rows = []
    for i, (dlon, dlat, shift) in enumerate(
        [(0.0, 0.0, -0.2), (0.12, 0.0, 0.2), (0.0, 0.1, -0.2), (0.12, 0.1, 0.2)]
    ):
        md = np.arange(0.0, 6000.0, 2.0)
        z = md / 1000.0
        write_las(tmp_path / f"w{i}.las", md, 304.8 / (vp_true(z) + shift))
        rows.append({"name": f"W{i}", "las_file": f"w{i}.las", "lon": LON0 + dlon,
                     "lat": LAT0 + dlat})  # fmt: skip
    tab = make_table(tmp_path, rows)
    cfg = WellsConfig(
        las_dir=tmp_path, table=tab, smoothing_m=0.0, interp="idw", max_distance_km=500,
        upscale="average",
    )  # fmt: skip
    r = build_wells_model(cfg, grid, background=4.0)
    ds = r.dataset
    assert set(ds.data_vars) >= {"vp", "vs", "mask", "weight"}
    assert ds.vp.shape == grid.shape_full
    assert ds["mask"].sel(depth=slice(0.3, 4.0)).all()
    k = 5
    z = ds.depth.values[k]
    west = float(ds.vp.isel(depth=k, lon=0).mean())
    east = float(ds.vp.isel(depth=k, lon=-1).mean())
    assert west < vp_true(z) < east
    assert float(ds.vs.isel(depth=k).mean()) == pytest.approx(
        float(ds.vp.isel(depth=k).mean()) / 1.75, rel=1e-6
    )
    assert set(r.profiles.well) == {"W0", "W1", "W2", "W3"}
    assert len(r.logs) > 0
    assert any(f.code == "gardner" for f in r.findings)


def test_coverage_mask_and_taper(tmp_path, grid):
    md = np.arange(0.0, 2000.0, 2.0)
    write_las(tmp_path / "w.las", md, linear_log(md))
    tab = make_table(tmp_path, [{"name": "W", "las_file": "w.las", "lon": LON0, "lat": LAT0}])
    cfg = WellsConfig(
        las_dir=tmp_path, table=tab, smoothing_m=0.0, interp="idw", max_distance_km=4.0,
        taper_km=2.0,
    )  # fmt: skip
    ds = build_wells_model(cfg, grid, background=lambda z: 6.0 + 0 * z).dataset
    wgt = ds.weight.values
    assert wgt.max() == pytest.approx(1.0) and wgt.min() == 0.0
    assert ds["mask"].values.sum() < wgt.size  # not everywhere
    k = int(np.argmin(np.abs(ds.depth.values - 1.0)))
    iy, ix = np.unravel_index(np.argmax(wgt[k]), wgt[k].shape)
    assert wgt[k, iy, ix] == pytest.approx(1.0)
    assert ds.vp.values[k, iy, ix] == pytest.approx(vp_true(ds.depth.values[k]), abs=0.05)
    # below the log bottom: pure background, mask False
    deep = ds.vp.isel(depth=-1).values
    assert np.allclose(deep, 6.0) and not ds["mask"].isel(depth=-1).any()
    # without a background the uncovered nodes are NaN
    ds2 = build_wells_model(cfg, grid).dataset
    assert np.isnan(ds2.vp.isel(depth=-1).values).all()


def test_deviated_well_position_varies_with_depth(tmp_path, grid):
    md = np.arange(0.0, 4000.0, 2.0)
    write_las(tmp_path / "d.las", md, np.full(md.size, 100.0))
    pd.DataFrame({"md": [0, 4000.0], "incl": [60, 60.0], "azi": [90, 90.0]}).to_csv(
        tmp_path / "dev.csv", index=False
    )
    tab = make_table(
        tmp_path,
        [{"name": "D", "las_file": "d.las", "lon": LON0, "lat": LAT0 + 0.05,
          "deviation_file": "dev.csv"}],
    )  # fmt: skip
    cfg = WellsConfig(las_dir=tmp_path, table=tab, smoothing_m=0.0, max_distance_km=1.5,
                      taper_km=0.0)  # fmt: skip
    ds = build_wells_model(cfg, grid).dataset
    # the constrained column moves east with depth (tan 60 deg ~ 1.7 km east per km)
    k1 = int(np.argmin(np.abs(ds.depth.values - 0.5)))
    k2 = int(np.argmin(np.abs(ds.depth.values - 2.0)))
    lon_shallow = ds.lon.values[ds.mask.isel(depth=k1).any("lat").values]
    lon_deep = ds.lon.values[ds.mask.isel(depth=k2).any("lat").values]
    assert lon_shallow.size and lon_deep.size
    assert lon_deep.mean() > lon_shallow.mean()


# ---------------------------------------------------------------------------------------
def _gradient_wells(tmp_path, offsets):
    rows = []
    for i, (dlon, dlat) in enumerate(offsets):
        md = np.arange(0.0, 4200.0, 2.0)
        z = md / 1000.0
        v = np.where(z < 1.5, 2.0 + 0.6 * z, 3.5 + 0.3 * (z - 1.5))
        rng = np.random.default_rng(i)
        v = v + rng.normal(0, 0.01, v.size)
        write_las(tmp_path / f"g{i}.las", md, 304.8 / v, dts=304.8 / (v / 1.8))
        rows.append({"name": f"G{i}", "las_file": f"g{i}.las", "lon": LON0 + dlon,
                     "lat": LAT0 + dlat})  # fmt: skip
    return make_table(tmp_path, rows)


def _horizons(grid):
    shp = (grid.lat_full.size, grid.lon_full.size)
    hz = [
        {"name": "U1", "vp": {"law": "gradient", "v0": 1.0, "k": 0.1}},
        {"name": "U2", "vp": {"law": "gradient", "v0": 1.0, "k": 0.1}},
    ]
    surf = {"U1": np.full(shp, 1.5), "U2": np.full(shp, 4.0)}
    return hz, surf


def test_geology_wells_recovers_gradient_law(tmp_path, grid):
    tab = _gradient_wells(tmp_path, [(0.03, 0.03), (0.09, 0.07)])
    cfg = WellsConfig(las_dir=tmp_path, table=tab, smoothing_m=0.0, despike_window_m=0.0)
    wells, _ = load_wells(cfg, grid)
    hz, surf = _horizons(grid)
    new, _, fits, findings = wells_horizons(hz, grid, wells, surfaces=surf)
    p1, p2 = fits.params
    assert p1["v0"] == pytest.approx(2.0, abs=0.01) and p1["k"] == pytest.approx(0.6, abs=0.01)
    assert p2["v0"] == pytest.approx(3.5, abs=0.02) and p2["k"] == pytest.approx(0.3, abs=0.02)
    assert (fits.r2 > 0.99).all() and (fits.rms_km_s < 0.05).all()
    assert new[0]["vpvs"] == pytest.approx(1.8, abs=0.02)
    assert any(f.code == "unit_fit" for f in findings)
    # input configuration is untouched
    assert hz[0]["vp"]["v0"] == 1.0

    ds, fits2, f2 = build_geology_wells_model(
        hz, grid, cfg, background=5.0, wells=wells, surfaces=surf
    )
    z = ds.depth.values
    prof = ds.vp.isel(lat=2, lon=2).values
    k = int(np.argmin(np.abs(z - 1.0)))
    # unit_top reference: U1 starts at the surface (0 km depth)
    assert prof[k] == pytest.approx(2.0 + 0.6 * z[k], abs=0.02)
    k2 = int(np.argmin(np.abs(z - 3.0)))
    assert prof[k2] == pytest.approx(3.5 + 0.3 * (z[k2] - 1.5), abs=0.03)
    # equals a direct geology build with the fitted laws
    ref, _ = build_geology_model(new, grid, 5.0, surfaces=surf, background_vpvs=cfg.vpvs)
    np.testing.assert_allclose(ds.vp.values, ref.vp.values, rtol=1e-12)
    assert "unit_fits" in ds.attrs


def test_geology_wells_constant_exponential_and_missing(tmp_path, grid):
    tab = _gradient_wells(tmp_path, [(0.05, 0.05)])
    cfg = WellsConfig(las_dir=tmp_path, table=tab, smoothing_m=0.0, despike_window_m=0.0)
    wells, _ = load_wells(cfg, grid)
    shp = (grid.lat_full.size, grid.lon_full.size)
    hz = [
        {"name": "A", "vp": {"law": "constant", "v0": 1.0}},
        {"name": "B", "vp": {"law": "exponential", "v0": 2.0, "vmax": 6.0, "lambda_km": 1.0}},
        {"name": "C", "vp": {"law": "constant", "v0": 7.5}},
    ]
    surf = {"A": np.full(shp, 0.2), "B": np.full(shp, 4.0), "C": np.full(shp, 4.9)}
    new, _, fits, findings = wells_horizons(hz, grid, wells, surfaces=surf)
    assert new[0]["vp"]["v0"] == pytest.approx(2.0 + 0.6 * 0.1, abs=0.05)
    assert fits.params[1] is not None and fits.rms_km_s[1] < 0.2
    # unit C (4.0-4.9 km) is only logged to 4.2 km: 100 samples, still fitted;
    # a unit with no log coverage keeps its law and reports it
    surf["C"] = np.full(shp, 4.9)
    hz.append({"name": "D", "vp": {"law": "constant", "v0": 7.7}})
    surf["D"] = np.full(shp, 5.0)
    new, _, fits, findings = wells_horizons(hz, grid, wells, surfaces=surf)
    assert new[3]["vp"]["v0"] == 7.7
    assert any(f.code == "unit_no_wells" for f in findings)


def test_config_schema_defaults():
    c = WellsConfig()
    assert c.upscale == "backus" and c.interp == "kriging" and c.depth_reference == "KB"
    assert c.variogram.model == "spherical"
    with pytest.raises(ValueError):
        WellsConfig(interp="cubic")
