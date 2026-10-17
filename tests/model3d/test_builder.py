"""Tests for osv.model3d.builder (blend, taper, smoothing, checks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.config.schema import Model3DConfig
from osv.model3d.builder import (
    StartingInputs,
    _smooth,
    blend_weight,
    build_starting_model,
)
from osv.model3d.extrude import extrude_1d
from osv.model3d.grid import define_grid

REPO = Path(__file__).resolve().parents[2]
RUN01 = REPO / "projects" / "rk-nm-2017" / "runs" / "01_prep"
RUN03 = REPO / "projects" / "rk-nm-2017" / "runs" / "03_velest_analyse"
LAT0, LON0 = -38.5, 176.2

MODEL = pd.DataFrame(
    {
        "depth_top_km": [-0.6, 0.2, 1.0, 3.0],
        "vp": [1.5, 3.0, 4.0, 6.0],
        "vpvs": [1.7] * 4,
    }
)
GRID_CFG = {"dx_km": 1.0, "dy_km": 1.0, "dz_km": 0.5, "pad_km": 2.0, "zmax_km": 6.0}


@pytest.fixture(scope="module")
def grid():
    la, lo = np.meshgrid(np.linspace(0, 0.2, 3), np.linspace(0, 0.24, 3), indexing="ij")
    st = pd.DataFrame(
        {
            "lat": LAT0 + la.ravel(),
            "lon": LON0 + lo.ravel(),
            "elevation_m": np.linspace(300.0, 600.0, la.size),
        }
    )
    return define_grid(st, None, Model3DConfig(grid=GRID_CFG, propgrid_refine=2))


def make_cfg(**kw):
    um = {"smoothing_km": 0.0, "blend_taper_km": 2.0}
    um.update(kw.pop("user_model", {}))
    return Model3DConfig(grid=GRID_CFG, propgrid_refine=2, user_model=um, **kw)


def box_user_grid(grid, vp=5.0, vs=None, lat_frac=(0.0, 0.5), depth_from=1.0):
    """User grid covering the southern half (lat) below ``depth_from`` km, constant ``vp``."""
    shp = grid.shape_full
    cov = np.zeros(shp, dtype=bool)
    nlat = shp[1]
    cov[:, : int(nlat * lat_frac[1]), :] = True
    cov &= (grid.depth_full >= depth_from)[:, None, None]
    dims = ("depth", "lat", "lon")
    ds = xr.Dataset(
        {"vp": (dims, np.where(cov, vp, np.nan))},
        coords={"depth": grid.depth_full, "lat": grid.lat_full, "lon": grid.lon_full},
    )
    if vs is not None:
        ds["vs"] = (dims, np.where(cov, vs, np.nan))
    return ds, xr.DataArray(cov, dims=dims, coords=ds.coords)


# ---------------------------------------------------------------------------------------
def test_velest_1d_mode(grid):
    sm = build_starting_model(grid, make_cfg(), MODEL)
    ref = extrude_1d(MODEL, grid, make_cfg().extrusion.mode)
    np.testing.assert_allclose(sm.dataset.vp, ref.vp)
    np.testing.assert_allclose(sm.dataset.vs, ref.vs)
    assert float(sm.dataset.weight.max()) == 0.0
    assert sm.summary["influenced_fraction"] == 0.0
    assert sm.summary["n_nan"] == 0
    assert {"vp", "vs", "vp_unc", "vs_unc", "weight"} <= set(sm.dataset.data_vars)
    vg = sm.to_vgrids()
    assert len(list(vg)) == 2


def test_blend_weight_edges_and_taper():
    mask = np.zeros((5, 20, 20), dtype=bool)
    mask[:, :, 5:] = True
    w = blend_weight(mask, (0.5, 1.0, 1.0), 4.0)
    assert (w[:, :, :5] == 0).all()
    assert w[2, 3, 5] == pytest.approx(0.25)  # one node (1 km) inside / 4 km taper
    assert w[2, 3, 9] == 1.0
    # model boundary is not a coverage edge: no taper at lon index 19 or the depth edges
    assert w[0, 3, 19] == 1.0
    assert (blend_weight(mask, (1, 1, 1), 0.0) == mask).all()
    assert (blend_weight(np.ones((3, 3, 3), bool), (1, 1, 1), 5.0) == 1).all()


def test_user_grid_blend_taper_continuity(grid):
    ug = box_user_grid(grid, vp=7.0, vs=4.0)
    cfg = make_cfg(source="user_grid", user_model={"grid_file": "x.nc", "blend_taper_km": 3.0})
    sm = build_starting_model(grid, cfg, MODEL, StartingInputs(user_grid=ug))
    ds = sm.dataset
    w = ds.weight.values
    cov = ug[1].values
    assert (w[~cov] == 0).all()
    assert (ds.vp.values[~cov] == ds.vp_bg.values[~cov]).all()
    assert w.max() == 1.0
    # weight is continuous: neighbouring nodes differ by at most spacing/taper
    sp = (grid.dz_km, grid.dy_km, grid.dx_km)
    for ax, s in enumerate(sp):
        assert np.abs(np.diff(w, axis=ax)).max() <= s / 3.0 + 1e-9
        # so velocity steps are bounded by |dv| * s / taper (no jump at the coverage edge)
        dv = np.abs(np.diff(ds.vp.values, axis=ax)).max()
        bound = (
            np.abs(7.0 - ds.vp_bg.values).max() * s / 3.0
            + np.abs(np.diff(ds.vp_bg.values, axis=ax)).max()
        )
        assert dv <= bound + 1e-9
    # blended = w*user + (1-w)*bg exactly
    exp = w * np.where(cov, 7.0, 0.0) + (1 - w) * ds.vp_bg.values
    np.testing.assert_allclose(ds.vp.values, exp)
    # Vs from the user model where provided
    full = w == 1
    np.testing.assert_allclose(ds.vs.values[full], 4.0)
    assert 0 < sm.summary["influenced_fraction"] < 1
    assert sm.summary["max_abs_dvp"] > 1.0


def test_user_grid_vs_from_background_ratio(grid):
    ug = box_user_grid(grid, vp=5.0)
    cfg = make_cfg(source="user_grid", user_model={"grid_file": "x.nc", "blend_taper_km": 0.0})
    sm = build_starting_model(grid, cfg, MODEL, StartingInputs(user_grid=ug))
    ds = sm.dataset
    full = ds.weight.values == 1
    np.testing.assert_allclose((ds.vp / ds.vs).values[full], 1.7)
    assert any(f.code == "vs_from_vpvs" for f in sm.findings)


def test_user_grid_from_csv_file(grid, tmp_path):
    # regular lattice covering the southern half of the model, lon/lat, depth km
    lat = np.linspace(grid.lat_full[0] - 0.01, grid.lat[len(grid.lat) // 2], 6)
    lon = np.linspace(grid.lon_full[0] - 0.01, grid.lon_full[-1] + 0.01, 6)
    z = np.linspace(0.0, 6.0, 7)
    zz, la, lo = np.meshgrid(z, lat, lon, indexing="ij")
    pd.DataFrame(
        {"x": lo.ravel(), "y": la.ravel(), "z": zz.ravel(), "vp": 4.0 + 0.2 * zz.ravel()}
    ).to_csv(tmp_path / "g.csv", index=False)
    cfg = make_cfg(source="user_grid", user_model={"grid_file": "g.csv"})
    sm = build_starting_model(grid, cfg, MODEL, StartingInputs(base_dir=tmp_path))
    assert 0 < sm.summary["influenced_fraction"] < 1
    assert sm.summary["n_nan"] == 0


def test_geology_source(grid):
    shp = (grid.lat_full.size, grid.lon_full.size)
    sfc = np.full(shp, 2.0)
    sfc[: shp[0] // 2] = np.nan  # undefined in the south -> background there
    cfg = make_cfg(
        source="geology",
        user_model={
            "horizons": [
                {
                    "name": "A",
                    "surface": "a.xyz",
                    "vp": {"law": "constant", "v0": 2.5},
                    "vpvs": 1.9,
                }
            ],
            "blend_taper_km": 1.0,
        },
    )
    st = pd.DataFrame({"elevation_m": [300.0]})
    sm = build_starting_model(grid, cfg, MODEL, StartingInputs(surfaces={"A": sfc}, stations=st))
    ds = sm.dataset
    assert "unit_id" in ds
    assert ds.attrs["unit_names"] == ["A"]
    north = ds.isel(lat=-2, lon=3)
    inside = (north.unit_id == 0) & (north.weight == 1)
    assert inside.any()
    np.testing.assert_allclose(north.vp.values[inside.values], 2.5)
    assert (ds.weight.isel(lat=0) == 0).all()
    assert 0 < sm.summary["influenced_fraction"] < 1


def test_smoothing_preserves_mean_and_smooths(grid):
    rng = np.random.default_rng(1)
    a = rng.normal(4.0, 0.5, grid.shape_full)
    sp = (grid.dz_km, grid.dy_km, grid.dx_km)
    s = _smooth(a, sp, 1.5)
    assert s.mean() == pytest.approx(a.mean(), rel=1e-12)
    assert s.std() < a.std()
    assert _smooth(a, sp, 0.0) is a
    holes = a.copy()
    holes[3:5] = np.nan
    sh = _smooth(holes, sp, 1.0)
    assert np.isnan(sh[3:5]).all() and np.isfinite(sh[:3]).all()

    ug = box_user_grid(grid, vp=7.0)
    base = {"source": "user_grid"}
    um = {"grid_file": "x.nc", "blend_taper_km": 2.0}
    raw = build_starting_model(
        grid, make_cfg(**base, user_model=um), MODEL, StartingInputs(user_grid=ug)
    )
    sm_b = build_starting_model(
        grid,
        make_cfg(**base, user_model={**um, "smoothing_km": 1.0}),
        MODEL,
        StartingInputs(user_grid=ug),
    )
    sm_u = build_starting_model(
        grid,
        make_cfg(**base, user_model={**um, "smoothing_km": 1.0, "smoothing_target": "user"}),
        MODEL,
        StartingInputs(user_grid=ug),
    )
    assert sm_b.dataset.vp.mean() == pytest.approx(raw.dataset.vp.mean(), rel=1e-3)
    assert sm_b.summary["max_gradient_km_s_per_km"] < raw.summary["max_gradient_km_s_per_km"]
    # user-only smoothing leaves the pure background untouched
    bgm = sm_u.dataset.weight.values == 0
    np.testing.assert_allclose(sm_u.dataset.vp.values[bgm], raw.dataset.vp.values[bgm])


def test_checks_fire(grid):
    ug = box_user_grid(grid, vp=12.0, vs=2.0)  # Vp too high, Vp/Vs = 6
    cfg = make_cfg(source="user_grid", user_model={"grid_file": "x.nc", "blend_taper_km": 0.0})
    sm = build_starting_model(grid, cfg, MODEL, StartingInputs(user_grid=ug))
    codes = {f.code: f for f in sm.findings}
    assert codes["vp_range"].n_nodes > 0
    assert codes["vpvs_range"].n_nodes > 0
    assert codes["large_gradient"].n_nodes > 0
    assert sm.summary["max_gradient_km_s_per_km"] > 3.0
    assert sm.summary["n_vp_out_of_range"] == codes["vp_range"].n_nodes
    assert codes["user_influence"].level == "info"
    # configurable thresholds
    cfg2 = make_cfg(
        source="user_grid",
        user_model={"grid_file": "x.nc", "blend_taper_km": 0.0},
        checks={"vp_max_km_s": 20.0, "vpvs_max": 10.0, "max_gradient_km_s_per_km": 100.0},
    )
    sm2 = build_starting_model(grid, cfg2, MODEL, StartingInputs(user_grid=ug))
    assert not {"vp_range", "vpvs_range", "large_gradient"} & {f.code for f in sm2.findings}


def test_nan_and_background_none(grid):
    ug = box_user_grid(grid, vp=5.0)
    cfg = make_cfg(source="user_grid", user_model={"grid_file": "x.nc", "background": "none"})
    sm = build_starting_model(grid, cfg, MODEL, StartingInputs(user_grid=ug))
    assert sm.summary["n_nan"] > 0
    assert any(f.code == "nan_velocity" and f.level == "error" for f in sm.findings)
    # where the user model exists with no background it stands alone
    cov = ug[1].values
    np.testing.assert_allclose(sm.dataset.vp.values[cov], 5.0)


def test_constant_background(grid):
    ug = box_user_grid(grid, vp=5.0)
    cfg = make_cfg(
        source="user_grid",
        user_model={
            "grid_file": "x.nc",
            "background": "constant",
            "background_vp_km_s": 3.0,
            "background_vpvs": 1.8,
            "blend_taper_km": 0.0,
        },
    )
    sm = build_starting_model(grid, cfg, None, StartingInputs(user_grid=ug))
    cov = ug[1].values
    np.testing.assert_allclose(sm.dataset.vp.values[~cov], 3.0)
    np.testing.assert_allclose(sm.dataset.vs.values[~cov], 3.0 / 1.8)
    np.testing.assert_allclose(sm.dataset.vs.values[cov], 5.0 / 1.8)
    with pytest.raises(Exception, match="model_1d"):
        build_starting_model(grid, make_cfg(), None)


def test_uncertainty_from_weight(grid):
    ug = box_user_grid(grid, vp=5.0)
    um = {"grid_file": "x.nc", "blend_taper_km": 2.0}
    plain = build_starting_model(
        grid, make_cfg(source="user_grid", user_model=um), MODEL, StartingInputs(user_grid=ug)
    )
    wt = build_starting_model(
        grid,
        make_cfg(source="user_grid", user_model=um, uncertainty_from_weight=True),
        MODEL,
        StartingInputs(user_grid=ug),
    )
    w = wt.dataset.weight.values
    np.testing.assert_allclose(plain.dataset.vp_unc.values, 0.3)
    np.testing.assert_allclose(wt.dataset.vp_unc.values[w == 1], 0.3)
    np.testing.assert_allclose(wt.dataset.vp_unc.values[w == 0], 0.6)
    vg = wt.to_vgrids()
    assert len(list(vg)) == 2


# ---------------------------------------------------------------------------------------
def _write_las(path, md, dt):
    lasio = pytest.importorskip("lasio")
    las = lasio.LASFile()
    las.well["WELL"].value = path.stem
    las.append_curve("DEPT", np.asarray(md), unit="m")
    las.append_curve("DT", np.asarray(dt), unit="US/F")
    with open(path, "w") as f:
        las.write(f, version=2.0)


def _wells_inputs(tmp_path, grid):
    md = np.arange(0.0, 6000.0, 2.0)
    v = 2.0 + 0.6 * md / 1000.0
    rows = []
    for i, (dlat, dlon) in enumerate([(0.0, 0.0), (0.0, 0.2)]):
        _write_las(tmp_path / f"w{i}.las", md, 304.8 / v)
        rows.append(
            {
                "name": f"W{i}",
                "las_file": f"w{i}.las",
                "lat": LAT0 + dlat,
                "lon": LON0 + dlon,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "wells.csv", index=False)


def test_wells_source(grid, tmp_path):
    _wells_inputs(tmp_path, grid)
    cfg = make_cfg(
        source="wells",
        user_model={
            "wells": {
                "las_dir": tmp_path,
                "table": tmp_path / "wells.csv",
                "max_distance_km": 6.0,
                "taper_km": 3.0,
                "smoothing_m": 0.0,
                "interp": "idw",
                "upscale": "average",
            },
            "blend_taper_km": 1.0,
        },
    )
    sm = build_starting_model(grid, cfg, MODEL, StartingInputs(base_dir=tmp_path))
    ds = sm.dataset
    w = ds.weight.values
    assert 0 < sm.summary["influenced_fraction"] < 1
    assert w.max() <= 1.0 and w.min() == 0.0
    # near the wells at depth the well velocity (2 + 0.6 z) dominates
    k = int(np.argmin(np.abs(ds.depth.values - 3.0)))
    j = int(np.argmin(np.abs(ds.lat.values - LAT0)))
    i = int(np.argmin(np.abs(ds.lon.values - LON0)))
    assert w[k, j, i] > 0.9
    assert ds.vp.values[k, j, i] == pytest.approx(2.0 + 0.6 * ds.depth.values[k], abs=0.3)
    assert sm.summary["n_nan"] == 0


def test_geology_wells_source(grid, tmp_path):
    _wells_inputs(tmp_path, grid)
    shp = (grid.lat_full.size, grid.lon_full.size)
    cfg = make_cfg(
        source="geology+wells",
        user_model={
            "horizons": [
                {
                    "name": "U1",
                    "surface": "u1.xyz",
                    "vp": {"law": "constant", "v0": 3.0},
                },
            ],
            "wells": {
                "las_dir": tmp_path,
                "table": tmp_path / "wells.csv",
                "smoothing_m": 0.0,
                "upscale": "average",
            },
        },
    )
    sm = build_starting_model(
        grid,
        cfg,
        MODEL,
        StartingInputs(base_dir=tmp_path, surfaces={"U1": np.full(shp, 3.0)}),
    )
    assert "unit_id" in sm.dataset
    assert sm.summary["influenced_fraction"] > 0.2
    assert sm.summary["n_nan"] == 0


# ---------------------------------------------------------------------------------------
@pytest.mark.integration
def test_real_project_velest_1d():
    if not (RUN03 / "final_model.csv").exists() or not (RUN01 / "stations.parquet").exists():
        pytest.skip("project rk-nm-2017 outputs not available")
    st = pd.read_parquet(RUN01 / "stations.parquet")
    ev = pd.read_parquet(RUN01 / "events.parquet")
    model = pd.read_csv(RUN03 / "final_model.csv")
    cfg = Model3DConfig()
    grid = define_grid(st, ev, cfg, strict=False)
    sm = build_starting_model(grid, cfg, model, StartingInputs(stations=st))
    s = sm.summary
    print(s)
    assert s["n_nan"] == 0
    assert s["influenced_fraction"] == 0.0
    assert len(list(sm.to_vgrids())) == 2
