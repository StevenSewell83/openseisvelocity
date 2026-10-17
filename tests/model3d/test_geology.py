"""Tests for osv.model3d.horizons_io and osv.model3d.geology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from osv.config.schema import Horizon, Model3DConfig, UserModelConfig
from osv.model3d.geology import apply_law, build_geology_model
from osv.model3d.grid import define_grid
from osv.model3d.horizons_io import (
    HorizonReadError,
    read_geotiff,
    read_horizon,
    read_tsurf,
    read_xyz,
)

NZTM = "EPSG:2193"
LAT0, LON0 = -38.55, 176.19


@pytest.fixture(scope="module")
def grid():
    la, lo = np.meshgrid(np.linspace(0, 0.1, 3), np.linspace(0, 0.12, 3), indexing="ij")
    st = pd.DataFrame(
        {
            "lat": LAT0 + la.ravel(),
            "lon": LON0 + lo.ravel(),
            "elevation_m": 300.0,
            "used": True,
        }
    )
    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 0.5, "pad_km": 2.0, "zmax_km": 5.0},
        propgrid_refine=2,
    )
    return define_grid(st, None, cfg)


def _nztm_nodes(grid):
    tr = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
    lo, la = np.meshgrid(grid.lon_full, grid.lat_full)
    return tr.transform(lo, la)


def _plane_depth_km(e, n, e0, n0):
    """Dipping plane: 1.0 km at the origin, deepening 0.1 m per m northward."""
    return 1.0 + 0.1 * (n - n0) / 1000.0


def _expected(grid):
    e, n = _nztm_nodes(grid)
    return _plane_depth_km(e, n, e.mean(), n.mean()), e.mean(), n.mean()


def _lattice(grid, margin=2000.0, step=500.0):
    e, n = _nztm_nodes(grid)
    ee = np.arange(e.min() - margin, e.max() + margin + step, step)
    nn = np.arange(n.min() - margin, n.max() + margin + step, step)
    return np.meshgrid(ee, nn)


# ---------------------------------------------------------------------------- readers


def test_read_xyz_nztm_header_and_elevation(grid, tmp_path):
    exp, e0, n0 = _expected(grid)
    E, N = _lattice(grid)
    z_depth_m = 1000.0 * _plane_depth_km(E, N, e0, n0)
    p = tmp_path / "h.xyz"
    rows = "\n".join(
        f"{x:.2f} {y:.2f} {-z:.3f}"
        for x, y, z in zip(E.ravel(), N.ravel(), z_depth_m.ravel(), strict=True)
    )
    p.write_text("# synthetic\nEasting Northing Elev\n" + rows + "\n")
    z = read_xyz(p, grid, crs=NZTM, z_convention="elevation", units="m")
    assert z.shape == (grid.lat_full.size, grid.lon_full.size)
    np.testing.assert_allclose(z, exp, atol=1e-6)
    # dispatcher by suffix
    z2 = read_horizon(p, grid, crs=NZTM, z_convention="elevation", units="m")
    np.testing.assert_allclose(z2, exp, atol=1e-6)


def test_read_xyz_no_overlap_errors(grid, tmp_path):
    p = tmp_path / "far.xyz"
    p.write_text("0 0 1\n1 0 1\n0 1 1\n1 1 1\n")
    with pytest.raises(HorizonReadError, match="overlap"):
        read_xyz(p, grid, crs=NZTM)


def test_read_xyz_partial_coverage_is_nan(grid, tmp_path):
    e, n = _nztm_nodes(grid)
    mid = n.mean()
    E, N = np.meshgrid(
        np.linspace(e.min() - 1e3, e.max() + 1e3, 12), np.linspace(n.min() - 1e3, mid, 12)
    )
    p = tmp_path / "half.xyz"
    p.write_text("\n".join(f"{a} {b} 1.5" for a, b in zip(E.ravel(), N.ravel(), strict=True)))
    z = read_xyz(p, grid, crs=NZTM, units="km")
    assert np.isfinite(z[0]).all() and np.isnan(z[-1]).all()
    np.testing.assert_allclose(z[np.isfinite(z)], 1.5)


def test_read_tsurf_nztm(grid, tmp_path):
    exp, e0, n0 = _expected(grid)
    E, N = _lattice(grid, step=1000.0)
    ny, nx = E.shape
    z_m = 1000.0 * _plane_depth_km(E, N, e0, n0)
    lines = ["GOCAD TSurface 1", "HEADER {", "name: base", "}", "ZPOSITIVE Depth"]
    lines.append("TFACE")
    ids = np.arange(ny * nx).reshape(ny, nx)
    for j in range(ny):
        for i in range(nx):
            lines.append(f"VRTX {ids[j, i]} {E[j, i]:.2f} {N[j, i]:.2f} {z_m[j, i]:.3f}")
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = ids[j, i], ids[j, i + 1], ids[j + 1, i], ids[j + 1, i + 1]
            lines += [f"TRGL {a} {b} {c}", f"TRGL {b} {d} {c}"]
    lines.append("END")
    p = tmp_path / "base.ts"
    p.write_text("\n".join(lines))
    z = read_tsurf(p, grid, crs=NZTM, units="m")
    np.testing.assert_allclose(z, exp, atol=1e-6)


def test_read_tsurf_bad(grid, tmp_path):
    p = tmp_path / "bad.ts"
    p.write_text("GOCAD TSurface 1\nVRTX 1 0 0 0\nEND\n")
    with pytest.raises(HorizonReadError, match="TRGL"):
        read_tsurf(p, grid)


def test_read_geotiff_nztm(grid, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.errors import CRSError

    from osv._proj import ensure_proj_data

    ensure_proj_data()
    from rasterio.transform import from_origin

    exp, e0, n0 = _expected(grid)
    e, n = _nztm_nodes(grid)
    res = 250.0
    x0, y1 = e.min() - 2000.0, n.max() + 2000.0
    nx = int((e.max() - e.min() + 4000.0) / res) + 1
    ny = int((n.max() - n.min() + 4000.0) / res) + 1
    xc = x0 + res * (np.arange(nx) + 0.5)
    yc = y1 - res * (np.arange(ny) + 0.5)
    XX, YY = np.meshgrid(xc, yc)
    data = (-1000.0 * _plane_depth_km(XX, YY, e0, n0)).astype("float32")  # elevation, m
    data[0, 0] = -9999.0
    p = tmp_path / "base.tif"
    try:
        rasterio.crs.CRS.from_user_input(NZTM)
    except CRSError as exc:  # e.g. a stale PROJ_DATA pointing to another PROJ
        pytest.skip(f"rasterio cannot resolve EPSG codes here: {str(exc)[:80]}")
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="float32",
        crs=NZTM,
        transform=from_origin(x0, y1, res, res),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    z = read_geotiff(p, grid, z_convention="elevation", units="m")
    np.testing.assert_allclose(z, exp, atol=1e-4)


def test_read_geotiff_missing_rasterio(grid, tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "rasterio":
            raise ImportError("no rasterio")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="rasterio"):
        read_geotiff(tmp_path / "x.tif", grid)


def test_unknown_format(grid, tmp_path):
    p = tmp_path / "h.foo"
    p.write_text("1 2 3")
    with pytest.raises(HorizonReadError, match="unknown surface format"):
        read_horizon(p, grid)


# ---------------------------------------------------------------------------- geology


def _flat(grid, depth):
    return np.full((grid.lat_full.size, grid.lon_full.size), depth)


def _cfg():
    return [
        Horizon(name="Huka", surface="a.xyz", vp={"law": "constant", "v0": 2.0}, vpvs=2.0),
        Horizon(
            name="Waiora",
            surface="b.xyz",
            vp={"law": "gradient", "v0": 3.0, "k": 0.5},
            vs={"law": "constant", "v0": 1.7},
        ),
    ]


def test_two_horizon_model_velocities(grid):
    surfaces = {"Huka": _flat(grid, 1.0), "Waiora": _flat(grid, 3.0)}
    ds, findings = build_geology_model(
        _cfg(), grid, background=lambda z: 5.0 + 0.0 * z, surfaces=surfaces
    )
    z = ds.depth.values[:, None, None] * np.ones(ds.vp.shape)
    vp, vs, uid = ds.vp.values, ds.vs.values, ds.unit_id.values
    u0, u1, ub = z < 1.0, (z >= 1.0) & (z < 3.0), z >= 3.0
    assert (uid[u0] == 0).all() and (uid[u1] == 1).all() and (uid[ub] == -1).all()
    np.testing.assert_allclose(vp[u0], 2.0)
    np.testing.assert_allclose(vs[u0], 1.0)  # vpvs = 2
    np.testing.assert_allclose(vp[u1], 3.0 + 0.5 * (z[u1] - 1.0))
    np.testing.assert_allclose(vs[u1], 1.7)
    np.testing.assert_allclose(vp[ub], 5.0)
    np.testing.assert_allclose(vs[ub], 5.0 / 1.75)
    assert ds.attrs["unit_names"] == ["Huka", "Waiora"]
    assert not [f for f in findings if f.level == "warning"]


def test_reference_options_and_exponential(grid):
    z = np.array([0.5, 2.0])
    gs = {"law": "gradient", "v0": 2.0, "k": 1.0}
    assert apply_law(gs, z)[1] == pytest.approx(4.0)
    ex = {"law": "exponential", "v0": 2.0, "vmax": 5.0, "lambda_km": 1.0}
    np.testing.assert_allclose(apply_law(ex, np.array([0.0, 1e3])), [2.0, 5.0])
    assert apply_law(ex, np.array([1.0]))[0] == pytest.approx(5.0 - 3.0 * np.exp(-1.0))
    # reference: sea_level for the second unit
    cfg = [
        {"name": "a", "surface": "", "vp": {"law": "constant", "v0": 2.0}},
        {
            "name": "b",
            "surface": "",
            "vp": {"law": "gradient", "v0": 3.0, "k": 1.0, "reference": "sea_level"},
        },
    ]
    surf = {"a": _flat(grid, 1.0), "b": _flat(grid, 4.0)}
    ds, _ = build_geology_model(cfg, grid, background=6.0, surfaces=surf)
    sel = ds.unit_id.values == 1
    zz = ds.depth.values[:, None, None] * np.ones(ds.vp.shape)
    np.testing.assert_allclose(ds.vp.values[sel], 3.0 + zz[sel])


def test_crossing_horizons_clipped(grid):
    base2 = _flat(grid, 3.0)
    base2[:2, :] = 0.5  # crosses above base 1 (1.0 km) at 2 rows
    n_bad = 2 * grid.lon_full.size
    ds, findings = build_geology_model(
        _cfg(), grid, background=5.0, surfaces={"Huka": _flat(grid, 1.0), "Waiora": base2}
    )
    f = [x for x in findings if x.code == "crossing_horizons"]
    assert len(f) == 1 and f[0].n_nodes == n_bad
    # zero-thickness unit 1 in those columns: only unit 0 then background
    col = ds.unit_id.values[:, 0, 0]
    assert set(np.unique(col)) == {0, -1}
    assert (ds.unit_id.values[:, -1, 0] == 1).any()


def test_nan_columns_use_background(grid):
    b1 = _flat(grid, 1.0)
    b1[:, 0] = np.nan
    ds, findings = build_geology_model(
        _cfg(), grid, background=4.0, surfaces={"Huka": b1, "Waiora": _flat(grid, 3.0)}
    )
    assert (ds.unit_id.values[:, :, 0] == -1).all()
    np.testing.assert_allclose(ds.vp.values[:, :, 0], 4.0)
    assert {"horizon_undefined", "background_columns"} <= {f.code for f in findings}


def test_no_background_gives_nan_and_finding(grid):
    ds, findings = build_geology_model(
        _cfg(), grid, surfaces={"Huka": _flat(grid, 1.0), "Waiora": _flat(grid, 3.0)}
    )
    assert np.isnan(ds.vp.values[ds.unit_id.values == -1]).all()
    assert any(f.code == "no_background" for f in findings)


def test_build_from_files_and_config(grid, tmp_path):
    for name, d in (("a.xyz", 1050.0), ("b.xyz", 3050.0)):
        E, N = _lattice(grid, step=2000.0)
        (tmp_path / name).write_text(
            "\n".join(f"{x} {y} {d}" for x, y in zip(E.ravel(), N.ravel(), strict=True))
        )
    um = UserModelConfig(
        horizons=[
            {"name": "A", "surface": "a.xyz", "crs": NZTM, "vp": {"law": "constant", "v0": 2.5}},
            {"name": "B", "surface": "b.xyz", "crs": NZTM, "vp": {"law": "constant", "v0": 4.0}},
        ]
    )
    ds, _ = build_geology_model(um.horizons, grid, background=5.5, base_dir=Path(tmp_path))
    z = ds.depth.values[:, None, None] * np.ones(ds.vp.shape)
    np.testing.assert_allclose(ds.vp.values[z < 1.05], 2.5)
    np.testing.assert_allclose(ds.vp.values[(z >= 1.05) & (z < 3.05)], 4.0)
    np.testing.assert_allclose(ds.vp.values[z >= 3.05], 5.5)
