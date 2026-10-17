"""Model exports (NetCDF, VTK, CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.analysis.exports import VTK_NODATA, export_model, local_km

DEPTH = np.array([0.0, 1.0, 2.0, 3.0])
LAT = np.array([-38.60, -38.58, -38.56])
LON = np.array([176.15, 176.17, 176.19, 176.21, 176.23])


def _model() -> xr.Dataset:
    shp = (len(DEPTH), len(LAT), len(LON))
    vp = (
        3.0
        + 0.5 * DEPTH[:, None, None]
        + 0.1 * np.arange(shp[2])[None, None, :]
        + 0 * np.zeros(shp)
    )
    vp[0, 0, 0] = np.nan
    vs = vp / 1.75
    return xr.Dataset(
        {"vp": (("depth", "lat", "lon"), vp), "vs": (("depth", "lat", "lon"), vs)},
        coords={"depth": DEPTH, "lat": LAT, "lon": LON},
    )


def _mask() -> xr.DataArray:
    m = np.zeros((len(DEPTH), len(LAT), len(LON)), bool)
    m[1:, :, 1:4] = True
    return xr.DataArray(m, dims=("depth", "lat", "lon"))


def _hypo() -> pd.DataFrame:
    return pd.DataFrame(
        {"lat": [-38.59, -38.57], "lon": [176.18, 176.2], "depth_km": [1.5, 2.5], "rms": [0.1, 0.2]}
    )


def test_netcdf_roundtrip(tmp_path) -> None:
    m = _model()
    files = export_model(m, tmp_path, ("netcdf",), masks={"hit": _mask()})
    assert set(files) == {"netcdf"}
    with xr.open_dataset(files["netcdf"]) as back:
        np.testing.assert_allclose(back["vp"].values, m["vp"].values)
        assert back["vp"].attrs["units"] == "km/s"
        assert back["depth"].attrs["positive"] == "down"
        assert back["lat"].attrs["units"] == "degrees_north"
        assert back["hit"].dtype == np.int8
        assert int(back["hit"].sum()) == int(_mask().sum())
        np.testing.assert_allclose(back["vpvs"].values[1:], 1.75)
        # 1-D reference: the lateral mean has zero mean perturbation
        assert abs(float(back["dvp_pct"].isel(depth=2).mean())) < 1e-9
        assert back.attrs["Conventions"].startswith("CF")


def test_legacy_vtk_structure(tmp_path) -> None:
    files = export_model(
        _model(),
        tmp_path,
        ("vtk",),
        masks={"hit": _mask()},
        hypocentres=_hypo(),
        stations=pd.DataFrame({"lat": [-38.57], "lon": [176.19], "depth_km": [-0.3]}),
        vtk_backend="legacy",
    )
    lines = files["vtk"].read_text().splitlines()
    assert lines[0].startswith("# vtk DataFile")
    assert "DATASET STRUCTURED_GRID" in lines
    nx, ny, nz = len(LON), len(LAT), len(DEPTH)
    assert f"DIMENSIONS {nx} {ny} {nz}" in lines
    n = nx * ny * nz
    ip = lines.index(f"POINTS {n} float")
    pts = np.array(" ".join(lines[ip + 1 : ip + 1 + n]).split(), float)
    assert pts.size == 3 * n
    pts = pts.reshape(-1, 3)
    # x fastest, z up (negative depth), origin at the grid centre
    assert pts[1, 0] > pts[0, 0] and pts[1, 1] == pts[0, 1]
    assert pts[0, 2] == 0.0 and pts[-1, 2] == -3.0
    x, y = local_km(LAT[0], LON[1], LAT.mean(), LON.mean())
    assert pts[1, 0] == pytest.approx(float(x), abs=1e-3)
    assert pts[1, 1] == pytest.approx(float(y), abs=1e-3)
    assert f"POINT_DATA {n}" in lines
    for name in ("vp", "vs", "vpvs", "dvp_pct", "hit"):
        assert f"SCALARS {name} float 1" in lines
    i = lines.index("SCALARS vp float 1") + 2
    vals = np.array(" ".join(lines[i : i + 3]).split(), float)
    assert vals[0] == VTK_NODATA  # the NaN node is first (lon, lat, depth all lowest)
    h = lines
    assert "DATASET POLYDATA" in files["hypocentres_vtk"].read_text()
    assert "VERTICES 2 4" in files["hypocentres_vtk"].read_text()
    assert "SCALARS rms float 1" in files["hypocentres_vtk"].read_text()
    assert "POINTS 1 float" in files["stations_vtk"].read_text()
    assert h


def test_pyvista_reads_vtk(tmp_path) -> None:
    pv = pytest.importorskip("pyvista")
    files = export_model(_model(), tmp_path, ("vtk",), masks={"hit": _mask()})
    grid = pv.read(str(files["vtk"]))
    assert grid.dimensions == (len(LON), len(LAT), len(DEPTH))
    assert "vp" in grid.point_data and "hit" in grid.point_data
    legacy = export_model(_model(), tmp_path / "l", ("vtk",), vtk_backend="legacy")["vtk"]
    g2 = pv.read(str(legacy))
    assert g2.n_points == grid.n_points


def test_csv_long_table(tmp_path) -> None:
    files = export_model(_model(), tmp_path, ("csv",), masks={"hit": _mask()}, hypocentres=_hypo())
    df = pd.read_csv(files["csv"])
    assert len(df) == DEPTH.size * LAT.size * LON.size
    for c in (
        "lat",
        "lon",
        "depth_km",
        "x_km",
        "y_km",
        "z_km",
        "vp",
        "vs",
        "vpvs",
        "dvp_pct",
        "hit",
    ):
        assert c in df
    assert (df["z_km"] == -df["depth_km"]).all()
    assert df["hit"].sum() == _mask().sum()
    hy = pd.read_csv(files["hypocentres_csv"])
    assert {"x_km", "y_km", "z_km"} <= set(hy) and (hy["z_km"] < 0).all()


def test_bad_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="format"):
        export_model(_model(), tmp_path, ("shp",))
    with pytest.raises(ValueError, match="shape"):
        export_model(_model(), tmp_path, ("csv",), masks={"m": _mask().isel(depth=slice(0, 2))})
