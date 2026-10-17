"""Ray coverage measures (``osv.analysis.coverage``)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from osv.analysis.coverage import (
    coverage_mask,
    derivative_weight_sum,
    dws_by_block,
    hit_count_from_frechet,
    node_volume,
    ray_hit_count,
    ray_length_per_cell,
    velocity_layout,
)
from osv.io.fmtomo import read_vgrids
from osv.io.fmtomo.controls import FrechetSpec
from osv.io.fmtomo.frechet import Frechet, parameter_layout, read_frechet
from osv.io.fmtomo.grids import velocity_dataset
from osv.io.fmtomo.rays import RayPaths, read_rays

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "tests" / "data" / "fmtomo" / "example1"
R0 = 6371.0


def _grid(nr=5, nlat=6, nlon=7, dr=1.0, dlat=0.01, dlon=0.01):
    """Grid with 1 cushion node; node 0 at radius R0-(nr-1)*dr (file order, deepest first)."""
    return velocity_dataset(
        np.full((nr, nlat, nlon), 3.0),
        r0_km=R0 - (nr - 1) * dr,
        dr_km=dr,
        lat0_rad=np.deg2rad(-40.0),
        dlat_rad=np.deg2rad(dlat),
        lon0_rad=np.deg2rad(170.0),
        dlon_rad=np.deg2rad(dlon),
        layer=1,
        vtype=1,
    )


def _frechet(rows: list[dict[int, float]], ncol: int) -> Frechet:
    m = sparse.lil_matrix((len(rows), ncol))
    for i, r in enumerate(rows):
        for c, v in r.items():
            m[i, c] = v
    rays = pd.DataFrame(
        {"receiver": 1, "source": 1, "ray": range(1, len(rows) + 1), "subray": 0, "n_deriv": 0}
    )
    return Frechet(rays, m.tocsr())


def test_dws_known_values_and_geometry():
    ds = _grid()
    nr, nlat, nlon = 5, 6, 7
    layout = parameter_layout(velocity=[(1, 1, (nr, nlat, nlon))])

    # file index of node (ir, ilat, ilon): radius fastest
    def col(ir, ilat, ilon):
        return (ilon * nlat + ilat) * nr + ir

    rows = [
        {col(1, 2, 3): -0.5, col(2, 2, 3): 0.25},
        {col(1, 2, 3): 0.1, col(0, 0, 0): 9.0},  # (0,0,0) is a cushion node
    ]
    fr = _frechet(rows, nr * nlat * nlon)
    dws = derivative_weight_sum(fr, layout, ds)
    assert dws.dims == ("depth", "lat", "lon")
    assert dws.shape == (3, 4, 5)
    # ir=1 (radius ascending) -> depth index nr-1-1=3 in full grid, 2 in interior
    assert dws.sel(depth=ds.depth[3], lat=ds.lat[2], lon=ds.lon[3]).item() == pytest.approx(0.6)
    assert dws.sel(depth=ds.depth[2], lat=ds.lat[2], lon=ds.lon[3]).item() == pytest.approx(0.25)
    assert float(dws.sum()) == pytest.approx(0.85)  # cushion node excluded
    full = derivative_weight_sum(fr, layout, ds, interior_only=False)
    assert float(full.sum()) == pytest.approx(9.85)
    # depth of the node: radius R0 - dr at ir = nr-2
    assert float(dws.depth[2]) == pytest.approx(3.0)  # radius R0 - 3 km
    hits = hit_count_from_frechet(fr, layout, ds)
    assert hits.sel(depth=ds.depth[3], lat=ds.lat[2], lon=ds.lon[3]).item() == 2
    assert int(hits.sum()) == 3


def test_dws_per_volume_and_padding():
    ds = _grid()
    layout = parameter_layout(velocity=[(1, 1, (5, 6, 7))])
    fr = _frechet([{16: 1.0}], 20)  # fewer columns than the layout: padded
    d = derivative_weight_sum(fr, layout, ds, interior_only=False)
    dv = derivative_weight_sum(fr, layout, ds, per_volume=True, interior_only=False)
    vol = node_volume(ds)
    np.testing.assert_allclose(dv.to_numpy() * vol.to_numpy(), d.to_numpy(), atol=1e-12)
    assert vol.isel(lat=0).min() < vol.isel(lat=-1).min()  # cos(lat) grows toward the equator


def test_blocks_selection_errors_and_sources():
    ds = _grid()
    layout = parameter_layout(velocity=[(1, 1, (5, 6, 7)), (1, 2, (5, 6, 7))], sources=[1])
    n = 2 * 210 + 4
    fr = _frechet([{0 + 210: 2.0, n - 1: 5.0}], n)
    with pytest.raises(ValueError, match="ambiguous"):
        derivative_weight_sum(fr, layout, ds)
    with pytest.raises(ValueError, match="no velocity block"):
        derivative_weight_sum(fr, layout, ds, layer=2, vtype=1)
    out = dws_by_block(fr, layout, read_vgrids_like(ds), interior_only=False)
    assert float(out[(1, 1)].sum()) == 0.0
    assert float(out[(1, 2)].sum()) == 2.0  # the source parameter is not a velocity node
    bad = parameter_layout(velocity=[(1, 1, (4, 6, 7))])
    with pytest.raises(ValueError, match="shape"):
        derivative_weight_sum(fr, bad, ds)


def read_vgrids_like(ds):
    from osv.io.fmtomo.grids import VelocityGrids

    return VelocityGrids([[ds], [ds]])


def exp_h(ds):
    r = R0 - ds.depth.to_numpy()[3]
    return 6 * np.deg2rad(0.01) * r * np.cos(np.deg2rad(ds.lat.to_numpy()[3]))


def _rays(paths: list[list[tuple[float, float, float]]]) -> RayPaths:
    """Rays from lists of (depth_km, lat_deg, lon_deg) points, one section each."""
    rays = pd.DataFrame(
        {
            "receiver": 1,
            "source": 1,
            "ray": range(1, len(paths) + 1),
            "subray": 0,
            "n_sections": 1,
            "valid": True,
        }
    )
    secs = pd.DataFrame(
        {
            "ray": range(len(paths)),
            "section": 1,
            "npoints": [len(p) for p in paths],
            "region": 1,
            "diffracted": False,
            "headwave": False,
        }
    )
    pts = []
    for i, p in enumerate(paths):
        for d, la, lo in p:
            pts.append((i, 1, R0 - d, np.deg2rad(la), np.deg2rad(lo), d, la, lo))
    points = pd.DataFrame(
        pts,
        columns=["ray", "section", "r_km", "lat_rad", "lon_rad", "depth_km", "lat", "lon"],
    )
    return RayPaths(rays, secs, points, R0)


def test_ray_length_and_hits_known_path():
    ds = _grid(nr=7, nlat=7, nlon=9, dr=1.0, dlat=0.01, dlon=0.01)
    lat_n = ds.lat.to_numpy()
    lon_n = ds.lon.to_numpy()
    d_n = ds.depth.to_numpy()
    # vertical ray down one column of interior node (ilat=3, ilon=4) from depth d[1]-0.5 to d[5]+0.5
    col = [(d, lat_n[3], lon_n[4]) for d in np.linspace(d_n[1] - 0.5, d_n[5] + 0.5, 3)]
    # a second ray crossing the same column horizontally at depth index 3 (E-W)
    row = [(d_n[3], lat_n[3], lon) for lon in np.linspace(lon_n[1], lon_n[7], 4)]
    rp = _rays([col, row])
    length = ray_length_per_cell(rp, ds, oversample=40)
    assert length.dims == ("depth", "lat", "lon")
    assert length.shape == (5, 5, 7)
    # the vertical ray gives 1 km in each of the 5 interior cells; total 5 km
    vert = length.isel(lat=2, lon=3)
    np.testing.assert_allclose(vert.to_numpy()[[0, 1, 3, 4]], 1.0, atol=0.05)
    assert float(length.sum()) == pytest.approx(5.0 + exp_h(ds), rel=0.03)
    # horizontal ray length: 6 * 0.01 deg of longitude at the latitude
    r = R0 - d_n[3]
    exp = 6 * np.deg2rad(0.01) * r * np.cos(np.deg2rad(lat_n[3]))
    horiz = length.isel(depth=2, lat=2).to_numpy()
    assert horiz[1:6].sum() > 0.9 * exp and horiz.sum() == pytest.approx(exp + 1.0, rel=0.02)
    assert horiz[3] == pytest.approx(exp / 6 + 1.0, rel=0.05)
    hits = ray_hit_count(rp, ds)
    assert hits.isel(depth=2, lat=2, lon=3).item() == 2  # both rays
    assert hits.isel(depth=0, lat=2, lon=3).item() == 1
    assert hits.isel(depth=2, lat=2, lon=1).item() == 1
    assert hits.max().item() == 2
    assert int((hits > 0).sum()) == 5 + 7 - 1
    # a ray outside the grid contributes nothing
    far = _rays([[(2.0, -10.0, 100.0), (3.0, -10.0, 100.1)]])
    assert float(ray_length_per_cell(far, ds).sum()) == 0.0


def test_sparse_points_do_not_skip_cells():
    ds = _grid(nr=5, nlat=5, nlon=12, dlon=0.01)
    lat, lon = ds.lat.to_numpy(), ds.lon.to_numpy()
    rp = _rays([[(2.0, lat[2], lon[1]), (2.0, lat[2], lon[10])]])  # 10 cells in one segment
    assert int((ray_hit_count(rp, ds) > 0).sum()) == 10  # cells of nodes 1..10


def test_coverage_mask_modes():
    import xarray as xr

    a = np.zeros((3, 3, 3))
    a[1, 1, 1] = 100.0
    a[0, 0, 0] = 5.0
    a[2, 2, 2] = 20.0
    m = xr.DataArray(a, dims=("depth", "lat", "lon"))
    ab = coverage_mask(m, 10.0)
    assert ab.dtype == bool and int(ab.sum()) == 2
    assert not ab[0, 0, 0]
    assert int(coverage_mask(m, 5.0).sum()) == 3
    assert int(coverage_mask(m, 0.0).sum()) == 3  # zeros are never covered
    pc = coverage_mask(m, 50, mode="percentile")  # median of {5, 20, 100} = 20
    assert int(pc.sum()) == 2
    dil = coverage_mask(m, 100.0, dilate=1)
    assert int(dil.sum()) == 7  # centre cell plus its six face neighbours
    with pytest.raises(ValueError):
        coverage_mask(m, 1, mode="bogus")
    with pytest.raises(ValueError):
        coverage_mask(m, 101, mode="percentile")


def test_example1_fixtures():
    fr = read_frechet(E1 / "frechet.dat")
    spec = FrechetSpec.read(E1 / "frechet.in")
    grids = read_vgrids(E1 / "vgrids.in")
    layout = velocity_layout(spec, grids)
    assert not [b for b in layout if b.kind == "velocity"]  # example 1 inverts sources only
    with pytest.raises(ValueError, match="no velocity block"):
        derivative_weight_sum(fr, layout, grids)
    # velocity blocks over the same matrix: a layout claiming velocity columns still works
    big = FrechetSpec(True, [(1, 1)], [], [])
    lay = velocity_layout(big, grids)
    n = lay[0].size
    dws = derivative_weight_sum(
        Frechet(fr.rays, sparse.csr_matrix((fr.matrix.shape[0], n))), lay, grids, layer=1, vtype=1
    )
    assert float(dws.sum()) == 0.0
    assert dws.sizes["lat"] == grids.get(1, 1).sizes["lat"] - 2


def test_example1_rays_excerpt_on_grid():
    rp = read_rays(E1 / "rays_excerpt.dat")
    grids = read_vgrids(E1 / "vgrids.in")
    length = ray_length_per_cell(rp, grids, layer=1, vtype=1)
    hits = ray_hit_count(rp, grids, layer=1, vtype=1)
    assert float(length.sum()) > 0
    assert hits.max().item() <= len(rp.rays)
    assert float(length.sum()) < 2 * float(
        np.linalg.norm(
            [
                rp.points["depth_km"].max() - rp.points["depth_km"].min(),
                1e4,
            ]
        )
    )
    assert ((hits > 0) == (length > 0)).all()


# ---------------------------------------------------------------------------------------
# integration: fm3d -> frechet + rays, DWS vs ray length
# ---------------------------------------------------------------------------------------

BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))


@pytest.mark.integration
@pytest.mark.skipif(not (BIN / "fm3d").exists(), reason="fm3d binary not built")
def test_fm3d_dws_correlates_with_ray_length(tmp_path: Path):
    import subprocess

    for n in (
        "vgrids.in",
        "interfaces.in",
        "propgrid.in",
        "mode_set.in",
        "sources.in",
        "receivers.in",
    ):
        shutil.copy(E1 / n, tmp_path / n)
    spec = FrechetSpec(True, [(1, 1)], [], [])
    (tmp_path / "frechet.in").write_text(spec.format())
    r = subprocess.run([str(BIN / "fm3d")], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1000:]
    grids = read_vgrids(tmp_path / "vgrids.in")
    layout = velocity_layout(spec, grids)
    fr = read_frechet(tmp_path / "frechet.dat", n_params=layout[0].size)
    rp = read_rays(tmp_path / "rays.dat")
    dws = derivative_weight_sum(fr, layout, grids)
    length = ray_length_per_cell(rp, grids, layer=1, vtype=1, region=1)
    hits_f = hit_count_from_frechet(fr, layout, grids)
    assert float(dws.sum()) > 0 and float(length.sum()) > 0
    x, y = dws.to_numpy().ravel(), length.to_numpy().ravel()
    sel = (x > 0) | (y > 0)
    assert np.corrcoef(x[sel], y[sel])[0, 1] > 0.8
    assert int((hits_f > 0).sum()) > 0
