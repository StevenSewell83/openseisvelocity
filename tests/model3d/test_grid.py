"""Tests for osv.model3d.grid (FMTOMO grid definition)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import GridConfig, Model3DConfig
from osv.coords import spherical_distance_km
from osv.io.fmtomo import grids as fg
from osv.model3d.grid import KM_PER_DEG, GridError, define_grid

REPO = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))
RUN01 = REPO / "projects" / "rk-nm-2017" / "runs" / "01_prep"


def _cfg(**grid: float) -> Model3DConfig:
    kw = {"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 2.0, "zmax_km": 6.0} | grid
    return Model3DConfig(grid=GridConfig(**kw), propgrid_refine=2)


@pytest.fixture
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    lat0, lon0 = -38.5, 176.2
    la, lo = np.meshgrid(np.linspace(0, 0.1, 4), np.linspace(0, 0.12, 4), indexing="ij")
    st = pd.DataFrame(
        {
            "sta": [f"S{i:02d}" for i in range(la.size)],
            "lat": lat0 + la.ravel(),
            "lon": lon0 + lo.ravel(),
            "elevation_m": np.linspace(300.0, 600.0, la.size),
            "used": True,
        }
    )
    ev = pd.DataFrame(
        {
            "lat": [lat0 + 0.03, lat0 + 0.07],
            "lon": [lon0 + 0.05, lon0 + 0.09],
            "depth_km_bsl": [2.0, 4.5],
        }
    )
    return st, ev


def test_node_counts_and_spacing(tables):
    st, ev = tables
    g = define_grid(st, ev, _cfg())
    lat_span = (st.lat.max() - st.lat.min()) * KM_PER_DEG + 4.0
    assert (g.lat[-1] - g.lat[0]) * KM_PER_DEG >= lat_span - 1e-6
    assert (g.lat[-1] - g.lat[0]) * KM_PER_DEG < lat_span + 1.0 + 1e-6
    assert g.dy_km == pytest.approx(1.0)
    # lon spacing scaled by cos(lat)
    assert g.dlon_deg == pytest.approx(g.dlat_deg / np.cos(np.deg2rad(g.centre_lat)))
    assert g.dx_km == pytest.approx(1.0)
    # vertical: top 0.2 km above the highest station (0.6 km), at least zmax deep
    assert g.top_depth_km == pytest.approx(-0.8)
    assert g.bottom_depth_km >= 6.0 and g.bottom_depth_km < 7.0
    assert np.allclose(np.diff(g.depth), 1.0)
    # cushions: one extra node per side, one spacing outside
    assert g.shape_full == tuple(n + 2 for n in g.shape)
    assert g.lat_full[0] == pytest.approx(g.lat[0] - g.dlat_deg)
    assert g.depth_full[-1] == pytest.approx(g.depth[-1] + 1.0)
    m = g.cushion_mask()
    assert int(m.sum()) == int(np.prod(g.shape_full)) - int(np.prod(g.shape))


def test_extent_from_stations_only(tables):
    st, ev = tables
    ev = ev.assign(lat=ev.lat + 0.5)  # far away events
    g_st = define_grid(st, ev, _cfg(extent_from="stations"), strict=False)
    g_both = define_grid(st, ev, _cfg(extent_from="both"))
    assert g_both.lat.size > g_st.lat.size
    assert any("outside the grid laterally" in w for w in g_st.warnings)


def test_containment_and_propgrid_inside_interior(tables):
    st, ev = tables
    g = define_grid(st, ev, _cfg())
    assert g.contains(st.lat, st.lon, -st.elevation_m / 1000).all()
    assert g.contains(ev.lat, ev.lon, ev.depth_km_bsl).all()
    assert not g.contains(st.lat.iloc[0] + 1, st.lon.iloc[0], 1.0)
    assert not g.contains(st.lat.iloc[0], st.lon.iloc[0], 50.0)
    pg = g.propgrid
    # FMTOMO needs 2 <= i <= n-2 of the velocity grid: propgrid strictly inside the interior
    assert pg.lat.min() > g.lat[0] and pg.lat.max() < g.lat[-1]
    assert pg.lon.min() > g.lon[0] and pg.lon.max() < g.lon[-1]
    assert pg.depth_km[0] > g.depth[0] and pg.depth_km[-1] < g.depth[-1]
    assert pg.dr_km <= g.dz_km / 2 and pg.dlat_deg <= g.dlat_deg / 2 + 1e-12
    assert (pg.refinement_factor, pg.n_refine_cells) == (5, 5)
    assert g.contains(g.lat[0], g.lon[0], g.top_depth_km, region="interior")


def test_zero_pad_is_raised_to_half_a_node(tables):
    st, ev = tables
    g = define_grid(st, ev, _cfg(pad_km=0.0))
    assert g.contains(st.lat, st.lon, 0.0).all()
    assert any("below half a node" in w for w in g.warnings)


def test_roundtrip_through_writers_and_validate_model(tables, tmp_path):
    st, ev = tables
    g = define_grid(st, ev, _cfg())
    z = g.depth_full[:, None, None] * np.ones(g.shape_full)
    vg = g.to_vgrids(2.0 + 0.4 * z, vs=(2.0 + 0.4 * z) / 1.75)
    itf = g.flat_interfaces()
    fg.validate_model(vg, itf, g.propgrid)
    fg.write_vgrids(tmp_path / "vgrids.in", vg)
    fg.write_interfaces(tmp_path / "interfaces.in", itf)
    fg.write_propgrid(tmp_path / "propgrid.in", g.propgrid)
    vg2 = fg.read_vgrids(tmp_path / "vgrids.in")
    itf2 = fg.read_interfaces(tmp_path / "interfaces.in")
    pg2 = fg.read_propgrid(tmp_path / "propgrid.in")
    fg.validate_model(vg2, itf2, pg2)
    assert vg2.n_types == 2 and vg2.n_layers == 1
    p = vg2.get(1, 1)
    assert p["velocity"].shape == g.shape_full
    np.testing.assert_allclose(p["depth"].values, g.depth_full, atol=1e-9)
    np.testing.assert_allclose(p["lat"].values, g.lat_full, atol=1e-9)
    np.testing.assert_allclose(p["velocity"].values, 2.0 + 0.4 * z, atol=1e-5)
    inner = fg.interior(p)
    np.testing.assert_allclose(inner["lon"].values, g.lon, atol=1e-9)
    assert pg2.nr == g.propgrid.nr


def test_to_vgrids_interior_shape_pads_cushions_and_rejects_bad(tables):
    st, ev = tables
    g = define_grid(st, ev, _cfg())
    vg = g.to_vgrids(np.full(g.shape, 3.0))
    assert vg.n_types == 1
    assert float(vg.get(1)["velocity"].min()) == 3.0
    with pytest.raises(GridError, match="shape"):
        g.to_vgrids(np.ones((3, 3, 3)))
    with pytest.raises(GridError, match="positive"):
        g.to_vgrids(0.0)


def test_template_dataset(tables):
    st, ev = tables
    g = define_grid(st, ev, _cfg())
    t = g.template()
    assert set(t.coords) == {"depth", "lat", "lon"} and t["velocity"].shape == g.shape_full
    assert g.template(1.0, full=False)["velocity"].shape == g.shape


def test_flat_interfaces_limits(tables):
    st, ev = tables
    g = define_grid(st, ev, _cfg())
    lo, hi = g.surface_limits_km
    assert lo < -0.6  # above the highest station
    with pytest.raises(GridError):
        g.flat_interfaces(top_depth_km=hi + 1)


def test_validation_errors(tables):
    st, ev = tables
    deep = ev.assign(depth_km_bsl=[2.0, 30.0])
    with pytest.raises(GridError, match="deeper than the grid bottom"):
        define_grid(st, deep, _cfg())
    g = define_grid(st, deep, _cfg(), strict=False)
    assert any("deeper than" in w for w in g.warnings)
    with pytest.raises(GridError, match="max_memory_gb"):
        define_grid(st, ev, _cfg(max_memory_gb=1e-6))
    above = ev.assign(depth_km_bsl=[-5.0, 1.0])
    with pytest.raises(GridError, match="above the model top"):
        define_grid(st, above, _cfg())


def test_spacing_warning(tables):
    st, ev = tables
    g = define_grid(st, ev, _cfg(dx_km=5.0, dy_km=5.0, pad_km=5.0))
    assert any("one third" in w for w in g.warnings)
    g = define_grid(st, ev, _cfg(dx_km=0.5, dy_km=0.5, dz_km=0.5, pad_km=1.0))
    assert not any("one third" in w for w in g.warnings)


# ---------------------------------------------------------------------------------------
# integration: fm3d accepts the grids and gives sensible times
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_fm3d_constant_gradient_real_project(tmp_path):
    from osv.io.fmtomo import controls as c
    from osv.io.fmtomo import sources as so
    from osv.io.fmtomo import times
    from osv.tools import run_tool

    if not (BIN / "fm3d").exists():
        pytest.skip(f"{BIN / 'fm3d'} not built")
    if not (RUN01 / "stations.parquet").exists() or not (RUN01 / "events.parquet").exists():
        pytest.skip("project run 01_prep not available")
    st = pd.read_parquet(RUN01 / "stations.parquet")
    ev = pd.read_parquet(RUN01 / "events.parquet")
    # default 0.5 km spacing; coarser propagation grid than the default (refine 5, 85 s) for speed
    g = define_grid(st, ev, Model3DConfig(propgrid_refine=2))
    print(g.summary())

    v0, k = 2.5, 0.4
    z = g.depth_full[:, None, None] * np.ones(g.shape_full)
    vg = g.to_vgrids(v0 + k * z)
    itf = g.flat_interfaces()
    fg.validate_model(vg, itf, g.propgrid)
    fg.write_vgrids(tmp_path / "vgrids.in", vg)
    fg.write_interfaces(tmp_path / "interfaces.in", itf)
    fg.write_propgrid(tmp_path / "propgrid.in", g.propgrid)

    evs = ev.sort_values("depth_km_bsl").iloc[[0, len(ev) // 2, -1]]
    sts = st[st["used"]].iloc[::12].head(5)
    sources = [
        so.Source(float(r.depth_km_bsl), float(r.lat), float(r.lon), [so.PathSpec.direct()])
        for r in evs.itertuples()
    ]
    so.write_sources(tmp_path / "sources.in", sources)
    pos = pd.DataFrame(
        {
            "depth_km": -sts["elevation_m"].to_numpy() / 1000.0,
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
    print(f"fm3d runtime {time.time() - t0:.1f} s")
    arr = times.read_arrivals(tmp_path / "arrivals.dat")
    assert len(arr) == len(rays) and arr["valid"].all()
    for row in arr.itertuples():
        s, r = sources[row.source - 1], pos.loc[row.receiver]
        epi = float(spherical_distance_km(s.lat, s.lon, r.lat, r.lon))
        dist = float(np.hypot(epi, s.depth_km - r.depth_km))
        v1, v2 = v0 + k * s.depth_km, v0 + k * r.depth_km
        analytic = np.arccosh(1 + (k * dist) ** 2 / (2 * v1 * v2)) / k
        straight = dist / (0.5 * (v1 + v2))
        assert row.time == pytest.approx(analytic, rel=0.03)
        assert row.time == pytest.approx(straight, rel=0.20)
