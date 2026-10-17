"""Comparison of the final model with the user geology / well model (osv.analysis.compare_user)."""

from __future__ import annotations

import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.analysis.compare_user import (
    compare_with_user_model,
    make_comparison_figures,
)
from osv.config.schema import Model3DConfig, WellsConfig
from osv.model3d.geology import build_geology_model
from osv.model3d.grid import define_grid
from osv.model3d.wells import build_wells_model

LAT0, LON0 = -38.55, 176.19
HORIZONS = [
    {"name": "shallow", "surface": "x", "vp": {"law": "constant", "v0": 2.5}},
    {"name": "mid", "surface": "x", "vp": {"law": "gradient", "v0": 3.0, "k": 0.3}},
]


def vp_true(z):
    return 2.0 + 0.5 * np.asarray(z)


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


@pytest.fixture(scope="module")
def setup(grid, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wells")
    shape = grid.shape_full[1:]
    geo, _ = build_geology_model(
        HORIZONS,
        grid,
        background=4.5,
        surfaces={"shallow": np.full(shape, 1.0), "mid": np.full(shape, 3.0)},
    )
    rows = []
    for i, (dlat, dlon) in enumerate([(0.03, 0.04), (0.07, 0.08)]):
        md = np.arange(0.0, 4500.0, 2.0)
        las = lasio.LASFile()
        las.well["WELL"].value = f"W{i}"
        las.append_curve("DEPT", md, unit="m")
        las.append_curve("DT", 304.8 / vp_true(md / 1000.0), unit="US/F")
        with open(tmp / f"w{i}.las", "w") as f:
            las.write(f, version=2.0)
        rows.append({"name": f"W{i}", "las_file": f"w{i}.las", "lon": LON0 + dlon,
                     "lat": LAT0 + dlat})  # fmt: skip
    pd.DataFrame(rows).to_csv(tmp / "wells.csv", index=False)
    cfg = WellsConfig(las_dir=tmp, table=tmp / "wells.csv", smoothing_m=0.0, max_distance_km=500)
    wells = build_wells_model(cfg, grid, background=4.0)

    def with_cushion(ds, **kw):
        ds = ds.copy()
        ds.attrs["n_cushion"] = grid.n_cushion
        return ds

    final = with_cushion(geo)
    final["vp"] = geo["vp"] * 1.05 + 0.1 * np.sin(geo["depth"])
    final["vs"] = final["vp"] / 1.75
    start = with_cushion(geo[["vp", "vs"]])
    return grid, geo, wells, final, start


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_difference_and_unit_stats(setup):
    _, geo, wells, final, start = setup
    r = compare_with_user_model(final, geo, wells, starting=start)
    d = r.difference
    assert {"dvp", "dvp_pct", "dvs_pct", "unit_id"} <= set(d.data_vars)
    pct = d["dvp_pct"].values
    assert np.nanmean(pct) > 4.0 and np.nanmean(pct) < 8.0
    st = r.unit_stats.set_index("name")
    assert {"shallow", "mid", "background"} <= set(st.index)
    # shallow unit has constant user vp 2.5, final = 1.05 * 2.5 + small sine term
    assert st.loc["shallow", "user_mean"] == pytest.approx(2.5)
    assert st.loc["shallow", "user_std"] == pytest.approx(0.0, abs=1e-9)
    assert st.loc["shallow", "bias"] > 0.1
    assert st.loc["mid", "final_std"] > 0
    assert r.summary["n_wells"] == 2


def test_well_misfit_final_vs_starting(setup):
    _, geo, wells, final, start = setup
    # final model close to the logs, starting model (geology) not
    close = final.copy()
    depth = close["depth"].broadcast_like(close["vp"])
    close["vp"] = vp_true(depth) + 0.0 * close["vp"]
    r = compare_with_user_model(close, geo, wells, starting=start)
    wp = r.well_profiles
    assert set(wp.well) == {"W0", "W1"}
    assert np.isfinite(wp.final_vp).any()
    m = r.well_misfit.set_index(["well", "model", "subset"])
    assert m.loc[("W0", "final", "all"), "rmse"] < 0.05
    assert m.loc[("W0", "starting", "all"), "rmse"] > m.loc[("W0", "final", "all"), "rmse"]
    assert m.loc[("W0", "final", "all"), "corr"] > 0.99
    assert len(r.well_curves) > len(wp)


def test_user_model_on_other_grid_and_unit_array(setup):
    _, geo, _, final, _ = setup
    coarse = geo.isel(depth=slice(None, None, 2))
    r = compare_with_user_model(final, coarse, unit_id=geo["unit_id"].values)
    assert r.difference["dvp_pct"].shape == tuple(
        final.sizes[d] - 2 * final.attrs["n_cushion"] for d in ("depth", "lat", "lon")
    )
    assert len(r.unit_stats) >= 2


def test_figures_synthetic_geology(setup):
    _, geo, wells, final, start = setup
    nc = final.attrs["n_cushion"]
    dws = xr.DataArray(
        np.where(final["depth"].values[nc:-nc, None, None] < 3.0, 100.0, 0.0)
        * np.ones(tuple(final.sizes[d] - 2 * nc for d in ("depth", "lat", "lon"))),
        dims=("depth", "lat", "lon"),
        coords={d: final[d].values[nc:-nc] for d in ("depth", "lat", "lon")},
    )
    r = compare_with_user_model(final, geo, wells, starting=start, coverage=dws)
    saved = {}

    def save(fig, name, caption, section):
        saved[name] = (fig, caption, section)

    names = make_comparison_figures(save, r, depths=[1.0, 2.5])
    assert {"user_well_profiles", "user_unit_boxplots", "user_diff_slices_vp"} <= set(names)
    assert any(n.startswith("user_diff_section_") for n in names)
    assert all(v[2] == "compare_user" for v in saved.values())
    fig = saved["user_diff_slices_vp"][0]
    text = " ".join(a.get_title() for a in fig.axes)
    assert "User-model" in text and "Starting" not in text
    # depth range below 3 km is uncovered: a shaded span exists in the well panel
    wfig = saved["user_well_profiles"][0]
    assert any(len(ax.patches) > 0 for ax in wfig.axes if ax.get_visible())
