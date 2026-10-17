"""Tests for osv.model3d.plots (figure functions on a synthetic model)."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from osv.config.schema import Model3DConfig
from osv.model3d.builder import StartingModel
from osv.model3d.grid import define_grid
from osv.model3d.plots import make_all_figures


@pytest.fixture(scope="module")
def world():
    st = pd.DataFrame(
        {
            "lat": [-38.55, -38.45, -38.50, -38.53],
            "lon": [176.15, 176.25, 176.30, 176.18],
            "elevation_m": [400.0, 300.0, 350.0, 380.0],
            "sensor_elevation_m": [400.0, 300.0, 350.0, 380.0],
        }
    )
    ev = pd.DataFrame({"lat": [-38.5, -38.51], "lon": [176.2, 176.21], "depth_km_bsl": [2.0, 3.0]})
    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 0.5, "pad_km": 2.0, "zmax_km": 6.0},
        propgrid_refine=2,
    )
    grid = define_grid(st, ev, cfg)
    shape = grid.shape_full
    z = grid.depth_full[:, None, None] * np.ones(shape)
    bg = 2.0 + 0.6 * np.maximum(z, 0)
    w = np.zeros(shape)
    w[:, : shape[1] // 2, :] = 1.0
    vp = bg * (1 + 0.1 * w)
    ds = grid.template().drop_vars("velocity")
    dims = ("depth", "lat", "lon")
    for k, v in (("vp", vp), ("vs", vp / 1.75), ("vp_bg", bg), ("vs_bg", bg / 1.75), ("weight", w)):
        ds[k] = (dims, v)
    unit = np.where(z < 2, 0, np.where(z < 4, 1, -1)).astype(np.int32)
    ds["unit_id"] = (dims, unit)
    ds.attrs["unit_names"] = ["A", "B"]
    res = pd.DataFrame(
        {
            "phase": ["P"] * 30 + ["S"] * 30,
            "station": list("ABCD") * 15,
            "res_start": np.random.default_rng(0).normal(0, 0.05, 60),
            "res_bg": np.random.default_rng(1).normal(0, 0.08, 60),
            "dist_km": np.linspace(1, 20, 60),
            "sta_lat": -38.5,
            "sta_lon": 176.2,
        }
    )
    return ds, grid, st, ev, res, StartingModel(grid, ds)


def test_make_all_figures(world):
    ds, grid, st, ev, res, _ = world
    w = SimpleNamespace(name="W1", lat=-38.52, lon=176.2, vp=np.array([2.5, 3.0]),
                        depth_km=np.array([1.0, 2.0]))  # fmt: skip
    prof = SimpleNamespace(name="A-A'", start=(-38.55, 176.15), end=(-38.45, 176.25))
    figs = make_all_figures(
        None,
        ds=ds,
        grid=grid,
        interfaces=grid.flat_interfaces(),
        stations=st,
        events=ev,
        residuals=res,
        wells=[w],
        profiles=[prof],
        depth_slices_km=[0, 1, 2, 50],
        checks=Model3DConfig().checks,
    )
    expected = {
        "vp_slices", "dvp_slices", "section_ew", "section_ns", "section_a_a", "well_profiles",
        "horizons", "weight_map", "vp_histogram", "gradient_map", "grid_geometry",
        "residual_histograms", "residual_distance", "station_residuals",
    }  # fmt: skip
    assert expected <= set(figs)
    plt.close("all")


def test_no_user_model_and_save(world, tmp_path):
    ds, grid, st, ev, res, _ = world
    plain = ds.drop_vars("unit_id").copy()
    plain["vp"], plain["vs"] = plain["vp_bg"], plain["vs_bg"]
    plain["weight"] = plain["weight"] * 0
    saved = []
    make_all_figures(
        lambda fig, name, caption, section=None: (saved.append(name), plt.close(fig)),
        ds=plain,
        grid=grid,
        interfaces=grid.flat_interfaces(),
        stations=st,
        events=ev,
        residuals=res,
    )
    assert "horizons" not in saved and "well_profiles" not in saved
    assert {"dvp_slices", "weight_map", "section_ew", "vp_histogram"} <= set(saved)
