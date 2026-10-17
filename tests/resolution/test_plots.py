"""Tests for osv.resolution.plots (Stage 6 figures)."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from osv.config.schema import Model3DConfig  # noqa: E402
from osv.model3d.grid import define_grid  # noqa: E402
from osv.resolution import plots as rp  # noqa: E402
from osv.resolution.metrics import run_metrics  # noqa: E402
from osv.resolution.models import checkerboard, spikes  # noqa: E402
from osv.resolution.runner import ResolutionRunResult  # noqa: E402

LAT0, LON0 = -38.5, 176.2


@pytest.fixture(scope="module")
def grid():
    la, lo = np.meshgrid(np.linspace(0, 0.09, 3), np.linspace(0, 0.11, 3), indexing="ij")
    st = pd.DataFrame({"lat": LAT0 + la.ravel(), "lon": LON0 + lo.ravel(), "elevation_m": 0.0})
    cfg = Model3DConfig(
        grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 1.0, "zmax_km": 6.0},
        propgrid_refine=2,
    )
    return define_grid(st, None, cfg)


def _recovered(true: xr.Dataset, var: str = "vp") -> xr.Dataset:
    d = 0.6 * true["dvp_pct"]
    return xr.Dataset(
        {f"d{var}_pct": d, var: true[f"{var}_bg"] * (1 + d / 100), f"{var}_bg": true[f"{var}_bg"]}
    )


@pytest.fixture(scope="module")
def suite(grid, tmp_path_factory):
    """True/recovered NetCDF of two checkerboards and a spike test, metrics and mask."""
    wd = tmp_path_factory.mktemp("res")
    rows = []
    specs = [("checker_3km_P", "checkerboard", {"cell_km": 3.0, "offset_half_cell": False})]
    specs.append(("checker_4km_P", "checkerboard", {"cell_km": 4.0, "offset_half_cell": False}))
    sp = [{"x_km": 0.0, "y_km": 0.0, "z_km": 3.0, "radius_km": 1.0, "pct": -8.0}]
    specs.append(("spikes_P", "spikes", {"spikes": sp}))
    import warnings

    for name, kind, params in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "checkerboard":
                tr = checkerboard(grid, 5.0, params["cell_km"], 10.0)
            else:
                tr = spikes(grid, 5.0, sp)
        rec = _recovered(tr)
        d = wd / name
        d.mkdir()
        tr.to_netcdf(d / "true_model.nc")
        rec.to_netcdf(d / "recovered_model.nc")
        rows.append(
            {
                "name": name,
                "kind": kind,
                "phase": "P",
                "params": json.dumps(params),
                "success": True,
                "true_model": str(d / "true_model.nc"),
                "recovered_model": str(d / "recovered_model.nc"),
            }
        )
    table = pd.DataFrame(rows)
    res = ResolutionRunResult(wd, table, 1.0, 1.0, 2, 1, 1)
    from osv.config.schema import ResolutionConfig

    out = run_metrics(res, grid, ResolutionConfig(), wd / "metrics")
    return table, out


def test_pick_depths(grid) -> None:
    z = rp.pick_depths(grid, [0, 1, 2, 3, 4, 5, 20, 100])
    assert 1 <= len(z) <= rp.MAX_DEPTHS and len(set(z)) == len(z)
    assert all(zi in set(grid.depth) for zi in z)
    assert rp.pick_depths(grid, [500.0])  # falls back to evenly spaced depths


def test_true_vs_recovered_symmetric_scale(grid) -> None:
    tr = checkerboard(grid, 5.0, 3.0, 10.0)
    rec = _recovered(tr)
    t, r = tr["dvp_pct"].values, rec["dvp_pct"].values
    fig = rp.plot_true_vs_recovered_slices(t, r, grid, [1.0, 3.0])
    assert isinstance(fig, Figure)
    meshes = [a.collections[0] for a in fig.axes[:4] if a.collections]
    norms = {(m.norm.vmin, m.norm.vmax) for m in meshes}
    assert len(norms) == 1
    vmin, vmax = norms.pop()
    assert vmin == -vmax and vmax == pytest.approx(np.abs(t[1:-1, 1:-1, 1:-1]).max())
    fig2 = rp.plot_true_vs_recovered_sections(t, r, grid, None)
    assert len(fig2.axes) >= 4  # 2 profiles x (true, recovered) + colour bar
    prof = [("A", (LAT0 + 0.01, LON0 + 0.01), (LAT0 + 0.08, LON0 + 0.1))]
    assert isinstance(rp.plot_true_vs_recovered_sections(t, r, grid, prof), Figure)


def test_individual_figures(grid, suite) -> None:
    table, out = suite
    summ, met, res = out["summary"], out["metrics"], out["resolvable"]
    zs = rp.pick_depths(grid, [1, 2, 3, 4])
    tr = checkerboard(grid, 5.0, 3.0, 10.0)
    cases = [("3 km", tr["dvp_pct"].values, 0.5 * tr["dvp_pct"].values)]
    assert isinstance(rp.plot_checkerboard_matrix(cases, grid, zs), Figure)
    rows = [("3 km", met["corr__checker_3km_P"].values)]
    assert isinstance(rp.plot_recovery_matrix(rows, grid, zs, "corr"), Figure)
    rows = [("3 km", met["amp__checker_3km_P"].values)]
    assert isinstance(rp.plot_recovery_matrix(rows, grid, zs, "amp"), Figure)
    assert isinstance(
        rp.plot_resolved_scale(res["resolved_scale_km_vp"].values, grid, zs, [3.0, 4.0]), Figure
    )
    sp = summ[summ["kind"] == "spikes"].iloc[0]["spikes"]
    assert isinstance(rp.plot_smearing({"spikes": sp}), Figure)
    assert isinstance(rp.plot_smearing({"spikes": []}), Figure)  # empty message figure
    assert isinstance(rp.plot_anisotropy(summ), Figure)
    assert isinstance(rp.plot_summary_metrics(summ), Figure)
    rng = np.random.default_rng(0)
    assert isinstance(rp.plot_noise(rng.normal(0, 0.03, 500), np.full(500, 0.03)), Figure)
    curve = [
        {"damping": d, "model_variance": 1e-3 * d, "variance_s2": 1e-2 / d}
        for d in (0.1, 1.0, 10.0)
    ]
    pick = {"curve": curve, "chosen": 1.0, "x_axis": "model_variance"}
    assert isinstance(rp.plot_synthetic_lcurve(pick, {"synthetic": pick}), Figure)


def test_make_all_figures(grid, suite) -> None:
    table, out = suite
    saved: list[tuple[str, str]] = []

    def save(fig, name, caption, section=None):
        assert isinstance(fig, Figure) and caption
        saved.append((name, caption))

    rng = np.random.default_rng(1)
    names = rp.make_all_figures(
        save,
        grid=grid,
        table=table,
        metrics=out["metrics"],
        summary=out["summary"],
        resolvable=out["resolvable"],
        depth_slices_km=[1, 2, 3, 4],
        noise={"P": (rng.normal(0, 0.02, 100), np.full(100, 0.02))},
        real_tradeoff={"damping": {"curve": [], "x_axis": "model_variance"}},
        synthetic_tradeoff={"s": {"damping": {"curve": [], "x_axis": "model_variance"}}},
    )
    assert names == [n for n, _ in saved] and len(set(names)) == len(names)
    for n in (
        "true_vs_recovered_slices_spikes_P",
        "true_vs_recovered_sections_checker_3km_P",
        "checkerboard_matrix_P",
        "recovery_corr_P",
        "recovery_amp_P",
        "resolved_scale_P",
        "smearing_spikes_P",
        "checkerboard_anisotropy_P",
        "noise_realisation_P",
        "summary_metrics",
        "lcurve_synthetic_vs_real_damping",
    ):
        assert n in names, n
    # detail figures: non-checkerboard first, cap respected
    capped: list[str] = []
    rp.make_all_figures(
        lambda f, n, c, s=None: capped.append(n), grid=grid, table=table, max_detail_tests=1
    )
    assert [n for n in capped if n.startswith("true_vs_recovered_slices")] == [
        "true_vs_recovered_slices_spikes_P"
    ]


def test_failed_tests_are_skipped(grid, suite) -> None:
    table, _ = suite
    bad = table.copy()
    bad["success"] = False
    assert rp.make_all_figures(lambda *a, **k: None, grid=grid, table=bad) == []
