"""Unit tests of the validation statistics, QC findings and figures (no binaries needed)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from osv.plotting.registry import FigureRegistry
from osv.validation.plots import (
    make_figures,
    plot_depth_change_hist,
    plot_map_sections,
    plot_rms_change_hist,
    plot_rms_scatter,
)
from osv.validation.relocate import (
    KM_PER_DEG,
    RelocateOptions,
    ValidationResult,
    compare_relocations,
    event_rms,
    hypocentre_offsets_km,
    nearest_neighbour_km,
    paired_rms_stats,
    pick_weights,
)


def test_event_rms_matches_velest_formula() -> None:
    res = np.array([0.1, -0.05, 0.02, 0.07])
    w = np.array([1.0, 1.0, 0.25, 0.0])  # last pick has zero weight: ignored
    wr = res[:3] * w[:3]
    expect = math.sqrt((np.sum(wr**2) - np.sum(wr) ** 2 / 3) / 2)
    assert event_rms(res, w) == pytest.approx(expect)
    assert math.isnan(event_rms(res[:1], w[:1]))
    # a constant residual is an origin-time error only: zero rms for equal weights
    assert event_rms(np.full(5, 0.3), np.ones(5)) == pytest.approx(0.0, abs=1e-12)


def test_pick_weights_classes_and_s_factor() -> None:
    pk = pd.DataFrame({"phase": ["P", "P", "S", "S"], "velest_weight": [0, 1, 0, 2]})
    np.testing.assert_allclose(pick_weights(pk, 0.5), [1.0, 0.25, 0.5, 0.5 / 16])


def test_hypocentre_offsets_and_nn() -> None:
    off = hypocentre_offsets_km([-38.5], [176.2], [3.0], [-38.5 + 1 / KM_PER_DEG], [176.2], [3.5])
    assert off["dy_km"][0] == pytest.approx(1.0)
    assert off["dx_km"][0] == pytest.approx(0.0, abs=1e-9)
    assert off["dz_km"][0] == pytest.approx(0.5)
    assert off["dh_km"][0] == pytest.approx(1.0)
    e = hypocentre_offsets_km([-38.5], [176.2], [0], [-38.5], [176.2 + 1 / KM_PER_DEG], [0])
    assert e["dx_km"][0] == pytest.approx(math.cos(math.radians(38.5)), rel=1e-4)
    lat = np.array([-38.5, -38.5, -38.5 + 5 / KM_PER_DEG])
    nn = nearest_neighbour_km(lat, [176.2, 176.2, 176.2], [1.0, 3.0, 1.0])
    np.testing.assert_allclose(nn, [2.0, 2.0, 5.0], rtol=1e-6)
    assert math.isnan(nearest_neighbour_km([1.0], [1.0], [1.0])[0])


def test_paired_stats_and_wilcoxon() -> None:
    rng = np.random.default_rng(1)
    a = rng.uniform(0.05, 0.15, 30)
    b = a - rng.uniform(0.005, 0.03, 30)  # 3-D always better
    s = paired_rms_stats(a, b)
    assert s["n"] == 30
    assert s["frac_3d_better"] == 1.0 and s["frac_3d_worse"] == 0.0
    assert s["median_improvement_s"] == pytest.approx(np.median(a - b))
    assert s["mean_improvement_pct"] > 0
    assert s["p_two_sided"] == pytest.approx(sps.wilcoxon(a - b).pvalue)
    assert s["p_3d_better"] < s["p_two_sided"] < 1e-4
    w = paired_rms_stats(b, a)  # 3-D always worse
    assert w["p_3d_better"] > 0.99 and w["frac_3d_worse"] == 1.0
    # nan pairs are skipped
    a2, b2 = a.copy(), b.copy()
    a2[0] = np.nan
    assert paired_rms_stats(a2, b2)["n"] == 29
    # identical values: no information, p is nan not an error
    z = paired_rms_stats([0.1, 0.1], [0.1, 0.1])
    assert math.isnan(z["p_two_sided"]) and z["frac_tied"] == 1.0
    assert paired_rms_stats([], [])["n"] == 0


def _events(n: int = 12, better: bool = True, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lat0 = -38.55 + rng.normal(0, 0.01, n)
    lon0 = 176.2 + rng.normal(0, 0.01, n)
    z0 = rng.uniform(2, 4, n)
    r1 = rng.uniform(0.08, 0.15, n)
    r3 = r1 - 0.02 if better else r1 + 0.05
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "lat0": lat0,
            "lon0": lon0,
            "depth0": z0,
            "lat_1d": lat0 + 0.002,
            "lon_1d": lon0,
            "depth_1d": z0 + 0.3,
            "rms_1d": r1,
            "lat_3d": lat0,
            "lon_3d": lon0 + 0.002,
            "depth_3d": z0 - 0.2,
            "rms_3d": r3,
            "n_picks": 20,
        }
    )


def test_compare_relocations_columns_and_findings() -> None:
    ev = _events()
    st, fnd = compare_relocations(ev)
    for c in ("rms_change", "dx_km", "dy_km", "dz_km", "dh_km"):
        assert c in ev
    np.testing.assert_allclose(ev["dz_km"], -0.5)
    assert st["hypocentre_difference"]["dz_median_km"] == pytest.approx(-0.5)
    assert {"nn_median_1d_km", "nn_median_3d_km", "nn_median_catalogue_km"} <= set(st["clustering"])
    codes = {f.code for f in fnd}
    assert "validation_3d_worse" not in codes and "validation_clustering" in codes
    assert st["rms"]["median_improvement_s"] == pytest.approx(0.02)


def test_qc_warns_when_3d_worse_for_most_events() -> None:
    ev = _events(better=False)
    _, fnd = compare_relocations(ev)
    f = next(f for f in fnd if f.code == "validation_3d_worse")
    assert f.severity == "warn" and f.count == len(ev)


def test_qc_few_events_and_large_shift_and_no_pairs() -> None:
    ev = _events(4)
    ev.loc[0, "lat_3d"] += 10 / KM_PER_DEG  # 10 km
    _, fnd = compare_relocations(ev, RelocateOptions(min_events_stat=6))
    codes = {f.code for f in fnd}
    assert {"validation_few_events", "validation_large_hypocentre_shift"} <= codes
    ev2 = _events(3)
    ev2["rms_3d"] = np.nan
    st, fnd2 = compare_relocations(ev2)
    assert st["rms"]["n"] == 0 and fnd2[0].code == "validation_no_pairs"


def test_result_summary_is_json_and_writes(tmp_path) -> None:
    ev = _events()
    st, fnd = compare_relocations(ev)
    res = ValidationResult(ev, st, fnd, workdir=tmp_path)
    json.dumps(res.summary())
    res.write()
    assert (tmp_path / "validation_events.csv").is_file()
    assert json.loads((tmp_path / "validation_summary.json").read_text())["n_events"] == 12


def _result(n: int = 12, better: bool = True) -> ValidationResult:
    ev = _events(n, better)
    st, fnd = compare_relocations(ev)
    return ValidationResult(ev, st, fnd)


def test_figures_are_produced(tmp_path) -> None:
    res = _result()
    for fn in (plot_rms_scatter, plot_rms_change_hist, plot_depth_change_hist, plot_map_sections):
        fig = fn(res)
        assert fig is not None
        fig.canvas.draw()
        import matplotlib.pyplot as plt

        plt.close(fig)
    reg = FigureRegistry(tmp_path / "figures", formats=("png",), dpi=60)
    names = make_figures(res, reg)
    assert names == [
        "validation_rms_scatter",
        "validation_rms_change",
        "validation_locations",
        "validation_depth_change",
    ]
    for n in names:
        assert (tmp_path / "figures" / f"{n}.png").stat().st_size > 1000
    assert len(reg.entries()) == 4


def test_figures_with_single_event_and_nan(tmp_path) -> None:
    res = _result(1)
    names = make_figures(res, tmp_path)
    assert "validation_rms_scatter" in names
    res2 = _result(5)
    res2.events["rms_3d"] = np.nan
    res2.events["rms_change"] = np.nan
    res2.stats["rms"] = paired_rms_stats(res2.events["rms_1d"], res2.events["rms_3d"])
    assert plot_rms_scatter(res2) is None and plot_rms_change_hist(res2) is None


def test_clamp_sources_keeps_events_below_surface_and_inside_grid() -> None:
    from osv.io.fmtomo import InterfaceGrids, PathSpec, PropGrid, Source
    from osv.io.fmtomo.grids import interface_dataset
    from osv.validation.relocate import _clamp_sources

    nlat, nlon = 5, 6
    lat0, lon0 = np.deg2rad(-38.6), np.deg2rad(176.1)
    d = np.deg2rad(0.05)
    radius = np.stack(
        [np.full((nlat, nlon), 6371.0 + 0.5), np.full((nlat, nlon), 6371.0 - 10.0)]
    )  # surface 0.5 km above sea level (depth -0.5), base at 10 km
    itf = InterfaceGrids(
        interface_dataset(radius, lat0_rad=lat0, dlat_rad=d, lon0_rad=lon0, dlon_rad=d)
    )
    pg = PropGrid(
        nr=12, nlat=nlat, nlon=nlon, dr_km=1.0, dlat_deg=0.05, dlon_deg=0.05,
        top_height_km=1.0, lat0_deg=-38.6, lon0_deg=176.1, refinement_factor=1, n_refine_cells=2,
    )  # fmt: skip
    path = [PathSpec.direct()]
    srcs = [
        Source(1.0, -38.5, 176.2, path),  # fine
        Source(-0.9, -38.5, 176.2, path),  # above the surface (-0.5)
        Source(30.0, -38.5, 176.2, path),  # below the grid
        Source(1.0, -39.5, 176.2, path),  # south of the grid
    ]
    out, n = _clamp_sources(srcs, itf, pg, 0.05)
    assert n == 3 and out[0] == srcs[0]
    assert out[1].depth_km == pytest.approx(-0.45)
    assert out[2].depth_km < float(pg.depth_km[-1])
    assert out[3].lat >= -38.6


def test_model_for_velest_drops_s_block_and_gets_vpvs() -> None:
    from osv.io.velest import ModelBlock, VelestModel
    from osv.validation.relocate import _model_for_velest

    lay = pd.DataFrame(
        {"vel": [2.0, 3.0, 5.0], "depth": [-0.5, 1.0, 3.0], "vdamp": 1.0, "reflector": " "}
    )
    slay = lay.assign(vel=lay["vel"] / 1.7)
    vm = VelestModel("t", ModelBlock(lay, label="P"), ModelBlock(slay, label="S"))
    m3, vpvs = _model_for_velest(vm, nsp=3)
    assert m3.s is None and vpvs == pytest.approx(1.7)
    m2, _ = _model_for_velest(vm, nsp=2)
    assert m2.s is not None
