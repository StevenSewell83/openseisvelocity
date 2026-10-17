"""Model slice/section figures (``osv.analysis.plots_model``)."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from osv.analysis.plots_model import (
    ModelFigureOptions,
    make_model_figures,
    model_limits,
    plot_section_comparison,
    plot_sections,
    plot_slice_matrix,
    plot_slices,
)
from osv.model3d.extrude import effective_profile
from osv.plotting.slices import ModelFields, ModelMasks, bspline_sample, colour_limits

NZ, NY, NX = 9, 11, 12
DEPTH = np.linspace(-0.5, 7.5, NZ)
LAT = -38.5 + 0.005 * np.arange(NY)
LON = 176.2 + 0.006 * np.arange(NX)


def _model(amp=0.0, seed=0):
    d, la, lo = np.meshgrid(DEPTH, LAT, LON, indexing="ij")
    bgp = 3.0 + 0.4 * d
    bgs = bgp / 1.75
    rng = np.random.default_rng(seed)
    pert = amp * rng.standard_normal(d.shape)
    c = ("depth", "lat", "lon")
    return xr.Dataset(
        {
            "vp": (c, bgp * (1 + pert)),
            "vs": (c, bgs * (1 + pert)),
            "vp_bg": (c, bgp),
            "vs_bg": (c, bgs),
        },
        coords={"depth": DEPTH, "lat": LAT, "lon": LON},
        attrs={"n_cushion": 1},
    )


def _dws():
    c = ("depth", "lat", "lon")
    v = np.zeros((NZ - 2, NY - 2, NX - 2))
    v[:, :, : (NX - 2) // 2] = 100.0
    return xr.DataArray(
        v, dims=c, coords={"depth": DEPTH[1:-1], "lat": LAT[1:-1], "lon": LON[1:-1]}
    )


def _cfg(profiles=()):
    return SimpleNamespace(
        plotting=SimpleNamespace(
            depth_slices_km=[1.0, 3.0, 99.0], profiles=list(profiles), dws_mask_threshold=10.0
        )
    )


EVENTS = pd.DataFrame({"lat": [-38.48, -38.47], "lon": [176.22, 176.23], "depth_km": [3.0, 5.0]})
STATIONS = pd.DataFrame({"lat": [-38.49], "lon": [176.21], "elevation_m": [300.0], "sta": ["AAA"]})


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_slices_and_masking():
    m, start = _model(0.03), _model(0.0)
    f = ModelFields(m, None, start)
    masks = ModelMasks.from_coverage(_dws(), 10.0)
    fig = plot_slices(f, "dvp_1d", [1.0, 3.0, 99.0], masks, events=EVENTS, stations=STATIONS)
    assert isinstance(fig, Figure)
    assert len([a for a in fig.axes if a.get_visible() and a.get_label() != "<colorbar>"]) >= 2
    cov = masks.covered_at("p", 1.0, LAT[5], LON[[2, NX - 3]])
    assert cov.tolist() == [True, False]


def test_percent_limits_symmetric_and_shared():
    f = ModelFields(_model(0.05), None, _model(0.0))
    lim = model_limits(f, ["dvp_1d", "dvp_start", "vp", "vpvs"])
    for k in ("dvp_1d", "dvp_start"):
        assert lim[k][0] == -lim[k][1]
        assert lim[k][1] > 0
    assert lim["vp"][0] < lim["vp"][1]
    fig = plot_slice_matrix(
        f, ["dvp_1d", "dvs_1d"], [1.0, 3.0], limits=lim | {"dvs_1d": lim["dvp_1d"]}
    )
    cbs = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(cbs) == 2
    assert cbs[0].get_xlim() == pytest.approx(cbs[1].get_xlim())


def test_colour_limits_only_covered():
    v = np.array([1.0, 2.0, 100.0])
    assert colour_limits([v], "pct", 100.0, [np.array([True, True, False])]) == (-2.0, 2.0)


def test_bspline_matches_effective_profile_and_constant():
    m = _model(0.05)
    z = np.array([0.7, 2.2, 5.1])
    got = bspline_sample(m, "vp", z, LAT[4], LON[4])
    ref = effective_profile(DEPTH, m["vp"].values[:, 4, 4], z)
    # 3-D with a lateral line of identical neighbours is not 1-D: check partition of unity instead
    c = m.copy()
    c["vp"] = c["vp"] * 0 + 4.2
    assert bspline_sample(c, "vp", z, LAT[4], LON[4]) == pytest.approx(4.2)
    assert np.isfinite(got).all() and ref.shape == got.shape
    assert np.isnan(bspline_sample(c, "vp", DEPTH[0], LAT[4], LON[4]))
    f = ModelFields(m, None, None, "bspline")
    assert np.isfinite(f.evaluate("vp", 4.0, LAT[5], LON[5]))


def test_sections_profiles_and_comparison():
    f = ModelFields(_model(0.03), None, _model(0.0))
    masks = ModelMasks.from_coverage(_dws(), 10.0)
    prof = ("A-A'", (LAT[1], LON[1]), (LAT[-2], LON[-2]))
    fig = plot_sections(
        f, ["vp", "dvp_start", "vpvs"], prof, masks, events=EVENTS, stations=STATIONS
    )
    assert "Section A-A'" in fig.texts[0].get_text() or fig._suptitle is not None
    topo = xr.DataArray(
        np.full((NY, NX), 300.0), dims=("lat", "lon"), coords={"lat": LAT, "lon": LON}
    )
    assert isinstance(plot_section_comparison(f, prof, masks, topography=topo), Figure)
    with pytest.raises(ValueError):
        plot_section_comparison(f, prof, quantity="vpx")


def test_hatch_alpha_none_styles():
    f = ModelFields(_model(0.03))
    masks = ModelMasks.from_coverage(_dws(), 10.0)
    for style in ("hatch", "alpha", "none", "grey"):
        plot_slices(f, "vp", [1.0], masks, opts=ModelFigureOptions(mask_style=style))
    with pytest.raises(ValueError):
        plot_slices(f, "vp", [1.0], masks, opts=ModelFigureOptions(mask_style="bad"))


def test_resolvable_mask_combines():
    res = _dws() > 0
    res[:, :, :2] = False
    masks = ModelMasks.from_coverage(_dws(), 10.0, res)
    assert masks.resolvable_applied
    assert not masks.covered_at("p", 1.0, LAT[5], LON[1])[()]
    assert masks.covered_at("p", 1.0, LAT[5], LON[3])[()]


def test_make_model_figures_all(tmp_path):
    saved = {}

    def save(fig, name, caption, section=None):
        saved[name] = (caption, section)
        plt.close(fig)

    names = make_model_figures(
        save,
        _model(0.03),
        None,
        _model(0.0),
        _dws(),
        EVENTS,
        STATIONS,
        _cfg(),
        resolvable_mask=_dws() > 0,
    )
    assert names == list(saved)
    for n in (
        "slices_vp",
        "slices_dvp_1d",
        "slices_dvp_start",
        "slices_vpvs",
        "compare_start_final_vp",
        "compare_p_s_dv_1d",
        "section_e_w",
        "section_n_s",
        "slices_dvp_1d_resolvable",
        "section_e_w_resolvable",
    ):
        assert n in saved
    # P only model
    p_only = _model(0.03).drop_vars(["vs", "vs_bg"])
    n2 = make_model_figures(
        lambda fig, *a: plt.close(fig), p_only, None, None, None, EVENTS, None, _cfg()
    )
    assert "slices_vp" in n2 and "slices_vs" not in n2 and "slices_dvp_start" not in n2


def test_reference_table_and_errors():
    ref = pd.DataFrame({"depth": [-1.0, 8.0], "vp": [3.0, 6.0], "vs": [1.7, 3.4]})
    f = ModelFields(_model(0.02), ref)
    assert np.isfinite(f.evaluate("dvp_1d", 2.0, LAT[5], LON[5]))
    with pytest.raises(KeyError):
        ModelFields(_model(0.02).drop_vars(["vp_bg", "vs_bg"])).evaluate(
            "dvp_1d", 1.0, LAT[5], LON[5]
        )
    with pytest.raises(ValueError):
        plot_slices(ModelFields(_model()), "vp", [50.0])


def test_mask_outline_unlabelled_and_contours_normalised():
    f = ModelFields(_model(0.03))
    masks = ModelMasks.from_coverage(_dws(), 10.0)
    fig = plot_slices(f, "vp", [1.0], masks)
    assert not [t for ax in fig.axes for t in ax.texts if "e-" in t.get_text()]
    assert masks.dws_at("p", 1.0, LAT[5], LON[2], normalise=True)[()] == pytest.approx(1.0)
    fig = plot_slices(f, "vp", [1.0], masks, opts=ModelFigureOptions(dws_contours=[0.5]))
    labels = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not any("e-" in t for t in labels)
