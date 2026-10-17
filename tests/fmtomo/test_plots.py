"""Stage 4 figures and QC checks (``osv.fmtomo.plots``), from synthetic tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from osv.fmtomo import plots as pl  # noqa: E402
from osv.fmtomo import tradeoff as to  # noqa: E402
from osv.io.fmtomo import write_vgrids  # noqa: E402
from osv.plotting.registry import FigureRegistry  # noqa: E402
from osv.resolution.models import checkerboard  # noqa: E402


def _L(p, corner):
    return 1.0 + (corner / p) ** 2, 1.0 + (p / corner) ** 2


def _pick(name: str, corner: float, manual: float, values) -> dict:
    p = np.array(values, float)
    model, data = _L(p, corner)
    cp = to.lcurve_corner(p, model, data)
    xcol = "model_variance" if name == "damping" else "pert_roughness_phys"
    return {
        "x_axis": xcol,
        "y_axis": "variance_s2",
        "curve": [
            {name: float(a), xcol: float(m), "variance_s2": float(d), "rms_s": 0.1}
            for a, m, d in zip(p, model, data, strict=True)
        ],
        "manual": manual,
        "max_curvature": cp.value,
        "curvature": [None if not math.isfinite(c) else float(c) for c in cp.curvature],
        "reason": cp.reason,
        "chosen": cp.value if cp.value is not None else manual,
        "source": "max_curvature" if cp.value is not None else "manual (fallback)",
    }


def _iter_table(rms, **extra) -> pd.DataFrame:
    n = len(rms)
    d = {
        "iter": np.arange(n),
        "rms_ms": rms,
        "variance_s2": (np.array(rms) / 1000) ** 2,
        "chi2": np.array(rms) / rms[-1],
        "n_valid": [1000.0] * n,
        "n_lost": [0.0] * n,
        "n_obs_invalid": [5.0] * n,
        "vel_min": [2.0] * n,
        "vel_max": [5.0] * n,
        "frac_at_min_velocity": [0.0] * n,
    }
    d.update(extra)
    return pd.DataFrame(d)


def _make_run(root: Path, d: float, s: float, rms: list[float]) -> Path:
    """A minimal run directory: summaries and vgrids of the start and the last iteration."""
    from osv.config.schema import Model3DConfig
    from osv.model3d.grid import define_grid

    run = root / f"d{d:g}_s{s:g}"
    cfg = Model3DConfig(grid={"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 1.0})
    st = pd.DataFrame({"lat": [-38.5, -38.45], "lon": [176.2, 176.25], "elevation_m": 0.0})
    grid = define_grid(st, None, cfg)
    ref = grid.to_vgrids(checkerboard(grid, 5.0, 2.0, 0.0)["vp"], uncertainty=0.5)
    pert = grid.to_vgrids(checkerboard(grid, 5.0, 2.0, 100.0 / (s + d))["vp"], uncertainty=0.5)
    write_vgrids(run / "vgridsref.in", ref)
    for i, r in enumerate(rms):
        di = run / "iterations" / f"{i:02d}"
        di.mkdir(parents=True)
        (di / "summary.json").write_text(json.dumps({"iter": i, "rms_ms": r, "n_valid": 10}))
        write_vgrids(di / "vgrids.in", ref if i == 0 else pert)
    return run


# ---------------------------------------------------------------------------------------
def test_lcurve_figure_with_manual_point():
    pk = _pick("damping", 20.0, 100.0, [1, 5, 10, 20, 50, 100, 200, 500, 1000])
    fig = pl.plot_lcurve(pk, "damping")
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["1", "5", "10", "20", "50", "100", "200", "500", "1000"]  # all labelled
    leg = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("chosen damping" in t for t in leg)
    assert any("manual damping = 100" in t for t in leg)  # differs from the chosen one
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert len(fig.axes) == 2 and fig.axes[1].lines  # curvature panel


def test_lcurve_figure_without_corner_and_empty():
    pk = _pick("smoothing", 5.0, 50.0, [1, 10])  # too few points
    assert pk["source"] == "manual (fallback)"
    fig = pl.plot_lcurve(pk, "smoothing")
    assert any("curvature undefined" in t.get_text() for t in fig.axes[1].texts)
    empty = pl.plot_lcurve({"curve": [], "chosen": 1.0}, "damping")
    assert any("no successful" in t.get_text() for t in empty.axes[0].texts)


def test_heatmap_marks_chosen_cell():
    rows = [
        {
            "damping": d,
            "smoothing": s,
            "success": True,
            "variance_s2": 1e-3 * (1 + d / s),
            "model_variance": 1e-2 / d,
            "pert_roughness_phys": 1e-4 / s,
        }
        for d in (1, 10, 100)
        for s in (1, 10)
    ]
    fig = pl.plot_tradeoff_heatmap(pd.DataFrame(rows), (10.0, 1.0))
    assert len(fig.axes) >= 3
    assert all(len(ax.patches) == 1 for ax in fig.axes[:3])


def test_run_iterations_and_misfit_figure(tmp_path):
    run = _make_run(tmp_path, 5, 10, [100.0, 60.0, 50.0])
    df = pl.read_run_iterations(run)
    assert list(df["rms_ms"]) == [100.0, 60.0, 50.0]
    assert pl.read_run_iterations(tmp_path / "missing").empty
    fig = pl.plot_misfit_vs_iteration({"a": df, "b": df * 1.1}, highlight="a")
    assert len(fig.axes[0].lines) == 2


def test_neighbour_slices(tmp_path):
    runs = [(f"damping {d}", _make_run(tmp_path, d, 10, [100.0, 50.0])) for d in (5, 20, 50)]
    fig = pl.plot_neighbour_slices(runs, depth_km=2.0)
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>" and a.collections]
    assert len(panels) == 3
    bad = pl.plot_neighbour_slices([("x", tmp_path / "nothing")], 2.0)
    assert any("no model" in t.get_text() for t in bad.axes[0].texts)


def test_inversion_figures_saved(tmp_path):
    it = _iter_table([100.0, 60.0, 45.0, 42.0])
    rng = np.random.default_rng(0)
    tables = {
        i: pd.DataFrame(
            {
                "residual": rng.normal(0, 0.1 / (1 + i), 400),
                "station": rng.choice(list("ABCD"), 400),
                "phase": "P",
                "sta_lat": -38.5,
                "sta_lon": 176.2,
            }
        )
        for i in range(4)
    }
    hyp = pd.DataFrame(
        [
            {
                "iter": i,
                "source": s,
                "lat": -38.5 + 0.001 * i * (s % 3),
                "lon": 176.2,
                "depth_km": 3.0 + 0.1 * i,
            }
            for i in range(4)
            for s in range(1, 10)
        ]
    )
    reg = FigureRegistry(tmp_path / "figs", ("png",), 60)
    names = pl.make_inversion_figures(
        reg.save, it, tables={i: t for i, t in tables.items() if i in (0, 3)}, hypocentres=hyp,
        min_velocity=0.5,
    )  # fmt: skip
    for n in (
        "misfit_vs_iteration",
        "residual_histograms_iterations",
        "hypocentre_shifts_iterations",
        "rays_lost_iterations",
        "velocity_range_iterations",
    ):
        assert n in names and (tmp_path / "figs" / f"{n}.png").stat().st_size > 1000
    assert [e["name"] for e in reg.entries()] == names


def test_tradeoff_figures_from_result(tmp_path):
    ds = [1, 5, 10, 20, 50, 100, 200]
    ss = [1, 10, 100]
    d_ch = 20.0
    rows = []
    for d in ds:
        rows.append(_row(d, 10.0, tmp_path))
    for s in ss:
        if s != 10:
            rows.append(_row(d_ch, s, tmp_path))
    pd_ = _pick("damping", d_ch, 50.0, ds)
    pd_["at_smoothing"] = 10.0
    ps_ = _pick("smoothing", 10.0, 100.0, ss + [30])
    ps_["at_damping"] = d_ch
    res = {
        "picks": {"damping": pd_, "smoothing": ps_},
        "table": rows,
        "mode": "sequential",
        "chosen_damping": d_ch,
        "chosen_smoothing": 10.0,
    }
    reg = FigureRegistry(tmp_path / "figs", ("png",), 60)
    names = pl.make_tradeoff_figures(reg.save, res, tmp_path, depth_km=2.0)
    assert names == [
        "lcurve_damping",
        "lcurve_smoothing",
        "misfit_vs_iteration_sweep",
        "neighbour_slices",
    ]
    res["mode"] = "full_grid"
    names = pl.make_tradeoff_figures(reg.save, res, tmp_path, suffix="_S")
    assert "tradeoff_heatmap_S" in names


def _row(d: float, s: float, root: Path) -> dict:
    run = _make_run(root, d, s, [100.0, 50.0 + d / 10])
    return {
        "damping": d,
        "smoothing": s,
        "success": True,
        "variance_s2": 1e-3,
        "model_variance": 1e-3 / d,
        "pert_roughness_phys": 1e-3 / s,
        "converged": True,
        "chi2_reduced": 1.0,
        "dir": str(run),
    }


# ---------------------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------------------
def _codes(fs):
    return {f.code: f for f in fs}


def test_inversion_findings_clean_run():
    it = _iter_table([100.0, 50.0, 30.0, 29.5], chi2=[3.0, 2.0, 1.1, 1.0])
    assert pl.inversion_findings(it, 0.5) == []


def test_inversion_findings_misfit_problems():
    up = _iter_table([100.0, 60.0, 70.0, 65.0])
    c = _codes(pl.inversion_findings(up))
    assert c["misfit_increasing"].severity == "warn"
    assert "after iteration 1" in c["misfit_increasing"].message
    worse = _codes(pl.inversion_findings(_iter_table([100.0, 110.0, 120.0])))
    assert worse["misfit_not_decreasing"].severity == "fail"
    flat = _codes(pl.inversion_findings(_iter_table([100.0, 99.9, 99.0, 98.5])))
    assert flat["misfit_not_decreasing"].severity == "warn"
    assert "no iteration results" in pl.inversion_findings(pd.DataFrame())[0].message


def test_inversion_findings_clipping_lost_chi2_convergence():
    it = _iter_table(
        [100.0, 70.0, 50.0, 40.0],
        frac_at_min_velocity=[0.0, 0.01, 0.02, 0.05],
        n_valid=[1000.0, 900.0, 880.0, 870.0],
        chi2=[9.0, 7.0, 5.0, 4.0],
    )
    c = _codes(pl.inversion_findings(it, 0.5, "S"))
    assert (
        c["min_velocity_clipped"].severity == "warn"
        and "5.00 %" in c["min_velocity_clipped"].message
    )
    assert c["rays_lost"].count == 130
    assert "chi-squared is 4.00" in c["chi2_far_from_one"].message
    assert "sqrt(chi2)" in c["chi2_far_from_one"].message  # comment on uncertainty scaling
    assert "uncertainties are too small" in c["chi2_far_from_one"].message
    assert c["no_convergence"].severity == "info"
    assert c["rays_lost"].message.startswith("S: ")
    low = pl.inversion_findings(_iter_table([100.0, 50.0, 30.0, 29.0], chi2=[1, 1, 0.3, 0.2]))
    assert "too large" in _codes(low)["chi2_far_from_one"].message
    assert pl.chi2_far_from_one(float("nan")) is False and pl.chi2_far_from_one(1.5) is False


def test_tradeoff_findings():
    ds = [1, 5, 10, 20, 50, 100]
    good = {
        "picks": {
            "damping": _pick("damping", 20.0, 50.0, ds),
            "smoothing": _pick("smoothing", 10.0, 50.0, [1, 5, 10, 20, 50]),
        },
        "table": [],
        "chosen_damping": 20.0,
        "chosen_smoothing": 10.0,
    }
    assert pl.tradeoff_findings(good) == []
    bad = {
        "picks": {
            "damping": _pick("damping", 20.0, 50.0, [1, 10]),  # no corner
            "smoothing": _pick("smoothing", 0.01, 50.0, [1, 5, 10, 20, 50]),  # corner at edge
        },
        "table": [
            {"damping": 1, "smoothing": 5, "success": False, "converged": False},
            {
                "damping": 5,
                "smoothing": 5,
                "success": True,
                "converged": False,
                "chi2_reduced": 5.0,
            },
        ],  # fmt: skip
        "chosen_damping": 5.0,
        "chosen_smoothing": 5.0,
    }
    c = _codes(pl.tradeoff_findings(bad, "P"))
    assert c["lcurve_corner"].severity == "warn" and c["lcurve_corner"].message.startswith("P: ")
    assert "sweep_failed" in c and "no_convergence" in c and "chi2_far_from_one" in c
    edge = _pick("smoothing", 5.0, 50.0, [1, 5, 10, 20, 50])
    edge.update(chosen=5.0, source="max_curvature")  # second value of the sweep
    c = _codes(pl.tradeoff_findings({**good, "picks": {**good["picks"], "smoothing": edge}}))
    assert c["lcurve_at_edge"].severity == "warn" and "smoothing" in c["lcurve_at_edge"].message
