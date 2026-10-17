"""Tests for osv.velest.plots (headless, synthetic inputs)."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from osv.config.schema import VelestSelect
from osv.velest import plots
from osv.velest.ensemble import EnsembleResult, Finding
from osv.velest.stability import StabilityResult

TOPS = np.array([-0.5, 0.5, 1.5, 2.5])
BOUNDS = [(2.0, 4.0), (3.0, 5.0), (4.0, 6.0), (5.0, 7.0)]
N_RUN = 14
N_EV = 12
STAS = [f"S{i:02d}" for i in range(8)]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _stations(rng):
    return pd.DataFrame(
        {
            "net": "NZ",
            "sta": STAS,
            "alias": STAS,
            "lat": -38.5 + rng.normal(0, 0.03, len(STAS)),
            "lon": 176.2 + rng.normal(0, 0.03, len(STAS)),
            "elevation_m": rng.uniform(250, 500, len(STAS)),
            "sensor_depth_m": [0.0] * 7 + [200.0],
            "sensor_elevation_m": [300.0] * 7 + [100.0],
        }
    )


def _ensemble(rng, with_s: bool = True):
    ids = [f"MC_{i:04d}" for i in range(N_RUN)]
    rms = np.sort(rng.uniform(0.07, 0.12, N_RUN))
    rows = []
    for rid in ids:
        v0 = np.sort(rng.uniform(*zip(*BOUNDS, strict=True)))
        v1 = v0 + rng.normal(0, 0.1, len(TOPS))
        for ph, f in (("P", 1.0), ("S", 1 / 1.72)):
            if ph == "S" and not with_s:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "id": rid,
                        "phase": ph,
                        "layer": range(len(TOPS)),
                        "depth_top": TOPS,
                        "v_initial": v0 * f,
                        "v_final": v1 * f,
                        "dv": (v1 - v0) * f,
                        "hits": 100.0,
                    }
                )
            )
    fm = pd.concat(rows, ignore_index=True)
    stop = ["ittmax"] * (N_RUN - 3) + ["four_backups", "ittmax", "ittmax"]
    runs = pd.DataFrame(
        {
            "id": ids,
            "status": "ok",
            "converged": [i % 5 == 0 for i in range(N_RUN)],
            "diverged": [False] * (N_RUN - 2) + [True, False],
            "failed": [False] * (N_RUN - 1) + [True],
            "finished": True,
            "n_iter": 20,
            "n_backups": 2,
            "initial_rms": 0.2,
            "final_rms": rms,
            "stop_reason": stop,
            "wall_time": 1.0,
            "has_model": True,
        }
    )
    runs["eligible"] = ~(runs["diverged"] | runs["failed"])
    elig = list(runs.loc[runs["eligible"], "id"])
    runs["selected"] = runs["id"].isin(elig[:3])
    sel = elig[:3]
    p = fm[(fm["phase"] == "P") & fm["id"].isin(sel)].pivot(
        index="id", columns="layer", values="v_final"
    )
    final = pd.DataFrame(
        {
            "layer": range(len(TOPS)),
            "depth_top": TOPS,
            "vp": p.median().to_numpy(),
            "vs": p.median().to_numpy() / 1.72 if with_s else np.nan,
            "vpvs": 1.72,
            "vp_std": p.std().to_numpy(),
            "vp_p05": p.min().to_numpy(),
            "vp_p95": p.max().to_numpy(),
            "vs_std": 0.05,
            "vs_p05": p.min().to_numpy() / 1.72,
            "vs_p95": p.max().to_numpy() / 1.72,
            "vp_initial_std": 0.5,
            "hits": 100.0,
        }
    )
    if not with_s:
        final[["vs_p05", "vs_p95", "vpvs"]] = np.nan
    hyp = []
    for rid in sel + [elig[3]]:
        h = pd.DataFrame(
            {
                "id": rid,
                "event": range(N_EV),
                "lat": -38.5 + 0.001 * rng.normal(size=N_EV) + 0.02 * np.arange(N_EV) / N_EV,
                "lon": 176.2 + 0.001 * rng.normal(size=N_EV),
                "depth": rng.uniform(1, 4, N_EV),
                "rms": rng.uniform(0.04, 0.1, N_EV),
            }
        )
        hyp.append(h)
    hyp = pd.concat(hyp, ignore_index=True)
    dl = pd.DataFrame(
        [
            {
                "id": rid,
                "sta": s,
                "ptcor": rng.normal(0, 0.1),
                "stcor": rng.normal(0, 0.2) if with_s else np.nan,
                "nobs": 0,
            }
            for rid in sel
            for s in STAS
        ]
    )
    ss = dl.groupby("sta").agg(
        ptcor_median=("ptcor", "median"),
        ptcor_std=("ptcor", "std"),
        ptcor_p05=("ptcor", lambda x: x.quantile(0.05)),
        ptcor_p95=("ptcor", lambda x: x.quantile(0.95)),
        stcor_median=("stcor", "median"),
        stcor_std=("stcor", "std"),
    )
    ss["nobs"] = 0.0
    ss = ss.reset_index()
    ens = EnsembleResult(
        runs=runs,
        final_models=fm,
        station_delays=dl,
        hypocentres=hyp,
        final_model=final,
        selected_ids=sel,
        eligible_ids=elig,
        closest_run_id=sel[1],
        closest_rms=float(rms[1]),
        best_run_id=elig[0],
        best_rms=float(rms[0]),
        station_summary=ss,
        select=VelestSelect(),
        findings=[Finding("x", "warn", "m", layer=1, data={"a": 1})],
        convergence_rate=0.3,
        rms_range=(0.07, 0.12),
        reference_station="S00",
    )
    return ens


def _inputs(ens):
    h = ens.hypocentres[ens.hypocentres["id"] == ens.closest_run_id]
    return pd.DataFrame(
        {
            "cnv_id": h["event"] + 1,
            "lat": h["lat"] + 0.002,
            "lon": h["lon"] - 0.002,
            "depth_km": h["depth"] + 0.5,
        }
    )


def _parsed(rng, n=8):
    rows = [
        {
            "run": 1,
            "entry": 0,
            "iteration": 0,
            "kind": "initial",
            "backup_nr": 0,
            "accepted": True,
            "rms": 0.15,
        }
    ]
    for k in range(1, n):
        rows.append(
            {
                "run": 1,
                "entry": 2 * k,
                "iteration": k,
                "kind": "iteration",
                "backup_nr": 0,
                "accepted": k % 3 != 0,
                "rms": 0.15 - 0.01 * k,
            }
        )
        if k % 3 == 0:
            rows.append(
                {
                    "run": 1,
                    "entry": 2 * k + 1,
                    "iteration": k,
                    "kind": "backup",
                    "backup_nr": 1,
                    "accepted": True,
                    "rms": 0.15 - 0.01 * k + 0.002,
                }
            )
    it = pd.DataFrame(rows)
    stats = pd.DataFrame(
        [
            {
                "sta": s,
                "phase": ph,
                "nobs": 10,
                "avres": rng.normal(0, 0.02),
                "avwres": 0.0,
                "std": 0.05,
                "wsum": 1.0,
                "delay": 0.0,
            }
            for s in STAS
            for ph in "PS"
        ]
    )
    return SimpleNamespace(iterations=it, station_stats=stats)


def _workflow(ens, rng):
    runs = ens.runs[["id", "final_rms"]].copy()
    runs["parsed"] = [_parsed(rng) for _ in range(len(runs))]
    hist = pd.DataFrame(
        {
            "round": 1,
            "stage_index": [1] * N_RUN + [2] * N_RUN,
            "stage": ["high_damping"] * N_RUN + ["final"] * N_RUN,
            "id": list(ens.runs["id"]) * 2,
            "status": "ok",
            "final_rms": rng.uniform(0.07, 0.12, 2 * N_RUN),
            "n_iterations": rng.integers(3, 30, 2 * N_RUN),
        }
    )
    return SimpleNamespace(runs=runs, history=hist)


def _stability(rng):
    ev, runs, mods, lay, dl = [], [], [], [], []

    def add_models(test, variant, rid, key, base):
        for k, t in enumerate(TOPS):
            mods.append(
                {
                    "test": test,
                    "variant": variant,
                    "id": rid,
                    "model_key": key,
                    "layer": k,
                    "depth_top": t,
                    "vp": base[k] + rng.normal(0, 0.03),
                    "vs": np.nan,
                    "hits": 10.0,
                }
            )

    base = np.array([3.0, 4.0, 5.0, 6.0])
    add_models("hypocentre_perturbation", "baseline_free", "bf", "final", base)
    runs.append(
        {
            "test": "hypocentre_perturbation",
            "variant": "baseline_free",
            "id": "bf",
            "status": "ok",
            "rms": 0.07,
        }
    )
    for variant in ("fixed", "free"):
        for j in range(3):
            rid = f"{variant}{j}"
            add_models("hypocentre_perturbation", variant, rid, "final", base)
            runs.append(
                {
                    "test": "hypocentre_perturbation",
                    "variant": variant,
                    "id": rid,
                    "status": "ok",
                    "rms": 0.07,
                }
            )
            for e in range(N_EV):
                pz = rng.uniform(-1, 1)
                ev.append(
                    {
                        "test": "hypocentre_perturbation",
                        "variant": variant,
                        "id": rid,
                        "event": e,
                        "pert_dx_km": 0.0,
                        "pert_dy_km": 0.0,
                        "pert_dz_km": pz,
                        "dx_km": 0.0,
                        "dy_km": 0.0,
                        "dh_km": abs(rng.normal(0, 0.05)),
                        "dz_km": rng.normal(0, 0.1),
                    }
                )
    for v in ("fixed", "free"):
        for k, t in enumerate(TOPS):
            m = np.array([x["vp"] for x in mods if x["variant"] == v and x["layer"] == k])
            lay.append(
                {
                    "test": "hypocentre_perturbation",
                    "variant": v,
                    "layer": k,
                    "depth_top": t,
                    "vp_start": base[k],
                    "vp_mean": m.mean(),
                    "vp_std": m.std(),
                    "vp_min": m.min(),
                    "vp_max": m.max(),
                    "vp_max_abs_dev": 0.1,
                    "vs_std": np.nan,
                    "hits": 10.0,
                }
            )
    for ref in ("S01", "S02"):
        rid = f"ref_{ref}"
        add_models("reference_station", f"ref_{ref}", rid, "final", base)
        runs.append(
            {
                "test": "reference_station",
                "variant": "swap",
                "id": rid,
                "status": "ok",
                "rms": 0.066,
            }
        )
        off = rng.normal(0, 0.05)
        for s in STAS:
            b = rng.normal(0, 0.1)
            dl.append(
                {
                    "test": "reference_station",
                    "id": rid,
                    "reference_station": ref,
                    "sta": s,
                    "ptcor": b + off,
                    "stcor": np.nan,
                    "nobs": 5,
                    "ptcor_diff": off,
                    "ptcor_resid": rng.normal(0, 0.01),
                    "stcor_diff": np.nan,
                    "stcor_resid": np.nan,
                }
            )
    add_models("reference_station", "baseline", "ref_baseline", "final", base)
    for key in ("final", "closest_MC_1"):
        for v, lvz in (("lvz0", 0), ("lvz1", 1)):
            rid = f"lvz_{key}_{lvz}"
            add_models("lvz", v, rid, key, base)
            runs.append(
                {"test": "lvz", "variant": v, "id": rid, "status": "ok", "rms": 0.07 - 0.001 * lvz}
            )
    return StabilityResult(
        runs=pd.DataFrame(runs),
        events=pd.DataFrame(ev),
        models=pd.DataFrame(mods),
        layers=pd.DataFrame(lay),
        delays=pd.DataFrame(dl),
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    ens = _ensemble(rng)
    return SimpleNamespace(
        ens=ens,
        stations=_stations(rng),
        inputs=_inputs(ens),
        wf=_workflow(ens, rng),
        stab=_stability(rng),
        rng=rng,
        grid=pd.DataFrame(
            {
                "vpvs": [1.64, 1.68, 1.72, 1.76],
                "final_rms": [0.071, 0.070, 0.0705, 0.073],
                "converged": [True, False, True, False],
            }
        ),
    )


def _ok(fig):
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    assert len(fig.axes) >= 1
    return fig


def test_spaghetti(data):
    fig = _ok(plots.plot_spaghetti(data.ens, station_elevations_m=[100, 300, 450]))
    assert len(fig.axes) >= 4  # Vp, Vs, depth histogram, colour bar
    assert fig.axes[0].yaxis_inverted()


def test_spaghetti_vpvs_panel_and_no_s(data):
    f = _ok(plots.plot_spaghetti(data.ens, second="vpvs", hypocentre_depths=[1, 2, 3]))
    assert "Vp/Vs" in f.axes[1].get_xlabel()
    ens = _ensemble(np.random.default_rng(1), with_s=False)
    f2 = _ok(plots.plot_spaghetti(ens))
    assert not any("Vs" in a.get_xlabel() for a in f2.axes)


def test_density_histograms_rms_scatter(data):
    _ok(plots.plot_velocity_density(data.ens, 30))
    f = _ok(plots.plot_layer_histograms(data.ens, BOUNDS))
    assert sum(a.has_data() for a in f.axes) == len(TOPS)
    _ok(plots.plot_rms_vs_layer_velocity(data.ens, BOUNDS))
    _ok(plots.plot_layer_histograms(data.ens))  # without bounds


def test_iteration_and_stage(data):
    f = _ok(plots.plot_rms_vs_iteration(data.wf.runs, data.ens.selected_ids))
    assert len(f.axes[0].lines) >= N_RUN
    _ok(plots.plot_stage_rms(data.wf.history))
    assert plots.plot_rms_vs_iteration(data.wf.runs.assign(parsed=None)) is None
    assert plots.plot_stage_rms(pd.DataFrame()) is None


def test_convergence_classes(data):
    c = plots.classify_runs(data.ens.runs)
    assert c.iloc[-1] == "failed" and c.iloc[-2] == "diverged"
    assert (c == "converged").sum() >= 1
    _ok(plots.plot_convergence(data.ens))


def test_station_delay_figures(data):
    f = _ok(plots.plot_station_delay_map(data.ens, data.stations))
    assert len(f.axes) == 4  # P and S maps with their colour bars
    f = _ok(plots.plot_delay_vs_elevation(data.ens, data.stations))
    assert len([a for a in f.axes if a.has_data()]) == 2
    # without S delays only the P panel remains; without stations: None
    ens = data.ens
    ens.station_summary = ens.station_summary.assign(stcor_median=np.nan)
    assert len(plots.plot_delay_vs_elevation(ens, data.stations).axes) == 1
    assert plots.plot_station_delay_map(ens, None) is None
    assert plots.plot_delay_vs_elevation(ens, None) is None


def test_hypocentres(data):
    _ok(plots.plot_hypocentre_shifts(data.ens, data.inputs, data.stations))
    assert plots.plot_hypocentre_shifts(data.ens, None) is None
    h = data.ens.hypocentres
    _ok(plots.plot_final_hypocentres(h[h["id"] == data.ens.closest_run_id], data.stations))
    _ok(plots.plot_final_hypocentres(h, None))
    assert plots.plot_final_hypocentres(pd.DataFrame()) is None


def test_hypocentre_shifts_from_precomputed_columns(data):
    h = data.ens.hypocentres.copy()
    h["dlat_km"], h["dlon_km"], h["ddepth_km"] = 0.2, -0.1, 0.3
    data.ens.hypocentres = h
    _ok(plots.plot_hypocentre_shifts(data.ens))


def test_stability_figures(data):
    _ok(plots.plot_hypocentre_perturbation(data.stab))
    _ok(plots.plot_stability_velocity_spread(data.stab))
    _ok(plots.plot_reference_swap(data.stab))
    f = _ok(plots.plot_lvz_comparison(data.stab))
    legend = [t.get_text() for t in f.axes[0].get_legend().get_texts()]
    assert any("RMS" in t for t in legend)
    empty = StabilityResult()
    for fn in (
        plots.plot_hypocentre_perturbation,
        plots.plot_stability_velocity_spread,
        plots.plot_reference_swap,
        plots.plot_lvz_comparison,
    ):
        assert fn(empty) is None


def test_residuals_and_vpvs(data):
    stats = data.wf.runs["parsed"].iloc[0].station_stats
    hy = data.ens.hypocentres.query("id == @data.ens.closest_run_id")
    _ok(plots.plot_residuals(stats, hy))
    _ok(plots.plot_residuals(stats))
    _ok(plots.plot_residuals(None, hy))
    assert plots.plot_residuals(None, None) is None
    g = data.grid.copy()
    g.attrs.update(best_vpvs=1.68, best_vpvs_parabola=1.70)
    _ok(plots.plot_vpvs(g, {"vpvs": 1.71, "vpvs_ci": [1.68, 1.74]}))
    _ok(plots.plot_vpvs(g))
    assert plots.plot_vpvs(None) is None
    assert plots.plot_vpvs(g.assign(final_rms=np.nan)) is None


def test_make_all_figures(data):
    saved = []

    def save(fig, name, caption, section=None):
        assert isinstance(fig, Figure) and caption and section
        saved.append(name)
        plt.close(fig)

    names = plots.make_all_figures(
        save,
        ensemble=data.ens,
        workflow=data.wf,
        stability=data.stab,
        vpvs_grid=data.grid,
        wadati={"vpvs": 1.7, "vpvs_ci": [1.65, 1.75]},
        stations=data.stations,
        input_hypocentres=data.inputs,
        vp_bounds=BOUNDS,
    )
    assert names == saved
    for want in (
        "velocity_spaghetti",
        "velocity_density",
        "layer_histograms",
        "rms_vs_layer_velocity",
        "rms_vs_iteration",
        "stage_rms",
        "convergence_summary",
        "station_delay_map",
        "delay_vs_elevation",
        "hypocentre_shifts",
        "final_hypocentres",
        "stability_hypocentre_perturbation",
        "stability_velocity_spread",
        "stability_reference_station",
        "stability_lvz",
        "residuals",
        "vpvs",
    ):
        assert want in names, want


def test_make_all_figures_minimal_and_failure(data, monkeypatch, caplog):
    saved = []
    names = plots.make_all_figures(
        lambda fig, name, cap, sec=None: saved.append(name), ensemble=data.ens
    )
    assert "velocity_spaghetti" in names and "stability_lvz" not in names
    assert "rms_vs_iteration" not in names and "vpvs" not in names

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(plots, "plot_velocity_density", boom)
    names = plots.make_all_figures(lambda *a, **k: None, ensemble=data.ens)
    assert "velocity_density" not in names and "velocity_spaghetti" in names


def test_to_qc_findings():
    out = plots.to_qc_findings(
        [Finding("a", "error", "bad", layer=2, data={"k": 1}), Finding("b", "info", "ok")]
    )
    assert [f.severity for f in out] == ["fail", "info"]
    assert out[0].details == {"k": 1, "layer": 2}
