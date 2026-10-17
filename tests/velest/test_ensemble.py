from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import VelestSelect
from osv.io.velest import ModelBlock, VelestModel, read_model
from osv.tools import ToolNotFoundError
from osv.velest.ensemble import (
    EnsembleError,
    QCThresholds,
    analyse_ensemble,
    final_model_checks,
    with_final_model,
    write_ensemble,
)
from osv.velest.models import TrialModel

DEPTHS = np.array([0.0, 1.0, 2.0, 3.0])
STAS = ["REF", "A", "B"]


def _block(v):
    return ModelBlock(pd.DataFrame({"vel": v, "depth": DEPTHS}))


def fake(
    vp0,
    vp1,
    rms,
    *,
    vpvs=1.75,
    hits=(500, 500, 500, 500),
    converged=True,
    diverged=False,
    failed=False,
    finished=True,
    ptcor=(0.0, 0.1, -0.1),
    n_iter=5,
):
    vp0, vp1 = np.asarray(vp0, float), np.asarray(vp1, float)
    rs = pd.DataFrame({"nlay": np.arange(1, 5), "nhit": hits})
    sc = pd.DataFrame(
        {"sta": STAS, "ptcor": ptcor, "stcor": np.array(ptcor) * 1.5, "nobs": [10, 20, 30]}
    )
    ev = pd.DataFrame(
        {"event": [0, 1], "lat": [-38.0, -38.1], "lon": [176.0, 176.1], "depth_km": [2.0, 3.0],
         "rms": [0.1, 0.1]}
    )  # fmt: skip
    return SimpleNamespace(
        converged=converged, diverged=diverged, failed=failed, finished=finished,
        n_iterations=n_iter, n_backups=0, initial_rms=rms * 3, final_rms=rms,
        stop_reason="delmin" if converged else "ittmax",
        initial_model=VelestModel("i", _block(vp0), _block(vp0 / vpvs)),
        final_model=VelestModel("f", _block(vp1), _block(vp1 / vpvs)),
        ray_statistics=rs, station_corrections=sc, cnv=SimpleNamespace(events=ev),
        final_hypocentres=pd.DataFrame(),
    )  # fmt: skip


def table(items):
    rows = [
        {"id": k, "status": "ok" if p is not None else "failed", "wall_time": 1.0,
         "run_dir": "none", "parsed": p}
        for k, p in items.items()
    ]  # fmt: skip
    return pd.DataFrame(rows)


def ensemble(n=10, seed=0):
    """10 monotonic runs, rms 0.10 .. 0.19, final model scattered around a truth."""
    rng = np.random.default_rng(seed)
    truth = np.array([2.0, 3.0, 4.0, 5.0])
    items = {}
    for k in range(n):
        v0 = np.sort(rng.uniform([1.5, 2.5, 3.5, 4.5], [3.0, 4.0, 5.0, 6.0]))
        v1 = truth + (k - n / 2) * 0.01 * np.array([1, -1, 1, -1])
        items[f"R{k:02d}"] = fake(v0, v1, 0.10 + 0.01 * k)
    return items


CFG = VelestSelect(method="best_n", value=5, statistic="median")


def codes(res, severity=None):
    return {f.code for f in res.findings if severity is None or f.severity == severity}


def test_tables_and_best_n_median():
    items = ensemble()
    res = analyse_ensemble(table(items), None, CFG)
    assert res.selected_ids == ["R00", "R01", "R02", "R03", "R04"]
    assert list(res.runs.columns[:8]) == [
        "id", "status", "converged", "diverged", "failed", "finished", "n_iter", "n_backups",
    ]  # fmt: skip
    assert {"v_initial", "v_final", "dv", "hits", "depth_top", "phase", "layer"} <= set(
        res.final_models.columns
    )
    assert len(res.final_models) == 10 * 4 * 2
    assert len(res.station_delays) == 30 and len(res.hypocentres) == 20
    v = np.array([items[i].final_model.p.layers["vel"].to_numpy() for i in res.selected_ids])
    assert np.allclose(res.final_model["vp"], np.median(v, axis=0))
    assert np.allclose(res.final_model["vp_std"], v.std(axis=0, ddof=1))
    assert np.allclose(res.final_model["vp_p05"], np.percentile(v, 5, axis=0))
    assert np.allclose(res.final_model["vpvs"], 1.75)
    assert res.rms_range == pytest.approx((0.10, 0.19))
    # median of 5 runs is the middle run's model for monotonic one-parameter spread
    assert res.closest_run_id == "R02"
    assert res.closest_rms == pytest.approx(0.12)
    assert res.best_run_id == "R00"


def test_mean_and_best_statistics():
    items = ensemble()
    mean = analyse_ensemble(
        table(items), None, VelestSelect(method="best_n", value=4, statistic="mean")
    )
    v = np.array([items[i].final_model.p.layers["vel"].to_numpy() for i in mean.selected_ids])
    assert np.allclose(mean.final_model["vp"], v.mean(axis=0))
    best = analyse_ensemble(
        table(items), None, VelestSelect(method="best_n", value=4, statistic="best")
    )
    assert np.allclose(best.final_model["vp"], items["R00"].final_model.p.layers["vel"])
    assert best.closest_run_id == "R00"


def test_best_percent_and_rms_below():
    items = ensemble()
    r = analyse_ensemble(table(items), None, VelestSelect(method="best_percent", value=25))
    assert len(r.selected_ids) == 3  # ceil(2.5)
    r = analyse_ensemble(table(items), None, VelestSelect(method="best_percent", value=1))
    assert r.selected_ids == ["R00"]
    r = analyse_ensemble(table(items), None, VelestSelect(method="rms_below", value=0.125))
    assert r.selected_ids == ["R00", "R01", "R02"]
    with pytest.raises(EnsembleError, match="below"):
        analyse_ensemble(table(items), None, VelestSelect(method="rms_below", value=0.01))


def test_only_good_runs_eligible():
    items = ensemble()
    items["bad_fail"] = fake([1, 2, 3, 4], [2, 3, 4, 5], 0.001, failed=True)
    items["bad_div"] = fake([1, 2, 3, 4], [2, 3, 4, 5], 0.002, diverged=True)
    items["bad_unfin"] = fake([1, 2, 3, 4], [2, 3, 4, 5], 0.003, finished=False)
    items["bad_status"] = None
    res = analyse_ensemble(table(items), None, CFG)
    assert "bad_fail" not in res.eligible_ids and "bad_div" not in res.eligible_ids
    assert "bad_unfin" not in res.eligible_ids and "bad_status" not in res.eligible_ids
    assert res.best_run_id == "R00"
    assert not res.runs.set_index("id").loc["bad_status", "eligible"]
    f = next(f for f in res.findings if f.code == "failures")
    assert f.severity == "warn" and f.data == {"failed": 2, "diverged": 1}
    with pytest.raises(EnsembleError, match="none of"):
        analyse_ensemble(
            table({"x": fake([1, 2, 3, 4], [1, 2, 3, 4], 0.1, failed=True)}), None, CFG
        )


def test_convergence_rate_warning():
    items = ensemble()
    res = analyse_ensemble(table(items), None, CFG)
    assert res.convergence_rate == 1.0
    assert "convergence_rate" not in codes(res, "warn")
    for k in list(items)[:3]:
        items[k] = fake([2, 3, 4, 5], [2, 3, 4, 5], 0.1, converged=False)
    res = analyse_ensemble(table(items), None, CFG)
    assert res.convergence_rate == pytest.approx(0.7)
    assert "convergence_rate" in codes(res, "warn")


def test_bound_cluster_warning():
    items = {}
    rng = np.random.default_rng(1)
    for k in range(8):
        v1 = np.array([2.0, 3.0, 4.0, 5.0 - 0.01 * k]) + np.array([0, 0, 0, 0.0])
        v1[0] = 1.5 + 0.005 * k  # at the lower bound of layer 0
        items[f"R{k}"] = fake(rng.uniform(1.5, 2.5, 4).cumsum() / 2, v1, 0.1 + 0.01 * k)
    bounds = [(1.5, 3.0), (2.5, 4.0), (3.5, 5.0), (4.5, 6.0)]
    res = analyse_ensemble(
        table(items), None, VelestSelect(method="best_n", value=6), vp_bounds=bounds
    )
    warn = [f for f in res.findings if f.code == "at_bound"]
    assert [(f.layer, f.data["side"]) for f in warn] == [(0, "lower")]


def test_bounds_from_trial_models():
    items = ensemble()
    tms = [
        TrialModel(
            k,
            DEPTHS,
            items[k].initial_model.p.layers["vel"],
            items[k].initial_model.s.layers["vel"],
            "u",
            0,
        )
        for k in items
    ]
    res = analyse_ensemble(table(items), tms, CFG)
    assert res.final_model.shape[0] == 4  # bounds derived without error


def test_unresolved_layer_and_low_hits():
    rng = np.random.default_rng(2)
    items = {}
    for k in range(12):
        v0 = np.array([2.0, 3.0, 4.0, rng.uniform(4.5, 6.0)])
        v1 = v0 + np.array([0.3, 0.3, 0.3, 0.0])  # last layer never moves
        items[f"R{k}"] = fake(v0, v1, 0.1 + 0.01 * k, hits=(500, 500, 500, 20))
    res = analyse_ensemble(table(items), None, VelestSelect(method="best_n", value=5))
    unres = {f.layer for f in res.findings if f.code == "unresolved_layer"}
    # layers 0-2 had no start spread -> sd_i == 0 and dv != 0, so only layer 3
    assert unres == {3}
    low = [f for f in res.findings if f.code == "low_hits"]
    assert [f.layer for f in low] == [3]
    assert res.final_model["hits"].iloc[3] == 20


def test_lvz_finding():
    items = {f"R{k}": fake([2, 3, 4, 5], [2.0, 3.5, 3.0, 5.0], 0.1 + 0.01 * k) for k in range(5)}
    res = analyse_ensemble(table(items), None, CFG)
    lvz = [f for f in res.findings if f.code == "lvz"]
    assert [(f.layer, f.severity) for f in lvz] == [(2, "warn")]
    items = {f"R{k}": fake([2, 3, 4, 5], [2.0, 3.5, 3.45, 5.0], 0.1 + 0.01 * k) for k in range(5)}
    lvz = [f for f in analyse_ensemble(table(items), None, CFG).findings if f.code == "lvz"]
    assert [(f.layer, f.severity) for f in lvz] == [(2, "info")]


def test_station_delay_findings():
    items = ensemble()
    res = analyse_ensemble(table(items), None, CFG)
    assert res.reference_station == "REF"  # inferred: zero in all selected runs
    assert next(f for f in res.findings if f.code == "reference_station").severity == "info"
    assert next(f for f in res.findings if f.code == "delay_spread").severity == "info"
    # drifting reference and wide spread
    for k, key in enumerate(items):
        items[key] = fake(
            [2, 3, 4, 5], [2, 3, 4, 5], 0.1 + 0.01 * k, ptcor=(0.05 * k, 0.5 * (-1) ** k, 0.0)
        )
    res = analyse_ensemble(table(items), None, CFG, reference_station="REF")
    assert next(f for f in res.findings if f.code == "reference_station").severity == "warn"
    ds = next(f for f in res.findings if f.code == "delay_spread")
    assert ds.severity == "warn" and "A" in ds.data["stations"]
    res = analyse_ensemble(table(items), None, CFG, reference_station="NOPE")
    assert "not in" in next(f for f in res.findings if f.code == "reference_station").message


def test_rms_improvement_warning_and_shifts():
    items = ensemble()
    for k in items.values():
        k.initial_rms = k.final_rms * 1.05
    res = analyse_ensemble(table(items), None, CFG)
    assert next(f for f in res.findings if f.code == "rms_improvement").severity == "warn"
    inp = pd.DataFrame(
        {"event": [0, 1], "lat": [-38.0, -38.0], "lon": [176.0, 176.0], "depth_km": [1.0, 3.0]}
    )
    res = analyse_ensemble(table(ensemble()), None, CFG, input_hypocentres=inp)
    h = res.hypocentres
    assert {"dlat_km", "dlon_km", "ddepth_km", "dh_km"} <= set(h.columns)
    e1 = h[(h["id"] == "R00") & (h["event"] == 1)].iloc[0]
    assert e1["dlat_km"] == pytest.approx(-0.1 * 111.195) and e1["ddepth_km"] == 0.0


def test_write_outputs(tmp_path):
    res = analyse_ensemble(table(ensemble()), None, CFG)
    paths = write_ensemble(res, tmp_path / "out", vdamp=0.5)
    names = {p.name for p in paths.values()}
    expected = {"final_model.mod", "final_model.csv", "station_delays.csv", "runs.parquet"}
    expected |= {"final_models.parquet", "station_delays_all.parquet", "summary.json"}
    assert expected <= names
    m = read_model(tmp_path / "out" / "final_model.mod")
    assert np.allclose(m.p.layers["vel"], res.final_model["vp"], atol=0.006)
    assert np.allclose(m.s.layers["vel"], res.final_model["vs"], atol=0.006)
    assert (m.p.layers["vdamp"] == 0.5).all()
    csv = pd.read_csv(tmp_path / "out" / "final_model.csv")
    assert {"depth", "vp", "vs", "vpvs", "vp_std", "vs_std"} <= set(csv.columns)
    sd = pd.read_csv(tmp_path / "out" / "station_delays.csv")
    assert {"ptcor_median", "ptcor_std", "ptcor_closest"} <= set(sd.columns)
    assert len(pd.read_parquet(tmp_path / "out" / "runs.parquet")) == 10
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["closest_run_id"] == res.closest_run_id and s["n_selected"] == 5


def test_thresholds_configurable():
    items = ensemble()
    res = analyse_ensemble(table(items), None, CFG, thresholds=QCThresholds(min_hits=1000))
    assert len([f for f in res.findings if f.code == "low_hits"]) == 4


def test_parses_run_dir_when_no_parsed(tmp_path, monkeypatch):
    import osv.io.velest_out as vo

    calls = []
    monkeypatch.setattr(
        vo, "read_run_dir", lambda d: calls.append(d) or fake([2, 3, 4, 5], [2, 3, 4, 5], 0.1)
    )
    t = pd.DataFrame(
        [{"id": "a", "status": "ok", "wall_time": 1.0, "run_dir": "/x/a", "parsed": None}]
    )
    res = analyse_ensemble(t, None, CFG)
    assert calls == ["/x/a"] and res.selected_ids == ["a"]


# --- integration on the real data -----------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_ensemble(tmp_path):
    from osv.config.schema import OSVConfig
    from osv.data.associate import associate_picks
    from osv.data.datum import apply_depth_datum
    from osv.data.quakeml import read_catalog
    from osv.data.select import select_data
    from osv.data.stations import read_stations as read_station_xml
    from osv.data.velest_inputs import write_velest_inputs
    from osv.tools import which_tool
    from osv.velest.models import generate_trial_models
    from osv.velest.runner import jobs_from_models, run_jobs

    try:
        which_tool("velest")
    except ToolNotFoundError:
        pytest.skip("velest binary not available (set OSV_TOOLS_DIR)")

    cfg = OSVConfig()
    cfg.velest.n_models = 59  # + reference model = 60
    cat = read_catalog(_CAT)
    st = read_station_xml(_FILES)
    cat.events = apply_depth_datum(cat.events, cfg.coordinates.depth_datum)
    sel = select_data(associate_picks(cat, st), cfg)
    vi = write_velest_inputs(sel, st, cfg, tmp_path / "inputs")
    models = generate_trial_models(cfg.velest, cfg.project.random_seed)
    vi.control.ittmax = cfg.velest.ittmax
    jobs = jobs_from_models(models, vi.control, vi.cnv_path, vi.sta_path)
    table_ = run_jobs(jobs, tmp_path / "runs", n_jobs=16, timeout=1800, compact=True)
    res = analyse_ensemble(table_, models, cfg.velest.select, vp_bounds=cfg.velest.vp_bounds)
    write_ensemble(res, tmp_path / "ens")
    print("\n", res.final_model.round(3).to_string())
    print("RMS range", res.rms_range, "convergence", res.convergence_rate)
    print("selected", len(res.selected_ids), "closest", res.closest_run_id, res.closest_rms)
    for f in res.findings:
        print(f.severity, f.code, f.message)
    assert len(res.selected_ids) >= 1
    assert (tmp_path / "ens" / "final_model.mod").is_file()


# --- final-model policy and final-model checks (osv-iu9.10) ----------------------------------


def _policy(policy, items=None, **kw):
    cfg = VelestSelect(method="best_n", value=5, statistic="median", final_model=policy)
    return analyse_ensemble(table(items or ensemble()), None, cfg, **kw)


def test_final_model_policy_median_best_closest():
    items = ensemble()
    v = lambda i: items[i].final_model.p.layers["vel"].to_numpy()  # noqa: E731
    med = _policy("median")
    assert med.final_model_policy == "median"
    assert np.allclose(med.final_model["vp"], np.median([v(i) for i in med.selected_ids], axis=0))
    best = _policy("best")
    assert best.final_model_policy == "best"
    assert np.allclose(best.final_model["vp"], v("R00"))
    assert np.allclose(best.final_model["vs"], v("R00") / 1.75)
    closest = _policy("closest")
    assert np.allclose(closest.final_model["vp"], v(closest.closest_run_id))
    assert closest.closest_run_id == "R02" and closest.final_model_policy == "closest"
    assert _policy("iterated").final_model_policy == "iterated"
    # uncertainty always describes the selected ensemble, whatever the policy
    for r in (med, best, closest):
        assert np.allclose(r.final_model["vp_p05"], med.final_model["vp_p05"])
        assert np.allclose(r.final_model["vp_std"], med.final_model["vp_std"])
    assert med.summary()["final_model_policy"] == "median"


def test_with_final_model_replaces_velocities_keeps_uncertainty():
    res = _policy("iterated")
    new = with_final_model(res, [2.5, 3.5, 4.5, 5.5])
    assert np.allclose(new.final_model["vp"], [2.5, 3.5, 4.5, 5.5])
    assert np.allclose(new.final_model["vs"], np.array([2.5, 3.5, 4.5, 5.5]) / 1.75)
    assert np.allclose(new.final_model["vpvs"], 1.75)
    assert np.allclose(new.final_model["vp_p95"], res.final_model["vp_p95"])
    assert new.closest_run_id == res.closest_run_id and new is not res
    explicit = with_final_model(res, [2, 3, 4, 5], vs=[1, 2, 3, 4], policy="x")
    assert (
        np.allclose(explicit.final_model["vs"], [1, 2, 3, 4]) and explicit.final_model_policy == "x"
    )
    with pytest.raises(EnsembleError):
        with_final_model(res, [1.0, 2.0])
    # lvz findings follow the new model
    assert "lvz" not in codes(res)
    assert "lvz" in codes(with_final_model(res, [2.0, 4.0, 3.0, 5.0]))
    assert "lvz" not in codes(
        with_final_model(with_final_model(res, [2.0, 4.0, 3.0, 5.0]), [2, 3, 4, 5])
    )


def test_final_model_checks_floor_bounds_divergence():
    bounds = [[1.5, 3.0], [2.5, 4.0], [3.5, 5.0], [4.5, 6.0]]
    res = _policy("median", vp_bounds=bounds)
    assert not {"vp_below_floor", "final_out_of_bounds", "divergence_rate"} & codes(res)
    bad = with_final_model(res, [0.7, 3.0, 4.0, 5.0])
    assert {"vp_below_floor", "final_out_of_bounds"} <= codes(bad, "warn")
    f = {x.code: x for x in bad.findings}
    assert f["vp_below_floor"].layer == 0 and f["final_out_of_bounds"].data["side"] == "lower"
    # inside 10 % of the range beyond the bound is tolerated: 1.5 - 0.15 = 1.35
    assert "final_out_of_bounds" not in codes(with_final_model(res, [1.4, 3.0, 4.0, 5.0]))
    assert "final_out_of_bounds" in codes(with_final_model(res, [1.3, 3.0, 4.0, 5.0]))
    assert "vp_below_floor" in codes(with_final_model(res, [1.4, 3.0, 4.0, 5.0]))  # 1.4 < 1.5
    high = with_final_model(res, [2.0, 3.0, 4.0, 6.4])
    assert {x.data["side"] for x in high.findings if x.code == "final_out_of_bounds"} == {"upper"}
    # thresholds are configurable
    th = QCThresholds(vp_floor_km_s=0.5, out_of_bounds_fraction=1.0)
    assert not {"vp_below_floor", "final_out_of_bounds"} & {
        f.code for f in final_model_checks(bad, th)
    }


def test_divergence_rate_finding():
    items = ensemble()
    for k in range(3):  # 3 of 10 diverged (the worst RMS runs)
        items[f"R{9 - k:02d}"].diverged = True
    res = _policy("median", items)
    assert "divergence_rate" in codes(res, "warn")  # 30 % > 20 %
    items["R07"].diverged = False
    items["R08"].diverged = False
    assert "divergence_rate" not in codes(_policy("median", items))  # 10 %
    cfg = VelestSelect(method="best_n", value=5)
    strict = analyse_ensemble(
        table(items), None, cfg, thresholds=QCThresholds(max_divergence_rate=0.05)
    )
    assert "divergence_rate" in codes(strict)
