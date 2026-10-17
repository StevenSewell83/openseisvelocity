from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import OSVConfig, VelestConfig
from osv.io.velest import (
    ModelBlock,
    VelestControl,
    VelestModel,
    read_cnv,
    read_control,
    read_model,
    read_stations,
    write_cnv,
    write_stations,
)
from osv.tools import ToolNotFoundError
from osv.velest import stability as st
from osv.velest.ensemble import Finding

DUMMY = """#!/bin/sh
echo "RMS RESIDUAL= 0.12" > velest.OUT
exit 0
"""

TOPS = [0.0, 1.0, 2.0, 3.0]
VP = [3.0, 4.0, 5.0, 6.0]
STAS = ["AAA", "BBB", "CCC", "DDD"]
TRUE_DEPTH = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
DELAY = {"AAA": 0.05, "BBB": -0.05, "CCC": 0.10, "DDD": 0.0}


def _block(v):
    return ModelBlock(pd.DataFrame({"vel": v, "depth": TOPS}))


def _fake_read_run_dir(path, **kw):
    """Parsed result computed from the run directory of the dummy VELEST."""
    d = Path(path)
    ctrl = read_control(d / "velest.cmn")
    model = read_model(d / "model.mod")
    free = ctrl.vthet < 1e4
    vp = model.p.layers["vel"].to_numpy(float)
    if free:  # a free inversion moves the velocities, a little less with lvz switched off
        vp = vp + (0.02 if ctrl.lowveloclay == 0 else 0.05) * np.arange(len(vp))
    sta = read_stations(d / ctrl.stationfile)
    use = sta[sta["icc"] > 0]
    ref = use.loc[use["icc"].idxmax(), "sta"]
    ptcor = [DELAY[s] - DELAY[ref] for s in sta["sta"]]  # delays relative to the reference
    corr = pd.DataFrame(
        {"sta": sta["sta"], "ptcor": ptcor, "stcor": np.array(ptcor) * 1.5, "nobs": 10}
    )
    ev = read_cnv(d / ctrl.phasefile).events
    perturbed = not np.allclose(ev["depth_km"], TRUE_DEPTH)
    dz = 0.0 if not perturbed else (0.3 if free else 0.05)
    cnv_ev = pd.DataFrame(
        {
            "event": ev["event"],
            "lat": -38.0 + 0.01 * ev["event"],
            "lon": 176.0 + 0.01 * ev["event"],
            "depth_km": TRUE_DEPTH + dz,
            "rms": 0.1,
        }
    )
    rs = pd.DataFrame({"nlay": np.arange(1, 5), "nhit": [100, 100, 100, 0]})
    return SimpleNamespace(
        converged=True, diverged=False, failed=False, finished=True, n_iterations=4,
        n_backups=0, initial_rms=0.3, final_rms=0.1 + (0.01 if ctrl.lowveloclay == 0 else 0.0),
        stop_reason="delmin", warnings=[],
        initial_model=None,
        final_model=VelestModel("f", _block(vp), _block(vp / 1.75)),
        ray_statistics=rs, station_corrections=corr, cnv=SimpleNamespace(events=cnv_ev),
        final_hypocentres=pd.DataFrame(),
    )  # fmt: skip


@pytest.fixture
def dummy(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "velest"
    exe.write_text(DUMMY)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("OSV_TOOLS_DIR", str(bindir))
    monkeypatch.setattr("osv.io.velest_out.read_run_dir", _fake_read_run_dir)


@pytest.fixture
def inputs(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    cnv, sta = d / "events.cnv", d / "stations.sta"
    n = len(TRUE_DEPTH)
    ev = pd.DataFrame(
        {
            "event": np.arange(n),
            "time": pd.date_range("2020-01-01", periods=n, freq="h"),
            "lat": -38.0 + 0.01 * np.arange(n),
            "lon": 176.0 + 0.01 * np.arange(n),
            "depth_km": TRUE_DEPTH,
            "mag": 0.0,
        }
    )
    picks = pd.DataFrame(
        [
            {"event": e, "sta": s, "phase": "P", "weight": 0, "tt": 1.0 + 0.1 * i}
            for e in range(n)
            for i, s in enumerate(STAS)
        ]
    )
    write_cnv(cnv, ev, picks)
    tab = pd.DataFrame(
        {
            "sta": STAS,
            "lat": [-38.0, -38.01, -38.02, -38.5],
            "lon": [176.0, 176.01, 176.02, 176.5],
            "elev_m": [10, 20, 30, 40],
            "icc": [1, 2, 3, 4],
        }
    )
    write_stations(sta, tab)
    ctrl = VelestControl(
        neqs=n, phasefile="events.cnv", stationfile="stations.sta", zmin=0.0, ittmax=3
    )
    return SimpleNamespace(cnv_path=cnv, sta_path=sta, control=ctrl, reference_station="DDD")


@pytest.fixture
def cfg():
    c = OSVConfig()
    c.velest.hypocentre_perturbation_km.n_tests = 4
    return c


MODEL = pd.DataFrame({"depth_top": TOPS, "vp": VP, "vs": np.array(VP) / 1.75})


# --- unit tests -----------------------------------------------------------------------------


def test_config_defaults():
    s = VelestConfig().stability
    assert s.fixed_vthet == 1e5 and s.n_reference_stations == 3 and s.lvz_test
    assert VelestConfig().hypocentre_perturbation_km.distribution == "uniform"


def test_as_frame_variants(tmp_path):
    ctrl = VelestControl(vpvs=1.7)
    df = st._as_frame(MODEL[["depth_top", "vp"]], ctrl)  # Vs filled from control.vpvs
    assert np.allclose(df["vs"], df["vp"] / 1.7)
    vm = VelestModel("m", _block(VP), _block(np.array(VP) / 1.8))
    assert np.allclose(st._as_frame(vm, ctrl)["vs"], np.array(VP) / 1.8)
    with pytest.raises(st.StabilityError):
        st._as_frame(pd.DataFrame({"x": [1]}), ctrl)
    p = st._write_model_file(MODEL, tmp_path / "m.mod")
    assert np.allclose(st._as_frame(p, ctrl)["vp"], VP)


def test_perturb_events_bounds_and_floor():
    from osv.config.schema import HypocentrePerturbation

    ev = pd.DataFrame(
        {"lat": [-38.0] * 200, "lon": [176.0] * 200, "depth_km": [0.3] * 100 + [5.0] * 100}
    )
    pert = HypocentrePerturbation(xy=1.0, z=2.0, n_tests=1)
    out = st._perturb_events(ev, pert, np.random.default_rng(1), 0.1)
    assert (out["depth_km"] >= 0.1 - 1e-12).all()
    assert out["pert_dx_km"].abs().max() <= 1.0 and out["pert_dy_km"].abs().max() <= 1.0
    dlat = (out["lat"] - ev["lat"]) * st.KM_PER_DEG
    assert np.allclose(dlat, out["pert_dy_km"])
    assert out["pert_dz_km"].abs().max() > 0.5
    g = st._perturb_events(
        ev, HypocentrePerturbation(xy=1.0, z=1.0, distribution="gaussian"),
        np.random.default_rng(2), -9.0,
    )  # fmt: skip
    assert g["pert_dx_km"].abs().max() > 1.0  # unbounded tails
    # a floor above the original depth does not move an event that was already shallower
    shallow = st._perturb_events(ev.iloc[:1], pert, np.random.default_rng(3), 1.0)
    assert shallow["depth_km"].iloc[0] == 0.3


def test_with_reference_and_auto_references(inputs):
    sta = read_stations(inputs.sta_path)
    assert st._reference_of(sta) == "DDD"
    new = st._with_reference(sta, "AAA")
    assert st._reference_of(new) == "AAA"
    assert sorted(new["icc"]) == [1, 2, 3, 4]
    with pytest.raises(st.StabilityError):
        st._with_reference(sta, "ZZZ")
    picks = read_cnv(inputs.cnv_path).picks
    auto = st._auto_references(sta, picks, "DDD", 3)
    assert "DDD" not in auto and len(auto) == 3 and len(set(auto)) == 3
    assert auto[0] in ("BBB", "CCC", "AAA")  # central ones first, far DDD excluded
    assert st._auto_references(sta, picks, "DDD", 0) == []


# --- orchestration with the dummy VELEST ----------------------------------------------------


def test_fixed_model_run(dummy, inputs, tmp_path):
    fx = st.fixed_model_run(MODEL, inputs, None, tmp_path / "fx")
    assert fx.ok and fx.max_velocity_change == pytest.approx(0.0)
    assert read_control(fx.run_dir / "velest.cmn").vthet == 1e5
    assert inputs.control.vthet == 1.0  # caller's control untouched
    assert list(fx.station_delays["sta"]) == STAS and len(fx.hypocentres) == 5
    assert np.isclose(fx.rms, 0.1)


def test_hypocentre_perturbation(dummy, inputs, cfg, tmp_path):
    res = st.hypocentre_perturbation_test(MODEL, inputs, None, cfg, tmp_path / "h", seed=3)
    r = res.runs
    assert len(r) == 1 + 2 * 4
    assert set(r["variant"]) == {"baseline_free", "fixed", "free"}
    assert (r["status"] == "ok").all()
    # dummy: fixed recovers to 0.05 km depth, free to 0.3 km; horizontal exact
    assert res.metric("hypocentre_perturbation", "fixed", "dz_mean_abs_km") == pytest.approx(0.05)
    assert res.metric("hypocentre_perturbation", "free", "dz_mean_abs_km") == pytest.approx(0.3)
    assert res.metric("hypocentre_perturbation", "free", "dh_mean_km") == pytest.approx(0, abs=1e-6)
    assert len(res.events) == 2 * 4 * 5
    assert {"pert_dx_km", "dh_km", "dz_km"} <= set(res.events.columns)
    # layers: spread is over the perturbed free runs only, hits come from the baseline
    lay = res.layers[res.layers["variant"] == "free"]
    assert len(lay) == 4 and (lay["vp_std"] == 0).all()
    assert lay["hits"].tolist() == [100, 100, 100, 0]
    assert lay["vp_max_abs_dev"].iloc[3] == pytest.approx(0.15)
    assert not res.warnings  # 0.3 km < 0.5 km
    # runs of the perturbed variants carry the applied perturbation
    assert (r[r["variant"] != "baseline_free"]["pert_dh_km"] > 0).all()


def test_hypocentre_perturbation_reproducible_and_warns(dummy, inputs, cfg, tmp_path):
    cfg.velest.stability.vertical_mislocation_warn_km = 0.2
    a = st.hypocentre_perturbation_test(MODEL, inputs, None, cfg, tmp_path / "a", seed=5, n_tests=2)
    b = st.hypocentre_perturbation_test(MODEL, inputs, None, cfg, tmp_path / "b", seed=5, n_tests=2)
    assert np.allclose(a.runs["pert_dh_km"].dropna(), b.runs["pert_dh_km"].dropna())
    assert [f.code for f in a.warnings] == ["hypocentre_vertical_mislocation"]  # free only
    assert a.warnings[0].data["variant"] == "free"


def test_velocity_spread_warning(dummy, inputs, cfg, tmp_path, monkeypatch):
    cfg.velest.stability.velocity_spread_warn_km_s = 0.001

    real = _fake_read_run_dir
    counter = {"n": 0}

    def noisy(path, **kw):
        out = real(path, **kw)
        counter["n"] += 1
        v = out.final_model.p.layers["vel"].to_numpy(float) + 0.01 * (counter["n"] % 3)
        out.final_model = VelestModel("f", _block(v), _block(v / 1.75))
        return out

    monkeypatch.setattr("osv.io.velest_out.read_run_dir", noisy)
    res = st.hypocentre_perturbation_test(MODEL, inputs, None, cfg, tmp_path / "s", n_tests=6)
    codes = {f.code for f in res.warnings}
    assert "velocity_unstable_hypocentres" in codes
    # layer 3 has no ray hits in the baseline and is never flagged
    flagged = {f.layer for f in res.warnings if f.code == "velocity_unstable_hypocentres"}
    assert 3 not in flagged and flagged


def test_hypocentre_perturbation_fails_without_reference(dummy, inputs, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "osv.io.velest_out.read_run_dir", lambda p, **k: (_ for _ in ()).throw(OSError("x"))
    )
    with pytest.raises(st.StabilityError, match="fixed-model run failed"):
        st.hypocentre_perturbation_test(MODEL, inputs, None, cfg, tmp_path / "f", n_tests=1)


def test_reference_station_test(dummy, inputs, cfg, tmp_path):
    res = st.reference_station_test(MODEL, inputs, None, cfg, tmp_path / "r")
    r = res.runs
    assert list(r["variant"]).count("swap") == 3 and list(r["variant"]).count("baseline") == 1
    assert "DDD" not in set(r[r["variant"] == "swap"]["reference_station"])
    # the swapped run's station file has the new reference as highest icc
    for row in r[r["variant"] == "swap"].itertuples():
        sta = read_stations(Path(row.run_dir) / "stations.sta")
        assert st._reference_of(sta) == row.reference_station
    # delays shift by a constant: residuals are zero, offset error is zero
    assert res.metric("reference_station", "all", "ptcor_resid_std_max") == pytest.approx(
        0, abs=1e-9
    )
    assert res.metric("reference_station", "all", "ptcor_offset_error_max") == pytest.approx(
        0, abs=1e-9
    )
    assert res.metric("reference_station", "all", "vp_dev_max") == pytest.approx(0, abs=1e-9)
    assert not res.warnings
    assert set(res.delays["reference_station"]) == set(
        r[r["variant"] == "swap"]["reference_station"]
    )


def test_reference_station_explicit_and_warning(dummy, inputs, cfg, tmp_path, monkeypatch):
    real = _fake_read_run_dir

    def shifted(path, **kw):
        out = real(path, **kw)
        if "ref_AAA" in str(path):
            v = out.final_model.p.layers["vel"].to_numpy(float) + [0, 0, 0.15, 0]
            out.final_model = VelestModel("f", _block(v), _block(v / 1.75))
        return out

    monkeypatch.setattr("osv.io.velest_out.read_run_dir", shifted)
    res = st.reference_station_test(
        MODEL, inputs, None, cfg, tmp_path / "r", reference_stations=["AAA", "DDD", "AAA"]
    )
    assert list(res.runs["reference_station"]) == ["DDD", "AAA"]
    w = res.warnings
    assert [f.code for f in w] == ["reference_swap_velocity_change"] and w[0].layer == 2
    assert res.metric("reference_station", "all", "vp_dev_max") == pytest.approx(0.15)


def test_lvz_test(dummy, inputs, cfg, tmp_path):
    extra = {"closest_X": MODEL.assign(vp=[3.0, 4.2, 4.1, 6.0])}
    res = st.lvz_test(MODEL, inputs, None, cfg, tmp_path / "l", extra_models=extra)
    r = res.runs
    assert len(r) == 4
    assert sorted(r["lowveloclay"].astype(int)) == [0, 0, 1, 1]
    for row in r.itertuples():
        assert read_control(Path(row.run_dir) / "velest.cmn").lowveloclay == row.lowveloclay
    assert res.metric("lvz", "final", "rms_gain") == pytest.approx(0.01)
    assert res.metric("lvz", "final", "vp_dev_max") == pytest.approx(0.09)  # 0.05*3 - 0.02*3
    assert res.metric("lvz", "closest_X", "n_lvz_layers") == 1  # decrease 4.2 -> 4.1(+0.1)
    assert set(res.models["variant"]) == {"lvz0", "lvz1"}
    assert not res.warnings


def test_run_stability_and_write(dummy, inputs, cfg, tmp_path):
    fm = pd.DataFrame({"layer": range(4), "depth_top": TOPS, "vp": VP, "vs": np.array(VP) / 1.75})
    long = pd.DataFrame(
        {
            "id": "MC_1", "phase": "P", "layer": range(4), "depth_top": TOPS,
            "v_initial": VP, "v_final": np.array(VP) + 0.1,
        }
    )  # fmt: skip
    ens = SimpleNamespace(
        final_model=fm, final_models=long, closest_run_id="MC_1", best_run_id="MC_1"
    )
    res = st.run_stability(ens, inputs, cfg, tmp_path / "s", seed=1, n_tests=2)
    assert res.fixed is not None and res.fixed.ok
    assert set(res.runs["test"]) == {"hypocentre_perturbation", "reference_station", "lvz"}
    assert {"final", "closest_MC_1"} <= set(res.summary[res.summary["test"] == "lvz"]["variant"])
    paths = st.write_stability(res, tmp_path / "out")
    for name in (
        "stability_runs.parquet", "stability_events.parquet", "stability_layers.parquet",
        "stability_summary.csv", "stability.json", "fixed_station_delays.csv",
    ):  # fmt: skip
        assert paths[name].is_file()
    back = pd.read_parquet(paths["stability_runs.parquet"])
    assert len(back) == len(res.runs)
    summ = pd.read_csv(paths["stability_summary.csv"])
    assert {"test", "variant", "metric", "value"} == set(summ.columns)
    assert all(isinstance(f, Finding) for f in res.findings)


# --- integration on the real data -----------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"

REAL_TOPS = [-0.6, 0.2, 0.7, 1.2, 1.7, 2.2, 3.0, 4.0]
REAL_VP = [3.41, 3.65, 3.91, 4.12, 4.71, 5.23, 5.63, 5.65]


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_stability(tmp_path):
    from osv.data.associate import associate_picks
    from osv.data.datum import apply_depth_datum
    from osv.data.quakeml import read_catalog
    from osv.data.select import select_data
    from osv.data.stations import read_stations as read_station_xml
    from osv.data.velest_inputs import write_velest_inputs
    from osv.tools import which_tool

    try:
        which_tool("velest")
    except ToolNotFoundError:
        pytest.skip("velest binary not available (set OSV_TOOLS_DIR)")

    cfg = OSVConfig()
    cfg.velest.hypocentre_perturbation_km.n_tests = 10
    cat = read_catalog(_CAT)
    stn = read_station_xml(_FILES)
    cat.events = apply_depth_datum(cat.events, cfg.coordinates.depth_datum)
    sel = select_data(associate_picks(cat, stn), cfg)
    vi = write_velest_inputs(sel, stn, cfg, tmp_path / "inputs")
    model = pd.DataFrame({"depth_top": REAL_TOPS, "vp": REAL_VP, "vs": np.array(REAL_VP) / 1.72})
    t0 = time.monotonic()

    fixed = st.fixed_model_run(model, vi, None, tmp_path / "fixed", cfg=cfg.velest.stability)
    assert fixed.ok and fixed.max_velocity_change < 0.011
    print(f"\nfixed run rms {fixed.rms:.4f}  max dv {fixed.max_velocity_change:.4f}")

    res = st.hypocentre_perturbation_test(
        model, vi, None, cfg, tmp_path / "hypo", seed=1, reference=fixed, n_jobs=8
    )
    res = res.merge(st.reference_station_test(model, vi, None, cfg, tmp_path / "ref", n_jobs=8))
    res = res.merge(st.lvz_test(model, vi, None, cfg, tmp_path / "lvz", n_jobs=8))
    print(f"stability wall time {time.monotonic() - t0:.0f}s")
    pd.set_option("display.width", 200)
    print(res.summary.to_string())
    print(res.layers.to_string())
    print(
        res.runs[res.runs["test"] != "hypocentre_perturbation"].drop(columns="run_dir").to_string()
    )
    for f in res.findings:
        print(f.severity, f.code, f.message)
    assert (res.runs["status"] == "ok").all()
    assert res.metric("hypocentre_perturbation", "fixed", "dh_mean_km") < 1.0
    paths = st.write_stability(res, tmp_path / "out")
    assert paths["stability.json"].is_file()


def test_stopped_early_finding(dummy, inputs, cfg, tmp_path, monkeypatch):
    def early(path, **kw):
        out = _fake_read_run_dir(path, **kw)
        out.stop_reason = "four_backups" if "ref_AAA" in str(path) else "delmin"
        return out

    monkeypatch.setattr("osv.io.velest_out.read_run_dir", early)
    res = st.reference_station_test(
        MODEL, inputs, None, cfg, tmp_path / "r", reference_stations=["AAA", "BBB"]
    )
    f = [x for x in res.warnings if x.code == "stability_runs_stopped_early"]
    assert len(f) == 1 and f[0].data["ids"] == ["ref_AAA"]
