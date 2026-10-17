from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import DampingStage, OSVConfig, VelestConfig
from osv.io.velest import (
    VelestControl,
    read_control,
    read_model,
    read_stations,
    write_stations,
)
from osv.tools import ToolNotFoundError
from osv.velest.models import TrialModel, generate_trial_models
from osv.velest.workflow import (
    VpVsResolutionError,
    _stage_control,
    iterate_final_model,
    perturb_model,
    resolve_stages,
    resolve_vdamp,
    resolve_vpvs,
    run_mc_workflow,
    select_seeds,
)

DUMMY = """#!/bin/sh
d=$(basename "$PWD")
[ -f velest.cmn ] || { echo "STOP control file velest.cmn not found!"; exit 0; }
case "$d" in
  fail*) echo "STOP something broke"; exit 1 ;;
esac
echo "RMS RESIDUAL= 0.12" > velest.OUT
cat events.cnv > velout.cnv
echo x >> velout.cnv
cp stations.sta velout.sta
exit 0
"""


def _fake_read_run_dir(path, **kwargs):
    """Stand-in parser: model +0.4/4**n km/s (n = earlier cycles), ptcor +0.1."""
    d = Path(path)
    n = (d / "events.cnv").read_text().count("x\n")
    delta = 0.4 / 4**n
    ctrl = read_control(d / "velest.cmn")
    model = read_model(d / "model.mod", nmod=ctrl.nmod)
    for blk in (model.p, model.s):
        if blk is not None:
            blk.layers["vel"] = blk.layers["vel"] + delta
    sta = read_stations(d / "stations.sta")
    base = sta["ptcor"].to_numpy() if ctrl.iusestacorr else np.zeros(len(sta))
    corr = pd.DataFrame({"sta": sta["sta"], "ptcor": base + 0.1, "stcor": base * 2 + 0.2})
    z = sum(0.4 / 4**j for j in range(n + 1))
    hypo = pd.DataFrame({"eq": [1, 2], "x": [0.0, 1.0], "y": [0.0, 0.0], "z": [z, z]})
    return SimpleNamespace(
        final_model=model,
        final_rms=1.0 / float(model.p.layers["vel"].iloc[0]),
        initial_rms=1.0,
        station_corrections=corr,
        final_hypocentres=hypo,
        n_iterations=3,
        converged=True,
        diverged=False,
        failed=False,
    )


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
    cnv.write_text("fake cnv\n")
    st = pd.DataFrame(
        {
            "sta": ["AAA", "BBB"],
            "lat": [-38.0, -38.1],
            "lon": [176.0, 176.1],
            "elev_m": [10, 20],
            "icc": [1, 2],
        }
    )
    write_stations(sta, st)
    ctrl = VelestControl(neqs=2, phasefile="events.cnv", stationfile="stations.sta")
    return SimpleNamespace(cnv_path=cnv, sta_path=sta, control=ctrl)


def _cfg(**kw) -> VelestConfig:
    return VelestConfig(include_reference_model=False, **kw)


def test_resolve_stages_inherits_user_damping():
    cfg = _cfg(ittmax=7, invertratio=2)
    cfg.damping.vthet = 3.0
    s1, s2 = resolve_stages(cfg)
    assert (s1.vthet, s1.stathet, s1.ittmax, s1.invertratio) == (10.0, 1.0, 7, 2)
    assert (s2.vthet, s2.stathet, s2.othet) == (3.0, cfg.damping.stathet, cfg.damping.othet)
    assert [s.name for s in resolve_stages(_cfg(stages=[]))] == ["single"]
    with pytest.raises(ValueError, match="unique"):
        _cfg(stages=[DampingStage(name="a"), DampingStage(name="a")])
    assert OSVConfig().velest.stages[0].vthet == 10.0


def test_stage_chaining_and_handoff(dummy, inputs, tmp_path):
    models = generate_trial_models(_cfg(n_models=3), 1)
    cfg = _cfg(n_models=3)
    cfg.damping.vthet = 2.0
    res = run_mc_workflow(models, inputs, cfg, tmp_path / "wf", n_jobs=2)
    assert len(res.runs) == 3 and (res.runs["status"] == "ok").all()
    assert list(res.history["stage_index"]) == [1, 1, 1, 2, 2, 2]
    m = models[0]
    s1, s2 = tmp_path / "wf/stage1" / m.id, tmp_path / "wf/stage2" / m.id
    # damping per stage
    assert read_control(s1 / "velest.cmn").vthet == 10.0
    assert read_control(s2 / "velest.cmn").vthet == 2.0
    # stage 2 model = stage 1 output (fake adds 0.4 km/s in stage 1)
    start1 = read_model(s1 / "model.mod", nmod=2).p.layers["vel"].to_numpy()
    start2 = read_model(s2 / "model.mod", nmod=2).p.layers["vel"].to_numpy()
    assert np.allclose(start2, start1 + 0.4, atol=0.011)
    # hypocentres: stage 2 input cnv is stage 1 velout.cnv
    assert (s2 / "events.cnv").read_text() == (s1 / "velout.cnv").read_text()
    assert (s2 / "events.cnv").read_text().count("x\n") == 1
    # station corrections: stage 1 ignores, stage 2 gets stage-1 delays and iusestacorr=1
    assert read_control(s1 / "velest.cmn").iusestacorr == 0
    c2 = read_control(s2 / "velest.cmn")
    assert c2.iusestacorr == 1
    sta2 = read_stations(s2 / "stations.sta")
    assert np.allclose(sta2["ptcor"], 0.1) and np.allclose(sta2["stcor"], 0.2)
    assert list(sta2["icc"]) == [1, 2]
    assert np.allclose(res.runs.iloc[0]["parsed"].station_corrections["ptcor"], 0.2)
    # final models and run table
    assert set(res.final_models) == {x.id for x in models}
    assert {"id", "status", "run_dir", "final_rms", "parsed", "round", "stage"} <= set(res.runs)
    assert all(
        r.endswith(f"stage2/{i}") for r, i in zip(res.runs["run_dir"], res.runs["id"], strict=True)
    )
    assert (tmp_path / "wf/workflow_history.csv").is_file()


def test_handoff_flags_off(dummy, inputs, tmp_path):
    models = generate_trial_models(_cfg(n_models=1), 1)
    cfg = _cfg(
        n_models=1,
        stages=[
            DampingStage(name="a"),
            DampingStage(
                name="b", use_previous_hypocentres=False, use_previous_station_corrections=False
            ),
        ],
    )
    run_mc_workflow(models, inputs, cfg, tmp_path / "wf", n_jobs=1)
    s2 = tmp_path / "wf/stage2" / models[0].id
    assert (s2 / "events.cnv").read_text() == "fake cnv\n"
    assert read_control(s2 / "velest.cmn").iusestacorr == 0
    # but the model is still chained
    assert read_model(s2 / "model.mod", nmod=2).p.layers["vel"].iloc[0] > models[0].vp[0] + 0.3


def test_failed_model_is_dropped(dummy, inputs, tmp_path):
    models = generate_trial_models(_cfg(n_models=2), 1)
    bad = TrialModel("fail_1", models[0].depths, models[0].vp, models[0].vs, "x", 0)
    res = run_mc_workflow([bad, models[1]], inputs, _cfg(), tmp_path / "wf", n_jobs=1)
    row = res.runs.set_index("id")
    assert (
        row.loc["fail_1", "status"] == "failed"
        and row.loc["fail_1", "stage"] == "stage1_high_damping"
    )
    assert (
        row.loc[models[1].id, "status"] == "ok" and row.loc[models[1].id, "stage"] == "stage2_final"
    )
    assert not (tmp_path / "wf/stage2/fail_1").exists()
    assert "fail_1" not in res.final_models


def test_select_seeds():
    def mk(i, rms, ok=True, div=False):
        p = SimpleNamespace(final_model=object(), diverged=div, failed=False)
        return {"id": i, "status": "ok" if ok else "failed", "parsed": p, "final_rms": rms}

    df = pd.DataFrame(
        [mk("a", 0.5), mk("b", 0.1), mk("c", 0.2), mk("d", 0.05, ok=False), mk("e", 0.01, div=True),
         mk("f", float("nan")), mk("g", 0.3)]
    )  # fmt: skip
    # eligible: a b c g -> 50 % = 2 best
    assert select_seeds(df, 50) == ["b", "c"]
    assert select_seeds(df, 1) == ["b"]
    assert select_seeds(df.iloc[[3]], 50) == []


def test_perturb_model_monotonic_and_vpvs():
    models = generate_trial_models(_cfg(n_models=1), 1)
    from osv.velest.workflow import _trial_to_velest

    vm = _trial_to_velest(models[0])
    out = perturb_model(vm, 0.3, np.random.default_rng(0))
    vp = out.p.layers["vel"].to_numpy()
    assert (np.diff(vp) >= 0).all() and not np.allclose(vp, vm.p.layers["vel"])
    r0 = vm.s.layers["vel"] / vm.p.layers["vel"]
    r1 = out.s.layers["vel"] / out.p.layers["vel"]
    assert np.allclose(r0, r1, atol=0.02)


def test_reseed_round(dummy, inputs, tmp_path):
    cfg = _cfg(n_models=6)
    cfg.reseed.enabled = True
    cfg.reseed.best_percent = 34  # ceil(0.34*6) = 3 seeds
    cfg.reseed.n_copies = 2
    cfg.reseed.perturb_vp_km_s = 0.05
    models = generate_trial_models(cfg, 1)
    res = run_mc_workflow(models, inputs, cfg, tmp_path / "wf", n_jobs=2, seed=5)
    r1 = res.runs[res.runs["round"] == 1]
    r2 = res.runs[res.runs["round"] == 2]
    assert len(r1) == 6 and len(r2) == 6
    best = set(r1.sort_values("final_rms")["id"][:3])  # fake: lowest rms == largest first Vp
    assert set(r2["parent"]) == best
    assert all(i.startswith(p) and "_r2_" in i for i, p in zip(r2["id"], r2["parent"], strict=True))
    assert (tmp_path / "wf/round2/stage1" / r2["id"].iloc[0]).is_dir()
    assert (tmp_path / "wf/round2/stage2" / r2["id"].iloc[0]).is_dir()
    # seed model = parent's final model (+/- perturbation); hypocentres inherited from parent
    pid = r2["parent"].iloc[0]
    parent_final = r1.set_index("id").loc[pid, "parsed"].final_model.p.layers["vel"].to_numpy()
    first = tmp_path / "wf/round2/stage1" / r2["id"].iloc[0]
    seed_vel = read_model(first / "model.mod", nmod=2).p.layers["vel"].to_numpy()
    assert 0 < np.abs(seed_vel - parent_final).max() < 0.5
    assert (first / "events.cnv").read_text().count("x\n") == 2  # parent's two stages
    assert read_control(first / "velest.cmn").iusestacorr == 1
    assert set(res.final_models) >= set(r2["id"])
    # unperturbed seeds -> single copy
    cfg.reseed.perturb_vp_km_s = 0.0
    res0 = run_mc_workflow(models, inputs, cfg, tmp_path / "wf0", n_jobs=2)
    assert (res0.runs["round"] == 2).sum() == 3


def test_reseed_not_inheriting(dummy, inputs, tmp_path):
    cfg = _cfg(n_models=3)
    cfg.reseed.enabled = True
    cfg.reseed.best_percent = 34
    cfg.reseed.n_copies = 1
    cfg.reseed.inherit_hypocentres = False
    cfg.reseed.inherit_station_corrections = False
    res = run_mc_workflow(generate_trial_models(cfg, 1), inputs, cfg, tmp_path / "wf", n_jobs=1)
    r2 = res.runs[res.runs["round"] == 2].iloc[0]
    first = tmp_path / "wf/round2/stage1" / r2["id"]
    assert (first / "events.cnv").read_text() == "fake cnv\n"
    assert read_control(first / "velest.cmn").iusestacorr == 0


def test_iterate_terminates_on_convergence(dummy, inputs, tmp_path):
    cfg = _cfg()  # defaults: dV < 0.02 and mean dH < 0.1
    model = generate_trial_models(cfg, 1)[0]
    out = iterate_final_model(model, inputs, cfg, tmp_path / "it")
    # fake changes: 0.4, 0.1, 0.025, 0.00625 km/s -> converged at cycle 4
    assert len(out) == 4 and out.converged
    h = out.history
    assert list(h["cycle"]) == [1, 2, 3, 4]
    assert np.allclose(h["model_change_km_s"], [0.4, 0.1, 0.025, 0.00625], atol=0.011)
    assert np.isnan(h["hypo_change_km"].iloc[0]) and h["hypo_change_km"].iloc[-1] < 0.1
    # each cycle starts from the previous outputs
    c3 = tmp_path / "it/iterate/cycle03"
    assert (c3 / "events.cnv").read_text().count("x\n") == 2
    assert read_control(c3 / "velest.cmn").iusestacorr == 1
    assert (tmp_path / "it/iterate/convergence.csv").is_file()


def _nsinv(root: Path) -> list[int]:
    return [read_control(d / "velest.cmn").nsinv for d in sorted((root / "iterate").glob("cycle*"))]


def test_iterate_freezes_delays_for_nsp3_only(dummy, inputs, tmp_path):
    """nsp=3: S delay = ptcor*vpvs without S partials, so handed-on delays are not re-inverted."""
    cfg = _cfg()
    cfg.iterate_until.max_cycles = 2
    model = generate_trial_models(cfg, 1)[0]
    iterate_final_model(model, inputs, cfg, tmp_path / "nsp2")
    assert _nsinv(tmp_path / "nsp2") == [1, 1]
    inputs.control.nsp, inputs.control.nmod = 3, 1
    inputs.control.vpvs = 1.75
    iterate_final_model(model, inputs, cfg, tmp_path / "nsp3")
    assert _nsinv(tmp_path / "nsp3") == [1, 0]  # cycle 1 has no delays yet
    iterate_final_model(model, inputs, cfg, tmp_path / "nsp3c", corrections=_corr())
    assert _nsinv(tmp_path / "nsp3c") == [0, 0]
    iterate_final_model(
        model, inputs, cfg, tmp_path / "nsp3i", corrections=_corr(), freeze_station_delays=False
    )
    assert _nsinv(tmp_path / "nsp3i") == [1, 1]


def _corr() -> pd.DataFrame:
    return pd.DataFrame({"sta": ["AAA", "BBB"], "ptcor": [0.1, 0.0], "stcor": [0.0, 0.0]})


def test_stage_invert_station_delays_option():
    base = VelestControl(neqs=2, phasefile="a", stationfile="b")
    assert base.nsinv == 1
    st = resolve_stages(_cfg(stages=[DampingStage(name="s")]))[0]
    assert _stage_control(base, st, False).nsinv == 1
    assert (
        _stage_control(base, st.model_copy(update={"invert_station_delays": False}), False).nsinv
        == 0
    )


def test_resolve_vpvs(tmp_path):
    import json

    cfg = _cfg()
    assert resolve_vpvs(cfg, tmp_path) == 1.75  # numeric: unchanged, no file needed
    cfg.vpvs.value = "wadati"
    with pytest.raises(VpVsResolutionError, match="does not exist"):
        resolve_vpvs(cfg, tmp_path)
    (tmp_path / "wadati.json").write_text(json.dumps({}))
    with pytest.raises(VpVsResolutionError, match="unusable"):
        resolve_vpvs(cfg, tmp_path)
    (tmp_path / "wadati.json").write_text(json.dumps({"vpvs": 0.9}))
    with pytest.raises(VpVsResolutionError, match="unusable"):
        resolve_vpvs(cfg, tmp_path)
    (tmp_path / "wadati.json").write_text("not json")
    with pytest.raises(VpVsResolutionError, match="cannot read"):
        resolve_vpvs(cfg, tmp_path)
    (tmp_path / "wadati.json").write_text(json.dumps({"vpvs": 1.7312, "n_pairs": 10}))
    assert resolve_vpvs(cfg, tmp_path) == pytest.approx(1.7312)
    assert resolve_vpvs(OSVConfig(velest=cfg), tmp_path) == pytest.approx(1.7312)  # full config


def test_wadati_value_schema_and_unresolved_guard():
    cfg = OSVConfig.model_validate({"velest": {"vpvs": {"value": "wadati"}}})
    assert cfg.velest.vpvs.value == "wadati"
    with pytest.raises(ValueError, match="resolve"):
        generate_trial_models(cfg.velest, 1)
    with pytest.raises(ValueError):
        OSVConfig.model_validate({"velest": {"vpvs": {"value": "other"}}})


def test_iterate_stops_at_max_cycles(dummy, inputs, tmp_path):
    cfg = _cfg()
    cfg.iterate_until.max_cycles = 2
    out = iterate_final_model(generate_trial_models(cfg, 1)[0], inputs, cfg, tmp_path / "it")
    assert len(out) == 2 and not out.converged
    cfg.iterate_until.max_cycles = 0
    assert (
        len(iterate_final_model(generate_trial_models(cfg, 1)[0], inputs, cfg, tmp_path / "i0"))
        == 0
    )


def test_iterate_failure_stops(dummy, inputs, tmp_path):
    cfg = _cfg()
    m = generate_trial_models(cfg, 1)[0]
    bad = TrialModel("x", m.depths, m.vp, m.vs, "x", 0)
    # a cycle directory called cycleNN never fails, so make the whole run fail via the tool
    import osv.velest.workflow as wf

    orig = wf.run_jobs

    def failing(jobs, *a, **k):
        for j in jobs:
            j.id = "fail" + j.id
        return orig(jobs, *a, **k)

    wf.run_jobs = failing
    try:
        out = iterate_final_model(bad, inputs, cfg, tmp_path / "it")
    finally:
        wf.run_jobs = orig
    assert len(out) == 0 and not out.converged and len(out.history) == 1


# --- integration on the real data -----------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_two_stage_and_iterate(tmp_path):
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
    cfg.velest.n_models = 11  # + reference = 12
    cfg.velest.reseed.enabled = True
    cfg.velest.reseed.best_percent = 25
    cfg.velest.reseed.n_copies = 2
    cat = read_catalog(_CAT)
    st = read_station_xml(_FILES)
    cat.events = apply_depth_datum(cat.events, cfg.coordinates.depth_datum)
    sel = select_data(associate_picks(cat, st), cfg)
    vi = write_velest_inputs(sel, st, cfg, tmp_path / "inputs")
    models = generate_trial_models(cfg.velest, cfg.project.random_seed)
    assert len(models) == 12
    t0 = time.monotonic()
    res = run_mc_workflow(models, vi, cfg, tmp_path / "wf", n_jobs=8, timeout=1800)
    print(f"\nworkflow wall time {time.monotonic() - t0:.0f}s")
    cols = ["round", "stage_index", "id", "status", "initial_rms", "final_rms", "n_iterations"]
    print(res.history[cols].to_string())
    print(res.history.groupby(["round", "stage_index"])["final_rms"].describe().to_string())
    # round 1 (stages 1-2 of the 12 Latin-hypercube models) must all work; reseeded perturbed
    # models may fail individually (VELEST can produce a NaN RMS for one event, see osv-iu9.9)
    h = res.history
    assert (h[h["round"] == 1]["status"] == "ok").all()
    r2 = res.runs[res.runs["round"] == 2]
    assert len(r2) == 6
    assert (r2["status"] == "ok").sum() >= 4, r2[["id", "status", "error"]]
    bad = res.runs[res.runs["status"] != "ok"]
    assert bad["error"].map(lambda e: isinstance(e, str) and len(e) > 0).all()
    assert set(bad["id"]) == set(h[h["status"] != "ok"]["id"])
    ok = res.runs[res.runs["status"] == "ok"]
    best = ok.sort_values("final_rms").iloc[0]
    out = iterate_final_model(
        res.final_models[best["id"]],
        vi,
        cfg,
        tmp_path / "wf" / "best",
        cnv_path=best["cnv_path"],
        corrections=best["parsed"].station_corrections,
    )
    print(out.history.to_string())
    # delays are handed on frozen (nsp=3): every cycle runs, the RMS stays at the stage-2 level
    assert len(out) == len(out.history) >= 2
    rms = out.history["rms"]
    assert rms.notna().all() and rms.max() <= best["final_rms"] * 1.05
    assert out.history["model_change_km_s"].iloc[-1] < out.history["model_change_km_s"].iloc[0]
    vp = out[-1].final_model.p.layers["vel"]
    assert vp.min() > 1.0 and vp.iloc[-1] > 5.0  # no collapse towards ~3 km/s


# --- vdamp ----------------------------------------------------------------------------------


def test_resolve_vdamp_number_list_auto():
    cfg = _cfg(layers_top_km=[-0.6, 0.2, 0.7, 1.2], vp_bounds=[(1, 2)] * 4)
    assert list(resolve_vdamp(cfg)) == [1.0] * 4
    assert list(resolve_vdamp(_cfg(layers_top_km=[0, 1], vp_bounds=[(1, 2)] * 2, vdamp=5))) == [
        5.0, 5.0,
    ]  # fmt: skip
    lst = _cfg(layers_top_km=[0, 1], vp_bounds=[(1, 2)] * 2, vdamp=[20, 1])
    assert list(resolve_vdamp(lst)) == [20.0, 1.0]
    auto = _cfg(layers_top_km=[-0.6, 0.2, 0.7, 1.2], vp_bounds=[(1, 2)] * 4, vdamp="auto")
    assert list(resolve_vdamp(auto)) == [1.0] * 4  # no geometry: nothing to damp
    # station depths -0.59..+0.16 km: only the layer starting at -0.6 is above the deepest one
    # (0.2 > 0.16); deepest event at 0.9 km: the layer starting at 1.2 is below it
    v = resolve_vdamp(auto, station_z_km=[-0.59, 0.16], event_depth_km=[0.5, 0.9])
    assert list(v) == [10.0, 1.0, 1.0, 3.0]
    auto.vdamp_auto.near_surface, auto.vdamp_auto.below_events = 50.0, 1.0
    assert list(resolve_vdamp(auto, station_z_km=[0.3], event_depth_km=[9.0])) == [50.0, 50.0, 1, 1]


def test_stage_swtfac_overrides_control():
    base = VelestControl(neqs=2, phasefile="a", stationfile="b", nsp=3, swtfac=0.75)
    st = resolve_stages(_cfg(stages=[DampingStage(name="s"), DampingStage(name="p", swtfac=0.01)]))
    assert _stage_control(base, st[0], False).swtfac == 0.75
    assert _stage_control(base, st[1], False).swtfac == 0.01


def _vdamp_of(path: Path) -> list[float]:
    return list(read_model(path, nmod=2).p.layers["vdamp"])


def test_vdamp_reaches_every_stage_and_iterate(dummy, inputs, tmp_path):
    cfg = _cfg(n_models=2)
    models = generate_trial_models(cfg, 1)
    vd = [30.0] + [1.0] * (models[0].n_layers - 1)
    run_mc_workflow(models, inputs, cfg, tmp_path / "wf", n_jobs=1, vdamp=vd)
    for stage in ("stage1", "stage2"):
        assert _vdamp_of(tmp_path / "wf" / stage / models[0].id / "model.mod") == vd
    cfg.iterate_until.max_cycles = 2
    iterate_final_model(models[0], inputs, cfg, tmp_path / "it", vdamp=vd)
    assert _vdamp_of(tmp_path / "it/iterate/cycle01/model.mod") == vd
    assert _vdamp_of(tmp_path / "it/iterate/cycle02/model.mod") == vd
    # default: vdamp 1 everywhere
    run_mc_workflow(models, inputs, cfg, tmp_path / "wf1", n_jobs=1)
    assert set(_vdamp_of(tmp_path / "wf1/stage2" / models[0].id / "model.mod")) == {1.0}
