"""Vp/Vs modes: nsp mapping, per-job ratios, ensemble Vs, grid search (velest.f:1118-1128)."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import VelestConfig, VelestSelect
from osv.io.velest import (
    ModelBlock,
    VelestControl,
    VelestModel,
    read_control,
    read_model,
    read_stations,
    write_stations,
)
from osv.tools import ToolNotFoundError
from osv.velest.ensemble import analyse_ensemble, write_ensemble
from osv.velest.models import generate_trial_models
from osv.velest.runner import jobs_from_models, prepare_run_dir
from osv.velest.vpvs import apply_vpvs_mode, model_vpvs, nsp_for_mode, vpvs_grid_search
from osv.velest.workflow import run_mc_workflow


def _cfg(mode: str, **kw) -> VelestConfig:
    return VelestConfig(
        vpvs={"mode": mode, "value": 1.72}, include_reference_model=False, n_models=4, **kw
    )


def test_nsp_mapping():
    assert nsp_for_mode("fixed") == 3
    assert nsp_for_mode("sampled") == 3
    assert nsp_for_mode("independent_s_model") == 2
    with pytest.raises(ValueError, match="unknown vpvs mode"):
        nsp_for_mode("nope")


@pytest.mark.parametrize(
    ("mode", "nsp", "nmod"), [("fixed", 3, 1), ("sampled", 3, 1), ("independent_s_model", 2, 2)]
)
def test_apply_vpvs_mode(mode, nsp, nmod):
    base = VelestControl()
    c = apply_vpvs_mode(base, mode, value=1.72)
    assert (c.nsp, c.nmod) == (nsp, nmod)
    assert c.vpvs == (1.72 if nsp == 3 else base.vpvs)
    assert base.nsp == 2  # input untouched


def test_sampled_jobs_get_their_own_ratio():
    models = generate_trial_models(_cfg("sampled"), 1)
    ctrl = apply_vpvs_mode(VelestControl(), "sampled", value=1.75)
    jobs = jobs_from_models(models, ctrl, "a.cnv", "a.sta")
    ratios = [j.control.vpvs for j in jobs]
    assert ratios == [model_vpvs(m) for m in models]
    assert len(set(ratios)) == len(ratios)
    assert all(1.65 <= r <= 1.85 for r in ratios)
    assert ctrl.vpvs == 1.75  # shared control not modified
    ctrl2 = apply_vpvs_mode(VelestControl(), "independent_s_model")
    jobs2 = jobs_from_models(models, ctrl2, "a.cnv", "a.sta")
    assert all(j.control.nsp == 2 for j in jobs2)


def test_run_dir_files_follow_nsp(tmp_path):
    (tmp_path / "e.cnv").write_text("x\n")
    sta = tmp_path / "s.sta"
    write_stations(
        sta,
        pd.DataFrame({"sta": ["AAA"], "lat": [-38.0], "lon": [176.0], "elev_m": [1], "icc": [1]}),
    )
    m = generate_trial_models(_cfg("sampled"), 3)[0]
    for mode, nmod in (("sampled", 1), ("independent_s_model", 2)):
        ctrl = apply_vpvs_mode(VelestControl(), mode)
        (job,) = jobs_from_models([m], ctrl, tmp_path / "e.cnv", sta)
        d = prepare_run_dir(job, tmp_path / mode)
        got = read_control(d / "velest.cmn")
        assert got.nmod == nmod
        mod = read_model(d / "model.mod", nmod=None)
        assert mod.nmod == nmod
        if nmod == 1:
            assert got.nsp == 3 and got.vpvs == pytest.approx(model_vpvs(m), abs=1e-9)
            np.testing.assert_allclose(mod.p.layers["vel"], np.round(m.vp, 2))


def _fake_parsed(nsp=3, vpvs=1.7, vp=(2.0, 3.0, 4.0, 5.0), rms=0.1):
    depths = np.array([0.0, 1.0, 2.0, 3.0])

    def blk(v):
        return ModelBlock(pd.DataFrame({"vel": v, "depth": depths}))

    p = np.asarray(vp, float)
    return SimpleNamespace(
        converged=True, diverged=False, failed=False, finished=True, n_iterations=5,
        n_backups=0, initial_rms=rms * 3, final_rms=rms, stop_reason="delmin",
        control={"nsp": nsp, "vpvs": vpvs},
        initial_model=VelestModel("i", blk(p * 0.9)),
        final_model=VelestModel("f", blk(p)),
        ray_statistics=pd.DataFrame({"nlay": [1, 2, 3, 4], "nhit": [5, 5, 5, 5]}),
        station_corrections=pd.DataFrame(
            {"sta": ["A"], "ptcor": [0.0], "stcor": [0.0], "nobs": [3]}
        ),
        cnv=SimpleNamespace(events=pd.DataFrame()),
        final_hypocentres=pd.DataFrame(),
    )  # fmt: skip


def _table(items):
    return pd.DataFrame(
        [
            {"id": k, "status": "ok", "wall_time": 1.0, "run_dir": "none", "parsed": p}
            for k, p in items.items()
        ]
    )


def test_ensemble_vs_follows_ratio(tmp_path):
    items = {
        "a": _fake_parsed(vpvs=1.70, vp=(2.0, 3.0, 4.0, 5.0), rms=0.10),
        "b": _fake_parsed(vpvs=1.80, vp=(2.2, 3.1, 4.2, 5.1), rms=0.11),
    }
    res = analyse_ensemble(
        _table(items), None, VelestSelect(method="best_n", value=1, statistic="median")
    )
    fm = res.final_model
    np.testing.assert_allclose(fm["vs"], np.array([2.0, 3.0, 4.0, 5.0]) / 1.70)
    np.testing.assert_allclose(fm["vpvs"], 1.70)
    both = analyse_ensemble(
        _table(items), None, VelestSelect(method="best_n", value=2, statistic="mean")
    )
    vp_a, vp_b = np.array([2.0, 3.0, 4.0, 5.0]), np.array([2.2, 3.1, 4.2, 5.1])
    np.testing.assert_allclose(both.final_model["vs"], np.mean([vp_a / 1.70, vp_b / 1.80], axis=0))
    s0 = res.final_models.query("id == 'a' and phase == 'S'")
    np.testing.assert_allclose(s0["v_initial"], vp_a * 0.9 / 1.70)
    write_ensemble(res, tmp_path)
    assert read_model(tmp_path / "final_model.mod").p.layers["vel"].iloc[0] == 2.0


def test_ensemble_without_ratio_has_no_vs():
    # no independent S model and no ratio: Vs stays undefined
    items = {"a": _fake_parsed(nsp=1), "b": _fake_parsed(nsp=1, rms=0.2)}
    res = analyse_ensemble(
        _table(items), None, VelestSelect(method="best_n", value=1, statistic="median")
    )
    assert res.final_model["vs"].isna().all()


# --- workflow and grid search with a dummy VELEST ---------------------------------------------

DUMMY = """#!/bin/sh
[ -f velest.cmn ] || { echo "STOP control file velest.cmn not found!"; exit 0; }
echo "RMS RESIDUAL= 0.12" > velest.OUT
cp events.cnv velout.cnv
cp stations.sta velout.sta
exit 0
"""


def _fake_read_run_dir(path, **kwargs):
    d = Path(path)
    ctrl = read_control(d / "velest.cmn")
    model = read_model(d / "model.mod", nmod=ctrl.nmod)
    sta = read_stations(d / "stations.sta")
    return SimpleNamespace(
        final_model=model, initial_rms=1.0, final_rms=1.0 + (ctrl.vpvs - 1.72) ** 2,
        station_corrections=pd.DataFrame({"sta": sta["sta"], "ptcor": 0.0, "stcor": 0.0}),
        final_hypocentres=pd.DataFrame(), n_iterations=3, converged=True, diverged=False,
        failed=False, control={"nsp": ctrl.nsp, "vpvs": ctrl.vpvs},
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
    d = tmp_path / "shared"
    d.mkdir()
    (d / "events.cnv").write_text("fake\n")
    write_stations(
        d / "stations.sta",
        pd.DataFrame({"sta": ["AAA"], "lat": [-38.0], "lon": [176.0], "elev_m": [1], "icc": [1]}),
    )
    ctrl = apply_vpvs_mode(
        VelestControl(neqs=1, phasefile="events.cnv", stationfile="stations.sta"),
        "sampled",
        value=1.75,
    )
    return SimpleNamespace(cnv_path=d / "events.cnv", sta_path=d / "stations.sta", control=ctrl)


def test_workflow_keeps_each_models_ratio_through_stages(dummy, tmp_path):
    cfg = _cfg("sampled")
    models = generate_trial_models(cfg, 5)
    res = run_mc_workflow(models, dummy, cfg, tmp_path / "wf", n_jobs=1)
    assert len(res.history) == len(models) * len(res.stages)
    for m in models:
        for k in (1, 2):
            ctrl = read_control(tmp_path / "wf" / f"stage{k}" / m.id / "velest.cmn")
            assert (ctrl.nsp, ctrl.nmod) == (3, 1)
            assert ctrl.vpvs == pytest.approx(model_vpvs(m), abs=1e-9)
        # stage 2 starts from a P-only model file
        assert read_model(tmp_path / "wf" / "stage2" / "_inputs" / f"{m.id}.mod").s is None
        assert res.final_models[m.id].vpvs[0] == pytest.approx(model_vpvs(m), abs=1e-9)


def test_vpvs_grid_search(dummy, tmp_path):
    m = generate_trial_models(_cfg("fixed"), 2)[0]
    dummy.control = apply_vpvs_mode(dummy.control, "fixed", value=1.75)
    df = vpvs_grid_search(m, dummy, tmp_path / "grid", n_jobs=1)
    assert list(df["vpvs"]) == [1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.8]
    assert df.attrs["best_vpvs"] == pytest.approx(1.72)
    assert df.attrs["best_vpvs_parabola"] == pytest.approx(1.72, abs=1e-6)
    ctrl = read_control(tmp_path / "grid" / "vpvs_1.680" / "velest.cmn")
    assert (ctrl.nsp, ctrl.vpvs) == (3, 1.68)


# --- integration -----------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_fixed_vpvs_workflow(tmp_path):
    from osv.config.schema import OSVConfig
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
    cfg.velest.vpvs.mode = "fixed"
    cfg.velest.vpvs.value = 1.72
    cfg.velest.n_models = 6
    cfg.velest.include_reference_model = False
    cat = read_catalog(_CAT)
    st = read_station_xml(_FILES)
    cat.events = apply_depth_datum(cat.events, cfg.coordinates.depth_datum)
    sel = select_data(associate_picks(cat, st), cfg)
    vi = write_velest_inputs(sel, st, cfg, tmp_path / "inputs")
    assert (vi.control.nsp, vi.control.nmod, vi.control.vpvs) == (3, 1, 1.72)
    models = generate_trial_models(cfg.velest, cfg.project.random_seed)
    res = run_mc_workflow(models, vi, cfg.velest, tmp_path / "wf", n_jobs=6, seed=1)
    assert len(res.ok) == len(models)
    out = res.ok.iloc[0].parsed
    # S picks are used with nsp = 3 (velest.f:1892 drops them only for nsp = 1)
    from osv.io.velest import read_cnv

    picks = read_cnv(vi.cnv_path).picks
    n_p = int((picks["phase"] == "P").sum())
    assert (picks["phase"] == "S").sum() > 0
    assert int(out.iterations["n_all"].iloc[0]) > n_p
    assert out.final_model.s is None
    ens = analyse_ensemble(res.ok, models, cfg.velest.select, vp_bounds=cfg.velest.vp_bounds)
    np.testing.assert_allclose(ens.final_model["vpvs"], 1.72, rtol=1e-6)
