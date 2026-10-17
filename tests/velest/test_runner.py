from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import VelestConfig
from osv.io.velest import VelestControl, read_control, read_stations, write_stations
from osv.tools import ToolNotFoundError
from osv.velest.models import generate_trial_models
from osv.velest.runner import (
    CONTROL_NAME,
    MODEL_NAME,
    OUT_NAME,
    RunnerError,
    VelestJob,
    jobs_from_models,
    prepare_run_dir,
    run_jobs,
)

DUMMY = """#!/bin/sh
# fake velest: behaviour chosen from the run-directory name
d=$(basename "$PWD")
[ -f velest.cmn ] || { echo "STOP INPUTPARAM>>> control file velest.cmn not found!"; exit 0; }
case "$d" in
  hang*) sleep 30 ;;
  fail*) echo "STOP something broke"; exit 1 ;;
  crash*) exit 3 ;;
  noout*) exit 0 ;;
esac
echo "RMS RESIDUAL= 0.12" > velest.OUT
echo cnv > velout.cnv
echo sta > velout.sta
echo big > velout.vel
echo big > velout.smp
exit 0
"""


@pytest.fixture
def dummy_tool(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "velest"
    exe.write_text(DUMMY)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("OSV_TOOLS_DIR", str(bindir))
    return exe


@pytest.fixture
def inputs(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    cnv, sta = d / "events.cnv", d / "stations.sta"
    cnv.write_text("fake cnv\n")
    st = pd.DataFrame(
        {"sta": ["AAA", "BBB"], "lat": [-38.0, -38.1], "lon": [176.0, 176.1], "elev_m": [10, 20]}
    )
    write_stations(sta, st)
    return cnv, sta


@pytest.fixture
def models():
    return generate_trial_models(VelestConfig(n_models=5, include_reference_model=False), 1)


def _jobs(models, inputs, ids=None):
    ctrl = VelestControl(neqs=3, phasefile="events.cnv", stationfile="stations.sta")
    jobs = jobs_from_models(models, ctrl, *inputs)
    for j, i in zip(jobs, ids or [], strict=False):
        j.id = i
    return jobs


def test_prepare_run_dir(tmp_path, models, inputs):
    ctrl = VelestControl(neqs=3, outfile="weird.out")
    job = VelestJob("t1", models[0], ctrl, *inputs)
    d = prepare_run_dir(job, tmp_path / "runs")
    assert d == tmp_path / "runs" / "t1"
    cmn = read_control(d / CONTROL_NAME)
    assert cmn.modelfile == MODEL_NAME and cmn.outfile == OUT_NAME
    assert cmn.phasefile == "events.cnv" and cmn.stationfile == "stations.sta"
    assert (d / MODEL_NAME).is_file()
    assert (d / "events.cnv").is_symlink()
    assert (d / "events.cnv").resolve() == inputs[0].resolve()
    assert ctrl.outfile == "weird.out" and ctrl.modelfile == ""  # caller's control untouched


def test_prepare_with_model_path_and_corrections(tmp_path, models, inputs):
    from osv.velest.models import write_trial_models

    mp = write_trial_models(models[:1], tmp_path, parquet=False)[0]
    job = VelestJob("c1", mp, VelestControl(), *inputs, station_corrections={"BBB": 0.25})
    d = prepare_run_dir(job, tmp_path / "runs")
    st = read_stations(d / "stations.sta").set_index("sta")
    assert not (d / "stations.sta").is_symlink()
    assert st.loc["BBB", "ptcor"] == pytest.approx(0.25) and st.loc["AAA", "ptcor"] == 0
    assert read_stations(inputs[1]).set_index("sta").loc["BBB", "ptcor"] == 0
    bad = VelestJob("c2", mp, VelestControl(), *inputs, station_corrections={"ZZZ": 1.0})
    with pytest.raises(RunnerError, match="ZZZ"):
        prepare_run_dir(bad, tmp_path / "runs")


def test_bad_job_id(models, inputs):
    with pytest.raises(RunnerError):
        VelestJob("../x", models[0], VelestControl(), *inputs)


def test_parallel_ok(tmp_path, dummy_tool, models, inputs):
    df = run_jobs(_jobs(models, inputs), tmp_path / "r", n_jobs=4, timeout=20)
    assert list(df["status"]) == ["ok"] * 5
    assert list(df["id"]) == [m.id for m in models]
    for r in df.itertuples():
        assert Path(r.out_path).is_file() and Path(r.cnv_path).is_file()
        assert json.loads((Path(r.run_dir) / "status.json").read_text())["status"] == "ok"
        assert (Path(r.run_dir) / "velout.vel").exists()  # not compacted


def test_failure_isolation_and_timeout(tmp_path, dummy_tool, models, inputs):
    ids = ["good1", "fail1", "hang1", "crash1", "noout1"]
    t0 = time.monotonic()
    df = run_jobs(_jobs(models, inputs, ids), tmp_path / "r", n_jobs=5, timeout=1.5)
    assert time.monotonic() - t0 < 15
    st = dict(zip(df["id"], df["status"], strict=True))
    assert st == {
        "good1": "ok",
        "fail1": "failed",
        "hang1": "timeout",
        "crash1": "failed",
        "noout1": "failed",
    }
    row = df.set_index("id")
    assert any("STOP" in e for e in row.loc["fail1", "detected_errors"])
    assert row.loc["crash1", "returncode"] == 3
    assert "no velest.OUT" in row.loc["noout1", "error"]


def test_per_job_timeout(tmp_path, dummy_tool, models, inputs):
    jobs = _jobs(models[:1], inputs, ["hang1"])
    jobs[0].timeout = 1.0
    df = run_jobs(jobs, tmp_path / "r", timeout=None)
    assert df.loc[0, "status"] == "timeout"


def test_resume_and_force(tmp_path, dummy_tool, models, inputs):
    root = tmp_path / "r"
    ids = ["good1", "good2", "fail1"]
    df1 = run_jobs(_jobs(models[:3], inputs, ids), root, n_jobs=2)
    assert not df1["skipped"].any()
    marker = root / "good1" / OUT_NAME
    marker.write_text("sentinel\n")
    df2 = run_jobs(_jobs(models[:3], inputs, ids), root, n_jobs=2)
    assert dict(zip(df2["id"], df2["skipped"], strict=True)) == {
        "good1": True,
        "good2": True,
        "fail1": False,  # failures are retried
    }
    assert marker.read_text() == "sentinel\n"
    df3 = run_jobs(_jobs(models[:3], inputs, ids), root, force=True)
    assert not df3["skipped"].any()
    assert "RMS" in marker.read_text()


def test_compact(tmp_path, dummy_tool, models, inputs):
    df = run_jobs(_jobs(models[:1], inputs), tmp_path / "r", compact=True)
    d = Path(df.loc[0, "run_dir"])
    names = {p.name for p in d.iterdir()}
    assert "velout.vel" not in names and "velout.smp" not in names
    assert "events.cnv" not in names
    assert {"velest.OUT", "velout.cnv", "velout.sta", "velest.cmn", "status.json"} <= names
    assert Path(df.loc[0, "out_path"]).is_file()


def test_missing_executable(tmp_path, monkeypatch, models, inputs):
    monkeypatch.setenv("OSV_TOOLS_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr("osv.tools.REPO_BUILD_BIN", tmp_path / "nope2")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    with pytest.raises(ToolNotFoundError):
        run_jobs(_jobs(models[:1], inputs), tmp_path / "r")


def test_duplicate_ids(tmp_path, dummy_tool, models, inputs):
    jobs = _jobs(models[:2], inputs, ["a", "a"])
    with pytest.raises(RunnerError, match="unique"):
        run_jobs(jobs, tmp_path / "r")


def test_missing_input_recorded_as_failure(tmp_path, dummy_tool, models, inputs):
    jobs = _jobs(models[:2], inputs)
    jobs[0].cnv_path = tmp_path / "missing.cnv"
    df = run_jobs(jobs, tmp_path / "r")
    assert list(df["status"]) == ["failed", "ok"]
    assert "missing.cnv" in df.loc[0, "error"]


def test_parsed_results_optional(tmp_path, dummy_tool, models, inputs, monkeypatch):
    import sys
    import types

    mod = types.ModuleType("osv.io.velest_out")

    class R:
        final_rms = 0.12

    mod.read_run_dir = lambda d: R()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "osv.io.velest_out", mod)
    df = run_jobs(_jobs(models[:2], inputs), tmp_path / "r")
    assert np.allclose(df["final_rms"], 0.12)


# --- integration on the real data -----------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_16_models_parallel(tmp_path):
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
    cfg.velest.n_models = 15  # + reference model = 16
    cat = read_catalog(_CAT)
    st = read_station_xml(_FILES)
    cat.events = apply_depth_datum(cat.events, cfg.coordinates.depth_datum)
    sel = select_data(associate_picks(cat, st), cfg)
    vi = write_velest_inputs(sel, st, cfg, tmp_path / "inputs")
    models = generate_trial_models(cfg.velest, cfg.project.random_seed)
    assert len(models) == 16
    vi.control.ittmax = cfg.velest.ittmax
    jobs = jobs_from_models(models, vi.control, vi.cnv_path, vi.sta_path)
    t0 = time.monotonic()
    df = run_jobs(jobs, tmp_path / "runs", n_jobs=8, timeout=1800)
    wall = time.monotonic() - t0
    print(f"\n16 VELEST runs on 8 workers: {wall:.1f}s, sum of job times {df.wall_time.sum():.1f}s")
    import re

    df["rms_last"] = [
        float(re.findall(r"RMS RESIDUAL=\s*([0-9.]+)", Path(p).read_text(errors="replace"))[-1])
        if p
        else np.nan
        for p in df["out_path"]
    ]
    print(df[["id", "status", "returncode", "wall_time", "rms_last"]].to_string())
    assert (df["status"] == "ok").all(), df[["id", "status", "error"]]
