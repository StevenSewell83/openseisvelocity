"""End-to-end test of the Stage 1 (prep) stage on the repository test dataset."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from osv import stages
from osv.config import OSVConfig
from osv.data.stage import PrepStage

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "manual_only_catalog.xml"
STATION_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
STATION_FILES = sorted(glob.glob(str(STATION_DIR / "*_STATIONXML.xml")))

pytestmark = pytest.mark.integration

OUTPUTS = (
    "events.parquet",
    "picks.parquet",
    "all_picks.parquet",
    "stations.parquet",
    "channels.parquet",
    "aliases.csv",
    "rejections.csv",
    "split.csv",
    "wadati.json",
    "report.html",
)
FIGURES = {
    "map_before_after",
    "section_ew",
    "section_ns",
    "section_profile_a_a",
    "histograms",
    "event_timeline",
    "picks_per_station",
    "travel_times",
    "wadati",
    "ray_map",
    "ray_section_ew",
    "ray_section_ns",
    "pick_uncertainty",
    "station_residuals",
    "station_residual_map",
    "rejections",
}


@pytest.mark.skipif(not (CATALOG.exists() and STATION_FILES), reason="test dataset not available")
def test_prep_end_to_end(tmp_path):
    cfg = OSVConfig.model_validate(
        {
            "project": {"name": "t", "workdir": str(tmp_path), "n_jobs": 1},
            "inputs": {"quakeml": [str(CATALOG)], "stationxml": STATION_FILES},
            "coordinates": {"origin": {"lat": -38.55, "lon": 176.194}},
            "plotting": {
                "format": ["png"],
                "dpi": 60,
                "profiles": [{"name": "A-A'", "start": [-38.62, 176.19], "end": [-38.46, 176.21]}],
            },
        }
    )
    assert stages.get_stage("prep") is PrepStage
    assert stages.run_pipeline(cfg, only="prep", project_dir=tmp_path) == {"prep": "ran"}
    d = tmp_path / "01_prep"
    for f in OUTPUTS:
        assert (d / f).exists(), f
    m = json.loads((d / "manifest.json").read_text())
    assert m["status"] == "complete"
    codes = {c["code"]: c["severity"] for c in m["checks"]}
    assert codes["duplicate_arrivals"] == "warn"
    assert codes["invalid_magnitudes"] == "warn"
    s = m["summary"]
    assert s["events_in"] == 79
    assert 1.5 < s["vpvs"] < 2.0
    assert len(pd.read_parquet(d / "events.parquet")) == s["events_selected"]
    figs = json.loads((d / "figures" / "figures.json").read_text())["figures"]
    assert FIGURES <= {e["name"] for e in figs}
    html = (d / "report.html").read_text()
    assert "duplicate_arrivals" in html and "invalid_magnitudes" in html
    assert "map_before_after" in html

    for p in (d / "figures").glob("*.png"):
        p.unlink()
    stages.plot_only(cfg, "prep", project_dir=tmp_path)
    assert (d / "figures" / "wadati.png").exists()


def _cfg_yaml(tmp_path: Path) -> Path:
    import yaml

    cfg = {
        "project": {"name": "t", "workdir": str(tmp_path / "runs"), "n_jobs": 1},
        "inputs": {"quakeml": [str(CATALOG)], "stationxml": STATION_FILES},
        "coordinates": {"origin": {"lat": -38.55, "lon": 176.194}},
        "plotting": {"format": ["png"], "dpi": 50, "profiles": []},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.skipif(not (CATALOG.exists() and STATION_FILES), reason="test dataset not available")
def test_prep_cli_velest_inputs_and_rerun(tmp_path):
    from typer.testing import CliRunner

    from osv.cli import app
    from osv.data.stage import load_velest_control

    runner = CliRunner()
    cfg_path = _cfg_yaml(tmp_path)
    res = runner.invoke(app, ["prep", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "prep: ran" in res.output
    d = tmp_path / "runs" / "01_prep"
    for f in OUTPUTS:
        assert (d / f).exists(), f
    for f in ("events.cnv", "stations.sta", "velest_inputs_summary.json", "event_map.csv"):
        assert (d / "velest" / f).exists(), f
    assert (d / "velest_validation" / "events.cnv").exists()
    ctrl = load_velest_control(d)
    assert ctrl.olat == pytest.approx(-38.55) and ctrl.olon == pytest.approx(-176.194)  # west +

    m = json.loads((d / "manifest.json").read_text())
    assert m["status"] == "complete"
    assert m["summary"]["velest_events"] > 0
    assert m["summary"]["velest_reference_station"]
    html = (d / "report.html").read_text()
    assert "duplicate_arrivals" in html and "invalid_magnitudes" in html

    res = runner.invoke(app, ["prep", "-c", str(cfg_path)])
    assert res.exit_code == 0 and "prep: skipped" in res.output, res.output
    res = runner.invoke(app, ["prep", "-c", str(cfg_path), "--force"])
    assert res.exit_code == 0 and "prep: ran" in res.output, res.output
    res = runner.invoke(app, ["run", "--to", "prep", "-c", str(cfg_path)])
    assert res.exit_code == 0 and "prep: skipped" in res.output, res.output
    res = runner.invoke(app, ["plot", "prep", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output

    # VELEST accepts the written inputs
    from osv.io.velest import ModelBlock, VelestModel, write_control, write_model
    from osv.tools import ToolNotFoundError, run_tool, which_tool

    try:
        which_tool("velest")
    except ToolNotFoundError:
        pytest.skip("velest binary not available (set OSV_TOOLS_DIR)")
    run = tmp_path / "velest_run"
    run.mkdir()
    for f in ("events.cnv", "stations.sta"):
        (run / f).write_bytes((d / "velest" / f).read_bytes())
    top = pd.DataFrame({"depth": [-0.6, 0.5, 1.5, 3.0], "vel": [2.5, 3.2, 4.2, 5.3]})
    sdf = top.assign(vel=(top["vel"] / 1.75).round(2))
    write_model(run / "model.mod", VelestModel("simple", ModelBlock(top), ModelBlock(sdf)))
    ctrl.modelfile = "model.mod"
    ctrl.phasefile = "events.cnv"
    ctrl.stationfile = "stations.sta"
    write_control(run / "velest.cmn", ctrl)
    out = run_tool("velest", run, timeout=600)
    assert out.ok
    assert "RMS RESIDUAL" in (run / "velest.out").read_text(errors="replace")
