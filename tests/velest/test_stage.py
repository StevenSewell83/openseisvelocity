"""End-to-end test of Stage 2 (velest_mc, velest_analyse) on the repository test dataset."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from osv import stages
from osv.tools import ToolNotFoundError, which_tool

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "manual_only_catalog.xml"
STATION_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
STATION_FILES = sorted(glob.glob(str(STATION_DIR / "*_STATIONXML.xml")))


def _have_velest() -> bool:
    try:
        which_tool("velest")
    except ToolNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (CATALOG.exists() and STATION_FILES), reason="test dataset missing"),
    pytest.mark.skipif(not _have_velest(), reason="velest binary not available"),
]


def _config(tmp_path: Path) -> Path:
    cfg = {
        "project": {"name": "t", "workdir": str(tmp_path / "runs"), "n_jobs": 4},
        "inputs": {"quakeml": [str(CATALOG)], "stationxml": STATION_FILES},
        "coordinates": {"origin": {"lat": -38.55, "lon": 176.194}},
        "plotting": {"format": ["png"], "dpi": 50, "profiles": []},
        "velest": {
            "n_models": 24,
            "ittmax": 9,
            "vpvs": {"mode": "fixed", "value": "wadati"},
            "hypocentre_perturbation_km": {"n_tests": 4},
            "stability": {"n_reference_stations": 1},
            "iterate_until": {"max_cycles": 2},
            "vpvs_grid": {"enabled": True, "values": [1.7, 1.75, 1.8]},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_stage2_end_to_end(tmp_path):
    from typer.testing import CliRunner

    from osv.cli import app
    from osv.config import load_config
    from osv.data.stage import PrepStage
    from osv.velest.stage import VelestAnalyseStage, VelestMCStage

    for cls in (PrepStage, VelestMCStage, VelestAnalyseStage):  # other tests may unregister
        stages.register(cls)
    runner = CliRunner()
    path = _config(tmp_path)
    for cmd in ("prep", "velest-mc", "velest-analyse"):
        res = runner.invoke(app, [cmd, "-c", str(path)])
        assert res.exit_code == 0, res.output

    runs = tmp_path / "runs"
    mc = runs / "02_velest_mc"
    for f in ("trial_models.parquet", "runs.parquet", "workflow_history.csv", "report.html"):
        assert (mc / f).exists(), f
    m = json.loads((mc / "manifest.json").read_text())
    assert m["status"] == "complete" and m["tools"]
    assert m["summary"]["trial_models"] == 25  # 24 + reference model
    assert m["summary"]["runs_ok"] > 0
    table = pd.read_parquet(mc / "runs.parquet")
    assert "parsed" not in table.columns and not table["run_dir"].str.startswith("/").any()

    an = runs / "03_velest_analyse"
    for f in (
        "final_model.mod",
        "final_model.csv",
        "station_delays.csv",
        "final_hypocentres.parquet",
        "summary.json",
        "vpvs_grid.csv",
        "report.html",
    ):
        assert (an / f).exists(), f
    fm = pd.read_csv(an / "final_model.csv")
    assert {"depth_top_km", "vp", "vs", "vpvs", "vp_p05", "vp_p95"} <= set(fm.columns)
    assert len(fm) == 8 and fm["vp"].is_monotonic_increasing
    sd = pd.read_csv(an / "station_delays.csv")
    assert list(sd.columns[:7]) == ["sta", "alias", "lat", "lon", "elevation", "ptcor", "stcor"]
    assert sd["lat"].notna().all()
    hyp = pd.read_parquet(an / "final_hypocentres.parquet")
    events = pd.read_parquet(runs / "01_prep" / "events.parquet")
    assert hyp["event_id"].isin(events["event_id"]).all() and len(hyp) > 10
    s = json.loads((an / "summary.json").read_text())
    assert s["n_selected"] >= 1 and s["final_model_rms"] > 0

    ma = json.loads((an / "manifest.json").read_text())
    assert ma["status"] == "complete" and ma["checks"]
    assert 1.5 < ma["summary"]["vpvs"] < 2.0
    figs = {e["name"] for e in json.loads((an / "figures" / "figures.json").read_text())["figures"]}
    assert {"velocity_spaghetti", "layer_histograms", "station_delay_map", "vpvs"} <= figs
    html = (an / "report.html").read_text()
    assert "velocity_spaghetti" in html

    # re-plot from the written outputs only
    for p in (an / "figures").glob("*.png"):
        p.unlink()
    cfg = load_config(path)
    stages.plot_only(cfg, "velest_analyse")
    assert (an / "figures" / "velocity_spaghetti.png").exists()
    assert (an / "figures" / "residuals.png").exists()

    # unchanged: both skipped; the pipeline reaches velest_analyse
    out = stages.run_pipeline(cfg, to="velest_analyse")
    assert set(out.values()) == {"skipped"}
