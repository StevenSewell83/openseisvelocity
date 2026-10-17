"""End-to-end test of Stage 3 (build_model): prep -> velest_mc -> velest_analyse -> build_model."""

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


def _have(tool: str) -> bool:
    try:
        which_tool(tool)
    except ToolNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (CATALOG.exists() and STATION_FILES), reason="test dataset missing"),
    pytest.mark.skipif(not (_have("velest") and _have("fm3d")), reason="binaries not built"),
]


def _config(tmp_path: Path) -> Path:
    cfg = {
        "project": {"name": "t", "workdir": str(tmp_path / "runs"), "n_jobs": 4},
        "inputs": {"quakeml": [str(CATALOG)], "stationxml": STATION_FILES},
        "coordinates": {"origin": {"lat": -38.55, "lon": 176.194}},
        "plotting": {"format": ["png"], "dpi": 50, "depth_slices_km": [0, 2, 4]},
        "fmtomo": {"phases": ["P", "S"]},
        "velest": {
            "n_models": 12,
            "ittmax": 6,
            "vpvs": {"mode": "fixed", "value": "wadati"},
            "hypocentre_perturbation_km": {"n_tests": 2},
            "stability": {"n_reference_stations": 1},
            "iterate_until": {"max_cycles": 1},
            "vpvs_grid": {"enabled": False},
        },
        "model3d": {
            "grid": {"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 2.0, "zmax_km": 6.0},
            "propgrid_refine": 2,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_stage3_end_to_end(tmp_path, restore_stage_registry):
    from typer.testing import CliRunner

    from osv.cli import app
    from osv.config import load_config
    from osv.data.stage import PrepStage
    from osv.model3d.stage import BuildModelStage
    from osv.velest.stage import VelestAnalyseStage, VelestMCStage

    for cls in (PrepStage, VelestMCStage, VelestAnalyseStage, BuildModelStage):
        stages.register(cls)
    runner = CliRunner()
    path = _config(tmp_path)
    for cmd in ("prep", "velest-mc", "velest-analyse", "build-model"):
        res = runner.invoke(app, [cmd, "-c", str(path)])
        assert res.exit_code == 0, res.output

    d = tmp_path / "runs" / "04_build_model"
    fm = d / "fmtomo"
    for f in (
        "vgrids.in", "interfaces.in", "propgrid.in", "sources.in", "receivers.in",
        "otimes_P.dat", "otimes_P_mapping.csv",
    ):  # fmt: skip
        assert (fm / f).exists(), f
    for f in ("starting_model.nc", "grid.json", "forward_check.json", "report.html"):
        assert (d / f).exists(), f
    m = json.loads((d / "manifest.json").read_text())
    assert m["status"] == "complete" and m["checks"] and m["tools"]
    assert m["summary"]["rms_P_starting_s"] < 1.0
    fc = json.loads((d / "forward_check.json").read_text())
    assert fc["phases"]["P"]["start"]["n"] == m["summary"]["observations_P"]
    if (fm / "otimes_S.dat").exists():  # S picks present
        assert (fm / "vgrids_S.in").exists() and "S" in fc["phases"]
    mp = pd.read_csv(fm / "otimes_P_mapping.csv")
    assert set(mp["phase"]) == {"P"}

    figs = {e["name"] for e in json.loads((d / "figures" / "figures.json").read_text())["figures"]}
    assert {"vp_slices", "section_ew", "grid_geometry", "residual_histograms"} <= figs
    assert "vp_slices" in (d / "report.html").read_text()

    for p in (d / "figures").glob("*.png"):
        p.unlink()
    cfg = load_config(path)
    stages.plot_only(cfg, "build_model")
    assert (d / "figures" / "vp_slices.png").exists()

    out = stages.run_pipeline(cfg, to="build_model")
    assert set(out.values()) == {"skipped"}
