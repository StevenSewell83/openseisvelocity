"""End-to-end regression baseline on the rk-nm-2017 test dataset (slow; opt in).

Runs the whole pipeline with a reduced configuration and compares key metrics with
``tests/data/e2e_baseline.json``.  Enable with ``OSV_RUN_E2E=1`` (needs the FMTOMO/VELEST
binaries, e.g. ``OSV_TOOLS_DIR=build/bin``, and the data referenced by
``projects/rk-nm-2017/config.yaml``).  ``OSV_UPDATE_E2E_BASELINE=1`` rewrites the baseline from the
run instead of asserting.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("OSV_RUN_E2E") != "1", reason="set OSV_RUN_E2E=1 to run"),
]

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "data" / "e2e_baseline.json"
PROJECT_CONFIG = ROOT / "projects" / "rk-nm-2017" / "config.yaml"
WORKDIR = ROOT / "build" / "e2e_baseline"

# tolerances written into a newly generated baseline
VP_TOL_KM_S = 0.35
RMS_REL_TOL = 0.5


def _reduced_config():
    from osv.config import OSVConfig
    from osv.config.loader import load_config

    base = load_config(PROJECT_CONFIG).model_dump(mode="python")
    base["project"].update(workdir=str(WORKDIR / "runs"), n_jobs=min(8, os.cpu_count() or 1))
    base["velest"].update(n_models=60)
    base["velest"]["stability"] = {"enabled": False}
    base["velest"]["vpvs_grid"] = {"enabled": False}
    base["model3d"]["grid"].update(dx_km=1.0, dy_km=1.0, dz_km=1.0)
    base["model3d"]["propgrid_refine"] = 2
    base["fmtomo"]["iterations"] = 2
    base["fmtomo"]["tradeoff"].update(
        damping=[1.0, 10.0, 100.0], smoothing=[10.0], iterations=1, max_concurrent=1
    )
    base["resolution"]["checkerboard"].update(cell_km=[3.0], offset_half_cell=False)
    base["resolution"]["spikes"] = []
    return OSVConfig.model_validate(base)


def _metrics(runs: Path) -> dict:
    prep = json.loads((runs / "01_prep" / "manifest.json").read_text())["summary"]
    fm = pd.read_csv(runs / "03_velest_analyse" / "final_model.csv")
    an = json.loads((runs / "07_analyse" / "summary.json").read_text())
    val = json.loads((runs / "09_validation" / "stats.json").read_text())
    return {
        "events_selected": prep["events_selected"],
        "picks_P_selected": prep["picks_P_selected"],
        "picks_S_selected": prep["picks_S_selected"],
        "vpvs": prep["vpvs"],
        "layer_vp_km_s": [round(float(v), 3) for v in fm["vp"]],
        "final_rms_ms_P": an["final_rms_ms_P"],
        "variance_reduction_pct_P": an["variance_reduction_pct_P"],
        "validation_median_3d_rms_s": val["stats"]["rms"]["median_3d"],
    }


def test_e2e_baseline():
    from osv.tools import ToolNotFoundError, which_tool

    try:
        for t in ("fm3d", "invert3d", "velest"):
            which_tool(t)
    except ToolNotFoundError as exc:
        pytest.skip(f"binaries not available: {exc}")
    from osv.stages import discover, run_pipeline

    discover()

    shutil.rmtree(WORKDIR, ignore_errors=True)
    cfg = _reduced_config()
    ran = run_pipeline(cfg, force=True)
    assert set(ran.values()) == {"ran"}, ran
    got = _metrics(WORKDIR / "runs")

    if os.environ.get("OSV_UPDATE_E2E_BASELINE") == "1":
        base = {
            "note": "regenerate with OSV_RUN_E2E=1 OSV_UPDATE_E2E_BASELINE=1 pytest "
            "tests/test_e2e_baseline.py",
            "expected": got,
            "tolerance": {
                "layer_vp_km_s": VP_TOL_KM_S,
                "rms_relative": RMS_REL_TOL,
                "min_variance_reduction_pct": max(5.0, 0.5 * got["variance_reduction_pct_P"]),
            },
        }
        BASELINE.write_text(json.dumps(base, indent=2) + "\n")
        return

    base = json.loads(BASELINE.read_text())
    exp, tol = base["expected"], base["tolerance"]
    for k in ("events_selected", "picks_P_selected", "picks_S_selected"):
        assert got[k] == exp[k], k
    assert got["vpvs"] == pytest.approx(exp["vpvs"], abs=0.01)
    assert len(got["layer_vp_km_s"]) == len(exp["layer_vp_km_s"])
    for i, (g, e) in enumerate(zip(got["layer_vp_km_s"], exp["layer_vp_km_s"], strict=True)):
        assert abs(g - e) <= tol["layer_vp_km_s"], f"layer {i}: {g} vs {e}"
    for k in ("final_rms_ms_P", "validation_median_3d_rms_s"):
        assert got[k] == pytest.approx(exp[k], rel=tol["rms_relative"]), k
    assert got["variance_reduction_pct_P"] >= tol["min_variance_reduction_pct"]
