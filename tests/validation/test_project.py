"""Integration: relocate the real project's validation events (skips without the outputs).

The Stage 3 starting model stands in for the 3-D model (Stage 4 is not wired yet).  It is the
1-D VELEST model extruded, so the 3-D relocation should be about as good as the 1-D one; this
checks the whole machinery on real picks and stations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.tools import ToolNotFoundError, which_tool

PROJECT = Path(__file__).resolve().parents[2] / "projects" / "rk-nm-2017"
RUNS = PROJECT / "runs"
NEEDED = [
    PROJECT / "config.yaml",
    RUNS / "01_prep" / "events.parquet",
    RUNS / "01_prep" / "picks.parquet",
    RUNS / "03_velest_analyse" / "final_model.mod",
    RUNS / "03_velest_analyse" / "station_delays.csv",
    RUNS / "04_build_model" / "fmtomo" / "vgrids.in",
    RUNS / "04_build_model" / "fmtomo" / "interfaces.in",
    RUNS / "04_build_model" / "fmtomo" / "propgrid.in",
]


def _have() -> bool:
    try:
        for b in ("fm3d", "frechgen", "invert3d", "residuals", "scorrect", "velest"):
            which_tool(b)
    except ToolNotFoundError:
        return False
    return all(p.exists() for p in NEEDED)


@pytest.mark.integration
@pytest.mark.skipif(not _have(), reason="binaries or the rk-nm-2017 project outputs missing")
def test_validation_events_start_model_vs_1d(tmp_path) -> None:
    from osv.config.loader import load_config
    from osv.validation.plots import make_figures
    from osv.validation.relocate import relocate_validation

    cfg = load_config(PROJECT / "config.yaml")
    ev = pd.read_parquet(RUNS / "01_prep" / "events.parquet")
    pk = pd.read_parquet(RUNS / "01_prep" / "picks.parquet")
    sd = pd.read_csv(RUNS / "03_velest_analyse" / "station_delays.csv")
    n_val = int((ev["split"] == "validation").sum())
    assert n_val > 0
    res = relocate_validation(
        ev,
        pk,
        RUNS / "03_velest_analyse" / "final_model.mod",
        sd,
        RUNS / "04_build_model" / "fmtomo",
        None,
        cfg,
        tmp_path,
        n_jobs=4,
    )
    e = res.events
    assert len(e) == n_val
    assert np.isfinite(e[["rms_1d", "rms_3d"]].to_numpy()).all()
    r = res.stats["rms"]
    h = res.stats["hypocentre_difference"]
    print(
        f"median RMS 1-D {1000 * r['median_1d']:.0f} ms, 3-D {1000 * r['median_3d']:.0f} ms, "
        f"p={r['p_two_sided']:.3g}; median dh {h['dh_median_km']:.2f} km"
    )
    # same 1-D-derived model: comparable fit (factor 2), hypocentres within a few km
    assert r["median_3d"] < 2.0 * r["median_1d"]
    assert res.stats["hypocentre_difference"]["dh_median_km"] < 3.0
    names = make_figures(res, tmp_path / "figures")
    assert len(names) == 4
