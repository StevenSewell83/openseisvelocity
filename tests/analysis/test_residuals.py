"""Residual/relocation tables and figures on a synthetic run directory (no binaries)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.analysis import plots_residuals as pr
from osv.analysis.residuals import (
    available_iterations,
    relocation_table,
    residual_stats_by,
    residual_table,
    residual_tables,
    variance_reduction,
)
from osv.io.fmtomo import (
    PathSpec,
    Receivers,
    Source,
    write_arrivals,
    write_observed,
    write_receivers,
    write_sources,
)
from osv.io.fmtomo.times import write_stimes
from osv.plotting.registry import FigureRegistry

KM = 111.195
STATIONS = [("STA1", -38.55, 176.15), ("STA2", -38.60, 176.25), ("STA3", -38.65, 176.18)]
EVENTS = [
    (1.5, -38.58, 176.20),
    (2.5, -38.60, 176.17),
    (3.5, -38.62, 176.22),
    (2.0, -38.57, 176.19),
]
SCALE = {0: 1.0, 1: 0.5, 2: 0.2}  # residual amplitude per iteration
N_ITER = 2


def _build(root: Path, nonlinear: bool = False) -> pd.DataFrame:
    """Write a 2-iteration run directory; returns the mapping."""
    rng = np.random.default_rng(1)
    root.mkdir(parents=True)
    paths = [PathSpec.direct(1), PathSpec.direct(2)]  # P, S
    ph = ["P", "S"]
    rows = []
    rays = []
    for r, (name, _, _) in enumerate(STATIONS, start=1):
        k = 0
        for s in range(1, len(EVENTS) + 1):
            for p in (1, 2):
                k += 1
                rays.append((r, s, p))
                rows.append(
                    {
                        "source_idx": s, "receiver_idx": r, "path_idx": p, "ray_idx": k,
                        "event_id": f"ev{s}", "station": name, "phase": ph[p - 1],
                        "pick_id": f"{name}-{s}-{ph[p - 1]}", "valid": True,
                    }
                )  # fmt: skip
    m = pd.DataFrame(rows)
    m["time"] = 2.0 + rng.uniform(0, 4, len(m)) * (1 + (m["phase"] == "S"))
    m["uncertainty"] = 0.05
    rec = Receivers(
        pd.DataFrame(
            {
                "depth_km": [-0.3] * 3,
                "lat": [s[1] for s in STATIONS],
                "lon": [s[2] for s in STATIONS],
            },
            index=pd.RangeIndex(1, 4, name="receiver"),
        ),
        pd.DataFrame(rays, columns=["receiver", "source", "path"], dtype="int64"),
    )
    srcs0 = [Source(d, la, lo, list(paths)) for d, la, lo in EVENTS]
    write_sources(root / "sourcesref.in", srcs0)
    write_receivers(root / "receivers.in", rec)
    obs = m.rename(columns={"receiver_idx": "receiver", "source_idx": "source", "ray_idx": "ray"})
    obs = obs[["receiver", "source", "ray", "time", "uncertainty"]].assign(subray=0)
    write_observed(root / "otimes.dat", obs)
    noise = rng.normal(0, 0.2, len(m))
    bias = np.where(m["station"] == "STA2", 0.3, 0.0)  # a station with a delay
    for it in range(N_ITER + 1):
        d = root / "iterations" / f"{it:02d}"
        d.mkdir(parents=True)
        calc = m["time"].to_numpy() - SCALE[it] * (noise + bias)
        arr = obs[["receiver", "source", "ray"]].assign(
            subray=0, time=calc, diffracted=False, headwave=False
        )
        write_arrivals(d / "mtimes.dat", arr)
        shift = 0.05 * it
        srcs = [
            Source(dd + shift, la + 0.01 * it, lo - 0.01 * it, list(paths)) for dd, la, lo in EVENTS
        ]
        write_sources(d / "sources.in", srcs)
        write_stimes(d / "stimes.dat", [0.01 * it * (i + 1) for i in range(len(EVENTS))])
        var = float(np.mean((SCALE[it] * (noise + bias)) ** 2))
        (d / "summary.json").write_text(
            json.dumps({"iter": it, "rms_ms": 1000 * var**0.5, "variance_s2": var, "chi2": 1.0})
        )
    write_arrivals(root / "rtimes.dat", arr.assign(time=m["time"] - noise - bias))
    write_arrivals(root / "mtimes.dat", arr)
    (root / "sources.in").write_text((root / "iterations/02/sources.in").read_text())
    return m


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    return root, _build(root)


def test_residual_table_values(run) -> None:
    root, m = run
    assert available_iterations(root) == [0, 1, 2]
    t0 = residual_table(root, m, "initial")
    t2 = residual_table(root, m, "final")
    assert len(t0) == len(m) == len(t2)
    assert (t0["iteration"] == 0).all() and (t2["iteration"] == 2).all()
    # residuals scale as constructed: final = 0.2 x initial
    np.testing.assert_allclose(t2["residual"], 0.2 * t0["residual"], atol=1e-5)
    np.testing.assert_allclose(t0["t_obs"] - t0["t_calc"], t0["residual"])
    # geometry: first row is STA1 / ev1
    r = t0.iloc[0]
    assert r["station"] == "STA1" and r["ev_depth_km"] == pytest.approx(1.5)
    d_epi = np.hypot((176.20 - 176.15) * KM * np.cos(np.deg2rad(-38.55)), (-38.58 + 38.55) * KM)
    assert r["dist_km"] == pytest.approx(np.hypot(d_epi, 1.8), rel=0.01)
    # station is north-west of the event
    assert 270 < r["azimuth_deg"] < 360
    assert (t2["ev_lat"] != t0["ev_lat"]).all()  # final uses the relocated hypocentres


def test_nonlinear_observed_times_are_gridloc_corrected(run) -> None:
    """gridloc rewrites otimes.dat with origin-time shifts; residuals must use those times."""
    from osv.analysis.residuals import _observed_times

    root, m = run
    obs = m.rename(columns={"receiver_idx": "receiver", "source_idx": "source", "ray_idx": "ray"})
    obs = obs[["receiver", "source", "ray", "time", "uncertainty"]].assign(subray=0)
    shift = 0.01 * m["source_idx"].to_numpy()  # one origin-time shift per event
    write_observed(root / "iterations" / "01" / "otimes.dat", obs.assign(time=obs["time"] + shift))
    write_observed(root / "otimes.dat", obs.assign(time=obs["time"] + 2 * shift))
    t = m["time"].to_numpy()
    np.testing.assert_allclose(_observed_times(root, 0, m, True), t)  # before any relocation
    np.testing.assert_allclose(_observed_times(root, 1, m, True), t + shift, atol=1e-5)
    # the final iteration without its own copy uses the run's otimes.dat
    np.testing.assert_allclose(_observed_times(root, 2, m, True), t + 2 * shift, atol=1e-5)
    np.testing.assert_allclose(_observed_times(root, 2, m, False), t)  # linear: Stage 3 times


def test_invalid_picks_dropped_and_checks(run) -> None:
    root, m = run
    bad = m.copy()
    bad.loc[0, "valid"] = False
    assert len(residual_table(root, bad, 1)) == len(m) - 1
    with pytest.raises(ValueError, match="rows"):
        residual_table(root, m.iloc[:-1], 1)
    swapped = m.copy()
    swapped.loc[[0, 1], ["source_idx"]] = [[4], [4]]
    with pytest.raises(ValueError, match="disagree"):
        residual_table(root, swapped, 1)


def test_stats_and_variance_reduction(run) -> None:
    root, m = run
    tabs = residual_tables(root, m)
    assert sorted(tabs) == [0, 1, 2]
    vr = variance_reduction(root, m, tabs)
    assert vr["vr_pct"].iloc[0] == pytest.approx(0.0)
    assert vr["vr_pct"].iloc[-1] == pytest.approx(100 * (1 - 0.04), abs=0.1)
    assert vr["vr_pct"].is_monotonic_increasing
    # program numbers agree with the picks (summary.json was built from the same residuals)
    np.testing.assert_allclose(vr["fmtomo_vr_pct"], vr["vr_pct"], atol=1e-3)
    by = residual_stats_by(tabs[0], ["station", "phase"])
    assert len(by) == 6
    sta2 = by[by["station"] == "STA2"]["mean_s"]
    assert (sta2 > 0.2).all()  # the delayed station stands out
    assert set(residual_stats_by(tabs[2], "phase")["phase"]) == {"P", "S"}


def test_relocation_table(run) -> None:
    root, m = run
    rl = relocation_table(root, m, "final")
    assert len(rl) == 4 and list(rl["event_id"]) == ["ev1", "ev2", "ev3", "ev4"]
    assert (rl["n_picks"] == 6).all()
    np.testing.assert_allclose(rl["dz_km"], 0.1)
    np.testing.assert_allclose(rl["dy_km"], 0.02 * KM, rtol=1e-3)
    assert (rl["dx_km"] < 0).all()
    np.testing.assert_allclose(rl["d3_km"], np.sqrt(rl["dh_km"] ** 2 + 0.01))
    np.testing.assert_allclose(rl["dt_s"], [0.02, 0.04, 0.06, 0.08])
    assert (relocation_table(root, m, 0)["d3_km"] == 0).all()


def test_missing_times_message(run) -> None:
    root, m = run
    (root / "iterations/01/mtimes.dat").unlink()
    with pytest.raises(FileNotFoundError, match="iteration 1"):
        residual_table(root, m, 1)
    assert sorted(residual_tables(root, m)) == [0, 2]  # skipped when not explicit


def test_figures(run, tmp_path) -> None:
    root, m = run
    reg = FigureRegistry(tmp_path / "figs", formats=("png",), dpi=50)
    out = pr.make_residual_figures(root, m, reg)
    names = {e["name"] for e in reg.entries()}
    assert names == {
        "residual_histograms", "residual_vs_distance", "residual_vs_azimuth",
        "station_residual_map", "event_rms_before_after", "variance_reduction",
        "hypocentre_shifts", "hypocentre_shift_histograms", "seismicity_before_after",
    }  # fmt: skip
    assert out["relocation"] is not None and set(out["figures"]) == names
    for e in reg.entries():
        assert (tmp_path / "figs" / e["files"][0]).stat().st_size > 1000


def test_no_relocation_skips_hypocentre_figures(run, tmp_path) -> None:
    root, m = run
    for it in (1, 2):  # sources never moved
        (root / f"iterations/{it:02d}/sources.in").write_text(
            (root / "iterations/00/sources.in").read_text()
        )
    reg = FigureRegistry(tmp_path / "figs2", formats=("png",), dpi=40)
    out = pr.make_residual_figures(root, m, reg)
    assert out["relocation"] is None
    assert "seismicity_before_after" not in {e["name"] for e in reg.entries()}
