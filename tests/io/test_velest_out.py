"""Tests for osv.io.velest_out (VELEST ``.OUT`` print-file parser).

Fixtures in ``tests/data/velest_out`` are real runs of the vendored VELEST (gfortran build) on
the Ngatamariki calibration set (67 events, 63 stations, 2514 readings, 6 + 6 layer model of
``MonteCarlo/NM_6Layer_MC_0001.mod``):

* ``real_sim``: simultaneous run, ``ittmax = 9`` (reached), with step-halving backups;
* ``diverged``: absurd alternating 9.5/0.5 km/s model, RMS becomes ``NaN``;
* ``open_error``: the station file is missing, VELEST stops while reading it (truncated .OUT);
* ``single``: ``isingle = 1`` (2 events, ``iturbo = 0``, ``invertratio = 0``);
* ``resolution``: simultaneous, ``ittmax = 3``, ``iresolcalc = 1``.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.io.velest import VelestModel, read_model
from osv.io.velest_out import (
    VelestRunResult,
    parse_velest_out,
    read_run_dir,
    read_velest_out,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "velest_out"


def _text(name: str) -> str:
    p = DATA / name
    if p.suffix == ".gz":
        return gzip.decompress(p.read_bytes()).decode("latin-1")
    return p.read_text(encoding="latin-1")


@pytest.fixture(scope="module")
def real() -> VelestRunResult:
    return read_run_dir(DATA / "real_sim")


@pytest.fixture(scope="module")
def diverged() -> VelestRunResult:
    return read_run_dir(DATA / "diverged")


@pytest.fixture(scope="module")
def open_error() -> VelestRunResult:
    return read_run_dir(DATA / "open_error")


@pytest.fixture(scope="module")
def single() -> VelestRunResult:
    return read_run_dir(DATA / "single")


@pytest.fixture(scope="module")
def resolution() -> VelestRunResult:
    return read_run_dir(DATA / "resolution")


# ---------------------------------------------------------------------------------------
# real simultaneous run
# ---------------------------------------------------------------------------------------


def test_real_header(real: VelestRunResult) -> None:
    assert real.title == "osv my-project calibration"
    assert real.version == "January 3, 1995"
    assert real.n_events == 67
    assert real.n_obs == 2514
    assert real.n_obs_p + real.n_obs_s == 2514
    assert real.control["ittmax"] == 9
    assert real.control["nsp"] == 2
    assert real.control["isingle"] == 0
    assert real.control["olat"] == pytest.approx(-38.569801)
    assert real.control["olon"] == pytest.approx(-176.199402)
    assert real.control["vpvs"] == pytest.approx(1.75)
    assert real.control_files["Modelfile"] == "model.mod"
    assert not real.single_event_mode


def test_real_iterations(real: VelestRunResult) -> None:
    it = real.iterations
    assert list(it["kind"].unique()) == ["initial", "iteration", "backup"]
    assert real.initial_rms == pytest.approx(0.247602)
    first = it[(it["iteration"] == 1) & (it["kind"] == "iteration")].iloc[0]
    assert first["rms"] == pytest.approx(0.113590)
    assert first["datvar"] == pytest.approx(0.014941)
    assert first["mean_sq_res"] == pytest.approx(0.012903)
    assert first["step_length"] == pytest.approx(3.899)
    assert first["damping"] == pytest.approx(0.58769)
    assert first["n_depth_constrained"] == 1
    assert first["n_direct"] == 2393 and first["n_refracted"] == 121 and first["n_all"] == 2514
    assert first["abs_res_all"] == pytest.approx(0.14282)
    assert first["mean_res_all"] == pytest.approx(0.13473)
    assert first["pct_change_abs_res"] == pytest.approx(-57.80)
    assert real.final_rms == pytest.approx(0.053455)
    assert real.n_iterations == 9


def test_real_backups(real: VelestRunResult) -> None:
    assert real.backup_iterations == [6, 9]
    assert real.n_backups == 4
    bk = real.iterations[real.iterations["kind"] == "backup"]
    assert list(bk["backup_nr"]) == [1, 2, 1, 2]
    assert (bk["step_length"] < 0).all()  # halved, reversed steps
    # iteration 6 first got worse (0.056003 > 0.053497), the second backup was accepted
    i6 = real.iterations[real.iterations["iteration"] == 6]
    assert list(i6["accepted"]) == [False, False, True]
    assert i6["rms"].iloc[0] > i6["rms"].iloc[-1]
    # exactly one accepted entry per iteration
    acc = real.accepted
    assert list(acc["iteration"]) == list(range(10))
    hist = real.rms_history
    assert hist.loc[0] == pytest.approx(0.247602)
    assert hist.loc[9] == pytest.approx(0.053455)
    assert hist.loc[2] < hist.loc[1] < hist.loc[0]


def test_real_stop_and_status(real: VelestRunResult) -> None:
    assert real.stop_reason == "ittmax"
    assert real.finished
    assert not real.failed and real.failure_reason is None
    assert not real.converged  # ittmax reached, never "STEP-LENGTH < DELMIN"
    assert not real.diverged
    assert real.summary["n_rays"] == 2514
    assert real.summary["n_refracted"] == 179
    assert real.summary["gap_min"] == 34 and real.summary["gap_max"] == 154
    assert real.summary["nvar"] == 405
    assert real.warnings == []


def test_real_hypocentre_shifts(real: VelestRunResult) -> None:
    sh = real.hypocentre_shifts
    s1 = sh[sh["iteration"] == 1].iloc[0]
    assert (s1["dot"], s1["dx"], s1["dy"], s1["dz"]) == pytest.approx(
        (0.134, -0.064, -0.076, -0.149)
    )
    assert (s1["adot"], s1["adx"], s1["ady"], s1["adz"]) == pytest.approx(
        (0.134, 0.178, 0.189, 0.195)
    )
    last = sh.iloc[-1]
    assert abs(last["adz"]) < 0.01  # converged: adjustments are tiny


def test_real_final_model(real: VelestRunResult) -> None:
    m = real.final_model
    assert isinstance(m, VelestModel)
    assert m.title == "NM_6Layer_MC_0001"
    np.testing.assert_allclose(m.p.layers["vel"], [2.492, 3.784, 2.711, 3.930, 2.701, 5.181])
    np.testing.assert_allclose(m.p.layers["depth"], [-0.6, 0.2, 0.7, 1.2, 1.7, 2.2])
    assert m.s is not None
    np.testing.assert_allclose(m.s.layers["vel"], [1.344, 2.305, 1.754, 2.616, 1.750, 3.113])
    init = real.initial_model
    assert init is not None
    np.testing.assert_allclose(init.p.layers["vel"], [2.54, 4.46, 2.83, 4.34, 2.70, 5.60])
    # same initial model as the input file
    assert (
        init.p.layers["vel"].tolist()
        == read_model(DATA / "real_sim" / "model.mod").p.layers["vel"].tolist()
    )
    # per-iteration models: 12 rows per printed velocity block
    models = real.models
    assert set(models["phase"]) == {"P", "S"}
    one = models[(models["iteration"] == 1) & ~models["readjusted"]]
    assert len(one) == 12
    p1 = one[one["phase"] == "P"].sort_values("layer")
    np.testing.assert_allclose(p1["vel"], [2.529, 4.350, 2.809, 4.304, 2.683, 5.400])
    np.testing.assert_allclose(p1["dvel"], [-0.011, -0.110, -0.021, -0.036, -0.017, -0.200])
    assert models["readjusted"].any()  # backups print readjustments


def test_real_station_corrections(real: VelestRunResult) -> None:
    sc = real.station_corrections
    assert len(sc) == 63
    row = sc.set_index("sta").loc["ALRZ"]
    assert row["ptcor"] == pytest.approx(-0.227)
    assert row["stcor"] == pytest.approx(-0.175)
    assert (row["nobs_p"], row["nobs_s"], row["nobs"]) == (39, 3, 42)
    assert sc["ptcor"].between(-0.4, 0.2).all()
    assert sc["stcor"].between(-0.4, 1.3).all()
    ref = sc.set_index("sta").loc["NS14"]  # reference station: P delay fixed at zero
    assert ref["ptcor"] == 0.0
    st = real.station_stats
    assert set(st["phase"]) == {"P", "S"}
    a = st[(st["sta"] == "ALRZ") & (st["phase"] == "P")].iloc[0]
    assert (a["nobs"], a["delay"]) == (39, pytest.approx(-0.2272))
    # stats delay agrees with the adjusted-correction table
    merged = st[st["phase"] == "P"].merge(sc, on="sta")
    np.testing.assert_allclose(merged["delay"], merged["ptcor"], atol=6e-4)


def test_real_final_hypocentres_and_output_files(real: VelestRunResult) -> None:
    h = real.final_hypocentres
    assert len(h) == 67
    r0 = h.iloc[0]
    assert r0["time"] == pd.Timestamp("2017-04-09 05:41:58.59", tz="UTC")
    assert r0["lat"] == pytest.approx(-38.5328)  # S letter -> negative
    assert r0["lon"] == pytest.approx(176.1891)
    assert r0["depth_km"] == pytest.approx(2.75)
    assert r0["nobs"] == 33
    assert real.cnv is not None and len(real.cnv.events) == 67
    assert real.stations is not None and len(real.stations) == 63
    # print file and output CNV agree on the hypocentres
    ev = real.cnv.events
    np.testing.assert_allclose(ev["lat"], h["lat"], atol=1e-4)
    np.testing.assert_allclose(ev["lon"], h["lon"], atol=1e-4)
    np.testing.assert_allclose(ev["depth_km"], h["depth_km"], atol=0.011)
    # output station file carries the same delays as the print file
    sta = real.stations.set_index("sta")
    sc = real.station_corrections.set_index("sta")
    np.testing.assert_allclose(sta.loc[sc.index, "ptcor"], sc["ptcor"], atol=0.006)


def test_real_ray_statistics(real: VelestRunResult) -> None:
    ray = real.ray_statistics
    assert len(ray) == 6
    assert ray["nhit"].iloc[0] == 2528
    assert ray["nref"].sum() == 165 + 14
    assert real.resolution is None


# ---------------------------------------------------------------------------------------
# diverged run
# ---------------------------------------------------------------------------------------


def test_diverged_run(diverged: VelestRunResult) -> None:
    assert diverged.finished  # the program ran to the end
    assert diverged.failed
    assert "non-finite RMS" in (diverged.failure_reason or "")
    assert diverged.diverged and not diverged.converged
    assert math.isnan(diverged.final_rms)
    assert diverged.initial_rms == pytest.approx(1.247537)
    assert diverged.stop_reason == "four_backups"
    it = diverged.iterations
    assert it["rms"].isna().sum() >= 6  # NaN fields parsed, not crashed
    bk = it[it["kind"] == "backup"]
    assert bk["backup_nr"].max() == 4
    # an accepted backup with a finite RMS exists (iteration 1) and has a finite model
    assert diverged.rms_history.loc[1] == pytest.approx(1.149776)
    assert diverged.final_model is not None
    assert len(diverged.final_model.p.layers) == 4
    # suspicious events are reported once per evaluation, station variances once
    assert any("suspicious" in w for w in diverged.warnings)
    assert any("negative station residual variance" in w for w in diverged.warnings)
    assert len(diverged.warnings) < 60


# ---------------------------------------------------------------------------------------
# crash while reading input (truncated output)
# ---------------------------------------------------------------------------------------


def test_open_error_run(open_error: VelestRunResult) -> None:
    r = open_error
    assert r.failed and not r.finished and not r.converged
    assert "file open error" in (r.failure_reason or "")
    assert "station-input-file" in (r.failure_reason or "")
    assert r.iterations.empty and r.models.empty and r.station_corrections.empty
    assert r.final_model is None
    assert r.control["neqs"] == 67 and r.n_events == 67
    assert r.n_obs is None
    assert r.title == "osv my-project calibration"
    assert r.warnings and "ERROR OPENING FILE" in r.warnings[0]
    assert r.cnv is None  # no velout.cnv written


@pytest.mark.parametrize("frac", [0.0, 0.01, 0.05, 0.3, 0.5, 0.97])
def test_truncated_real_output_never_raises(frac: float) -> None:
    text = _text("real_sim/velest.out.gz")
    cut = text[: int(len(text) * frac)]
    r = parse_velest_out(cut)
    assert isinstance(r, VelestRunResult)
    assert r.failed and not r.finished and not r.converged
    assert r.failure_reason
    if frac >= 0.3:
        assert len(r.iterations) >= 1
        assert r.iterations["rms"].iloc[0] == pytest.approx(0.247602)


def test_truncated_in_the_middle_of_a_line() -> None:
    text = _text("real_sim/velest.out.gz")
    pos = text.index("RMS RESIDUAL=     0.113590")
    r = parse_velest_out(text[: pos + 20])  # "RMS RESIDUAL=     0.11"
    assert r.failed
    assert r.iterations["iteration"].max() == 1


def test_garbage_and_empty_input() -> None:
    r = parse_velest_out("")
    assert r.failed and r.failure_reason == "empty output"
    r = parse_velest_out("\x00\xff garbage\n\n*** olat\n abc\n" * 3)
    assert r.failed and r.iterations.empty


def test_missing_file_and_directory(tmp_path: Path) -> None:
    r = read_velest_out(tmp_path / "nope.out")
    assert r.failed and "cannot read" in (r.failure_reason or "")
    r = read_run_dir(tmp_path)
    assert r.failed and "no VELEST print output" in (r.failure_reason or "")


# ---------------------------------------------------------------------------------------
# single-event mode and resolution
# ---------------------------------------------------------------------------------------


def test_single_event_mode(single: VelestRunResult) -> None:
    r = single
    assert r.single_event_mode
    assert r.n_events == 2
    assert r.n_obs == 33 + 23
    assert sorted(r.iterations["run"].unique()) == [1, 2]
    per_run = r.iterations.groupby("run")["iteration"].max()
    assert list(per_run) == [6, 7]
    assert r.stop_reasons == ["delmin", "delmin"]
    assert r.stop_reason == "delmin"
    assert r.converged and not r.failed and r.finished
    first = r.iterations[r.iterations["run"] == 1]
    assert first["rms"].iloc[0] == pytest.approx(0.227397)
    assert first["rms"].iloc[-1] == pytest.approx(0.078245)
    assert r.final_model is r.initial_model  # invertratio = 0: model fixed
    assert len(r.final_hypocentres) == 2
    # VELEST's single-event mode rewrites velout.cnv per event: only one event is left
    assert r.cnv is not None and len(r.cnv.events) == 1


def test_resolution_diagonal(resolution: VelestRunResult) -> None:
    r = resolution
    res = r.resolution
    assert res is not None
    assert list(res["param"][:6]) == ["ot1", "x1", "y1", "z1", "vp1", "vp2"]
    assert len(res) == 16 and res["param"].iloc[-1] == "vs6"
    assert res["rdiag"].iloc[0] == pytest.approx(0.9982)
    assert res["rdiag"].iloc[3] == pytest.approx(0.8989)
    assert res["rdiag"].iloc[-1] == pytest.approx(0.7379)
    assert res["sigma"].iloc[0] == pytest.approx(0.3928)
    assert r.stop_reason == "ittmax" and not r.converged and not r.failed
    assert r.n_iterations == 3


def test_converged_flag_from_delmin_text() -> None:
    """A run that stops on STEP-LENGTH < DELMIN is converged (synthetic edit of a real one)."""
    text = _text("resolution/velest.out.gz")
    assert "NITT = ITTMAX  --> stop iterations." in text
    t2 = text.replace(
        "NITT = ITTMAX  --> stop iterations.", "STEP-LENGTH < DELMIN  --> stop iterations."
    )
    r = parse_velest_out(t2)
    assert r.stop_reason == "delmin" and r.converged
