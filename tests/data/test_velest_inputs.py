"""Tests for osv.data.velest_inputs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import OSVConfig
from osv.data.select import Selection
from osv.data.stations import StationTables
from osv.data.velest_inputs import (
    VelestInputError,
    VelestLimitError,
    choose_reference_station,
    write_velest_inputs,
)
from osv.io.velest import read_cnv, read_control, read_stations
from osv.tools import ToolNotFoundError, which_tool

REPO = Path(__file__).resolve().parents[2]
T0 = pd.Timestamp("2017-06-01 12:00:00", tz="UTC")

# alias -> (lat, lon, surface elevation m, sensor depth m)
STA = {
    "AAA": (-38.50, 176.10, 400.0, 0.0),
    "BBB": (-38.55, 176.20, 300.0, 0.0),
    "CCC": (-38.60, 176.15, 350.0, 150.0),  # borehole
    "DDD": (-38.58, 176.12, 320.0, 0.0),
}


def _cfg(**coords) -> OSVConfig:
    return OSVConfig.model_validate({"coordinates": coords})


def _stations() -> StationTables:
    rows = [
        {
            "net": "NZ",
            "sta": a,
            "alias": a,
            "start": pd.Timestamp("2010-01-01", tz="UTC"),
            "lat": la,
            "lon": lo,
            "elevation_m": el,
            "sensor_depth_m": d,
            "sensor_elevation_m": el - d,
        }
        for a, (la, lo, el, d) in STA.items()
    ]
    return StationTables(
        stations=pd.DataFrame(rows), channels=pd.DataFrame(), aliases=pd.DataFrame()
    )


def _selection(n_events=3, picks_per_event=None, **overrides) -> Selection:
    ev, pk = [], []
    for i in range(n_events):
        eid = f"e{i}"
        ev.append(
            {
                "event_id": eid,
                "time": T0 + pd.Timedelta(minutes=10 * (n_events - i)),  # reverse time order
                "lat": -38.55 + 0.01 * i,
                "lon": 176.15,
                "depth_km": 2.0 + i,
                "depth_km_bsl": 2.0 + i - 0.5,
                "mag": -6.19,
                "mag_valid": False,
                "split": "validation" if i == n_events - 1 else "calibration",
            }
        )
        for a, (la, lo, el, d) in STA.items():
            for ph, tt in (("P", 1.0 + 0.1 * i), ("S", 1.7 + 0.1 * i)):
                pk.append(
                    {
                        "event_id": eid,
                        "alias": a,
                        "phase": ph,
                        "travel_time_s": tt,
                        "velest_weight": 0 if ph == "P" else 1,
                        "sigma_s": 0.02,
                        "station_lat": la,
                        "station_lon": lo,
                        "station_elevation_m": el,
                        "sensor_elevation_m": el - d,
                    }
                )
    events = pd.DataFrame(ev)
    picks = pd.DataFrame(pk)
    split = pd.Series(events["split"].to_numpy(), index=events["event_id"], name="split")
    sel = Selection(events=events, picks=picks, rejections=pd.DataFrame(), split=split)
    for k, v in overrides.items():
        setattr(sel, k, v)
    return sel


def test_round_trip_and_ordering(tmp_path):
    sel = _selection()
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path)
    assert vi.n_events == 2 and vi.subset == "calibration"  # one event is validation
    assert vi.n_picks == 2 * 8
    assert vi.control.neqs == 2 and vi.control.zshift == 0.0 and vi.control.iuseelev == 1
    assert vi.control.stationfile == "stations.sta" and vi.control.phasefile == "events.cnv"
    cnv = read_cnv(vi.cnv_path)
    # sorted by time: e1 (earlier of the calibration pair) then e0; file order = event_map order
    assert list(vi.events["event_id"]) == ["e1", "e0"]
    # (read_cnv numbers events from 0 in file order; event_map.csv cnv_id is 1-based)
    assert list(cnv.events["event"]) == [0, 1]
    assert cnv.events["time"].is_monotonic_increasing
    # depth is below sea level (depth_km_bsl), true hemisphere letters
    assert np.allclose(cnv.events["depth_km"], [2.5, 1.5])
    assert np.allclose(cnv.events["lat"], [-38.54, -38.55], atol=1e-4)
    assert "S " in vi.cnv_path.read_text().splitlines()[0]
    assert cnv.picks["tt"].between(0, 100).all()
    assert set(cnv.picks["sta"]) == set(STA)
    assert (cnv.picks.groupby("event").size() == 8).all()
    # picks carry weights
    assert set(cnv.picks.loc[cnv.picks.phase == "S", "weight"]) == {1}
    # summary file and event map
    summ = json.loads(vi.summary_path.read_text())
    assert summ["n_events"] == 2 and summ["reference_station"] == vi.reference_station
    assert pd.read_csv(vi.event_map_path)["event_id"].tolist() == ["e1", "e0"]


def test_subsets(tmp_path):
    sel = _selection()
    cfg = _cfg()
    assert write_velest_inputs(sel, _stations(), cfg, tmp_path / "v", "validation").n_events == 1
    assert write_velest_inputs(sel, _stations(), cfg, tmp_path / "a", "all").n_events == 3
    with pytest.raises(ValueError, match="subset"):
        write_velest_inputs(sel, _stations(), cfg, tmp_path / "x", "bogus")  # type: ignore[arg-type]


def test_stations_reference_last_and_borehole(tmp_path):
    vi = write_velest_inputs(_selection(), _stations(), _cfg(), tmp_path)
    sta = read_stations(vi.sta_path)
    assert list(sta["icc"]) == [1, 2, 3, 4]
    # reference station gets the highest icc (velest.f:1580-1585, 1622)
    assert sta["sta"].iloc[-1] == vi.reference_station
    assert vi.stations["icc"].iloc[-1] == 4
    assert (sta["model"] == 1).all() and (sta["ptcor"] == 0).all() and (sta["stcor"] == 0).all()
    s = sta.set_index("sta")
    assert s.loc["CCC", "elev_m"] == 200  # 350 - 150 m borehole sensor
    assert s.loc["AAA", "elev_m"] == 400
    assert s.loc["AAA", "lat"] == pytest.approx(-38.50, abs=1e-4)  # true hemisphere
    assert s.loc["AAA", "lon"] == pytest.approx(176.10, abs=1e-4)
    # file ends with a blank record
    assert vi.sta_path.read_text().endswith("\n\n")


def test_ignore_depth_uses_surface_elevation(tmp_path):
    cfg = _cfg(borehole_sensors="ignore_depth")
    vi = write_velest_inputs(_selection(), _stations(), cfg, tmp_path)
    assert read_stations(vi.sta_path).set_index("sta").loc["CCC", "elev_m"] == 350


def test_positions_from_station_table_when_picks_lack_them(tmp_path):
    sel = _selection()
    sel.picks = sel.picks.drop(
        columns=["station_lat", "station_lon", "station_elevation_m", "sensor_elevation_m"]
    )
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path)
    assert read_stations(vi.sta_path).set_index("sta").loc["CCC", "elev_m"] == 200


def test_reference_station_choice():
    st = pd.DataFrame(
        {
            "alias": ["A", "B", "C", "D"],
            "lat": [0.0, 0.0, 0.0, 10.0],
            "lon": [0.0, 1.0, 2.0, 1.0],
        }
    )
    n = pd.Series({"A": 100, "B": 90, "C": 100, "D": 100})
    # centroid is (2.5, 1.0): B is nearest, and has >= 50% of the max count
    assert choose_reference_station(st, n) == "B"
    # a central station with too few picks is not eligible
    assert choose_reference_station(st, pd.Series({"A": 100, "B": 10, "C": 100, "D": 0})) == "A"


def test_explicit_reference_station(tmp_path):
    vi = write_velest_inputs(_selection(), _stations(), _cfg(), tmp_path, reference_station="AAA")
    assert vi.reference_station == "AAA"
    assert read_stations(vi.sta_path)["sta"].iloc[-1] == "AAA"
    with pytest.raises(VelestInputError, match="reference station"):
        write_velest_inputs(
            _selection(), _stations(), _cfg(), tmp_path / "x", reference_station="ZZZ"
        )


def test_magnitudes(tmp_path):
    sel = _selection(n_events=3)
    sel.events["mag_valid"] = [True, True, False]
    sel.events["mag"] = [2.5, -0.4, -6.19]
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all")
    mags = dict(zip(vi.events["event_id"], vi.events["mag"], strict=True))
    assert mags == {"e0": 2.5, "e1": 0.0, "e2": 0.0}
    assert list(read_cnv(vi.cnv_path).events["mag"]) == [0.0, 0.0, 2.5]  # time order e2, e1, e0
    codes = {f.code for f in vi.findings}
    assert {"magnitude_written_zero", "magnitude_negative_clamped"} <= codes


def test_invalid_travel_times_dropped(tmp_path):
    sel = _selection(n_events=2)
    p = sel.picks
    idx = p.index[(p.event_id == "e0") & (p.alias == "AAA")]
    p.loc[idx[0], "travel_time_s"] = -0.5
    p.loc[idx[1], "travel_time_s"] = 100.4
    p.loc[
        p.index[(p.event_id == "e0") & (p.alias == "BBB") & (p.phase == "P")], "travel_time_s"
    ] = 0.0
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all")
    assert vi.n_picks == len(p) - 3 and vi.n_picks_dropped == 3
    f = next(f for f in vi.findings if f.code == "picks_dropped_travel_time")
    assert f.count == 3
    assert read_cnv(vi.cnv_path).picks["tt"].between(0, 100).all()


def test_duplicate_picks_keep_best_weight(tmp_path):
    sel = _selection(n_events=1)
    extra = sel.picks[(sel.picks.alias == "AAA") & (sel.picks.phase == "P")].copy()
    extra["velest_weight"] = 2
    extra["travel_time_s"] = 1.05
    sel.picks = pd.concat([sel.picks, extra], ignore_index=True)
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all")
    assert vi.n_picks == 8 and vi.n_picks_dropped == 1
    pk = read_cnv(vi.cnv_path).picks
    assert pk[(pk.sta == "AAA") & (pk.phase == "P")]["weight"].tolist() == [0]


def test_weight4_written_and_counted(tmp_path):
    sel = _selection(n_events=1)
    sel.picks.loc[sel.picks.index[0], "velest_weight"] = 4
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all")
    assert vi.n_picks_by_weight[4] == 1 and vi.n_picks_unused == 1
    assert (read_cnv(vi.cnv_path).picks["weight"] == 4).sum() == 1
    assert any(f.code == "weight_4_picks_written" for f in vi.findings)


def test_event_with_only_weight4_picks_dropped(tmp_path):
    sel = _selection(n_events=2)
    sel.picks.loc[sel.picks.event_id == "e0", "velest_weight"] = 4
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all")
    assert vi.n_events == 1 and vi.control.neqs == 1
    assert any(f.code == "events_dropped_no_usable_picks" for f in vi.findings)


def test_limits_fail_clearly(tmp_path):
    sel = _selection(n_events=3)
    with pytest.raises(VelestLimitError, match="ieq=2"):
        write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all", max_events=2)
    with pytest.raises(VelestLimitError, match="ist=3"):
        write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all", max_stations=3)
    with pytest.raises(VelestLimitError, match="maxobsperevent=5"):
        write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all", max_obs_per_event=5)
    # default limits: the real VELEST maxima
    assert issubclass(VelestLimitError, VelestInputError)


def test_empty_subset_error(tmp_path):
    sel = _selection(n_events=2)
    sel.events["split"] = "calibration"
    with pytest.raises(VelestInputError, match="validation"):
        write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "validation")


def test_control_origin_and_zmin(tmp_path):
    # no origin: station centroid
    vi = write_velest_inputs(_selection(), _stations(), _cfg(), tmp_path / "a")
    lat, lon = vi.control.origin()
    assert lat == pytest.approx(np.mean([v[0] for v in STA.values()]), abs=1e-3)
    assert lon == pytest.approx(np.mean([v[1] for v in STA.values()]), abs=1e-3)
    assert any(f.code == "origin_from_station_centroid" for f in vi.findings)
    assert vi.control.zmin == 0.0
    # configured origin; events above sea level lower zmin
    sel = _selection()
    sel.events["depth_km_bsl"] = [-0.35, 1.0, 2.0]
    vi = write_velest_inputs(
        sel, _stations(), _cfg(origin={"lat": -38.55, "lon": 176.19}), tmp_path / "b", "all"
    )
    assert vi.control.origin() == (-38.55, 176.19)
    assert vi.control.zmin == pytest.approx(-0.5)
    c2 = read_control(_write_ctrl(vi, tmp_path))
    assert c2.neqs == vi.control.neqs


def _write_ctrl(vi, tmp_path):
    from osv.io.velest import write_control

    return write_control(tmp_path / "velest.cmn", vi.control)


def test_depth_km_fallback_finding(tmp_path):
    sel = _selection(n_events=1)
    sel.events = sel.events.drop(columns=["depth_km_bsl"])
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all")
    assert read_cnv(vi.cnv_path).events["depth_km"].iloc[0] == pytest.approx(2.0)
    assert any(f.code == "depth_datum_assumed" for f in vi.findings)


def test_moved_station_collapsed(tmp_path):
    sel = _selection(n_events=2)
    m = (sel.picks.event_id == "e0") & (sel.picks.alias == "AAA")
    sel.picks.loc[m, "station_lat"] += 0.01
    vi = write_velest_inputs(sel, _stations(), _cfg(), tmp_path, "all")
    f = next(f for f in vi.findings if f.code == "station_epochs_collapsed")
    assert f.count == 1
    # the most-used position (the unmoved one) is written
    assert read_stations(vi.sta_path).set_index("sta").loc["AAA", "lat"] == pytest.approx(-38.5)


# --- integration on the real data ---------------------------------------------------------

_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"
_MOD = Path("/home/steve/seismic/velest/MonteCarlo/NM_6Layer_MC_0001.mod")


def _rms_values(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"RMS RESIDUAL=\s*([0-9.]+)", text)]


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_data_inputs_accepted_by_velest(tmp_path):
    from osv.data.associate import associate_picks
    from osv.data.datum import apply_depth_datum
    from osv.data.quakeml import read_catalog
    from osv.data.select import select_data
    from osv.data.stations import read_stations as read_station_xml
    from osv.io.velest import read_model, write_control, write_model
    from osv.tools import run_tool

    cfg = OSVConfig()
    cat = read_catalog(_CAT)
    st = read_station_xml(_FILES)
    cat.events = apply_depth_datum(cat.events, cfg.coordinates.depth_datum)
    sel = select_data(associate_picks(cat, st), cfg)
    vi = write_velest_inputs(sel, st, cfg, tmp_path)

    assert vi.n_events == int((sel.events["split"] == "calibration").sum()) > 0
    cnv = read_cnv(vi.cnv_path)
    assert len(cnv.events) == vi.n_events and len(cnv.picks) == vi.n_picks
    sta = read_stations(vi.sta_path)
    assert len(sta) == vi.n_stations <= 300
    assert sta["sta"].iloc[-1] == vi.reference_station
    assert set(cnv.picks["sta"]) <= set(sta["sta"])
    assert set(sta["sta"]) <= set(cnv.picks["sta"])
    assert (cnv.picks.groupby("event").size() <= 180).all()

    try:
        which_tool("velest")
    except ToolNotFoundError:
        pytest.skip("velest binary not available (set OSV_TOOLS_DIR)")
    if not _MOD.exists():
        pytest.skip("legacy 1-D model not available")

    write_model(tmp_path / "model.mod", read_model(_MOD))
    c = vi.control
    c.modelfile = "model.mod"
    c.outfile, c.cnvfile, c.stafile = "velest.out", "velout.cnv", "velout.sta"
    write_control(tmp_path / "velest.cmn", c)
    res = run_tool("velest", tmp_path, timeout=600)
    assert res.ok
    out = (tmp_path / "velest.out").read_text(errors="replace")
    rms = _rms_values(out)
    print(f"\nVELEST RMS per iteration: {rms}")
    assert len(rms) >= 2 and rms[-1] < rms[0]
    final = read_cnv(tmp_path / "velout.cnv")
    assert len(final.events) == vi.n_events
