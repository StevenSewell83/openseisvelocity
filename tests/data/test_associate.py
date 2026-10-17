"""Tests for osv.data.associate."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import InputsConfig
from osv.data.associate import _azimuthal_gap, associate_picks
from osv.data.quakeml import CatalogTables
from osv.data.stations import StationTables

REPO = Path(__file__).resolve().parents[2]
T0 = pd.Timestamp("2017-06-01", tz="UTC")


def _stations(rows):
    df = pd.DataFrame(rows)
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return StationTables(stations=df, channels=pd.DataFrame(), aliases=pd.DataFrame())


def _sta(sta, lat, lon, elev=0.0, depth=0.0, start="2015-01-01", end=None):
    return {
        "net": "NZ",
        "sta": sta,
        "alias": sta,
        "loc": "10",
        "start": start,
        "end": end,
        "lat": lat,
        "lon": lon,
        "elevation_m": elev,
        "sensor_depth_m": depth,
        "sensor_elevation_m": elev - depth,
    }


def _pk(eid, sta, phase, tt, cha="EHZ", unc=0.02, w=1.0, dt=0.0):
    return {
        "event_id": eid,
        "pick_id": f"{eid}.{sta}.{phase}.{cha}.{tt}",
        "net": "NZ",
        "sta": sta,
        "loc": "10",
        "cha": cha,
        "phase": phase,
        "time": T0 + pd.Timedelta(seconds=tt + dt),
        "unc_s": unc,
        "comment_weight": w,
        "travel_time_s": tt,
    }


def _events():
    return pd.DataFrame(
        {"event_id": ["e1"], "time": [T0], "lat": [0.0], "lon": [0.0], "depth_km": [2.0]}
    )


def _cat(picks, events=None):
    cols = ["unc_lower", "unc_upper"]
    p = pd.DataFrame(picks)
    for c in cols:
        p[c] = np.nan
    return CatalogTables(events=events if events is not None else _events(), picks=p)


STA = [
    _sta("N", 0.1, 0.0),
    _sta("E", 0.0, 0.1),
    _sta("S", -0.1, 0.0),
    _sta("W", 0.0, -0.1, elev=100.0, depth=50.0),
]


def _basic_picks():
    return [_pk("e1", s, "P", 3.0) for s in "NESW"]


def test_basic_geometry_and_events():
    picks = [
        *_basic_picks(),
        _pk("e1", "N", "S", 5.0, cha="EHT"),
        _pk("e1", "E", "S", 5.2, cha="EH1"),
    ]
    r = associate_picks(_cat(picks), _stations(STA))
    p = r.picks
    assert len(p) == 6
    n = p[(p.sta == "N") & (p.phase == "P")].iloc[0]
    assert n.epicentral_km == pytest.approx(11.06, abs=0.1)
    assert n.hypocentral_km == pytest.approx(np.hypot(n.epicentral_km, 2.0), abs=1e-6)
    assert n.azimuth_deg == pytest.approx(0, abs=0.1)
    assert n.backazimuth_deg == pytest.approx(180, abs=0.1)
    assert n.apparent_velocity_kms == pytest.approx(n.hypocentral_km / 3.0)
    w = p[(p.sta == "W") & (p.phase == "P")].iloc[0]  # sensor at 50 m elevation
    assert w.sensor_elevation_m == 50.0
    assert w.hypocentral_km == pytest.approx(np.hypot(w.epicentral_km, 2.05), abs=1e-6)
    # S on horizontals is kept; Ts-Tp only for S with P at same station
    s = p[p.phase == "S"].set_index("sta")
    assert set(s.cha) == {"EHT", "EH1"}
    assert s.loc["N", "ts_minus_tp_s"] == pytest.approx(2.0)
    assert p[p.phase == "P"].ts_minus_tp_s.isna().all()
    ev = r.events.iloc[0]
    assert (ev.n_P, ev.n_S, ev.n_stations) == (4, 2, 4)
    assert ev.azgap_deg == pytest.approx(90.0, abs=1.0)
    assert ev.nearest_station_km == pytest.approx(11.06, abs=0.1)


def test_s_without_p_has_nan_tstp():
    r = associate_picks(_cat([_pk("e1", "N", "S", 5.0, cha="EHN")]), _stations(STA))
    assert len(r.picks) == 1
    assert np.isnan(r.picks.ts_minus_tp_s.iloc[0])


def test_missing_station_and_outside_epoch():
    stas = [_sta("N", 0.1, 0.0, start="2017-07-01")]  # starts after the pick
    picks = [_pk("e1", "N", "P", 3.0), _pk("e1", "ZZZ", "P", 3.0), _pk("e1", "ZZZ", "S", 5.0)]
    r = associate_picks(_cat(picks), _stations(stas))
    assert r.picks.empty
    f = {x.code: x for x in r.findings}
    assert f["pick_no_station"].count == 2
    assert f["pick_no_station"].details["per_station"] == {"NZ.ZZZ": 2}
    assert f["pick_outside_epoch"].count == 1
    assert f["pick_outside_epoch"].details["per_station"] == {"NZ.N": 1}
    assert r.events.n_stations.iloc[0] == 0
    assert r.events.azgap_deg.iloc[0] == 360.0


def test_epoch_selects_right_station_row():
    stas = [
        _sta("N", 0.1, 0.0, end="2017-01-01"),
        _sta("N", 0.2, 0.0, start="2017-01-02"),
    ]
    r = associate_picks(_cat([_pk("e1", "N", "P", 3.0)]), _stations(stas))
    assert r.picks.station_lat.iloc[0] == 0.2


def test_phase_filter():
    picks = [
        _pk("e1", "N", "P", 3.0),
        {**_pk("e1", "E", "P", 3.0), "phase": None},
        {**_pk("e1", "S", "P", 3.0), "phase": "Pn"},
    ]
    r = associate_picks(_cat(picks), _stations(STA))
    assert len(r.picks) == 1
    f = {x.code: x for x in r.findings}
    assert f["phase_dropped"].count == 2


def test_duplicates_keep_best():
    picks = [
        _pk("e1", "N", "P", 3.0, unc=0.05, cha="EHZ"),
        _pk("e1", "N", "P", 3.1, unc=0.01, cha="EH1", w=0.5),
        _pk("e1", "N", "P", 3.2, unc=0.01, cha="EH2", w=1.0),  # same unc, higher weight
        _pk("e1", "N", "P", 3.3, unc=0.01, cha="EHN", w=1.0, dt=0.0),  # tie: later time
        _pk("e1", "N", "S", 5.0),
    ]
    r = associate_picks(_cat(picks), _stations(STA))
    assert len(r.picks) == 2
    kept = r.picks[r.picks.phase == "P"].iloc[0]
    assert kept.cha == "EH2"
    assert {x.code: x for x in r.findings}["duplicate_picks"].count == 3


def test_duplicate_missing_unc_loses():
    picks = [_pk("e1", "N", "P", 3.0, unc=np.nan), _pk("e1", "N", "P", 3.1, unc=0.04)]
    r = associate_picks(_cat(picks), _stations(STA))
    assert r.picks.travel_time_s.iloc[0] == 3.1


def _wpicks():
    return [
        _pk("e1", "N", "P", 3.0, unc=0.02, w=1.0),
        _pk("e1", "E", "P", 3.0, unc=0.04, w=0.5),
        _pk("e1", "S", "P", 3.0, unc=0.06, w=1.0),
        _pk("e1", "W", "P", 3.0, unc=np.nan, w=np.nan),
    ]


def _sig(mode):
    cfg = InputsConfig(weight_source=mode)
    r = associate_picks(_cat(_wpicks()), _stations(STA), cfg)
    return r.picks.set_index("sta").sigma_s, r


def test_weight_uncertainty():
    s, r = _sig("uncertainty")
    assert s["N"] == 0.02 and s["E"] == 0.04
    assert s["W"] == pytest.approx(0.04)  # median of 0.02, 0.04, 0.06
    assert r.picks.set_index("sta").sigma_filled["W"]
    assert "missing_sigma" in {x.code for x in r.findings}


def test_weight_comment():
    s, _ = _sig("comment")
    assert s["N"] == pytest.approx(0.04)  # base median 0.04 / 1
    assert s["E"] == pytest.approx(0.08)  # / 0.5
    assert s["W"] == pytest.approx(0.04)  # missing weight -> 1


def test_weight_both():
    s, r = _sig("both")
    assert s["N"] == pytest.approx(0.02)
    assert s["E"] == pytest.approx(0.08)
    assert s["S"] == pytest.approx(0.06)
    assert s["W"] == pytest.approx(0.06 / 1)  # NaN unc: median of 0.02, 0.08, 0.06
    assert "missing_comment_weight" in {x.code for x in r.findings}


def test_zero_weight_gives_inf():
    picks = [_pk("e1", "N", "P", 3.0, unc=0.02, w=0.0), _pk("e1", "E", "P", 3.0, unc=0.02)]
    r = associate_picks(_cat(picks), _stations(STA), InputsConfig(weight_source="both"))
    assert np.isinf(r.picks.set_index("sta").sigma_s["N"])


def test_depth_bsl_preferred():
    ev = _events()
    ev["depth_km_bsl"] = 4.0
    r = associate_picks(_cat([_pk("e1", "N", "P", 3.0)], ev), _stations(STA))
    p = r.picks.iloc[0]
    assert p.hypocentral_km == pytest.approx(np.hypot(p.epicentral_km, 4.0))
    assert r.events.attrs["depth_column"] == "depth_km_bsl"


def test_orphan_event_dropped():
    r = associate_picks(_cat([_pk("nope", "N", "P", 3.0)]), _stations(STA))
    assert r.picks.empty
    assert "pick_without_event" in {x.code for x in r.findings}


def test_gap_function():
    assert _azimuthal_gap(np.array([0, 90, 180, 270])) == pytest.approx(90)
    assert _azimuthal_gap(np.array([10, 20, 30])) == pytest.approx(340)
    assert _azimuthal_gap(np.array([350, 10])) == pytest.approx(340)
    assert _azimuthal_gap(np.array([45])) == 360.0
    assert _azimuthal_gap(np.array([])) == 360.0


# --- integration on the real data ----------------------------------------------------

_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_data():
    from osv.data.quakeml import read_catalog
    from osv.data.stations import read_stations

    cat = read_catalog(_CAT)
    st = read_stations(_FILES)
    r = associate_picks(cat, st)
    p = r.picks
    assert len(p) == 2940
    assert (p.phase == "P").sum() == 2231
    assert (p.phase == "S").sum() == 709
    assert p[["net", "sta"]].drop_duplicates().shape[0] == 63
    assert p.alias.notna().all()
    assert len(r.events) == 79
    assert r.events.n_P.sum() == 2231
