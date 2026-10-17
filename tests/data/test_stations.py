"""Tests for osv.data.stations."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Site, Station

from osv.config.schema import CoordinatesConfig
from osv.data.stations import build_aliases, read_stations, station_at

REPO = Path(__file__).resolve().parents[2]
T0 = UTCDateTime(2015, 1, 1)
T1 = UTCDateTime(2018, 1, 1)


def _cha(code="EHZ", loc="", lat=-38.5, lon=176.2, elev=400.0, depth=0.0, **kw):
    return Channel(
        code=code,
        location_code=loc,
        latitude=lat,
        longitude=lon,
        elevation=elev,
        depth=depth,
        azimuth=0.0,
        dip=-90.0,
        sample_rate=100.0,
        start_date=kw.pop("start", T0),
        end_date=kw.pop("end", None),
    )


def _sta(code, lat=-38.5, lon=176.2, elev=400.0, start=T0, end=None, channels=None):
    return Station(
        code=code,
        latitude=lat,
        longitude=lon,
        elevation=elev,
        site=Site(name=code),
        start_date=start,
        end_date=end,
        channels=channels if channels is not None else [_cha(lat=lat, lon=lon, elev=elev)],
    )


def _write(tmp_path: Path, name: str, nets: dict[str, list[Station]]) -> Path:
    inv = Inventory(networks=[Network(code=c, stations=s) for c, s in nets.items()], source="t")
    p = tmp_path / name
    inv.write(str(p), format="STATIONXML")
    return p


def test_basic_and_borehole(tmp_path):
    bh = _sta(
        "NS12",
        elev=500.0,
        channels=[_cha(elev=500.0, depth=514.0), _cha("EHN", elev=500.0, depth=514.0)],
    )
    p = _write(tmp_path, "a.xml", {"NZ": [_sta("ABCD"), bh]})
    t = read_stations([p])
    assert list(t.stations.sta) == ["ABCD", "NS12"]
    r = t.stations.set_index("sta")
    assert r.loc["NS12", "sensor_depth_m"] == 514.0
    assert r.loc["NS12", "sensor_elevation_m"] == pytest.approx(-14.0)
    assert r.loc["ABCD", "sensor_elevation_m"] == 400.0
    assert r.loc["NS12", "channels"] == ["EHN", "EHZ"]
    assert r.loc["NS12", "sample_rates"] == [100.0]
    assert len(t.channels) == 3
    assert t.finding("borehole_sensors").details["depths_m"] == {"NZ.NS12": 514.0}
    assert t.finding("channel_coordinates_inconsistent") is None

    t2 = read_stations([p], coords_cfg=CoordinatesConfig(borehole_sensors="ignore_depth"))
    r2 = t2.stations.set_index("sta")
    assert r2.loc["NS12", "sensor_elevation_m"] == 500.0
    assert r2.loc["NS12", "sensor_depth_m"] == 514.0


def test_channel_depths_differ_uses_max(tmp_path):
    s = _sta("XYZ1", channels=[_cha("EHZ", depth=0.0), _cha("EHN", depth=80.0)])
    t = read_stations([_write(tmp_path, "a.xml", {"NZ": [s]})])
    assert t.stations.sensor_depth_m.iloc[0] == 80.0
    assert t.finding("channel_depths_differ").count == 1


def test_epochs_moved_and_lookup(tmp_path):
    e1 = _sta("MOVE", start=T0, end=UTCDateTime(2016, 1, 1))
    e2 = _sta("MOVE", lat=-38.51, start=UTCDateTime(2016, 1, 1) + 1, end=None)
    same = _sta("SAME", start=T0, end=UTCDateTime(2016, 1, 1))
    same2 = _sta("SAME", start=UTCDateTime(2016, 1, 1) + 1)
    p = _write(tmp_path, "a.xml", {"NZ": [e1, e2, same, same2]})
    t = read_stations([p, p])  # repeated file must not duplicate epochs
    assert len(t.stations) == 4
    f = t.finding("station_moved")
    assert f.count == 1 and f.details["epochs"][0]["station"] == "NZ.MOVE"
    assert f.details["epochs"][0]["moved_m"] > 1000

    times = ["2015-06-01", "2017-06-01", "2014-01-01", "2017-06-01", "2015-06-01"]
    got = station_at(t.stations, ["NZ"] * 5, ["MOVE", "MOVE", "MOVE", "NOPE", "SAME"], times)
    assert list(got.found) == [True, True, False, False, True]
    assert got.loc[0, "lat"] == pytest.approx(-38.5)
    assert got.loc[1, "lat"] == pytest.approx(-38.51)
    assert np.isnan(got.loc[2, "lat"])
    # net omitted matches any network
    got2 = t.station_at(None, ["MOVE"], ["2017-01-01"])
    assert got2.found.all()
    idx = pd.Timestamp("2016-01-01", tz="UTC")  # boundary inclusive of first epoch
    assert station_at(t.stations, ["NZ"], ["MOVE"], [idx]).found.all()


def test_missing_elevation_and_long_codes(tmp_path):
    p = _write(
        tmp_path,
        "a.xml",
        {"NZ": [_sta("ZERO", elev=0.0), _sta("LONGCODE1"), _sta("LONGCODE2")]},
    )
    t = read_stations([p])
    assert t.finding("elevation_missing_or_zero").details["stations"] == ["NZ.ZERO"]
    assert t.finding("station_code_too_long").count == 2
    assert t.finding("station_code_truncation_clash").details["groups"] == {
        "LONG": ["LONGCODE1", "LONGCODE2"]
    }
    al = t.aliases.set_index("sta").alias
    assert al["ZERO"] == "ZERO"
    assert al["LONGCODE1"] == "LONG"
    assert al["LONGCODE2"] != "LONG" and len(al["LONGCODE2"]) == 4
    assert t.stations.set_index("sta").alias["LONGCODE2"] == al["LONGCODE2"]


def test_duplicate_codes_across_networks_and_inconsistent_channels(tmp_path):
    bad = _sta("DUPE", channels=[_cha(lat=-38.5, elev=400.0), _cha("EHN", lat=-38.6)])
    p = _write(tmp_path, "a.xml", {"NZ": [_sta("DUPE"), _sta("OK01")], "XX": [bad]})
    t = read_stations([p])
    assert t.finding("duplicate_station_codes").details["codes"] == {"DUPE": ["NZ", "XX"]}
    assert t.finding("channel_coordinates_inconsistent").count == 1
    al = t.aliases
    assert al.alias.str.upper().is_unique and (al.alias.str.len() <= 4).all()
    assert al.set_index("sta").alias["OK01"] == "OK01"


def test_alias_collisions_deterministic():
    pairs = pd.DataFrame(
        {
            "net": ["A", "A", "A", "B", "B"],
            "sta": ["ABCDE", "ABCDF", "ABC0", "ABC1", "abcd"],
        }
    )
    a1 = build_aliases(pairs)
    a2 = build_aliases(pairs.iloc[::-1])
    pd.testing.assert_frame_equal(a1, a2)
    assert a1.alias.str.upper().is_unique and (a1.alias.str.len() == 4).all()
    d = dict(zip(a1.sta, a1.alias, strict=True))
    assert d["ABC0"] == "ABC0" and d["ABC1"] == "ABC1"
    assert d["ABCDE"] not in ("ABC0", "ABC1")


def test_aliases_user_csv_and_dict(tmp_path):
    p = _write(tmp_path, "a.xml", {"NZ": [_sta("LONGCODE"), _sta("OK01")]})
    csv = tmp_path / "user.csv"
    csv.write_text("code,alias\nLONGCODE,LC01\n")
    t = read_stations([p], aliases=csv)
    assert t.aliases.set_index("sta").alias.to_dict() == {"LONGCODE": "LC01", "OK01": "OK01"}
    t = read_stations([p], aliases={"NZ.LONGCODE": "LC02"})
    assert t.aliases.set_index("sta").alias["LONGCODE"] == "LC02"
    with pytest.raises(ValueError, match="1-4"):
        read_stations([p], aliases={"LONGCODE": "TOOLONG"})
    with pytest.raises(ValueError, match="unique"):
        read_stations([p], aliases={"LONGCODE": "LC01", "OK01": "lc01"})
    # an automatic alias steps around a user alias that took its code
    t = read_stations([p], aliases={"LONGCODE": "OK01"})
    assert t.aliases.set_index("sta").alias["OK01"] != "OK01"


def test_write_and_roundtrip_csv(tmp_path):
    p = _write(tmp_path, "a.xml", {"NZ": [_sta("LONGCODE"), _sta("OK01")]})
    t = read_stations([p])
    out = t.write(tmp_path / "out")
    assert out["stations"].exists() and out["channels"].exists()
    csv = t.write_aliases_csv(tmp_path / "aliases.csv")
    again = read_stations([p], aliases=csv)
    assert again.alias_map() == t.alias_map()
    assert pd.read_parquet(out["stations"]).shape[0] == 2


def test_no_paths_raises():
    with pytest.raises(ValueError, match="StationXML"):
        read_stations([])


# --- integration on the real station directory ---------------------------------------

_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []


@pytest.mark.integration
@pytest.mark.skipif(not _FILES, reason="real StationXML directory not available")
def test_real_station_directory():
    t = read_stations(_FILES)
    s = t.stations
    r = s.drop_duplicates(["net", "sta"]).set_index("sta")
    assert len(r) >= 63
    assert s.elevation_m.between(250, 590).all()
    bh = {k: v for k, v in r.sensor_depth_m.items() if v > 0}
    assert bh == {"NS12": 514.0, "NS13": 250.0, "NS14": 250.0, "THQ2": 80.0}
    assert t.aliases.alias.str.upper().is_unique
    assert (t.aliases.alias.str.len() <= 4).all()

    cat = REPO / "manual_only_catalog.xml"
    if cat.exists():
        from obspy import read_events

        ev = read_events(str(cat))
        keys, times = [], []
        for e in ev:
            for pk in e.picks:
                w = pk.waveform_id
                keys.append((w.network_code, w.station_code))
                times.append(pk.time.datetime)
        got = station_at(
            s, [k[0] for k in keys], [k[1] for k in keys], pd.to_datetime(times, utc=True)
        )
        assert got.found.all()
        assert len(set(keys)) == 63
