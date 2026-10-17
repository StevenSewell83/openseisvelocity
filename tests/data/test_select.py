"""Tests for osv.data.select."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import OSVConfig
from osv.data.associate import AssociatedPicks, associate_picks
from osv.data.select import (
    assign_velest_weight,
    load_geojson_region,
    points_in_region,
    select_data,
    split_events,
)

REPO = Path(__file__).resolve().parents[2]
T0 = pd.Timestamp("2017-06-01", tz="UTC")
AZ = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}


def _cfg(**ds) -> OSVConfig:
    base = {"min_p": 1, "max_azgap": 360.0, "validation_fraction": 0.0, "depth_km": [-1, 10]}
    base.update(ds)
    return OSVConfig.model_validate({"data_selection": base})


def _picks(eid, stations="NESW", phases="P", epi=5.0, sigma=0.02, mag_az=True):
    rows = []
    for s in stations:
        for ph in phases:
            rows.append(
                {
                    "event_id": eid,
                    "pick_id": f"{eid}.{s}.{ph}",
                    "net": "NZ",
                    "sta": s,
                    "phase": ph,
                    "sigma_s": sigma,
                    "epicentral_km": epi,
                    "azimuth_deg": AZ[s],
                }
            )
    return rows


def _ev(eid, lat=0.0, lon=0.0, depth=2.0, rms=0.1, mag=-6.19, mag_valid=False):
    return {
        "event_id": eid,
        "lat": lat,
        "lon": lon,
        "depth_km": depth,
        "rms": rms,
        "mag": mag,
        "mag_valid": mag_valid,
    }


def _assoc(events, picks) -> AssociatedPicks:
    return AssociatedPicks(picks=pd.DataFrame(picks), events=pd.DataFrame(events))


def _reasons(sel, level="event"):
    r = sel.rejections
    r = r[r.level == level]
    return {k: sorted(v.reason) for k, v in r.groupby("id")}


def test_clean_event_passes():
    sel = select_data(_assoc([_ev("e1")], _picks("e1")), _cfg())
    assert list(sel.events.event_id) == ["e1"]
    assert len(sel.picks) == 4 and sel.rejections.empty
    assert sel.summary["events_selected"] == 1
    assert (sel.picks.velest_weight == 0).all()
    assert (sel.split == "calibration").all()
    assert list(sel.events.columns).count("split") == 1


def test_epicentral_distance_pick_filter_and_recount():
    picks = _picks("e1", "NES") + _picks("e1", "W", epi=30.0)
    sel = select_data(_assoc([_ev("e1")], picks), _cfg(max_epi_dist_km=25.0))
    assert len(sel.picks) == 3
    r = sel.rejections
    assert list(r.level) == ["pick"] and r.reason.iloc[0] == "epicentral_distance"
    assert r.id.iloc[0] == "e1.W.P" and r.value.iloc[0] == 30.0 and r.threshold.iloc[0] == 25.0
    ev = sel.events.iloc[0]
    assert ev.n_P == 3 and ev.n_stations == 3
    assert ev.azgap_deg == pytest.approx(180.0)  # N, E, S remain


def test_gap_recomputed_after_pick_removal_rejects_event():
    picks = _picks("e1", "NES") + _picks("e1", "W", epi=30.0)
    sel = select_data(_assoc([_ev("e1")], picks), _cfg(max_azgap=100.0))
    assert sel.events.empty
    assert _reasons(sel) == {"e1": ["max_azgap"]}


def test_zero_weight_and_outliers():
    picks = _picks("e1")
    picks[0]["sigma_s"] = np.inf
    a = _assoc([_ev("e1")], picks)
    flags = pd.Series(False, index=a.picks.index)
    flags.iloc[1] = True
    sel = select_data(a, _cfg(), outlier_flags=flags)
    pr = sel.rejections[sel.rejections.level == "pick"].set_index("id").reason
    assert pr["e1.N.P"] == "zero_weight" and pr["e1.E.P"] == "outlier"
    assert len(sel.picks) == 2 and sel.events.iloc[0].n_P == 2
    assert sel.summary["pick_filters"] == {
        "epicentral_distance": 0,
        "zero_weight": 1,
        "outlier": 1,
    }


def test_pick_with_multiple_reasons():
    picks = _picks("e1", "NESW")
    picks[0]["sigma_s"] = np.inf
    picks[0]["epicentral_km"] = 99.0
    sel = select_data(_assoc([_ev("e1")], picks), _cfg())
    r = sel.rejections[sel.rejections.id == "e1.N.P"]
    assert sorted(r.reason) == ["epicentral_distance", "zero_weight"]


@pytest.mark.parametrize(
    ("ev_kw", "ds_kw", "reason"),
    [
        ({"depth": 12.0}, {}, "depth"),
        ({"depth": -3.0}, {}, "depth"),
        ({"rms": 0.5}, {"max_rms": 0.2}, "max_rms"),
        ({}, {"min_p": 5}, "min_p"),
        ({}, {"min_s": 1}, "min_s"),
        ({"lat": 5.0}, {"region_polygon": [[-1, -1], [-1, 1], [1, 1], [1, -1]]}, "region"),
    ],
)
def test_each_event_filter(ev_kw, ds_kw, reason):
    sel = select_data(_assoc([_ev("e1", **ev_kw)], _picks("e1")), _cfg(**ds_kw))
    assert sel.events.empty and sel.picks.empty
    assert _reasons(sel) == {"e1": [reason]}
    assert sel.summary["event_filters"][reason] == 1


def test_region_polygon_config_is_lat_lon():
    poly = [[-1, 10], [-1, 12], [1, 12], [1, 10]]  # lat, lon
    sel = select_data(
        _assoc([_ev("in", lon=11.0), _ev("out", lon=0.0)], _picks("in") + _picks("out")),
        _cfg(region_polygon=poly),
    )
    assert list(sel.events.event_id) == ["in"]


def test_azgap_filter_default_events():
    picks = _picks("e1", "NE")
    sel = select_data(_assoc([_ev("e1")], picks), _cfg(max_azgap=180.0))
    assert _reasons(sel) == {"e1": ["max_azgap"]}


def test_multiple_reasons_recorded():
    ev = _ev("e1", depth=20.0, rms=0.9)
    sel = select_data(_assoc([ev], _picks("e1")), _cfg(min_p=10, min_s=2, max_rms=0.2))
    assert _reasons(sel) == {"e1": ["depth", "max_rms", "min_p", "min_s"]}
    r = sel.rejections.set_index("reason")
    assert r.loc["max_rms", "value"] == 0.9 and r.loc["max_rms", "threshold"] == 0.2
    assert r.loc["min_p", "value"] == 4 and r.loc["min_p", "threshold"] == 10


def test_magnitude_filter_skipped_when_invalid():
    sel = select_data(_assoc([_ev("e1")], _picks("e1")), _cfg(magnitude={"min": 1.0}))
    assert len(sel.events) == 1
    assert any(f.code == "magnitude_filter_skipped" for f in sel.findings)
    assert sel.summary["magnitude_filter"] == "skipped"


def test_magnitude_filter_applied_when_valid():
    evs = [_ev("lo", mag=0.5, mag_valid=True), _ev("hi", mag=2.0, mag_valid=True)]
    sel = select_data(_assoc(evs, _picks("lo") + _picks("hi")), _cfg(magnitude={"min": 1.0}))
    assert list(sel.events.event_id) == ["hi"]
    assert _reasons(sel) == {"lo": ["magnitude"]}
    # not skipped but ignore_if_invalid False and all invalid -> applied, all fail
    sel = select_data(
        _assoc([_ev("e1")], _picks("e1")),
        _cfg(magnitude={"min": 1.0, "ignore_if_invalid": False}),
    )
    assert sel.events.empty
    assert any(f.code == "no_events_selected" for f in sel.findings)


def test_depth_uses_bsl_column():
    ev = _ev("e1", depth=2.0)
    ev["depth_km_bsl"] = 12.0
    sel = select_data(_assoc([ev], _picks("e1")), _cfg())
    assert _reasons(sel) == {"e1": ["depth"]}


def test_picks_of_rejected_events_dropped():
    evs = [_ev("ok"), _ev("bad", rms=1.0)]
    sel = select_data(_assoc(evs, _picks("ok") + _picks("bad")), _cfg())
    assert set(sel.picks.event_id) == {"ok"}
    assert sel.summary["picks_rejected_with_events"] == 4


def test_weight_mapping_boundaries():
    m = {0.02: 0, 0.05: 1, 0.10: 2, 0.20: 3}
    s = [0.0, 0.02, 0.0200001, 0.05, 0.0500001, 0.10, 0.1000001, 0.20, 0.2000001, 5.0]
    assert list(assign_velest_weight(s, m)) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert list(assign_velest_weight([np.inf, np.nan], m)) == [4, 4]
    # float noise from lower/upper bound subtraction stays in class
    assert assign_velest_weight([0.3 - 0.25], m)[0] == 1


def test_weight_class_4_finding_and_sigma_kept():
    picks = _picks("e1", sigma=0.5)
    sel = select_data(_assoc([_ev("e1")], picks), _cfg())
    assert (sel.picks.velest_weight == 4).all()
    assert (sel.picks.sigma_s == 0.5).all()
    assert sel.summary["weight_classes"][4] == 4
    assert any(f.code == "weight_class_4_picks" for f in sel.findings)


def _many(n=40):
    rng = np.random.default_rng(0)
    evs = [_ev(f"e{i:02d}", depth=float(d)) for i, d in enumerate(rng.uniform(0, 6, n))]
    return pd.DataFrame(evs)


def test_split_fraction_reproducible_and_stratified():
    ev = _many(40)
    s1 = split_events(ev, ev.depth_km, 0.25, 7)
    s2 = split_events(ev, ev.depth_km, 0.25, 7)
    s3 = split_events(ev, ev.depth_km, 0.25, 8)
    assert s1.equals(s2) and not s1.equals(s3)
    assert (s1 == "validation").sum() == 10
    d = ev.set_index("event_id").depth_km
    val = d[s1[s1 == "validation"].index]
    q = pd.qcut(d, 4, labels=False)  # validation spans every depth quartile
    assert set(q[val.index]) == {0, 1, 2, 3}
    assert val.min() < d.quantile(0.2) and val.max() > d.quantile(0.8)


def test_split_zero_fraction_and_tiny():
    ev = _many(5)
    assert (split_events(ev, ev.depth_km, 0.0, 1) == "calibration").all()
    one = split_events(ev.iloc[:1], ev.depth_km.iloc[:1], 0.5, 1)
    assert (one == "calibration").all()
    s = split_events(ev, ev.depth_km, 0.15, 1)  # round(0.75)=1 -> at least one
    assert (s == "validation").sum() == 1


def test_select_data_split_uses_seed():
    evs = [_ev(f"e{i}", depth=float(i) / 4) for i in range(20)]
    picks = [p for e in evs for p in _picks(e["event_id"])]
    cfg = _cfg(validation_fraction=0.2)
    a = select_data(_assoc(evs, picks), cfg)
    b = select_data(_assoc(evs, picks), cfg)
    assert a.split.equals(b.split)
    assert a.summary["n_validation"] == 4 and a.summary["n_calibration"] == 16
    assert set(a.events.split) == {"calibration", "validation"}
    cfg2 = OSVConfig.model_validate(
        {
            "project": {"random_seed": 1},
            "data_selection": {
                "min_p": 1,
                "max_azgap": 360,
                "validation_fraction": 0.2,
                "depth_km": [-1, 10],
            },
        }
    )
    assert not select_data(_assoc(evs, picks), cfg2).split.equals(a.split)


# --- polygons -----------------------------------------------------------------------------

SQUARE_WITH_HOLE = {
    "type": "Polygon",
    "coordinates": [
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
    ],
}


def test_polygon_with_hole(tmp_path):
    f = tmp_path / "r.geojson"
    f.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": SQUARE_WITH_HOLE}))
    region = load_geojson_region(f)
    lon = np.array([1.0, 5.0, 9.0, 11.0, np.nan])
    lat = np.array([1.0, 5.0, 9.0, 5.0, 5.0])
    assert list(points_in_region(lon, lat, region)) == [True, False, True, False, False]

    evs = [_ev("a", lon=1, lat=1), _ev("hole", lon=5, lat=5), _ev("out", lon=20, lat=5)]
    picks = [p for e in evs for p in _picks(e["event_id"])]
    sel = select_data(_assoc(evs, picks), _cfg(region_geojson=str(f)))
    assert list(sel.events.event_id) == ["a"]
    assert _reasons(sel) == {"hole": ["region"], "out": ["region"]}


def test_multipolygon_and_collection(tmp_path):
    mp = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]],
        ],
    }
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"type": "FeatureCollection", "features": [{"geometry": mp}]}))
    region = load_geojson_region(f)
    inside = points_in_region(np.array([0.5, 5.5, 3.0]), np.array([0.5, 5.5, 3.0]), region)
    assert list(inside) == [True, True, False]
    f.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
    with pytest.raises(ValueError, match="Polygon"):
        load_geojson_region(f)
    with pytest.raises(FileNotFoundError):
        load_geojson_region(tmp_path / "nope.json")


def test_config_region_geojson_key():
    cfg = OSVConfig.model_validate({"data_selection": {"region_geojson": "x.geojson"}})
    assert cfg.data_selection.region_geojson == Path("x.geojson")
    assert OSVConfig().data_selection.region_geojson is None


# --- integration on the real data ----------------------------------------------------------

_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_data_defaults():
    from osv.data.quakeml import read_catalog
    from osv.data.stations import read_stations

    assoc = associate_picks(read_catalog(_CAT), read_stations(_FILES))
    cfg = OSVConfig()
    sel = select_data(assoc, cfg)
    print("\nSUMMARY", json.dumps(sel.summary, indent=1, default=str))
    print(sel.rejections.groupby(["level", "reason"]).size())
    s = sel.summary
    assert s["events_in"] == 79
    assert s["events_selected"] == len(sel.events) > 0
    assert len(sel.picks) == s["picks_selected"]
    assert sel.picks.velest_weight.between(0, 4).all()
    assert s["n_calibration"] + s["n_validation"] == len(sel.events)
    # every rejected event has at least one reason
    rejected = set(assoc.events.event_id) - set(sel.events.event_id)
    ev_rej = set(sel.rejections[sel.rejections.level == "event"].id)
    assert rejected == ev_rej
