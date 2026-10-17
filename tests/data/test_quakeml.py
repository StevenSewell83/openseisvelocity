"""Tests for osv.data.quakeml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.core.event import (
    Arrival,
    Catalog,
    Comment,
    CreationInfo,
    Event,
    Magnitude,
    Origin,
    Pick,
    QuantityError,
    ResourceIdentifier,
    WaveformStreamID,
)

from osv.config.schema import DataSelectionConfig, InputsConfig
from osv.data.quakeml import normalise_phase, parse_comment_weight, read_catalog

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "manual_only_catalog.xml"
T0 = UTCDateTime(2017, 1, 1)


def _pick(eid, off, hint="P", sta="AAAA", cha="EHZ", **kw):
    return Pick(
        resource_id=ResourceIdentifier(f"smi:local/t/pick/{eid}"),
        time=T0 + off,
        waveform_id=WaveformStreamID("NZ", sta, "10", cha),
        phase_hint=hint,
        **kw,
    )


def _origin(oid, dt=0.0, **kw):
    return Origin(
        resource_id=ResourceIdentifier(f"smi:local/t/origin/{oid}"),
        time=T0 + dt,
        latitude=-38.5,
        longitude=176.2,
        depth=2500.0,
        **kw,
    )


def _write(tmp_path, events, name="cat.xml"):
    p = tmp_path / name
    Catalog(events=events).write(str(p), format="QUAKEML")
    return p


def _simple_event(n=0, mag=1.0, **evkw):
    o = _origin(f"o{n}")
    picks = [_pick(f"{n}a", 2.0, "P"), _pick(f"{n}b", 3.5, "S", sta="BBBB", cha="EH1")]
    ev = Event(
        resource_id=ResourceIdentifier(f"smi:local/t/ev{n}"),
        origins=[o],
        picks=picks,
        magnitudes=[Magnitude(mag=mag, magnitude_type="ML", origin_id=o.resource_id)],
        preferred_origin_id=o.resource_id,
        **evkw,
    )
    return ev, o, picks


def _cfg(**kw):
    return InputsConfig(**kw)


def _codes(res):
    return {f.code for f in res.findings}


# --------------------------------------------------------------------------- helpers


def test_normalise_phase():
    assert [normalise_phase(x) for x in ("P", "Pg", "s", "Sn", "x", "", None)] == [
        "P", "P", "S", "S", None, None, None,
    ]  # fmt: skip


def test_parse_comment_weight():
    assert parse_comment_weight([Comment(text="wgt = 0.8")]) == 0.8
    assert parse_comment_weight([Comment(text="foo"), Comment(text="WGT=1")]) == 1.0
    assert np.isnan(parse_comment_weight([Comment(text="nothing")]))
    assert np.isnan(parse_comment_weight([]))


# --------------------------------------------------------------------------- real file


@pytest.fixture(scope="module")
def real():
    return read_catalog(CATALOG, InputsConfig())


def test_real_counts(real):
    assert len(real.events) == 79
    assert len(real.picks) == 2940
    assert (real.picks.phase == "P").sum() == 2231
    assert (real.picks.phase == "S").sum() == 709
    assert str(real.events.time.dtype) == "datetime64[ns, UTC]"
    assert str(real.picks.time.dtype) == "datetime64[ns, UTC]"


def test_real_findings(real):
    f = {x.code: x for x in real.findings}
    assert f["duplicate_arrivals"].count == 2880
    assert f["duplicate_arrivals"].details["n_events"] == 79
    assert f["unreferenced_picks"].count == 2861
    assert f["asymmetric_uncertainty"].count == 546
    assert f["eval_mode_unset"].count == 2940
    assert f["invalid_magnitudes"].severity == "warn"
    assert f["invalid_magnitudes"].details["identical"] is True
    assert not real.events.mag_valid.any()
    assert "missing_uncertainty" not in f
    assert "unknown_phase" not in f


def test_real_values(real):
    assert set(real.picks.comment_weight.unique()) == {0.8, 1.0}
    assert real.picks.unc_s.notna().all()
    assert real.picks.travel_time_s.between(0, 10).all()
    assert real.events.depth_km.between(0.1, 6).all()
    assert real.events.lat.lt(0).all()
    assert set(real.picks.event_id) == set(real.events.event_id)


def test_real_use_arrivals():
    res = read_catalog(CATALOG, _cfg(use_arrivals=True))
    assert len(res.picks) == 79  # one referenced pick per event
    assert (res.picks.phase == "S").sum() == 78


def test_real_write_parquet(real, tmp_path):
    paths = real.write(tmp_path / "out")
    assert len(pd.read_parquet(paths["events"])) == 79
    assert len(pd.read_parquet(paths["picks"])) == 2940
    assert paths["findings"].exists()


# --------------------------------------------------------------------------- synthetic


def test_basic_synthetic(tmp_path):
    ev, o, picks = _simple_event(mag=2.0)
    o.arrivals = [Arrival(pick_id=p.resource_id, phase=p.phase_hint) for p in picks]
    o.quality = None
    p = _write(tmp_path, [ev])
    res = read_catalog([p], _cfg())
    assert len(res.events) == 1
    row = res.events.iloc[0]
    assert row.depth_km == pytest.approx(2.5)
    assert row.mag == 2.0 and bool(row.mag_valid)
    assert list(res.picks.phase) == ["P", "S"]
    assert res.picks.travel_time_s.tolist() == pytest.approx([2.0, 3.5])
    assert res.picks["loc"].tolist() == ["10", "10"]
    # missing uncertainties reported
    assert res.finding("missing_uncertainty").count == 2
    assert np.isnan(res.picks.unc_s).all()
    assert "duplicate_arrivals" not in _codes(res)
    assert "unreferenced_picks" not in _codes(res)


def test_uncertainty_variants(tmp_path):
    ev, o, picks = _simple_event()
    picks[0].time_errors = QuantityError(uncertainty=0.05)
    picks[1].time_errors = QuantityError(lower_uncertainty=0.02, upper_uncertainty=0.06)
    res = read_catalog(_write(tmp_path, [ev]), _cfg())
    assert res.picks.unc_s.tolist() == pytest.approx([0.05, 0.04])
    assert res.picks.unc_lower[1] == pytest.approx(0.02)
    assert res.finding("asymmetric_uncertainty").count == 1
    assert "missing_uncertainty" not in _codes(res)


def test_duplicate_and_unreferenced(tmp_path):
    ev, o, picks = _simple_event()
    a = Arrival(pick_id=picks[0].resource_id, phase="P")
    o.arrivals = [a, a, a]
    extra = _pick("x", 1.0, "P")
    ev.picks.append(extra)
    p = _write(tmp_path, [ev])
    res = read_catalog(p, _cfg())
    assert res.finding("duplicate_arrivals").count == 2
    assert res.finding("unreferenced_picks").count == 2
    assert len(res.picks) == 3
    res2 = read_catalog(p, _cfg(use_arrivals=True))
    assert len(res2.picks) == 1
    assert res2.finding("unreferenced_picks").severity == "warn"


def test_use_arrivals_phase_from_arrival(tmp_path):
    ev, o, picks = _simple_event()
    o.arrivals = [Arrival(pick_id=picks[0].resource_id, phase="S")]  # contradicts hint P
    p = _write(tmp_path, [ev])
    assert read_catalog(p, _cfg(use_arrivals=True)).picks.phase.tolist() == ["S"]
    assert read_catalog(p, _cfg(use_arrivals=False)).picks.phase.tolist() == ["P", "S"]


def test_arrival_without_pick(tmp_path):
    ev, o, picks = _simple_event()
    o.arrivals = [Arrival(pick_id=ResourceIdentifier("smi:local/t/pick/ghost"), phase="P")]
    res = read_catalog(_write(tmp_path, [ev]), _cfg())
    assert res.finding("arrival_without_pick").count == 1


def test_unknown_phase(tmp_path):
    ev, o, picks = _simple_event()
    picks[0].phase_hint = "amp"
    picks[1].phase_hint = None
    res = read_catalog(_write(tmp_path, [ev]), _cfg())
    assert res.finding("unknown_phase").count == 2
    assert res.picks.phase.isna().all()
    assert res.picks.phase_hint.iloc[0] == "amp"


def test_evaluation_mode_filter(tmp_path):
    ev, o, picks = _simple_event()
    picks[0].evaluation_mode = "manual"
    picks[1].evaluation_mode = "automatic"
    ev.picks.append(_pick("u", 4.0, "P"))  # unset
    p = _write(tmp_path, [ev])
    assert len(read_catalog(p, _cfg(evaluation_mode="any")).picks) == 3
    res = read_catalog(p, _cfg(evaluation_mode="manual"))
    assert sorted(res.picks.evaluation_mode.fillna("unset")) == ["manual", "unset"]
    assert res.finding("eval_mode_unset").severity == "warn"
    assert res.finding("eval_mode_filtered").count == 1
    res = read_catalog(p, _cfg(evaluation_mode="automatic"))
    assert sorted(res.picks.evaluation_mode.fillna("unset")) == ["automatic", "unset"]
    assert read_catalog(p, _cfg()).finding("eval_mode_unset").severity == "info"


def _two_origin_event():
    o1 = _origin(
        "o1",
        0.0,
        creation_info=CreationInfo(creation_time=T0 + 100),
        method_id=ResourceIdentifier("smi:local/t/method/A"),
    )
    o2 = _origin(
        "o2",
        0.5,
        creation_info=CreationInfo(creation_time=T0 + 50),
        earth_model_id=ResourceIdentifier("smi:local/t/model/B"),
    )
    o3 = _origin("o3", 0.2, creation_info=CreationInfo(creation_time=T0 + 10))
    return Event(
        resource_id=ResourceIdentifier("smi:local/t/ev"),
        origins=[o1, o2, o3],
        picks=[_pick("a", 2.0)],
        preferred_origin_id=o2.resource_id,
    )  # fmt: skip


def test_origin_selection(tmp_path):
    p = _write(tmp_path, [_two_origin_event()])
    pref = read_catalog(p, _cfg(origin_selection="preferred"))
    assert pref.events.origin_id[0].endswith("o2")
    assert pref.picks.travel_time_s[0] == pytest.approx(1.5)
    assert pref.finding("multiple_origins").count == 1
    lat = read_catalog(p, _cfg(origin_selection="latest"))
    assert lat.events.origin_id[0].endswith("o1")
    assert lat.picks.travel_time_s[0] == pytest.approx(2.0)
    bm = read_catalog(p, _cfg(origin_selection="by_method:method/A"))
    assert bm.events.origin_id[0].endswith("o1")
    bm = read_catalog(p, _cfg(origin_selection="by_method:model/B"))
    assert bm.events.origin_id[0].endswith("o2")
    none = read_catalog(p, _cfg(origin_selection="by_method:zzz"))
    assert len(none.events) == 0 and none.finding("origin_method_not_found").count == 1


def test_preferred_fallback_to_last(tmp_path):
    ev = _two_origin_event()
    ev.preferred_origin_id = None
    res = read_catalog(_write(tmp_path, [ev]), _cfg())
    assert res.events.origin_id[0].endswith("o3")
    assert res.finding("no_preferred_origin").severity == "warn"


def test_bad_origin_selection_raises(tmp_path):
    p = _write(tmp_path, [_simple_event()[0]])
    cfg = InputsConfig.model_construct(origin_selection="bogus")
    with pytest.raises(ValueError, match="origin_selection"):
        read_catalog(p, cfg)


def test_magnitude_validity(tmp_path):
    evs = [_simple_event(i, mag=m)[0] for i, m in enumerate([1.0, 2.0, -5.0, 12.0])]
    p = _write(tmp_path, evs)
    res = read_catalog(p, _cfg())
    assert res.events.mag_valid.tolist() == [True, True, False, False]
    assert res.finding("invalid_magnitudes").count == 2
    res = read_catalog(p, _cfg(), DataSelectionConfig(magnitude={"ignore_if_invalid": False}))
    assert res.events.mag_valid.all()
    assert res.finding("invalid_magnitudes").severity == "warn"


def test_identical_magnitudes(tmp_path):
    evs = [_simple_event(i, mag=1.5)[0] for i in range(3)]
    res = read_catalog(_write(tmp_path, evs), _cfg())
    assert not res.events.mag_valid.any()
    assert res.finding("invalid_magnitudes").details["identical"] is True


def test_valid_magnitudes_no_finding(tmp_path):
    evs = [_simple_event(i, mag=1.0 + i)[0] for i in range(3)]
    res = read_catalog(_write(tmp_path, evs), _cfg())
    assert res.events.mag_valid.all()
    assert "invalid_magnitudes" not in _codes(res)


def test_comment_weight_in_table(tmp_path):
    ev, o, picks = _simple_event()
    picks[0].comments = [Comment(text="wgt = 0.8")]
    res = read_catalog(_write(tmp_path, [ev]), _cfg())
    assert res.picks.comment_weight[0] == 0.8
    assert np.isnan(res.picks.comment_weight[1])


def test_multiple_files_and_errors(tmp_path):
    p1 = _write(tmp_path, [_simple_event(1)[0]], "a.xml")
    p2 = _write(tmp_path, [_simple_event(2)[0]], "b.xml")
    res = read_catalog([p1, p2], _cfg())
    assert len(res.events) == 2
    with pytest.raises(FileNotFoundError):
        read_catalog([tmp_path / "nope.xml"], _cfg())
    with pytest.raises(ValueError, match="quakeml"):
        read_catalog([], _cfg())
    # paths default to config
    assert len(read_catalog(None, _cfg(quakeml=[p1])).events) == 1
