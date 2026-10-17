"""Tests for osv.io.fmtomo.sources and .times (sources.in, receivers.in, otimes.dat, ...)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.io.fmtomo import sources as s
from osv.io.fmtomo import times as t
from osv.io.fmtomo._fortran import FmtomoFormatError

DATA = Path(__file__).resolve().parents[1] / "data" / "fmtomo"
E1 = DATA / "example1"


def test_sources_example1_local_with_reflection_path():
    src = s.read_sources(E1 / "sources.in")
    assert len(src) == 5
    s0 = src[0]
    assert not s0.teleseismic and s0.phase is None
    # file record is "depth lat lon"
    assert (s0.depth_km, s0.lat, s0.lon) == (49.1324768, -41.0005379, 141.401596)
    assert len(s0.paths) == 1
    assert s0.paths[0].steps == ((0, 2), (2, 1)) and s0.paths[0].vtypes == (1, 1)
    assert s0.paths[0] == s.PathSpec.reflected(2)
    assert s0.uncertainty is None


def test_sourcesref_has_uncertainties():
    ref = s.read_sources(E1 / "sourcesref.in")
    assert ref[0].uncertainty == (2.0, 2.0, 2.0)
    assert (ref[0].depth_km, ref[0].lat, ref[0].lon) == (60.0, -41.2, 141.2)
    text = s.format_sources(ref)
    assert (
        text.splitlines()[2]
        == "    60.00000   -41.20000   141.20000     2.00000     2.00000     2.00000"
    )


@pytest.mark.parametrize("name", ["sources.in", "sourcesref.in"])
def test_sources_roundtrip(tmp_path, name):
    src = s.read_sources(E1 / name)
    out = s.write_sources(tmp_path / name, src)
    assert s.read_sources(out) == src


def test_sources_teleseismic_example2():
    src = s.read_sources(DATA / "example2" / "sources.in")
    assert len(src) == 21
    assert not src[0].teleseismic
    assert [len(p.paths) for p in src[:2]] == [2, 2]
    assert src[0].paths[1] == s.PathSpec.direct()
    tele = [x for x in src if x.teleseismic]
    assert len(tele) == 12
    assert tele[0].phase == "P"
    assert tele[0].depth_km == pytest.approx(33.0)
    assert tele[0].paths[0].steps == ((3, 2), (2, 1))
    df = s.sources_frame(src)
    assert df.index[0] == 1 and df["teleseismic"].sum() == 12
    text = s.format_sources(src)
    assert s.parse_sources(text) == src


def test_sources_wrapped_path_lists_and_errors():
    # a path with 5 steps wraps over two lines of the file (list-directed read continues)
    p = s.PathSpec.from_sequence([0, 2, 2, 3, 3, 2, 2, 3, 2, 1], [1, 1, 2, 2, 1])
    src = [s.Source(10.0, -41.0, 141.0, [p, s.PathSpec.direct(2)])]
    text = s.format_sources(src)
    assert s.parse_sources(text) == src
    with pytest.raises(FmtomoFormatError, match="path 1"):
        s.parse_sources("1\n 0\n 10 -41 141\n 1\n 2\n 0 2 2 1\n 1 3\n")
    with pytest.raises(FmtomoFormatError, match="even number"):
        s.PathSpec.from_sequence([0, 2, 1], [1])
    with pytest.raises(FmtomoFormatError, match="velocity types"):
        s.PathSpec(((0, 1),), (3,))
    with pytest.raises(FmtomoFormatError, match="phase"):
        s.format_sources([s.Source(1, 2, 3, [], teleseismic=True, phase="WAYTOOLONGPHASE")])
    with pytest.raises(FmtomoFormatError, match="unexpected end of file"):
        s.parse_sources("2\n 0\n 10 -41 141\n 0\n")


def test_sources_extra_id_variant_roundtrip():
    text = "1\n 0\n 0.35 -31.3235 138.89931 7\n 1\n 1\n 0 1\n 1\n"
    src = s.parse_sources(text)
    assert src[0].extra_id == 7 and src[0].uncertainty is None
    assert s.parse_sources(s.format_sources(src)) == src


def test_validate_sources():
    src = [s.Source(10.0, -41.0, 141.0, [s.PathSpec.direct(2), s.PathSpec.reflected(7)])]
    with pytest.raises(FmtomoFormatError) as exc:
        s.validate_sources(src, n_interfaces=3, n_vtypes=1)
    msg = str(exc.value)
    assert "interface number 7" in msg and "velocity type 2" in msg
    s.validate_sources(src, n_interfaces=8, n_vtypes=2)


def test_receivers_example1():
    rec = s.read_receivers(E1 / "receivers.in")
    assert len(rec.positions) == 100
    first = rec.positions.loc[1]
    assert (first.depth_km, first.lat, first.lon) == (-1.0, -41.8, 140.2)
    # every receiver sees all 5 sources, each along path 1
    assert len(rec.rays) == 500
    assert set(rec.rays["path"]) == {1}
    assert list(rec.rays[rec.rays.receiver == 1]["source"]) == [1, 2, 3, 4, 5]
    src = s.read_sources(E1 / "sources.in")
    s.validate_receivers(rec, src)


def test_receivers_roundtrip_example1_and_example2(tmp_path):
    for p in (E1 / "receivers.in", DATA / "example2" / "receivers.in"):
        rec = s.read_receivers(p)
        out = s.write_receivers(tmp_path / p.parent.name, rec)
        back = s.read_receivers(out)
        pd.testing.assert_frame_equal(back.positions, rec.positions)
        pd.testing.assert_frame_equal(back.rays, rec.rays)
    rec2 = s.read_receivers(DATA / "example2" / "receivers.in")
    assert len(rec2.rays) == 6 * 30
    assert rec2.rays["path"].max() == 2  # example2 has sources with two paths


def test_receivers_validation_messages():
    rec = s.read_receivers(E1 / "receivers.in")
    src = s.read_sources(E1 / "sources.in")[:3]
    with pytest.raises(FmtomoFormatError, match="non-existent source 4"):
        s.validate_receivers(rec, src)
    src = s.read_sources(E1 / "sources.in")
    rec.rays.loc[0, "path"] = 2
    with pytest.raises(FmtomoFormatError, match="path 2 of source 1, which has 1 path"):
        s.validate_receivers(rec, src)


def test_receivers_without_rays_roundtrip():
    rec = s.Receivers(
        pd.DataFrame(
            {"depth_km": [-1.0, 0.5], "lat": [-41.0, -41.1], "lon": [140.0, 140.1]},
            index=pd.RangeIndex(1, 3, name="receiver"),
        ),
        pd.DataFrame({"receiver": [2], "source": [1], "path": [1]}),
    )
    back = s.parse_receivers(s.format_receivers(rec))
    pd.testing.assert_frame_equal(back.rays, rec.rays)
    assert len(back.positions) == 2


# ---- times ----------------------------------------------------------------------------


def test_observed_times_example1_and_invalid_flag():
    df = t.read_observed(E1 / "otimes.dat")
    assert len(df) == 500
    r0 = df.iloc[0]
    assert (r0.receiver, r0.source, r0.ray, r0.subray) == (1, 1, 1, 0)
    assert r0.time == 21.649653 and r0.uncertainty == 0.1
    # receiver 6 / source 1 was an invalid pick (-100 plus 0.25 s DC offset)
    bad = df[~df.valid]
    assert len(bad) == 21 and (bad.time < -50).all()
    assert df.loc[5, "time"] == -99.75
    # third column is the ray's position in the receiver's list (here == source id), not the path id
    assert (df["ray"] == df["source"]).all()


def test_observed_times_roundtrip(tmp_path):
    df = t.read_observed(E1 / "otimes.dat")
    out = t.write_observed(tmp_path / "otimes.dat", df)
    back = t.read_observed(out)
    pd.testing.assert_frame_equal(back, df)
    # text round trip of the 4 integer columns / counts
    assert out.read_text().splitlines()[0].split() == ["500"]


def test_arrivals_example1_roundtrip_text_equal(tmp_path):
    for name in ("arrivals.dat", "rtimes.dat"):
        text = (E1 / name).read_text()
        df = t.parse_arrivals(text)
        assert len(df) == 500 and list(df.columns[:7]) == t.ARRIVAL_COLUMNS
        assert df["valid"].sum() == 481  # 19 rays have no valid path (-1) in the example
        assert df["diffracted"].dtype == bool and df["headwave"].dtype == bool
        assert t.format_arrivals(df) == text  # (4i6,f15.6,2l5) is reproduced exactly


def test_arrivals_invalid_and_flags():
    text = (
        "     1     1     1     0      -1.000000    F    F\n"
        "     2     1     1     0      12.500000    T    T\n"
    )
    df = t.parse_arrivals(text)
    assert list(df["valid"]) == [False, True]
    assert list(df["diffracted"]) == [False, True] and list(df["headwave"]) == [False, True]
    assert t.format_arrivals(df) == text
    with pytest.raises(FmtomoFormatError, match="logical"):
        t.parse_arrivals("1 1 1 0 3.0 X F\n")


def test_stimes_sourcederivs_roundtrip(tmp_path):
    st = t.read_stimes(E1 / "stimes.dat")
    assert st.shape == (5,) and st[0] == pytest.approx(-0.287421495)
    np.testing.assert_array_equal(t.read_stimes(t.write_stimes(tmp_path / "a", st)), st)
    ids = t.read_source_ids(E1 / "sourcederivs.in")
    assert list(ids) == [1, 2, 3, 4, 5]
    assert list(t.read_source_ids(t.write_source_ids(tmp_path / "b", ids))) == [1, 2, 3, 4, 5]
    with pytest.raises(FmtomoFormatError):
        t.parse_stimes("3\n 1.0\n 2.0\n")


def test_residuals_dat_example1(tmp_path):
    df = t.read_residuals(E1 / "residuals.dat")
    assert list(df.columns) == ["rms_ms", "variance_s2", "chi2"]
    assert len(df) == 7 and df.index.name == "iteration"
    assert df.loc[0, "rms_ms"] == 2174.77 and df.loc[6, "chi2"] == 9.73203
    assert t.format_residuals(df) == (E1 / "residuals.dat").read_text()
    with pytest.raises(FmtomoFormatError, match="3 values"):
        t.parse_residuals("1.0 2.0\n")


def test_gridsave_roundtrip():
    text = "1 2\n1 2\n1 1\n3 1\n2\n1\n"
    entries = t.parse_gridsave(text)
    assert entries == [t.GridSave(1, (1, 2), (1, 1)), t.GridSave(3, (2,), (1,))]
    assert t.parse_gridsave(t.format_gridsave(entries)) == entries


def test_attach_path_ids_example2():
    """In example 2 a receiver lists (source 1, path 1), (source 1, path 2), ...: ray != path."""
    rec = s.read_receivers(DATA / "example2" / "receivers.in")
    arr = pd.DataFrame(
        {
            "receiver": [1, 1, 2],
            "source": [1, 1, 1],
            "ray": [1, 2, 1],
            "subray": 0,
            "time": [15.0, 5.0, 15.1],
            "diffracted": False,
            "headwave": False,
        }
    )
    out = t.attach_path_ids(arr, rec)
    assert list(out["path"]) == [1, 2, 1]
    arr.loc[1, "source"] = 2
    with pytest.raises(FmtomoFormatError, match="do not match"):
        t.attach_path_ids(arr, rec)
