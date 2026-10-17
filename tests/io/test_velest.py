"""Tests for osv.io.velest (CNV, station, model and control file I/O).

Fixtures in ``tests/data/velest`` are excerpts of the legacy Ngatamariki runs
(``/home/steve/seismic/velest/MonteCarlo`` and ``/MC``) and of a VELEST run on a synthetic
dataset (``synth_*``).  The ``mc_*`` files use the legacy point-mirrored convention.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv import coords
from osv.io.velest import (
    CNV_PICK_COLUMNS,
    CnvData,
    ModelBlock,
    VelestControl,
    VelestFormatError,
    VelestModel,
    format_cnv,
    format_control,
    format_model,
    format_stations,
    parse_cnv,
    parse_control,
    parse_model,
    parse_stations,
    read_cnv,
    read_control,
    read_model,
    read_stations,
    write_cnv,
    write_control,
    write_model,
    write_stations,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "velest"


def noncomment(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.replace("\r", "").split("\n") if not ln.startswith("*")]


def make_events(n: int = 1, **over: object) -> pd.DataFrame:
    rows = []
    for k in range(n):
        rows.append(
            {
                "event": k,
                "time": pd.Timestamp("2019-03-04 05:06:07.89") + pd.Timedelta(minutes=k),
                "lat": -38.5 - 0.01 * k,
                "lon": 176.2,
                "depth_km": 2.5,
                "mag": 1.2,
                "gap": pd.NA,
                "rms": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["gap"] = pd.array(df["gap"], dtype="Int64")
    for k, v in over.items():
        df[k] = v
    return df


def make_picks(counts: list[int]) -> pd.DataFrame:
    rows = []
    for ev, n in enumerate(counts):
        for i in range(n):
            rows.append(
                (ev, f"S{i:03d}", "P" if i % 2 == 0 else "S", i % 5, round(1.0 + 0.37 * i, 2))
            )
    return pd.DataFrame(rows, columns=CNV_PICK_COLUMNS)


def fortran_header(line: str) -> dict[str, float | str]:
    """Reference reader: the fixed columns of INPUTCNV (velest.f:3447-3451), 0-based."""
    p = line.ljust(80)
    return {
        "yy": int(p[0:2]),
        "mo": int(p[2:4]),
        "dd": int(p[4:6]),
        "hh": int(p[7:9]),
        "mi": int(p[9:11]),
        "sec": float(p[12:17]),
        "lat": float(p[18:25]),
        "ns": p[25],
        "lon": float(p[27:35]),
        "ew": p[35],
        "depth": float(p[37:44]),
        "mag": float(p[46:51]),
    }


# ----------------------------------------------------------------------------------------
# CNV
# ----------------------------------------------------------------------------------------


def test_cnv_velest_output_is_byte_identical_roundtrip() -> None:
    path = DATA / "mc_velout.cnv"
    data = read_cnv(path, allow_negative_tt=True)
    assert len(data.events) == 25
    assert data.picks["tt"].min() < 0  # VELEST output holds negative times
    text = format_cnv(data.events, data.picks, allow_negative_tt=True)
    assert text == path.read_text()


def test_cnv_negative_times_are_an_error_by_default() -> None:
    with pytest.raises(VelestFormatError, match="allow_negative_tt"):
        read_cnv(DATA / "mc_velout.cnv")


def test_cnv_output_variant_values() -> None:
    d = read_cnv(DATA / "mc_velout.cnv", allow_negative_tt=True)
    e = d.events.iloc[0]
    assert e["time"] == pd.Timestamp("2012-06-07 00:11:17.49")
    assert (e["lat"], e["lon"], e["depth_km"], e["mag"]) == (-38.6109, 176.2059, 3.34, 1.0)
    assert e["gap"] == 106 and e["rms"] == pytest.approx(0.11)
    first = d.picks_of(0).iloc[0]
    assert (first["sta"], first["phase"], first["weight"], first["tt"]) == ("ARAZ", "P", 0, 2.51)
    assert set(d.picks["phase"]) == {"P", "S"}


@pytest.mark.parametrize("name", ["mc_input_south.cnv", "mc_input_north.cnv", "mc_input_rk.cnv"])
def test_cnv_input_files_roundtrip_parsed_equal(name: str) -> None:
    d = read_cnv(DATA / name, allow_negative_tt=True)
    assert len(d.events) >= 24
    text = format_cnv(d.events, d.picks, allow_negative_tt=True)
    back = parse_cnv(text, allow_negative_tt=True)
    assert back.equals(d)
    assert d.events["gap"].isna().all()
    # second pass is a text fixed point
    assert format_cnv(back.events, back.picks, allow_negative_tt=True) == text


def test_cnv_legacy_input_file_values_and_mirror() -> None:
    raw = read_cnv(DATA / "mc_input_south.cnv")
    e = raw.events.iloc[0]
    assert e["time"] == pd.Timestamp("2012-07-17 06:27:50.48")
    # the file says N / W: returned as is
    assert (e["lat"], e["lon"]) == (38.5621, -176.1913)
    mirrored = read_cnv(DATA / "mc_input_south.cnv", legacy_mirror=True)
    assert (mirrored.events["lat"] == -raw.events["lat"]).all()
    assert (mirrored.events["lon"] == -raw.events["lon"]).all()
    # writing with the same option restores the legacy letters
    again = parse_cnv(format_cnv(mirrored.events, mirrored.picks, legacy_mirror=True))
    assert again.equals(raw)
    assert "38.5621S" not in format_cnv(raw.events, raw.picks)
    assert "38.5621N 176.1913W" in format_cnv(raw.events, raw.picks)


def test_cnv_picks_and_weights_from_fixture() -> None:
    d = read_cnv(DATA / "mc_input_south.cnv")
    p = d.picks_of(0)
    assert len(p) == 11
    assert list(p["sta"][:3]) == ["NS06", "NS09", "NS02"]
    assert list(p["phase"]) == list("PPPPPPPPSSS")
    assert (p["weight"] == 0).all()


def test_cnv_writer_header_matches_velest_fixed_columns() -> None:
    ev = make_events(1, depth_km=-0.75, mag=-1.5)
    text = format_cnv(ev, make_picks([3]))
    h = fortran_header(text.splitlines()[0])
    assert h == {
        "yy": 19,
        "mo": 3,
        "dd": 4,
        "hh": 5,
        "mi": 6,
        "sec": 7.89,
        "lat": 38.5,
        "ns": "S",
        "lon": 176.2,
        "ew": "E",
        "depth": -0.75,
        "mag": -1.5,
    }
    # same via the reader
    d = parse_cnv(text)
    assert d.events.loc[0, "depth_km"] == -0.75 and d.events.loc[0, "mag"] == -1.5


def test_cnv_output_style_layout_for_gap_rms() -> None:
    ev = make_events(1)
    ev["gap"] = pd.array([123], dtype="Int64")
    ev["rms"] = [0.456]
    text = format_cnv(ev, make_picks([2]))
    head = text.splitlines()[0]
    assert head.endswith("   2.50   1.20    123      0.46")
    d = parse_cnv(text)
    assert d.events.loc[0, "gap"] == 123 and d.events.loc[0, "rms"] == pytest.approx(0.46)


@pytest.mark.parametrize("n", [1, 5, 6, 7, 12, 13, 18])
def test_cnv_pick_line_wrap(n: int) -> None:
    ev, pk = make_events(2), make_picks([n, 3])
    text = format_cnv(ev, pk)
    lines = text.split("\n")
    assert lines[-1] == ""  # ends with a newline
    body = lines[1 : 1 + -(-n // 6)]
    assert all(len(ln) == 12 * 6 for ln in body[:-1])
    assert len(body[-1]) == 12 * (n - 6 * (len(body) - 1))
    assert lines[1 + len(body)] == ""  # blank line terminates the event (even when n % 6 == 0)
    back = parse_cnv(text)
    assert back.picks.equals(pk.astype(back.picks.dtypes.to_dict()))
    assert len(back.events) == 2


def test_cnv_missing_blank_after_full_line_is_detected() -> None:
    text = format_cnv(make_events(2), make_picks([6, 2])).replace("\n\n", "\n", 1)
    with pytest.raises(VelestFormatError, match="pick group"):
        parse_cnv(text)


def test_cnv_end_markers_and_blank_handling() -> None:
    good = format_cnv(make_events(2), make_picks([3, 4]))
    assert len(parse_cnv(good).events) == 2
    assert len(parse_cnv(good.rstrip("\n")).events) == 2  # no final newline / blank line
    spaced = "\n \n" + good.replace("\n\n", "\n \n\n\n")  # blank-looking lines are skipped
    assert len(parse_cnv(spaced).events) == 2
    stop = good + "9999\n" + good
    assert len(parse_cnv(stop).events) == 2  # 9999 ends the file (velest.f:3445)
    crlf = good.replace("\n", "\r\n")
    assert parse_cnv(crlf).equals(parse_cnv(good))


def test_cnv_s_picks_and_reflection_phases() -> None:
    pk = pd.DataFrame(
        [(0, "AAAA", "S", 1, 2.0), (0, "BBBB", "M", 0, 3.0), (0, "CCCC", "-", 2, 0.5)],
        columns=CNV_PICK_COLUMNS,
    )
    back = parse_cnv(format_cnv(make_events(1), pk))
    assert list(back.picks["phase"]) == ["S", "M", "-"]
    assert list(back.picks["weight"]) == [1, 0, 2]


def test_cnv_short_station_names_are_padded_and_stripped() -> None:
    pk = pd.DataFrame([(0, "AB", "P", 0, 1.0)], columns=CNV_PICK_COLUMNS)
    text = format_cnv(make_events(1), pk)
    assert text.splitlines()[1] == "AB  P0  1.00"
    assert parse_cnv(text).picks.loc[0, "sta"] == "AB"


@pytest.mark.parametrize(
    ("mut", "msg"),
    [
        (lambda p: p.assign(tt=-0.5), "outside"),
        (lambda p: p.assign(tt=100.0), "outside"),
        (lambda p: p.assign(tt=99.996), "outside"),
        (lambda p: p.assign(tt=np.nan), "outside"),
        (lambda p: p.assign(phase="X"), "phase"),
        (lambda p: p.assign(weight=6), "weight"),
        (lambda p: p.assign(weight=-1), "weight"),
        (lambda p: p.assign(sta="TOOLONG"), "longer than 4"),
        (lambda p: p.assign(sta=" "), "blank"),
    ],
)
def test_cnv_writer_rejects_bad_picks(mut, msg: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(VelestFormatError, match=msg):
        format_cnv(make_events(1), mut(make_picks([2])))


def test_cnv_writer_structural_errors() -> None:
    ev = make_events(1)
    dup = pd.DataFrame([(0, "A", "P", 0, 1.0), (0, "A", "P", 0, 2.0)], columns=CNV_PICK_COLUMNS)
    with pytest.raises(VelestFormatError, match="duplicate P pick"):
        format_cnv(ev, dup)
    with pytest.raises(VelestFormatError, match="no picks"):
        format_cnv(make_events(2), make_picks([2]))
    with pytest.raises(VelestFormatError, match="unknown events"):
        format_cnv(ev, make_picks([2, 2]))
    with pytest.raises(VelestFormatError, match="at most 4"):
        format_cnv(ev, make_picks([5]), max_obs=4)
    with pytest.raises(VelestFormatError, match="year"):
        format_cnv(make_events(1, time=pd.Timestamp("2100-01-01")), make_picks([2]))
    with pytest.raises(VelestFormatError, match="magnitude"):
        format_cnv(make_events(1, mag=123.0), make_picks([2]))
    with pytest.raises(VelestFormatError, match="duplicate event"):
        format_cnv(make_events(1).pipe(lambda d: pd.concat([d, d])), make_picks([2]))


def test_cnv_reader_errors() -> None:
    hdr = "120717  627 50.48 38.5621N 176.1913W    3.20    0.74"
    with pytest.raises(VelestFormatError, match="upper case"):
        parse_cnv(hdr.replace("N 176", "n 176") + "\nAAAAP0  1.00\n\n")
    with pytest.raises(VelestFormatError, match="too short"):
        parse_cnv("1207\n")
    with pytest.raises(VelestFormatError, match="outside"):
        parse_cnv(hdr + "\nAAAAP0100.00\n\n")
    with pytest.raises(VelestFormatError, match="outside"):
        parse_cnv(hdr + "\nAAAAP0 -1.00\n\n")
    with pytest.raises(VelestFormatError, match="unknown phase"):
        parse_cnv(hdr + "\nAAAAZ0  1.00\n\n")
    with pytest.raises(VelestFormatError, match="invalid date"):
        parse_cnv(hdr.replace("120717", "121317") + "\nAAAAP0  1.00\n\n")


def test_cnv_zshift_roundtrip() -> None:
    ev = make_events(1, depth_km=2.0)
    text = format_cnv(ev, make_picks([2]), zshift_km=0.9)
    assert fortran_header(text.splitlines()[0])["depth"] == pytest.approx(1.1)
    assert parse_cnv(text, zshift_km=0.9).events.loc[0, "depth_km"] == pytest.approx(2.0)
    assert parse_cnv(text).events.loc[0, "depth_km"] == pytest.approx(1.1)


def test_cnv_seconds_rounding_never_prints_60() -> None:
    ev = make_events(1, time=pd.Timestamp("2020-12-31 23:59:59.996"))
    d = parse_cnv(format_cnv(ev, make_picks([2])))
    assert d.events.loc[0, "time"] == pd.Timestamp("2021-01-01 00:00:00")


def test_cnv_year_pivot() -> None:
    ev = make_events(1, time=pd.Timestamp("1985-05-05 01:02:03"))
    text = format_cnv(ev, make_picks([2]), century_pivot=1950)
    assert parse_cnv(text, century_pivot=1950).events.loc[0, "time"].year == 1985
    assert parse_cnv(text, century_pivot=1980).events.loc[0, "time"].year == 1985
    assert parse_cnv(text, century_pivot=1990).events.loc[0, "time"].year == 2085


def test_cnv_write_cnv_files(tmp_path: Path) -> None:
    d = CnvData(make_events(3), make_picks([4, 7, 6]))
    p = write_cnv(tmp_path / "sub" / "a.cnv", d)
    q = write_cnv(tmp_path / "b.cnv", d.events, d.picks)
    assert p.read_text() == q.read_text()
    assert read_cnv(p).picks["tt"].tolist() == pytest.approx(d.picks["tt"].tolist())
    with pytest.raises(TypeError):
        write_cnv(tmp_path / "c.cnv", d.events)


def test_cnv_synthetic_velest_output_reads() -> None:
    d = read_cnv(DATA / "synth_velout.cnv")
    assert len(d.events) == 8 and len(d.picks) == 8 * 16
    assert d.events["gap"].notna().all()


# ----------------------------------------------------------------------------------------
# Station file
# ----------------------------------------------------------------------------------------


def test_stations_legacy_input_values_and_mirror() -> None:
    raw = read_stations(DATA / "mc_nmstats.sta")
    assert len(raw) == 15
    r = raw.iloc[0]
    assert (r["sta"], r["lat"], r["lon"], r["elev_m"], r["icc"], r["flag"]) == (
        "WPRZ",
        38.5196,
        -176.1624,
        519,
        13,
        1,
    )
    assert raw.attrs["format"] == "(a4,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)"
    mir = read_stations(DATA / "mc_nmstats.sta", legacy_mirror=True)
    assert mir.loc[0, "lat"] == -38.5196 and mir.loc[0, "lon"] == 176.1624
    text = format_stations(mir, legacy_mirror=True)
    assert parse_stations(text).drop(columns="flag").equals(raw.drop(columns="flag"))
    assert text.splitlines()[1].startswith("WPRZ38.5196N 176.1624W  519 1  13")
    assert text.endswith("\n\n")  # blank terminator record


def test_stations_input_roundtrip_parsed_equal() -> None:
    raw = read_stations(DATA / "mc_nmstats.sta")
    back = parse_stations(format_stations(raw))
    assert back.equals(raw)
    assert format_stations(back) == format_stations(raw)


def test_stations_velest_output_variant() -> None:
    path = DATA / "mc_velout.sta"
    s = read_stations(path)
    assert len(s) == 53
    assert s["flag"].isna().all()  # first-line comment is not a flag
    assert s.loc[0, ["sta", "lat", "lon", "elev_m", "model", "icc"]].tolist() == [
        "RT05",
        -38.6318,
        176.1960,
        369,
        1,
        1,
    ]
    assert s.loc[0, "ptcor"] == 0.24 and s.loc[0, "stcor"] == 0.33
    # VELEST writes alternating model numbers for nsp=2 runs: read as is
    assert s["model"].tolist()[:4] == [1, 2, 1, 2]
    # byte equal up to the free-text comment and the 80 column padding
    got = format_stations(s).split("\n")
    want = path.read_text().split("\n")
    assert [g.rstrip() for g in got[2:]] == [w.rstrip() for w in want[2:]]
    assert got[1] == want[1][:46]
    assert parse_stations(format_stations(s)).equals(s)


def test_stations_duplicate_names_warn_and_roundtrip(caplog: pytest.LogCaptureFixture) -> None:
    s = read_stations(DATA / "mc_rk_stats_all.sta")
    assert s["sta"].duplicated().any()
    with caplog.at_level(logging.WARNING, logger="osv.io.velest"):
        text = format_stations(s, with_flag=True)
    assert "duplicate station names" in caplog.text
    assert parse_stations(text).equals(s)


def test_stations_float_elevation_is_an_error() -> None:
    with pytest.raises(VelestFormatError, match="cannot read station record"):
        read_stations(DATA / "mc_stats_float_elev.sta")


def test_stations_negative_elevation_lowercase_letters_and_no_terminator() -> None:
    fm = "(a4,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)"
    text = (
        fm
        + "\nAB  45.0000s   7.4396e -999 1   1  0.10   0.20"
        + "\nCD  45.0000n   7.4396w 9999 2   2 -0.10  -0.20"
    )
    s = parse_stations(text)
    assert s["lat"].tolist() == [-45.0, 45.0]
    assert s["lon"].tolist() == [7.4396, -7.4396]
    assert s["elev_m"].tolist() == [-999, 9999]
    assert s["ptcor"].tolist() == [0.1, -0.1] and s["stcor"].tolist() == [0.2, -0.2]
    back = parse_stations(format_stations(s))
    assert back.equals(s)


def test_stations_stop_at_blank_name_and_other_format() -> None:
    fm = "(a4,f7.4,a1,f9.4,a1,i5,i2,i4,f6.2,f7.2)"
    rec = "AB  45.0000N   7.4396E  -12 3  17  0.10   0.20"
    s = parse_stations(f"{fm}\n{rec}\n    \nZZ  45.0000N   7.4396E 100 1   1  0.00   0.00\n")
    assert len(s) == 1
    assert s.loc[0, ["elev_m", "model", "icc"]].tolist() == [-12, 3, 17]
    assert s.loc[0, "lon"] == 7.4396


def test_stations_errors() -> None:
    fm = "(a4,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)"
    with pytest.raises(VelestFormatError, match="no format line"):
        parse_stations("")
    with pytest.raises(VelestFormatError, match="must describe"):
        parse_stations("(a4,f7.4,a1)\nAB  45.0000N\n")
    with pytest.raises(VelestFormatError, match="unsupported"):
        parse_stations("(a4,2(f7.4))\n")
    with pytest.raises(VelestFormatError, match="letters"):
        parse_stations(fm + "\nAB  45.0000X   7.4396E    0 1   1  0.00   0.00\n")
    base = pd.DataFrame({"sta": ["AB"], "lat": [1.0], "lon": [2.0], "elev_m": [0]})
    for col, val, msg in [
        ("elev_m", 10000, "i4"),
        ("elev_m", -1000, "i4"),
        ("lat", 100.0, "latitude"),
        ("model", 10, "model"),
        ("icc", 1000, "icc"),
        ("ptcor", 100.0, "ptcor"),
        ("sta", "ABCDE", "longer than 4"),
    ]:
        with pytest.raises(VelestFormatError, match=msg):
            format_stations(base.assign(**{col: val}))
    with pytest.raises(VelestFormatError, match="standard"):
        format_stations(base, fmt="(a4)")


def test_stations_defaults_and_files(tmp_path: Path) -> None:
    base = pd.DataFrame(
        {"sta": ["AA", "BB", "CC"], "lat": [-38.1, -38.2, -38.3], "lon": [176.0] * 3, "elev_m": 0}
    )
    p = write_stations(tmp_path / "x" / "s.sta", base)
    s = read_stations(p)
    assert s["icc"].tolist() == [1, 2, 3]  # CC (highest) is the reference station
    assert s["model"].tolist() == [1, 1, 1] and (s["ptcor"] == 0).all()
    assert s["lat"].tolist() == [-38.1, -38.2, -38.3]


# ----------------------------------------------------------------------------------------
# Model file
# ----------------------------------------------------------------------------------------


def test_model_legacy_crlf_files() -> None:
    m = read_model(DATA / "mc_6layer_crlf.mod")
    assert m.title == "NM_6Layer_MC_0001" and m.nmod == 2
    assert m.p.layers["vel"].tolist() == [2.54, 4.46, 2.83, 4.34, 2.70, 5.60]
    assert m.p.layers["depth"].tolist() == [-0.6, 0.2, 0.7, 1.2, 1.7, 2.2]
    assert m.s is not None and m.s.layers["vel"].tolist()[0] == 1.45
    assert (m.p.layers["vdamp"] == 1.0).all()
    assert (m.p.layers["reflector"] == " ").all()
    again = parse_model(format_model(m))
    assert again.equals(m)
    assert format_model(again) == format_model(m)
    other = read_model(DATA / "mc_6layer_input_0002.txt")
    assert len(other.p.layers) == 6 and other.s is not None


def test_model_writer_layout_matches_format() -> None:
    lay = pd.DataFrame(
        {"vel": [4.8, 5.0], "depth": [-3.0, 0.05], "vdamp": [1.0, 999.0], "reflector": [" ", "m"]}
    )
    m = VelestModel("test", ModelBlock(lay, label="P-VELOCITY MODEL"))
    lines = format_model(m).splitlines()
    assert lines[0] == "test"
    assert lines[1].startswith(" 02 ") and lines[1][:3] == " 02"
    # (f5.2,5x,f7.2,2x,f7.3,3x,a1,1x,a40)
    assert lines[2] == " 4.80       -3.00    1.000     P-VELOCITY MODEL"
    assert lines[3] == " 5.00        0.05  999.000   m"
    back = parse_model("\n".join(lines))
    assert back.p.label == "P-VELOCITY MODEL"
    assert back.p.layers["reflector"].tolist() == [" ", "m"]
    assert back.p.layers["vdamp"].tolist() == [1.0, 999.0]


def test_model_missing_s_block() -> None:
    p = ModelBlock(pd.DataFrame({"vel": [5.0, 6.0], "depth": [0.0, 5.0]}))
    only_p = format_model(VelestModel("x", p))
    m = parse_model(only_p)
    assert m.s is None and m.nmod == 1
    assert parse_model(only_p, nmod=1).s is None
    with pytest.raises(VelestFormatError, match="needs an S block"):
        parse_model(only_p, nmod=2)
    both = format_model(VelestModel("x", p, ModelBlock(p.layers)))
    assert parse_model(both, nmod=1).s is None  # nsp != 2: S block is ignored like VELEST
    assert parse_model(both, nmod=2).s is not None


def test_model_velest_output_variant() -> None:
    m = read_model(DATA / "synth_velout.mod")
    assert m.title == " Output model:"  # write(12,*) leaves a leading blank
    assert m.p.layers["vel"].tolist() == [4.5, 5.0, 5.51, 6.0]
    assert m.s is not None and m.s.layers["depth"].tolist() == [-3.0, 0.05, 3.0, 6.0]
    # VELEST's own i3 read would see a blank count here; our reader uses the first integer
    assert (DATA / "synth_velout.mod").read_text().splitlines()[1].startswith("  ")


def test_model_errors() -> None:
    ok = pd.DataFrame({"vel": [5.0, 6.0], "depth": [0.0, 5.0]})
    for bad, msg in [
        (pd.DataFrame({"vel": [5.0, 6.0], "depth": [0.0, 0.0]}), "increasing"),
        (pd.DataFrame({"vel": [0.0, 6.0], "depth": [0.0, 1.0]}), "velocities"),
        (pd.DataFrame({"vel": [100.0], "depth": [0.0]}), "velocities"),
        (ok.assign(vdamp=1000.0), "vdamp"),
        (ok.assign(vdamp=-1.0), "vdamp"),
        (ok.assign(reflector="x"), "reflector"),
        (pd.DataFrame({"vel": [], "depth": []}), "layers"),
    ]:
        with pytest.raises(VelestFormatError, match=msg):
            ModelBlock(bad)
    with pytest.raises(VelestFormatError, match="40"):
        format_model(VelestModel("x" * 41, ModelBlock(ok)))
    with pytest.raises(VelestFormatError, match="empty"):
        parse_model("")
    with pytest.raises(VelestFormatError, match="no velocity block"):
        parse_model("title\n")
    with pytest.raises(VelestFormatError, match="layer count"):
        parse_model("title\nabc\n")
    with pytest.raises(VelestFormatError, match="announces 3"):
        parse_model("title\n 3\n 5.00        0.00    1.00\n")
    with pytest.raises(VelestFormatError, match="cannot read layer"):
        parse_model("title\n 1\n 5.xx        0.00    1.00\n")
    with pytest.raises(ValueError, match="nmod"):
        parse_model("t", nmod=3)  # type: ignore[arg-type]


def test_model_negative_depth_and_files(tmp_path: Path) -> None:
    lay = pd.DataFrame({"vel": [2.5, 5.5], "depth": [-1.2, 0.05]})
    p = write_model(tmp_path / "d" / "m.mod", VelestModel("neg", ModelBlock(lay)))
    assert read_model(p).p.layers["depth"].tolist() == [-1.2, 0.05]


# ----------------------------------------------------------------------------------------
# Control file
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mc_montecarlo.cmn", "mc_velest.cmn"])
def test_control_legacy_files_roundtrip(name: str) -> None:
    path = DATA / name
    c = read_control(path)
    text = format_control(c)
    assert noncomment(text) == noncomment(path.read_text())
    assert parse_control(text).equals(c)


def test_control_montecarlo_values() -> None:
    c = read_control(DATA / "mc_montecarlo.cmn")
    assert c.title == "Ngatamariki VELEST MonteCarlo Run1"
    assert (c.olat, c.olon, c.neqs, c.nsp, c.nmod, c.vpvs, c.swtfac) == (
        38.55,
        176.194,
        459,
        2,
        2,
        1.75,
        0.5,
    )
    assert (c.dmax, c.zadj, c.ittmax, c.invertratio, c.iuseelev) == (1000.0, 1.0, 9, 1, 1)
    assert c.modelfile == "NM_6Layer_MC_0001.mod" and c.stationfile == "nmstats.sta"
    assert c.cnvfile == "NM_6Layer_MC_0001.CNV" and c.stafile == "NM_6Layer_MC_0001.STA"
    assert c.seismofile == "" and c.resfile == ""
    # legacy point mirror: file says N/W-style values for a south/east site
    assert c.origin() == (38.55, -176.194)
    assert c.origin(legacy_mirror=True) == (-38.55, 176.194)


def test_control_from_true_origin_uses_coords_helpers() -> None:
    c = VelestControl.from_origin(-38.55, 176.194, neqs=10)
    assert c.olat == coords.velest_olat(-38.55) == -38.55
    assert c.olon == pytest.approx(coords.velest_olon(176.194), abs=1e-9)
    assert c.olon == -176.194
    assert c.origin() == (-38.55, 176.194)
    back = parse_control(format_control(c))
    assert back.origin() == (-38.55, 176.194)
    west = VelestControl.from_origin(34.0, -118.25)
    assert (west.olat, west.olon) == (34.0, 118.25)
    with pytest.raises(ValueError, match="latitude"):
        VelestControl.from_origin(95.0, 0.0)


def test_control_defaults_and_nmod() -> None:
    assert VelestControl().nmod == 2
    assert VelestControl(nsp=1).nmod == 1 and VelestControl(nsp=3).nmod == 1
    with pytest.raises(ValueError, match="nmod must be 1"):
        VelestControl(nsp=1, nmod=2)


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"olat": 91.0}, "olat"),
        ({"olon": 181.0}, "olon"),
        ({"icoordsystem": 3}, "icoordsystem"),
        ({"itrial": 2}, "itrial"),
        ({"ised": 3}, "ised"),
        ({"neqs": -1}, "neqs"),
        ({"rotate": 400.0}, "rotate"),
        ({"isingle": 2}, "isingle"),
        ({"iresolcalc": 2}, "iresolcalc"),
        ({"dmax": -1.0}, "dmax"),
        ({"itopo": 3}, "itopo"),
        ({"veladj": -0.1}, "veladj"),
        ({"lowveloclay": 2}, "lowveloclay"),
        ({"nsp": 4}, "nsp"),
        ({"swtfac": -1.0}, "swtfac"),
        ({"vpvs": 0.0}, "vpvs"),
        ({"vthet": -1.0}, "vthet"),
        ({"nsinv": 2}, "nsinv"),
        ({"iuseelev": 2}, "iuseelev"),
        ({"istaout": 3}, "istaout"),
        ({"iresout": 2}, "iresout"),
        ({"ittmax": 100}, "ittmax"),
        ({"invertratio": -1}, "invertratio"),
        ({"delmin": -1.0}, "delmin"),
        ({"title": "* hidden"}, "title"),
        ({"modelfile": "x" * 81}, "modelfile"),
        ({"stationfile": "*x"}, "stationfile"),
        ({"cnvfile": " a.cnv"}, "cnvfile"),
    ],
)
def test_control_validation(kw: dict[str, object], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        VelestControl(**kw)  # type: ignore[arg-type]


def test_control_short_24_line_form(tmp_path: Path) -> None:
    c = VelestControl.from_origin(-38.55, 176.19, neqs=5, modelfile="m.mod", stationfile="s.sta")
    text = format_control(c, extended_files=False)
    assert len([ln for ln in text.split("\n")[:-1] if not ln.startswith("*")]) == 24
    back = parse_control(text)
    assert back.equals(c)
    with pytest.raises(ValueError, match="extended_files"):
        format_control(VelestControl(smpfile="x.smp"), extended_files=False)
    p = write_control(tmp_path / "run" / "velest.cmn", c)
    assert read_control(p).equals(c)


def test_control_blank_lines_count_like_velest() -> None:
    text = format_control(VelestControl())
    cards = [ln for ln in text.split("\n")[:-1] if not ln.startswith("*")]
    assert len(cards) == 32  # blank file-name lines are real lines
    with pytest.raises(VelestFormatError, match="24 or 32"):
        parse_control(text + "\n")  # a stray blank line is one card too many (velest.f:1011)
    with pytest.raises(VelestFormatError, match="24 or 32"):
        parse_control("title\n1 2 3\n")


def replace_card(text: str, index: int, new: str) -> str:
    """Replace the ``index``-th non-comment line (0 = title) of a control file."""
    lines = text.split("\n")
    seen = -1
    for i, ln in enumerate(lines):
        if not ln.startswith("*"):
            seen += 1
            if seen == index:
                lines[i] = new
                return "\n".join(lines)
    raise AssertionError("no such card")


def test_control_free_format_and_errors() -> None:
    base = format_control(VelestControl())
    c = parse_control(replace_card(base, 6, "0.02,0.03 0.04 2 0.05 extra"))
    assert (c.othet, c.xythet, c.zthet, c.vthet, c.stathet) == (0.02, 0.03, 0.04, 2.0, 0.05)
    with pytest.raises(VelestFormatError, match="needs 7 values"):
        parse_control(replace_card(base, 1, "1.0 2.0 0"))
    with pytest.raises(VelestFormatError, match="neqs must be an integer"):
        parse_control(replace_card(base, 2, "x 0 0.0"))
    with pytest.raises(VelestFormatError, match="vpvs must be a number"):
        parse_control(replace_card(base, 5, "2 0.5 abc 2"))
    with pytest.raises(VelestFormatError, match="nsp must be"):
        parse_control(replace_card(base, 5, "4 0.5 1.7 1"))
    with pytest.raises(VelestFormatError, match="control file"):
        parse_control(replace_card(base, 4, "1000.0 0 0.0 0.2 1.0 1.5"))


def test_control_every_field_roundtrips() -> None:
    c = VelestControl(
        title="all fields",
        olat=-12.3456,
        olon=-98.7654,
        icoordsystem=1,
        zshift=0.9,
        itrial=1,
        ztrial=-2.5,
        ised=1,
        neqs=1234,
        nshot=3,
        rotate=-45.5,
        isingle=1,
        iresolcalc=1,
        dmax=250.5,
        itopo=2,
        zmin=-1.25,
        veladj=0.35,
        zadj=2.5,
        lowveloclay=0,
        nsp=3,
        swtfac=0.8,
        vpvs=1.725,
        othet=0.1,
        xythet=0.2,
        zthet=0.3,
        vthet=2.0,
        stathet=0.05,
        nsinv=0,
        nshcor=1,
        nshfix=1,
        iuseelev=0,
        iusestacorr=1,
        iturbo=0,
        icnvout=0,
        istaout=2,
        ismpout=1,
        irayout=1,
        idrvout=1,
        ialeout=1,
        idspout=1,
        irflout=1,
        irfrout=1,
        iresout=1,
        delmin=0.005,
        ittmax=15,
        invertratio=3,
        modelfile="a.mod",
        stationfile="b.sta",
        seismofile="c",
        regnamfile="d",
        regkoordfile="e",
        topo1file="f",
        topo2file="g",
        phasefile="h.cnv",
        shotfile="i",
        outfile="j.out",
        velfile="k",
        cnvfile="l.cnv",
        stafile="m.sta",
        smpfile="n",
        rayfile="o",
        drvfile="p",
        alefile="q",
        dsprfile="r",
        rflfile="s",
        rfrfile="t",
        resfile="u",
    )
    text = format_control(c)
    back = parse_control(text)
    assert back.equals(c)
    assert back.nmod == 1
    assert format_control(back) == text


# ----------------------------------------------------------------------------------------
# Integration: write a tiny synthetic dataset and run VELEST on it
# ----------------------------------------------------------------------------------------


def write_synthetic_dataset(directory: Path) -> VelestControl:
    """Eight events around eight stations in a homogeneous-ish medium (Vp 5.5, Vp/Vs 1.73)."""
    rng = np.random.default_rng(1)
    lat0, lon0 = -38.55, 176.19
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    r = np.array([3, 5, 4, 6, 3.5, 5.5, 4.5, 7.0]) / 111.0
    st = pd.DataFrame(
        {
            "sta": [f"ST{i:02d}" for i in range(8)],
            "lat": lat0 + r * np.cos(ang),
            "lon": lon0 + r * np.sin(ang) / np.cos(np.deg2rad(lat0)),
            "elev_m": 0,
        }
    )
    ev_rows, pk_rows = [], []
    for k in range(8):
        la, lo, z = lat0 + rng.normal(0, 0.01), lon0 + rng.normal(0, 0.01), rng.uniform(2, 5)
        ev_rows.append(
            {
                "event": k,
                "time": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=k, seconds=12.34),
                "lat": la,
                "lon": lo,
                "depth_km": z,
                "mag": 1.5,
            }
        )
        for s in st.itertuples():
            epi = float(coords.epicentral_distance_km(la, lo, s.lat, s.lon))
            h = float(np.hypot(epi, z))
            pk_rows.append((k, s.sta, "P", 0, h / 5.5))
            pk_rows.append((k, s.sta, "S", 0, h / 5.5 * 1.73))
    write_cnv(
        directory / "ev.cnv", pd.DataFrame(ev_rows), pd.DataFrame(pk_rows, columns=CNV_PICK_COLUMNS)
    )
    write_stations(directory / "st.sta", st)
    lay = pd.DataFrame({"vel": [4.5, 5.0, 5.5, 6.0], "depth": [-3.0, 0.05, 3.0, 6.0]})
    s_lay = lay.assign(vel=(lay["vel"] / 1.73).round(2))
    write_model(
        directory / "m.mod",
        VelestModel("synthetic", ModelBlock(lay, "P-VELOCITY MODEL"), ModelBlock(s_lay, "S")),
    )
    ctrl = VelestControl.from_origin(
        lat0,
        lon0,
        title="synthetic",
        neqs=8,
        iuseelev=0,
        istaout=2,
        modelfile="m.mod",
        stationfile="st.sta",
        phasefile="ev.cnv",
        outfile="velest.out",
        cnvfile="velout.cnv",
        stafile="velout.sta",
    )
    write_control(directory / "velest.cmn", ctrl)
    return ctrl


def test_synthetic_dataset_files_are_readable(tmp_path: Path) -> None:
    ctrl = write_synthetic_dataset(tmp_path)
    assert read_control(tmp_path / "velest.cmn").equals(ctrl)
    assert len(read_cnv(tmp_path / "ev.cnv").events) == ctrl.neqs
    assert len(read_stations(tmp_path / "st.sta")) == 8
    assert read_model(tmp_path / "m.mod", nmod=ctrl.nmod).s is not None


@pytest.mark.integration
def test_velest_runs_on_synthetic_dataset(tmp_path: Path) -> None:
    from osv.tools import ToolNotFoundError, which_tool

    try:
        exe = which_tool("velest")
    except ToolNotFoundError:
        pytest.skip("velest executable not found")
    ctrl = write_synthetic_dataset(tmp_path)
    proc = subprocess.run(
        [str(exe)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-500:]
    out = read_cnv(tmp_path / "velout.cnv", allow_negative_tt=True)
    assert len(out.events) == ctrl.neqs
    assert out.events["gap"].notna().all()
    sta = read_stations(tmp_path / "velout.sta")
    assert len(sta) == 8 and (sta["elev_m"] == 0).all()
    mod = read_model(tmp_path / "velout.mod")
    assert mod.s is not None and len(mod.p.layers) == 4
    # relocations stay near the true origin (lat -38.55, lon 176.19)
    assert (out.events["lat"] + 38.55).abs().max() < 0.1
    assert (out.events["lon"] - 176.19).abs().max() < 0.1


@pytest.mark.parametrize("val", [1e-4, 0.5, 1e5, 1e8, 123456789.5])
def test_control_extreme_damping_roundtrips(val: float) -> None:
    c = VelestControl(vthet=val, othet=val, zthet=val, xythet=val, stathet=val)
    text = format_control(c)
    back = parse_control(text)
    for n in ("vthet", "othet", "zthet", "xythet", "stathet"):
        assert getattr(back, n) == pytest.approx(val, rel=1e-9)
    # every numeric field is its own list-directed token (no fused values)
    line = next(
        ln
        for ln in text.splitlines()
        if not ln.startswith("*")
        and len(ln.split()) == 5
        and ln.split()[0].replace(".", "").isdigit()
        and abs(float(ln.split()[3]) - val) < 1e-6 * val + 1e-9
    )
    assert len(line.split()) == 5


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 1e20])
def test_control_unrepresentable_value_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        format_control(VelestControl(vthet=bad))


@pytest.mark.integration
def test_velest_runs_with_large_vthet(tmp_path: Path) -> None:
    from osv.tools import ToolNotFoundError, which_tool

    try:
        exe = which_tool("velest")
    except ToolNotFoundError:
        pytest.skip("velest executable not found")
    ctrl = write_synthetic_dataset(tmp_path)
    ctrl.vthet = 1e8
    write_control(tmp_path / "velest.cmn", ctrl)
    proc = subprocess.run(
        [str(exe)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-500:]
    assert "Bad real number" not in proc.stdout + proc.stderr
    assert len(read_cnv(tmp_path / "velout.cnv", allow_negative_tt=True).events) == ctrl.neqs
