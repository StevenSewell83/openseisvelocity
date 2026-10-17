"""Readers and writers for the VELEST file formats.

Covers the four files VELEST reads or writes in a normal run:

* **CNV** earthquake files (``read_cnv`` / ``write_cnv``): input *and* output variants
  (``INPUTCNV`` ``velest.f:3403-3497``, output ``FINALHYPOCOUT`` ``velest.f:3268-3287``).
* **Station files** (``read_stations`` / ``write_stations``): input and the
  ``velout.sta`` output variant (``velest.f:1520-1573``, ``6123-6162``).
* **Model files** (``read_model`` / ``write_model``): P block and optional S block
  (``velest.f:1373-1398``).
* **Control file** ``velest.cmn`` (:class:`VelestControl`, ``read_control`` /
  ``write_control``): ``INPUTPARAM`` ``velest.f:985-1063``.

Conventions
-----------
Everything in memory uses true signed coordinates (north and east positive, depth km
positive down, elevation m positive up).  Conversion to the VELEST file conventions uses
:mod:`osv.coords` only.  The readers return **what the file says**: the legacy
Ngatamariki files (``MonteCarlo/nmstats.sta``, ``*.cnv``, ``velest.cmn``) are point
mirrored (``N``/``W`` letters for a site that is really S/E).  Pass
``legacy_mirror=True`` to the CNV/station readers (and writers, which then apply the
inverse, the mirror being an involution) to work in true coordinates; use
:meth:`VelestControl.origin` for the control file.

Tables
------
* CNV: ``events`` DataFrame (``event`` int id, ``time`` UTC ``datetime64[ns]``, ``lat``,
  ``lon``, ``depth_km``, ``mag``, ``gap`` nullable int, ``rms``) and ``picks`` DataFrame
  (``event``, ``sta``, ``phase``, ``weight``, ``tt`` seconds after origin time).
* Stations: ``sta``, ``lat``, ``lon``, ``elev_m``, ``model``, ``icc``, ``ptcor``,
  ``stcor``, ``flag`` (nullable int; the optional trailing integer after the format
  span that the legacy files carry).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from osv.coords import (
    format_velest_lat,
    format_velest_lon,
    legacy_mirror,
    parse_velest_lat,
    parse_velest_lon,
    velest_depth_from_file,
    velest_depth_to_file,
    velest_elevation_field,
    velest_olat,
    velest_olon,
)

__all__ = [
    "CNV_EVENT_COLUMNS",
    "CNV_PICK_COLUMNS",
    "MAX_OBS_PER_EVENT",
    "STATION_COLUMNS",
    "STATION_FORMAT",
    "CnvData",
    "ModelBlock",
    "VelestControl",
    "VelestFormatError",
    "VelestModel",
    "format_control",
    "format_cnv",
    "format_model",
    "format_stations",
    "parse_cnv",
    "parse_control",
    "parse_model",
    "parse_stations",
    "read_cnv",
    "read_control",
    "read_model",
    "read_stations",
    "write_cnv",
    "write_control",
    "write_model",
    "write_stations",
]

logger = logging.getLogger(__name__)

#: ``maxobsperevent`` of ``INPUTCNV`` (``velest.f``, parameter in the subroutine
#: declarations).  Override with the ``max_obs`` argument when the vendored build is patched.
MAX_OBS_PER_EVENT = 180
#: Phase characters VELEST knows (``velest.f:3278-3282``): P, S, reflected (M), S-P (-).
PHASE_CHARS = "PSM-"
#: Observation weights: 0-4 are used, 5 means "no reading" (user guide p. 22 / ``velest.f``).
MAX_WEIGHT = 5

CNV_EVENT_COLUMNS = ["event", "time", "lat", "lon", "depth_km", "mag", "gap", "rms"]
CNV_PICK_COLUMNS = ["event", "sta", "phase", "weight", "tt"]
STATION_COLUMNS = ["sta", "lat", "lon", "elev_m", "model", "icc", "ptcor", "stcor", "flag"]
#: Station-file format line written by default (user guide p. 20, ``nmstats.sta``).
STATION_FORMAT = "(a4,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)"
_STATION_FORMAT_FIELDS = ("a", "f", "a", "f", "a", "i", "i", "i", "f", "f")


class VelestFormatError(ValueError):
    """A VELEST file is malformed (or the data cannot be represented in the format)."""


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="latin-1")


def _write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("latin-1"))
    return p


def _lines(text: str) -> list[str]:
    """Split into lines, dropping ``\\r`` (the legacy model files are CRLF)."""
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()  # the newline that ends the last line is not an extra (blank) line
    return [ln.rstrip("\r") for ln in parts]


def _where(source: str | None, lineno: int) -> str:
    return f"{source or '<text>'}:{lineno}"


def _fnum(field_text: str, decimals: int = 0) -> float:
    """Fortran ``Fw.d`` read: blank is 0, a decimal point overrides ``d``."""
    t = field_text.strip().replace("D", "e").replace("d", "e")
    if not t:
        return 0.0
    if "." in t or "e" in t.lower():
        return float(t)
    return int(t) / 10.0**decimals


def _inum(field_text: str) -> int:
    """Fortran ``Iw`` read: blank is 0; anything else must be an integer."""
    t = field_text.strip()
    return int(t) if t else 0


def _check_name(name: str, what: str) -> str:
    n = str(name)
    if not n.strip():
        raise VelestFormatError(f"{what}: station name is blank")
    if len(n) > 4:
        raise VelestFormatError(
            f"{what}: station name {n!r} is longer than 4 characters; VELEST needs "
            "<= 4-character codes (use station aliases)"
        )
    return n


def _mirror_pair(lat: float, lon: float) -> tuple[float, float]:
    la, lo = legacy_mirror(lat, lon)
    return float(la), float(lo)


# ----------------------------------------------------------------------------------------
# CNV
# ----------------------------------------------------------------------------------------


@dataclass(eq=False)
class CnvData:
    """Events and picks of a CNV file.

    Attributes
    ----------
    events
        One row per event (see module docstring for columns).
    picks
        One row per pick; ``event`` refers to ``events["event"]``.
    """

    events: pd.DataFrame
    picks: pd.DataFrame

    def picks_of(self, event: int) -> pd.DataFrame:
        """Picks of one event."""
        return self.picks[self.picks["event"] == event]

    def equals(self, other: CnvData) -> bool:
        """Table equality (exact)."""
        return self.events.equals(other.events) and self.picks.equals(other.picks)


# Header, input reader velest.f:3447-3451: (3i2,1x,2i2,1x,f5.2,1x,f7.4,a1,1x,f8.4,a1,1x,f7.2,
# 2x,f5.2): depth in columns 38-44, magnitude in 47-51 (1-based).  VELEST's own output,
# velest.f:3280-3281: (...,a1,2f7.2,4x,i3,5x,f5.2): depth 37-43, magnitude 44-50, gap, rms.
# The user guide (p. 21) and the legacy files use yet other widths, so depth/mag are read as
# the first two blank-separated numbers after the longitude letter (fixed columns of the
# input reader as a fallback).


def _year_from_yy(yy: int, pivot: int) -> int:
    """Two-digit year to four digits: the century that puts it in ``[pivot, pivot+99]``."""
    year = pivot - pivot % 100 + yy
    if year < pivot:
        year += 100
    return year


def _parse_cnv_header(
    line: str, source: str | None, lineno: int, century_pivot: int
) -> dict[str, object]:
    where = _where(source, lineno)
    if len(line) < 43:
        raise VelestFormatError(f"{where}: CNV header line too short: {line!r}")
    padded = line.ljust(80)
    try:
        yy = _inum(padded[0:2])
        mo = _inum(padded[2:4])
        dd = _inum(padded[4:6])
        hh = _inum(padded[7:9])
        mi = _inum(padded[9:11])
        sec = _fnum(padded[12:17])
        lat_txt = padded[18:25]
        lat_ch = padded[25]
        lon_txt = padded[27:35]
        lon_ch = padded[35]
        rest = padded[36:].split()
        try:
            depth, mag = float(rest[0]), float(rest[1])
        except (IndexError, ValueError):
            depth, mag = _fnum(padded[36:43]), _fnum(padded[45:50])
    except ValueError as exc:
        raise VelestFormatError(f"{where}: cannot parse CNV header {line!r}: {exc}") from exc
    # VELEST tests the letters case sensitively (velest.f:3459-3460): demand upper case.
    if lat_ch not in "NS" or lon_ch not in "EW":
        raise VelestFormatError(
            f"{where}: latitude/longitude letters must be upper case N/S and E/W "
            f"(got {lat_ch!r}, {lon_ch!r}) in {line!r}"
        )
    lat = parse_velest_lat(lat_txt + lat_ch)
    lon = parse_velest_lon(lon_txt + lon_ch)
    gap: int | None = None
    rms = float("nan")
    if len(rest) >= 4:  # output variant: depth mag gap rms (a lone trailing IFX is ignored)
        try:
            gap, rms = int(rest[2]), float(rest[3])
        except ValueError:
            gap, rms = None, float("nan")
    year = _year_from_yy(yy, century_pivot)
    if sec >= 60.0 or sec < 0.0 or not (1 <= mo <= 12) or not (1 <= dd <= 31):
        raise VelestFormatError(f"{where}: invalid date/time in CNV header {line!r}")
    try:
        t = pd.Timestamp(year=year, month=mo, day=dd, hour=hh, minute=mi) + pd.Timedelta(
            milliseconds=int(round(sec * 100.0)) * 10
        )
    except ValueError as exc:
        raise VelestFormatError(f"{where}: invalid date/time in CNV header {line!r}") from exc
    return {
        "time": t,
        "lat": lat,
        "lon": lon,
        "depth_km": depth,
        "mag": mag,
        "gap": gap,
        "rms": rms,
    }


def parse_cnv(
    text: str,
    *,
    legacy_mirror: bool = False,
    zshift_km: float = 0.0,
    century_pivot: int = 1970,
    allow_negative_tt: bool = False,
    source: str | None = None,
) -> CnvData:
    """Parse CNV text (input or VELEST output variant).

    Follows ``INPUTCNV`` (``velest.f:3440-3497``): blank lines between events are skipped,
    a line ``9999`` ends the file, a header is followed by pick lines of six
    ``a4,a1,i1,f6.2`` groups, and the first group with a blank station name ends the event.

    Parameters
    ----------
    legacy_mirror
        Return true coordinates for point-mirrored legacy files (``lat, lon -> -lat, -lon``).
    zshift_km
        VELEST's ``zshift``; input depth in the file is shifted by it (``velest.f:1865``),
        so ``depth_km = file_depth + zshift_km``.  Leave 0 for VELEST *output* files, which
        already contain the internal depth.
    century_pivot
        Two-digit years map to ``[century_pivot, century_pivot + 99]``.
    allow_negative_tt
        Travel times ``>= 100`` s are always an error.  Negative times are an error too unless
        this is true: VELEST accepts them and its relocation output contains them when a pick
        precedes the new origin (e.g. ``MC/velout.cnv``, ``MC/RK-NM_2012-13_manpicksDec16.CNV``).
    """
    lines = _lines(text)
    ev_rows: list[dict[str, object]] = []
    pk_rows: list[tuple[int, str, str, int, float]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        if line == "9999":
            break
        hdr = _parse_cnv_header(line, source, i, century_pivot)
        ev = len(ev_rows)
        ev_rows.append({"event": ev, **hdr})
        done = False
        while not done and i < n:
            pl = lines[i].ljust(72)
            lineno = i + 1
            i += 1
            for j in range(6):
                g = pl[12 * j : 12 * j + 12]
                if not g[:4].strip():
                    done = True
                    break
                try:
                    phase = g[4]
                    weight = _inum(g[5])
                    tt = _fnum(g[6:12])
                except ValueError as exc:
                    raise VelestFormatError(
                        f"{_where(source, lineno)}: bad pick group {g!r} in {lines[lineno - 1]!r} "
                        f"(a missing blank line after an event with 6k picks?): {exc}"
                    ) from exc
                if phase not in PHASE_CHARS:
                    raise VelestFormatError(
                        f"{_where(source, lineno)}: unknown phase {phase!r} in pick group {g!r}"
                    )
                if tt >= 100.0 or (tt < 0.0 and not allow_negative_tt):
                    raise VelestFormatError(
                        f"{_where(source, lineno)}: travel time {tt} s outside [0, 100) s "
                        f"({g!r})"
                        + (
                            "; pass allow_negative_tt=True to accept negative times, which "
                            "VELEST itself writes and reads"
                            if tt < 0.0
                            else ""
                        )
                    )
                pk_rows.append((ev, g[:4].rstrip(), phase, weight, tt))
        # A header followed directly by EOF is an event without picks; keep it.
    events = pd.DataFrame(ev_rows, columns=CNV_EVENT_COLUMNS)
    picks = pd.DataFrame(pk_rows, columns=CNV_PICK_COLUMNS)
    return _finish_cnv_tables(events, picks, legacy_mirror, zshift_km)


def _finish_cnv_tables(
    events: pd.DataFrame, picks: pd.DataFrame, mirror: bool, zshift_km: float
) -> CnvData:
    events = events.copy()
    events["event"] = events["event"].astype("int64")
    events["time"] = pd.to_datetime(events["time"]).astype("datetime64[ns]")
    for c in ("lat", "lon", "depth_km", "mag", "rms"):
        events[c] = events[c].astype("float64")
    events["gap"] = pd.array(events["gap"].tolist(), dtype="Int64")
    if len(events):
        if mirror:
            la, lo = legacy_mirror(events["lat"].to_numpy(), events["lon"].to_numpy())
            events["lat"], events["lon"] = la, lo
        if zshift_km:
            events["depth_km"] = velest_depth_from_file(events["depth_km"].to_numpy(), zshift_km)
    picks = picks.copy()
    picks["event"] = picks["event"].astype("int64")
    picks["sta"] = picks["sta"].astype(object)
    picks["phase"] = picks["phase"].astype(object)
    picks["weight"] = picks["weight"].astype("int64")
    picks["tt"] = picks["tt"].astype("float64")
    return CnvData(events=events.reset_index(drop=True), picks=picks.reset_index(drop=True))


def read_cnv(path: str | Path, **kwargs: object) -> CnvData:
    """Read a CNV file; see :func:`parse_cnv` for the options."""
    return parse_cnv(_read_text(path), source=str(path), **kwargs)  # type: ignore[arg-type]


def _cs_components(t: pd.Timestamp) -> tuple[int, int, int, int, int, float]:
    """Round a timestamp to 10 ms and split it so seconds never format as ``60.00``."""
    ns = pd.Timestamp(t).value
    cs = int(round(ns / 1e7))
    t2 = pd.Timestamp(cs * 10_000_000)
    return (
        t2.year,
        t2.month,
        t2.day,
        t2.hour,
        t2.minute,
        t2.second + (t2.value // 10**7 % 100) / 100.0,
    )


def format_cnv(
    events: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    legacy_mirror: bool = False,
    zshift_km: float = 0.0,
    century_pivot: int = 1970,
    output_style: bool | None = None,
    max_obs: int = MAX_OBS_PER_EVENT,
    allow_negative_tt: bool = False,
) -> str:
    """Format events and picks as CNV text that VELEST reads.

    Header ``(3i2,1x,2i2,1x,f5.2,1x,f7.4,a1,1x,f8.4,a1,f7.2,f7.2)`` plus, with
    ``output_style`` (default: when any ``gap``/``rms`` is present), VELEST's output
    extension ``4x,i3,5x,f5.2`` (``velest.f:3280-3281``).  Picks follow, six
    ``(a4,a1,i1,f6.2)`` groups per line, then a **blank line** after every event (required
    as terminator when the pick count is a multiple of six).  The file ends after the last
    blank line (VELEST stops at EOF, ``velest.f:3441``).

    Raises
    ------
    VelestFormatError
        For travel times ``< 0`` or ``>= 100`` s, unknown phases or weights, more than
        ``max_obs`` picks, duplicate (station, phase) in an event, events without picks,
        or values that do not fit the format.  ``allow_negative_tt`` permits negative times
        (only meaningful when re-writing VELEST output files).
    """
    ev = events.reset_index(drop=True)
    if "event" not in ev:
        ev = ev.assign(event=np.arange(len(ev)))
    if ev["event"].duplicated().any():
        raise VelestFormatError("duplicate event ids in events table")
    unknown = set(picks["event"]) - set(ev["event"])
    if unknown:
        raise VelestFormatError(f"picks refer to unknown events: {sorted(unknown)[:5]}")
    has_gap = "gap" in ev and ev["gap"].notna().any()
    has_rms = "rms" in ev and ev["rms"].notna().any()
    if output_style is None:
        output_style = bool(has_gap or has_rms)

    by_event = {k: g for k, g in picks.groupby("event", sort=False)}
    out: list[str] = []
    for row in ev.itertuples(index=False):
        eid = row.event
        grp = by_event.get(eid)
        if grp is None or not len(grp):
            raise VelestFormatError(
                f"event {eid} has no picks; VELEST would skip it and neqs would be wrong"
            )
        if len(grp) > max_obs:
            raise VelestFormatError(
                f"event {eid} has {len(grp)} picks; VELEST handles at most {max_obs} per event"
            )
        lat, lon = float(row.lat), float(row.lon)
        if legacy_mirror:
            lat, lon = _mirror_pair(lat, lon)
        y, mo, dd, hh, mi, sec = _cs_components(row.time)
        if not century_pivot <= y <= century_pivot + 99:
            raise VelestFormatError(
                f"event {eid}: year {y} not representable with 2-digit years and "
                f"century_pivot={century_pivot}"
            )
        depth = float(velest_depth_to_file(float(row.depth_km), zshift_km))
        mag = float(getattr(row, "mag", 0.0))
        if np.isnan(mag):
            mag = 0.0
        if not -999.99 <= depth <= 9999.99:
            raise VelestFormatError(f"event {eid}: depth {depth} does not fit f7.2")
        mag_lo, mag_hi = (-999.99, 9999.99) if output_style else (-9.99, 99.99)
        if not mag_lo <= mag <= mag_hi:
            raise VelestFormatError(
                f"event {eid}: magnitude {mag} does not fit the CNV field "
                f"({'f7.2' if output_style else 'f5.2'})"
            )
        try:
            lat_s = format_velest_lat(lat)
        except ValueError as exc:
            raise VelestFormatError(f"event {eid}: {exc}") from exc
        if output_style:  # VELEST's own output: i2 fields and (a1,2f7.2) (velest.f:3280)
            stamp = f"{y % 100:2d}{mo:2d}{dd:2d} {hh:2d}{mi:2d}"
            dm = f"{depth:7.2f}{mag:7.2f}"
        else:  # user guide p. 21 (3i2.2,1x,2i2.2,...) and the input reader (velest.f:3447)
            stamp = f"{y % 100:02d}{mo:02d}{dd:02d} {hh:02d}{mi:02d}"
            dm = f"{depth:8.2f}{mag:7.2f}"
        head = f"{stamp} {sec:5.2f} {lat_s} {format_velest_lon(lon)}{dm}"
        if output_style:
            gap = getattr(row, "gap", None)
            rms = getattr(row, "rms", None)
            gap_v = 0 if gap is None or pd.isna(gap) else int(gap)
            rms_v = 0.0 if rms is None or pd.isna(rms) else float(rms)
            head += f"    {gap_v:3d}     {rms_v:5.2f}"
        out.append(head)
        out.extend(_format_picks(eid, grp, allow_negative_tt))
        out.append("")
    return "".join(ln + "\n" for ln in out)


def _format_picks(eid: object, grp: pd.DataFrame, allow_negative: bool) -> list[str]:
    groups: list[str] = []
    seen: set[tuple[str, str]] = set()
    for sta, phase, weight, tt in zip(
        grp["sta"], grp["phase"], grp["weight"], grp["tt"], strict=True
    ):
        sta = _check_name(sta, f"event {eid}")
        if phase not in PHASE_CHARS or len(phase) != 1:
            raise VelestFormatError(f"event {eid}: phase {phase!r} not one of {list(PHASE_CHARS)}")
        w = int(weight)
        if not 0 <= w <= MAX_WEIGHT:
            raise VelestFormatError(
                f"event {eid}: weight {weight} for {sta} {phase} outside 0..{MAX_WEIGHT}"
            )
        t = float(tt)
        if not np.isfinite(t) or (t < 0.0 and not allow_negative) or round(t, 2) >= 100.0:
            raise VelestFormatError(
                f"event {eid}: travel time {tt} s for {sta} {phase} outside [0, 100) s; "
                "VELEST cannot read it (f6.2 relative to the origin time)"
            )
        key = (sta, phase)
        if key in seen:
            raise VelestFormatError(
                f"event {eid}: duplicate {phase} pick for station {sta}; VELEST accepts one "
                "phase of each type per station and event"
            )
        seen.add(key)
        groups.append(f"{sta:<4s}{phase}{w:1d}{t:6.2f}")
    return ["".join(groups[k : k + 6]) for k in range(0, len(groups), 6)]


def write_cnv(
    path: str | Path,
    data: CnvData | pd.DataFrame,
    picks: pd.DataFrame | None = None,
    **kwargs: object,
) -> Path:
    """Write a CNV file from a :class:`CnvData` (or ``events`` plus ``picks`` tables)."""
    if isinstance(data, CnvData):
        events, pk = data.events, data.picks
    else:
        if picks is None:
            raise TypeError("write_cnv(path, events, picks) needs the picks table")
        events, pk = data, picks
    return _write_text(path, format_cnv(events, pk, **kwargs))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------
# Station file
# ----------------------------------------------------------------------------------------

_FMT_ITEM = re.compile(r"^(\d*)([xX])$|^(\d*)([aAiIfF])(\d+)(?:\.(\d+))?$")


def _parse_fortran_format(fm: str) -> list[tuple[str, int, int]]:
    """Parse a flat Fortran format (``a``, ``i``, ``f`` edit descriptors and ``nX``).

    Returns ``(kind, width, decimals)`` items; ``kind`` is ``a``, ``i``, ``f`` or ``x``.
    """
    s = fm.strip()
    if s.startswith("(") and ")" in s:
        s = s[1 : s.rindex(")")]
    items: list[tuple[str, int, int]] = []
    for tok in (t.strip() for t in s.split(",")):
        if not tok:
            continue
        m = _FMT_ITEM.match(tok)
        if not m:
            raise VelestFormatError(f"unsupported station-file format item {tok!r} in {fm!r}")
        if m.group(2):
            items.append(("x", int(m.group(1) or 1), 0))
        else:
            rep = int(m.group(3) or 1)
            items.extend([(m.group(4).lower(), int(m.group(5)), int(m.group(6) or 0))] * rep)
    return items


def parse_stations(
    text: str,
    *,
    legacy_mirror: bool = False,
    source: str | None = None,
) -> pd.DataFrame:
    """Parse a station file (input, or VELEST output ``velout.sta``).

    The first line is the Fortran format used to read the records (``velest.f:1520``);
    it must describe ``a,f,a,f,a,i,i,i,f,f`` (name, lat, N/S, lon, E/W, elevation, model,
    icc, ptcor, stcor).  Reading stops at the first blank station name
    (``velest.f:1541``) or at EOF.  Letters are case-folded (``velest.f:1536-1537``);
    latitude/longitude are returned as signed values (S and W negative) **as the file
    says** unless ``legacy_mirror``.  The model column is read but VELEST ignores it
    (``mode=1``, ``velest.f:1543``); VELEST's own output writes alternating 1/2 there for
    ``nsp=2`` runs.  The elevation is an integer (metres); a float such as ``369.0`` in
    the i4 field is an error (VELEST fails on it too).  Characters after the format span
    are ignored except a trailing integer, kept as ``flag``.

    The format line is kept in ``df.attrs["format"]``.
    """
    lines = _lines(text)
    if not lines or not lines[0].strip():
        raise VelestFormatError(f"{source or '<text>'}: station file has no format line")
    fm = lines[0].strip()
    spec = _parse_fortran_format(fm)
    data_kinds = tuple(k for k, _, _ in spec if k != "x")
    if data_kinds != _STATION_FORMAT_FIELDS:
        raise VelestFormatError(
            f"{source or '<text>'}:1: station format {fm!r} must describe name, lat, N/S, "
            "lon, E/W, elevation, model, icc, ptcor, stcor (a,f,a,f,a,i,i,i,f,f)"
        )
    span = sum(w for _, w, _ in spec)
    rows: list[list[object]] = []
    for lineno, line in enumerate(lines[1:], start=2):
        padded = line.ljust(span)
        pos = 0
        vals: list[object] = []
        try:
            for kind, w, d in spec:
                chunk = padded[pos : pos + w]
                pos += w
                if kind == "x":
                    continue
                if kind == "a":
                    vals.append(chunk)
                elif kind == "i":
                    vals.append(_inum(chunk))
                else:
                    vals.append(_fnum(chunk, d))
        except ValueError as exc:
            raise VelestFormatError(
                f"{_where(source, lineno)}: cannot read station record {line!r} with format "
                f"{fm!r}: {exc}"
            ) from exc
        name = str(vals[0])
        if not name.strip():
            break  # blank station name terminates the list (velest.f:1541)
        ns, ew = str(vals[2]).upper(), str(vals[4]).upper()
        if ns not in ("N", "S") or ew not in ("E", "W"):
            raise VelestFormatError(
                f"{_where(source, lineno)}: latitude/longitude letters must be N/S and E/W, "
                f"got {vals[2]!r}/{vals[4]!r} in {line!r}"
            )
        lat = parse_velest_lat(f"{vals[1]}{ns}")
        lon = parse_velest_lon(f"{vals[3]}{ew}")
        tail = line[span:].strip()
        flag = int(tail) if re.fullmatch(r"[+-]?\d+", tail) else None
        rows.append([name.rstrip(), lat, lon, vals[5], vals[6], vals[7], vals[8], vals[9], flag])
    df = pd.DataFrame(rows, columns=STATION_COLUMNS)
    df = df.astype(
        {
            "sta": object,
            "lat": "float64",
            "lon": "float64",
            "elev_m": "int64",
            "model": "int64",
            "icc": "int64",
            "ptcor": "float64",
            "stcor": "float64",
        }
    )
    df["flag"] = pd.array(df["flag"].tolist(), dtype="Int64")
    if legacy_mirror and len(df):
        df["lat"], df["lon"] = legacy_mirror_arrays(df["lat"], df["lon"])
    df.attrs["format"] = fm
    return df


def legacy_mirror_arrays(lat: pd.Series, lon: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`osv.coords.legacy_mirror` for table columns."""
    la, lo = legacy_mirror(lat.to_numpy(), lon.to_numpy())
    return np.asarray(la), np.asarray(lo)


def read_stations(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read a station file; see :func:`parse_stations`."""
    return parse_stations(_read_text(path), source=str(path), **kwargs)  # type: ignore[arg-type]


def format_stations(
    stations: pd.DataFrame,
    *,
    legacy_mirror: bool = False,
    fmt: str = STATION_FORMAT,
    with_flag: bool | None = None,
) -> str:
    """Format a station table as a VELEST station file.

    ``stations`` needs ``sta, lat, lon, elev_m`` (true signed coordinates, elevation m up);
    ``model`` (default 1), ``icc`` (default: running number 1..n; the **highest** icc is
    the reference station whose P delay stays fixed, user guide p. 20), ``ptcor`` and
    ``stcor`` (default 0) are optional.  The file always ends with a blank record (VELEST
    stops there; without it the Fortran read hits EOF).  ``fmt`` must be the standard
    layout; ``with_flag`` appends the ``flag`` column (default: when present and non-null).
    """
    if fmt.replace(" ", "").lower() != STATION_FORMAT.replace(" ", ""):
        raise VelestFormatError(
            f"writer only supports the standard station format {STATION_FORMAT!r}, got {fmt!r}"
        )
    df = stations.reset_index(drop=True)
    if df["sta"].duplicated().any():
        dup = df.loc[df["sta"].duplicated(), "sta"].tolist()
        # legacy files (MC/RK-NMstats_ALL.sta) repeat names with different icc; VELEST
        # accepts that but only one record is matched per pick, so warn.
        logger.warning("duplicate station names in station table: %s", sorted(set(dup))[:5])
    n = len(df)
    model = df["model"] if "model" in df else pd.Series([1] * n)
    icc = df["icc"] if "icc" in df else pd.Series(range(1, n + 1))
    ptcor = df["ptcor"] if "ptcor" in df else pd.Series([0.0] * n)
    stcor = df["stcor"] if "stcor" in df else pd.Series([0.0] * n)
    if with_flag is None:
        with_flag = "flag" in df and bool(df["flag"].notna().any())
    out = [fmt]
    for i in range(n):
        sta = _check_name(df["sta"][i], f"station row {i}")
        lat, lon = float(df["lat"][i]), float(df["lon"][i])
        if legacy_mirror:
            lat, lon = _mirror_pair(lat, lon)
        try:
            lat_s = format_velest_lat(lat)
            lon_s = format_velest_lon(lon)
            elev_s = velest_elevation_field(float(df["elev_m"][i]))
        except ValueError as exc:
            raise VelestFormatError(f"station {sta}: {exc}") from exc
        m, c = int(model[i]), int(icc[i])
        if not 0 <= m <= 9:
            raise VelestFormatError(f"station {sta}: model index {m} does not fit i1")
        if not 0 <= c <= 999:
            raise VelestFormatError(f"station {sta}: icc {c} does not fit i3 (0..999)")
        for what, v in (("ptcor", float(ptcor[i])), ("stcor", float(stcor[i]))):
            if not -9.99 <= v <= 99.99:
                raise VelestFormatError(f"station {sta}: {what} {v} does not fit f5.2")
        line = (
            f"{sta:<4s}{lat_s} {lon_s} {elev_s} {m:1d} {c:3d} "
            f"{float(ptcor[i]):5.2f}  {float(stcor[i]):5.2f}"
        )
        if with_flag and "flag" in df and pd.notna(df["flag"][i]):
            line += f"{int(df['flag'][i]):4d}"
        out.append(line)
    out.append("")
    return "".join(ln + "\n" for ln in out)


def write_stations(path: str | Path, stations: pd.DataFrame, **kwargs: object) -> Path:
    """Write a station file; see :func:`format_stations`."""
    return _write_text(path, format_stations(stations, **kwargs))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------
# Model file
# ----------------------------------------------------------------------------------------

MODEL_COMMENT = "        vel,depth,vdamp,phase (f5.2,5x,f7.2,2x,f7.3,3x,a1)"
MODEL_COLUMNS = ["vel", "depth", "vdamp", "reflector"]


@dataclass(eq=False)
class ModelBlock:
    """One velocity structure (P or S): layer table plus the free-text fields.

    ``layers`` has ``vel`` (km/s), ``depth`` (km, top of layer, positive down, may be
    negative), ``vdamp`` (relative damping, 0..999.99) and ``reflector`` (``' '`` or
    ``'m'``/``'M'``).  ``label`` is the text after the first layer line (``a40``),
    ``comment`` the text after the layer count on the count line (ignored by VELEST).
    """

    layers: pd.DataFrame
    label: str = ""
    comment: str = MODEL_COMMENT

    def __post_init__(self) -> None:
        lay = self.layers.reset_index(drop=True).copy()
        if "vdamp" not in lay:
            lay["vdamp"] = 1.0
        if "reflector" not in lay:
            lay["reflector"] = " "
        lay = lay[MODEL_COLUMNS]
        lay["vel"] = lay["vel"].astype("float64")
        lay["depth"] = lay["depth"].astype("float64")
        lay["vdamp"] = lay["vdamp"].astype("float64")
        lay["reflector"] = lay["reflector"].astype(object).fillna(" ")
        self.layers = lay
        self.validate()

    def validate(self) -> None:
        """Check ranges (the f5.2/f7.2/f7.3 fields) and layer ordering."""
        lay = self.layers
        if not 1 <= len(lay) <= 999:
            raise VelestFormatError(f"model block has {len(lay)} layers (need 1..999)")
        if not ((lay["vel"] > 0) & (lay["vel"] < 100)).all():
            raise VelestFormatError("layer velocities must be in (0, 100) km/s (f5.2)")
        if not (lay["depth"].abs() < 9999.995).all():
            raise VelestFormatError("layer depths do not fit f7.2")
        if not ((lay["vdamp"] >= 0) & (lay["vdamp"] <= 999.999)).all():
            raise VelestFormatError("vdamp must be in 0..999.99 (user guide p. 21)")
        if (np.diff(lay["depth"].to_numpy()) <= 0).any():
            raise VelestFormatError("layer top depths must be strictly increasing")
        if not lay["reflector"].isin([" ", "m", "M"]).all():
            raise VelestFormatError("reflector mark must be blank, 'm' or 'M' (velest.f:1394)")

    def equals(self, other: ModelBlock) -> bool:
        """Equality of layers and text fields."""
        return (
            self.layers.equals(other.layers)
            and self.label == other.label
            and self.comment == other.comment
        )


@dataclass(eq=False)
class VelestModel:
    """A VELEST model file: title, P block and optional S block.

    VELEST reads ``nmod`` blocks, with ``nmod = 2`` only for ``nsp = 2``
    (``velest.f:1120-1128``); otherwise only the P block is read.
    """

    title: str
    p: ModelBlock
    s: ModelBlock | None = None

    def equals(self, other: VelestModel) -> bool:
        """Equality of title and blocks."""
        if self.title != other.title or (self.s is None) != (other.s is None):
            return False
        if not self.p.equals(other.p):
            return False
        return self.s is None or other.s is None or self.s.equals(other.s)

    @property
    def nmod(self) -> int:
        """Number of blocks (1 or 2)."""
        return 1 if self.s is None else 2


def parse_model(text: str, *, nmod: int | None = None, source: str | None = None) -> VelestModel:
    """Parse model-file text (input files and the ``velout.mod`` output variant).

    Layout (``velest.f:1373-1398``): title (``a40``), then per block a layer count (the
    first integer on the line; VELEST reads ``i3`` but its own ``velout.mod`` writes the
    count list-directed) and one ``(f5.2,5x,f7.2,2x,f7.3,3x,a1[,1x,a40])`` line per layer.
    ``nmod=None`` reads an S block when more content follows; ``nmod=2`` requires it and
    ``nmod=1`` ignores it.  CRLF line ends (legacy files) are accepted.
    """
    if nmod not in (None, 1, 2):
        raise ValueError("nmod must be None, 1 or 2")
    lines = _lines(text)
    if not lines:
        raise VelestFormatError(f"{source or '<text>'}: empty model file")
    title = lines[0][:40].rstrip()
    pos = 1
    blocks: list[ModelBlock] = []
    want = 2 if nmod is None else nmod
    while len(blocks) < want:
        while pos < len(lines) and not lines[pos].strip():
            pos += 1
        if pos >= len(lines):
            break
        cl = lines[pos]
        pos += 1
        toks = cl.split()
        try:
            count = int(toks[0])
        except (IndexError, ValueError) as exc:
            raise VelestFormatError(
                f"{_where(source, pos)}: expected the layer count, got {cl!r}"
            ) from exc
        comment_start = cl.index(toks[0]) + len(toks[0])
        comment = cl[comment_start:] if len(cl) > comment_start else ""
        rows: list[list[object]] = []
        label = ""
        for k in range(count):
            if pos >= len(lines):
                raise VelestFormatError(
                    f"{source or '<text>'}: model block {len(blocks) + 1} announces {count} "
                    f"layers but the file ends after {k}"
                )
            ll = lines[pos].ljust(31)
            pos += 1
            try:
                vel = _fnum(ll[0:5], 2)
                depth = _fnum(ll[10:17], 2)
                vdamp = _fnum(ll[19:26], 3)
            except ValueError as exc:
                raise VelestFormatError(
                    f"{_where(source, pos)}: cannot read layer {lines[pos - 1]!r}: {exc}"
                ) from exc
            rows.append([vel, depth, vdamp, ll[29]])
            if k == 0:
                label = lines[pos - 1][31:71].rstrip() if len(lines[pos - 1]) > 31 else ""
        blocks.append(
            ModelBlock(pd.DataFrame(rows, columns=MODEL_COLUMNS), label=label, comment=comment)
        )
    if not blocks:
        raise VelestFormatError(f"{source or '<text>'}: model file has no velocity block")
    if nmod == 2 and len(blocks) < 2:
        raise VelestFormatError(
            f"{source or '<text>'}: nmod=2 (nsp=2) needs an S block but the file has only a P block"
        )
    return VelestModel(title=title, p=blocks[0], s=blocks[1] if len(blocks) > 1 else None)


def read_model(path: str | Path, *, nmod: int | None = None) -> VelestModel:
    """Read a model file; see :func:`parse_model`."""
    return parse_model(_read_text(path), nmod=nmod, source=str(path))


def format_model(model: VelestModel) -> str:
    """Format a model file (``f5.2,5x,f7.2,2x,f7.3,3x,a1``; first layer line gets the label)."""
    if len(model.title) > 40:
        raise VelestFormatError("model title is longer than 40 characters (a40)")
    out = [model.title]
    for block in [model.p, *([model.s] if model.s is not None else [])]:
        block.validate()
        out.append(f"{f' {len(block.layers):02d}':>3s}{block.comment}".rstrip())
        for k, r in enumerate(block.layers.itertuples(index=False)):
            line = f"{r.vel:5.2f}{'':5s}{r.depth:7.2f}  {r.vdamp:7.3f}   {r.reflector}"
            if k == 0 and block.label:
                line += f" {block.label[:40]}"
            out.append(
                line.rstrip() if r.reflector == " " and not (k == 0 and block.label) else line
            )
    return "".join(ln + "\n" for ln in out)


def write_model(path: str | Path, model: VelestModel) -> Path:
    """Write a model file."""
    return _write_text(path, format_model(model))


# ----------------------------------------------------------------------------------------
# Control file (velest.cmn)
# ----------------------------------------------------------------------------------------

#: File-name lines of velest.cmn in order (``velest.f:1025-1062``): attribute names.
_FILE_FIELDS = (
    "modelfile",
    "stationfile",
    "seismofile",
    "regnamfile",
    "regkoordfile",
    "topo1file",
    "topo2file",
    "phasefile",
    "shotfile",
    "outfile",
    "velfile",
    "cnvfile",
    "stafile",
    "smpfile",
    "rayfile",
    "drvfile",
    "alefile",
    "dsprfile",
    "rflfile",
    "rfrfile",
    "resfile",
)
_EXTENDED_FILE_FIELDS = _FILE_FIELDS[13:]


def _bin(name: str, v: int) -> None:
    if v not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {v}")


@dataclass
class VelestControl:
    """All parameters of ``velest.cmn`` (user guide section 6.1, ``velest.f:985-1063``).

    Field names are the VELEST variable names.  ``olat``/``olon`` are the **file** values
    (north positive; **west positive**): construct from true coordinates with
    :meth:`from_origin` (uses :func:`osv.coords.velest_olat` / ``velest_olon``) and read
    them back with :meth:`origin`.  Empty strings are blank file-name lines.

    ``nmod`` is derived from ``nsp`` when ``None`` (VELEST forces ``nmod = 2`` for
    ``nsp = 2`` and 1 otherwise, ``velest.f:1119-1128``); an inconsistent value raises.
    """

    title: str = "osv VELEST run"
    olat: float = 0.0
    olon: float = 0.0
    icoordsystem: int = 0
    zshift: float = 0.0
    itrial: int = 0
    ztrial: float = 0.0
    ised: int = 0
    neqs: int = 0
    nshot: int = 0
    rotate: float = 0.0
    isingle: int = 0
    iresolcalc: int = 0
    dmax: float = 150.0
    itopo: int = 0
    zmin: float = 0.0
    veladj: float = 0.2
    zadj: float = 5.0
    lowveloclay: int = 1
    nsp: int = 2
    swtfac: float = 0.75
    vpvs: float = 1.73
    nmod: int | None = None
    othet: float = 0.01
    xythet: float = 0.01
    zthet: float = 0.01
    vthet: float = 1.0
    stathet: float = 0.01
    nsinv: int = 1
    nshcor: int = 0
    nshfix: int = 0
    iuseelev: int = 1
    iusestacorr: int = 0
    iturbo: int = 1
    icnvout: int = 1
    istaout: int = 1
    ismpout: int = 0
    irayout: int = 0
    idrvout: int = 0
    ialeout: int = 0
    idspout: int = 0
    irflout: int = 0
    irfrout: int = 0
    iresout: int = 0
    delmin: float = 0.001
    ittmax: int = 9
    invertratio: int = 1
    modelfile: str = ""
    stationfile: str = ""
    seismofile: str = ""
    regnamfile: str = ""
    regkoordfile: str = ""
    topo1file: str = ""
    topo2file: str = ""
    phasefile: str = ""
    shotfile: str = ""
    outfile: str = "velest.out"
    velfile: str = ""
    cnvfile: str = "velout.cnv"
    stafile: str = "velout.sta"
    smpfile: str = ""
    rayfile: str = ""
    drvfile: str = ""
    alefile: str = ""
    dsprfile: str = ""
    rflfile: str = ""
    rfrfile: str = ""
    resfile: str = ""

    def __post_init__(self) -> None:
        if self.nmod is None:
            self.nmod = 2 if self.nsp == 2 else 1
        self.validate()

    @classmethod
    def from_origin(cls, lat: float, lon_east: float, **kwargs: object) -> VelestControl:
        """Build from the true origin (``lat`` north+, ``lon_east`` east+)."""
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"latitude {lat} outside [-90, 90]")
        # round away the float noise of wrap_lon so that file values are tidy decimals
        olon = round(float(velest_olon(lon_east)), 9)
        return cls(olat=velest_olat(lat), olon=olon, **kwargs)  # type: ignore[arg-type]

    def origin(self, *, legacy_mirror: bool = False) -> tuple[float, float]:
        """True ``(lat, lon_east)`` of the origin as the file states it.

        With ``legacy_mirror`` the legacy NZ convention (point mirror) is undone.
        """
        # velest_olon(lon_east) = -lon_east is an involution, so it also undoes itself
        lat, lon = velest_olat(self.olat), round(float(velest_olon(self.olon)), 9)
        if legacy_mirror:
            lat, lon = _mirror_pair(lat, lon)
        return float(lat), float(lon)

    def validate(self) -> None:
        """Check every field range; raise ``ValueError`` naming the field."""
        if not -90.0 <= self.olat <= 90.0:
            raise ValueError(f"olat must be in [-90, 90], got {self.olat}")
        if not -180.0 <= self.olon <= 180.0:
            raise ValueError(f"olon must be in [-180, 180], got {self.olon}")
        if self.icoordsystem not in (0, 1, 2):
            raise ValueError(f"icoordsystem must be 0, 1 or 2, got {self.icoordsystem}")
        if self.itrial not in (0, 1):
            raise ValueError(f"itrial must be 0 or 1, got {self.itrial}")
        if self.ised not in (0, 1, 2):
            raise ValueError(f"ised must be 0, 1 or 2, got {self.ised}")
        if self.neqs < 0 or self.nshot < 0:
            raise ValueError("neqs and nshot must be >= 0")
        if not -360.0 <= self.rotate <= 360.0:
            raise ValueError(f"rotate must be in [-360, 360], got {self.rotate}")
        if self.isingle not in (0, 1):
            raise ValueError(f"isingle must be 0 or 1, got {self.isingle}")
        _bin("iresolcalc", self.iresolcalc)
        if self.dmax < 0:
            raise ValueError(f"dmax must be >= 0, got {self.dmax}")
        if self.itopo not in (0, 1, 2):
            raise ValueError(f"itopo must be 0, 1 or 2, got {self.itopo}")
        if self.veladj < 0 or self.zadj < 0:
            raise ValueError("veladj and zadj must be >= 0")
        _bin("lowveloclay", self.lowveloclay)
        if self.nsp not in (1, 2, 3):
            raise ValueError(f"nsp must be 1, 2 or 3, got {self.nsp}")
        if self.swtfac < 0:
            raise ValueError(f"swtfac must be >= 0, got {self.swtfac}")
        if self.vpvs <= 0:
            raise ValueError(f"vpvs must be > 0, got {self.vpvs}")
        expected_nmod = 2 if self.nsp == 2 else 1
        if self.nmod != expected_nmod:
            raise ValueError(
                f"nmod must be {expected_nmod} for nsp={self.nsp} (VELEST overrides it), "
                f"got {self.nmod}"
            )
        for name in ("othet", "xythet", "zthet", "vthet", "stathet", "delmin"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("nsinv", "nshcor", "nshfix", "iuseelev", "iusestacorr", "iturbo", "icnvout"):
            _bin(name, getattr(self, name))
        if self.istaout not in (0, 1, 2):
            raise ValueError(f"istaout must be 0, 1 or 2, got {self.istaout}")
        for name in (
            "ismpout",
            "irayout",
            "idrvout",
            "ialeout",
            "idspout",
            "irflout",
            "irfrout",
            "iresout",
        ):
            _bin(name, getattr(self, name))
        if not 0 <= self.ittmax <= 99:
            raise ValueError(f"ittmax must be in 0..99, got {self.ittmax}")
        if not 0 <= self.invertratio <= 99:
            raise ValueError(f"invertratio must be in 0..99, got {self.invertratio}")
        if self.title.startswith("*") or len(self.title) > 80 or "\n" in self.title:
            raise ValueError("title must be one line of <= 80 chars not starting with '*'")
        for name in _FILE_FIELDS:
            v = getattr(self, name)
            if len(v) > 80 or v.startswith("*") or "\n" in v:
                raise ValueError(f"{name}: file name must be <= 80 chars, not start with '*'")
            if v != v.strip():
                raise ValueError(f"{name}: file name {v!r} has leading/trailing blanks")

    def equals(self, other: VelestControl) -> bool:
        """Field-wise equality."""
        return all(getattr(self, f.name) == getattr(other, f.name) for f in fields(self))


_CMN_HEADER = """\
******* CONTROL-FILE FOR PROGRAM  V E L E S T  (28-SEPT1993) *******
***
*** ( all lines starting with  *  are ignored! )
*** ( where no filename is specified,
***   leave the line BLANK. Do NOT delete!)
***
*** next line contains a title (printed on output):
"""


_F_MAX = 1e15


def _f(x: float, width: int, decimals: int) -> str:
    """Right-aligned ``F{width}.{decimals}`` that gains decimals when needed to be exact.

    The legacy layout (``decimals``) is kept whenever it represents the value; VELEST reads
    the file list-directed, so a longer field is harmless -- but only if it stays separated
    from its neighbour.  A value wider than ``width`` (e.g. ``vthet=1e8``) would otherwise
    fuse with the previous field into one bad token, so at least one leading blank is
    always kept.  Plain fixed-point is used (no exponent) and non-finite or absurdly large
    values are rejected.
    """
    if not math.isfinite(x) or abs(x) >= _F_MAX:
        raise ValueError(f"value {x!r} cannot be written to velest.cmn (need finite, |x| < 1e15)")
    txt = f"{x:{width}.9f}"
    for d in range(decimals, 10):
        cand = f"{x:{width}.{d}f}"
        if abs(float(cand) - x) <= 1e-9 * max(1.0, abs(x)):
            txt = cand
            break
    return txt if txt.startswith(" ") else " " + txt


def format_control(ctrl: VelestControl, *, extended_files: bool = True) -> str:
    """Format ``velest.cmn`` with the layout of ``MonteCarlo/velest.cmn``.

    ``extended_files=False`` writes the short 24-line form (VELEST accepts 24 or 32
    non-comment lines, ``velest.f:1011``); then ``smpfile`` ... ``resfile`` must be empty.
    Blank file names stay blank lines (VELEST substitutes ``velout.*`` defaults for the
    output files ``outfile``, ``velfile``, ``cnvfile``, ``stafile`` and, with 32 lines,
    the extended ones).  Comment lines start with ``*`` in column 1.
    """
    ctrl.validate()
    c = ctrl
    if not extended_files and any(getattr(c, n) for n in _EXTENDED_FILE_FIELDS):
        raise ValueError("extended_files=False but smpfile..resfile are set")
    out: list[str] = [_CMN_HEADER.rstrip("\n")]
    out.append(c.title)
    out.append("***  olat       olon   icoordsystem      zshift   itrial ztrial    ised")
    out.append(
        f"{_f(c.olat, 9, 4)}{_f(c.olon, 12, 4)}{c.icoordsystem:7d}{_f(c.zshift, 17, 3)}"
        f"{c.itrial:7d}{_f(c.ztrial, 9, 2)}{c.ised:8d}"
    )
    out.append("***")
    out.append("*** neqs   nshot   rotate")
    out.append(f"{c.neqs:8d}{c.nshot:7d}{_f(c.rotate, 9, 1)}")
    out.append("***")
    out.append("*** isingle   iresolcalc")
    out.append(f"{c.isingle:8d}{c.iresolcalc:11d}")
    out.append("***")
    out.append("*** dmax    itopo    zmin     veladj    zadj   lowveloclay")
    out.append(
        f"{_f(c.dmax, 9, 1)}{c.itopo:6d}{_f(c.zmin, 10, 2)}{_f(c.veladj, 11, 2)}"
        f"{_f(c.zadj, 8, 2)}{c.lowveloclay:8d}"
    )
    out.append("***")
    out.append("*** nsp    swtfac   vpvs       nmod")
    out.append(f"{c.nsp:6d}{_f(c.swtfac, 10, 2)}{_f(c.vpvs, 8, 2)}{c.nmod:10d}")
    out.append("***")
    out.append("***   othet   xythet    zthet    vthet   stathet")
    out.append(
        f"{_f(c.othet, 11, 2)}{_f(c.xythet, 9, 2)}{_f(c.zthet, 9, 2)}{_f(c.vthet, 9, 1)}"
        f"{_f(c.stathet, 10, 2)}"
    )
    out.append("***")
    out.append("*** nsinv   nshcor   nshfix     iuseelev    iusestacorr")
    out.append(f"{c.nsinv:8d}{c.nshcor:8d}{c.nshfix:8d}{c.iuseelev:12d}{c.iusestacorr:13d}")
    out.append("***")
    out.append("*** iturbo    icnvout   istaout   ismpout")
    out.append(f"{c.iturbo:8d}{c.icnvout:10d}{c.istaout:10d}{c.ismpout:10d}")
    out.append("***")
    out.append("*** irayout   idrvout   ialeout   idspout   irflout   irfrout   iresout")
    out.append(
        f"{c.irayout:8d}{c.idrvout:10d}{c.ialeout:10d}{c.idspout:10d}"
        f"{c.irflout:10d}{c.irfrout:10d}{c.iresout:10d}"
    )
    out.append("***")
    out.append("*** delmin   ittmax   invertratio")
    out.append(f"{_f(c.delmin, 9, 3)}{f'{c.ittmax:02d}':>8s}{c.invertratio:11d}")

    def fname(label: str, attr: str) -> None:
        out.append("***")
        out.append(f"*** {label}")
        out.append(getattr(c, attr))

    fname("Modelfile:", "modelfile")
    fname("Stationfile:", "stationfile")
    fname("Seismofile:", "seismofile")
    fname("File with region names:", "regnamfile")
    fname("File with region coordinates:", "regkoordfile")
    fname("File #1 with topo data:", "topo1file")
    fname("File #2 with topo data:", "topo2file")
    out.append("***")
    out.append("*** DATA INPUT files:")
    fname("File with Earthquake data:", "phasefile")
    fname("File with Shot data:", "shotfile")
    out.append("***")
    out.append("*** OUTPUT files:")
    fname("Main print output file:", "outfile")
    fname("File with single event locations:", "velfile")
    fname("File with final hypocenters in *.cnv format:", "cnvfile")
    fname("File with new station corrections:", "stafile")
    if extended_files:
        fname("File with summary cards (e.g. for plotting):", "smpfile")
        fname("File with raypoints:", "rayfile")
        fname("File with derivatives:", "drvfile")
        fname("File with ALEs:", "alefile")
        fname("File with Dirichlet spreads:", "dsprfile")
        fname("File with reflection points:", "rflfile")
        fname("File with refraction points:", "rfrfile")
        fname("File with residuals:", "resfile")
    out.append("***")
    out.append("******* END OF THE CONTROL-FILE FOR PROGRAM  V E L E S T  *******")
    return "\n".join(out) + "\n"


def write_control(path: str | Path, ctrl: VelestControl, *, extended_files: bool = True) -> Path:
    """Write ``velest.cmn`` (VELEST reads exactly this name from its working directory)."""
    return _write_text(path, format_control(ctrl, extended_files=extended_files))


def _tokens(line: str) -> list[str]:
    return [t for t in re.split(r"[\s,]+", line.strip()) if t]


def _to_int(tok: str, name: str) -> int:
    try:
        return int(tok)
    except ValueError as exc:
        raise VelestFormatError(f"control file: {name} must be an integer, got {tok!r}") from exc


def _to_float(tok: str, name: str) -> float:
    try:
        return float(tok.replace("D", "e").replace("d", "e"))
    except ValueError as exc:
        raise VelestFormatError(f"control file: {name} must be a number, got {tok!r}") from exc


def parse_control(text: str, *, source: str | None = None) -> VelestControl:
    """Parse ``velest.cmn`` text.

    Mirrors ``INPUTPARAM`` (``velest.f:1000-1062``): a line is a comment only when it
    starts with ``*`` in column 1; the remaining lines (including blank ones) must number
    24 or 32; numeric lines are free format (blanks or commas).  A trailing blank line
    that VELEST would count raises the same error as in VELEST.
    """
    where = source or "<text>"
    cards = [ln for ln in _lines(text) if not ln.startswith("*")]
    if len(cards) not in (24, 32):
        raise VelestFormatError(
            f"{where}: control file has {len(cards)} non-comment lines; VELEST needs exactly "
            "24 or 32 (blank file-name lines count, do not delete them)"
        )

    def need(idx: int, n: int, names: str) -> list[str]:
        toks = _tokens(cards[idx])
        if len(toks) < n:
            raise VelestFormatError(
                f"{where}: control line {idx + 1} needs {n} values ({names}), got {cards[idx]!r}"
            )
        return toks

    kw: dict[str, object] = {"title": cards[0].rstrip()}
    t = need(1, 7, "olat olon icoordsystem zshift itrial ztrial ised")
    kw.update(
        olat=_to_float(t[0], "olat"),
        olon=_to_float(t[1], "olon"),
        icoordsystem=_to_int(t[2], "icoordsystem"),
        zshift=_to_float(t[3], "zshift"),
        itrial=_to_int(t[4], "itrial"),
        ztrial=_to_float(t[5], "ztrial"),
        ised=_to_int(t[6], "ised"),
    )
    t = need(2, 3, "neqs nshot rotate")
    kw.update(
        neqs=_to_int(t[0], "neqs"), nshot=_to_int(t[1], "nshot"), rotate=_to_float(t[2], "rotate")
    )
    t = need(3, 2, "isingle iresolcalc")
    kw.update(isingle=_to_int(t[0], "isingle"), iresolcalc=_to_int(t[1], "iresolcalc"))
    t = need(4, 6, "dmax itopo zmin veladj zadj lowveloclay")
    kw.update(
        dmax=_to_float(t[0], "dmax"),
        itopo=_to_int(t[1], "itopo"),
        zmin=_to_float(t[2], "zmin"),
        veladj=_to_float(t[3], "veladj"),
        zadj=_to_float(t[4], "zadj"),
        lowveloclay=_to_int(t[5], "lowveloclay"),
    )
    t = need(5, 4, "nsp swtfac vpvs nmod")
    kw.update(
        nsp=_to_int(t[0], "nsp"),
        swtfac=_to_float(t[1], "swtfac"),
        vpvs=_to_float(t[2], "vpvs"),
        nmod=_to_int(t[3], "nmod"),
    )
    t = need(6, 5, "othet xythet zthet vthet stathet")
    for name, tok in zip(("othet", "xythet", "zthet", "vthet", "stathet"), t, strict=False):
        kw[name] = _to_float(tok, name)
    t = need(7, 5, "nsinv nshcor nshfix iuseelev iusestacorr")
    for name, tok in zip(("nsinv", "nshcor", "nshfix", "iuseelev", "iusestacorr"), t, strict=False):
        kw[name] = _to_int(tok, name)
    t = need(8, 4, "iturbo icnvout istaout ismpout")
    for name, tok in zip(("iturbo", "icnvout", "istaout", "ismpout"), t, strict=False):
        kw[name] = _to_int(tok, name)
    t = need(9, 7, "irayout idrvout ialeout idspout irflout irfrout iresout")
    names7 = ("irayout", "idrvout", "ialeout", "idspout", "irflout", "irfrout", "iresout")
    for name, tok in zip(names7, t, strict=False):
        kw[name] = _to_int(tok, name)
    t = need(10, 3, "delmin ittmax invertratio")
    kw.update(
        delmin=_to_float(t[0], "delmin"),
        ittmax=_to_int(t[1], "ittmax"),
        invertratio=_to_int(t[2], "invertratio"),
    )
    nfile = len(cards) - 11
    for name, card in zip(_FILE_FIELDS[:nfile], cards[11:], strict=False):
        kw[name] = card.strip()
    for name in _FILE_FIELDS[nfile:]:
        kw[name] = ""
    try:
        return VelestControl(**kw)  # type: ignore[arg-type]
    except ValueError as exc:
        raise VelestFormatError(f"{where}: {exc}") from exc


def read_control(path: str | Path) -> VelestControl:
    """Read a ``velest.cmn``; see :func:`parse_control`."""
    return parse_control(_read_text(path), source=str(path))
