"""Control and input files of the generators ``arraygen``, ``moddata``, ``obsdata``, ``synthdata``.

``fm3d`` itself needs ``sources.in``/``receivers.in``; FMTOMO ships small programs that build
them from simpler input.  osv writes the final files directly, but these control files are
part of the package's examples and of the manual, so they are covered here.

* :class:`Arraygen` ``arraygen.in`` - ``arraygen.f90:24-29`` (``(a20)`` name, then
  list-directed).
* :class:`Synthdata` ``synthdata.in`` - ``synthdata.f90:44-54`` (``(a26)`` names).  Output
  ``otimes.dat`` columns: ``synthdata.f90:122`` (``i1,i2,i3,i4,tt,cv``).
* :class:`Moddata` ``moddata.in`` - ``moddata.f90:56-67`` and ``:105-132``: 10 header records,
  then per source file 3 banner records, ``(a26)`` name, local/teleseismic, derivative flag,
  number of paths, per path: number of steps, one skipped comment record, sequence, velocity
  types.
* :class:`Obsdata` ``obsdata.in`` - ``obsdata.f90:68-81`` and ``:141-179``: same group layout;
  the Extras ``obsdatai.f90:89-90`` adds ``extract source-receiver`` and its file after the
  times file.

Input formats of those programs (all lat, lon, depth order, unlike ``sources.in``): moddata
source files ``n`` then ``lat lon depth [covlat covlon covdep]`` (+ ``phase`` record for
teleseisms, ``moddata.f90:140-147``); receiver files ``n`` then ``lat lon depth``
(``arraygen.out``); obsdata pick files ``n`` (+ ``lat lon depth`` and ``phase`` for teleseisms)
then ``rlat rlon rdep [tt err]`` (``obsdata.f90:213-224``); obsdata source files per source
``location`` (local only), number of pick files, then ``path_type rank filename`` records
(``obsdata.f90:104-130,195-203``): the first integer selects the path of the group's path list,
the second is the position among the source's emitted paths (manual p. 44-45 and
``obsdata.f90:65-66``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._control import ControlFile, F
from ._fortran import FmtomoFormatError, RecordReader, fint, fline, read_text, write_text
from .sources import PathSpec, Source

__all__ = [
    "Arraygen",
    "Moddata",
    "ObsSource",
    "Obsdata",
    "PickFile",
    "SourceGroup",
    "Synthdata",
    "format_input_receivers",
    "format_input_sources",
    "format_obs_sources",
    "format_picks",
    "parse_input_receivers",
    "parse_input_sources",
    "parse_obs_sources",
    "parse_picks",
]

_C66 = "c" * 66


@dataclass
class Arraygen(ControlFile):
    """``arraygen.in``: a regular lat/lon grid of receivers (or sources) at one depth."""

    output_file: str = "arraygen.out"
    lat_bounds: tuple[float, float] = (-32.0, -34.0)  # (north, south)
    lon_bounds: tuple[float, float] = (138.5, 141.5)  # (west, east)
    depth_km: float = 0.0
    n_lat_lon: tuple[int, int] = (10, 10)

    _pad = 29
    _layout = (
        F("output_file", "a", "Output file name", 20),
        F("lat_bounds", "ff", "N-S bounds of grid", joins=("   ",)),
        F("lon_bounds", "ff", "W-E bounds of grid", joins=("   ",)),
        F("depth_km", "f", "Depth of receiver grid"),
        F("n_lat_lon", "ii", "Number of receivers in lat and long", joins=("    ",)),
    )

    def points(self) -> pd.DataFrame:
        """The generated points (``arraygen.f90:33-47``): latitude slowest, south to north."""
        thn, ths = self.lat_bounds
        phw, phe = self.lon_bounds
        nth, nph = self.n_lat_lon
        th = np.array(
            [(thn + ths) / 2.0]
            if nth == 1
            else [ths + i * (thn - ths) / (nth - 1) for i in range(nth)]
        )
        ph = np.array(
            [(phw + phe) / 2.0]
            if nph == 1
            else [phw + j * (phe - phw) / (nph - 1) for j in range(nph)]
        )
        lat, lon = np.meshgrid(th, ph, indexing="ij")
        return pd.DataFrame(
            {"lat": lat.ravel(), "lon": lon.ravel(), "depth_km": float(self.depth_km)}
        )


@dataclass
class Synthdata(ControlFile):
    """``synthdata.in``: turn model times into pseudo-observed ``otimes.dat``."""

    input_times_file: str = "syntimes.dat"
    data_uncertainty_s: float = 0.1
    output_file: str = "otimes.dat"
    source_file: str = "sourcesref.in"
    rtimes_file: str = "rtimes.dat"
    dc_offset_s: float = 0.0
    add_noise: int = 1
    noise_sd_local_s: float = 0.1
    noise_sd_teleseismic_s: float = 0.077
    weight_by: int = 1
    random_seed: int = 99827374

    _pad = 31
    _layout = (
        F("input_times_file", "a", "Input synthetic times", 26),
        F("data_uncertainty_s", "f", "Data covariance (s)"),
        F("output_file", "a", 'Output "observed traveltime" file', 26),
        F("source_file", "a", "Source file", 26),
        F("rtimes_file", "a", "Reference teleseismic times", 26),
        F("dc_offset_s", "f", "DC offset to traveltimes"),
        F("add_noise", "i", "Add noise to synthetic times (0=no,1=yes)"),
        F("noise_sd_local_s", "f", "Standard deviation of added noise (local)"),
        F("noise_sd_teleseismic_s", "f", "Standard deviation of added noise (teleseismic)"),
        F("weight_by", "i", "Weight with standard deviation (1) or noise (2)"),
        F("random_seed", "i", "Random seed for noise generation"),
    )


# --------------------------------------------------------------------------------------
# moddata.in / obsdata.in
# --------------------------------------------------------------------------------------

_ORDINALS = [
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth",
]


@dataclass
class SourceGroup:
    """One input source file of ``moddata.in``/``obsdata.in`` with its path definitions."""

    input_file: str = "sources.in"
    teleseismic: bool = False
    compute_source_derivatives: bool = False
    paths: list[PathSpec] = field(default_factory=lambda: [PathSpec.direct()])


def _kv(value: str, comment: str, pad: int = 28) -> str:
    return f"{value:<{pad}}c: {comment}"


def _check_a(name: str, what: str, width: int = 26) -> str:
    if len(name) > width:
        raise FmtomoFormatError(
            f"{what}: {name!r} is {len(name)} characters; the Fortran reader (a{width}) takes only "
            f"the first {width}"
        )
    return name


def _format_group(g: SourceGroup, index: int, *, lowercase_first: bool) -> list[str]:
    word = _ORDINALS[index] if index < len(_ORDINALS) else f"{index + 1}th"
    if index == 0 and lowercase_first:
        word = word.lower()
    out = [
        _C66,
        f"c {word} input source file and associated path information",
        _C66,
        _kv(_check_a(g.input_file, "source group input file"), "Input source file"),
        _kv("1" if g.teleseismic else "0", "Local (0) or teleseismic sources (1)"),
        _kv(
            "1" if g.compute_source_derivatives else "0", "Compute source derivatives (0=no, 1=yes)"
        ),
        _kv(str(len(g.paths)), "Number of paths from these sources"),
    ]
    for j, p in enumerate(g.paths, start=1):
        out.append(_kv(str(p.n_steps), f"Number of path segments for path {j}"))
        out.append("c: Path sequence information plus velocity fields below")
        pairs = [f"{a} {b}" for a, b in p.steps]
        out.append("  ".join(pairs))
        out.append(
            "".join(f"{v:<{len(s) + 2}}" for v, s in zip(p.vtypes, pairs, strict=True)).rstrip()
        )
    return out


def _parse_group(rd: RecordReader) -> SourceGroup:
    rd.skip(3)
    name = rd.a(26)
    lot, csdif, nps = rd.int(), rd.int(), rd.int()
    paths = []
    for _ in range(nps):
        n = rd.int()
        rd.skip()
        seq = rd.ints(2 * n)
        vt = rd.ints(n)
        paths.append(PathSpec.from_sequence(seq, vt))
    return SourceGroup(name, bool(lot), bool(csdif), paths)


_HEAD_BANNER_MOD = (
    _C66,
    "c Input file for generating source and receiver files",
    "c for use by 3D FMM code. Every receiver must",
    "c detect a traveltime from every source.",
    _C66,
)
_HEAD_BANNER_OBS = (
    _C66,
    "c Input file for generating source and receiver files",
    "c for use by 3D FMM code. Will also generate traveltime",
    "c file if required.",
    _C66,
)


@dataclass
class Moddata:
    """``moddata.in``: build ``sources.in``/``receivers.in`` for synthetic tests."""

    sources_file: str = "sources.in"
    receivers_file: str = "receivers.in"
    source_derivs_file: str = "sourcederivs.in"
    input_receiver_file: str = "arraygen.out"
    groups: list[SourceGroup] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str, source: str | Path | None = None) -> Moddata:
        rd = RecordReader(text, source)
        rd.skip(5)
        names = [rd.a(26) for _ in range(4)]
        n = rd.int()
        return cls(*names, [_parse_group(rd) for _ in range(n)])

    @classmethod
    def read(cls, path: str | Path) -> Moddata:
        return cls.parse(read_text(path), path)

    def format(self) -> str:
        out = [
            *_HEAD_BANNER_MOD,
            _kv(_check_a(self.sources_file, "sources_file"), "Output source file"),
            _kv(_check_a(self.receivers_file, "receivers_file"), "Output receiver file"),
            _kv(
                _check_a(self.source_derivs_file, "source_derivs_file"),
                "Output source derivative file",
            ),
            _kv(_check_a(self.input_receiver_file, "input_receiver_file"), "Input receiver file"),
            _kv(str(len(self.groups)), "Number of input source files"),
        ]
        for i, g in enumerate(self.groups):
            out.extend(_format_group(g, i, lowercase_first=False))
        return "\n".join(out) + "\n"

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.format())


@dataclass
class Obsdata:
    """``obsdata.in``: build ``sources.in``, ``receivers.in`` and ``otimes.dat`` from pick files.

    The Extras variant (``obsdatai.in``) also has ``source_receiver_file`` (written when not
    ``None``; parse with ``extras=True``).
    """

    sources_file: str = "sources.in"
    receivers_file: str = "receivers.in"
    source_derivs_file: str = "sourcederivs.in"
    picks_dir: str = "picks/"
    extract_times: bool = True
    otimes_file: str = "otimes.dat"
    source_receiver_file: str | None = None
    groups: list[SourceGroup] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str, source: str | Path | None = None, *, extras: bool = False) -> Obsdata:
        rd = RecordReader(text, source)
        rd.skip(5)
        names = [rd.a(26) for _ in range(4)]
        n = rd.int()
        extt = rd.int()
        otimes = rd.a(26)
        srf = None
        if extras:
            rd.int()  # extsr switch (Extras obsdatai.f90:89)
            srf = rd.a(26)
        return cls(
            names[0],
            names[1],
            names[2],
            names[3],
            bool(extt),
            otimes,
            srf,
            [_parse_group(rd) for _ in range(n)],
        )

    @classmethod
    def read(cls, path: str | Path, *, extras: bool = False) -> Obsdata:
        return cls.parse(read_text(path), path, extras=extras)

    def format(self) -> str:
        out = [
            *_HEAD_BANNER_OBS,
            _kv(_check_a(self.sources_file, "sources_file"), "Output source file"),
            _kv(_check_a(self.receivers_file, "receivers_file"), "Output receiver file"),
            _kv(
                _check_a(self.source_derivs_file, "source_derivs_file"),
                "Output source derivative file",
            ),
            _kv(_check_a(self.picks_dir, "picks_dir"), "Directory containing input receiver files"),
            _kv(str(len(self.groups)), "Number of input source files"),
            _kv("1" if self.extract_times else "0", "Extract traveltimes (0=no, 1=yes)"),
            _kv(_check_a(self.otimes_file, "otimes_file"), "File containing extracted traveltimes"),
        ]
        if self.source_receiver_file is not None:
            out.append(_kv("1", "Extract source-receiver association"))
            out.append(
                _kv(
                    _check_a(self.source_receiver_file, "source_receiver_file"),
                    "File containing source-receiver info",
                )
            )
        for i, g in enumerate(self.groups):
            out.extend(_format_group(g, i, lowercase_first=True))
        return "\n".join(out) + "\n"

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.format())


# --------------------------------------------------------------------------------------
# Input files of moddata/obsdata
# --------------------------------------------------------------------------------------


def parse_input_sources(
    text: str, *, teleseismic: bool = False, source: str | Path | None = None
) -> list[Source]:
    """Parse a moddata input source file: ``lat lon depth [covlat covlon covdep]`` (+ phase)."""
    rd = RecordReader(text, source)
    n = rd.int()
    out = []
    for i in range(n):
        toks = rd.record_tokens(3)
        try:
            lat, lon, depth = (float(t) for t in toks[:3])
            unc = None
            if len(toks) >= 6:
                clat, clon, cdep = (float(t) for t in toks[3:6])
                unc = (cdep, clat, clon)
        except ValueError:
            raise rd.error(f"source {i + 1}: bad record {toks!r}") from None
        phase = rd.a(10) if teleseismic else None
        out.append(Source(depth, lat, lon, [], teleseismic, phase, unc))
    return out


def format_input_sources(sources: list[Source]) -> str:
    """Format moddata input sources (lat lon depth order)."""
    out = [fint(len(sources))]
    for s in sources:
        if s.uncertainty is not None:
            cdep, clat, clon = s.uncertainty
            out.append("".join(f"{v:15.5f}" for v in (s.lat, s.lon, s.depth_km, clat, clon, cdep)))
        else:
            out.append("".join(f"{v:15.5f}" for v in (s.lat, s.lon, s.depth_km)))
        if s.teleseismic:
            out.append(f"{s.phase or 'P':<10}")
    return "\n".join(out) + "\n"


def parse_input_receivers(text: str, *, source: str | Path | None = None) -> pd.DataFrame:
    """Parse a receiver input file (``arraygen.out``): ``lat lon depth`` per receiver."""
    rd = RecordReader(text, source)
    n = rd.int()
    rows = [rd.floats(3) for _ in range(n)]
    return pd.DataFrame(
        rows, columns=["lat", "lon", "depth_km"], index=pd.RangeIndex(1, n + 1, name="receiver")
    )


def format_input_receivers(df: pd.DataFrame) -> str:
    """Format receivers as ``arraygen.out``."""
    out = [fint(len(df))]
    out.extend(
        fline(float(r.lat), float(r.lon), float(r.depth_km)) for r in df.itertuples(index=False)
    )
    return "\n".join(out) + "\n"


@dataclass
class PickFile:
    """An obsdata pick file: for teleseisms the source (``Source`` with phase) and the picks."""

    picks: pd.DataFrame  # lat, lon, depth_km [, time, uncertainty]
    source: Source | None = None


def parse_picks(
    text: str,
    *,
    teleseismic: bool = False,
    with_times: bool = True,
    source: str | Path | None = None,
) -> PickFile:
    """Parse a pick file (``obsdata.f90:213-224``)."""
    rd = RecordReader(text, source)
    n = rd.int()
    src = None
    if teleseismic:
        lat, lon, depth = rd.floats(3)
        src = Source(depth, lat, lon, [], True, rd.word())
    ncol = 5 if with_times else 3
    rows = [rd.floats(ncol) for _ in range(n)]
    cols = ["lat", "lon", "depth_km", "time", "uncertainty"][:ncol]
    return PickFile(pd.DataFrame(rows, columns=cols), src)


def format_picks(pf: PickFile) -> str:
    """Format a pick file."""
    df = pf.picks
    out = [fint(len(df))]
    if pf.source is not None:
        s = pf.source
        out.append(fline(s.lat, s.lon, s.depth_km))
        out.append(s.phase or "P")
    for row in df.itertuples(index=False):
        out.append(fline(*[float(v) for v in row]))
    return "\n".join(out) + "\n"


@dataclass
class ObsSource:
    """One source of an obsdata source input file: location (local only) and pick files.

    ``picks`` are ``(path_type, rank, filename)`` triples.
    """

    source: Source | None
    picks: list[tuple[int, int, str]]


def parse_obs_sources(
    text: str, *, teleseismic: bool = False, source: str | Path | None = None
) -> list[ObsSource]:
    """Parse an obsdata source input file (``obsdata.f90:104-130,185-203``)."""
    rd = RecordReader(text, source)
    n = rd.int()
    out = []
    for i in range(n):
        loc = None
        if not teleseismic:
            toks = rd.record_tokens(3)
            lat, lon, depth = (float(t) for t in toks[:3])
            unc = (float(toks[5]), float(toks[3]), float(toks[4])) if len(toks) >= 6 else None
            loc = Source(depth, lat, lon, [], False, None, unc)
            npath = rd.int()
        else:
            npath = 1
        picks = []
        for _ in range(npath):
            rec = rd.record_tokens(3)
            if len(rec) < 3:
                raise rd.error(f"source {i + 1}: expected 'path_type rank filename'")
            picks.append((int(rec[0]), int(rec[1]), rec[2]))
        out.append(ObsSource(loc, picks))
    return out


def format_obs_sources(items: list[ObsSource], *, teleseismic: bool = False) -> str:
    """Format an obsdata source input file."""
    out = [fint(len(items))]
    for it in items:
        if not teleseismic:
            if it.source is None:
                raise FmtomoFormatError("local obsdata sources need a location")
            s = it.source
            if s.uncertainty is not None:
                cdep, clat, clon = s.uncertainty
                out.append(
                    "".join(f"{v:15.5f}" for v in (s.lat, s.lon, s.depth_km, clat, clon, cdep))
                )
            else:
                out.append("".join(f"{v:15.5f}" for v in (s.lat, s.lon, s.depth_km)))
            out.append(str(len(it.picks)))
        out.extend(f"{a} {b}   {name}" for a, b, name in it.picks)
    return "\n".join(out) + "\n"
