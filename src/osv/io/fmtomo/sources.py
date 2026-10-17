"""``sources.in`` (and ``sourcesref.in``/``sourcestrue.in``) and ``receivers.in``.

``sources.in`` (read by ``fm3d`` at ``3dfm_main.f90:164-290``; written by ``moddata.f90:105-175``,
``obsdata.f90:236-265``, ``invert3d.f90:1036-1049`` and ``scorrect.f90:138-150``)::

    n_sources
    per source:
        is_teleseismic                      0 = local (inside the model), 1 = teleseismic
        [phase]                             only if teleseismic, e.g. P, PcP (character*8)
        depth lat lon                       km (positive down, negative above the sphere), deg, deg
        n_paths
        per path:
            n_steps                         number of interface-pair steps
            i1 j1 i2 j2 ...                 2*n_steps integers (the path signature)
            v1 v2 ...                       n_steps velocity types (1 or 2)

``fm3d`` reads exactly three values from the position record (``3dfm_main.f90:207,220``);
``moddata``/``obsdata`` append the a priori uncertainties ``depth lat lon`` (format ``6f12.5``,
``moddata.f90:151``) that ``invert3d`` reads for sources it relocates
(``invert3d.f90:485-487``); the Extras ``obsdatai`` appends an integer source id
(``obsdatai.f90`` ``WRITE(20,*)depth,lat,long,srcid``).  All variants are accepted and
reproduced (:attr:`Source.uncertainty`, :attr:`Source.extra_id`).

**Path signature.**  Interfaces are numbered from 1 at the top.  ``0 2  2 1`` for a source in
layer 1 means: wavefront from the source (0) to interface 2, then from interface 2 back to
interface 1 (the surface) - a reflection off interface 2; ``0 1`` is the direct/refracted wave
to the surface.  Equal numbers (``1 1``) request a reflection-fit (PP-like) step; fm3d refuses
these in ``no_pp_mode`` (``3dfm_main.f90:279-287``).  The velocity type of each step is 1
(first velocity set, usually P) or 2 (second, usually S); consecutive different values give
a conversion.  A P direct path is ``PathSpec.direct(1)``, an S direct path ``PathSpec.direct(2)``.

``receivers.in`` (``3dfm_main.f90:414-470``; written by ``moddata.f90:207-218``)::

    n_receivers
    per receiver:
        depth lat lon                       km (negative = above the sphere), deg, deg
        n_rays
        source ids (n_rays integers)        position of the source in sources.in
        path ids  (n_rays integers)         position in that source's path list
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ._fortran import FmtomoFormatError, RecordReader, fint, fints, fline, read_text, write_text

__all__ = [
    "PathSpec",
    "Receivers",
    "Source",
    "format_receivers",
    "format_sources",
    "parse_receivers",
    "parse_sources",
    "read_receivers",
    "read_sources",
    "sources_frame",
    "validate_receivers",
    "validate_sources",
    "write_receivers",
    "write_sources",
]

MAX_PHASE_CHARS = 8  # character(len=8) :: teleseismic_phase (mod_3dfm.f90:145)


@dataclass(frozen=True)
class PathSpec:
    """One path (phase) of a source: interface-pair steps and a velocity type per step."""

    steps: tuple[tuple[int, int], ...]
    vtypes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise FmtomoFormatError("a path needs at least one step")
        if len(self.steps) != len(self.vtypes):
            raise FmtomoFormatError(
                f"path has {len(self.steps)} steps but {len(self.vtypes)} velocity types"
            )
        if any(v not in (1, 2) for v in self.vtypes):
            raise FmtomoFormatError(f"velocity types must be 1 or 2, got {self.vtypes}")

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def sequence(self) -> tuple[int, ...]:
        """The flat integer sequence as written in the file."""
        return tuple(i for pair in self.steps for i in pair)

    @classmethod
    def from_sequence(
        cls, sequence: list[int] | tuple[int, ...], vtypes: list[int] | tuple[int, ...]
    ) -> PathSpec:
        if len(sequence) % 2:
            raise FmtomoFormatError(
                f"path sequence needs an even number of integers, got {len(sequence)}"
            )
        steps = tuple((int(sequence[i]), int(sequence[i + 1])) for i in range(0, len(sequence), 2))
        return cls(steps, tuple(int(v) for v in vtypes))

    @classmethod
    def direct(cls, vtype: int = 1, top_interface: int = 1) -> PathSpec:
        """Direct (and refracted) wave from a source in the top layer to the surface: ``0 1``."""
        return cls(((0, top_interface),), (vtype,))

    @classmethod
    def reflected(cls, interface: int, vtype: int = 1, top_interface: int = 1) -> PathSpec:
        """Reflection off ``interface`` for a source in the top layer: ``0 i  i 1``."""
        return cls(((0, interface), (interface, top_interface)), (vtype, vtype))

    @property
    def is_reflection_fit(self) -> bool:
        """True if any step has equal numbers (reflection matching, PP-like)."""
        return any(a == b for a, b in self.steps)


@dataclass
class Source:
    """One source of ``sources.in``.

    ``depth_km`` is positive down and relative to the 6371 km sphere (so it is the
    ``depth_km_bsl`` of the catalogue); ``lat``/``lon`` are degrees.  ``uncertainty`` is the
    optional a priori ``(depth_km, lat_deg, lon_deg)`` standard deviation triple of
    ``sourcesref.in``; ``extra_id`` the integer source id of Extras files.
    """

    depth_km: float
    lat: float
    lon: float
    paths: list[PathSpec] = field(default_factory=list)
    teleseismic: bool = False
    phase: str | None = None
    uncertainty: tuple[float, float, float] | None = None
    extra_id: int | None = None


def parse_sources(text: str, *, source: str | Path | None = None) -> list[Source]:
    """Parse the text of a ``sources.in``-format file into a list of :class:`Source`."""
    rd = RecordReader(text, source)
    n = rd.int()
    if n < 0:
        raise rd.error(f"negative number of sources ({n})")
    out: list[Source] = []
    for i in range(1, n + 1):
        flag = rd.int()
        if flag not in (0, 1):
            raise rd.error(f"source {i}: teleseismic flag must be 0 or 1, got {flag}")
        phase = rd.word() if flag == 1 else None
        toks = rd.record_tokens(3)
        try:
            depth, lat, lon = (float(t.replace("d", "e").replace("D", "e")) for t in toks[:3])
            unc = None
            extra = None
            if len(toks) >= 6:
                unc = tuple(float(t.replace("d", "e")) for t in toks[3:6])
            elif len(toks) == 4:
                extra = int(toks[3])
        except ValueError:
            raise rd.error(f"source {i}: cannot read position record {toks!r}") from None
        n_paths = rd.int()
        if n_paths < 0:
            raise rd.error(f"source {i}: negative number of paths")
        paths = []
        for p in range(1, n_paths + 1):
            n_tf = rd.int()
            if n_tf < 1:
                raise rd.error(f"source {i} path {p}: needs at least one step, got {n_tf}")
            seq = rd.ints(2 * n_tf)
            vt = rd.ints(n_tf)
            try:
                paths.append(PathSpec.from_sequence(seq, vt))
            except FmtomoFormatError as exc:
                raise rd.error(f"source {i} path {p}: {exc}") from None
        out.append(Source(depth, lat, lon, paths, flag == 1, phase, unc, extra))  # type: ignore[arg-type]
    rd.expect_eof(f"source {n}")
    return out


def read_sources(path: str | Path) -> list[Source]:
    """Read a ``sources.in``-format file."""
    return parse_sources(read_text(path), source=path)


def format_sources(sources: list[Source]) -> str:
    """Format as ``sources.in`` (6f12.5 position record if ``uncertainty`` is given)."""
    out = [fint(len(sources))]
    for i, s in enumerate(sources, start=1):
        out.append(fint(1 if s.teleseismic else 0))
        if s.teleseismic:
            if not s.phase:
                raise FmtomoFormatError(f"source {i} is teleseismic but has no phase")
            if len(s.phase) > MAX_PHASE_CHARS or any(c.isspace() or c in ",/'\"" for c in s.phase):
                raise FmtomoFormatError(
                    f"source {i}: phase {s.phase!r} must be at most {MAX_PHASE_CHARS} characters "
                    "without blanks, commas or slashes (mod_3dfm.f90:145)"
                )
            out.append(f"{s.phase:<10}")
        if s.uncertainty is not None:
            u = s.uncertainty
            out.append("".join(f"{v:12.5f}" for v in (s.depth_km, s.lat, s.lon, *u)))
        elif s.extra_id is not None:
            out.append(fline(s.depth_km, s.lat, s.lon, int(s.extra_id)))
        else:
            out.append(fline(s.depth_km, s.lat, s.lon))
        out.append(fint(len(s.paths)))
        for p in s.paths:
            out.append(fint(p.n_steps))
            out.extend(fints(p.sequence))
            out.extend(fints(p.vtypes))
    return "\n".join(out) + "\n"


def write_sources(path: str | Path, sources: list[Source]) -> Path:
    """Write a ``sources.in``-format file."""
    return write_text(path, format_sources(sources))


def sources_frame(sources: list[Source]) -> pd.DataFrame:
    """Sources as a DataFrame indexed by FMTOMO source id (1-based)."""
    df = pd.DataFrame(
        {
            "teleseismic": [s.teleseismic for s in sources],
            "phase": [s.phase for s in sources],
            "depth_km": [s.depth_km for s in sources],
            "lat": [s.lat for s in sources],
            "lon": [s.lon for s in sources],
            "n_paths": [len(s.paths) for s in sources],
        },
        index=pd.RangeIndex(1, len(sources) + 1, name="source"),
    )
    return df


def validate_sources(
    sources: list[Source], *, n_interfaces: int | None = None, n_vtypes: int | None = None
) -> None:
    """Check path signatures against the model (interface numbers, velocity types).

    Reproduces the checks of ``fm3d`` (``3dfm_main.f90:274-279``: type 2 needs two velocity
    types) and requires interface numbers in ``0..n_interfaces``.
    """
    problems: list[str] = []
    for i, s in enumerate(sources, start=1):
        if s.teleseismic and not s.phase:
            problems.append(f"source {i}: teleseismic without phase")
        for p, path in enumerate(s.paths, start=1):
            for a, b in path.steps:
                for v in (a, b):
                    if v < 0 or (n_interfaces is not None and v > n_interfaces):
                        problems.append(
                            f"source {i} path {p}: interface number {v} outside 0..{n_interfaces}"
                        )
            if n_vtypes == 1 and 2 in path.vtypes:
                problems.append(
                    f"source {i} path {p}: velocity type 2 requested but vgrids has one type"
                )
    if problems:
        raise FmtomoFormatError("invalid sources:\n  - " + "\n  - ".join(problems))


@dataclass
class Receivers:
    """Contents of ``receivers.in``.

    ``positions``: DataFrame indexed by receiver id (1-based) with ``depth_km`` (negative above
    the sphere), ``lat``, ``lon`` (degrees).  ``rays``: DataFrame in file order with columns
    ``receiver``, ``source``, ``path`` - one row per (source, path) requested at the receiver.
    """

    positions: pd.DataFrame
    rays: pd.DataFrame


def parse_receivers(text: str, *, source: str | Path | None = None) -> Receivers:
    """Parse the text of a ``receivers.in``-format file."""
    rd = RecordReader(text, source)
    n = rd.int()
    if n < 0:
        raise rd.error(f"negative number of receivers ({n})")
    pos: list[tuple[float, float, float]] = []
    r_ids: list[int] = []
    s_ids: list[int] = []
    p_ids: list[int] = []
    for i in range(1, n + 1):
        pos.append(tuple(rd.floats(3)))  # type: ignore[arg-type]
        nr = rd.int()
        if nr < 0:
            raise rd.error(f"receiver {i}: negative number of rays")
        src = rd.ints(nr)
        pth = rd.ints(nr)
        r_ids.extend([i] * nr)
        s_ids.extend(src)
        p_ids.extend(pth)
    rd.expect_eof(f"receiver {n}")
    positions = pd.DataFrame(
        pos, columns=["depth_km", "lat", "lon"], index=pd.RangeIndex(1, n + 1, name="receiver")
    )
    rays = pd.DataFrame({"receiver": r_ids, "source": s_ids, "path": p_ids}, dtype="int64")
    return Receivers(positions, rays)


def read_receivers(path: str | Path) -> Receivers:
    """Read a ``receivers.in``-format file."""
    return parse_receivers(read_text(path), source=path)


def format_receivers(rec: Receivers) -> str:
    """Format as ``receivers.in``; ray lists wrap at six integers per line like gfortran."""
    pos = rec.positions
    n = len(pos)
    expected = list(range(1, n + 1))
    if list(pos.index) != expected:
        raise FmtomoFormatError("receiver positions must be indexed 1..n in file order")
    rays = rec.rays
    bad = sorted(set(rays["receiver"]) - set(expected))
    if bad:
        raise FmtomoFormatError(f"rays refer to receivers {bad[:5]} that have no position")
    out = [fint(n)]
    groups = {k: g for k, g in rays.groupby("receiver", sort=False)}
    for i in expected:
        row = pos.loc[i]
        out.append(fline(float(row["depth_km"]), float(row["lat"]), float(row["lon"])))
        g = groups.get(i)
        k = 0 if g is None else len(g)
        out.append(fint(k))
        if k:
            out.extend(fints(list(g["source"])))
            out.extend(fints(list(g["path"])))
        else:
            out.extend(["", ""])  # a read with a zero-length list still consumes a record
    return "\n".join(out) + "\n"


def write_receivers(path: str | Path, rec: Receivers) -> Path:
    """Write a ``receivers.in``-format file."""
    return write_text(path, format_receivers(rec))


def validate_receivers(rec: Receivers, sources: list[Source]) -> None:
    """Reproduce the consistency checks of ``fm3d`` (``3dfm_main.f90:448-473``)."""
    problems: list[str] = []
    n_src = len(sources)
    for r in rec.rays.itertuples(index=False):
        if not 1 <= r.source <= n_src:
            problems.append(
                f"receiver {r.receiver} requests paths from non-existent source {r.source}"
            )
            continue
        n_paths = len(sources[r.source - 1].paths)
        if not 1 <= r.path <= n_paths:
            problems.append(
                f"receiver {r.receiver} requests path {r.path} of source {r.source}, which has "
                f"{n_paths} path(s)"
            )
    if problems:
        raise FmtomoFormatError(
            "inconsistent receivers/sources:\n  - " + "\n  - ".join(problems[:20])
        )
