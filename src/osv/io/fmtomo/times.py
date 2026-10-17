"""Travel-time files and small vector files.

Observed times ``otimes.dat`` (read ``invert3d.f90:763``, ``residuals.f90:125``; written by
``obsdata.f90:231``, ``synthdata.f90:122``)::

    n_obs
    receiver source ray subray time uncertainty   (list-directed: usually 2 lines per observation)

Model/reference times ``arrivals.dat`` = ``mtimes.dat`` = ``rtimes.dat`` (written by ``fm3d``,
``3dfm_main.f90:1113,1125,1173``, format ``(4i6,f15.6,2l5)``; no count line)::

    receiver source ray subray time diffracted headwave

``receiver``/``source`` are the 1-based positions in ``receivers.in``/``sources.in``.  **``ray``
is the 1-based position of the ray in that receiver's own ray list** (loop index ``m`` of
``3dfm_main.f90:1095-1113``, the same in ``rays.dat`` and ``frechet.dat``), *not* the path id
(``README_0.7`` says "position in the list of paths from each source", which is only true when a
receiver requests each source once, as in example 1); :func:`attach_path_ids` maps it to the path
id with the receivers table.  ``obsdata`` writes the path id into this column of ``otimes.dat``
(``obsdata.f90:231``), but ``invert3d``/``residuals`` ignore the first four columns
(``invert3d.f90:763``) and rely on the row order matching ``arrivals.dat``.  ``subray`` is 0 for a
normal path and the sub-ray number for reflection-fit (multi-ray) paths (``k``,
``3dfm_main.f90:1112``); the observed-file fourth integer is ``nval`` (``obsdata.f90:231``, 0).
**Validity**: ``fm3d`` writes time ``-1`` for an invalid ray (``3dfm_main.f90:1121``);
``synthdata``/``obsdata`` mark invalid observed picks with ``-100`` (``synthdata.f90:102``) and the
programs treat observed times below ``-50`` as invalid (``invert3d.f90:770``,
``residuals.f90:127``).  The ``valid`` column applies those rules (observed: ``time >= -50``;
model: ``time > 0``).  Teleseismic *residual* times may legitimately be negative.

Other files: ``stimes.dat`` (``n`` then ``n`` origin-time corrections, ``invert3d.f90:1051-1056``),
``sourcederivs.in`` (``n`` then ``n`` source ids, ``moddata.f90:181-182``), ``gridsave.in``
(``3dfm_main.f90:708-730``) and ``residuals.dat`` (one ``(f10.2,2X,f10.5,2X,f10.5)`` line
per iteration written by ``residuals.f90:155``: RMS in **milliseconds**, variance in s^2,
chi-square per datum).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._fortran import (
    FmtomoFormatError,
    RecordReader,
    fint,
    fints,
    fline,
    fnum,
    read_text,
    write_text,
)
from .sources import Receivers

__all__ = [
    "INVALID_MODEL_TIME",
    "INVALID_OBSERVED_BELOW",
    "ARRIVAL_COLUMNS",
    "OBSERVED_COLUMNS",
    "GridSave",
    "attach_path_ids",
    "format_arrivals",
    "format_gridsave",
    "format_observed",
    "format_residuals",
    "format_source_ids",
    "format_stimes",
    "parse_arrivals",
    "parse_gridsave",
    "parse_observed",
    "parse_residuals",
    "parse_source_ids",
    "parse_stimes",
    "read_arrivals",
    "read_gridsave",
    "read_observed",
    "read_residuals",
    "read_source_ids",
    "read_stimes",
    "write_arrivals",
    "write_gridsave",
    "write_observed",
    "write_residuals",
    "write_source_ids",
    "write_stimes",
]

INVALID_MODEL_TIME = -1.0
INVALID_OBSERVED_BELOW = -50.0
OBSERVED_COLUMNS = ["receiver", "source", "ray", "subray", "time", "uncertainty"]
ARRIVAL_COLUMNS = ["receiver", "source", "ray", "subray", "time", "diffracted", "headwave"]
_KEY = ["receiver", "source", "ray", "subray"]


def _num(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return float(tok.replace("d", "e").replace("D", "e"))


def _tf(v: object) -> str:
    return "T" if v else "F"


def _logical(tok: str, rd: RecordReader) -> bool:
    t = tok.lstrip(".").upper()[:1]
    if t == "T":
        return True
    if t == "F":
        return False
    raise rd.error(f"expected a logical (T/F), got {tok!r}")


def parse_observed(text: str, *, source: str | Path | None = None) -> pd.DataFrame:
    """Parse ``otimes.dat``; columns :data:`OBSERVED_COLUMNS` plus boolean ``valid``."""
    rd = RecordReader(text, source)
    n = rd.int()
    if n < 0:
        raise rd.error(f"negative observation count {n}")
    if rd.remaining() < n:
        raise rd.error(
            f"header declares {n} observations but only {rd.remaining()} record(s) remain"
        )
    rows = np.empty((n, 6))
    for i in range(n):
        try:
            rows[i] = [_num(t) for t in rd.tokens(6)]
        except ValueError:
            raise rd.error(f"observation {i + 1}: non-numeric value") from None
    rd.expect_eof("the last observation")
    df = pd.DataFrame(rows, columns=OBSERVED_COLUMNS)
    for c in _KEY:
        df[c] = df[c].astype("int64")
    df["valid"] = df["time"] >= INVALID_OBSERVED_BELOW
    return df


def read_observed(path: str | Path) -> pd.DataFrame:
    """Read ``otimes.dat``."""
    return parse_observed(read_text(path), source=path)


def format_observed(df: pd.DataFrame) -> str:
    """Format as ``otimes.dat`` (one observation per two lines like the Fortran output)."""
    missing = [c for c in OBSERVED_COLUMNS if c not in df.columns]
    if missing:
        raise FmtomoFormatError(f"observed-times table lacks columns {missing}")
    out = [fint(len(df))]
    for r in df[OBSERVED_COLUMNS].itertuples(index=False):
        out.append(fline(int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])))
        out.append(fline(float(r[5])))
    return "\n".join(out) + "\n"


def write_observed(path: str | Path, df: pd.DataFrame) -> Path:
    """Write ``otimes.dat``."""
    return write_text(path, format_observed(df))


def parse_arrivals(text: str, *, source: str | Path | None = None) -> pd.DataFrame:
    """Parse ``arrivals.dat``/``mtimes.dat``/``rtimes.dat``; adds boolean ``valid`` (time > 0)."""
    rd = RecordReader(text, source)
    rows: list[tuple] = []
    while not rd.at_eof():
        toks = rd.tokens(7)
        try:
            rows.append(
                (
                    int(toks[0]),
                    int(toks[1]),
                    int(toks[2]),
                    int(toks[3]),
                    _num(toks[4]),
                    _logical(toks[5], rd),
                    _logical(toks[6], rd),
                )
            )
        except FmtomoFormatError:
            raise
        except ValueError:
            raise rd.error(f"cannot parse arrival record {toks!r}") from None
    df = pd.DataFrame(rows, columns=ARRIVAL_COLUMNS)
    for c in _KEY:
        df[c] = df[c].astype("int64")
    df["valid"] = df["time"] > 0.0
    return df


def read_arrivals(path: str | Path) -> pd.DataFrame:
    """Read ``arrivals.dat`` / ``mtimes.dat`` / ``rtimes.dat``."""
    return parse_arrivals(read_text(path), source=path)


def format_arrivals(df: pd.DataFrame) -> str:
    """Format as ``arrivals.dat`` (``(4i6,f15.6,2l5)``)."""
    missing = [c for c in ARRIVAL_COLUMNS if c not in df.columns]
    if missing:
        raise FmtomoFormatError(f"arrival table lacks columns {missing}")
    lines = [
        f"{int(a):6d}{int(b):6d}{int(c):6d}{int(d):6d}{float(t):15.6f}{_tf(x):>5}{_tf(y):>5}"
        for a, b, c, d, t, x, y in df[ARRIVAL_COLUMNS].itertuples(index=False)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_arrivals(path: str | Path, df: pd.DataFrame) -> Path:
    """Write ``arrivals.dat`` / ``mtimes.dat`` / ``rtimes.dat``."""
    return write_text(path, format_arrivals(df))


def attach_path_ids(table: pd.DataFrame, receivers: Receivers) -> pd.DataFrame:
    """Add column ``path`` (the source's path id) to a table keyed by ``receiver``/``ray``.

    ``ray`` is the position in the receiver's ray list; ``receivers.rays`` holds that list in
    file order.  The ``source`` column of ``table`` is checked against it.
    """
    rays = receivers.rays.copy()
    rays["ray"] = rays.groupby("receiver").cumcount() + 1
    key = rays.set_index(["receiver", "ray"])
    idx = pd.MultiIndex.from_arrays([table["receiver"], table["ray"]])
    try:
        found = key.loc[idx]
    except KeyError as exc:
        raise FmtomoFormatError(
            f"table refers to a receiver ray not in receivers.in: {exc}"
        ) from None
    if not (found["source"].to_numpy() == table["source"].to_numpy()).all():
        raise FmtomoFormatError("source ids of the table do not match the receivers' ray lists")
    out = table.copy()
    out["path"] = found["path"].to_numpy()
    return out


def parse_stimes(text: str, *, source: str | Path | None = None) -> np.ndarray:
    """Parse ``stimes.dat``: per-source origin-time perturbations (s)."""
    rd = RecordReader(text, source)
    n = rd.int()
    out = np.array(rd.floats(n)) if n > 0 else np.empty(0)
    rd.expect_eof("stimes")
    return out


def read_stimes(path: str | Path) -> np.ndarray:
    """Read ``stimes.dat``."""
    return parse_stimes(read_text(path), source=path)


def format_stimes(values: np.ndarray | list[float]) -> str:
    """Format as ``stimes.dat`` (one value per line)."""
    v = [float(x) for x in values]
    return "\n".join([fint(len(v)), *[fnum(x) for x in v]]) + "\n"


def write_stimes(path: str | Path, values: np.ndarray | list[float]) -> Path:
    """Write ``stimes.dat``."""
    return write_text(path, format_stimes(values))


def parse_source_ids(text: str, *, source: str | Path | None = None) -> np.ndarray:
    """Parse ``sourcederivs.in``: ids of the sources whose location is inverted."""
    rd = RecordReader(text, source)
    n = rd.int()
    out = np.array(rd.ints(n), dtype=int) if n > 0 else np.empty(0, dtype=int)
    rd.expect_eof("sourcederivs")
    return out


def read_source_ids(path: str | Path) -> np.ndarray:
    """Read ``sourcederivs.in``."""
    return parse_source_ids(read_text(path), source=path)


def format_source_ids(ids: np.ndarray | list[int]) -> str:
    """Format as ``sourcederivs.in``."""
    v = [int(x) for x in ids]
    return "\n".join([fint(len(v)), *fints(v)]) + "\n"


def write_source_ids(path: str | Path, ids: np.ndarray | list[int]) -> Path:
    """Write ``sourcederivs.in``."""
    return write_text(path, format_source_ids(ids))


@dataclass(frozen=True)
class GridSave:
    """One ``gridsave.in`` entry: arrival-time grids to save for ``source``'s ``paths``."""

    source: int
    paths: tuple[int, ...]
    first_timefield: tuple[int, ...]


def parse_gridsave(text: str, *, source: str | Path | None = None) -> list[GridSave]:
    """Parse ``gridsave.in`` (``3dfm_main.f90:718-729``): repeated ``source n``/paths/first-tf."""
    rd = RecordReader(text, source)
    out: list[GridSave] = []
    while not rd.at_eof():
        n, nsave = rd.ints(2)
        if nsave > 0:
            paths = tuple(rd.ints(nsave))
            first = tuple(rd.ints(nsave))
            out.append(GridSave(n, paths, first))
    return out


def read_gridsave(path: str | Path) -> list[GridSave]:
    """Read ``gridsave.in``."""
    return parse_gridsave(read_text(path), source=path)


def format_gridsave(entries: list[GridSave]) -> str:
    """Format as ``gridsave.in``."""
    out: list[str] = []
    for e in entries:
        if len(e.paths) != len(e.first_timefield):
            raise FmtomoFormatError("gridsave entry needs one first-timefield per path")
        out.append(fint(e.source) + fint(len(e.paths)))
        out.extend(fints(e.paths))
        out.extend(fints(e.first_timefield))
    return "\n".join(out) + "\n"


def write_gridsave(path: str | Path, entries: list[GridSave]) -> Path:
    """Write ``gridsave.in``."""
    return write_text(path, format_gridsave(entries))


def parse_residuals(text: str, *, source: str | Path | None = None) -> pd.DataFrame:
    """Parse ``residuals.dat``: ``rms_ms``, ``variance_s2``, ``chi2``; index is the iteration."""
    rows = []
    for k, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        toks = line.split()
        if len(toks) != 3:
            raise FmtomoFormatError(
                f"{source or '<text>'}:{k}: expected 3 values (rms, variance, chi2), got {line!r}"
            )
        try:
            rows.append([_num(t) for t in toks])
        except ValueError:
            raise FmtomoFormatError(
                f"{source or '<text>'}:{k}: non-numeric residual line {line!r}"
            ) from None
    return pd.DataFrame(
        rows,
        columns=["rms_ms", "variance_s2", "chi2"],
        index=pd.RangeIndex(len(rows), name="iteration"),
    )


def read_residuals(path: str | Path) -> pd.DataFrame:
    """Read ``residuals.dat``."""
    return parse_residuals(read_text(path), source=path)


def format_residuals(df: pd.DataFrame) -> str:
    """Format as ``residuals.dat`` the way ``tomo3d`` leaves it.

    ``residuals`` prints ``(f10.2,2X,f10.5,2X,f10.5)`` but the ksh script passes the line through
    ``read ttres; print -u5 $ttres`` (``bin/tomo3d:116``), which strips and collapses the blanks.
    """
    lines = [
        f"{float(a):.2f} {float(b):.5f} {float(c):.5f}"
        for a, b, c in df[["rms_ms", "variance_s2", "chi2"]].itertuples(index=False)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_residuals(path: str | Path, df: pd.DataFrame) -> Path:
    """Write ``residuals.dat``."""
    return write_text(path, format_residuals(df))
