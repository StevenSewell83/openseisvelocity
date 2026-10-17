"""FMTOMO Extras (``fortran/fmtomo_extras``): grid relocation and Vp/Vs files.

Control files (all positional; widths are those of the Fortran ``(a<N>)`` reads):

* :class:`GridLoc` ``gridloc.in`` - ``gridloc.f90:74-83`` (``(a20)``; the limits file is
  list-directed).
* :class:`GridLocJ` ``gridlocj.in`` - ``gridlocj.f90:101-120`` (``(a30)``); joint P/S relocation.
* :class:`JoinTimes` ``jointimes.in`` - ``jointimes.f90:23-28`` (``(a30)``).
* :class:`MkVpVs` ``mkvpvs.in`` - ``mkvpvs.f90:14-19`` (list-directed names, no ``/``).
* :class:`VpVsInv` ``vpvsinv.in`` - ``vpvsinv.f90:100-115`` (``(a25)``).
* :class:`FrVpVs` ``frvpvs.in`` - ``frvpvs.f90:111-126`` (list-directed).
* :class:`VpVsResids` ``vpvsresids.in`` - ``vpvsresids.f90:13-15`` (list-directed).

Data files: ``src.dat`` (source-receiver associations, :func:`parse_src_dat`, written by
``obsdatai``
/ ``jointimes`` and read by ``gridloc``/``gridlocj``: ``n``; per block ``block n_picks``;
then ``n_picks`` records ``id time uncertainty``, ``jointimes.f90:104-110``,
``gridlocj.f90:280-286``); ``sandr.dat`` (counts, :func:`read_sandr`); ``otimess.dat`` /
``otimesp.dat`` are ordinary ``otimes.dat`` files and ``vgridsp.in``/``vgridss.in``/
``vgridvpvs.in`` ordinary ``vgrids.in`` files (the Vp/Vs grid has no uncertainty column:
``mkvpvs.f90:48``) - use :mod:`osv.io.fmtomo.times` and :mod:`osv.io.fmtomo.grids`.
``otimes.dat`` of the Extras has the same 6 columns (``gridloc.f90:444``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ._control import ControlFile, F, L
from ._fortran import RecordReader, fint, fline, read_text, write_text

__all__ = [
    "FrVpVs",
    "GridLoc",
    "GridLocJ",
    "JoinTimes",
    "MkVpVs",
    "VpVsInv",
    "SERROR_COLUMNS",
    "VpVsResids",
    "format_sandr",
    "format_src_dat",
    "parse_sandr",
    "parse_serror",
    "parse_src_dat",
    "read_sandr",
    "read_serror",
    "read_src_dat",
    "write_sandr",
    "write_src_dat",
]


@dataclass
class GridLoc(ControlFile):
    """``gridloc.in``: grid-search relocation of events from model travel-time fields."""

    src_file: str = "src.dat"
    arrtimes_file: str = "arrtimes.dat"
    min_stations: int = 5
    max_origin_time_perturbation: float = 5.0
    limits_file: str = "interfacesref.in"
    otimes_file: str = "otimes.dat"
    receivers_file: str = "receivers.in"
    refined_search: int = 1
    subcell_dicing: int = 10

    _pad = 23
    _layout = (
        F("src_file", "a", "File containing source-receiver associations", 20),
        F("arrtimes_file", "a", "File containing all model traveltime fields", 20),
        F("min_stations", "i", "Ignore relocate if less than this number of stations"),
        F("max_origin_time_perturbation", "f", "Maximum origin time perturbation"),
        F("limits_file", "w", "Define limits of relocation"),
        F("otimes_file", "a", "Observed traveltime file", 20),
        F("receivers_file", "a", "Receiver file (in reality the sources)", 20),
        F("refined_search", "i", "Apply refined sub-cell search (0=no, 1=yes)"),
        F(
            "subcell_dicing",
            "i",
            "Sub-cell dicing for refined search (~5-10 should do; must be >1)",
        ),
    )


@dataclass
class GridLocJ(ControlFile):
    """``gridlocj.in``: joint P and S grid-search relocation."""

    src_file: str = "src.dat"
    arrtimes_p_file: str = "invert_p/arrtimes.dat"
    arrtimes_s_file: str = "invert_s/arrtimes.dat"
    sources_p_file: str = "invert_p/sources.in"
    sources_s_file: str = "invert_s/sources.in"
    min_combinations: int = 5
    max_origin_time_perturbation: float = 5.0
    limits_file: str = "invert_p/interfacesref.in"
    otimes_p_file: str = "invert_p/otimes.dat"
    otimes_s_file: str = "invert_s/otimes.dat"
    receivers_p_file: str = "invert_p/receivers.in"
    receivers_s_file: str = "invert_s/receivers.in"
    refined_search: int = 1
    subcell_dicing: int = 10
    confidence_interval: int = 0
    confidence_file: str = "serror.dat"
    s_weight: float = 3.0
    sp_weight: float = 100.0
    relocation_based_on: int = 1

    _pad = 32
    _layout = (
        F("src_file", "a", "File containing source-receiver associations", 30),
        F("arrtimes_p_file", "a", "File containing all model traveltime fields for P", 30),
        F("arrtimes_s_file", "a", "File containing all model traveltime fields for S", 30),
        F("sources_p_file", "a", "File containing all source locations for P", 30),
        F("sources_s_file", "a", "File containing all source locations for S", 30),
        F(
            "min_combinations",
            "i",
            "Ignore relocate if less than this number of source-receiver combinations",
        ),
        F("max_origin_time_perturbation", "f", "Maximum origin time perturbation"),
        F("limits_file", "a", "Define limits of relocation (P = S)", 30),
        F("otimes_p_file", "a", "Observed traveltime file for P", 30),
        F("otimes_s_file", "a", "Observed traveltime file for S", 30),
        F("receivers_p_file", "a", "Receiver file (in reality the sources) for P", 30),
        F("receivers_s_file", "a", "Receiver file (in reality the sources) for S", 30),
        F("refined_search", "i", "Apply refined sub-cell search (0=no, 1=yes)"),
        F(
            "subcell_dicing",
            "i",
            "Sub-cell dicing for refined search (~5-10 should do; must be >1)",
        ),
        F("confidence_interval", "i", "Determine 95% confidence interval (0=no,     1=yes)"),
        F("confidence_file", "a", "Output of confidence interval", 30),
        F("s_weight", "f", "Weighting factor for S (higher means S-times less important)"),
        F("sp_weight", "f", "Weighting factor for S-P residuals"),
        F("relocation_based_on", "i", "Relocation based on:"),
        L(" " * 34 + "1 = P-times only"),
        L(" " * 34 + "2 = P-times and unique S-times"),
        L(" " * 34 + "3 = P-times and S-P times and unique S-times"),
        L(" " * 34 + "4 = P-times and S-times"),
    )


@dataclass
class JoinTimes(ControlFile):
    """``jointimes.in``: merge P and S source-receiver association files."""

    src_p_file: str = "invert_p/src.dat"
    src_s_file: str = "invert_s/src.dat"
    receivers_p_file: str = "invert_p/receiversref.in"
    receivers_s_file: str = "invert_s/receiversref.in"
    src_joint_file: str = "src.dat"

    _pad = 32
    _layout = (
        F("src_p_file", "a", "Source-receiver association file for P", 30),
        F("src_s_file", "a", "Source-receiver association file for S", 30),
        F("receivers_p_file", "a", "Receiver file for P", 30),
        F("receivers_s_file", "a", "Receiver file for S", 30),
        F("src_joint_file", "a", "Source-receiver association file for both P and S", 30),
    )


@dataclass
class MkVpVs(ControlFile):
    """``mkvpvs.in``: build a Vp/Vs grid from a Vp and a Vs velocity grid."""

    vp_file: str = "vgridsp.in"
    vs_file: str = "vgridss.in"
    vpvs_out_file: str = "vgridsref.in"
    vpvs_calc_file: str = "vgridvpvs.in"
    max_depth_km: float = 50.0

    _pad = 27
    _layout = (
        F("vp_file", "w", "Input Vp file"),
        F("vs_file", "w", "Input Vs file"),
        F("vpvs_out_file", "w", "Output Vp/Vs file"),
        F("vpvs_calc_file", "w", "Output calculated Vp/Vs"),
        F("max_depth_km", "f", "maximum depth of Vp/Vs"),
    )


_C70 = "c" * 70


@dataclass
class VpVsInv(ControlFile):
    """``vpvsinv.in``: Vp/Vs inversion parameters."""

    start_grid_file: str = "vgridsref.in"
    current_grid_file: str = "vgrids.in"
    otimes_file: str = "otimes.dat"
    mtimes_file: str = "mtimes.dat"
    frechet_file: str = "frechet.dat"
    sandr_file: str = "sandr.dat"
    damping: float = 1.0
    subspace_dimension: int = 20
    second_derivative_smoothing: int = 1
    smoothing: float = 0.5
    sparse_fraction: float = 0.1
    earth_radius_km: float = 6371.0

    _pad = 26
    _layout = (
        L(_C70),
        L("c INPUT PARAMETERS"),
        L(_C70),
        F("start_grid_file", "a", "File containing starting vpvs grid", 25),
        F("current_grid_file", "a", "File containing current vpvs grid", 25),
        F("otimes_file", "a", "File containing observed data", 25),
        F("mtimes_file", "a", "File containing model data", 25),
        F("frechet_file", "a", "File containing Frechet derivatives", 25),
        F("sandr_file", "a", "File containing numbers of sources & receivers", 25),
        F("damping", "f", "Damping factor (epsilon)"),
        F("subspace_dimension", "i", "Subspace dimension"),
        F("second_derivative_smoothing", "i", "Apply 2nd derivative smoothing (0=no, 1=yes)"),
        F("smoothing", "f", "Smoothing factor (eta) -"),
        F("sparse_fraction", "f", "Fraction of max. G size for sparse matrix"),
        F("earth_radius_km", "f", "Earth radius in km"),
    )


@dataclass
class FrVpVs(ControlFile):
    """``frvpvs.in``: Frechet derivatives for Vp/Vs from S ray paths."""

    ray_file: str = "rays.dat"
    vpvs_grid_file: str = "vgrids.in"
    vpvs_ref_grid_file: str = "vgridsref.in"
    vp_ref_grid_file: str = "vgridspr.in"
    vs_ref_grid_file: str = "vgridssr.in"
    vs_grid_file: str = "vgridss.in"
    src_file: str = "src.dat"
    otimes_s_file: str = "otimess.dat"
    otimes_p_file: str = "otimesp.dat"
    otimes_file: str = "otimes.dat"
    mtimes_file: str = "mtimes.dat"
    receivers_p_file: str = "sourcesp.in"
    receivers_s_file: str = "sourcess.in"
    frechet_file: str = "frechet.dat"
    sandr_file: str = "sandr.dat"

    _pad = 23
    _layout = (
        F("ray_file", "w", "S-ray path file"),
        F("vpvs_grid_file", "w", "Vp/Vs grid file"),
        F("vpvs_ref_grid_file", "w", "Reference Vp/Vs grid file"),
        F("vp_ref_grid_file", "w", "Vp reference grid file"),
        F("vs_ref_grid_file", "w", "Vs reference grid file"),
        F("vs_grid_file", "w", "Vs current grid file"),
        F("src_file", "w", "Source-receiver associations"),
        F("otimes_s_file", "w", "Observed times for S"),
        F("otimes_p_file", "w", "Observed times for P"),
        F("otimes_file", "w", "Observed data"),
        F("mtimes_file", "w", "Model data"),
        F("receivers_p_file", "w", "P receivers"),
        F("receivers_s_file", "w", "S receivers"),
        F("frechet_file", "w", "Output Frechet derivatives"),
        F("sandr_file", "w", "Total number of sources and receivers"),
    )


@dataclass
class VpVsResids(ControlFile):
    """``vpvsresids.in``: residual files to compare."""

    otimes_file: str = "otimes.dat"
    mtimes_file: str = "mtimes.dat"

    _pad = 32
    _layout = (
        F("otimes_file", "w", "Observed residuals"),
        F("mtimes_file", "w", "Model residuals"),
    )


# ---- data files -----------------------------------------------------------------------


def parse_src_dat(text: str, *, source: str | Path | None = None) -> pd.DataFrame:
    """Parse ``src.dat``: DataFrame ``block``, ``id``, ``time``, ``uncertainty`` (one row per pick).

    ``block`` is the (1-based) position of the block in the file; blocks without picks leave no
    row, so ``attrs['n_blocks']`` keeps the block count.
    """
    rd = RecordReader(text, source)
    nb = rd.int()
    rows = []
    for _ in range(nb):
        blk, n = rd.ints(2)
        for _ in range(n):
            ident, t, e = rd.tokens(3)
            rows.append((blk, int(ident), float(t), float(e)))
    rd.expect_eof("src.dat")
    df = pd.DataFrame(rows, columns=["block", "id", "time", "uncertainty"])
    df["block"] = df["block"].astype("int64")
    df["id"] = df["id"].astype("int64")
    df.attrs["n_blocks"] = nb
    return df


def read_src_dat(path: str | Path) -> pd.DataFrame:
    """Read ``src.dat``."""
    return parse_src_dat(read_text(path), source=path)


def format_src_dat(df: pd.DataFrame, *, n_blocks: int | None = None) -> str:
    """Format as ``src.dat`` (blocks in order of ``block``; missing blocks written with 0 picks)."""
    nb = (
        n_blocks
        if n_blocks is not None
        else int(df.attrs.get("n_blocks", df["block"].max() if len(df) else 0))
    )
    groups = {k: g for k, g in df.groupby("block", sort=False)}
    out = [fint(nb)]
    for b in range(1, nb + 1):
        g = groups.get(b)
        n = 0 if g is None else len(g)
        out.append(fint(b) + fint(n))
        if g is not None:
            out.extend(
                fline(int(r.id), float(r.time), float(r.uncertainty))
                for r in g.itertuples(index=False)
            )
    return "\n".join(out) + "\n"


def write_src_dat(path: str | Path, df: pd.DataFrame, **kwargs: int | None) -> Path:
    """Write ``src.dat``."""
    return write_text(path, format_src_dat(df, **kwargs))


def parse_sandr(text: str, *, source: str | Path | None = None) -> tuple[int, ...]:
    """Parse ``sandr.dat``: the integers of the file (``vpvsinv.f90:135`` reads only the first)."""
    out: list[int] = []
    rd = RecordReader(text, source)
    while not rd.at_eof():
        out.append(rd.int())
    return tuple(out)


def read_sandr(path: str | Path) -> tuple[int, ...]:
    """Read ``sandr.dat``."""
    return parse_sandr(read_text(path), source=path)


def format_sandr(values: tuple[int, ...] | list[int]) -> str:
    """Format as ``sandr.dat`` (one integer per line)."""
    return "\n".join(fint(v) for v in values) + "\n"


def write_sandr(path: str | Path, values: tuple[int, ...] | list[int]) -> Path:
    """Write ``sandr.dat``."""
    return write_text(path, format_sandr(values))


SERROR_COLUMNS = ["depth_km", "lat", "lon", "err_depth_km", "err_lat", "err_lon"]


def parse_serror(text: str, *, source: str | Path | None = None) -> pd.DataFrame:
    """Parse ``serror.dat`` of ``gridlocj`` (``gridlocj.f90:885``): one line per relocated event.

    Columns ``-srad, slat, slon`` (the relocated position as the program stores it; ``-srad`` is
    written as ``depth_km``, an event that could not be relocated has all three set to 0) and the
    ``edep, elat, elon`` confidence-interval half widths.  The file has no header.
    """
    rd = RecordReader(text, source)
    rows = []
    while not rd.at_eof():
        rows.append(rd.floats(6))
    return pd.DataFrame(rows, columns=SERROR_COLUMNS)


def read_serror(path: str | Path) -> pd.DataFrame:
    """Read ``serror.dat``."""
    return parse_serror(read_text(path), source=path)
