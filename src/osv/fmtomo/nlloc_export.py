"""Export the final 3-D model and its station corrections for NonLinLoc.

Writes ``<root>.P.mod.{hdr,buf}`` and ``<root>.S.mod.{hdr,buf}`` (``SLOW_LEN``, cubic cells,
``TRANS SIMPLE`` about ``origin``) and ``<root>_station_delays.csv`` (``station``, ``phase``,
``delay_s``, ``n``: the corrections the model was inverted with, for ``LOCDELAY``).

The model is evaluated as ``fm3d`` evaluates it, the tensor-product cubic B-spline of the nodes
(:func:`osv.plotting.slices.bspline_sample`).  The NonLinLoc grid usually reaches beyond the
FMTOMO grid (NonLinLoc's location grid must lie inside it); there the node grid is padded by
repeating its edge nodes, which are the unchanged 1-D starting model (the cushion), so the
export continues the 1-D model laterally, the top layer above and the half-space below.  For a
P-only inversion the S grid keeps the starting model's Vp/Vs.

The model only applies with the same station corrections: the observed times it was inverted
with had them subtracted (Stage 3 terms, iterated terms or VELEST delays, whichever was used).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.io.nlloc import NllocGrid, simple_xy_to_latlon, write_nlloc_grid
from osv.plotting.slices import bspline_sample

log = logging.getLogger(__name__)

__all__ = ["NllocExport", "export_nlloc", "pad_model", "station_corrections"]


class NllocExportError(RuntimeError):
    """The project has no usable final model or the request is invalid."""


@dataclass
class NllocExport:
    """Files and summary of :func:`export_nlloc`."""

    out_dir: Path
    roots: dict[str, Path]
    delays_csv: Path | None
    summary: dict[str, Any] = field(default_factory=dict)


def pad_model(
    ds: xr.Dataset, depth: tuple[float, float], lat: tuple[float, float], lon: tuple[float, float]
) -> xr.Dataset:
    """Extend the node grid by edge replication so that the B-spline covers the given ranges.

    ``fm3d``'s B-spline is valid from the second node to the second-to-last node of each axis;
    the grid is padded (same spacing, values of the outermost nodes) until ``[lo, hi]`` of every
    axis lies inside that range.
    """
    pads: dict[str, tuple[int, int]] = {}
    coords: dict[str, np.ndarray] = {}
    for dim, (lo, hi) in (("depth", depth), ("lat", lat), ("lon", lon)):
        x = ds[dim].values.astype(float)
        d = (x[-1] - x[0]) / (x.size - 1)
        n_lo = max(0, math.ceil((x[1] - lo) / d) + 1)
        n_hi = max(0, math.ceil((hi - x[-2]) / d) + 1)
        pads[dim] = (n_lo, n_hi)
        coords[dim] = x[0] + d * np.arange(-n_lo, x.size + n_hi)
    out = {}
    for name, v in ds.data_vars.items():
        if set(v.dims) != {"depth", "lat", "lon"}:
            continue
        a = v.transpose("depth", "lat", "lon").values
        out[name] = (
            ("depth", "lat", "lon"),
            np.pad(a, [pads["depth"], pads["lat"], pads["lon"]], mode="edge"),
        )
    return xr.Dataset(out, coords=coords)


def station_corrections(project_dir: Path, cfg: Any) -> tuple[pd.DataFrame | None, str]:
    """The station corrections the 3-D model was inverted with.

    Returns ``(table, source)``; ``table`` has ``station`` (station code), ``phase``,
    ``delay_s`` and ``n`` (``None`` if no corrections were used).
    """
    from osv.fmtomo.station_terms import TERMS_FILE
    from osv.model3d.stage import FMTOMO_DIR

    p = Path(project_dir)
    alias = _alias_map(p)
    terms = None
    if getattr(cfg.model3d, "estimate_station_terms", False):
        f = p / "04_build_model" / FMTOMO_DIR / TERMS_FILE
        source = f"Stage 3 station terms ({f})"
        terms = pd.read_csv(f)
    elif (p / "06_invert" / "station_terms.csv").is_file():
        f = p / "06_invert" / "station_terms.csv"
        source = f"iterated station terms ({f})"
        terms = pd.read_csv(f)
    elif cfg.model3d.apply_velest_station_delays:
        f = p / "03_velest_analyse" / "station_delays.csv"
        dl = pd.read_csv(f)
        excl = cfg.model3d.velest_station_delays_exclude
        if excl == "borehole":
            raise NllocExportError(
                "velest_station_delays_exclude: borehole is not supported by the export; list the "
                "station codes instead"
            )
        key = "alias" if "alias" in dl else "sta"
        dl = dl[~dl[key].astype(str).isin([str(s) for s in excl])]
        rows = []
        for ph, col in (("P", "ptcor"), ("S", "stcor")):
            if col in dl:
                for r in dl.itertuples(index=False):
                    v = getattr(r, col)
                    if pd.notna(v):
                        nobs = int(getattr(r, "nobs", 0))
                        rows.append((str(getattr(r, key)), ph, float(v), nobs))
        terms = pd.DataFrame(rows, columns=["station", "phase", "term_s", "n"])
        source = f"VELEST station delays ({f})"
    else:
        return None, "none (the observed times were not corrected)"
    out = pd.DataFrame(
        {
            "station": terms["station"].astype(str).map(lambda s: alias.get(s, s)),
            "phase": terms["phase"].astype(str).str.upper(),
            "delay_s": terms["term_s"].astype(float).round(5),
            "n": terms["n"].fillna(0).astype(int),
        }
    )
    return out.sort_values(["phase", "station"]).reset_index(drop=True), source


def _alias_map(project_dir: Path) -> dict[str, str]:
    """VELEST alias -> station code (Stage 1 ``aliases.csv``)."""
    f = project_dir / "01_prep" / "aliases.csv"
    if not f.is_file():
        return {}
    a = pd.read_csv(f)
    return dict(zip(a["alias"].astype(str), a["sta"].astype(str), strict=True))


def _s_model(project_dir: Path, final: xr.Dataset) -> tuple[xr.DataArray, str]:
    if "vs" in final:
        return final["vs"], "inverted Vs"
    start = xr.load_dataset(project_dir / "04_build_model" / "starting_model.nc")
    ratio = (start["vp"] / start["vs"]).transpose("depth", "lat", "lon")
    if ratio.shape != final["vp"].transpose("depth", "lat", "lon").shape:
        raise NllocExportError("starting_model.nc and final_model.nc grids differ")
    vs = final["vp"].transpose("depth", "lat", "lon") / ratio.values
    return vs, "final Vp / starting-model Vp/Vs"


def export_nlloc(
    project_dir: str | Path,
    cfg: Any,
    out_dir: str | Path,
    *,
    root: str = "model",
    origin: tuple[float, float] | None = None,
    half_width_km: float = 10.5,
    z_range_km: tuple[float, float] = (-2.5, 10.5),
    spacing_km: float = 0.05,
) -> NllocExport:
    """Write the NonLinLoc P and S model grids and the station-correction CSV.

    Parameters
    ----------
    project_dir
        ``project.workdir`` with a completed ``06_invert``.
    out_dir, root
        Files ``<out_dir>/<root>.{P,S}.mod.{hdr,buf}``, ``<root>_station_delays.csv`` and
        ``<root>_export.json``.
    origin
        ``TRANS SIMPLE`` origin ``(lat, lon)``; must equal NonLinLoc's (OpenMEQ
        ``nlloc.grid.origin``).  Default ``coordinates.origin``.
    half_width_km, z_range_km
        Grid extent: x and y in ``[-half_width_km, half_width_km]``, z (km, positive down from
        sea level) in ``z_range_km``; it must contain NonLinLoc's location grid.
    spacing_km
        Node spacing (cubic cells, as ``Grid2Time``'s finite differences require).  The
        Podvin-Lecomte scheme overestimates times near a station in a slow, steep-gradient top
        layer roughly in proportion to the spacing (MacBan, vertical rays: +19 ms at 0.1 km,
        +8 ms at 0.05 km, +4 ms at 0.025 km); most of it is a per-station constant that the
        origin time and LOCDELAY absorb.

    Raises
    ------
    NllocExportError
        Without a completed ``06_invert`` or an origin.
    """
    p = Path(project_dir)
    fm = p / "06_invert" / "final_model.nc"
    if not fm.is_file():
        raise NllocExportError(f"{fm} not found: run the invert stage first")
    if origin is None:
        o = cfg.coordinates.origin
        if o is None:
            raise NllocExportError("no origin: pass origin=(lat, lon) or set coordinates.origin")
        origin = (float(o.lat), float(o.lon))
    lat0, lon0 = origin
    if spacing_km <= 0 or half_width_km <= 0 or z_range_km[1] <= z_range_km[0]:
        raise NllocExportError("spacing, half width and z range must be positive")
    final = xr.load_dataset(fm)
    n = int(round(2 * half_width_km / spacing_km)) + 1
    xs = -half_width_km + spacing_km * np.arange(n)
    nz = int(round((z_range_km[1] - z_range_km[0]) / spacing_km)) + 1
    zs = z_range_km[0] + spacing_km * np.arange(nz)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    LAT, LON = simple_xy_to_latlon(X, Y, lat0, lon0)
    vs, s_source = _s_model(p, final)
    ds = xr.Dataset({"vp": final["vp"], "vs": vs.transpose("depth", "lat", "lon")})
    padded = pad_model(
        ds,
        (float(zs[0]), float(zs[-1])),
        (float(LAT.min()), float(LAT.max())),
        (float(LON.min()), float(LON.max())),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots: dict[str, Path] = {}
    stats: dict[str, Any] = {}
    for ph, var in (("P", "vp"), ("S", "vs")):
        v = bspline_sample(padded, var, zs[None, None, :], LAT[:, :, None], LON[:, :, None])
        if not np.isfinite(v).all() or (v <= 0).any():
            raise NllocExportError(f"{ph}: non-finite or non-positive velocities in the export")
        g = NllocGrid(
            (spacing_km / v).astype(np.float32),
            (float(xs[0]), float(xs[0]), float(zs[0])),
            (spacing_km,) * 3,
            "SLOW_LEN",
            lat0,
            lon0,
        )
        r = out / f"{root}.{ph}.mod"
        write_nlloc_grid(r, g)
        roots[ph] = r
        stats[ph] = {"v_min_km_s": round(float(v.min()), 3), "v_max_km_s": round(float(v.max()), 3)}
    delays, dsrc = station_corrections(p, cfg)
    dcsv = None
    if delays is not None:
        dcsv = out / f"{root}_station_delays.csv"
        delays.to_csv(dcsv, index=False)
    inside = (
        (LAT >= float(final["lat"].values[1]))
        & (LAT <= float(final["lat"].values[-2]))
        & (LON >= float(final["lon"].values[1]))
        & (LON <= float(final["lon"].values[-2]))
    )
    summary = {
        "project": str(p),
        "model": str(fm),
        "s_model": s_source,
        "station_corrections": dsrc,
        "origin": [lat0, lon0],
        "transform": f"TRANS SIMPLE {lat0} {lon0} 0.0",
        "grid": {
            "x_km": [float(xs[0]), float(xs[-1])],
            "y_km": [float(xs[0]), float(xs[-1])],
            "z_km": [float(zs[0]), float(zs[-1])],
            "spacing_km": spacing_km,
            "nodes": [n, n, nz],
            "type": "SLOW_LEN",
        },
        "fraction_inside_fmtomo_grid": round(float(inside.mean()), 3),
        "velocities": stats,
        "note": "use with the station corrections in the CSV (LOCDELAY); outside the FMTOMO grid "
        "the 1-D starting model continues",
    }
    (out / f"{root}_export.json").write_text(json.dumps(summary, indent=1))
    log.info("NonLinLoc export written to %s (%s)", out, ", ".join(str(r) for r in roots.values()))
    return NllocExport(out, roots, dcsv, summary)
