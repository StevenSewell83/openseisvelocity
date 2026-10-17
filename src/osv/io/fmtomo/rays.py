"""``rays.dat``: ray paths written by ``fm3d`` (``save_rays_mode``).

Format (``3dfmlib.f90:4752-4817``, ``README_0.7``; read back by ``gmtslice.f90:1357-1376``)::

    receiver source ray subray n_sections              (5i6; n_sections = 0 for an invalid ray)
    per section:
        npoints region diffracted headwave            (2i6,2l5)
        npoints records "r lat lon"                   (3f17.8)

``r`` is the radius in km, ``lat`` and ``lon`` are **radians** (``gmtslice.f90:1369-1372``
converts with ``*180/pi`` and subtracts the Earth radius).  Points are written from index
``npoints`` down to 1 (``3dfmlib.f90:4779``); the file order is kept here and runs from the source
(first point of the first section) to the receiver (last point of the last section), verified
with a real ``fm3d`` run on example 1.  A ray section is the part of a ray inside one region
(between two interfaces); ``region`` is the region id.  No example ``rays.dat`` ships with the
vendored distribution; ``tests/data/fmtomo/example1/rays_excerpt.dat`` holds three rays of a real
run (``save_rays_mode = T``) and the unit tests also use a hand-built file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from osv.coords import FMTOMO_EARTH_RADIUS_KM

from ._fortran import FmtomoFormatError, RecordReader, read_text, write_text

__all__ = ["RayPaths", "format_rays", "parse_rays", "read_rays", "ray_hit_count", "write_rays"]


@dataclass
class RayPaths:
    """Ray paths of a ``rays.dat`` file.

    ``rays``: one row per ray in file order (``receiver``, ``source``, ``ray``, ``subray``,
    ``n_sections``, ``valid``).  ``sections``: ``ray`` (row position in ``rays``), ``section``
    (1-based), ``npoints``, ``region``, ``diffracted``, ``headwave``.  ``points``: ``ray``,
    ``section``, ``r_km``, ``lat_rad``, ``lon_rad`` (as in the file) and derived ``depth_km``,
    ``lat``, ``lon`` (degrees).
    """

    rays: pd.DataFrame
    sections: pd.DataFrame
    points: pd.DataFrame
    earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM


def _tf(v: object) -> str:
    return "T" if v else "F"


def _logical(tok: str, rd: RecordReader) -> bool:
    t = tok.lstrip(".").upper()[:1]
    if t not in ("T", "F"):
        raise rd.error(f"expected a logical (T/F), got {tok!r}")
    return t == "T"


def parse_rays(
    text: str, *, source: str | Path | None = None, earth_radius_km: float = FMTOMO_EARTH_RADIUS_KM
) -> RayPaths:
    """Parse the text of ``rays.dat``."""
    rd = RecordReader(text, source)
    ray_rows: list[tuple] = []
    sec_rows: list[tuple] = []
    pts: list[list[float]] = []
    pt_ray: list[int] = []
    pt_sec: list[int] = []
    while not rd.at_eof():
        rec, src, pth, sub, nsec = (int(t) for t in _ints(rd, 5))
        ray_idx = len(ray_rows)
        if nsec < 0:
            raise rd.error(f"ray {ray_idx + 1}: negative number of sections")
        ray_rows.append((rec, src, pth, sub, nsec, nsec > 0))
        for s in range(1, nsec + 1):
            toks = rd.tokens(4)
            try:
                npts, region = int(toks[0]), int(toks[1])
            except ValueError:
                raise rd.error(
                    f"ray {ray_idx + 1} section {s}: bad section header {toks!r}"
                ) from None
            sec_rows.append(
                (ray_idx, s, npts, region, _logical(toks[2], rd), _logical(toks[3], rd))
            )
            if rd.remaining() < npts:
                raise rd.error(
                    f"ray {ray_idx + 1} section {s} declares {npts} points but the file ends early"
                )
            for _ in range(npts):
                parts = rd.next_record().split()
                while not parts:
                    parts = rd.next_record().split()
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except (ValueError, IndexError):
                    raise rd.error(
                        f"ray {ray_idx + 1} section {s}: bad point record {parts!r}"
                    ) from None
                pt_ray.append(ray_idx)
                pt_sec.append(s)
    rays = pd.DataFrame(
        ray_rows, columns=["receiver", "source", "ray", "subray", "n_sections", "valid"]
    )
    sections = pd.DataFrame(
        sec_rows, columns=["ray", "section", "npoints", "region", "diffracted", "headwave"]
    )
    arr = np.array(pts, dtype=float).reshape(-1, 3)
    points = pd.DataFrame(
        {
            "ray": np.array(pt_ray, dtype="int64"),
            "section": np.array(pt_sec, dtype="int64"),
            "r_km": arr[:, 0],
            "lat_rad": arr[:, 1],
            "lon_rad": arr[:, 2],
        }
    )
    points["depth_km"] = earth_radius_km - points["r_km"]
    points["lat"] = np.rad2deg(points["lat_rad"])
    points["lon"] = np.rad2deg(points["lon_rad"])
    return RayPaths(rays, sections, points, earth_radius_km)


def _ints(rd: RecordReader, n: int) -> list[int]:
    return rd.ints(n)


def read_rays(path: str | Path, **kwargs: object) -> RayPaths:
    """Read ``rays.dat``."""
    return parse_rays(read_text(path), source=path, **kwargs)  # type: ignore[arg-type]


def format_rays(rp: RayPaths) -> str:
    """Format as ``rays.dat`` (``(5i6)``, ``(2i6,2l5)``, ``(3f17.8)``)."""
    pts = rp.points
    if {"lat_rad", "lon_rad", "r_km"} <= set(pts.columns):
        r, la, lo = pts["r_km"].to_numpy(), pts["lat_rad"].to_numpy(), pts["lon_rad"].to_numpy()
    elif {"depth_km", "lat", "lon"} <= set(pts.columns):
        r = rp.earth_radius_km - pts["depth_km"].to_numpy()
        la, lo = np.deg2rad(pts["lat"].to_numpy()), np.deg2rad(pts["lon"].to_numpy())
    else:
        raise FmtomoFormatError("points table needs r_km/lat_rad/lon_rad or depth_km/lat/lon")
    index = pts.groupby(["ray", "section"], sort=False).indices
    sec_by_ray = {k: g for k, g in rp.sections.groupby("ray", sort=False)}
    out: list[str] = []
    for i, row in enumerate(rp.rays.itertuples(index=False)):
        secs = sec_by_ray.get(i)
        n_sec = 0 if secs is None else len(secs)
        if bool(row.valid) is False and n_sec:
            n_sec = 0
        out.append(f"{row.receiver:6d}{row.source:6d}{row.ray:6d}{row.subray:6d}{n_sec:6d}")
        if not n_sec:
            continue
        for s in secs.itertuples(index=False):
            idx = index.get((i, s.section))
            n = 0 if idx is None else len(idx)
            if n != s.npoints:
                raise FmtomoFormatError(
                    f"ray {i + 1} section {s.section}: npoints={s.npoints} but {n} point records "
                    "present"
                )
            out.append(f"{s.npoints:6d}{s.region:6d}{_tf(s.diffracted):>5}{_tf(s.headwave):>5}")
            out.extend(f"{r[j]:17.8f}{la[j]:17.8f}{lo[j]:17.8f}" for j in idx)
    return "\n".join(out) + "\n"


def write_rays(path: str | Path, rp: RayPaths) -> Path:
    """Write ``rays.dat``."""
    return write_text(path, format_rays(rp))


def ray_hit_count(
    rp: RayPaths,
    *,
    depth_edges: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    valid_only: bool = True,
) -> xr.DataArray:
    """Number of ray points per cell of a regular box grid (a simple ray-coverage measure).

    Ray points are spaced roughly one propagation-grid step apart, so counts are proportional
    to path length per cell.  Edges are in km (depth, positive down) and degrees.
    """
    pts = rp.points
    if valid_only:
        ok = rp.rays["valid"].to_numpy()[pts["ray"].to_numpy()]
        pts = pts[ok]
    hist, _ = np.histogramdd(
        np.column_stack([pts["depth_km"], pts["lat"], pts["lon"]]),
        bins=[np.asarray(depth_edges), np.asarray(lat_edges), np.asarray(lon_edges)],
    )
    de, la, lo = (np.asarray(e, dtype=float) for e in (depth_edges, lat_edges, lon_edges))
    return xr.DataArray(
        hist,
        dims=("depth", "lat", "lon"),
        coords={
            "depth": 0.5 * (de[:-1] + de[1:]),
            "lat": 0.5 * (la[:-1] + la[1:]),
            "lon": 0.5 * (lo[:-1] + lo[1:]),
        },
        name="ray_points",
    )
