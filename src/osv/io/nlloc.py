"""NonLinLoc 3-D grid files (``<root>.<PHASE>.mod.hdr`` / ``.buf``) and the SIMPLE transform.

Conventions are those of the NonLinLoc sources (``GridLib.c``):

* **Header** (``WriteGrid3dHdr``): line 1 ``numx numy numz origx origy origz dx dy dz
  gridType floatType``; a model grid has no source line; then the transform line
  ``TRANSFORM  SIMPLE LatOrig <lat>  LongOrig <lon>  RotCW <rot>``.  Coordinates are km in
  the transform's frame, ``z`` positive down from the transform depth (sea level here).
* **Buffer** (``ReadGrid3dBuf``/``ReadGrid3dBufSheet``): 4-byte floats (the usual build has
  ``GRID_FLOAT_TYPE_DOUBLE`` off), x outermost, then y, z fastest - ``array[ix][iy][iz]``.
* **Model values**: ``SLOW_LEN`` = slowness (s/km) x node spacing (km), what ``Vel2Grid``
  writes and the Podvin-Lecomte finite-difference ``Grid2Time`` expects (cubic cells).
* **SIMPLE transform** (``latlon2rect``): ``x = dlon * c111 * cos(lat)`` (the point's own
  latitude), ``y = dlat * c111``, ``c111 = DEG2KM = pi * 6371.0087714 / 180`` km per degree.

``LOCDELAY`` (``NLLocLib.c:2542``) subtracts the delay from the observed time, the same sign
as the osv station terms and VELEST delays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DEG2KM",
    "NllocGrid",
    "read_nlloc_grid",
    "simple_latlon_to_xy",
    "simple_xy_to_latlon",
    "write_nlloc_grid",
]

#: NonLinLoc's km per degree (``geo.h``: ``PI * AVG_ERAD / 180``, ``AVG_ERAD = 6371.0087714``)
DEG2KM = math.pi * 6371.0087714 / 180.0
_MODEL_TYPES = ("VELOCITY", "VELOCITY_METERS", "SLOWNESS", "VEL2", "SLOW2", "SLOW_LEN")


@dataclass
class NllocGrid:
    """A NonLinLoc 3-D grid (values ``(numx, numy, numz)``, km, SIMPLE transform)."""

    values: np.ndarray
    orig: tuple[float, float, float]
    spacing: tuple[float, float, float]
    grid_type: str
    lat0: float
    lon0: float
    rot_cw: float = 0.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.values.shape)  # type: ignore[return-value]

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Node coordinates ``x, y, z`` (km)."""
        sizes = zip(self.orig, self.spacing, self.shape, strict=True)
        return tuple(o + d * np.arange(n) for o, d, n in sizes)  # type: ignore[return-value]

    def transform_line(self) -> str:
        return (
            f"TRANSFORM  SIMPLE LatOrig {self.lat0:.6f}  LongOrig {self.lon0:.6f}  "
            f"RotCW {self.rot_cw:.6f}"
        )


def simple_latlon_to_xy(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    """NonLinLoc ``TRANS SIMPLE`` (no rotation): lat/lon (degrees) to x east, y north (km)."""
    lat = np.asarray(lat, dtype=float)
    dlon = (np.asarray(lon, dtype=float) - lon0 + 180.0) % 360.0 - 180.0
    return dlon * DEG2KM * np.cos(np.deg2rad(lat)), (lat - lat0) * DEG2KM


def simple_xy_to_latlon(
    x: np.ndarray, y: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`simple_latlon_to_xy` (``rect2latlon``)."""
    lat = lat0 + np.asarray(y, dtype=float) / DEG2KM
    lon = lon0 + np.asarray(x, dtype=float) / (DEG2KM * np.cos(np.deg2rad(lat)))
    return lat, lon


def write_nlloc_grid(root: str | Path, grid: NllocGrid) -> tuple[Path, Path]:
    """Write ``<root>.hdr`` and ``<root>.buf`` (``root`` e.g. ``out/model.P.mod``).

    Raises
    ------
    ValueError
        If the grid type is not a model type, the values are not 3-D or not finite.
    """
    if grid.grid_type not in _MODEL_TYPES:
        raise ValueError(f"grid type must be one of {_MODEL_TYPES}, got {grid.grid_type!r}")
    v = np.asarray(grid.values)
    if v.ndim != 3:
        raise ValueError(f"grid values must be 3-D (numx, numy, numz), got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("grid values must be finite")
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = v.shape
    (ox, oy, oz), (dx, dy, dz) = grid.orig, grid.spacing
    hdr = Path(f"{root}.hdr")
    buf = Path(f"{root}.buf")
    hdr.write_text(
        f"{nx} {ny} {nz}  {ox:.6f} {oy:.6f} {oz:.6f}  {dx:.6f} {dy:.6f} {dz:.6f} "
        f"{grid.grid_type} FLOAT\n{grid.transform_line()}\n"
    )
    np.ascontiguousarray(v, dtype="<f4").tofile(buf)
    return hdr, buf


def read_nlloc_grid(root: str | Path) -> NllocGrid:
    """Read ``<root>.hdr`` and ``<root>.buf`` (model or travel-time grids, SIMPLE transform).

    Raises
    ------
    ValueError
        If the header is malformed, not FLOAT, or the buffer size does not match.
    """
    root = Path(root)
    lines = Path(f"{root}.hdr").read_text().splitlines()
    f = lines[0].split()
    if len(f) < 11:
        raise ValueError(f"{root}.hdr: line 1 needs 11 fields, got {lines[0]!r}")
    nx, ny, nz = (int(t) for t in f[:3])
    orig = tuple(float(t) for t in f[3:6])
    sp = tuple(float(t) for t in f[6:9])
    if f[10] != "FLOAT":
        raise ValueError(f"{root}.hdr: only FLOAT grids are supported, got {f[10]!r}")
    lat0 = lon0 = rot = math.nan
    for ln in lines[1:]:
        t = ln.split()
        if t and t[0] == "TRANSFORM" and len(t) > 1 and t[1] == "SIMPLE":
            lat0 = float(t[t.index("LatOrig") + 1])
            lon0 = float(t[t.index("LongOrig") + 1])
            rot = float(t[t.index("RotCW") + 1])
    data = np.fromfile(f"{root}.buf", dtype="<f4")
    if data.size != nx * ny * nz:
        raise ValueError(f"{root}.buf has {data.size} values, header says {nx * ny * nz}")
    return NllocGrid(
        data.reshape(nx, ny, nz),
        orig,  # type: ignore[arg-type]
        sp,  # type: ignore[arg-type]
        f[9],
        lat0,
        lon0,
        rot,
    )
