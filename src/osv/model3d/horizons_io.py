"""Readers that grid geological surfaces onto the model's lateral nodes (Stage 3, task 4.5).

Every reader returns ``z(lat, lon)`` as a 2-D array on ``(grid.lat_full, grid.lon_full)``
(or the interior nodes with ``full=False``) in **depth, km, positive down, below sea level**
(NaN where the surface is undefined).  The file's own convention is declared by
``z_convention`` (``"depth"`` positive down or ``"elevation"`` positive up) and ``units``
(``"m"`` or ``"km"``); the horizontal CRS by ``crs`` (anything :class:`pyproj.CRS` accepts).
With ``crs=None`` x/y are taken as longitude/latitude (GeoTIFF: the file's own CRS).

Interpolation is done in the file's native horizontal coordinates: the model nodes are
transformed into the file CRS (not the other way round), so the surface is not distorted by
the projection.  Scattered points and triangulated surfaces are interpolated linearly
(no extrapolation outside the convex hull / mesh); rasters bilinearly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from osv.model3d.grid import ModelGrid

log = logging.getLogger(__name__)

__all__ = [
    "HorizonReadError",
    "read_geotiff",
    "read_horizon",
    "read_tsurf",
    "read_xyz",
    "z_to_depth_km",
]

WGS84 = CRS.from_epsg(4326)
_UNIT_TO_KM = {"m": 1e-3, "km": 1.0}


class HorizonReadError(ValueError):
    """A surface file cannot be read or interpreted (message says what to fix)."""


def z_to_depth_km(
    z: Any, z_convention: Literal["depth", "elevation"] = "depth", units: str = "m"
) -> np.ndarray:
    """Convert z values to depth below sea level in km (positive down)."""
    if units not in _UNIT_TO_KM:
        raise HorizonReadError(f"units must be 'm' or 'km', got {units!r}")
    if z_convention not in ("depth", "elevation"):
        raise HorizonReadError(f"z_convention must be 'depth' or 'elevation', got {z_convention!r}")
    a = np.asarray(z, dtype=float) * _UNIT_TO_KM[units]
    return -a if z_convention == "elevation" else a


def _axes(grid: ModelGrid, full: bool) -> tuple[np.ndarray, np.ndarray]:
    return (grid.lat_full, grid.lon_full) if full else (grid.lat, grid.lon)


def nodes_in_crs(
    lat: np.ndarray, lon: np.ndarray, crs: Any | None
) -> tuple[np.ndarray, np.ndarray]:
    """Lateral node positions ``(X, Y)`` with shape ``(nlat, nlon)`` in ``crs`` (None = lon/lat)."""
    lo2, la2 = np.meshgrid(lon, lat)
    if crs is None or CRS.from_user_input(crs) == WGS84:
        return lo2, la2
    tr = Transformer.from_crs(WGS84, CRS.from_user_input(crs), always_xy=True)
    x, y = tr.transform(lo2, la2)
    return np.asarray(x, float), np.asarray(y, float)


def _check_overlap(z: np.ndarray, source: Path, what: str) -> np.ndarray:
    if not np.isfinite(z).any():
        raise HorizonReadError(
            f"{source}: {what} does not overlap the model grid; check crs/units "
            "(is the file in the CRS given by 'crs'?)"
        )
    return z


# ----------------------------------------------------------------------------------------
# XYZ ASCII
# ----------------------------------------------------------------------------------------


def _load_xyz(path: Path) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            t = line.strip()
            if not t or t[0] in "#>/!":
                continue
            parts = re.split(r"[,;\s]+", t)
            try:
                vals = [float(p) for p in parts[:3]]
            except ValueError:
                if not rows:  # header line before any data
                    continue
                raise HorizonReadError(f"{path}:{ln}: cannot parse '{t[:60]}' as x y z") from None
            if len(vals) < 3:
                raise HorizonReadError(f"{path}:{ln}: expected 3 columns (x y z), got {len(vals)}")
            rows.append(vals)
    if len(rows) < 3:
        raise HorizonReadError(f"{path}: fewer than 3 data rows; need x y z points")
    return np.asarray(rows, float)


def read_xyz(
    path: str | Path,
    grid: ModelGrid,
    *,
    crs: Any | None = None,
    z_convention: Literal["depth", "elevation"] = "depth",
    units: str = "m",
    full: bool = True,
) -> np.ndarray:
    """Grid an ASCII ``x y z`` point file (whitespace/comma separated, optional header).

    Lines starting with ``#`` and non-numeric header lines before the first data row are
    skipped.  Points are interpolated linearly (Delaunay); NaN outside their convex hull.
    """
    path = Path(path)
    pts = _load_xyz(path)
    pts = pts[np.isfinite(pts).all(axis=1)]
    lat, lon = _axes(grid, full)
    X, Y = nodes_in_crs(lat, lon, crs)
    # centre/scale for numerical robustness of the triangulation
    x0, y0 = pts[:, 0].mean(), pts[:, 1].mean()
    sc = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-12)
    try:
        f = LinearNDInterpolator(
            np.column_stack([(pts[:, 0] - x0) / sc, (pts[:, 1] - y0) / sc]), pts[:, 2]
        )
    except Exception as exc:  # QhullError for degenerate (collinear) input
        raise HorizonReadError(f"{path}: cannot triangulate points ({exc})") from exc
    z = f((X - x0) / sc, (Y - y0) / sc)
    return _check_overlap(z_to_depth_km(z, z_convention, units), path, "surface")


# ----------------------------------------------------------------------------------------
# GeoTIFF
# ----------------------------------------------------------------------------------------


def read_geotiff(
    path: str | Path,
    grid: ModelGrid,
    *,
    crs: Any | None = None,
    z_convention: Literal["depth", "elevation"] = "depth",
    units: str = "m",
    band: int = 1,
    full: bool = True,
) -> np.ndarray:
    """Sample band ``band`` of a GeoTIFF bilinearly at the model nodes (needs ``rasterio``).

    The CRS stored in the file is used unless ``crs`` overrides it; nodata cells become NaN.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Reading GeoTIFF horizons needs the optional 'rasterio' package: "
            "install it with `pip install rasterio` (or `pip install openseisvelocity[geo]`)"
        ) from exc
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read(band).astype(float)
        nodata = src.nodata
        transform = src.transform
        file_crs = src.crs
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    use_crs = crs if crs is not None else (file_crs.to_wkt() if file_crs else None)
    if use_crs is None:
        raise HorizonReadError(f"{path}: file has no CRS; set the horizon 'crs' option")
    if transform.b != 0 or transform.d != 0:
        raise HorizonReadError(f"{path}: rotated rasters are not supported; reproject first")
    ny, nx = data.shape
    xs = transform.c + transform.a * (np.arange(nx) + 0.5)
    ys = transform.f + transform.e * (np.arange(ny) + 0.5)
    if ys[0] > ys[-1]:  # north-up rasters have decreasing y
        ys, data = ys[::-1], data[::-1]
    if xs[0] > xs[-1]:
        xs, data = xs[::-1], data[:, ::-1]
    lat, lon = _axes(grid, full)
    X, Y = nodes_in_crs(lat, lon, use_crs)
    f = RegularGridInterpolator((ys, xs), data, bounds_error=False, fill_value=np.nan)
    z = f(np.column_stack([Y.ravel(), X.ravel()])).reshape(X.shape)
    return _check_overlap(z_to_depth_km(z, z_convention, units), path, "raster")


# ----------------------------------------------------------------------------------------
# GOCAD TSurf
# ----------------------------------------------------------------------------------------


def _parse_tsurf(path: Path) -> tuple[np.ndarray, np.ndarray, str | None]:
    verts: dict[int, tuple[float, float, float]] = {}
    tris: list[tuple[int, int, int]] = []
    zpos: str | None = None
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            p = line.split()
            if not p:
                continue
            key = p[0]
            try:
                if key in ("VRTX", "PVRTX"):
                    verts[int(p[1])] = (float(p[2]), float(p[3]), float(p[4]))
                elif key in ("ATOM", "PATOM"):
                    verts[int(p[1])] = verts[int(p[2])]
                elif key == "TRGL":
                    tris.append((int(p[1]), int(p[2]), int(p[3])))
                elif key == "ZPOSITIVE":
                    zpos = p[1]
            except (ValueError, IndexError, KeyError) as exc:
                raise HorizonReadError(
                    f"{path}:{ln}: malformed TSurf record '{line.strip()}'"
                ) from exc
    if not verts or not tris:
        raise HorizonReadError(f"{path}: no VRTX/PVRTX vertices or TRGL triangles found")
    ids = sorted(verts)
    index = {v: i for i, v in enumerate(ids)}
    xyz = np.array([verts[i] for i in ids], float)
    try:
        tri = np.array([[index[a], index[b], index[c]] for a, b, c in tris], int)
    except KeyError as exc:
        raise HorizonReadError(f"{path}: TRGL refers to undefined vertex {exc}") from exc
    return xyz, tri, zpos


def read_tsurf(
    path: str | Path,
    grid: ModelGrid,
    *,
    crs: Any | None = None,
    z_convention: Literal["depth", "elevation"] = "depth",
    units: str = "m",
    full: bool = True,
) -> np.ndarray:
    """Interpolate a GOCAD TSurf (``VRTX``/``PVRTX`` + ``TRGL``) at the model nodes.

    Barycentric (linear) interpolation inside each triangle, NaN outside the mesh.  The
    ``ZPOSITIVE`` header is *not* trusted: ``z_convention`` from the configuration rules
    (a mismatch with the header is logged).
    """
    import matplotlib.tri as mtri

    path = Path(path)
    xyz, tri, zpos = _parse_tsurf(path)
    if zpos is not None:
        header = "elevation" if zpos.lower() == "elevation" else "depth"
        if header != z_convention:
            log.warning(
                "%s: header says ZPOSITIVE %s but z_convention='%s' is configured; using the "
                "configuration",
                path.name,
                zpos,
                z_convention,
            )
    lat, lon = _axes(grid, full)
    X, Y = nodes_in_crs(lat, lon, crs)
    x0, y0 = xyz[:, 0].mean(), xyz[:, 1].mean()
    triang = mtri.Triangulation(xyz[:, 0] - x0, xyz[:, 1] - y0, tri)
    interp = mtri.LinearTriInterpolator(triang, xyz[:, 2])
    z = np.ma.filled(interp(X - x0, Y - y0), np.nan).astype(float)
    return _check_overlap(z_to_depth_km(z, z_convention, units), path, "surface")


# ----------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------

_SUFFIX = {
    ".xyz": "xyz",
    ".txt": "xyz",
    ".csv": "xyz",
    ".dat": "xyz",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".gtif": "geotiff",
    ".ts": "tsurf",
    ".tsurf": "tsurf",
}


def read_horizon(
    path: str | Path,
    grid: ModelGrid,
    *,
    crs: Any | None = None,
    z_convention: Literal["depth", "elevation"] = "depth",
    units: str = "m",
    fmt: Literal["xyz", "geotiff", "tsurf"] | None = None,
    full: bool = True,
) -> np.ndarray:
    """Read a surface of any supported format (chosen from the suffix unless ``fmt`` given).

    Returns depth in km below sea level on the lateral model nodes, shape ``(nlat, nlon)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"surface file not found: {path}")
    kind = fmt or _SUFFIX.get(path.suffix.lower())
    if kind is None:
        raise HorizonReadError(
            f"{path}: unknown surface format '{path.suffix}'; use .xyz/.txt/.csv, .tif or .ts "
            "(or pass fmt=)"
        )
    fn = {"xyz": read_xyz, "geotiff": read_geotiff, "tsurf": read_tsurf}[kind]
    return fn(path, grid, crs=crs, z_convention=z_convention, units=units, full=full)
