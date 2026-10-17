"""Surface and base interfaces for FMTOMO (``interfaces.in``, Stage 3 task 4.3).

Interface 1 is the top (the free surface, where the velocity model ends and where ``fm3d``
refuses sources above it), interface 2 the base.  Both live on the lateral node grid of the
velocity model (cushions included, ``interfaces.in`` has one lateral grid).

Surface modes (``model3d.surface``)
-----------------------------------
``station_elevations``
    Surface elevations of the stations (``elevation_m``; borehole sensor elevations are *not*
    used, the surface above the sensor is) are interpolated linearly in a local km frame
    (``scipy.interpolate.griddata``), nodes outside the convex hull take the nearest value, then
    the field is smoothed with a Gaussian of ``surface_smoothing_km``.
``dem``
    GeoTIFF read with ``rasterio`` (optional dependency, extra ``geo``; CRS handled with
    ``pyproj``) or an ``xarray.DataArray`` of elevations (m) with coords ``lat``/``lon`` for
    tests.  Sampled bilinearly on a four times finer lateral mesh, smoothed with
    ``surface_smoothing_km`` and resampled to the interface nodes (voids: nearest fill).
``flat``
    ``surface_elevation_m`` everywhere.

Constraints enforced (``fm3d`` checks in ``3dfmlib.f90:241-242, 1575-1582`` and
``3dfm_main.f90:431-436``)
-----------------------------------------------------------------------------------------
* Every node (cushions too) must lie inside the propagation grid vertically: the surface is
  clipped to ``grid.surface_limits_km``.
* A receiver must not be above the *B-spline-interpolated* surface (``fm3d`` evaluates the
  interface with the same approximating uniform cubic B-spline as the velocity; smoothing and
  the spline both lower local peaks, so a surface station can end up above the surface).
  **Choice:** the surface is lifted locally.  The spline weights are non-negative and sum to
  one, so lowering the depth of the 16 nodes around the receiver by the violation fixes it
  exactly and never harms other receivers.  Borehole sensors are below the surface by
  construction, they are only checked.  The receiver margin is ``receiver_margin_km``.
* Sources must be below the surface: :func:`place_sources` moves events that are above it down
  to ``surface + margin_km`` (finding reported).  Events below the base are reported only.

All findings are appended to the ``findings`` list (and stored in
``InterfaceGrids.data.attrs['findings']``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, griddata

from osv.coords import FMTOMO_EARTH_RADIUS_KM
from osv.io.fmtomo.grids import InterfaceGrids, interface_dataset
from osv.model3d.grid import KM_PER_DEG, GridError, ModelGrid

log = logging.getLogger(__name__)

__all__ = [
    "InterfaceError",
    "build_interfaces",
    "interface_depth_at",
    "place_sources",
    "receiver_depths",
]

#: Receivers are kept at least this far below the surface (km); ``fm3d`` tolerates 0.1 tolerance.
RECEIVER_MARGIN_KM = 0.001
#: Oversampling of the lateral mesh used to resample a DEM.
DEM_OVERSAMPLE = 4


class InterfaceError(GridError):
    """Interfaces cannot be built (message says what to fix)."""


# ---------------------------------------------------------------------------------------
# surface construction
# ---------------------------------------------------------------------------------------


def _local_km(grid: ModelGrid, lat: Any, lon: Any) -> tuple[np.ndarray, np.ndarray]:
    la, lo = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    x = (lo - grid.centre_lon) * KM_PER_DEG * np.cos(np.deg2rad(grid.centre_lat))
    y = (la - grid.centre_lat) * KM_PER_DEG
    return x, y


def _smooth(grid: ModelGrid, field: np.ndarray, sigma_km: float) -> np.ndarray:
    if sigma_km <= 0:
        return field
    sig = (sigma_km / grid.dy_km, sigma_km / grid.dx_km)
    return ndimage.gaussian_filter(field, sigma=sig, mode="nearest")


def _fill_nearest(z: np.ndarray) -> np.ndarray:
    """Replace NaNs of the 2-D field ``z`` by the nearest valid node."""
    bad = ~np.isfinite(z)
    if not bad.any():
        return z
    if bad.all():
        raise InterfaceError("no valid elevation values to build the surface from")
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def _station_surface(grid: ModelGrid, stations: pd.DataFrame, findings: list[str]) -> np.ndarray:
    st = stations.dropna(subset=["lat", "lon", "elevation_m"])
    if len(st) == 0:
        raise InterfaceError("surface='station_elevations' needs stations with elevation_m")
    lonm, latm = np.meshgrid(grid.lon_full, grid.lat_full)
    X, Y = _local_km(grid, latm, lonm)
    xs, ys = _local_km(grid, st["lat"].to_numpy(float), st["lon"].to_numpy(float))
    z = st["elevation_m"].to_numpy(float) / 1000.0
    pts = np.column_stack([xs, ys])
    try:
        out = griddata(pts, z, (X, Y), method="linear")
    except Exception as exc:  # QhullError: < 3 points or collinear
        findings.append(
            f"station surface: linear interpolation impossible ({type(exc).__name__}); "
            "using nearest-station elevations"
        )
        out = np.full(X.shape, np.nan)
    if np.isnan(out).any():
        near = griddata(pts, z, (X, Y), method="nearest")
        out = np.where(np.isnan(out), near, out)
    return out  # km above sea level


def _dem_surface(grid: ModelGrid, dem: Any, findings: list[str]) -> np.ndarray:
    """DEM elevations (m) on the fine lateral mesh (``DEM_OVERSAMPLE`` x the node grid)."""
    k = DEM_OVERSAMPLE
    fla = np.linspace(grid.lat_full[0], grid.lat_full[-1], (grid.lat_full.size - 1) * k + 1)
    flo = np.linspace(grid.lon_full[0], grid.lon_full[-1], (grid.lon_full.size - 1) * k + 1)
    if isinstance(dem, xr.DataArray):
        da = dem.sortby("lat").sortby("lon")
        interp = RegularGridInterpolator(
            (np.asarray(da["lat"], dtype=float), np.asarray(da["lon"], dtype=float)),
            np.asarray(da.to_numpy(), dtype=float),
            bounds_error=False,
            fill_value=np.nan,
        )
        LA, LO = np.meshgrid(fla, flo, indexing="ij")
        fine = interp(np.column_stack([LA.ravel(), LO.ravel()])).reshape(LA.shape)
    else:
        fine = _sample_geotiff(Path(dem), fla, flo, findings)
    if np.isnan(fine).all():
        raise InterfaceError("the DEM does not cover the model grid")
    nan_frac = float(np.isnan(fine).mean())
    if nan_frac > 0:
        findings.append(
            f"DEM: {100 * nan_frac:.1f} % of the grid is outside the DEM or nodata; "
            "filled with the nearest valid elevation"
        )
    return _fill_nearest(fine)


def _sample_geotiff(
    path: Path, fla: np.ndarray, flo: np.ndarray, findings: list[str]
) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise InterfaceError(
            "reading a GeoTIFF DEM needs rasterio: install the 'geo' extra "
            "(uv pip install 'openseisvelocity[geo]')"
        ) from exc
    if not path.exists():
        raise InterfaceError(f"DEM file not found: {path}")
    LA, LO = np.meshgrid(fla, flo, indexing="ij")
    with rasterio.open(path) as src:
        if src.crs is not None and not src.crs.is_geographic:
            from pyproj import Transformer

            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xx, yy = tr.transform(LO.ravel(), LA.ravel())
        else:
            xx, yy = LO.ravel(), LA.ravel()
        data = src.read(1, masked=True).astype(float).filled(np.nan)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        inv = ~src.transform
        col, row = inv * (np.asarray(xx), np.asarray(yy))
        # pixel centres are at +0.5
        col, row = col - 0.5, row - 0.5
        vals = ndimage.map_coordinates(
            np.nan_to_num(data, nan=0.0), [row, col], order=1, mode="constant", cval=np.nan
        )
        valid = ndimage.map_coordinates(
            (~np.isnan(data)).astype(float), [row, col], order=1, mode="constant", cval=0.0
        )
        vals = np.where(valid > 0.999, vals, np.nan)
    findings.append(f"DEM {path.name}: bilinear sampling, CRS {src.crs}")
    return vals.reshape(LA.shape)


def _surface_from_dem(
    grid: ModelGrid, dem: Any, sigma_km: float, findings: list[str]
) -> np.ndarray:
    k = DEM_OVERSAMPLE
    fine = _dem_surface(grid, dem, findings)
    if sigma_km > 0:
        sig = (sigma_km / (grid.dy_km / k), sigma_km / (grid.dx_km / k))
        fine = ndimage.gaussian_filter(fine, sigma=sig, mode="nearest")
    return fine[::k, ::k] / 1000.0  # km above sea level


# ---------------------------------------------------------------------------------------
# B-spline evaluation of the interface (as fm3d)
# ---------------------------------------------------------------------------------------


def _bspline_weights(u: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            (1 - u) ** 3 / 6,
            (4 - 6 * u**2 + 3 * u**3) / 6,
            (1 + 3 * u + 3 * u**2 - 3 * u**3) / 6,
            u**3 / 6,
        ]
    )


def _spline_index(
    grid: ModelGrid, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Node indices (0-based, lower-left of the 4x4 stencil + 1) and fractions."""
    ilat = np.floor((lat - grid.lat_full[0]) / grid.dlat_deg).astype(int)
    ilon = np.floor((lon - grid.lon_full[0]) / grid.dlon_deg).astype(int)
    nlat, nlon = grid.lat_full.size, grid.lon_full.size
    if ((ilat < 1) | (ilat > nlat - 3) | (ilon < 1) | (ilon > nlon - 3)).any():
        raise InterfaceError("points lie outside the interior of the interface grid")
    u = (lat - grid.lat_full[ilat]) / grid.dlat_deg
    v = (lon - grid.lon_full[ilon]) / grid.dlon_deg
    return ilat, ilon, u, v


def _eval_spline(
    grid: ModelGrid, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    ilat, ilon, u, v = _spline_index(grid, lat, lon)
    bu, bv = _bspline_weights(u), _bspline_weights(v)
    out = np.zeros(lat.shape)
    for i in range(4):
        for j in range(4):
            out += bu[i] * bv[j] * depth[ilat + i - 1, ilon + j - 1]
    return out


def interface_depth_at(
    interfaces: InterfaceGrids, grid: ModelGrid, lat: Any, lon: Any, which: int = 0
) -> np.ndarray:
    """Depth (km below sea level) of interface ``which`` (0 = surface) as ``fm3d`` sees it."""
    la = np.atleast_1d(np.asarray(lat, dtype=float))
    lo = np.atleast_1d(np.asarray(lon, dtype=float))
    d = interfaces.data["interface_depth"].isel(interface=which).to_numpy()
    return _eval_spline(grid, d, la, lo)


def _make_interfaces(
    grid: ModelGrid, top_depth: np.ndarray, base_depth: float, findings: list[str]
) -> InterfaceGrids:
    rad = np.empty((2, grid.lat_full.size, grid.lon_full.size))
    rad[0] = FMTOMO_EARTH_RADIUS_KM - top_depth
    rad[1] = FMTOMO_EARTH_RADIUS_KM - base_depth
    ds = interface_dataset(
        rad,
        lat0_rad=float(np.deg2rad(grid.lat_full[0])),
        dlat_rad=float(np.deg2rad(grid.dlat_deg)),
        lon0_rad=float(np.deg2rad(grid.lon_full[0])),
        dlon_rad=float(np.deg2rad(grid.dlon_deg)),
        n_cushion=grid.n_cushion,
    )
    ds.attrs["findings"] = "\n".join(findings)
    return InterfaceGrids(ds)


# ---------------------------------------------------------------------------------------
# receivers / sources
# ---------------------------------------------------------------------------------------


def receiver_depths(stations: pd.DataFrame) -> np.ndarray:
    """Receiver depth (km below sea level) = ``-sensor_elevation_m / 1000`` (borehole aware)."""
    col = "sensor_elevation_m" if "sensor_elevation_m" in stations else "elevation_m"
    return -stations[col].to_numpy(float) / 1000.0


def _lift_for_receivers(
    grid: ModelGrid,
    top: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    rdepth: np.ndarray,
    names: list[str],
    margin: float,
    findings: list[str],
    limit: float,
) -> np.ndarray:
    top = top.copy()
    ilat, ilon, _, _ = _spline_index(grid, lat, lon)
    lifted: dict[int, float] = {}
    for _ in range(3):  # sequential passes: a later lift can only help earlier receivers
        changed = False
        for n in range(lat.size):
            s = _eval_spline(grid, top, lat[n : n + 1], lon[n : n + 1])[0]
            viol = s - (rdepth[n] - margin)
            if viol > 1e-9:
                top[ilat[n] - 1 : ilat[n] + 3, ilon[n] - 1 : ilon[n] + 3] -= viol + 1e-6
                lifted[n] = lifted.get(n, 0.0) + float(viol)
                changed = True
        top = np.maximum(top, limit)
        if not changed:
            break
    if lifted:
        worst = max(lifted.values())
        findings.append(
            f"{len(lifted)} receiver(s) were above the interpolated surface; surface lifted "
            f"locally around them (max {1000 * worst:.1f} m): "
            + ", ".join(f"{names[n]}" for n in sorted(lifted)[:8])
        )
    return top


def _receiver_arrays(
    grid: ModelGrid, stations: pd.DataFrame, findings: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    st = stations[stations["used"].astype(bool)] if "used" in stations else stations
    st = st.dropna(subset=["lat", "lon"])
    lat, lon = st["lat"].to_numpy(float), st["lon"].to_numpy(float)
    ok = grid.contains(lat, lon, np.zeros(len(st)), region="propgrid")
    if not ok.all():
        findings.append(
            f"{int((~ok).sum())} station(s) outside the propagation grid laterally are ignored "
            "for the surface constraint"
        )
    names = (st["sta"].astype(str) if "sta" in st else pd.Series(st.index.astype(str))).tolist()
    return (
        lat[ok],
        lon[ok],
        receiver_depths(st)[ok],
        [n for n, k in zip(names, ok, strict=True) if k],
    )


def build_interfaces(
    grid: ModelGrid,
    stations: pd.DataFrame,
    cfg: Any,
    dem: Any = None,
    *,
    findings: list[str] | None = None,
    receiver_margin_km: float = RECEIVER_MARGIN_KM,
    base_depth_km: float | None = None,
) -> InterfaceGrids:
    """Build the surface (interface 1) and flat base (interface 2) of ``interfaces.in``.

    Parameters
    ----------
    grid
        From :func:`osv.model3d.grid.define_grid`.
    stations
        ``lat``, ``lon``, ``elevation_m`` (surface elevation), optionally ``sensor_elevation_m``,
        ``sta`` and ``used``.
    cfg
        ``Model3DConfig`` (or ``OSVConfig``): ``surface`` (``dem`` | ``flat`` |
        ``station_elevations``), ``surface_elevation_m`` and ``surface_smoothing_km``.
    dem
        Path to a GeoTIFF (``inputs.dem``) or an ``xarray.DataArray`` (``lat``, ``lon``; m).
    findings
        Optional list to which human-readable findings are appended.
    receiver_margin_km
        Minimum depth of every receiver below the interpolated surface.
    base_depth_km
        Depth of the flat base; default the lower limit inside the propagation grid
        (``grid.surface_limits_km[1]``), the deepest position ``fm3d`` accepts.

    Returns
    -------
    InterfaceGrids
        Two interfaces on the lateral velocity-grid nodes (cushions included).  The findings are
        also stored in ``.data.attrs['findings']``.
    """
    m3 = getattr(cfg, "model3d", cfg)
    fnd: list[str] = [] if findings is None else findings
    start = len(fnd)
    lo, hi = grid.surface_limits_km
    mode = m3.surface
    sigma = float(m3.surface_smoothing_km)
    shape = (grid.lat_full.size, grid.lon_full.size)

    if mode == "flat":
        elev = np.full(shape, float(m3.surface_elevation_m) / 1000.0)
    elif mode == "station_elevations":
        elev = _smooth(grid, _station_surface(grid, stations, fnd), sigma)
    elif mode == "dem":
        if dem is None:
            raise InterfaceError(
                "surface='dem' needs a DEM: set inputs.dem to a GeoTIFF (or pass dem=...)"
            )
        elev = _surface_from_dem(grid, dem, sigma, fnd)
    else:
        raise InterfaceError(f"unknown surface mode {mode!r}")
    top = -elev  # depth, km

    if (top < lo).any():
        fnd.append(
            f"surface clipped to the top of the propagation grid ({lo:g} km) at "
            f"{int((top < lo).sum())} node(s); increase grid.top_margin_km"
        )
        top = np.maximum(top, lo)

    base = hi if base_depth_km is None else float(base_depth_km)
    if not (lo < base <= hi + 1e-9):
        raise InterfaceError(f"base depth {base:g} km must lie in ({lo:g}, {hi:g}] km")

    lat, lon, rdep, names = _receiver_arrays(grid, stations, fnd)
    if lat.size:
        top = _lift_for_receivers(grid, top, lat, lon, rdep, names, receiver_margin_km, fnd, lo)
        s = _eval_spline(grid, top, lat, lon)
        still = s > rdep - receiver_margin_km + 1e-6
        if still.any():
            fnd.append(
                f"{int(still.sum())} receiver(s) remain above the surface after clipping "
                f"(deepest allowed surface is {lo:g} km): "
                + ", ".join(n for n, k in zip(names, still, strict=True) if k)
            )
    if (top >= base).any():
        raise InterfaceError(
            f"the surface reaches the base interface ({base:g} km); increase grid.zmax_km"
        )
    surf_elev = -top
    fnd.append(
        f"surface ({mode}): {1000 * surf_elev.min():.0f} to {1000 * surf_elev.max():.0f} m "
        f"elevation; base flat at {base:g} km depth"
    )
    out = _make_interfaces(grid, top, base, fnd)
    for msg in fnd[start:]:
        log.info("interfaces: %s", msg)
    return out


def place_sources(
    interfaces: InterfaceGrids,
    grid: ModelGrid,
    events: pd.DataFrame,
    *,
    margin_km: float = 0.01,
    findings: list[str] | None = None,
) -> pd.DataFrame:
    """Return ``events`` with sources moved below the surface (``depth_km_bsl``).

    Events above ``surface + margin_km`` are moved down to that depth; events laterally outside
    the propagation grid or below the base are reported but unchanged.  A column ``moved_km``
    holds the shift.
    """
    fnd: list[str] = [] if findings is None else findings
    col = "depth_km_bsl" if "depth_km_bsl" in events else "depth_km"
    ev = events.copy()
    z = ev[col].to_numpy(float)
    lat, lon = ev["lat"].to_numpy(float), ev["lon"].to_numpy(float)
    inside = grid.contains(lat, lon, np.zeros(len(ev)), region="propgrid")
    if not inside.all():
        fnd.append(f"{int((~inside).sum())} event(s) outside the propagation grid laterally")
    surf = np.full(len(ev), np.nan)
    base = np.full(len(ev), np.nan)
    if inside.any():
        surf[inside] = interface_depth_at(interfaces, grid, lat[inside], lon[inside], 0)
        base[inside] = interface_depth_at(interfaces, grid, lat[inside], lon[inside], 1)
    need = np.where(inside, surf + margin_km, -np.inf)
    moved = np.where(z < need, need - z, 0.0)
    if (moved > 0).any():
        idx = np.flatnonzero(moved > 0)
        fnd.append(
            f"{idx.size} event(s) above the surface moved down (max {moved.max():.3f} km): "
            + ", ".join(
                str(ev["event_id"].iloc[i]) if "event_id" in ev else str(ev.index[i])
                for i in idx[:8]
            )
        )
    ev[col] = z + moved
    ev["moved_km"] = moved
    deep = inside & (ev[col].to_numpy(float) > base)
    if deep.any():
        fnd.append(f"{int(deep.sum())} event(s) below the base interface (not moved)")
    for msg in fnd:
        log.info("sources: %s", msg)
    return ev
