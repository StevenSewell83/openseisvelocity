"""Import of a user-supplied 3-D velocity grid (NetCDF or CSV) onto the model grid (task 4.4).

Supported inputs
----------------
* **NetCDF** (xarray): coordinates ``lat``/``lon``/``depth`` or ``x``/``y``/``z`` (names can
  be changed with ``columns``), variables ``vp`` and optionally ``vs``.  For ``x``/``y`` the
  CRS comes from ``crs`` or the file attribute ``crs`` / ``crs_wkt``.  The data must lie on a
  regular (possibly non-uniform) axis grid; it is interpolated trilinearly.
* **CSV**: columns ``x, y, z, vp[, vs]`` (names configurable).  Complete regular lattices are
  detected and interpolated trilinearly; scattered points use Delaunay linear interpolation,
  restricted to nodes within ``max_distance_km`` of a data point.

``z`` is depth (positive down) or elevation (positive up) according to ``z_convention``, in
``units`` (``"m"``/``"km"``).  Velocities are km/s; values that look like m/s (median above
100) are converted with a warning.  Output is on the *full* grid (including cushions) with
NaN where the file has no data; the coverage mask is True where Vp is defined.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree

from osv.model3d.grid import KM_PER_DEG, ModelGrid
from osv.model3d.horizons_io import WGS84, nodes_in_crs, z_to_depth_km

log = logging.getLogger(__name__)

__all__ = ["UserGridError", "load_user_grid"]

DEFAULT_COLUMNS = {"x": "x", "y": "y", "z": "z", "vp": "vp", "vs": "vs"}
_NC_GEOGRAPHIC = {"x": "lon", "y": "lat", "z": "depth"}
#: Scattered data: nodes farther than this many median nearest-neighbour spacings are masked.
SCATTER_MAX_DISTANCE_FACTOR = 3.0


class UserGridError(ValueError):
    """The user grid file cannot be interpreted (message says what to fix)."""


def _velocity_km_s(v: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
    if med > 100.0:
        log.warning("%s: median %.0f looks like m/s; converting to km/s", name, med)
        v = v / 1000.0
    return v


def _regular(
    xs: np.ndarray, ys: np.ndarray, zs_km: np.ndarray, vals: np.ndarray, grid: ModelGrid, crs: Any
) -> np.ndarray:
    """Trilinear interpolation of a ``(nz, ny, nx)`` array onto the model nodes (NaN outside)."""
    for a, nm in ((xs, "x"), (ys, "y"), (zs_km, "z")):
        if a.size < 2:
            raise UserGridError(f"axis {nm} has fewer than 2 values; a 3-D grid is required")
    out = np.full(grid.shape_full, np.nan)
    X, Y = nodes_in_crs(grid.lat_full, grid.lon_full, crs)
    pts_xy = np.column_stack([Y.ravel(), X.ravel()])
    f = RegularGridInterpolator((zs_km, ys, xs), vals, bounds_error=False, fill_value=np.nan)
    for k, d in enumerate(grid.depth_full):
        pts = np.column_stack([np.full(len(pts_xy), d), pts_xy])
        out[k] = f(pts).reshape(X.shape)
    return out


def _sorted_axis(a: np.ndarray, vals: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(a)
    return a[order], np.take(vals, order, axis=axis)


def _local_km(lat: np.ndarray, lon: np.ndarray, depth: np.ndarray, grid: ModelGrid) -> np.ndarray:
    cl = np.cos(np.deg2rad(grid.centre_lat))
    return np.column_stack(
        [
            (np.asarray(lon) - grid.centre_lon) * KM_PER_DEG * cl,
            (np.asarray(lat) - grid.centre_lat) * KM_PER_DEG,
            np.asarray(depth),
        ]
    )


def _scattered(
    x: np.ndarray,
    y: np.ndarray,
    z_km: np.ndarray,
    values: dict[str, np.ndarray],
    grid: ModelGrid,
    crs: Any,
    max_distance_km: float | None,
) -> dict[str, np.ndarray]:
    if crs is not None and CRS.from_user_input(crs) != WGS84:
        tr = Transformer.from_crs(CRS.from_user_input(crs), WGS84, always_xy=True)
        lon, lat = tr.transform(x, y)
    else:
        lon, lat = x, y
    pts = _local_km(lat, lon, z_km, grid)
    zz, la2, lo2 = np.meshgrid(grid.depth_full, grid.lat_full, grid.lon_full, indexing="ij")
    tgt = _local_km(la2.ravel(), lo2.ravel(), zz.ravel(), grid)
    tree = cKDTree(pts)
    if max_distance_km is None:
        nn = tree.query(pts, k=2)[0][:, 1]
        nn = nn[nn > 0]
        max_distance_km = SCATTER_MAX_DISTANCE_FACTOR * float(np.median(nn)) if nn.size else 0.0
    dist = tree.query(tgt)[0]
    far = dist > max_distance_km
    out = {}
    for name, v in values.items():
        try:
            f = LinearNDInterpolator(pts, v)
        except Exception as exc:
            raise UserGridError(f"cannot triangulate scattered points ({exc})") from exc
        r = f(tgt)
        r[far] = np.nan
        out[name] = r.reshape(grid.shape_full)
    return out


def _load_csv(path: Path, cols: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    df.columns = [str(c).strip() for c in df.columns]
    need = [cols["x"], cols["y"], cols["z"], cols["vp"]]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise UserGridError(
            f"{path}: missing column(s) {missing}; found {list(df.columns)}. "
            "Set model3d.user_model.grid_columns to map the names."
        )
    return df


def _regular_lattice(
    df: pd.DataFrame, cols: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    """Return axes and ``(nz, ny, nx)`` arrays if the rows form a complete lattice."""
    ux = np.unique(df[cols["x"]])
    uy = np.unique(df[cols["y"]])
    uz = np.unique(df[cols["z"]])
    if ux.size * uy.size * uz.size != len(df) or min(ux.size, uy.size, uz.size) < 2:
        return None
    ix = np.searchsorted(ux, df[cols["x"]].to_numpy())
    iy = np.searchsorted(uy, df[cols["y"]].to_numpy())
    iz = np.searchsorted(uz, df[cols["z"]].to_numpy())
    flat = (iz * uy.size + iy) * ux.size + ix
    if np.unique(flat).size != len(df):
        return None
    arrs = {}
    for key in ("vp", "vs"):
        if cols[key] in df.columns:
            a = np.full((uz.size, uy.size, ux.size), np.nan)
            a[iz, iy, ix] = df[cols[key]].to_numpy(float)
            arrs[key] = a
    return ux, uy, uz, arrs


def load_user_grid(
    path: str | Path,
    grid: ModelGrid,
    crs: Any | None = None,
    columns: dict[str, str] | None = None,
    *,
    z_convention: Literal["depth", "elevation"] = "depth",
    units: str = "km",
    max_distance_km: float | None = None,
) -> tuple[xr.Dataset, xr.DataArray]:
    """Read a NetCDF/CSV velocity grid and resample it onto ``grid`` (full node set).

    Parameters
    ----------
    path
        ``.nc``/``.nc4``/``.cdf`` (NetCDF) or any other suffix (read as CSV).
    grid
        Target :class:`~osv.model3d.grid.ModelGrid`.
    crs
        CRS of the horizontal coordinates (e.g. ``"EPSG:2193"``); None = lon/lat, or for
        NetCDF the ``crs``/``crs_wkt`` attribute when present.
    columns
        Names of ``x``, ``y``, ``z``, ``vp``, ``vs`` columns/variables (partial dict allowed).
        NetCDF defaults: coordinates ``lon``, ``lat``, ``depth`` if present, else ``x``, ``y``,
        ``z``.
    z_convention, units
        Meaning and unit of the z coordinate (``depth``/``elevation``; ``m``/``km``).
    max_distance_km
        Scattered input only: mask nodes farther than this from any data point (default:
        3 x the median nearest-neighbour spacing).

    Returns
    -------
    (xarray.Dataset, xarray.DataArray)
        Dataset with ``vp`` (and ``vs`` when available), dims ``(depth, lat, lon)``, NaN
        where the file has no data; and the boolean coverage mask (Vp finite).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"user grid file not found: {path}")
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    if path.suffix.lower() in (".nc", ".nc4", ".cdf", ".netcdf"):
        result = _from_netcdf(path, grid, crs, columns, z_convention, units)
    else:
        result = _from_csv(path, grid, crs, cols, z_convention, units, max_distance_km)

    coords = {
        "depth": ("depth", grid.depth_full, {"units": "km", "positive": "down"}),
        "lat": ("lat", grid.lat_full, {"units": "degrees_north"}),
        "lon": ("lon", grid.lon_full, {"units": "degrees_east"}),
    }
    dims = ("depth", "lat", "lon")
    ds = xr.Dataset(
        {k: (dims, _velocity_km_s(v, k), {"units": "km/s"}) for k, v in result.items()},
        coords=coords,
        attrs={"source": str(path)},
    )
    coverage = xr.DataArray(
        np.isfinite(ds["vp"].values), dims=dims, coords=ds.coords, name="coverage"
    )
    frac = float(coverage.mean())
    if frac == 0.0:
        raise UserGridError(
            f"{path}: the grid does not overlap the model volume; check crs, units and z_convention"
        )
    log.info("user grid %s covers %.1f%% of the model nodes", path.name, 100 * frac)
    return ds, coverage


def _from_netcdf(
    path: Path,
    grid: ModelGrid,
    crs: Any,
    columns: dict[str, str] | None,
    z_convention: str,
    units: str,
) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    cols = dict(columns or {})
    for k, v in DEFAULT_COLUMNS.items():
        cols.setdefault(k, v)
    if "x" not in (columns or {}) and all(
        c in ds.coords or c in ds.dims for c in _NC_GEOGRAPHIC.values()
    ):
        cols.update(_NC_GEOGRAPHIC)
        file_crs = None
    else:
        file_crs = ds.attrs.get("crs") or ds.attrs.get("crs_wkt")
        if crs is None and file_crs is not None:
            crs = file_crs
        elif crs is None and cols["x"] != "lon":
            log.warning("%s: no CRS given for x/y coordinates; assuming lon/lat", path.name)
    for key in ("x", "y", "z", "vp"):
        if cols[key] not in ds.variables:
            raise UserGridError(
                f"{path}: variable '{cols[key]}' not found; has {list(ds.variables)}. "
                "Set grid_columns to map the names."
            )
    da = ds[cols["vp"]]
    if da.ndim != 3:
        raise UserGridError(f"{path}: '{cols['vp']}' must be 3-D (z, y, x), has dims {da.dims}")
    order = (cols["z"], cols["y"], cols["x"])
    if set(da.dims) != set(order):
        raise UserGridError(f"{path}: '{cols['vp']}' dims {da.dims} do not match {order}")
    out = {}
    for key in ("vp", "vs"):
        if cols[key] in ds.variables:
            out[key] = ds[cols[key]].transpose(*order).values.astype(float)
    xs, ys = ds[cols["x"]].values.astype(float), ds[cols["y"]].values.astype(float)
    zs = z_to_depth_km(ds[cols["z"]].values, z_convention, units)
    # sort every axis ascending
    xs, out = xs.copy(), dict(out)
    for key in out:
        v = out[key]
        zs_s, v = _sorted_axis(zs, v, 0)
        ys_s, v = _sorted_axis(ys, v, 1)
        xs_s, v = _sorted_axis(xs, v, 2)
        out[key] = v
    return {k: _regular(xs_s, ys_s, zs_s, v, grid, crs) for k, v in out.items()}


def _from_csv(
    path: Path,
    grid: ModelGrid,
    crs: Any,
    cols: dict[str, str],
    z_convention: str,
    units: str,
    max_distance_km: float | None,
) -> dict[str, np.ndarray]:
    df = _load_csv(path, cols)
    df = df.dropna(subset=[cols["x"], cols["y"], cols["z"], cols["vp"]])
    df = df.assign(**{cols["z"]: z_to_depth_km(df[cols["z"]].to_numpy(float), z_convention, units)})
    lat = _regular_lattice(df, cols)
    if lat is not None:
        ux, uy, uz, arrs = lat
        return {k: _regular(ux, uy, uz, a, grid, crs) for k, a in arrs.items()}
    values = {"vp": df[cols["vp"]].to_numpy(float)}
    if cols["vs"] in df.columns:
        values["vs"] = df[cols["vs"]].to_numpy(float)
    return _scattered(
        df[cols["x"]].to_numpy(float),
        df[cols["y"]].to_numpy(float),
        df[cols["z"]].to_numpy(float),
        values,
        grid,
        crs,
        max_distance_km,
    )
