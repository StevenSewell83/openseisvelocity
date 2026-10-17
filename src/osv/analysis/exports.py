"""Export final models, masks and hypocentres: NetCDF, VTK (ParaView) and CSV (task 6.4).

:func:`export_model` writes, per requested format:

``netcdf``
    ``model.nc`` - the dataset (``depth``, ``lat``, ``lon``; ``vp``, ``vs``, ``vpvs``, ``dvp_pct``,
    ``dvs_pct`` and the masks) with CF-style attributes (units, ``long_name``, ``positive="down"``
    for depth, ``Conventions``).  Boolean masks are stored as ``int8`` (``flag_values``).
``vtk``
    ``model.vtk`` (legacy ASCII ``STRUCTURED_GRID``; or ``model.vts`` when ``pyvista`` is
    installed and ``vtk_backend`` allows) in **local km coordinates**: x east, y north, z up
    (``z = -depth``), origin at the grid centre.  Point data: ``vp``, ``vs``, ``vpvs``,
    ``dvp_pct``, ``dvs_pct`` and every mask.  NaN values are written as :data:`VTK_NODATA` in the
    legacy file (the VTK ASCII reader does not read ``nan`` reliably).  Hypocentres and stations
    go to ``hypocentres.vtk`` / ``stations.vtk`` (``POLYDATA`` vertices with their numeric
    columns as point data).
``csv``
    ``model.csv`` - one row per grid node: ``lat, lon, depth_km, x_km, y_km, z_km`` and the same
    variables.  ``hypocentres.csv``/``stations.csv`` repeat the tables with local coordinates.

Percentage perturbations are relative to the reference ``vp_ref``/``vs_ref`` (or ``vp_bg``/
``vs_bg``) of the dataset if present, else the laterally averaged 1-D profile of the model.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from osv.model3d.grid import KM_PER_DEG

log = logging.getLogger(__name__)

__all__ = ["VTK_NODATA", "export_model", "local_km", "model_variables", "write_vtk_points"]

#: Value standing in for NaN in legacy VTK files.
VTK_NODATA = -9999.0
_DIMS = ("depth", "lat", "lon")
_FORMATS = ("netcdf", "vtk", "csv")


def local_km(lat: Any, lon: Any, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """East/north km of ``lat``/``lon`` relative to ``(lat0, lon0)`` (FMTOMO sphere)."""
    y = (np.asarray(lat, float) - lat0) * KM_PER_DEG
    x = (np.asarray(lon, float) - lon0) * KM_PER_DEG * np.cos(np.deg2rad(lat0))
    return x, y


def _centre(ds: xr.Dataset) -> tuple[float, float]:
    lat0 = float(ds.attrs.get("centre_lat", ds["lat"].mean()))
    lon0 = float(ds.attrs.get("centre_lon", ds["lon"].mean()))
    return lat0, lon0


def _reference(ds: xr.Dataset, name: str) -> xr.DataArray:
    for ref in (f"{name}_ref", f"{name}_bg"):
        if ref in ds:
            return ds[ref]
    return ds[name].mean(("lat", "lon"), skipna=True)  # 1-D profile per depth


def model_variables(
    model_ds: xr.Dataset, masks: Mapping[str, xr.DataArray] | None = None
) -> xr.Dataset:
    """Dataset ``(depth, lat, lon)`` with vp, vs, vpvs, dvp_pct, dvs_pct and the masks.

    Parameters
    ----------
    model_ds
        Needs ``vp`` and/or ``vs`` (km/s) with dims ``depth``, ``lat``, ``lon``.
    masks
        Optional ``name -> DataArray`` (bool or numeric) on the same grid (e.g. ray-hit or
        resolution masks); added as variables.
    """
    if not any(v in model_ds for v in ("vp", "vs")):
        raise ValueError("model dataset needs a 'vp' and/or 'vs' variable")
    for d in _DIMS:
        if d not in model_ds.dims:
            raise ValueError(f"model dataset lacks dimension {d!r} (has {tuple(model_ds.dims)})")
    out = xr.Dataset(coords={d: model_ds[d] for d in _DIMS}, attrs=dict(model_ds.attrs))
    for v in ("vp", "vs"):
        if v in model_ds:
            out[v] = model_ds[v].transpose(*_DIMS).astype("float64")
            out[v].attrs.update(units="km/s", long_name=f"{v.upper()[0]}{v[1]} velocity")
            ref = _reference(model_ds, v)
            pct = 100.0 * (out[v] - ref) / ref
            out[f"d{v}_pct"] = pct.transpose(*_DIMS)
            out[f"d{v}_pct"].attrs.update(units="percent", long_name=f"{v} perturbation")
    if "vp" in out and "vs" in out:
        out["vpvs"] = out["vp"] / out["vs"]
        out["vpvs"].attrs.update(units="1", long_name="Vp/Vs ratio")
    ref_shape = tuple(out.sizes[d] for d in _DIMS)
    for name, m in (masks or {}).items():
        m = xr.DataArray(m) if not isinstance(m, xr.DataArray) else m
        if not set(_DIMS) <= set(m.dims):
            raise ValueError(f"mask {name!r} needs dims {_DIMS}, has {m.dims}")
        m = m.transpose(*_DIMS)
        if m.shape != ref_shape:
            raise ValueError(f"mask {name!r} has shape {m.shape}, the model {ref_shape}")
        # same shape: take the values on the model's own coordinates
        m = xr.DataArray(m.values, coords=out.coords, dims=_DIMS)
        out[name] = m.astype("float32") if m.dtype == bool else m
        out[name].attrs.update(long_name=f"mask {name}", units="1")
        out[name].attrs.pop("flag_values", None)
    return out


def _write_netcdf(ds: xr.Dataset, path: Path, masks: set[str]) -> Path:
    out = ds.copy()
    for m in masks:
        out[m] = out[m].fillna(0).astype("int8")
        out[m].attrs.update(
            long_name=f"mask {m}",
            flag_values=np.array([0, 1], "int8"),
            flag_meanings="excluded included",
            units="1",
        )
    out["depth"].attrs.update(
        units="km", long_name="depth below sea level", positive="down", axis="Z"
    )
    out["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    out["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    out.attrs.update(
        Conventions="CF-1.8",
        title="openSeisVelocity model export",
        source="openSeisVelocity (FMTOMO inversion)",
    )
    for k in list(out.attrs):  # netCDF attributes cannot hold arbitrary objects
        if not isinstance(out.attrs[k], (str, int, float, np.number, np.ndarray, list, tuple)):
            out.attrs[k] = str(out.attrs[k])
    out.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------------------
# VTK
# ---------------------------------------------------------------------------------------
def _fmt(a: np.ndarray, per_line: int = 6) -> str:
    a = np.where(np.isfinite(a), a, VTK_NODATA)
    lines = [" ".join(f"{v:.6g}" for v in a[i : i + per_line]) for i in range(0, len(a), per_line)]
    return "\n".join(lines)


def _vtk_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in str(name))


def _grid_points(ds: xr.Dataset) -> tuple[np.ndarray, tuple[int, int, int]]:
    lat0, lon0 = _centre(ds)
    z = -ds["depth"].to_numpy().astype(float)
    x, y = local_km(
        ds["lat"].to_numpy().astype(float), ds["lon"].to_numpy().astype(float), lat0, lon0
    )
    # x (lon) fastest, then y (lat), then z; but x depends on lat via the cos(lat0) only
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])
    return pts, (len(x), len(y), len(z))


def _vtk_scalars(ds: xr.Dataset, names: list[str]) -> dict[str, np.ndarray]:
    # (depth, lat, lon) -> (lon, lat, depth) flattened with lon fastest == order "F"
    return {
        n: np.asarray(ds[n].transpose("lon", "lat", "depth").values, float).ravel(order="F")
        for n in names
    }


def _write_vtk_structured(ds: xr.Dataset, path: Path, names: list[str], backend: str) -> Path:
    pts, dims = _grid_points(ds)
    data = _vtk_scalars(ds, names)
    if backend != "legacy":
        try:
            import pyvista as pv  # type: ignore[import-not-found]
        except ImportError:
            if backend == "pyvista":
                raise
        else:
            grid = pv.StructuredGrid()
            grid.points = pts
            grid.dimensions = dims
            for n, v in data.items():
                grid.point_data[_vtk_name(n)] = v
            out = path.with_suffix(".vts")
            grid.save(out)
            return out
    n = len(pts)
    parts = [
        "# vtk DataFile Version 3.0",
        "openSeisVelocity model (x east km, y north km, z up km)",
        "ASCII",
        "DATASET STRUCTURED_GRID",
        f"DIMENSIONS {dims[0]} {dims[1]} {dims[2]}",
        f"POINTS {n} float",
        _fmt(pts.ravel(), 3),
        f"POINT_DATA {n}",
    ]
    for name, v in data.items():
        parts += [f"SCALARS {_vtk_name(name)} float 1", "LOOKUP_TABLE default", _fmt(v)]
    path.write_text("\n".join(parts) + "\n")
    return path


def write_vtk_points(
    path: str | Path, xyz: np.ndarray, data: Mapping[str, np.ndarray] | None = None
) -> Path:
    """Write a legacy ASCII VTK ``POLYDATA`` point cloud (vertices) with scalar point data."""
    p = Path(path)
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    n = len(xyz)
    parts = [
        "# vtk DataFile Version 3.0",
        "openSeisVelocity points (x east km, y north km, z up km)",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {n} float",
        _fmt(xyz.ravel(), 3),
        f"VERTICES {n} {2 * n}",
        *[f"1 {i}" for i in range(n)],
    ]
    if data:
        parts.append(f"POINT_DATA {n}")
        for name, v in data.items():
            parts += [
                f"SCALARS {_vtk_name(name)} float 1",
                "LOOKUP_TABLE default",
                _fmt(np.asarray(v, float)),
            ]
    p.write_text("\n".join(parts) + "\n")
    return p


def _points_table(
    df: pd.DataFrame, lat0: float, lon0: float, depth_col: str | None
) -> pd.DataFrame:
    d = df.copy()
    if depth_col is None:
        depth_col = next((c for c in ("depth_km", "depth_km_bsl", "depth") if c in d), None)
    if depth_col is None:
        raise ValueError("points table needs a depth column (depth_km)")
    x, y = local_km(d["lat"], d["lon"], lat0, lon0)
    d["x_km"], d["y_km"] = x, y
    d["z_km"] = -d[depth_col].to_numpy().astype(float)
    d["depth_km"] = d[depth_col].to_numpy().astype(float)
    return d


def _write_points(df: pd.DataFrame, path: Path) -> Path:
    num = [
        c
        for c in df.columns
        if c not in ("x_km", "y_km", "z_km") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return write_vtk_points(
        path,
        df[["x_km", "y_km", "z_km"]].to_numpy().astype(float),
        {c: df[c].to_numpy(float) for c in num},
    )


def export_model(
    model_ds: xr.Dataset,
    outdir: str | Path,
    formats: tuple[str, ...] | list[str] = _FORMATS,
    masks: Mapping[str, xr.DataArray] | None = None,
    hypocentres: pd.DataFrame | None = None,
    stations: pd.DataFrame | None = None,
    *,
    vtk_backend: str = "auto",
) -> dict[str, Path]:
    """Export a model (and optionally masks, hypocentres, stations) to files.

    Parameters
    ----------
    model_ds
        Model dataset with ``vp`` and/or ``vs`` on ``(depth, lat, lon)``.
    outdir
        Output directory (created).
    formats
        Subset of ``("netcdf", "vtk", "csv")``.
    masks
        ``name -> DataArray`` masks on the model grid (written as variables).
    hypocentres, stations
        Tables with ``lat``, ``lon`` and ``depth_km`` (stations: depth, negative above sea
        level); numeric columns are kept as point data.
    vtk_backend
        ``"auto"`` (pyvista ``.vts`` if installed, else legacy ``.vtk``), ``"pyvista"`` or
        ``"legacy"``.

    Returns
    -------
    dict
        Output key (``netcdf``, ``vtk``, ``csv``, ``hypocentres_vtk``, ``hypocentres_csv``,
        ``stations_vtk``, ``stations_csv``) to written path.
    """
    bad = [f for f in formats if f not in _FORMATS]
    if bad:
        raise ValueError(f"unknown export format(s) {bad}; choose from {_FORMATS}")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ds = model_variables(model_ds, masks)
    mask_names = list(masks or {})
    scalars = [v for v in ds.data_vars]
    lat0, lon0 = _centre(ds)
    ds.attrs.update(local_origin_lat=lat0, local_origin_lon=lon0)
    files: dict[str, Path] = {}
    if "netcdf" in formats:
        files["netcdf"] = _write_netcdf(ds, out / "model.nc", set(mask_names))
    if "vtk" in formats:
        files["vtk"] = _write_vtk_structured(ds, out / "model.vtk", scalars, vtk_backend)
    if "csv" in formats:
        df = ds.to_dataframe().reset_index().rename(columns={"depth": "depth_km"})
        x, y = local_km(df["lat"], df["lon"], lat0, lon0)
        df.insert(3, "x_km", x)
        df.insert(4, "y_km", y)
        df.insert(5, "z_km", -df["depth_km"].to_numpy())
        files["csv"] = out / "model.csv"
        df.to_csv(files["csv"], index=False, float_format="%.6g")
    for key, tab in (("hypocentres", hypocentres), ("stations", stations)):
        if tab is None or not len(tab):
            continue
        pt = _points_table(tab, lat0, lon0, None)
        if "vtk" in formats:
            files[f"{key}_vtk"] = _write_points(pt, out / f"{key}.vtk")
        if "csv" in formats:
            files[f"{key}_csv"] = out / f"{key}.csv"
            pt.to_csv(files[f"{key}_csv"], index=False)
    return files
