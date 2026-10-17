"""Synthetic test models for Stage 6: checkerboard, spikes and input models.

All generators return an :class:`xarray.Dataset` on the *full* node set of the
:class:`~osv.model3d.grid.ModelGrid` (cushions included), dims ``(depth, lat, lon)`` with

``dvp_pct``
    the perturbation pattern in percent (relative to the background Vp),
``vp``
    ``background * (1 + dvp_pct / 100)`` (km/s),
``vs``
    ``vs_background * (1 + vs_scale * dvp_pct / 100)``; the default ``vs_scale=1`` preserves
    Vp/Vs exactly, ``vs_scale=0`` leaves Vs unperturbed (P-only test), other values give a
    different Vs pattern amplitude.

Pattern geometry is defined in kilometres on a local transverse-Mercator projection centred
on the grid centre (not in degrees), depth positive down.  ``phase`` shifts are in km.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from osv.coords import LocalProjection
from osv.model3d.grid import ModelGrid
from osv.model3d.user_grid import load_user_grid

log = logging.getLogger(__name__)

__all__ = [
    "UnderSamplingWarning",
    "anomaly_bodies",
    "checkerboard",
    "grid_xyz",
    "input_model",
    "pattern_summary",
    "spikes",
    "write_vgrids",
]

DEFAULT_VPVS = 1.73
DIMS = ("depth", "lat", "lon")


class UnderSamplingWarning(UserWarning):
    """The test pattern is too fine for the inversion grid's node spacing."""


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


def _projection(grid: ModelGrid, projection: LocalProjection | None) -> LocalProjection:
    return projection or LocalProjection(grid.centre_lat, grid.centre_lon)


def grid_xyz(
    grid: ModelGrid, projection: LocalProjection | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast local ``x`` (east km), ``y`` (north km) and depth ``z`` (km) on the full grid."""
    proj = _projection(grid, projection)
    la, lo = np.meshgrid(grid.lat_full, grid.lon_full, indexing="ij")
    x2, y2 = proj.forward(la, lo)
    shape = grid.shape_full
    x = np.broadcast_to(np.asarray(x2)[None], shape)
    y = np.broadcast_to(np.asarray(y2)[None], shape)
    z = np.broadcast_to(grid.depth_full[:, None, None], shape)
    return x, y, z


def _as_full(grid: ModelGrid, v: Any, name: str) -> np.ndarray:
    a = np.asarray(v.values if hasattr(v, "values") else v, dtype=float)
    if a.ndim == 0:
        return np.full(grid.shape_full, float(a))
    if a.shape == grid.shape_full:
        return a.copy()
    if a.shape == grid.shape:
        return np.pad(a, grid.n_cushion, mode="edge")
    if a.ndim == 1 and a.size == grid.depth_full.size:  # 1-D profile on the full depth axis
        return np.broadcast_to(a[:, None, None], grid.shape_full).copy()
    raise ValueError(f"{name}: shape {a.shape} does not match the grid {grid.shape_full}")


def _background(grid: ModelGrid, background: Any, vpvs: float) -> tuple[np.ndarray, np.ndarray]:
    """Background ``(vp, vs)`` on the full grid from a scalar/array/DataArray/Dataset."""
    if isinstance(background, xr.Dataset):
        vp = _as_full(grid, background["vp"], "background vp")
        vs = _as_full(grid, background["vs"], "background vs") if "vs" in background else vp / vpvs
    else:
        vp = _as_full(grid, background, "background")
        vs = vp / vpvs
    if not np.all(np.isfinite(vp)) or np.any(vp <= 0):
        raise ValueError("background Vp must be finite and positive")
    return vp, vs


def _dataset(
    grid: ModelGrid,
    dvp: np.ndarray,
    bg_vp: np.ndarray,
    bg_vs: np.ndarray,
    vs_scale: float,
    attrs: dict[str, Any],
) -> xr.Dataset:
    coords = {
        "depth": ("depth", grid.depth_full, {"units": "km", "positive": "down"}),
        "lat": ("lat", grid.lat_full, {"units": "degrees_north"}),
        "lon": ("lon", grid.lon_full, {"units": "degrees_east"}),
    }
    return xr.Dataset(
        {
            "dvp_pct": (DIMS, dvp, {"units": "percent"}),
            "vp": (DIMS, bg_vp * (1.0 + dvp / 100.0), {"units": "km/s"}),
            "vs": (DIMS, bg_vs * (1.0 + vs_scale * dvp / 100.0), {"units": "km/s"}),
            "vp_bg": (DIMS, bg_vp, {"units": "km/s"}),
            "vs_bg": (DIMS, bg_vs, {"units": "km/s"}),
        },
        coords=coords,
        attrs={"n_cushion": grid.n_cushion, "vs_scale": vs_scale, **attrs},
    )


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _region_weight(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    proj: LocalProjection,
    region: pd.DataFrame,
    taper_km: float,
) -> np.ndarray:
    """1 inside the bounding box of ``region`` (lat, lon[, depth_km]), 0 beyond ``taper_km``."""
    rx, ry = proj.forward(region["lat"].to_numpy(float), region["lon"].to_numpy(float))
    lo = [np.min(rx), np.min(ry)]
    hi = [np.max(rx), np.max(ry)]
    dx = np.maximum(np.maximum(lo[0] - x, x - hi[0]), 0.0)
    dy = np.maximum(np.maximum(lo[1] - y, y - hi[1]), 0.0)
    dist = np.hypot(dx, dy)
    for col in ("depth_km", "depth_km_bsl"):
        if col in region:
            zd = region[col].to_numpy(float)
            dist = np.hypot(dist, np.maximum(np.maximum(zd.min() - z, z - zd.max()), 0.0))
            break
    if taper_km <= 0:
        return (dist <= 0).astype(float)
    return 1.0 - _smoothstep(dist / taper_km)


def _check_sampling(grid: ModelGrid, cell_km: float, depth_cells: bool) -> list[str]:
    msgs = []
    lateral = max(grid.dx_km, grid.dy_km)
    if cell_km < 2.0 * lateral:
        msgs.append(
            f"checkerboard cell {cell_km:g} km < 2 x lateral node spacing "
            f"({lateral:.3g} km): the pattern is under-sampled by the grid"
        )
    if depth_cells and cell_km < 2.0 * grid.dz_km:
        msgs.append(
            f"checkerboard cell {cell_km:g} km < 2 x vertical node spacing "
            f"({grid.dz_km:.3g} km): the pattern is under-sampled in depth"
        )
    for m in msgs:
        log.warning(m)
        warnings.warn(m, UnderSamplingWarning, stacklevel=3)
    return msgs


# ---------------------------------------------------------------------------------------
# checkerboard
# ---------------------------------------------------------------------------------------


def checkerboard(
    grid: ModelGrid,
    background: Any,
    cell_km: float,
    amplitude_pct: float,
    offset_half_cell: bool = False,
    *,
    shape: Literal["sin", "block"] = "sin",
    depth_cells: bool = True,
    phase: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vs_scale: float = 1.0,
    vpvs: float = DEFAULT_VPVS,
    taper_to_zero_outside: pd.DataFrame | None = None,
    taper_km: float = 1.0,
    projection: LocalProjection | None = None,
) -> xr.Dataset:
    """Checkerboard of alternating +/- velocity anomalies.

    The pattern is ``A * f(pi (x-px)/c) f(pi (y-py)/c) [f(pi (z-pz)/c)]`` with ``f = sin``
    (smooth, recommended for B-spline grids; ``A`` is then the peak amplitude) or
    ``f = sign(sin)`` (blocks of constant ``+-A``), so a *cell* (one sign) is ``cell_km`` wide
    along each axis and the period is ``2 cell_km``.  The origin of the pattern is the grid
    centre, so a node-centred pattern has its zero crossings at the centre.

    Parameters
    ----------
    cell_km
        Cell size in km (on the local projection, not degrees).
    offset_half_cell
        Shift the pattern by half a cell in x, y (and z when ``depth_cells``): the
        complementary checkerboard used to test the dependence on the node phase.
    depth_cells
        False: laterally varying only (constant in depth, a 2-D pattern).
    phase
        Extra ``(x, y, z)`` shift in km.
    vs_scale
        ``dvs = vs_scale * dvp`` (1 preserves Vp/Vs; 0 = P-only perturbation).
    taper_to_zero_outside
        Optional table with ``lat``, ``lon`` (and ``depth_km``) of the stations/events: the
        pattern is kept inside their bounding box and tapered to zero over ``taper_km``.

    Warns
    -----
    UnderSamplingWarning
        If ``cell_km < 2 x`` node spacing (laterally, or in depth when ``depth_cells``).
    """
    if cell_km <= 0:
        raise ValueError("cell_km must be positive")
    if shape not in ("sin", "block"):
        raise ValueError("shape must be 'sin' or 'block'")
    proj = _projection(grid, projection)
    bg_vp, bg_vs = _background(grid, background, vpvs)
    x, y, z = grid_xyz(grid, proj)
    warns = _check_sampling(grid, cell_km, depth_cells)
    half = 0.5 * cell_km if offset_half_cell else 0.0
    f = np.sin if shape == "sin" else lambda a: np.where(np.sin(a) >= 0, 1.0, -1.0)
    # sign(sin) at exact zero crossings is +1; nudge by a tiny phase to avoid grid-node ties
    eps = 1e-9 if shape == "block" else 0.0
    k = np.pi / cell_km
    pat = f(k * (x - phase[0] - half) + eps) * f(k * (y - phase[1] - half) + eps)
    if depth_cells:
        pat = pat * f(k * (z - phase[2] - half) + eps)
    if taper_to_zero_outside is not None:
        pat = pat * _region_weight(x, y, z, proj, taper_to_zero_outside, taper_km)
    dvp = amplitude_pct * pat
    return _dataset(
        grid,
        dvp,
        bg_vp,
        bg_vs,
        vs_scale,
        {
            "kind": "checkerboard",
            "cell_km": float(cell_km),
            "amplitude_pct": float(amplitude_pct),
            "offset_half_cell": int(offset_half_cell),
            "shape": shape,
            "depth_cells": int(depth_cells),
            "warnings": "; ".join(warns),
        },
    )


# ---------------------------------------------------------------------------------------
# spikes
# ---------------------------------------------------------------------------------------


def _get(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _spike_pattern(
    spec: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray, proj: LocalProjection
) -> np.ndarray:
    if _get(spec, "x_km") is not None:
        cx, cy, cz = float(spec["x_km"] if isinstance(spec, dict) else spec.x_km), 0.0, 0.0
        cy = float(_get(spec, "y_km", 0.0))
        cz = float(_get(spec, "z_km", 0.0))
    elif _get(spec, "lat") is not None:
        cx, cy = (float(v) for v in proj.forward(_get(spec, "lat"), _get(spec, "lon")))
        cz = float(_get(spec, "depth"))
    else:
        raise ValueError("spike needs x_km/y_km/z_km or lat/lon/depth")
    radius = float(_get(spec, "radius_km"))
    if radius <= 0:
        raise ValueError("spike radius_km must be positive")
    pct = float(_get(spec, "pct"))
    shape = _get(spec, "shape", "gaussian")
    r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    if shape == "gaussian":
        w = np.exp(-r2 / (2.0 * (0.5 * radius) ** 2))
    elif shape == "sphere":
        w = (r2 <= radius**2).astype(float)
    else:
        raise ValueError("spike shape must be 'gaussian' or 'sphere'")
    return pct * w


def spikes(
    grid: ModelGrid,
    background: Any,
    specs: Sequence[Any],
    *,
    vs_scale: float = 1.0,
    vpvs: float = DEFAULT_VPVS,
    projection: LocalProjection | None = None,
) -> xr.Dataset:
    """Spike (point-anomaly) model.

    Each spec (dict or ``SpikeConfig``) has ``x_km, y_km, z_km`` (local km from the grid
    centre, depth down) or ``lat, lon, depth``; ``radius_km``; ``pct`` and ``shape``
    (``'gaussian'`` with sigma = radius/2, peak = ``pct`` at the centre, or ``'sphere'``,
    uniform ``pct`` within the radius).  Overlapping spikes add.
    """
    proj = _projection(grid, projection)
    bg_vp, bg_vs = _background(grid, background, vpvs)
    x, y, z = grid_xyz(grid, proj)
    dvp = np.zeros(grid.shape_full)
    for s in specs:
        dvp += _spike_pattern(s, x, y, z, proj)
    return _dataset(grid, dvp, bg_vp, bg_vs, vs_scale, {"kind": "spikes", "n_spikes": len(specs)})


# ---------------------------------------------------------------------------------------
# anomaly bodies
# ---------------------------------------------------------------------------------------


def _body_weight(
    spec: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray, proj: LocalProjection
) -> np.ndarray:
    kind = _get(spec, "kind")
    if _get(spec, "x_km") is not None:
        cx, cy = float(_get(spec, "x_km")), float(_get(spec, "y_km", 0.0))
    else:
        cx, cy = (float(v) for v in proj.forward(_get(spec, "lat"), _get(spec, "lon")))
    cz = float(_get(spec, "z_km", _get(spec, "depth", 0.0)))
    strike = np.deg2rad(float(_get(spec, "strike_deg", 0.0)))
    dip = np.deg2rad(float(_get(spec, "dip_deg", 90.0)))
    edge = float(_get(spec, "edge_km", 0.0))
    length = float(_get(spec, "length_km", 2.0))
    width = float(_get(spec, "width_km", 2.0))
    d = np.stack([x - cx, y - cy, z - cz])  # (east, north, down)
    s = np.array([np.sin(strike), np.cos(strike), 0.0])  # along strike (azimuth)
    h = np.array([np.cos(strike), -np.sin(strike), 0.0])  # horizontal dip direction (right-hand)
    if kind == "block":
        u = np.tensordot(s, d, 1)
        v = np.tensordot(h, d, 1)
        w = d[2]
        half = np.array([length, width, float(_get(spec, "height_km", 2.0))]) / 2.0
        sd = np.maximum.reduce([np.abs(u) - half[0], np.abs(v) - half[1], np.abs(w) - half[2]])
    else:
        dd = np.array([np.cos(dip) * h[0], np.cos(dip) * h[1], np.sin(dip)])  # down-dip, unit
        nn = np.cross(s, dd)  # plane normal
        u = np.tensordot(s, d, 1)
        v = np.tensordot(dd, d, 1)
        w = np.tensordot(nn, d, 1)
        if kind == "slab":
            half = np.array([length, width, float(_get(spec, "thickness_km", 0.5))]) / 2.0
            sd = np.maximum.reduce([np.abs(u) - half[0], np.abs(v) - half[1], np.abs(w) - half[2]])
        elif kind == "cylinder":  # axis along the down-dip direction (dip 90 = vertical)
            radius = float(_get(spec, "radius_km", 1.0))
            sd = np.maximum(np.hypot(u, w) - radius, np.abs(v) - 0.5 * length)
        else:
            raise ValueError(f"unknown anomaly kind {kind!r}")
    if edge <= 0:
        return (sd <= 0).astype(float)
    return 1.0 - _smoothstep(sd / edge + 0.5)


def anomaly_bodies(
    grid: ModelGrid,
    background: Any,
    bodies: Sequence[Any],
    *,
    vs_scale: float = 1.0,
    vpvs: float = DEFAULT_VPVS,
    projection: LocalProjection | None = None,
) -> xr.Dataset:
    """Custom anomaly bodies (``AnomalyBodyConfig`` or dicts): block, cylinder, dipping slab.

    ``strike_deg`` is the azimuth (clockwise from north) of the long axis, ``dip_deg`` the
    dip below horizontal to the right of the strike direction (right-hand rule).  A slab is
    ``length_km`` along strike x ``width_km`` down dip x ``thickness_km`` normal to the
    plane; a block ``length_km`` along strike x ``width_km`` across x ``height_km`` vertical;
    a cylinder has its axis along the down-dip direction (``dip_deg=90``: vertical).
    ``edge_km > 0`` smooths the body edge with a cosine-like taper centred on the surface.
    Overlapping bodies add.
    """
    proj = _projection(grid, projection)
    bg_vp, bg_vs = _background(grid, background, vpvs)
    x, y, z = grid_xyz(grid, proj)
    dvp = np.zeros(grid.shape_full)
    for b in bodies:
        dvp += float(_get(b, "pct")) * _body_weight(b, x, y, z, proj)
    return _dataset(grid, dvp, bg_vp, bg_vs, vs_scale, {"kind": "anomaly", "n_bodies": len(bodies)})


# ---------------------------------------------------------------------------------------
# input models
# ---------------------------------------------------------------------------------------


def _regrid(grid: ModelGrid, ds: xr.Dataset) -> xr.Dataset:
    """Interpolate a dataset with ``vp[, vs]`` and lat/lon/depth coords onto the full grid."""
    if all(n in ds.dims for n in DIMS) and ds["vp"].shape == grid.shape_full:
        if all(
            np.allclose(ds[d].values, c)
            for d, c in zip(DIMS, (grid.depth_full, grid.lat_full, grid.lon_full), strict=True)
        ):
            return ds
    out = ds[[v for v in ("vp", "vs") if v in ds]].interp(
        depth=grid.depth_full, lat=grid.lat_full, lon=grid.lon_full, method="linear"
    )
    return out.transpose(*DIMS)


def input_model(
    kind: Literal["final_3d", "user_model", "starting", "file", "anomaly"],
    grid: ModelGrid,
    background: Any | None = None,
    *,
    dataset: xr.Dataset | None = None,
    path: str | Path | None = None,
    cfg: Any = None,
    model_1d: pd.DataFrame | None = None,
    inputs: Any = None,
    bodies: Sequence[Any] | None = None,
    vs_scale: float = 1.0,
    vpvs: float = DEFAULT_VPVS,
    projection: LocalProjection | None = None,
    **load_kw: Any,
) -> xr.Dataset:
    """Build a synthetic *input model* for recovery tests on ``grid``.

    kind
        ``'final_3d'``: a Dataset (``dataset``) with ``vp[, vs]`` or a NetCDF ``path`` (the
        inversion result; interpolated onto the grid when its axes differ).
        ``'user_model'`` / ``'starting'``: :func:`osv.model3d.builder.build_starting_model`
        with ``cfg``, ``model_1d`` and ``inputs``.
        ``'file'``: NetCDF/CSV via :func:`osv.model3d.user_grid.load_user_grid` (``path`` and
        ``**load_kw``: ``crs``, ``columns``, ``z_convention``, ``units``).
        ``'anomaly'``: ``bodies`` added to ``background`` (see :func:`anomaly_bodies`).

    Returns
    -------
    xarray.Dataset
        ``vp``, ``vs``, ``dvp_pct`` (percent relative to ``background``, or to the laterally
        averaged model of each depth when no background is given), ``vp_bg``, ``vs_bg``.
        Nodes missing in the source (NaN) are filled with the background.
    """
    if kind == "anomaly":
        if background is None or bodies is None:
            raise ValueError("kind='anomaly' needs background and bodies")
        return anomaly_bodies(
            grid, background, bodies, vs_scale=vs_scale, vpvs=vpvs, projection=projection
        )
    if kind == "final_3d":
        if dataset is None and path is None:
            raise ValueError("kind='final_3d' needs dataset or path")
        src = dataset if dataset is not None else xr.load_dataset(path)
        src = _regrid(grid, src)
    elif kind in ("user_model", "starting"):
        from osv.model3d.builder import build_starting_model

        if cfg is None:
            raise ValueError(f"kind={kind!r} needs cfg (Model3DConfig)")
        src = build_starting_model(grid, cfg, model_1d, inputs).dataset
    elif kind == "file":
        if path is None:
            raise ValueError("kind='file' needs path")
        src, _ = load_user_grid(path, grid, **load_kw)
    else:
        raise ValueError(f"unknown input model kind {kind!r}")

    vp = _as_full(grid, src["vp"], "vp")
    vs = _as_full(grid, src["vs"], "vs") if "vs" in src else vp / vpvs
    if background is not None:
        bg_vp, bg_vs = _background(grid, background, vpvs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m_vp = np.nanmean(vp, axis=(1, 2), keepdims=True)
            m_vs = np.nanmean(vs, axis=(1, 2), keepdims=True)
        bg_vp = np.broadcast_to(m_vp, vp.shape).copy()
        bg_vs = np.broadcast_to(m_vs, vs.shape).copy()
    vp = np.where(np.isfinite(vp), vp, bg_vp)
    vs = np.where(np.isfinite(vs), vs, bg_vs)
    dvp = 100.0 * (vp / bg_vp - 1.0)
    ds = _dataset(grid, dvp, bg_vp, bg_vs, vs_scale, {"kind": f"input:{kind}"})
    ds["vp"] = (DIMS, vp, {"units": "km/s"})
    ds["vs"] = (DIMS, vs, {"units": "km/s"})
    return ds


# ---------------------------------------------------------------------------------------
# summary / output
# ---------------------------------------------------------------------------------------


def pattern_summary(ds: xr.Dataset, grid: ModelGrid | None = None) -> dict[str, float]:
    """Amplitude statistics of ``dvp_pct`` (cushion nodes excluded when ``grid`` is given)."""
    a = ds["dvp_pct"]
    if grid is not None:
        a = a.where(~grid.cushion_mask())
    v = a.values[np.isfinite(a.values)]
    if v.size == 0:
        raise ValueError("no finite pattern values")
    out = {
        "min_pct": float(v.min()),
        "max_pct": float(v.max()),
        "mean_pct": float(v.mean()),
        "rms_pct": float(np.sqrt(np.mean(v**2))),
        "peak_abs_pct": float(np.abs(v).max()),
        "frac_positive": float(np.mean(v > 1e-9)),
        "frac_negative": float(np.mean(v < -1e-9)),
        "n_nodes": float(v.size),
    }
    if {"vp", "vs"} <= set(ds.data_vars):
        r = (ds["vp"] / ds["vs"]).values
        out["vpvs_min"], out["vpvs_max"] = float(np.nanmin(r)), float(np.nanmax(r))
    return out


def write_vgrids(
    grid: ModelGrid, ds: xr.Dataset, *, with_s: bool = True, uncertainty: float | None = None
) -> Any:
    """:class:`~osv.io.fmtomo.grids.VelocityGrids` (Vp, optionally Vs) of a test model."""
    return grid.to_vgrids(ds["vp"], ds["vs"] if with_s else None, uncertainty=uncertainty)
