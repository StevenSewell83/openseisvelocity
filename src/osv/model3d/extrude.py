"""Extension of the 1-D VELEST model to a 3-D FMTOMO velocity grid (Stage 3, task 4.2).

Modes
-----
``layered``
    VELEST layers are constant-velocity slabs *below* each ``depth_top_km``.  The node at depth
    ``z`` takes the velocity of the deepest layer whose top is ``<= z`` (a node exactly on a top
    belongs to the layer below).  Above the first top the first layer is extended upwards; the
    last layer extends to the bottom of the grid.  ``smoothing_km`` convolves the step profile with
    a Gaussian of that standard deviation (sampled on a 10 m grid, edge values extended).
``gradient``
    Linear interpolation between the **mid-depths** of the layers (each layer is represented by
    its velocity at its centre, so the layer average is approximately kept).  The last layer has
    no base: its mid-depth is placed half the thickness of the previous layer below its top.
    Above the first mid-depth and below the last one the velocity is constant (the first and last
    layer velocities), so the extended model above the first top (the part of the grid above sea
    level, e.g. above ``-0.6 km``) is filled with the first-layer velocity.

``vs`` comes from ``vpvs`` (scalar, or ``None`` to use the ``vs`` column, else the ``vpvs``
column of the table).

FMTOMO's own interpolation (verified in ``3dfmlib.f90:468-560``)
-----------------------------------------------------------------
``interpolate_velocity`` evaluates a *uniform cubic B-spline* whose control points are the node
values.  That spline is **approximating, not interpolating**: at a node it returns
``(v[i-1] + 4 v[i] + v[i+1]) / 6`` and every basis function is non-negative, so a step in the node
values is smeared over about three nodes but **never rings or overshoots**.  Ringing only
appears if one pre-filters the nodes to make the spline interpolate (not done here).  Measured on
a 2 -> 4 km/s step on a node (:func:`effective_profile`, ``dz = 0.5 km``): the effective profile
is monotonic with zero overshoot (< 1e-15 km/s) and a 10-90 % transition width of 0.66 km (about
one node), equal to the nodal step smoothed with ``[1, 4, 1] / 6``.  A Gaussian ``smoothing_km``
(sigma) of 0.25 / 0.5 / 1.0 km widens that to 1.0 / 1.5 / 2.7 km; overshoot stays below 1e-3 km/s.
Smoothing is therefore optional for the solver's sake (no ringing to suppress); use it to avoid
the eikonal solver meeting velocity jumps of several km/s within one propagation cell.
The :func:`effective_profile` helper evaluates the profile
actually seen by ``fm3d`` so the effect can be checked on any model.

``vgrids.in`` second column (``invert3d.f90:555-560``): ``cm = covvm**2`` is the a priori model
variance and the damping term is ``epsv / cm``; the column is therefore a standard deviation in
km/s (larger = freer to change).  Default: constant 0.3 km/s, or ``spread`` = half of the
p05-p95 range of each 1-D layer (clipped).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from osv.io.fmtomo.grids import VelocityGrids
from osv.model3d.grid import GridError, ModelGrid

log = logging.getLogger(__name__)

__all__ = [
    "ExtrusionError",
    "effective_profile",
    "extrude_1d",
    "extrusion_to_vgrids",
    "profile_1d",
]

#: Sampling step (km) of the fine depth axis used for Gaussian smoothing.
SMOOTH_STEP_KM = 0.01


class ExtrusionError(GridError):
    """The 1-D model cannot be extruded (message says what to fix)."""


def _layers(model_1d: pd.DataFrame) -> pd.DataFrame:
    if "depth_top_km" not in model_1d or "vp" not in model_1d:
        raise ExtrusionError(
            "model_1d needs columns 'depth_top_km' and 'vp' (Stage 2 final_model.csv)"
        )
    m = model_1d.sort_values("depth_top_km").reset_index(drop=True)
    if m["depth_top_km"].duplicated().any():
        raise ExtrusionError("model_1d has duplicate depth_top_km values")
    if not np.isfinite(m["vp"]).all() or (m["vp"] <= 0).any():
        raise ExtrusionError("model_1d: vp must be finite and positive")
    return m


def _gradient_nodes(tops: np.ndarray) -> np.ndarray:
    """Mid-depth of each layer; the last layer borrows the previous thickness."""
    if tops.size == 1:
        return tops.copy()
    mids = np.empty_like(tops)
    mids[:-1] = 0.5 * (tops[:-1] + tops[1:])
    mids[-1] = tops[-1] + 0.5 * (tops[-1] - tops[-2])
    return mids


def by_depth_profile(depth: np.ndarray, pairs: Any) -> np.ndarray:
    """Piecewise-constant profile of ``[depth_top_km, value]`` pairs at ``depth`` (km).

    Each value holds from its top down to the next top; depths above the first top take the
    first value.

    Raises
    ------
    ExtrusionError
        If ``pairs`` is empty, the tops are not strictly ascending or a value is not positive.
    """
    arr = np.asarray(pairs, dtype=float).reshape(-1, 2)
    if not len(arr):
        raise ExtrusionError("uncertainty.by_depth needs at least one [depth_top_km, km_s] pair")
    tops, vals = arr[:, 0], arr[:, 1]
    if np.any(np.diff(tops) <= 0) or np.any(vals <= 0):
        raise ExtrusionError("uncertainty.by_depth: tops must ascend and values be > 0")
    idx = np.clip(np.searchsorted(tops, np.asarray(depth, dtype=float), side="right") - 1, 0, None)
    return vals[idx]


def profile_1d(
    depth: np.ndarray,
    tops: np.ndarray,
    values: np.ndarray,
    *,
    mode: Literal["gradient", "layered"] = "gradient",
    smoothing_km: float = 0.0,
) -> np.ndarray:
    """Evaluate a layered/gradient 1-D profile at ``depth`` (km, any order).

    ``tops`` ascending layer tops, ``values`` the layer values.  See the module docstring.
    """
    depth = np.asarray(depth, dtype=float)
    tops = np.asarray(tops, dtype=float)
    values = np.asarray(values, dtype=float)
    if mode == "gradient":
        if smoothing_km > 0:
            log.info("smoothing_km is ignored in gradient mode (the profile is already smooth)")
        return np.interp(depth, _gradient_nodes(tops), values)
    if mode != "layered":
        raise ExtrusionError(f"mode must be 'gradient' or 'layered', got {mode!r}")

    def step(z: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(tops, z + 1e-9, side="right") - 1
        return values[np.clip(idx, 0, tops.size - 1)]

    if smoothing_km <= 0:
        return step(depth)
    h = min(SMOOTH_STEP_KM, smoothing_km / 5.0)
    half = int(np.ceil(4.0 * smoothing_km / h))
    kz = np.arange(-half, half + 1) * h
    kern = np.exp(-0.5 * (kz / smoothing_km) ** 2)
    kern /= kern.sum()
    z0 = float(min(depth.min(), tops.min())) - half * h
    z1 = float(depth.max()) + half * h
    zf = z0 + h * np.arange(int(np.ceil((z1 - z0) / h)) + 1)
    sm = np.convolve(step(zf), kern, mode="same")
    return np.interp(depth, zf, sm)


def _vs_values(m: pd.DataFrame, vpvs: float | None) -> np.ndarray:
    if vpvs is not None:
        if vpvs <= 0:
            raise ExtrusionError("vpvs must be positive")
        return m["vp"].to_numpy(float) / float(vpvs)
    if "vs" in m and m["vs"].notna().all():
        return m["vs"].to_numpy(float)
    if "vpvs" in m and m["vpvs"].notna().all():
        return m["vp"].to_numpy(float) / m["vpvs"].to_numpy(float)
    raise ExtrusionError("no S model: pass vpvs=<ratio> or give a 'vs' or 'vpvs' column")


def _spread(m: pd.DataFrame, name: str) -> np.ndarray | None:
    lo, hi = f"{name}_p05", f"{name}_p95"
    if lo in m and hi in m and m[[lo, hi]].notna().all().all():
        return 0.5 * (m[hi].to_numpy(float) - m[lo].to_numpy(float))
    return None


def extrude_1d(
    model_1d: pd.DataFrame,
    grid: ModelGrid,
    mode: Literal["gradient", "layered"] = "gradient",
    vpvs: float | None = None,
    *,
    smoothing_km: float = 0.0,
    uncertainty: Any = None,
) -> xr.Dataset:
    """Extend a 1-D VELEST model to the full FMTOMO grid (cushions included).

    Parameters
    ----------
    model_1d
        Table with ``depth_top_km`` (km below sea level, may be negative), ``vp`` and ``vs`` or
        ``vpvs`` (Stage 2 ``final_model.csv``); ``vp_p05``/``vp_p95`` (and the ``vs`` pair) are
        used for the ``spread`` uncertainty.
    grid
        :class:`~osv.model3d.grid.ModelGrid` from ``define_grid``.
    mode, smoothing_km
        See the module docstring; ``smoothing_km`` only affects ``layered``.
    vpvs
        Vp/Vs ratio overriding the table's S model (``vs = vp / vpvs``).
    uncertainty
        ``ExtrusionUncertaintyConfig`` (or anything with ``mode``, ``value_km_s``, ``min_km_s``,
        ``max_km_s``); default constant 0.3 km/s.

    Returns
    -------
    xarray.Dataset
        Variables ``vp``, ``vs``, ``vp_unc``, ``vs_unc`` (km/s) with dims ``depth, lat, lon``
        (depth ascending), on ``grid.template()`` coordinates.  ``attrs`` record the mode.
    """
    m = _layers(model_1d)
    tops = m["depth_top_km"].to_numpy(float)
    vp = m["vp"].to_numpy(float)
    vs = _vs_values(m, vpvs)
    if (vs <= 0).any() or not np.isfinite(vs).all():
        raise ExtrusionError("S velocities must be finite and positive")

    u_mode = getattr(uncertainty, "mode", "constant")
    u_val = float(getattr(uncertainty, "value_km_s", 0.3))
    u_lo, u_hi = (
        float(getattr(uncertainty, "min_km_s", 0.1)),
        float(getattr(uncertainty, "max_km_s", 1.0)),
    )
    if u_mode == "spread":
        sp_p, sp_s = _spread(m, "vp"), _spread(m, "vs")
        if sp_p is None:
            raise ExtrusionError("uncertainty.mode='spread' needs vp_p05 and vp_p95 columns")
        if sp_s is None:
            sp_s = sp_p / (vpvs if vpvs else float(np.mean(vp / vs)))
        unc_p, unc_s = np.clip(sp_p, u_lo, u_hi), np.clip(sp_s, u_lo, u_hi)
    elif u_mode in ("constant", "by_depth"):
        unc_p = unc_s = np.full(tops.size, u_val)
    else:
        raise ExtrusionError(
            f"uncertainty.mode must be 'constant', 'spread' or 'by_depth', got {u_mode!r}"
        )

    z = grid.depth_full
    ds = grid.template()
    ds = ds.drop_vars("velocity")
    shape = grid.shape_full
    for name, arr in (("vp", vp), ("vs", vs), ("vp_unc", unc_p), ("vs_unc", unc_s)):
        prof = profile_1d(z, tops, arr, mode=mode, smoothing_km=smoothing_km)
        ds[name] = (
            ("depth", "lat", "lon"),
            np.broadcast_to(prof[:, None, None], shape).copy(),
            {"units": "km/s"},
        )
    if u_mode == "by_depth":
        prof = by_depth_profile(z, getattr(uncertainty, "by_depth", []))
        for name in ("vp_unc", "vs_unc"):
            ds[name] = (
                ("depth", "lat", "lon"),
                np.broadcast_to(prof[:, None, None], shape).copy(),
                {"units": "km/s"},
            )
    ds.attrs.update(
        {"extrusion_mode": mode, "smoothing_km": float(smoothing_km), "uncertainty_mode": u_mode}
    )
    if z[0] < tops[0] or z[-1] > tops[-1]:
        log.info(
            "grid depth %.2f..%.2f km extends beyond the 1-D layer tops %.2f..%.2f: "
            "first/last layer velocities extended",
            z[0],
            z[-1],
            tops[0],
            tops[-1],
        )
    return ds


def extrusion_to_vgrids(grid: ModelGrid, ds: xr.Dataset, *, with_s: bool = True) -> VelocityGrids:
    """:class:`VelocityGrids` (type 1 = Vp, type 2 = Vs) with the per-node uncertainty column.

    ``ds`` as returned by :func:`extrude_1d` (or any dataset with ``vp``, ``vs`` and optional
    ``vp_unc``/``vs_unc``).  Uncertainties are written whenever present.
    """
    vg = grid.to_vgrids(ds["vp"], ds["vs"] if with_s else None)
    for _layer, vtype, vds in vg:
        key = "vp_unc" if vtype == 1 else "vs_unc"
        if key in ds:
            unc = grid._full_values(ds[key], key)
            vds["uncertainty"] = (
                ("depth", "lat", "lon"),
                unc,
                {
                    "units": "km/s",
                    "note": "second column of vgrids.in ('covariance'); squared by invert3d",
                },
            )
    return vg


def effective_profile(depth_nodes: np.ndarray, values: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Velocity ``fm3d`` evaluates along depth for node ``values`` (uniform cubic B-spline).

    Mirrors ``interpolate_velocity`` (``3dfmlib.f90:468-560``): valid for depths between the
    second and the second-to-last node; others give NaN.  ``depth_nodes`` must be regularly
    spaced (either order); ``values`` are the node values in the same order.
    """
    x = np.asarray(depth_nodes, dtype=float)
    v = np.asarray(values, dtype=float)
    if x[0] > x[-1]:
        x, v = x[::-1], v[::-1]
    dz = x[1] - x[0]
    z = np.asarray(depth, dtype=float)
    i = np.floor((z - x[0]) / dz).astype(int)  # 0-based node at or below z
    out = np.full(z.shape, np.nan)
    ok = (i >= 1) & (i <= x.size - 3)
    u = (z[ok] - x[i[ok]]) / dz
    b = (
        (1 - u) ** 3 / 6,
        (4 - 6 * u**2 + 3 * u**3) / 6,
        (1 + 3 * u + 3 * u**2 - 3 * u**3) / 6,
        u**3 / 6,
    )
    out[ok] = sum(b[k] * v[i[ok] - 1 + k] for k in range(4))
    return out
