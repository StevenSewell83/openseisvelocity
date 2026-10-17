"""Layer-cake geology model: stack units between horizons and assign velocities (task 4.5).

Conventions
-----------
* ``horizons`` are ordered **top to bottom**; each horizon is the **base** of its unit.
  Unit ``i`` occupies ``base(i-1) <= depth < base(i)``; unit 0 extends from the top of the
  model down to ``base(0)``.  Below the last base the *background* is used (unit id ``-1``).
* All depths are km below sea level, positive down.
* A velocity law is evaluated at ``h = max(depth - reference, 0)`` where ``reference`` is
  the top of the unit (``unit_top``, default; for unit 0 this is the ground surface), the
  ground surface (``surface``, ``surface_depth_km`` argument, default 0) or sea level.
* A column with an undefined (NaN) base for any horizon is assigned to the background
  entirely (the stack cannot be built there) and counted in the findings.
* Crossing horizons (negative thickness) are repaired with
  ``base(i) = max(base(i), base(i-1))``; the number of affected nodes is reported.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from osv.model3d.grid import ModelGrid
from osv.model3d.horizons_io import read_horizon

log = logging.getLogger(__name__)

__all__ = ["GeologyFinding", "apply_law", "build_geology_model"]

#: Default Vp/Vs of the background when it is given as Vp only.
BACKGROUND_VPVS = 1.75

Background = Callable[[np.ndarray], np.ndarray] | float | None


@dataclass
class GeologyFinding:
    """A QC observation (``level`` is ``info``, ``warning`` or ``error``)."""

    level: str
    code: str
    message: str
    n_nodes: int = 0


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def apply_law(law: Any, h_km: np.ndarray) -> np.ndarray:
    """Velocity (km/s) of ``law`` (config object or dict) at depth ``h_km`` below its reference.

    ``constant``: ``v0``; ``gradient``: ``v0 + k h``; ``exponential`` (compaction):
    ``vmax - (vmax - v0) exp(-h / lambda_km)``.
    """
    kind = _get(law, "law")
    h = np.maximum(np.asarray(h_km, dtype=float), 0.0)
    v0 = float(_get(law, "v0"))
    if kind == "constant":
        return np.full(h.shape, v0)
    if kind == "gradient":
        return v0 + float(_get(law, "k")) * h
    if kind == "exponential":
        vmax, lam = float(_get(law, "vmax")), float(_get(law, "lambda_km"))
        return vmax - (vmax - v0) * np.exp(-h / lam)
    raise ValueError(f"unknown velocity law {kind!r}; use constant, gradient or exponential")


def _background(bg: Background, depth: np.ndarray, vpvs: float) -> tuple[np.ndarray, np.ndarray]:
    if bg is None:
        v = np.full(depth.shape, np.nan)
    elif callable(bg):
        v = np.asarray(bg(depth), dtype=float)
    else:
        v = np.full(depth.shape, float(bg))
    return v, v / vpvs


def build_geology_model(
    horizons_cfg: Sequence[Any],
    grid: ModelGrid,
    background: Background = None,
    *,
    surfaces: Mapping[str, np.ndarray] | None = None,
    surface_depth_km: np.ndarray | float = 0.0,
    background_vpvs: float = BACKGROUND_VPVS,
    base_dir: str | Path | None = None,
) -> tuple[xr.Dataset, list[GeologyFinding]]:
    """Stack the geological units and assign Vp/Vs on the (full) model grid.

    Parameters
    ----------
    horizons_cfg
        ``user_model.horizons`` entries (config objects or dicts with ``name``, ``surface``,
        ``vp``, optional ``vs``, ``vpvs``, ``crs``, ``z_convention``, ``units``), top to bottom.
    grid
        The :class:`~osv.model3d.grid.ModelGrid`; output is on its full grid.
    background
        Below the last horizon and in columns without a complete stack: a function
        ``f(depth_km) -> vp`` (e.g. the VELEST 1-D model), a constant Vp, or None (NaN).
        Background Vs is ``vp / background_vpvs``.
    surfaces
        Optional ``{horizon name: depth_km array (nlat, nlon)}`` used instead of reading
        the file (already gridded surfaces, depth km bsl).
    surface_depth_km
        Ground-surface depth (km, scalar or ``(nlat, nlon)``) used as the top reference of
        unit 0 and for ``reference: surface`` laws.
    base_dir
        Directory against which relative surface paths are resolved.

    Returns
    -------
    (xarray.Dataset, list[GeologyFinding])
        Dataset with ``vp``, ``vs`` (km/s) and integer ``unit_id`` (``-1`` = background; the
        unit names are in ``attrs['unit_names']``), dims ``(depth, lat, lon)``.
    """
    if not len(horizons_cfg):
        raise ValueError("horizons_cfg: at least one horizon is required")
    findings: list[GeologyFinding] = []
    lat, lon, depth = grid.lat_full, grid.lon_full, grid.depth_full
    nlat, nlon = lat.size, lon.size
    names = [str(_get(h, "name")) for h in horizons_cfg]
    if len(set(names)) != len(names):
        raise ValueError("horizon names must be unique")

    bases = np.empty((len(names), nlat, nlon))
    for i, h in enumerate(horizons_cfg):
        if surfaces is not None and names[i] in surfaces:
            z = np.asarray(surfaces[names[i]], dtype=float)
            if z.shape != (nlat, nlon):
                raise ValueError(f"surfaces[{names[i]!r}]: shape {z.shape} != {(nlat, nlon)}")
        else:
            p = Path(_get(h, "surface"))
            if base_dir is not None and not p.is_absolute():
                p = Path(base_dir) / p
            z = read_horizon(
                p,
                grid,
                crs=_get(h, "crs"),
                z_convention=_get(h, "z_convention", "depth"),
                units=_get(h, "units", "m"),
            )
        bases[i] = z

    # columns where the stack cannot be built
    undefined = ~np.isfinite(bases).all(axis=0)
    for i, nm in enumerate(names):
        n_nan = int((~np.isfinite(bases[i])).sum())
        if n_nan:
            findings.append(
                GeologyFinding(
                    "warning",
                    "horizon_undefined",
                    f"horizon '{nm}' is undefined at {n_nan} of {nlat * nlon} lateral nodes",
                    n_nan,
                )
            )
    if undefined.any():
        findings.append(
            GeologyFinding(
                "warning",
                "background_columns",
                f"{int(undefined.sum())} lateral nodes lack a complete horizon stack; "
                "the background model is used there",
                int(undefined.sum()),
            )
        )
    filled = np.where(undefined[None], 0.0, bases)

    # crossing horizons -> clip so thickness >= 0
    for i in range(1, len(names)):
        bad = (filled[i] < filled[i - 1]) & ~undefined
        if bad.any():
            worst = float((filled[i - 1] - filled[i])[bad].max())
            findings.append(
                GeologyFinding(
                    "warning",
                    "crossing_horizons",
                    f"horizon '{names[i]}' lies above '{names[i - 1]}' at {int(bad.sum())} "
                    f"lateral nodes (max overlap {worst:.3f} km); clipped to zero thickness",
                    int(bad.sum()),
                )
            )
            filled[i] = np.where(bad, filled[i - 1], filled[i])

    nz = depth.size
    z3 = depth[:, None, None]
    unit = np.zeros((nz, nlat, nlon), dtype=np.int32)
    for i in range(len(names)):
        unit += z3 >= filled[i][None]
    unit[unit >= len(names)] = -1
    unit[:, undefined] = -1

    surf = np.broadcast_to(np.asarray(surface_depth_km, dtype=float), (nlat, nlon))
    vp, vs = _background(background, np.broadcast_to(z3, unit.shape), background_vpvs)
    vp, vs = vp.copy(), vs.copy()
    for i, h in enumerate(horizons_cfg):
        sel = unit == i
        n = int(sel.sum())
        if n == 0:
            findings.append(
                GeologyFinding("info", "empty_unit", f"unit '{names[i]}' contains no nodes", 0)
            )
            continue
        top = surf if i == 0 else filled[i - 1]
        zz = np.broadcast_to(z3, unit.shape)

        def evaluate(law: Any, _top: np.ndarray = top, _zz: np.ndarray = zz) -> np.ndarray:
            ref = _get(law, "reference", "unit_top")
            origin = {"unit_top": _top, "surface": surf, "sea_level": np.zeros_like(surf)}.get(ref)
            if origin is None:
                raise ValueError(f"unknown law reference {ref!r}")
            return apply_law(law, _zz - origin[None])

        vp_i = evaluate(_get(h, "vp"))
        law_s = _get(h, "vs")
        vs_i = evaluate(law_s) if law_s is not None else vp_i / float(_get(h, "vpvs", 1.75))
        vp[sel] = vp_i[sel]
        vs[sel] = vs_i[sel]

    n_bg = int((unit == -1).sum())
    if n_bg and background is None:
        findings.append(
            GeologyFinding(
                "warning",
                "no_background",
                f"{n_bg} nodes are outside all units and no background model was given (NaN)",
                n_bg,
            )
        )
    dims = ("depth", "lat", "lon")
    ds = xr.Dataset(
        {
            "vp": (dims, vp, {"units": "km/s"}),
            "vs": (dims, vs, {"units": "km/s"}),
            "unit_id": (dims, unit, {"long_name": "unit index (0 = top unit), -1 = background"}),
        },
        coords={
            "depth": ("depth", depth, {"units": "km", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"unit_names": names},
    )
    for f in findings:
        log.log(logging.WARNING if f.level == "warning" else logging.INFO, f.message)
    return ds, findings
