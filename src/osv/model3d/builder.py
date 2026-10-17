"""Starting-model builder: blend the user model into the 1-D background (Stage 3, task 4.7).

:func:`build_starting_model` dispatches on ``model3d.source``, builds the user model (user grid,
geology, wells or geology+wells) on the full FMTOMO grid and blends it into the extruded 1-D
background

``v = w * v_user + (1 - w) * v_background``.

Blending weight
---------------
``w`` is built from the user model's *coverage mask*: ``w = clip(d / blend_taper_km, 0, 1)``
where ``d`` is the Euclidean distance (km, per-axis node spacings of depth/lat/lon, via
``scipy.ndimage.distance_transform_edt``) of a covered node to the nearest uncovered node.  The
weight therefore rises linearly from 0 at the coverage edge to 1 one taper width inside, and is
continuous (adjacent nodes differ by at most ``spacing / blend_taper_km``).  The model
boundary is *not* a coverage edge (the mask is edge-padded), so a user model that reaches the grid
boundary is not tapered there.  ``blend_taper_km = 0`` gives a hard switch.  Wells carry their
own distance weight, which is multiplied with the taper.

Smoothing
---------
``smoothing_km`` is a Gaussian sigma applied along each axis with sigma/spacing nodes
(``mode='reflect'``, which conserves the mean); ``smoothing_target='blend'`` smooths the
blended model, ``'user'`` only the user model before blending.

Vs
--
The user model's Vs where it has one, else ``vp / (vp_bg / vs_bg)`` using the background Vp/Vs at
each node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from osv.io.fmtomo.grids import VelocityGrids
from osv.model3d.extrude import ExtrusionError, extrude_1d, extrusion_to_vgrids
from osv.model3d.geology import GeologyFinding, build_geology_model
from osv.model3d.grid import GridError, ModelGrid
from osv.model3d.user_grid import load_user_grid
from osv.model3d.wells import build_geology_wells_model, build_wells_model

log = logging.getLogger(__name__)

__all__ = ["StartingInputs", "StartingModel", "blend_weight", "build_starting_model"]

DIMS = ("depth", "lat", "lon")
#: Weights above this count a node as influenced by the user model.
INFLUENCE_EPS = 1e-6


@dataclass
class StartingInputs:
    """Inputs of :func:`build_starting_model` resolved from the project (all optional).

    Attributes
    ----------
    base_dir
        Directory against which relative paths of ``user_model`` (grid, surfaces, wells) are
        resolved (the project directory).
    stations
        Station table (``elevation_m``) used for the ground-surface depth of the geology.
    surface_depth_km
        Ground-surface depth (km bsl; scalar or ``(nlat, nlon)`` on the full lateral grid);
        default the negative median station elevation (0 without stations).
    user_grid
        Preloaded ``(dataset, coverage)`` from :func:`~osv.model3d.user_grid.load_user_grid`.
    surfaces
        Preloaded gridded horizons ``{name: depth_km (nlat, nlon)}`` for the geology.
    wells
        Preprocessed wells (:func:`~osv.model3d.wells.load_wells`).
    """

    base_dir: Path | str | None = None
    stations: pd.DataFrame | None = None
    surface_depth_km: Any = None
    user_grid: tuple[xr.Dataset, xr.DataArray] | None = None
    surfaces: dict[str, np.ndarray] | None = None
    wells: list[Any] | None = None


@dataclass
class StartingModel:
    """Result of :func:`build_starting_model`.

    ``dataset`` has ``vp``, ``vs``, ``vp_unc``, ``vs_unc``, ``weight`` (user-model blending
    weight), ``vp_bg``, ``vs_bg`` (background) and ``unit_id`` (geology sources), all on the
    full grid with dims ``(depth, lat, lon)``.
    """

    grid: ModelGrid
    dataset: xr.Dataset
    findings: list[GeologyFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_vgrids(self, *, with_s: bool = True, uncertainty: bool = True) -> VelocityGrids:
        """:class:`VelocityGrids` (type 1 = Vp, type 2 = Vs) with the a priori uncertainty."""
        ds = self.dataset if uncertainty else self.dataset.drop_vars(["vp_unc", "vs_unc"])
        return extrusion_to_vgrids(self.grid, ds, with_s=with_s)


def _spacing(grid: ModelGrid) -> tuple[float, float, float]:
    """Node spacings in km along ``(depth, lat, lon)``."""
    return (float(grid.dz_km), float(grid.dy_km), float(grid.dx_km))


def blend_weight(
    mask: np.ndarray, spacing_km: tuple[float, float, float], taper_km: float
) -> np.ndarray:
    """Taper weight in [0, 1] from a boolean coverage mask (see the module docstring)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.all():
        return np.ones(mask.shape)
    if not mask.any():
        return np.zeros(mask.shape)
    if taper_km <= 0:
        return mask.astype(float)
    padded = np.pad(mask, 1, mode="edge")
    d = ndimage.distance_transform_edt(padded, sampling=spacing_km)[1:-1, 1:-1, 1:-1]
    return np.clip(d / taper_km, 0.0, 1.0)


def _smooth(a: np.ndarray, spacing_km: tuple[float, float, float], sigma_km: float) -> np.ndarray:
    """Gaussian smoothing, NaN-aware (normalised convolution), sigma in km per axis."""
    if sigma_km <= 0:
        return a
    sig = [sigma_km / s for s in spacing_km]
    fin = np.isfinite(a)
    if fin.all():
        return ndimage.gaussian_filter(a, sig, mode="reflect")
    num = ndimage.gaussian_filter(np.where(fin, a, 0.0), sig, mode="reflect")
    den = ndimage.gaussian_filter(fin.astype(float), sig, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(fin & (den > 0), out, np.nan)


def _background(
    grid: ModelGrid, m3: Any, model_1d: pd.DataFrame | None
) -> tuple[xr.Dataset, list[GeologyFinding]]:
    """Background dataset (vp, vs, vp_unc, vs_unc) per ``user_model.background``."""
    um = m3.user_model
    ext = m3.extrusion
    findings: list[GeologyFinding] = []
    kind = "velest_1d" if m3.source == "velest_1d" else um.background
    if kind == "velest_1d" or kind == "none":
        if model_1d is None:
            if kind == "velest_1d":
                raise ExtrusionError("model_1d (Stage 2 final_model.csv) is required")
        else:
            bg = extrude_1d(
                model_1d,
                grid,
                ext.mode,
                smoothing_km=ext.smoothing_km,
                uncertainty=ext.uncertainty,
            )
            if kind == "none":
                bg["vp"][:] = np.nan
                bg["vs"][:] = np.nan
            return bg, findings
    if kind == "constant" and um.background_vp_km_s is None:
        raise GridError("user_model.background_vp_km_s is required for background 'constant'")
    v = float(um.background_vp_km_s) if kind == "constant" else np.nan
    vs = v / float(um.background_vpvs)
    bg = grid.template().drop_vars("velocity")
    u = float(ext.uncertainty.value_km_s)
    for name, val in (("vp", v), ("vs", vs), ("vp_unc", u), ("vs_unc", u)):
        bg[name] = (DIMS, np.full(grid.shape_full, val), {"units": "km/s"})
    bg.attrs["extrusion_mode"] = "constant"
    return bg, findings


def _bg_profile(bg: xr.Dataset) -> Any:
    """``f(depth_km) -> vp`` of the (laterally uniform or not) background: lateral mean."""
    vp = bg["vp"].values
    if not np.isfinite(vp).any():
        return None
    prof = np.nanmean(vp, axis=(1, 2))
    z = bg["depth"].values
    return lambda d: np.interp(np.asarray(d, dtype=float), z, prof)


def _surface_depth(grid: ModelGrid, inputs: StartingInputs) -> Any:
    if inputs.surface_depth_km is not None:
        return inputs.surface_depth_km
    st = inputs.stations
    if st is not None and len(st) and "elevation_m" in st:
        return -float(np.nanmedian(st["elevation_m"].to_numpy(float))) / 1000.0
    return 0.0


def _resolve(path: Any, base: Path | str | None) -> Path:
    p = Path(path)
    return p if p.is_absolute() or base is None else Path(base) / p


def _build_user(
    grid: ModelGrid, m3: Any, bg: xr.Dataset, inputs: StartingInputs
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[GeologyFinding]]:
    """User-model fields ``{vp, vs?, unit_id?}``, coverage mask, own weight, findings."""
    um = m3.user_model
    src = m3.source
    findings: list[GeologyFinding] = []
    ones = np.ones(grid.shape_full)
    base = inputs.base_dir
    bgf = _bg_profile(bg)
    if src == "user_grid":
        if inputs.user_grid is not None:
            ds, cov = inputs.user_grid
        else:
            ds, cov = load_user_grid(
                _resolve(um.grid_file, base),
                grid,
                crs=um.grid_crs,
                columns=um.grid_columns or None,
                z_convention=um.z_convention,
                units=um.units,
            )
        fields = {k: ds[k].values.astype(float) for k in ("vp", "vs") if k in ds}
        if "vs" not in fields:
            findings.append(
                GeologyFinding(
                    "info", "vs_from_vpvs", "user grid has no Vs; Vs = Vp / background Vp/Vs"
                )
            )
        return fields, np.asarray(cov.values, dtype=bool), ones, findings
    if src in ("geology", "geology+wells"):
        surf = _surface_depth(grid, inputs)
        if src == "geology":
            ds, f = build_geology_model(
                um.horizons,
                grid,
                bgf,
                surfaces=inputs.surfaces,
                surface_depth_km=surf,
                base_dir=base,
            )
        else:
            ds, _fits, f = build_geology_wells_model(
                um.horizons,
                grid,
                um.wells,
                bgf,
                wells=inputs.wells,
                surfaces=inputs.surfaces,
                surface_depth_km=surf,
                base_dir=base,
            )
        findings += f
        unit = ds["unit_id"].values
        fields = {"vp": ds["vp"].values, "vs": ds["vs"].values, "unit_id": unit}
        return fields, unit >= 0, ones, findings
    if src == "wells":
        r = build_wells_model(um.wells, grid, None, wells=inputs.wells, base_dir=base)
        findings += r.findings
        w = r.dataset["weight"].values
        fields = {"vp": r.dataset["vp"].values, "vs": r.dataset["vs"].values}
        return fields, w > 0, w, findings
    raise GridError(f"unknown model3d.source {src!r}")


def _checks(
    grid: ModelGrid, ds: xr.Dataset, chk: Any, findings: list[GeologyFinding]
) -> dict[str, Any]:
    """Physical checks; append findings and return summary numbers."""
    vp, vs = ds["vp"].values, ds["vs"].values
    out: dict[str, Any] = {}
    n_nan = int((~np.isfinite(vp)).sum() + (~np.isfinite(vs)).sum())
    out["n_nan"] = n_nan
    if n_nan:
        findings.append(
            GeologyFinding(
                "error",
                "nan_velocity",
                f"{n_nan} Vp/Vs nodes are NaN (no background and no user-model coverage); "
                "the vgrids cannot be written",
                n_nan,
            )
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        low, high = vp < chk.vp_min_km_s, vp > chk.vp_max_km_s
        ratio = vp / vs
        rlow, rhigh = ratio < chk.vpvs_min, ratio > chk.vpvs_max
    for sel, code, what, arr in (
        (low | high, "vp_range", f"Vp outside [{chk.vp_min_km_s:g}, {chk.vp_max_km_s:g}] km/s", vp),
        (
            rlow | rhigh,
            "vpvs_range",
            f"Vp/Vs outside [{chk.vpvs_min:g}, {chk.vpvs_max:g}]",
            ratio,
        ),
    ):
        n = int(sel.sum())
        if n:
            lo, hi = float(np.nanmin(arr[sel])), float(np.nanmax(arr[sel]))
            findings.append(
                GeologyFinding(
                    "warning",
                    code,
                    f"{what} at {n} of {sel.size} nodes (offending values {lo:.3g}..{hi:.3g})",
                    n,
                )
            )
    out["n_vp_out_of_range"] = int((low | high).sum())
    out["n_vpvs_out_of_range"] = int((rlow | rhigh).sum())
    sp = _spacing(grid)
    g = np.sqrt(sum(np.nan_to_num(np.gradient(vp, s, axis=i)) ** 2 for i, s in enumerate(sp)))
    n_g = int((g > chk.max_gradient_km_s_per_km).sum())
    out["max_gradient_km_s_per_km"] = float(g.max())
    out["n_large_gradient"] = n_g
    if n_g:
        k = np.unravel_index(int(np.argmax(g)), g.shape)
        findings.append(
            GeologyFinding(
                "warning",
                "large_gradient",
                f"|grad Vp| > {chk.max_gradient_km_s_per_km:g} (km/s)/km at {n_g} nodes; max "
                f"{g.max():.2f} at depth {grid.depth_full[k[0]]:.2f} km, lat "
                f"{grid.lat_full[k[1]]:.4f}, lon {grid.lon_full[k[2]]:.4f}; sharp jumps "
                "can upset the eikonal solver",
                n_g,
            )
        )
    return out


def build_starting_model(
    grid: ModelGrid,
    cfg: Any,
    model_1d: pd.DataFrame | None,
    inputs: StartingInputs | None = None,
) -> StartingModel:
    """Build the 3-D starting model: user model blended into the 1-D background.

    Parameters
    ----------
    grid
        :class:`~osv.model3d.grid.ModelGrid`; all outputs are on its full grid.
    cfg
        ``Model3DConfig`` (or anything with a ``.model3d`` attribute, e.g. ``OSVConfig``).
    model_1d
        Stage 2 ``final_model.csv`` table (``depth_top_km``, ``vp``, ``vs``/``vpvs``); may be
        None only with ``user_model.background`` ``'constant'`` or ``'none'`` and a user source.
    inputs
        :class:`StartingInputs` (paths base directory, stations, preloaded data).

    Returns
    -------
    StartingModel
        ``dataset`` (vp, vs, vp_unc, vs_unc, weight, vp_bg, vs_bg[, unit_id]), ``findings``
        (physical checks and notes from the sources) and a ``summary`` dict.
    """
    m3 = getattr(cfg, "model3d", cfg)
    um = m3.user_model
    inputs = inputs or StartingInputs()
    sp = _spacing(grid)
    shape = grid.shape_full
    bg, findings = _background(grid, m3, model_1d)
    bg_vp, bg_vs = bg["vp"].values, bg["vs"].values

    fields: dict[str, np.ndarray] = {}
    if m3.source == "velest_1d":
        weight = np.zeros(shape)
        vp, vs = bg_vp.copy(), bg_vs.copy()
    else:
        fields, mask, own, f = _build_user(grid, m3, bg, inputs)
        findings += f
        weight = own * blend_weight(mask, sp, um.blend_taper_km)
        weight = np.where(mask, weight, 0.0)
        u_vp = fields["vp"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_bg = bg_vp / bg_vs
            u_vs = fields.get("vs")
            if u_vs is None:
                u_vs = u_vp / ratio_bg
            else:
                u_vs = np.where(np.isfinite(u_vs), u_vs, u_vp / ratio_bg)
        if um.smoothing_km > 0 and um.smoothing_target == "user":
            # smooth the user model alone (holes filled by the background before filtering)
            u_vp = _smooth(np.where(np.isfinite(u_vp), u_vp, bg_vp), sp, um.smoothing_km)
            u_vs = _smooth(np.where(np.isfinite(u_vs), u_vs, bg_vs), sp, um.smoothing_km)

        def blend(u: np.ndarray, b: np.ndarray) -> np.ndarray:
            ok_u, ok_b = np.isfinite(u), np.isfinite(b)
            w = np.where(ok_u, weight, 0.0)
            return np.where(
                ok_b, np.where(ok_u, w * np.nan_to_num(u) + (1 - w) * np.nan_to_num(b), b), u
            )

        vp, vs = blend(u_vp, bg_vp), blend(u_vs, bg_vs)
        # without a background the user value stands alone (effective weight 1)
        weight = np.where(np.isfinite(bg_vp) | ~np.isfinite(u_vp), weight, np.where(mask, 1.0, 0.0))
        if um.smoothing_km > 0 and um.smoothing_target == "blend":
            vp, vs = _smooth(vp, sp, um.smoothing_km), _smooth(vs, sp, um.smoothing_km)

    unc = {k: bg[k].values.copy() for k in ("vp_unc", "vs_unc")}
    if m3.uncertainty_from_weight and m3.source != "velest_1d":
        fac = 1.0 + (float(m3.uncertainty_low_weight_factor) - 1.0) * (1.0 - weight)
        cap = float(m3.extrusion.uncertainty.max_km_s)
        for k in unc:
            unc[k] = np.minimum(unc[k] * fac, np.maximum(cap, unc[k]))

    out = grid.template().drop_vars("velocity")
    for name, arr in (
        ("vp", vp),
        ("vs", vs),
        ("vp_unc", unc["vp_unc"]),
        ("vs_unc", unc["vs_unc"]),
    ):
        out[name] = (DIMS, arr, {"units": "km/s"})
    out["weight"] = (DIMS, weight, {"long_name": "user-model blending weight (0 = background)"})
    out["vp_bg"] = (DIMS, bg_vp, {"units": "km/s"})
    out["vs_bg"] = (DIMS, bg_vs, {"units": "km/s"})
    if "unit_id" in fields:
        out["unit_id"] = (DIMS, fields["unit_id"].astype(np.int32), {})
        out.attrs["unit_names"] = [str(h.name) for h in um.horizons]
    out.attrs.update(
        {
            "source": m3.source,
            "blend_taper_km": float(um.blend_taper_km),
            "smoothing_km": float(um.smoothing_km),
            "background": "velest_1d" if m3.source == "velest_1d" else um.background,
        }
    )

    summary = _checks(grid, out, m3.checks, findings)
    infl = weight > INFLUENCE_EPS
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = vp / vs
        dvp = np.abs(vp - bg_vp)
        dvs = np.abs(vs - bg_vs)
    summary.update(
        {
            "source": m3.source,
            "n_nodes": int(weight.size),
            "influenced_fraction": float(infl.mean()),
            "mean_weight": float(weight.mean()),
            "max_abs_dvp": float(np.nanmax(dvp)) if np.isfinite(dvp).any() else float("nan"),
            "median_abs_dvp": float(np.nanmedian(dvp[infl]))
            if (infl & np.isfinite(dvp)).any()
            else 0.0,
            "max_abs_dvs": float(np.nanmax(dvs)) if np.isfinite(dvs).any() else float("nan"),
            "vp_min": float(np.nanmin(vp)) if np.isfinite(vp).any() else float("nan"),
            "vp_max": float(np.nanmax(vp)) if np.isfinite(vp).any() else float("nan"),
            "vpvs_median": float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan"),
        }
    )
    if m3.source != "velest_1d":
        findings.append(
            GeologyFinding(
                "info",
                "user_influence",
                f"user model influences {100 * summary['influenced_fraction']:.1f}% of the grid; "
                f"|dVp| vs background: max {summary['max_abs_dvp']:.2f}, median "
                f"{summary['median_abs_dvp']:.2f} km/s",
                int(infl.sum()),
            )
        )
    for f in findings:
        if f.level in ("warning", "error"):
            log.warning("%s: %s", f.code, f.message)
    log.info("starting model (%s): %s", m3.source, summary)
    return StartingModel(grid, out, findings, summary)
