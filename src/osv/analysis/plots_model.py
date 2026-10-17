"""Stage 5 figures of the final model: depth slices and vertical cross-sections (task 6.2).

All functions return :class:`matplotlib.figure.Figure` s and work from plain data (model
datasets, 1-D reference, starting model, DWS coverage, event and station tables).  Quantities
(see :data:`osv.plotting.slices.QUANTITIES`): absolute ``vp``/``vs``, ``vpvs``, percentage change
against the 1-D model (``dvp_1d``, ``dvs_1d``) and against the starting model (``dvp_start``,
``dvs_start``), plus the starting model itself (``start_vp`` ...).

* Colour limits are shared by every panel of a quantity (symmetric about 0 for percentages),
  computed from the well-sampled nodes only, so slices and sections of one quantity agree.
* The coverage mask (``ModelMasks``: DWS above ``plotting.dws_mask_threshold``, optionally AND-ed
  with a Stage 6 *resolvable* mask) hides the rest in grey, translucent grey or hatching.
* Slices show seismicity within half a slab thickness of the depth; sections project events and
  stations inside a swath onto the profile.
* ``interpolation="linear"`` (default) is a display choice; ``"bspline"`` shows the model as
  ``fm3d`` evaluates it (cubic B-spline); a footer says which was used.

:func:`make_model_figures` builds the standard set and hands each figure to ``save``.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from osv.plotting.slices import (
    QUANTITIES,
    ModelFields,
    ModelMasks,
    colour_limits,
    default_profiles,
    draw_mask,
    project_swath,
)
from osv.plotting.style import (
    PERTURBATION_CMAP,
    add_scale_bar,
    apply_style,
    setup_map_axes,
    velocity_cmap,
)

log = logging.getLogger(__name__)

__all__ = [
    "ModelFigureOptions",
    "make_model_figures",
    "model_limits",
    "plot_section_comparison",
    "plot_sections",
    "plot_slice_matrix",
    "plot_slices",
]

Saver = Callable[..., Any]
Profile = Any  # ProfileConfig-like (name, start, end) or a (name, start, end) tuple


@dataclass
class ModelFigureOptions:
    """Appearance and sampling options of the model figures."""

    interpolation: str = "linear"
    #: ``"grey"``, ``"alpha"``, ``"hatch"`` or ``"none"`` (see ``draw_mask``).
    mask_style: str = "grey"
    mask_alpha: float = 0.65
    #: Contour levels of the DWS normalised to its maximum (0-1), labelled; ``None`` = none.
    #: The coverage-mask outline itself is never labelled.
    dws_contours: Sequence[float] | None = None
    clip_percentile: float = 99.0
    #: Display grid is this many times finer than the node grid.
    oversample: int = 2
    #: Slab thickness (km) for seismicity on slices; default one node spacing in depth.
    slab_thickness_km: float | None = None
    #: Half-width (km) of the swath around a profile.
    swath_halfwidth_km: float = 1.0
    #: Vertical exaggeration of sections (figure height follows from it).
    vertical_exaggeration: float = 1.5
    max_columns: int = 4
    abs_cmap: str | None = None
    ratio_cmap: str = "cividis"
    station_labels: bool = False
    #: Draw coastlines when cartopy and its data are available (silently skipped otherwise).
    coastline: bool = False
    #: Largest number of depth slices in one comparison matrix.
    max_matrix_rows: int = 4


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower() or "profile"


def _profile(p: Profile) -> tuple[str, tuple[float, float], tuple[float, float]]:
    if isinstance(p, tuple | list):
        name, s, e = p
    else:
        name, s, e = p.name, p.start, p.end
    return str(name), (float(s[0]), float(s[1])), (float(e[0]), float(e[1]))


def _cmap(key: str, opts: ModelFigureOptions):
    kind = QUANTITIES[key].kind
    if kind == "pct":
        return plt.get_cmap(PERTURBATION_CMAP)
    if kind == "ratio":
        return plt.get_cmap(opts.ratio_cmap)
    return velocity_cmap(opts.abs_cmap)


def _depth_of(df: pd.DataFrame) -> np.ndarray:
    for c in ("depth_km_bsl", "depth_km", "depth"):
        if c in df:
            return df[c].to_numpy(float)
    raise KeyError("events table needs a depth column (depth_km_bsl or depth_km)")


def _nodes(fields: ModelFields):
    return np.meshgrid(fields.depth, fields.lat, fields.lon, indexing="ij")


def model_limits(
    fields: ModelFields,
    keys: Sequence[str],
    masks: ModelMasks | None = None,
    opts: ModelFigureOptions | None = None,
) -> dict[str, tuple[float, float]]:
    """Colour limits per quantity from the covered interior nodes (symmetric for percent)."""
    opts = opts or ModelFigureOptions()
    d, la, lo = _nodes(fields)
    out: dict[str, tuple[float, float]] = {}
    for k in keys:
        q = QUANTITIES[k]
        v = fields.evaluate(k, d, la, lo)
        cov = masks.covered_at(q.phase, d, la, lo) if masks is not None else None
        out[k] = colour_limits([v], q.kind, opts.clip_percentile, [cov])
    return out


def _fine(a: np.ndarray, oversample: int) -> np.ndarray:
    n = max((a.size - 1) * max(int(oversample), 1) + 1, 2)
    return np.linspace(a[0], a[-1], n)


def _norm_kw(key: str, lim: tuple[float, float], opts: ModelFigureOptions) -> dict[str, Any]:
    kw: dict[str, Any] = {"cmap": _cmap(key, opts), "vmin": lim[0], "vmax": lim[1]}
    return kw


def _add_coastline(ax: Any) -> None:
    try:
        import cartopy.feature as cfeature

        xl, yl = ax.get_xlim(), ax.get_ylim()
        for geom in cfeature.COASTLINE.geometries():
            if hasattr(geom, "geoms"):
                parts = list(geom.geoms)
            else:
                parts = [geom]
            for g in parts:
                x, y = g.xy
                ax.plot(x, y, color="k", lw=0.8, zorder=3)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
    except Exception as exc:  # cartopy missing, no data (offline), ...
        log.debug("coastline skipped: %s", exc)


def _footer(fig: Figure, fields: ModelFields, masks: ModelMasks | None, extra: str = "") -> None:
    parts = [fields.note()]
    if masks is not None and masks.covered:
        parts.append(
            "Masked: DWS below threshold"
            + (" or not resolvable (Stage 6)." if masks.resolvable_applied else ".")
        )
    if extra:
        parts.append(extra)
    fig.supxlabel(" ".join(parts), fontsize=6.5, color="0.35")


# ----------------------------------------------------------------------------------------
# depth slices
# ----------------------------------------------------------------------------------------


def _slice_depths(fields: ModelFields, depths: Sequence[float]) -> list[float]:
    """Requested depths inside the model, snapped to nothing (evaluated exactly)."""
    half = 0.5 * float(np.median(np.diff(fields.depth))) if fields.depth.size > 1 else 0.0
    out: list[float] = []
    for d in depths:
        if fields.depth[0] - half <= d <= fields.depth[-1] + half and d not in out:
            out.append(float(d))
        elif not (fields.depth[0] - half <= d <= fields.depth[-1] + half):
            log.info("depth slice %.2f km is outside the model: skipped", d)
    return out


def _draw_slice(
    ax: Any,
    fields: ModelFields,
    key: str,
    depth: float,
    lim: tuple[float, float],
    masks: ModelMasks | None,
    opts: ModelFigureOptions,
    events: pd.DataFrame | None,
    stations: pd.DataFrame | None,
    slab: float,
) -> Any:
    q = QUANTITIES[key]
    lon, lat = _fine(fields.lon, opts.oversample), _fine(fields.lat, opts.oversample)
    LO, LA = np.meshgrid(lon, lat)
    val = fields.evaluate(key, depth, LA, LO)
    mesh = ax.pcolormesh(
        lon, lat, val, shading="nearest", rasterized=True, **_norm_kw(key, lim, opts)
    )
    if masks is not None:
        draw_mask(
            ax, lon, lat, masks.covered_at(q.phase, depth, LA, LO), opts.mask_style, opts.mask_alpha
        )
        if opts.dws_contours:
            dws = masks.dws_at(q.phase, depth, LA, LO, normalise=True)
            if dws is not None and np.nanmax(dws) > min(opts.dws_contours):
                cs = ax.contour(
                    lon,
                    lat,
                    dws,
                    levels=sorted(opts.dws_contours),
                    colors="k",
                    linewidths=0.6,
                    zorder=6,
                )
                ax.clabel(cs, fontsize=5, fmt="%.2g")
    if events is not None and len(events):
        z = _depth_of(events)
        sel = np.abs(z - depth) <= 0.5 * slab
        ax.scatter(
            events["lon"].to_numpy(float)[sel],
            events["lat"].to_numpy(float)[sel],
            s=9,
            c="k",
            edgecolors="white",
            linewidths=0.4,
            zorder=7,
        )
        ax.text(
            0.97,
            0.03,
            f"{int(sel.sum())} events",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            zorder=11,
            bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1},
        )
    if stations is not None and len(stations):
        ax.scatter(
            stations["lon"],
            stations["lat"],
            marker="^",
            s=14,
            c="white",
            edgecolors="k",
            linewidths=0.5,
            zorder=8,
        )
        if opts.station_labels and "sta" in stations:
            for x, y, t in zip(stations["lon"], stations["lat"], stations["sta"], strict=False):
                ax.annotate(
                    str(t), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=5, zorder=9
                )
    ax.set_xlim(lon[0], lon[-1])
    ax.set_ylim(lat[0], lat[-1])
    setup_map_axes(ax, float(fields.lat.mean()))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if opts.coastline:
        _add_coastline(ax)
    return mesh


def _slab(fields: ModelFields, opts: ModelFigureOptions) -> float:
    if opts.slab_thickness_km is not None:
        return float(opts.slab_thickness_km)
    return float(np.median(np.diff(fields.depth))) if fields.depth.size > 1 else 1.0


def plot_slice_matrix(
    fields: ModelFields,
    keys: Sequence[str],
    depths: Sequence[float],
    masks: ModelMasks | None = None,
    *,
    events: pd.DataFrame | None = None,
    stations: pd.DataFrame | None = None,
    opts: ModelFigureOptions | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
) -> Figure:
    """Map panels with one row per depth and one column per quantity (side-by-side compare).

    Use ``keys=("start_vp", "vp", "dvp_start")`` for starting vs final, ``("dvp_1d", "dvs_1d")``
    or ``("vp", "vs")`` for P vs S.  Each column has its own colour bar; limits are shared
    down the column.
    """
    opts = opts or ModelFigureOptions()
    apply_style()
    zs = _slice_depths(fields, depths)[: opts.max_matrix_rows]
    if not zs:
        raise ValueError("none of the requested depths lies inside the model")
    limits = limits or model_limits(fields, keys, masks, opts)
    nr, nc = len(zs), len(keys)
    width = 3.6
    asp = 1.0 / max(math.cos(math.radians(float(fields.lat.mean()))), 1e-3)
    ext_x = (fields.lon[-1] - fields.lon[0]) / asp
    ext_y = fields.lat[-1] - fields.lat[0]
    height = width * ext_y / ext_x
    fig, axs = plt.subplots(
        nr,
        nc,
        figsize=(width * nc + 0.8, height * nr + 0.9),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    slab = _slab(fields, opts)
    for j, k in enumerate(keys):
        mesh = None
        for i, z in enumerate(zs):
            ax = axs[i, j]
            mesh = _draw_slice(ax, fields, k, z, limits[k], masks, opts, events, stations, slab)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i == 0:
                ax.set_title(QUANTITIES[k].title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"z = {z:g} km\nLat (deg)", fontsize=8)
            if i == nr - 1:
                ax.set_xlabel("Lon (deg)", fontsize=8)
        cb = fig.colorbar(
            mesh,
            ax=axs[:, j],
            label=QUANTITIES[k].label,
            location="bottom",
            shrink=0.8,
            pad=0.03,
            aspect=30,
        )
        cb.ax.tick_params(labelsize=6)
    add_scale_bar(axs[-1, 0], lat0=float(fields.lat.mean()))
    if title:
        fig.suptitle(title, fontsize=10)
    _footer(fig, fields, masks, f"Events within +/-{slab / 2:g} km of each depth.")
    return fig


def plot_slices(
    fields: ModelFields,
    key: str,
    depths: Sequence[float],
    masks: ModelMasks | None = None,
    *,
    events: pd.DataFrame | None = None,
    stations: pd.DataFrame | None = None,
    opts: ModelFigureOptions | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
) -> Figure:
    """Depth-slice panels of one quantity with one shared colour bar.

    Parameters
    ----------
    fields
        :class:`~osv.plotting.slices.ModelFields` (model, 1-D reference, starting model).
    key
        Quantity key, e.g. ``"vp"``, ``"dvp_1d"``, ``"dvp_start"``, ``"vpvs"``.
    depths
        Slice depths (km, positive down); those outside the model are skipped.
    masks
        Optional :class:`~osv.plotting.slices.ModelMasks` hiding poorly sampled model.
    """
    opts = opts or ModelFigureOptions()
    apply_style()
    zs = _slice_depths(fields, depths)
    if not zs:
        raise ValueError("none of the requested depths lies inside the model")
    limits = limits or model_limits(fields, [key], masks, opts)
    n = len(zs)
    ncol = min(opts.max_columns, n)
    nrow = math.ceil(n / ncol)
    asp = 1.0 / max(math.cos(math.radians(float(fields.lat.mean()))), 1e-3)
    ext_x = (fields.lon[-1] - fields.lon[0]) / asp
    ext_y = fields.lat[-1] - fields.lat[0]
    width = 3.3
    height = width * ext_y / ext_x
    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(width * ncol + 1.0, (height + 0.35) * nrow + 0.8),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    slab = _slab(fields, opts)
    mesh = None
    for ax, z in zip(axs.ravel(), zs, strict=False):
        mesh = _draw_slice(ax, fields, key, z, limits[key], masks, opts, events, stations, slab)
        ax.set_title(f"{z:g} km", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axs[:, 0]:
        ax.set_ylabel("Lat (deg)", fontsize=8)
    for ax in axs[-1, :]:
        ax.set_xlabel("Lon (deg)", fontsize=8)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    cb = fig.colorbar(mesh, ax=axs, label=QUANTITIES[key].label, shrink=0.8)
    cb.ax.tick_params(labelsize=7)
    add_scale_bar(axs[-1, 0], lat0=float(fields.lat.mean()))
    fig.suptitle(title or f"{QUANTITIES[key].title}: depth slices", fontsize=10)
    _footer(fig, fields, masks, f"Events within +/-{slab / 2:g} km of each depth.")
    return fig


# ----------------------------------------------------------------------------------------
# cross-sections
# ----------------------------------------------------------------------------------------


def _section_grid(fields: ModelFields, start, end, opts: ModelFigureOptions):
    _, _, length = project_swath(*end, start, end)
    dx = min(
        abs(fields.lon[1] - fields.lon[0]) * 111.195 * math.cos(math.radians(start[0])),
        abs(fields.lat[1] - fields.lat[0]) * 111.195,
    )
    n = max(int(math.ceil(length / (dx / max(opts.oversample, 1)))) + 1, 2)
    t = np.linspace(0, 1, n)
    lat = start[0] + t * (end[0] - start[0])
    lon = start[1] + t * (end[1] - start[1])
    depth = _fine(fields.depth, opts.oversample)
    return t * length, depth, lat, lon, length


def _draw_section(
    ax: Any,
    fields: ModelFields,
    key: str,
    prof: tuple[str, tuple[float, float], tuple[float, float]],
    lim: tuple[float, float],
    masks: ModelMasks | None,
    opts: ModelFigureOptions,
    events: pd.DataFrame | None,
    stations: pd.DataFrame | None,
    topography: xr.DataArray | None,
) -> Any:
    _, start, end = prof
    q = QUANTITIES[key]
    dist, depth, lat, lon, length = _section_grid(fields, start, end, opts)
    D, LA, LO = np.broadcast_arrays(depth[:, None], lat[None, :], lon[None, :])
    val = fields.evaluate(key, D, LA, LO)
    surf = None
    if topography is not None:
        surf = (
            -np.asarray(
                topography.interp(lat=("p", lat), lon=("p", lon), method="linear").values, float
            )
            / 1000.0
        )
        val = np.where(D < surf[None, :], np.nan, val)
    mesh = ax.pcolormesh(
        dist, depth, val, shading="nearest", rasterized=True, **_norm_kw(key, lim, opts)
    )
    if masks is not None:
        cov = masks.covered_at(q.phase, D, LA, LO)
        draw_mask(ax, dist, depth, cov, opts.mask_style, opts.mask_alpha)
        if opts.dws_contours:
            dws = masks.dws_at(q.phase, D, LA, LO, normalise=True)
            if dws is not None and np.nanmax(dws) > min(opts.dws_contours):
                cs = ax.contour(
                    dist,
                    depth,
                    dws,
                    levels=sorted(opts.dws_contours),
                    colors="k",
                    linewidths=0.6,
                    zorder=6,
                )
                ax.clabel(cs, fontsize=5, fmt="%.2g")
    if surf is not None:
        ax.plot(dist, surf, "k-", lw=1.2, zorder=8, label="topography")
    else:
        ax.axhline(0.0, color="k", lw=0.8, ls="--", zorder=8, label="sea level")
    if stations is not None and len(stations):
        s, c, _ = project_swath(stations["lat"], stations["lon"], start, end)
        sel = (c <= opts.swath_halfwidth_km) & (s >= 0) & (s <= length)
        if "elevation_m" in stations or "sensor_elevation_m" in stations:
            col = "sensor_elevation_m" if "sensor_elevation_m" in stations else "elevation_m"
            z = -stations[col].to_numpy(float) / 1000.0
        else:
            z = np.zeros(len(stations))
        ax.scatter(
            s[sel],
            z[sel],
            marker="^",
            s=24,
            c="white",
            edgecolors="k",
            linewidths=0.6,
            zorder=9,
            label="stations",
            clip_on=False,
        )
    if events is not None and len(events):
        s, c, _ = project_swath(events["lat"], events["lon"], start, end)
        sel = (c <= opts.swath_halfwidth_km) & (s >= 0) & (s <= length)
        ax.scatter(
            s[sel],
            _depth_of(events)[sel],
            s=6,
            c="k",
            edgecolors="white",
            linewidths=0.2,
            alpha=0.8,
            zorder=7,
            label=f"events (n={int(sel.sum())})",
        )
    ax.set_xlim(0, length)
    ax.set_ylim(depth[-1], min(depth[0], -0.5 if surf is not None else depth[0]))
    return mesh


def _section_size(fields: ModelFields, length: float, opts: ModelFigureOptions, n: int):
    zr = float(fields.depth[-1] - fields.depth[0])
    width = 8.0
    h = width * zr * opts.vertical_exaggeration / max(length, 1e-6)
    h = float(np.clip(h, 1.6, 4.0))
    return width, h * n + 0.9


def plot_sections(
    fields: ModelFields,
    keys: Sequence[str],
    profile: Profile,
    masks: ModelMasks | None = None,
    *,
    events: pd.DataFrame | None = None,
    stations: pd.DataFrame | None = None,
    topography: xr.DataArray | None = None,
    opts: ModelFigureOptions | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
) -> Figure:
    """Vertical cross-section along ``profile``: one stacked panel per quantity.

    ``profile`` is a config ``ProfileConfig`` or ``(name, (lat, lon), (lat, lon))``.  Events and
    stations within ``opts.swath_halfwidth_km`` of the line are projected onto it; the surface is
    ``topography`` (elevation in m, dims ``lat``/``lon``) or sea level; masked as for slices.
    """
    opts = opts or ModelFigureOptions()
    apply_style()
    prof = _profile(profile)
    name, start, end = prof
    limits = limits or model_limits(fields, keys, masks, opts)
    _, _, length = project_swath(*end, start, end)
    w, h = _section_size(fields, length, opts, len(keys))
    fig, axs = plt.subplots(
        len(keys), 1, figsize=(w, h), squeeze=False, constrained_layout=True, sharex=True
    )
    for ax, k in zip(axs[:, 0], keys, strict=True):
        mesh = _draw_section(
            ax, fields, k, prof, limits[k], masks, opts, events, stations, topography
        )
        cb = fig.colorbar(mesh, ax=ax, label=QUANTITIES[k].label, pad=0.01)
        cb.ax.tick_params(labelsize=7)
        ax.set_ylabel("Depth (km)")
        ax.text(
            0.01,
            0.04,
            QUANTITIES[k].title,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            zorder=11,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.5},
        )
    axs[0, 0].legend(loc="upper right", ncol=3, fontsize=6.5, framealpha=0.85).set_zorder(12)
    axs[-1, 0].set_xlabel(f"Distance along {name} (km)")
    fig.suptitle(
        title
        or (
            f"Section {name}: ({start[0]:.3f}, {start[1]:.3f}) to ({end[0]:.3f}, {end[1]:.3f}); "
            f"swath +/-{opts.swath_halfwidth_km:g} km; VE {opts.vertical_exaggeration:g}x"
        ),
        fontsize=9,
    )
    _footer(fig, fields, masks)
    return fig


def plot_section_comparison(
    fields: ModelFields,
    profile: Profile,
    masks: ModelMasks | None = None,
    *,
    quantity: str = "vp",
    events: pd.DataFrame | None = None,
    stations: pd.DataFrame | None = None,
    topography: xr.DataArray | None = None,
    opts: ModelFigureOptions | None = None,
) -> Figure:
    """Starting model, final model and their percentage difference along one profile."""
    suffix = quantity[-2:]  # 'vp' / 'vs'
    if suffix not in ("vp", "vs"):
        raise ValueError("quantity must be 'vp' or 'vs'")
    keys = [f"start_{suffix}", suffix, f"d{suffix}_start"]
    opts = opts or ModelFigureOptions()
    lim = model_limits(fields, keys, masks, opts)
    # starting and final share one absolute scale
    lo = min(lim[keys[0]][0], lim[keys[1]][0])
    hi = max(lim[keys[0]][1], lim[keys[1]][1])
    lim[keys[0]] = lim[keys[1]] = (lo, hi)
    return plot_sections(
        fields,
        keys,
        profile,
        masks,
        events=events,
        stations=stations,
        topography=topography,
        opts=opts,
        limits=lim,
    )


# ----------------------------------------------------------------------------------------
# all figures
# ----------------------------------------------------------------------------------------


def make_model_figures(
    save: Saver,
    model: xr.Dataset,
    reference_1d: Any,
    starting: xr.Dataset | None,
    coverage: Any,
    events: pd.DataFrame | None,
    stations: pd.DataFrame | None,
    cfg: Any,
    resolvable_mask: Any = None,
    *,
    options: ModelFigureOptions | None = None,
    topography: xr.DataArray | None = None,
) -> list[str]:
    """Build the standard model figures and pass each to ``save(fig, name, caption, section)``.

    Parameters
    ----------
    save
        ``FigureRegistry.save``-like callable (it closes the figure).
    model, reference_1d, starting
        Final model dataset, 1-D reference (see :class:`~osv.plotting.slices.ModelFields`) and
        starting model (``None`` disables the percentage-vs-start and comparison figures).
    coverage
        DWS (``DataArray`` / ``Dataset`` / mapping, see :meth:`ModelMasks.from_coverage`), or
        ``None`` for no masking.
    events, stations
        Tables with ``lat``, ``lon`` and (events) ``depth_km_bsl`` or ``depth_km``.
    cfg
        Project config (uses ``cfg.plotting.depth_slices_km``, ``profiles`` and
        ``dws_mask_threshold``).
    resolvable_mask
        Optional Stage 6 mask (boolean); adds ``*_resolvable`` figures masked by DWS AND resolvable.

    Returns
    -------
    list of str
        Names of the figures saved.
    """
    opts = options or ModelFigureOptions()
    pc = cfg.plotting
    fields = ModelFields(model, reference_1d, starting, opts.interpolation)
    avail = fields.available()
    depths = list(pc.depth_slices_km)
    profiles = [_profile(p) for p in pc.profiles] or default_profiles(fields.lat, fields.lon)
    base = (
        ModelMasks.from_coverage(coverage, pc.dws_mask_threshold) if coverage is not None else None
    )
    variants: list[tuple[str, ModelMasks | None, str]] = [("", base, "")]
    if resolvable_mask is not None:
        res = (
            ModelMasks.from_coverage(coverage, pc.dws_mask_threshold, resolvable_mask)
            if coverage is not None
            else ModelMasks.from_coverage(None, 0.0, resolvable_mask)
        )
        variants.append(("_resolvable", res, " Masked by DWS and the Stage 6 resolvable mask."))
    saved: list[str] = []

    def emit(fig: Figure, name: str, caption: str) -> None:
        save(fig, name, caption, "model")
        saved.append(name)

    kw = {"events": events, "stations": stations, "opts": opts}
    for suffix, masks, cap_extra in variants:
        slice_keys = [
            k
            for k in ("vp", "vs", "dvp_1d", "dvs_1d", "dvp_start", "dvs_start", "vpvs")
            if k in avail
        ]
        lims = model_limits(fields, slice_keys, masks, opts)
        for k in slice_keys:
            emit(
                plot_slices(fields, k, depths, masks, limits=lims, **kw),
                f"slices_{k}{suffix}",
                f"{QUANTITIES[k].title} at depth slices; uncovered model masked, "
                "events within the slab, stations as triangles." + cap_extra,
            )
        for k in ("vp", "vs"):
            keys = [kk for kk in (f"start_{k}", k, f"d{k}_start") if kk in avail]
            if len(keys) == 3:
                lm = {**model_limits(fields, keys, masks, opts)}
                lo, hi = min(lm[keys[0]][0], lm[keys[1]][0]), max(lm[keys[0]][1], lm[keys[1]][1])
                lm[keys[0]] = lm[keys[1]] = (lo, hi)
                emit(
                    plot_slice_matrix(
                        fields,
                        keys,
                        depths,
                        masks,
                        limits=lm,
                        title=f"Starting vs final {k.upper()}",
                        **kw,
                    ),
                    f"compare_start_final_{k}{suffix}",
                    f"Starting model, final model and percentage change for {k.upper()}."
                    + cap_extra,
                )
        for pair, nm in (
            (("dvp_1d", "dvs_1d"), "dv_1d"),
            (("dvp_start", "dvs_start"), "dv_start"),
            (("vp", "vs"), "vp_vs"),
        ):
            if all(x in avail for x in pair):
                lm = model_limits(fields, pair, masks, opts)
                if QUANTITIES[pair[0]].kind == "pct":
                    m = max(lm[pair[0]][1], lm[pair[1]][1])
                    lm = {pair[0]: (-m, m), pair[1]: (-m, m)}
                emit(
                    plot_slice_matrix(fields, pair, depths, masks, limits=lm, title="P vs S", **kw),
                    f"compare_p_s_{nm}{suffix}",
                    "P and S side by side (shared symmetric scale for percentages)." + cap_extra,
                )
        sec_sets = [("", [k for k in ("vp", "dvp_1d", "dvp_start", "vpvs") if k in avail])]
        sec_sets.append(("_s", [k for k in ("vs", "dvs_1d") if k in avail]))
        slims = model_limits(fields, [k for _, ks in sec_sets for k in ks], masks, opts)
        for name, st, en in profiles:
            for sfx, ks in sec_sets:
                if not ks:
                    continue
                emit(
                    plot_sections(
                        fields, ks, (name, st, en), masks, topography=topography, limits=slims, **kw
                    ),
                    f"section_{_slug(name)}{sfx}{suffix}",
                    f"Cross-section {name}: "
                    + ", ".join(QUANTITIES[k].title for k in ks)
                    + "."
                    + cap_extra,
                )
            if suffix == "" and "start_vp" in avail:
                emit(
                    plot_section_comparison(
                        fields, (name, st, en), masks, topography=topography, **kw
                    ),
                    f"section_{_slug(name)}_compare",
                    f"Cross-section {name}: starting Vp, final Vp and change.",
                )
    return saved
