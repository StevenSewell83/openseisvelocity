"""Stage 3 diagnostic figures (task 4.10).

Every function returns a :class:`matplotlib.figure.Figure` and works from plain data (the
starting-model dataset, the :class:`~osv.model3d.grid.ModelGrid`, the interfaces, the station and
event tables and the forward-check residuals), so figures can be regenerated from the files the
stage wrote.  :func:`make_all_figures` creates all that the inputs allow and hands them to a
``save(fig, name, caption, section)`` callable (``ctx.figures.save``).

Conventions: maps are lon/lat with the ``1/cos(lat)`` aspect; depths are km below sea level,
positive down; dVp is ``100 * (Vp - Vp_background) / Vp_background`` in percent; the model
region is the grid *interior* (cushion nodes are not drawn).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator

from osv.model3d.grid import KM_PER_DEG, ModelGrid
from osv.model3d.interfaces import interface_depth_at
from osv.plotting.style import (
    PERTURBATION_CMAP,
    add_scale_bar,
    apply_style,
    perturbation_norm,
    plot_events,
    plot_stations,
    setup_map_axes,
    velocity_cmap,
)

log = logging.getLogger(__name__)

__all__ = [
    "depth_slice_figure",
    "gradient_map",
    "grid_geometry",
    "horizon_figure",
    "make_all_figures",
    "residual_histograms",
    "residual_vs_distance",
    "section_figure",
    "station_residual_map",
    "vp_histogram",
    "weight_map",
    "well_profiles",
]

#: Most panels drawn in one slice figure.
MAX_SLICES = 8
#: Columns of the slice figures.
SLICE_COLUMNS = 4
#: Half-width (km) of the band around a section within which stations and events are drawn.
SECTION_BAND_KM = 1.5
#: Sections are sampled at this fraction of the smaller node spacing.
SECTION_SAMPLING = 0.5
#: Percentile clip of the dVp colour scale.
DVP_CLIP_PERCENTILE = 99.0
#: Weights above this count as "user model applies" in the weight map.
WEIGHT_APPLIED = 0.999

Saver = Callable[..., Any]


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def _interior(da: Any, grid: ModelGrid) -> np.ndarray:
    c = grid.n_cushion
    return np.asarray(da.values if hasattr(da, "values") else da, dtype=float)[c:-c, c:-c, c:-c]


def _dvp(ds: Any, grid: ModelGrid) -> np.ndarray:
    vp, bg = _interior(ds["vp"], grid), _interior(ds["vp_bg"], grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * (vp - bg) / bg


def _differs(ds: Any) -> bool:
    return bool(np.nanmax(np.abs(ds["vp"].values - ds["vp_bg"].values)) > 1e-9)


def _extent(grid: ModelGrid) -> tuple[float, float, float, float]:
    return float(grid.lon[0]), float(grid.lon[-1]), float(grid.lat[0]), float(grid.lat[-1])


def _map_axes(ax: Any, grid: ModelGrid) -> None:
    x0, x1, y0, y1 = _extent(grid)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    setup_map_axes(ax, grid.centre_lat)


def _stations_xy(stations: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    if stations is None or not len(stations):
        return np.array([]), np.array([])
    return stations["lon"].to_numpy(float), stations["lat"].to_numpy(float)


def _local_km(lat: Any, lon: Any, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """East/north km of points relative to ``(lat0, lon0)`` (FMTOMO sphere)."""
    y = (np.asarray(lat, float) - lat0) * KM_PER_DEG
    x = (np.asarray(lon, float) - lon0) * KM_PER_DEG * np.cos(np.deg2rad(lat0))
    return x, y


def _colorbar(fig: Figure, mappable: Any, ax: Any, label: str, **kw: Any) -> Any:
    cb = fig.colorbar(mappable, ax=ax, label=label, **kw)
    cb.ax.tick_params(labelsize=7)
    return cb


def _nearest_index(values: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(values - x)))


def _interface_at(interfaces: Any, grid: ModelGrid, lat: Any, lon: Any, which: int) -> np.ndarray:
    """Interface depth with the points clipped to the range ``fm3d`` can interpolate.

    The B-spline is defined from the first interior node up to (not including) the last one.
    """
    eps = 1e-4
    la = np.clip(
        np.asarray(lat, float),
        grid.lat[0] + eps * grid.dlat_deg,
        grid.lat[-1] - eps * grid.dlat_deg,
    )
    lo = np.clip(
        np.asarray(lon, float),
        grid.lon[0] + eps * grid.dlon_deg,
        grid.lon[-1] - eps * grid.dlon_deg,
    )
    return interface_depth_at(interfaces, grid, la, lo, which)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "profile"


def _grid_interp(ds: Any, name: str, grid: ModelGrid, *, fill: float = np.nan) -> Any:
    return RegularGridInterpolator(
        (grid.depth_full, grid.lat_full, grid.lon_full),
        np.asarray(ds[name].values, dtype=float),
        bounds_error=False,
        fill_value=fill,
    )


# ----------------------------------------------------------------------------------------
# depth slices
# ----------------------------------------------------------------------------------------


def depth_slice_figure(
    ds: Any,
    grid: ModelGrid,
    depths_km: Sequence[float],
    *,
    perturbation: bool = False,
    stations: pd.DataFrame | None = None,
) -> Figure:
    """Map views of Vp (``perturbation=False``) or dVp against the 1-D background (percent).

    Slices are taken at the interior node nearest to each requested depth; depths outside the
    model are skipped (at most :data:`MAX_SLICES` panels).
    """
    apply_style()
    idx: list[int] = []
    for d in depths_km:
        if grid.depth[0] - grid.dz_km / 2 <= d <= grid.depth[-1] + grid.dz_km / 2:
            k = _nearest_index(grid.depth, d)
            if k not in idx:
                idx.append(k)
    idx = idx[:MAX_SLICES] or [0]
    data = _dvp(ds, grid) if perturbation else _interior(ds["vp"], grid)
    n = len(idx)
    ncol = min(SLICE_COLUMNS, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(3.4 * ncol + 1.0, 3.2 * nrow + 0.6),
        squeeze=False,
        constrained_layout=True,
    )
    if perturbation:
        lim = (
            float(np.nanpercentile(np.abs(data), DVP_CLIP_PERCENTILE))
            if np.isfinite(data).any()
            else 1.0
        )
        lim = max(lim, 1e-6)
        kw = {"cmap": PERTURBATION_CMAP, "norm": perturbation_norm(vmax=lim)}
        label = "dVp (%)"
    else:
        kw = {
            "cmap": velocity_cmap(),
            "vmin": float(np.nanmin(data)),
            "vmax": float(np.nanmax(data)),
        }
        label = "Vp (km/s)"
    sx, sy = _stations_xy(stations)
    mesh = None
    for ax, k in zip(axs.ravel(), idx, strict=False):
        mesh = ax.pcolormesh(grid.lon, grid.lat, data[k], shading="nearest", **kw)
        if sx.size:
            ax.scatter(sx, sy, marker="^", s=8, c="white", edgecolors="k", linewidths=0.4, zorder=5)
        ax.set_title(f"depth {grid.depth[k]:.2f} km", fontsize=9)
        _map_axes(ax, grid)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    if mesh is not None:
        _colorbar(fig, mesh, axs, label, shrink=0.8)
    if perturbation and not _differs(ds):
        fig.suptitle(
            "dVp vs 1-D background: no user model, identical to the background", fontsize=10
        )
    return fig


# ----------------------------------------------------------------------------------------
# cross sections
# ----------------------------------------------------------------------------------------


def section_figure(
    ds: Any,
    grid: ModelGrid,
    interfaces: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "A-A'",
    stations: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    band_km: float = SECTION_BAND_KM,
) -> Figure:
    """Vertical section from ``start`` to ``end`` (``(lat, lon)``): Vp and, for a user model, dVp.

    Stations and events within ``band_km`` of the line are projected on it; the ground surface
    (interface 1) is drawn and nodes above it are masked.
    """
    apply_style()
    lat0, lon0 = start
    x1, y1 = _local_km(end[0], end[1], lat0, lon0)
    length = float(np.hypot(x1, y1))
    step = SECTION_SAMPLING * min(grid.dx_km, grid.dy_km)
    n = max(int(np.ceil(length / step)) + 1, 2)
    t = np.linspace(0, 1, n)
    lat = lat0 + t * (end[0] - lat0)
    lon = lon0 + t * (end[1] - lon0)
    dist = t * length
    depth = grid.depth
    pts = np.stack(
        np.broadcast_arrays(depth[:, None], lat[None, :], lon[None, :]), axis=-1
    ).reshape(-1, 3)
    vp = _grid_interp(ds, "vp", grid)(pts).reshape(depth.size, n)
    surf = _interface_at(interfaces, grid, lat, lon, 0)
    base = _interface_at(interfaces, grid, lat, lon, 1)
    mask = depth[:, None] < surf[None, :]
    rows = [("Vp (km/s)", np.ma.array(vp, mask=mask), "vp")]
    if _differs(ds):
        bg = _grid_interp(ds, "vp_bg", grid)(pts).reshape(depth.size, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(("dVp (%)", np.ma.array(100 * (vp - bg) / bg, mask=mask), "dvp"))
    fig, axs = plt.subplots(
        len(rows),
        1,
        figsize=(8.0, 3.4 * len(rows) + 0.6),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
    )
    ux, uy = (x1 / length, y1 / length) if length > 0 else (1.0, 0.0)

    def project(la: Any, lo: Any) -> tuple[np.ndarray, np.ndarray]:
        px, py = _local_km(la, lo, lat0, lon0)
        return px * ux + py * uy, np.abs(-px * uy + py * ux)

    for ax, (name, arr, kind) in zip(axs[:, 0], rows, strict=True):
        if kind == "vp":
            m = ax.pcolormesh(dist, depth, arr, shading="nearest", cmap=velocity_cmap())
        else:
            lim = max(float(np.nanpercentile(np.abs(arr.compressed()), DVP_CLIP_PERCENTILE)), 1e-6)
            m = ax.pcolormesh(
                dist,
                depth,
                arr,
                shading="nearest",
                cmap=PERTURBATION_CMAP,
                norm=perturbation_norm(vmax=lim),
            )
        _colorbar(fig, m, ax, name)
        ax.plot(dist, surf, "k-", lw=1.2, label="surface")
        ax.plot(dist, base, "k--", lw=0.8, label="base")
        if stations is not None and len(stations):
            s, c = project(stations["lat"], stations["lon"])
            sel = (c <= band_km) & (s >= 0) & (s <= length)
            z = (
                -stations["sensor_elevation_m"].to_numpy(float) / 1000
                if ("sensor_elevation_m" in stations)
                else -stations["elevation_m"].to_numpy(float) / 1000
            )
            ax.scatter(
                s[sel],
                z[sel],
                **{
                    "marker": "^",
                    "s": 30,
                    "c": "white",
                    "edgecolors": "k",
                    "linewidths": 0.7,
                    "zorder": 6,
                    "label": "stations",
                },
            )
        if events is not None and len(events):
            s, c = project(events["lat"], events["lon"])
            sel = (c <= band_km) & (s >= 0) & (s <= length)
            ax.scatter(
                s[sel],
                events["depth_km_bsl"].to_numpy(float)[sel],
                s=9,
                c="red",
                edgecolors="k",
                linewidths=0.2,
                zorder=5,
                label="events",
            )
        ax.set_ylim(depth[-1], depth[0])
        ax.set_ylabel("Depth (km)")
    axs[-1, 0].set_xlabel(f"Distance along {label} (km)")
    axs[0, 0].legend(loc="lower right", ncol=4, fontsize=7)
    axs[0, 0].set_title(
        f"Section {label}: ({lat0:.3f}, {lon0:.3f}) to ({end[0]:.3f}, {end[1]:.3f}), "
        f"events/stations within {band_km:g} km",
        fontsize=9,
    )
    return fig


# ----------------------------------------------------------------------------------------
# wells and geology
# ----------------------------------------------------------------------------------------


def well_profiles(ds: Any, grid: ModelGrid, wells: Sequence[Any]) -> Figure:
    """Vp against depth at each well: the model, the 1-D background and the conditioned log."""
    apply_style()
    n = len(wells)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(2.9 * ncol + 0.5, 4.2 * nrow),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    f_vp, f_bg = _grid_interp(ds, "vp", grid), _grid_interp(ds, "vp_bg", grid)
    z = grid.depth
    for ax, w in zip(axs.ravel(), wells, strict=False):
        pts = np.stack([z, np.full_like(z, w.lat), np.full_like(z, w.lon)], axis=-1)
        ax.plot(np.asarray(w.vp), np.asarray(w.depth_km), color="0.55", lw=0.9, label="log (Vp)")
        ax.step(
            f_bg(pts), z, where="mid", color="tab:blue", lw=1.0, ls="--", label="1-D background"
        )
        ax.plot(f_vp(pts), z, color="tab:red", lw=1.6, label="starting model")
        ax.set_title(str(w.name), fontsize=9)
        ax.set_xlabel("Vp (km/s)")
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    axs[0, 0].set_ylim(z[-1], z[0])
    for r in range(nrow):
        axs[r, 0].set_ylabel("Depth (km)")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    return fig


def horizon_figure(ds: Any, grid: ModelGrid) -> Figure:
    """Base depth of each geological unit (maps) and the unit-id volume (two sections)."""
    apply_style()
    unit = _interior(ds["unit_id"], grid).astype(int)
    names = [str(x) for x in ds.attrs.get("unit_names", [])]
    n_units = int(unit.max()) + 1 if (unit >= 0).any() else 0
    z = grid.depth
    ncol = max(n_units, 2)
    fig = plt.figure(figsize=(3.6 * ncol + 1.0, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.8])
    mesh = None
    for i in range(n_units):
        ax = fig.add_subplot(gs[0, i])
        inside = (unit >= 0) & (unit <= i)
        base = np.where(
            inside.any(axis=0), np.where(inside, z[:, None, None], -np.inf).max(axis=0), np.nan
        )
        base = base + 0.5 * grid.dz_km
        mesh = ax.pcolormesh(grid.lon, grid.lat, base, shading="nearest", cmap="cividis_r")
        _colorbar(fig, mesh, ax, "base depth (km)", shrink=0.8)
        ax.set_title(f"base of {names[i] if i < len(names) else f'unit {i}'}", fontsize=9)
        _map_axes(ax, grid)
        ax.tick_params(labelsize=7)
    cmap = plt.get_cmap("tab10", max(n_units + 1, 2))
    ilat, ilon = grid.lat.size // 2, grid.lon.size // 2
    for j, (title, sec, ax_x, xlab) in enumerate(
        (
            ("E-W through the centre", unit[:, ilat, :], grid.lon, "Longitude (deg)"),
            ("N-S through the centre", unit[:, :, ilon], grid.lat, "Latitude (deg)"),
        )
    ):
        ax = fig.add_subplot(gs[1, : ncol // 2] if j == 0 else gs[1, ncol // 2 :])
        ax.pcolormesh(
            ax_x,
            z,
            np.where(sec >= 0, sec, np.nan),
            shading="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=max(n_units, 1) + 0.5,
        )
        ax.set_ylim(z[-1], z[0])
        ax.set_title(f"unit id, {title} (white = background)", fontsize=9)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Depth (km)")
    return fig


# ----------------------------------------------------------------------------------------
# weight, histogram, gradient
# ----------------------------------------------------------------------------------------


def weight_map(ds: Any, grid: ModelGrid, stations: pd.DataFrame | None = None) -> Figure:
    """Where the user model applies: maximum weight over depth (map) and a section."""
    apply_style()
    w = _interior(ds["weight"], grid)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    m = axs[0].pcolormesh(
        grid.lon, grid.lat, np.nanmax(w, axis=0), shading="nearest", cmap="magma_r", vmin=0, vmax=1
    )
    _colorbar(fig, m, axs[0], "max weight over depth")
    sx, sy = _stations_xy(stations)
    if sx.size:
        axs[0].scatter(sx, sy, marker="^", s=8, c="white", edgecolors="k", linewidths=0.4)
    _map_axes(axs[0], grid)
    frac = float((w > WEIGHT_APPLIED).mean())
    infl = float((w > 1e-6).mean())
    axs[0].set_title("user-model weight (1 = user model, 0 = 1-D background)", fontsize=9)
    ilat = grid.lat.size // 2
    m2 = axs[1].pcolormesh(
        grid.lon, grid.depth, w[:, ilat, :], shading="nearest", cmap="magma_r", vmin=0, vmax=1
    )
    axs[1].set_ylim(grid.depth[-1], grid.depth[0])
    axs[1].set_xlabel("Longitude (deg)")
    axs[1].set_ylabel("Depth (km)")
    axs[1].set_title(
        f"E-W section at lat {grid.lat[ilat]:.3f}; full weight {100 * frac:.1f}% of nodes, "
        f"any influence {100 * infl:.1f}%",
        fontsize=9,
    )
    _colorbar(fig, m2, axs[1], "weight")
    if infl == 0:
        fig.suptitle("velest_1d source: the 1-D model applies everywhere (weight 0)", fontsize=10)
    return fig


def vp_histogram(ds: Any, grid: ModelGrid, checks: Any = None) -> Figure:
    """Histograms of Vp (model and background) and Vp/Vs over the interior nodes."""
    apply_style()
    vp, bg = _interior(ds["vp"], grid).ravel(), _interior(ds["vp_bg"], grid).ravel()
    vs = _interior(ds["vs"], grid).ravel()
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    vp, bg = vp[np.isfinite(vp)], bg[np.isfinite(bg)]
    bins = np.linspace(
        min(vp.min(), bg.min() if bg.size else vp.min()),
        max(vp.max(), bg.max() if bg.size else vp.max()),
        60,
    )
    axs[0].hist(vp, bins=bins, color="tab:red", alpha=0.7, label="starting model")
    if bg.size and _differs(ds):
        axs[0].hist(
            bg, bins=bins, histtype="step", color="tab:blue", lw=1.4, label="1-D background"
        )
    if checks is not None:
        for v in (checks.vp_min_km_s, checks.vp_max_km_s):
            if bins[0] <= v <= bins[-1]:
                axs[0].axvline(v, color="k", ls=":", lw=1)
    axs[0].set(xlabel="Vp (km/s)", ylabel="Nodes", title=f"Vp: {vp.min():.2f}-{vp.max():.2f} km/s")
    axs[0].legend()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = _interior(ds["vp"], grid).ravel() / vs
    ratio = ratio[np.isfinite(ratio)]
    pad = max(0.02 * float(np.ptp(ratio)), 0.01) if ratio.size else 0.1
    axs[1].hist(
        ratio, bins=60, range=(ratio.min() - pad, ratio.max() + pad), color="tab:green", alpha=0.8
    )
    if checks is not None:
        for v in (checks.vpvs_min, checks.vpvs_max):
            if ratio.size and ratio.min() - 0.1 <= v <= ratio.max() + 0.1:
                axs[1].axvline(v, color="k", ls=":", lw=1)
    axs[1].set(xlabel="Vp/Vs", ylabel="Nodes", title=f"median Vp/Vs {np.median(ratio):.2f}")
    return fig


def gradient_map(ds: Any, grid: ModelGrid, checks: Any = None) -> Figure:
    """Largest ``|grad Vp|`` over depth (map) and its maximum per depth (profile)."""
    apply_style()
    vp = np.asarray(ds["vp"].values, dtype=float)
    sp = (grid.dz_km, grid.dy_km, grid.dx_km)
    g = np.sqrt(sum(np.nan_to_num(np.gradient(vp, s, axis=i)) ** 2 for i, s in enumerate(sp)))
    c = grid.n_cushion
    g = g[c:-c, c:-c, c:-c]
    thr = float(getattr(checks, "max_gradient_km_s_per_km", np.nan))
    fig, axs = plt.subplots(
        1, 2, figsize=(10.5, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]}
    )
    m = axs[0].pcolormesh(grid.lon, grid.lat, g.max(axis=0), shading="nearest", cmap="inferno")
    _colorbar(fig, m, axs[0], "max |grad Vp| over depth ((km/s)/km)")
    _map_axes(axs[0], grid)
    axs[0].set_title(f"velocity-gradient magnitude, max {g.max():.2f} (km/s)/km", fontsize=9)
    gmax = g.reshape(g.shape[0], -1).max(axis=1)
    axs[1].plot(gmax, grid.depth, "k-")
    if np.isfinite(thr):
        axs[1].axvline(thr, color="r", ls="--", label=f"limit {thr:g}")
        axs[1].legend()
    axs[1].set_ylim(grid.depth[-1], grid.depth[0])
    axs[1].set(xlabel="max |grad Vp| ((km/s)/km)", ylabel="Depth (km)")
    return fig


# ----------------------------------------------------------------------------------------
# grid geometry
# ----------------------------------------------------------------------------------------


def grid_geometry(
    grid: ModelGrid,
    interfaces: Any,
    stations: pd.DataFrame | None,
    events: pd.DataFrame | None,
) -> Figure:
    """Velocity and propagation grid bounds against stations and events (map and sections)."""
    apply_style()
    pg = grid.propgrid
    fig = plt.figure(figsize=(11.5, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    lon_f, lat_f = grid.lon_full, grid.lat_full
    ax.add_patch(
        Rectangle(
            (lon_f[0], lat_f[0]),
            lon_f[-1] - lon_f[0],
            lat_f[-1] - lat_f[0],
            fill=False,
            ec="0.5",
            ls=":",
            label="velocity grid incl. cushion",
        )
    )
    ax.add_patch(
        Rectangle(
            (grid.lon[0], grid.lat[0]),
            grid.lon[-1] - grid.lon[0],
            grid.lat[-1] - grid.lat[0],
            fill=False,
            ec="tab:blue",
            lw=1.4,
            label="model interior",
        )
    )
    plon0, plat0 = pg.lon0_deg, pg.lat0_deg
    ax.add_patch(
        Rectangle(
            (plon0, plat0),
            (pg.nlon - 1) * pg.dlon_deg,
            (pg.nlat - 1) * pg.dlat_deg,
            fill=False,
            ec="tab:green",
            lw=1.4,
            ls="--",
            label="propagation grid",
        )
    )
    ax.scatter(*np.meshgrid(grid.lon, grid.lat), s=1.5, c="0.7", zorder=1, label="velocity nodes")
    sc = None
    if events is not None and len(events):
        sc = plot_events(
            ax, events["lon"], events["lat"], events["depth_km_bsl"], zorder=3, label="events"
        )
    if stations is not None and len(stations):
        plot_stations(ax, stations["lon"], stations["lat"])
    if sc is not None:
        _colorbar(fig, sc, ax, "event depth (km)", shrink=0.7)
    ax.set_xlim(lon_f[0] - 0.02 * (lon_f[-1] - lon_f[0]), lon_f[-1] + 0.02 * (lon_f[-1] - lon_f[0]))
    ax.set_ylim(lat_f[0] - 0.02 * (lat_f[-1] - lat_f[0]), lat_f[-1] + 0.02 * (lat_f[-1] - lat_f[0]))
    setup_map_axes(ax, grid.centre_lat)
    add_scale_bar(ax)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        f"grid {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} nodes, spacing "
        f"{grid.dx_km:.2f} x {grid.dy_km:.2f} x {grid.dz_km:.2f} km; propagation grid "
        f"{pg.nr}x{pg.nlat}x{pg.nlon} (dr {pg.dr_km:.3f} km)",
        fontsize=8,
    )
    z0, z1 = float(pg.depth_km[0]), float(pg.depth_km[-1])
    for row, (xs, _ys, xlab, lo, hi, key) in enumerate(
        (
            ("lon", "lat", "Longitude (deg)", plon0, plon0 + (pg.nlon - 1) * pg.dlon_deg, "lon"),
            ("lat", "lon", "Latitude (deg)", plat0, plat0 + (pg.nlat - 1) * pg.dlat_deg, "lat"),
        )
    ):
        a = fig.add_subplot(gs[row, 1])
        axis = grid.lon if key == "lon" else grid.lat
        centre = grid.centre_lat if key == "lon" else grid.centre_lon
        la = np.full_like(axis, centre) if key == "lon" else axis
        lo_ = axis if key == "lon" else np.full_like(axis, centre)
        a.plot(axis, _interface_at(interfaces, grid, la, lo_, 0), "k-", label="surface")
        a.plot(axis, _interface_at(interfaces, grid, la, lo_, 1), "k--", label="base")
        a.add_patch(Rectangle((lo, z0), hi - lo, z1 - z0, fill=False, ec="tab:green", ls="--"))
        a.add_patch(
            Rectangle(
                (axis[0], grid.depth[0]),
                axis[-1] - axis[0],
                grid.depth[-1] - grid.depth[0],
                fill=False,
                ec="tab:blue",
            )
        )
        if events is not None and len(events):
            a.scatter(events[xs], events["depth_km_bsl"], s=6, c="red", zorder=3, label="events")
        if stations is not None and len(stations):
            elev = "sensor_elevation_m" if "sensor_elevation_m" in stations else "elevation_m"
            a.scatter(
                stations[xs],
                -stations[elev] / 1000,
                marker="^",
                s=18,
                c="white",
                edgecolors="k",
                linewidths=0.6,
                zorder=4,
                label="stations",
            )
        a.set_ylim(z1 + grid.dz_km, z0 - grid.dz_km)
        a.set_xlabel(xlab)
        a.set_ylabel("Depth (km)")
        a.grid(alpha=0.3)
        if row == 0:
            a.legend(fontsize=7, ncol=4, loc="lower center")
            a.set_title(
                "projection on longitude / latitude (surface at the grid centre line)", fontsize=8
            )
    return fig


# ----------------------------------------------------------------------------------------
# residuals
# ----------------------------------------------------------------------------------------


def _phases(res: pd.DataFrame) -> list[str]:
    return [p for p in ("P", "S") if (res["phase"] == p).any()]


def residual_histograms(res: pd.DataFrame, *, background: bool = True) -> Figure:
    """Histograms of the residuals (observed - calculated) of the 1-D background and the model."""
    apply_style()
    phases = _phases(res)
    fig, axs = plt.subplots(
        1,
        max(len(phases), 1),
        figsize=(5.2 * max(len(phases), 1), 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, ph in zip(axs[0], phases, strict=False):
        g = res[res["phase"] == ph]
        cols = [
            ("res_start", "starting model" if background else "starting model (= 1-D)", "tab:red")
        ]
        if background:
            cols.insert(0, ("res_bg", "1-D background", "tab:blue"))
        allr = np.concatenate([g[c].dropna().to_numpy() for c, _, _ in cols])
        lim = float(np.nanpercentile(np.abs(allr), 99.5)) if allr.size else 1.0
        bins = np.linspace(-lim, lim, 61)
        for i, (c, name, colr) in enumerate(cols):
            r = g[c].dropna().to_numpy()
            rms = float(np.sqrt(np.mean(r**2))) if r.size else np.nan
            ax.hist(
                r,
                bins=bins,
                histtype="stepfilled" if i else "step",
                alpha=0.55 if i else 1.0,
                color=colr,
                lw=1.4,
                label=f"{name}: RMS {rms:.3f} s, mean {np.mean(r):+.3f} s",
            )
        ax.axvline(0, color="k", lw=0.6)
        ax.set(
            xlabel="residual: observed - calculated (s)",
            ylabel="Picks",
            title=f"{ph} ({len(g)} picks)",
        )
        ax.legend(fontsize=7)
    return fig


def residual_vs_distance(res: pd.DataFrame, *, background: bool = True) -> Figure:
    """Residual against hypocentral distance, with a running median, per phase."""
    apply_style()
    phases = _phases(res)
    fig, axs = plt.subplots(
        1,
        max(len(phases), 1),
        figsize=(5.6 * max(len(phases), 1), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, ph in zip(axs[0], phases, strict=False):
        g = res[res["phase"] == ph]
        cols = [("res_start", "starting model", "tab:red")]
        if background:
            cols.insert(0, ("res_bg", "1-D background", "tab:blue"))
        for c, name, colr in cols:
            ax.scatter(g["dist_km"], g[c], s=5, alpha=0.35, color=colr, label=name)
            ok = g[c].notna()
            if ok.sum() > 10:
                edges = np.quantile(g.loc[ok, "dist_km"], np.linspace(0, 1, 9))
                cut = pd.cut(g.loc[ok, "dist_km"], np.unique(edges), include_lowest=True)
                med = (
                    g.loc[ok]
                    .groupby(cut, observed=True)
                    .agg(d=("dist_km", "median"), r=(c, "median"))
                )
                ax.plot(med["d"], med["r"], "-o", color=colr, ms=4, lw=1.6, mec="k", mew=0.4)
        ax.axhline(0, color="k", lw=0.6)
        ax.set(
            xlabel="Hypocentral distance (km)",
            ylabel="Residual (s)",
            title=f"{ph} residual vs distance",
        )
        ax.legend(fontsize=7, markerscale=2)
    return fig


def station_residual_map(
    res: pd.DataFrame, grid: ModelGrid, stations: pd.DataFrame | None = None
) -> Figure:
    """Mean residual of the starting model per station (colour) and pick count (size)."""
    apply_style()
    phases = _phases(res)
    fig, axs = plt.subplots(
        1,
        max(len(phases), 1),
        figsize=(5.6 * max(len(phases), 1) + 1.2, 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, ph in zip(axs[0], phases, strict=False):
        g = res[res["phase"] == ph]
        s = g.groupby("station").agg(
            lat=("sta_lat", "first"),
            lon=("sta_lon", "first"),
            mean=("res_start", "mean"),
            n=("res_start", "count"),
        )
        if stations is not None and len(stations):
            ax.scatter(stations["lon"], stations["lat"], marker="^", s=8, c="0.8", zorder=1)
        lim = max(float(np.nanpercentile(np.abs(s["mean"]), 98)) if len(s) else 0.1, 1e-3)
        m = ax.scatter(
            s["lon"],
            s["lat"],
            c=s["mean"],
            s=25 + 8 * np.sqrt(s["n"]),
            cmap=PERTURBATION_CMAP,
            norm=perturbation_norm(vmax=lim),
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
        )
        for st, r in s.iterrows():
            ax.annotate(
                str(st), (r["lon"], r["lat"]), xytext=(3, 3), textcoords="offset points", fontsize=5
            )
        _colorbar(fig, m, ax, "mean (obs - calc) (s)", shrink=0.8)
        _map_axes(ax, grid)
        ax.set_title(f"{ph}: mean residual per station (size ~ picks)", fontsize=9)
    return fig


# ----------------------------------------------------------------------------------------
# all figures
# ----------------------------------------------------------------------------------------


def make_all_figures(
    save: Saver | None = None,
    *,
    ds: Any,
    grid: ModelGrid,
    interfaces: Any,
    stations: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    residuals: pd.DataFrame | None = None,
    wells: Sequence[Any] | None = None,
    profiles: Sequence[Any] = (),
    depth_slices_km: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    checks: Any = None,
) -> dict[str, Figure]:
    """Create every figure the inputs allow.

    Parameters
    ----------
    save
        ``save(fig, name, caption, section)`` (e.g. ``ctx.figures.save``), which also closes the
        figure; with ``None`` the figures are returned open.
    ds
        Starting-model dataset (``vp``, ``vs``, ``vp_bg``, ``weight`` and optional ``unit_id``).
    grid, interfaces
        Model grid and interfaces.
    stations, events
        Tables with ``lat``, ``lon`` (``sensor_elevation_m``/``elevation_m`` for stations,
        ``depth_km_bsl`` for events).
    residuals
        ``ForwardCheck.residuals``.
    wells
        Processed well profiles (``WellProfile``), if any.
    profiles
        Config profiles (objects with ``name``, ``start``, ``end``).
    depth_slices_km
        Slice depths (``plotting.depth_slices_km``).
    checks
        ``ModelChecksConfig`` for the threshold lines.

    Returns
    -------
    dict
        ``{name: Figure}`` of the figures made (empty values when saved and closed).
    """
    out: dict[str, Figure] = {}

    def emit(name: str, caption: str, section: str, make: Callable[[], Figure]) -> None:
        try:
            fig = make()
        except Exception:  # a failing figure must not lose the others
            log.warning("figure %s failed", name, exc_info=True)
            return
        if save is None:
            out[name] = fig
        else:
            save(fig, name, caption, section)
            out[name] = fig

    differs = _differs(ds)
    emit(
        "vp_slices",
        "Vp (km/s) in horizontal slices of the starting model; triangles are stations.",
        "Starting model",
        lambda: depth_slice_figure(ds, grid, depth_slices_km, stations=stations),
    )
    emit(
        "dvp_slices",
        "dVp (%) of the starting model against the 1-D background in "
        "horizontal slices." + ("" if differs else " Identical here: no user model."),
        "Starting model",
        lambda: depth_slice_figure(ds, grid, depth_slices_km, perturbation=True, stations=stations),
    )
    lat_c, lon_c = grid.centre_lat, grid.centre_lon
    sec_args = {"stations": stations, "events": events}
    emit(
        "section_ew",
        "East-west section through the grid centre (Vp, and dVp for a user model).",
        "Starting model",
        lambda: section_figure(
            ds,
            grid,
            interfaces,
            (lat_c, grid.lon[0]),
            (lat_c, grid.lon[-1]),
            label="E-W",
            **sec_args,
        ),
    )
    emit(
        "section_ns",
        "North-south section through the grid centre (Vp, and dVp for a user model).",
        "Starting model",
        lambda: section_figure(
            ds,
            grid,
            interfaces,
            (grid.lat[0], lon_c),
            (grid.lat[-1], lon_c),
            label="N-S",
            **sec_args,
        ),
    )
    for p in profiles:
        emit(
            f"section_{_slug(p.name)}",
            f"Section {p.name} from the configured profile.",
            "Starting model",
            lambda p=p: section_figure(
                ds, grid, interfaces, tuple(p.start), tuple(p.end), label=p.name, **sec_args
            ),
        )
    if wells:
        emit(
            "well_profiles",
            "Vp at each well: conditioned log, 1-D background and starting model.",
            "Geology and wells",
            lambda: well_profiles(ds, grid, wells),
        )
    if "unit_id" in ds:
        emit(
            "horizons",
            "Base depth of each geological unit and the unit-id volume.",
            "Geology and wells",
            lambda: horizon_figure(ds, grid),
        )
    emit(
        "weight_map",
        "Where the user model applies (weight 1) and where it is tapered into the "
        "1-D background (0 < weight < 1).",
        "Quality control",
        lambda: weight_map(ds, grid, stations),
    )
    emit(
        "vp_histogram",
        "Distribution of node velocities and Vp/Vs; dotted lines are the check limits.",
        "Quality control",
        lambda: vp_histogram(ds, grid, checks),
    )
    emit(
        "gradient_map",
        "Velocity-gradient magnitude; sharp jumps upset the eikonal solver.",
        "Quality control",
        lambda: gradient_map(ds, grid, checks),
    )
    emit(
        "grid_geometry",
        "Velocity and propagation grids against stations and events (map and "
        "projections), with the surface and base interfaces.",
        "Grid",
        lambda: grid_geometry(grid, interfaces, stations, events),
    )
    if residuals is not None and len(residuals):
        emit(
            "residual_histograms",
            "Starting residuals (observed - calculated): 1-D background "
            "against the starting model.",
            "Forward check",
            lambda: residual_histograms(residuals, background=differs),
        )
        emit(
            "residual_distance",
            "Starting residuals against hypocentral distance.",
            "Forward check",
            lambda: residual_vs_distance(residuals, background=differs),
        )
        emit(
            "station_residuals",
            "Mean starting residual per station.",
            "Forward check",
            lambda: station_residual_map(residuals, grid, stations),
        )
    return out
