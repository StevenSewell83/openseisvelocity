"""Stage 6 figures: true vs recovered models, recovery maps, resolved scale, smearing, noise.

All functions take plain xarray/pandas objects and return :class:`matplotlib.figure.Figure`;
:func:`make_all_figures` builds the whole set and hands each figure to a
``save(fig, name, caption, section)`` callable (``FigureRegistry.save``).

Fields are on the full grid ``(depth, lat, lon)`` of :class:`~osv.model3d.grid.ModelGrid`
(cushion nodes are cropped before plotting).  Perturbations are percent of the background.

Figures (PLAN.md Stage 6)
    * ``true_vs_recovered_slices_<test>`` / ``true_vs_recovered_sections_<test>``: the true and
      recovered perturbation side by side, one symmetric colour scale (set by the true model).
    * ``checkerboard_matrix_<phase>``: recovered pattern, rows = cell sizes, columns = depths
      (zero contour of the true pattern overlaid).
    * ``recovery_corr_<phase>`` / ``recovery_amp_<phase>``: local correlation and amplitude
      recovery of the same matrix.
    * ``resolved_scale_<phase>``: smallest checkerboard cell size that passes the thresholds, in
      depth slices and a section.
    * ``smearing_<test>``: spike centroid offsets and principal axes on a map and a section.
    * ``checkerboard_anisotropy_<phase>``: per-axis wavenumber retention of the checkerboards.
    * ``noise_realisation_<phase>``: noise of the synthetic data vs the pick uncertainties.
    * ``lcurve_synthetic_vs_real``: optional, when synthetic trade-off curves are supplied.
    * ``summary_metrics``: global correlation, semblance and amplitude of every test.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from osv.plotting.style import PERTURBATION_CMAP, new_figure, perturbation_norm, setup_map_axes

log = logging.getLogger(__name__)

__all__ = [
    "crop",
    "make_all_figures",
    "pick_depths",
    "plot_anisotropy",
    "plot_checkerboard_matrix",
    "plot_noise",
    "plot_recovery_matrix",
    "plot_resolved_scale",
    "plot_smearing",
    "plot_summary_metrics",
    "plot_synthetic_lcurve",
    "plot_true_vs_recovered_sections",
    "plot_true_vs_recovered_slices",
]

KM_PER_DEG = 111.194926645
#: depths shown per figure at most
MAX_DEPTHS = 5


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def crop(da: xr.DataArray, grid: Any) -> xr.DataArray:
    """Drop the cushion nodes of a full-grid DataArray."""
    c = int(grid.n_cushion)
    if c <= 0:
        return da
    return da.isel(depth=slice(c, -c), lat=slice(c, -c), lon=slice(c, -c))


def _field(values: Any, grid: Any, name: str = "f") -> xr.DataArray:
    da = xr.DataArray(
        np.asarray(values, float),
        dims=("depth", "lat", "lon"),
        coords={"depth": grid.depth_full, "lat": grid.lat_full, "lon": grid.lon_full},
        name=name,
    )
    return crop(da, grid)


def pick_depths(grid: Any, wanted: Sequence[float] | None, n: int = MAX_DEPTHS) -> list[float]:
    """Node depths nearest to ``wanted`` (within the interior grid), at most ``n``, unique."""
    z = np.asarray(grid.depth, float)
    if not wanted:
        idx = np.unique(np.round(np.linspace(0, len(z) - 1, n)).astype(int))
        return [float(z[i]) for i in idx]
    inside = [float(w) for w in wanted if z[0] - 0.5 * grid.dz_km <= w <= z[-1] + 0.5 * grid.dz_km]
    if not inside:
        return pick_depths(grid, None, n)
    if len(inside) > n:
        inside = [
            inside[i] for i in np.unique(np.round(np.linspace(0, len(inside) - 1, n)).astype(int))
        ]
    out: list[float] = []
    for w in inside:
        v = float(z[int(np.argmin(np.abs(z - w)))])
        if v not in out:
            out.append(v)
    return out


def _vmax(*arrs: Any) -> float:
    m = 0.0
    for a in arrs:
        v = np.asarray(a, float)
        v = v[np.isfinite(v)]
        if v.size:
            m = max(m, float(np.abs(v).max()))
    return m or 1.0


def _map(
    ax: Any,
    da2d: xr.DataArray,
    *,
    cmap: Any,
    norm: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    kw: dict[str, Any] = {"cmap": cmap, "shading": "nearest"}
    if norm is not None:
        kw["norm"] = norm
    else:
        kw.update(vmin=vmin, vmax=vmax)
    m = ax.pcolormesh(da2d["lon"].values, da2d["lat"].values, da2d.values, **kw)
    setup_map_axes(ax, float(np.mean(da2d["lat"].values)))
    return m


def _label_depth(ax: Any, z: float, loc: str = "upper left") -> None:
    ax.text(
        0.03,
        0.96,
        f"{z:g} km",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"fc": "white", "alpha": 0.7, "lw": 0, "pad": 1.5},
    )


def _default_profiles(grid: Any) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    lat = np.asarray(grid.lat, float)
    lon = np.asarray(grid.lon, float)
    clat, clon = float(lat.mean()), float(lon.mean())
    return [
        ("W-E", (clat, float(lon[0])), (clat, float(lon[-1]))),
        ("S-N", (float(lat[0]), clon), (float(lat[-1]), clon)),
    ]


def _profile_tuple(p: Any) -> tuple[str, tuple[float, float], tuple[float, float]]:
    if isinstance(p, tuple):
        return p
    g = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
    return str(g("name")), tuple(map(float, g("start"))), tuple(map(float, g("end")))  # type: ignore[return-value]


def _section(
    da: xr.DataArray, start: tuple[float, float], end: tuple[float, float], n: int = 120
) -> tuple[np.ndarray, xr.DataArray]:
    """Linear interpolation of ``da`` along a lat/lon line -> (distance km, (depth, n) array)."""
    t = np.linspace(0.0, 1.0, n)
    lat = start[0] + t * (end[0] - start[0])
    lon = start[1] + t * (end[1] - start[1])
    lat = np.clip(lat, da["lat"].values.min(), da["lat"].values.max())
    lon = np.clip(lon, da["lon"].values.min(), da["lon"].values.max())
    out = da.interp(
        lat=xr.DataArray(lat, dims="s"), lon=xr.DataArray(lon, dims="s"), method="linear"
    )
    dx = (end[1] - start[1]) * KM_PER_DEG * math.cos(math.radians(0.5 * (start[0] + end[0])))
    dy = (end[0] - start[0]) * KM_PER_DEG
    return t * math.hypot(dx, dy), out


def _empty(msg: str) -> Figure:
    fig, ax = new_figure(figsize=(5, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


# ---------------------------------------------------------------------------------------
# true vs recovered
# ---------------------------------------------------------------------------------------
def plot_true_vs_recovered_slices(
    true: Any, rec: Any, grid: Any, depths: Sequence[float], *, title: str = ""
) -> Figure:
    """Depth slices of the true and recovered perturbation (percent), one symmetric scale."""
    t, r = _field(true, grid), _field(rec, grid)
    zs = list(depths)
    norm = perturbation_norm(vmax=_vmax(t.values))
    fig, axes = new_figure(
        nrows=len(zs), ncols=2, figsize=(7.2, 2.6 * len(zs) + 0.6), squeeze=False
    )
    mesh = None
    for i, z in enumerate(zs):
        for j, (f, lab) in enumerate(((t, "true"), (r, "recovered"))):
            ax = axes[i, j]
            mesh = _map(ax, f.sel(depth=z, method="nearest"), cmap=PERTURBATION_CMAP, norm=norm)
            _label_depth(ax, z)
            if i == 0:
                ax.set_title(lab)
            if j:
                ax.set_ylabel("")
    fig.colorbar(mesh, ax=axes, shrink=0.7, label="perturbation (%)")
    if title:
        fig.suptitle(title)
    return fig


def plot_true_vs_recovered_sections(
    true: Any,
    rec: Any,
    grid: Any,
    profiles: Sequence[Any] | None = None,
    *,
    title: str = "",
) -> Figure:
    """Vertical sections (configured profiles, else W-E and S-N through the centre)."""
    t, r = _field(true, grid), _field(rec, grid)
    profs = [_profile_tuple(p) for p in (profiles or _default_profiles(grid))]
    norm = perturbation_norm(vmax=_vmax(t.values))
    fig, axes = new_figure(
        nrows=len(profs), ncols=2, figsize=(9.0, 3.0 * len(profs) + 0.6), squeeze=False
    )
    mesh = None
    for i, (name, a, b) in enumerate(profs):
        for j, (f, lab) in enumerate(((t, "true"), (r, "recovered"))):
            ax = axes[i, j]
            dist, sec = _section(f, a, b)
            mesh = ax.pcolormesh(
                dist, sec["depth"].values, sec.values, cmap=PERTURBATION_CMAP, norm=norm,
                shading="nearest",
            )  # fmt: skip
            ax.invert_yaxis()
            ax.set_title(f"{lab}  {name}", fontsize=9)
            ax.set_xlabel("distance along profile (km)")
            if j == 0:
                ax.set_ylabel("depth (km)")
    fig.colorbar(mesh, ax=axes, shrink=0.7, label="perturbation (%)")
    if title:
        fig.suptitle(title)
    return fig


# ---------------------------------------------------------------------------------------
# matrices (checkerboard, recovery maps)
# ---------------------------------------------------------------------------------------
def _matrix(
    rows: Sequence[tuple[str, Any]],
    grid: Any,
    depths: Sequence[float],
    *,
    cmap: Any,
    norm: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    contours: Sequence[Any] | None = None,
    title: str = "",
) -> Figure:
    zs = list(depths)
    fig, axes = new_figure(
        nrows=len(rows),
        ncols=len(zs),
        figsize=(2.6 * len(zs) + 1.0, 2.5 * len(rows) + 0.7),
        squeeze=False,
    )
    mesh = None
    for i, (lab, vals) in enumerate(rows):
        f = _field(vals, grid)
        for j, z in enumerate(zs):
            ax = axes[i, j]
            mesh = _map(
                ax, f.sel(depth=z, method="nearest"), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax
            )
            if contours is not None and contours[i] is not None:
                c = _field(contours[i], grid).sel(depth=z, method="nearest")
                if np.nanmax(np.abs(c.values)) > 0:
                    ax.contour(
                        c["lon"], c["lat"], c.values, levels=[0.0], colors="k", linewidths=0.4
                    )
            if i == 0:
                ax.set_title(f"{z:g} km", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{lab}\nLatitude (deg)")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if i < len(rows) - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)
    fig.colorbar(mesh, ax=axes, shrink=0.7, label=cbar_label)
    if title:
        fig.suptitle(title)
    return fig


def plot_checkerboard_matrix(
    cases: Sequence[tuple[str, Any, Any]], grid: Any, depths: Sequence[float], *, title: str = ""
) -> Figure:
    """Recovered checkerboards: rows = ``(label, true, recovered)``, columns = depths."""
    vmax = _vmax(*(c[1] for c in cases))
    return _matrix(
        [(lab, rec) for lab, _, rec in cases],
        grid,
        depths,
        cmap=PERTURBATION_CMAP,
        norm=perturbation_norm(vmax=vmax),
        cbar_label="recovered perturbation (%)",
        contours=[t for _, t, _ in cases],
        title=title,
    )


def plot_recovery_matrix(
    rows: Sequence[tuple[str, Any]],
    grid: Any,
    depths: Sequence[float],
    quantity: str = "corr",
    *,
    title: str = "",
) -> Figure:
    """Local recovery maps (``corr`` or ``amp``), rows = cell sizes, columns = depths."""
    if quantity == "corr":
        kw: dict[str, Any] = {"vmin": -1.0, "vmax": 1.0, "cbar_label": "local correlation"}
        cmap = "RdBu_r"
    else:
        kw = {"vmin": 0.0, "vmax": 1.5, "cbar_label": "amplitude recovery (1 = full)"}
        cmap = "PuOr"
    return _matrix(rows, grid, depths, cmap=cmap, title=title, **kw)


def plot_resolved_scale(
    scale: Any,
    grid: Any,
    depths: Sequence[float],
    cell_sizes: Sequence[float],
    profiles: Sequence[Any] | None = None,
    *,
    title: str = "",
) -> Figure:
    """``resolved_scale_km`` in depth slices (top row) and sections (bottom row)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    sizes = sorted({float(c) for c in cell_sizes})
    f = _field(scale, grid)
    if not sizes:
        return _empty("no checkerboard sizes")
    cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(sizes)))
    cmap = ListedColormap(cols).with_extremes(bad="0.85")
    edges = [sizes[0] - 0.5 * (sizes[1] - sizes[0] if len(sizes) > 1 else 1.0)]
    edges += [0.5 * (a + b) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
    edges += [sizes[-1] + 0.5 * (sizes[-1] - sizes[-2] if len(sizes) > 1 else 1.0)]
    norm = BoundaryNorm(edges, cmap.N)
    zs = list(depths)
    profs = [_profile_tuple(p) for p in (profiles or _default_profiles(grid))][:2]
    ncol = max(len(zs), len(profs))
    fig, axes = new_figure(nrows=2, ncols=ncol, figsize=(2.8 * ncol + 1.0, 6.0), squeeze=False)
    mesh = None
    for j in range(ncol):
        ax = axes[0, j]
        if j < len(zs):
            mesh = _map(ax, f.sel(depth=zs[j], method="nearest"), cmap=cmap, norm=norm)
            ax.set_title(f"{zs[j]:g} km", fontsize=9)
        else:
            ax.set_axis_off()
        ax = axes[1, j]
        if j < len(profs):
            name, a, b = profs[j]
            dist, sec = _section(f, a, b)
            mesh = ax.pcolormesh(
                dist, sec["depth"].values, sec.values, cmap=cmap, norm=norm, shading="nearest"
            )
            ax.invert_yaxis()
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("distance (km)")
            ax.set_ylabel("depth (km)")
        else:
            ax.set_axis_off()
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes, shrink=0.7, ticks=sizes, label="resolved scale (km)")
        cb.ax.set_yticklabels([f"{s:g}" for s in sizes])
    fig.text(0.01, 0.01, "grey: not resolved at any tested scale", fontsize=7)
    if title:
        fig.suptitle(title)
    return fig


# ---------------------------------------------------------------------------------------
# smearing
# ---------------------------------------------------------------------------------------
def _spike_glyphs(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for s in summary.get("spikes", []) or []:
        c = np.asarray(s.get("true_centre_km", [np.nan] * 3), float)
        off = np.asarray(s.get("centroid_offset_km", [np.nan] * 3), float)
        sg = np.asarray(s.get("rec_sigma_km", [np.nan] * 3), float)
        if not np.isfinite(c).all():
            continue
        out.append(
            {
                "centre": c,
                "offset": off,
                "sigma": sg,
                "az": float(s.get("major_azimuth_deg", math.nan)),
                "plunge": float(s.get("major_plunge_deg", math.nan)),
                "peak_ratio": float(s.get("peak_ratio", math.nan)),
            }
        )
    return out


def plot_smearing(summary: Mapping[str, Any], *, title: str = "") -> Figure:
    """Spike smearing glyphs on a map (east, north) and a section (east, depth).

    Open circle = true spike, filled dot = recovered centroid (arrow = offset), bar = principal
    (major) axis of the recovered spread (length ``2 sigma``).
    """
    gl = _spike_glyphs(summary)
    if not gl:
        return _empty("no spike was recovered")
    fig, (am, asec) = new_figure(ncols=2, figsize=(10.0, 4.6))
    for g in gl:
        cx, cy, cz = g["centre"]
        o = np.nan_to_num(g["offset"])
        rx, ry, rz = cx + o[0], cy + o[1], cz + o[2]
        am.plot(cx, cy, "o", mfc="none", mec="k", ms=9)
        asec.plot(cx, cz, "o", mfc="none", mec="k", ms=9)
        am.annotate("", (rx, ry), (cx, cy), arrowprops={"arrowstyle": "->", "color": "C3"})
        asec.annotate("", (rx, rz), (cx, cz), arrowprops={"arrowstyle": "->", "color": "C3"})
        am.plot(rx, ry, "o", color="C3", ms=4)
        asec.plot(rx, rz, "o", color="C3", ms=4)
        sg = g["sigma"]
        if np.isfinite(g["az"]) and np.isfinite(sg[0]):
            az, pl = math.radians(g["az"]), math.radians(g["plunge"])
            half = sg[0]
            hx, hy = half * math.cos(pl) * math.sin(az), half * math.cos(pl) * math.cos(az)
            am.plot([rx - hx, rx + hx], [ry - hy, ry + hy], "-", color="C0", lw=3, alpha=0.8)
            asec.plot(
                [rx - hx, rx + hx], [rz - half * math.sin(pl), rz + half * math.sin(pl)], "-",
                color="C0", lw=3, alpha=0.8,
            )  # fmt: skip
            # lateral extent of the secondary axis as an ellipse on the map
            am.add_patch(
                Ellipse(
                    (rx, ry), 2 * sg[0] * max(math.cos(pl), 0.05), 2 * max(sg[1], 1e-3),
                    angle=-g["az"] + 90.0, fill=False, ec="C0", lw=0.8,
                )
            )  # fmt: skip
        if np.isfinite(g["peak_ratio"]):
            am.annotate(
                f"{g['peak_ratio']:.2f}",
                (cx, cy),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=7,
            )
    am.set_xlabel("east (km)")
    am.set_ylabel("north (km)")
    am.set_aspect("equal")
    am.set_title("map: offset (red) and major axis (blue); number = peak ratio")
    asec.set_xlabel("east (km)")
    asec.set_ylabel("depth (km)")
    asec.invert_yaxis()
    asec.set_title("east-depth section")
    if title:
        fig.suptitle(title)
    return fig


def plot_anisotropy(summary: pd.DataFrame, *, title: str = "") -> Figure:
    """Wavenumber retention per axis for the checkerboard tests (1 = as sharp as the truth)."""
    cols = [
        c for c in ("aniso_retention_x", "aniso_retention_y", "aniso_retention_z") if c in summary
    ]
    s = summary[summary["kind"] == "checkerboard"] if "kind" in summary else summary
    if not cols or s.empty:
        return _empty("no checkerboard metrics")
    fig, ax = new_figure(figsize=(max(5.0, 0.7 * len(s) + 2), 3.6))
    x = np.arange(len(s))
    for k, c in enumerate(cols):
        ax.bar(x + (k - 1) * 0.27, s[c].to_numpy(float), 0.27, label=c.split("_")[-1])
    ax.axhline(1.0, color="0.4", lw=0.7)
    ax.set_xticks(x, [str(n) for n in s["name"]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("wavenumber retention")
    ax.legend(title="axis")
    if title:
        ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------------------
# noise, L-curve, summary
# ---------------------------------------------------------------------------------------
def plot_noise(
    noise_s: Any, sigma_s: Any, *, phase: str = "P", rms_final: float | None = None
) -> Figure:
    """Histogram of the noise realisation (ms) vs the pick uncertainties of the real data."""
    n = np.asarray(noise_s, float) * 1e3
    s = np.asarray(sigma_s, float) * 1e3
    n, s = n[np.isfinite(n)], s[np.isfinite(s)]
    fig, (a, b) = new_figure(ncols=2, figsize=(9.0, 3.6))
    if n.size:
        a.hist(n, bins=50, density=True, color="0.6")
        sd = float(n.std())
        if sd > 0:
            x = np.linspace(n.min(), n.max(), 200)
            pdf = np.exp(-0.5 * ((x - n.mean()) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            a.plot(x, pdf, "C3-", label=f"normal, std {sd:.1f} ms")
            a.legend()
        a.set_title(f"{phase}: synthetic noise (n = {n.size})")
    a.set_xlabel("noise (ms)")
    a.set_ylabel("density")
    if s.size:
        b.hist(s, bins=40, color="C0")
        b.axvline(
            float(np.sqrt(np.mean(s**2))),
            color="k",
            ls="--",
            label=f"rms {np.sqrt(np.mean(s**2)):.1f} ms",
        )
        if rms_final is not None and math.isfinite(rms_final):
            b.axvline(
                rms_final * 1e3,
                color="C3",
                ls=":",
                label=f"final data rms {rms_final * 1e3:.1f} ms",
            )
        b.legend()
    b.set_title(f"{phase}: pick uncertainties")
    b.set_xlabel("uncertainty (ms)")
    b.set_ylabel("picks")
    return fig


def plot_synthetic_lcurve(
    real: Mapping[str, Any], synthetic: Mapping[str, Mapping[str, Any]], kind: str = "damping"
) -> Figure:
    """Synthetic L-curves over the real one (``picks[kind]`` dicts of ``tradeoff.json``).

    Each curve is normalised to its own maximum of data variance and x value so that curves of
    data with different noise level can be compared by shape.
    """
    fig, ax = new_figure(figsize=(6.0, 4.5))
    for lab, pick, style in [("real", real, "k-o"), *((k, v, "-s") for k, v in synthetic.items())]:
        cur = pd.DataFrame(pick.get("curve", []))
        xcol = str(pick.get("x_axis", "model_variance"))
        if cur.empty or xcol not in cur or "variance_s2" not in cur:
            continue
        cur = cur.sort_values(kind)
        x = cur[xcol].to_numpy(float)
        y = cur["variance_s2"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if ok.any():
            ax.plot(x[ok] / x[ok].max(), y[ok] / y[ok].max(), style, ms=4, lw=1, label=lab)
            ch = pick.get("chosen")
            sel = ok & np.isclose(cur[kind].to_numpy(float), float(ch)) if ch is not None else None
            if sel is not None and sel.any():
                ax.plot(
                    x[sel] / x[ok].max(), y[sel] / y[ok].max(), "o", mfc="none", mec="C1", ms=12
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model measure (normalised)")
    ax.set_ylabel("data variance (normalised)")
    ax.set_title(f"{kind.capitalize()} L-curve: synthetic vs real (circle = chosen)")
    if not ax.lines:
        ax.text(0.5, 0.5, "no curves", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend()
    return fig


def plot_summary_metrics(summary: pd.DataFrame) -> Figure:
    """Grouped bars of the global correlation, semblance and amplitude recovery per test."""
    cols = [
        ("global_corr", "correlation"),
        ("global_semblance", "semblance"),
        ("global_amp", "amplitude"),
    ]
    cols = [(c, lab) for c, lab in cols if c in summary]
    if summary.empty or not cols:
        return _empty("no test metrics")
    fig, ax = new_figure(figsize=(max(6.0, 0.8 * len(summary) + 2), 4.0))
    x = np.arange(len(summary))
    w = 0.8 / len(cols)
    for k, (c, lab) in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, summary[c].to_numpy(float), w, label=lab)
    ax.axhline(1.0, color="0.4", lw=0.6)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_xticks(x, [str(n) for n in summary["name"]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("global metric (covered volume)")
    ax.legend()
    return fig


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def _params(row: Mapping[str, Any]) -> dict[str, Any]:
    p = row.get("params", {})
    return p if isinstance(p, dict) else json.loads(p)


def _load(path: Any) -> xr.Dataset:
    with xr.open_dataset(path) as d:
        return d.load()


def _pct(true: xr.Dataset, rec: xr.Dataset, phase: str) -> tuple[np.ndarray, np.ndarray]:
    if phase == "P":
        return true["dvp_pct"].values, rec["dvp_pct"].values
    if "vs_bg" in true and "vs" in true:
        t = 100.0 * (true["vs"].values / true["vs_bg"].values - 1.0)
    else:
        t = true["dvp_pct"].values
    return t, rec["dvs_pct"].values


def make_all_figures(
    save: Callable[..., Any],
    *,
    grid: Any,
    table: pd.DataFrame,
    metrics: xr.Dataset | None = None,
    summary: pd.DataFrame | None = None,
    resolvable: xr.Dataset | None = None,
    depth_slices_km: Sequence[float] | None = None,
    profiles: Sequence[Any] | None = None,
    noise: Mapping[str, tuple[Any, Any]] | None = None,
    synthetic_tradeoff: Mapping[str, Any] | None = None,
    real_tradeoff: Mapping[str, Any] | None = None,
    final_rms_s: Mapping[str, float] | None = None,
    max_detail_tests: int = 6,
) -> list[str]:
    """Build and save every Stage 6 figure; returns the names saved.

    Parameters
    ----------
    save
        ``save(fig, name, caption, section=None)`` (``FigureRegistry.save``).
    table
        ``ResolutionRunResult.table`` (``name``, ``kind``, ``phase``, ``params``, ``success``,
        ``true_model``, ``recovered_model``).
    metrics
        ``metrics.nc`` of :func:`osv.resolution.metrics.run_metrics` (``<field>__<test>``).
    summary
        Its per-test summary table (``summary.csv`` content, plus the ``spikes`` list column if
        available).
    resolvable
        ``resolvable_mask.nc`` (``resolved_scale_km_vp`` ...).
    noise
        ``{phase: (noise_s, sigma_s)}`` arrays of the synthetic data.
    synthetic_tradeoff, real_tradeoff
        Optional ``{label: picks[kind]}`` of synthetic trade-off sweeps and the real ``picks``
        entry (``tradeoff.json`` ``results[phase]['picks']``) for the L-curve comparison.
    max_detail_tests
        Detailed side-by-side figures are made for at most this many tests (non-checkerboard
        first, then the largest cell sizes).
    """
    done: list[str] = []

    def put(fig: Figure, name: str, caption: str, section: str = "resolution") -> None:
        save(fig, name, caption, section)
        done.append(name)

    ok = table[table["success"].astype(bool)] if len(table) else table
    zs = pick_depths(grid, depth_slices_km)
    # --- true vs recovered
    order = sorted(
        range(len(ok)),
        key=lambda i: (
            ok.iloc[i]["kind"] == "checkerboard",
            -float(_params(ok.iloc[i]).get("cell_km", 0) or 0),
        ),
    )
    cache: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for i in range(len(ok)):
        row = ok.iloc[i]
        try:
            t, r = _pct(_load(row["true_model"]), _load(row["recovered_model"]), str(row["phase"]))
            cache[str(row["name"])] = (t, r, str(row["phase"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot load test %s for plotting: %s", row["name"], exc)
    for i in order[:max_detail_tests]:
        name = str(ok.iloc[i]["name"])
        if name not in cache:
            continue
        t, r, ph = cache[name]
        put(
            plot_true_vs_recovered_slices(t, r, grid, zs, title=name),
            f"true_vs_recovered_slices_{name}",
            f"{name}: true (left) and recovered (right) {ph} perturbation in depth slices; "
            "common symmetric colour scale.",
        )
        put(
            plot_true_vs_recovered_sections(t, r, grid, profiles, title=name),
            f"true_vs_recovered_sections_{name}",
            f"{name}: true and recovered {ph} perturbation along vertical sections.",
        )
    # --- checkerboard matrices and recovery maps (unshifted copies)
    for ph in sorted({str(p) for p in ok["phase"]}) if len(ok) else []:
        cbs = []
        for i in range(len(ok)):
            row = ok.iloc[i]
            p = _params(row)
            if (
                row["kind"] == "checkerboard"
                and str(row["phase"]) == ph
                and not p.get("offset_half_cell")
                and str(row["name"]) in cache
            ):
                cbs.append((float(p["cell_km"]), str(row["name"])))
        cbs.sort()
        if not cbs:
            continue
        cases = [(f"{c:g} km", cache[n][0], cache[n][1]) for c, n in cbs]
        put(
            plot_checkerboard_matrix(cases, grid, zs, title=f"{ph} checkerboards"),
            f"checkerboard_matrix_{ph}",
            f"{ph}: recovered checkerboard patterns; rows = cell size, columns = depth; black "
            "line = zero crossings of the true pattern.",
        )
        if metrics is not None:
            for q, lab in (("corr", "local correlation"), ("amp", "amplitude recovery")):
                rows = [
                    (f"{c:g} km", metrics[f"{q}__{n}"].values)
                    for c, n in cbs
                    if f"{q}__{n}" in metrics
                ]
                if rows:
                    put(
                        plot_recovery_matrix(rows, grid, zs, q, title=f"{ph}: {lab}"),
                        f"recovery_{q}_{ph}",
                        f"{ph}: {lab} (Gaussian moving window) of the checkerboard tests.",
                    )
        if resolvable is not None and f"resolved_scale_km_v{ph.lower()}" in resolvable:
            put(
                plot_resolved_scale(
                    resolvable[f"resolved_scale_km_v{ph.lower()}"].values,
                    grid,
                    zs,
                    [c for c, _ in cbs],
                    profiles,
                    title=f"{ph}: resolved scale",
                ),
                f"resolved_scale_{ph}",
                f"{ph}: smallest checkerboard cell size recovered at each node (correlation and "
                "amplitude above the thresholds).",
            )
    # --- smearing
    if summary is not None and len(summary):
        for _, srow in summary.iterrows():
            sp = srow.get("spikes") if hasattr(srow, "get") else None
            if isinstance(sp, list) and sp:
                put(
                    plot_smearing({"spikes": sp}, title=str(srow["name"])),
                    f"smearing_{srow['name']}",
                    f"{srow['name']}: spike centroid offsets and principal axes of the recovered "
                    "spread.",
                )
        for ph in sorted({str(p) for p in summary["phase"]}):
            sub = summary[(summary["phase"] == ph) & (summary["kind"] == "checkerboard")]
            if len(sub) and any(c.startswith("aniso_retention") for c in sub):
                put(
                    plot_anisotropy(sub, title=f"{ph}: per-axis retention"),
                    f"checkerboard_anisotropy_{ph}",
                    f"{ph}: wavenumber retention of the recovered checkerboard along each axis "
                    "(< 1 = smeared along that axis).",
                )
        put(
            plot_summary_metrics(summary),
            "summary_metrics",
            "Global correlation, semblance and amplitude recovery of every test over the covered "
            "volume.",
        )
    # --- noise
    for ph, (n, s) in (noise or {}).items():
        put(
            plot_noise(n, s, phase=ph, rms_final=(final_rms_s or {}).get(ph)),
            f"noise_realisation_{ph}",
            f"{ph}: noise added to the synthetic data against the pick uncertainties.",
        )
    # --- synthetic vs real L-curve
    if real_tradeoff and synthetic_tradeoff:
        for kind in ("damping", "smoothing"):
            if kind in real_tradeoff:
                put(
                    plot_synthetic_lcurve(
                        real_tradeoff[kind],
                        {k: v[kind] for k, v in synthetic_tradeoff.items() if kind in v},
                        kind,
                    ),
                    f"lcurve_synthetic_vs_real_{kind}",
                    f"{kind} L-curve of the real data and of synthetic data (normalised).",
                )
    return done
