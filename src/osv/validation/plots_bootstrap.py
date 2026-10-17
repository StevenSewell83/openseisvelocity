"""Figures of the bootstrap / jackknife model uncertainty (Stage 7).

Every function takes a :class:`~osv.validation.bootstrap.BootstrapResult` and returns a
matplotlib figure (or ``None`` when there is nothing to show).  :func:`make_figures` saves them
through a :class:`~osv.plotting.registry.FigureRegistry` with captions.  Depth slices and
sections show the *interior* of the model (cushion nodes removed,
:func:`osv.plotting.slices.crop_interior`) and, when the result carries a ``dws`` variable, grey
out nodes below ``dws_threshold``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from osv.plotting.registry import FigureRegistry
from osv.plotting.slices import crop_interior, default_profiles, draw_mask
from osv.plotting.style import new_figure, setup_map_axes
from osv.validation.bootstrap import BootstrapResult

__all__ = [
    "make_figures",
    "plot_hypocentre_scatter",
    "plot_sample_misfit",
    "plot_std_sections",
    "plot_std_slices",
    "plot_std_vs_dws",
]

CMAP = "viridis"


def _var(res: BootstrapResult) -> str:
    return "vp" if "dvp_std_pct" in res.stats else "vs"


def _covered(ds: Any, threshold: float | None) -> Any:
    if threshold is None or "dws" not in ds:
        return None
    return ds["dws"].values >= threshold


def plot_std_slices(
    res: BootstrapResult,
    depths: list[float] | None = None,
    dws_threshold: float | None = None,
    max_panels: int = 6,
) -> Any:
    """Depth slices of the standard deviation of the velocity perturbation (percent)."""
    ds = crop_interior(res.stats)
    v = _var(res)
    da = ds[f"d{v}_std_pct"]
    if depths is None:
        z = da["depth"].values
        n = min(max_panels, len(z))
        depths = [float(z[i]) for i in np.unique(np.linspace(0, len(z) - 1, n).round().astype(int))]
    n = len(depths)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = new_figure(ncols=ncol, nrows=nrow, figsize=(4.2 * ncol, 3.6 * nrow),
                           squeeze=False, constrained_layout=True)  # fmt: skip
    vmax = float(np.nanpercentile(da.values, 99)) or 1.0
    lon, lat = ds["lon"].values, ds["lat"].values
    last = None
    for ax, z in zip(axes.ravel(), depths, strict=False):
        sl = da.sel(depth=z, method="nearest")
        last = ax.pcolormesh(lon, lat, sl.values, shading="auto", cmap=CMAP, vmin=0, vmax=vmax)
        cov = _covered(ds.sel(depth=z, method="nearest"), dws_threshold)
        if cov is not None:
            draw_mask(ax, lon, lat, cov, "alpha")
        ax.set_title(f"{float(sl['depth']):.1f} km depth")
        setup_map_axes(ax, float(lat.mean()))
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.colorbar(last, ax=axes, shrink=0.8, label=f"std of d{v.capitalize()} ({res.method}, %)")
    return fig


def plot_std_sections(res: BootstrapResult, dws_threshold: float | None = None) -> Any:
    """E-W and N-S sections through the model centre of the standard deviation (percent)."""
    ds = crop_interior(res.stats)
    v = _var(res)
    prof = default_profiles(ds["lat"].values, ds["lon"].values)
    fig, axes = new_figure(ncols=len(prof), figsize=(5.0 * len(prof), 3.6), squeeze=False,
                           constrained_layout=True)  # fmt: skip
    da = ds[f"d{v}_std_pct"]
    vmax = float(np.nanpercentile(da.values, 99)) or 1.0
    last = None
    for ax, (name, a, b) in zip(axes.ravel(), prof, strict=True):
        if abs(a[0] - b[0]) < 1e-12:  # E-W: constant latitude
            sec = da.sel(lat=a[0], method="nearest")
            x, xl = ds["lon"].values, "Longitude (deg)"
            cov = _covered(ds.sel(lat=a[0], method="nearest"), dws_threshold)
        else:
            sec = da.sel(lon=a[1], method="nearest")
            x, xl = ds["lat"].values, "Latitude (deg)"
            cov = _covered(ds.sel(lon=a[1], method="nearest"), dws_threshold)
        z = ds["depth"].values
        last = ax.pcolormesh(x, z, sec.values, shading="auto", cmap=CMAP, vmin=0, vmax=vmax)
        if cov is not None:
            draw_mask(ax, x, z, cov, "alpha")
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel(xl)
        ax.set_ylabel("Depth (km)")
    fig.colorbar(last, ax=axes, shrink=0.8, label=f"std of d{v.capitalize()} ({res.method}, %)")
    return fig


def plot_std_vs_dws(res: BootstrapResult, max_points: int = 20000, seed: int = 0) -> Any:
    """Scatter of the per-node standard deviation against the derivative weight sum."""
    if "dws" not in res.stats:
        return None
    ds = crop_interior(res.stats)
    v = _var(res)
    x = ds["dws"].values.ravel().astype(float)
    y = ds[f"d{v}_std_pct"].values.ravel().astype(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not ok.any():
        return None
    x, y = x[ok], y[ok]
    if len(x) > max_points:
        i = np.random.default_rng(seed).choice(len(x), max_points, replace=False)
        x, y = x[i], y[i]
    fig, ax = new_figure(figsize=(5.0, 4.0))
    ax.scatter(x, y, s=6, alpha=0.35, c="#1f77b4", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Derivative weight sum")
    ax.set_ylabel(f"std of d{v.capitalize()} (%)")
    ax.set_title(f"{res.method}: uncertainty against ray coverage")
    return fig


def plot_sample_misfit(res: BootstrapResult) -> Any:
    """Initial and final data RMS (ms) of every sample; failed samples are marked."""
    t = res.table
    if t.empty:
        return None
    x = np.arange(len(t))
    fig, ax = new_figure(figsize=(max(5.0, 0.35 * len(t) + 2), 3.6))
    ax.bar(x - 0.2, 1000 * t["misfit_initial_s"].astype(float), 0.4, color="0.7", label="start")
    ax.bar(x + 0.2, 1000 * t["misfit_final_s"].astype(float), 0.4, color="#ff7f0e", label="final")
    for i, ok in enumerate(t["success"]):
        if not ok:
            ax.annotate(
                "failed", (i, 0), ha="center", va="bottom", color="r", rotation=90, fontsize=7
            )
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in t["name"]], rotation=90, fontsize=7)
    ax.set_ylabel("Data RMS (ms)")
    ax.legend()
    return fig


def plot_hypocentre_scatter(res: BootstrapResult) -> Any:
    """Histograms of the horizontal and vertical scatter of events present in >= 2 samples."""
    h = res.hypocentre_stats
    if h is None or h.empty:
        return None
    fig, axes = new_figure(ncols=2, figsize=(7.0, 3.2), constrained_layout=True)
    axes[0].hist(h["std_h_km"].dropna(), bins=20, color="#1f77b4", edgecolor="k", linewidth=0.4)
    axes[0].set_xlabel("Horizontal std (km)")
    axes[1].hist(h["std_depth_km"].dropna(), bins=20, color="#ff7f0e", edgecolor="k", linewidth=0.4)
    axes[1].set_xlabel("Depth std (km)")
    axes[0].set_ylabel("Events")
    return fig


def make_figures(
    res: BootstrapResult,
    figures: FigureRegistry | str | Path,
    section: str = "Bootstrap",
    dws_threshold: float | None = None,
) -> list[str]:
    """Save all bootstrap figures; returns the names written.

    ``dws_threshold`` greys out nodes with a smaller derivative weight sum (needs ``dws`` in the
    result).  ``figures`` is a registry or a directory (png and pdf are written then).
    """
    reg = figures if isinstance(figures, FigureRegistry) else FigureRegistry(Path(figures))
    m = res.method
    items = [
        (
            "bootstrap_std_slices",
            lambda r: plot_std_slices(r, dws_threshold=dws_threshold),
            f"Standard deviation of the velocity perturbation over {int(res.n_ok)} {m} "
            "inversions of event-resampled data, depth slices. Large values are poorly "
            "constrained nodes.",
        ),
        (
            "bootstrap_std_sections",
            lambda r: plot_std_sections(r, dws_threshold=dws_threshold),
            f"East-west and north-south sections of the {m} standard deviation through the "
            "centre of the model.",
        ),
        (
            "bootstrap_std_vs_dws",
            plot_std_vs_dws,
            f"{m.capitalize()} standard deviation of every interior node against its derivative "
            "weight sum: uncertainty should fall with ray coverage.",
        ),
        (
            "bootstrap_sample_misfit",
            plot_sample_misfit,
            "Data RMS before and after the inversion of each resampled data set.",
        ),
        (
            "bootstrap_hypocentre_scatter",
            plot_hypocentre_scatter,
            "Scatter of the relocated hypocentres of events present in at least two samples.",
        ),
    ]
    out: list[str] = []
    for name, fn, caption in items:
        fig = fn(res)
        if fig is None:
            continue
        reg.save(fig, name, caption, section)
        out.append(name)
    return out
