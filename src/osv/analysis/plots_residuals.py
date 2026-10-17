"""Residual and relocation figures (task 6.3).

All functions take the tables of :mod:`osv.analysis.residuals` and return a matplotlib figure;
:func:`make_residual_figures` builds the whole set from a run directory and saves it through a
:class:`~osv.plotting.registry.FigureRegistry`.  "Initial" is the starting model (iteration 0),
"final" the last iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.analysis.exports import local_km
from osv.analysis.residuals import (
    available_iterations,
    relocation_table,
    residual_stats_by,
    residual_tables,
    variance_reduction,
)
from osv.plotting.registry import FigureRegistry
from osv.plotting.style import (
    DEPTH_CMAP,
    PERTURBATION_CMAP,
    new_figure,
    perturbation_norm,
    setup_map_axes,
)

log = logging.getLogger(__name__)

__all__ = [
    "make_residual_figures",
    "plot_event_rms",
    "plot_hypocentre_shifts",
    "plot_residual_histograms",
    "plot_residual_vs",
    "plot_seismicity_before_after",
    "plot_shift_histograms",
    "plot_station_residual_map",
    "plot_variance_reduction",
]

C_INIT = "#7f7f7f"
C_FINAL = "#d95f02"
#: Residual axis limit is this many robust sigmas (1.4826 MAD) of the initial residuals.
_LIM_SIGMA = 5.0


def _lim(*dfs: pd.DataFrame, col: str = "residual") -> float:
    r = np.concatenate([d[col].to_numpy(float) for d in dfs if len(d)] or [np.array([1.0])])
    r = r[np.isfinite(r)]
    if not r.size:
        return 1.0
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    return float(max(_LIM_SIGMA * mad, 1e-3, np.abs(np.median(r))))


def _phases(*dfs: pd.DataFrame) -> list[str]:
    return sorted({p for d in dfs for p in d["phase"].unique()})


def plot_residual_histograms(initial: pd.DataFrame, final: pd.DataFrame):
    """Residual histograms, initial vs final, one panel per phase (mean and RMS annotated)."""
    ph = _phases(initial, final) or ["all"]
    fig, axs = new_figure(ncols=len(ph), figsize=(4.2 * len(ph), 3.6), squeeze=False)
    lim = _lim(initial, final)
    bins = np.linspace(-lim, lim, 61)
    for ax, p in zip(axs[0], ph, strict=True):
        for d, lab, c in ((initial, "initial", C_INIT), (final, "final", C_FINAL)):
            g = d[d["phase"] == p] if p != "all" else d
            r = g["residual"].to_numpy(float)
            r = r[np.isfinite(r)]
            rms = np.sqrt(np.mean(r**2)) if r.size else np.nan
            ax.hist(r, bins=bins, alpha=0.6, color=c, label=f"{lab}: rms {rms:.3f} s, n={r.size}")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{p} residuals")
        ax.set_xlabel("observed - calculated (s)")
        ax.set_ylabel("picks")
        ax.legend()
    return fig


def plot_residual_vs(initial: pd.DataFrame, final: pd.DataFrame, x: str = "dist_km"):
    """Residual against hypocentral distance (``x="dist_km"``) or azimuth (``"azimuth_deg"``).

    Rows are phases, columns initial/final; the running median is overlaid.
    """
    if x not in ("dist_km", "azimuth_deg"):
        raise ValueError("x must be 'dist_km' or 'azimuth_deg'")
    ph = _phases(initial, final) or ["all"]
    fig, axs = new_figure(
        nrows=len(ph), ncols=2, figsize=(8, 2.8 * len(ph)), squeeze=False, sharex=True, sharey=True
    )
    lim = _lim(initial, final)
    label = "hypocentral distance (km)" if x == "dist_km" else "azimuth event to station (deg)"
    for i, p in enumerate(ph):
        for j, (d, lab, c) in enumerate(((initial, "initial", C_INIT), (final, "final", C_FINAL))):
            ax = axs[i, j]
            g = d[d["phase"] == p] if p != "all" else d
            ax.scatter(g[x], g["residual"], s=4, alpha=0.35, color=c, rasterized=True)
            if len(g) >= 10:
                q = pd.qcut(g[x], min(12, max(2, len(g) // 10)), duplicates="drop")
                med = g.groupby(q, observed=True).agg(x=(x, "median"), r=("residual", "median"))
                ax.plot(med["x"], med["r"], "k-o", ms=3, lw=1.2, label="median")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{p} {lab}")
            if i == len(ph) - 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel("residual (s)")
    if x == "azimuth_deg":
        axs[0, 0].set_xlim(0, 360)
    return fig


def _station_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("station").agg(
        lat=("sta_lat", "first"),
        lon=("sta_lon", "first"),
        mean=("residual", "mean"),
        n=("residual", "size"),
    )
    return g


def plot_station_residual_map(initial: pd.DataFrame, final: pd.DataFrame, label_stations=True):
    """Map of per-station mean residual, initial and final (shared diverging scale)."""
    fig, axs = new_figure(ncols=2, figsize=(10, 4.6), squeeze=False)
    tabs = [_station_table(initial), _station_table(final)]
    vmax = max([float(np.nanmax(np.abs(t["mean"]))) for t in tabs if len(t)] or [1.0]) or 1.0
    norm = perturbation_norm(vmax=vmax)
    sc = None
    for ax, t, lab in zip(axs[0], tabs, ("initial", "final"), strict=True):
        sc = ax.scatter(
            t["lon"],
            t["lat"],
            c=t["mean"],
            cmap=PERTURBATION_CMAP,
            norm=norm,
            marker="^",
            s=70,
            edgecolors="k",
            linewidths=0.6,
            zorder=5,
        )
        if label_stations:
            for name, r in t.iterrows():
                ax.annotate(
                    str(name),
                    (r["lon"], r["lat"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
        ax.set_title(f"mean residual per station, {lab}")
        if len(t):
            setup_map_axes(ax, float(t["lat"].mean()))
    if sc is not None:
        fig.colorbar(sc, ax=list(axs[0]), label="mean residual (s); blue: model too fast")
    return fig


def plot_event_rms(initial: pd.DataFrame, final: pd.DataFrame):
    """Per-event RMS before vs after (scatter with 1:1 line) and the distribution of the change."""

    def rms(d: pd.DataFrame) -> pd.Series:
        return d.groupby("event_id")["residual"].apply(lambda r: float(np.sqrt(np.mean(r**2))))

    a, b = rms(initial), rms(final)
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    fig, (ax1, ax2) = new_figure(ncols=2, figsize=(9, 4))
    ax1.scatter(j["a"], j["b"], s=10, color=C_FINAL, alpha=0.7)
    top = float(j.to_numpy().max()) if len(j) else 1.0
    ax1.plot([0, top], [0, top], "k-", lw=0.8)
    ax1.set_xlabel("event RMS, initial (s)")
    ax1.set_ylabel("event RMS, final (s)")
    ax1.set_aspect("equal")
    ax1.set_title(f"{len(j)} events ({int((j['b'] < j['a']).sum())} improved)")
    d = j["b"] - j["a"]
    ax2.hist(d, bins=40, color=C_FINAL, alpha=0.8)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("change in event RMS (s)")
    ax2.set_ylabel("events")
    ax2.set_title(f"median change {d.median():.3f} s" if len(d) else "no events")
    return fig


def plot_variance_reduction(vr: pd.DataFrame):
    """Variance reduction (%) and RMS per iteration."""
    fig, (ax1, ax2) = new_figure(ncols=2, figsize=(9, 3.8))
    if vr["vr_pct"].notna().any():
        ax1.plot(vr["iteration"], vr["vr_pct"], "o-", color=C_FINAL, label="picks")
    if "fmtomo_vr_pct" in vr and vr["fmtomo_vr_pct"].notna().any():
        ax1.plot(
            vr["iteration"], vr["fmtomo_vr_pct"], "s--", color=C_INIT, label="residuals program"
        )
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("variance reduction (%)")
    ax1.legend()
    if vr["rms_s"].notna().any():
        ax2.plot(vr["iteration"], vr["rms_s"] * 1000, "o-", color=C_FINAL, label="picks")
    if "fmtomo_rms_ms" in vr and vr["fmtomo_rms_ms"].notna().any():
        ax2.plot(
            vr["iteration"], vr["fmtomo_rms_ms"], "s--", color=C_INIT, label="residuals program"
        )
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("RMS residual (ms)")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
    return fig


def _local(reloc: pd.DataFrame, stations: pd.DataFrame | None):
    lat0, lon0 = float(reloc["lat0"].mean()), float(reloc["lon0"].mean())
    x0, y0 = local_km(reloc["lat0"], reloc["lon0"], lat0, lon0)
    x1, y1 = local_km(reloc["lat"], reloc["lon"], lat0, lon0)
    st = None
    if stations is not None and len(stations):
        st = local_km(stations["lat"], stations["lon"], lat0, lon0)
    return (x0, y0, x1, y1), st


def plot_hypocentre_shifts(reloc: pd.DataFrame, stations: pd.DataFrame | None = None):
    """Shift arrows on a map (coloured by depth change) plus a depth section (x-z) of the shifts."""
    (x0, y0, x1, y1), st = _local(reloc, stations)
    fig, (ax1, ax2) = new_figure(ncols=2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1]})
    vmax = float(np.nanmax(np.abs(reloc["dz_km"]))) if len(reloc) else 1.0
    norm = perturbation_norm(vmax=vmax or 1.0)
    q = ax1.quiver(
        x0,
        y0,
        x1 - x0,
        y1 - y0,
        reloc["dz_km"],
        cmap=PERTURBATION_CMAP,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
    )
    if st is not None:
        ax1.scatter(*st, marker="^", c="white", edgecolors="k", s=30, zorder=5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("east (km)")
    ax1.set_ylabel("north (km)")
    ax1.set_title("epicentre shifts (arrow = initial to relocated)")
    fig.colorbar(q, ax=ax1, label="depth change (km), red: shallower")
    z0, z1 = reloc["depth0_km"].to_numpy(), reloc["depth_km"].to_numpy()
    ax2.quiver(
        x0, z0, x1 - x0, z1 - z0, angles="xy", scale_units="xy", scale=1, width=0.003, color=C_FINAL
    )
    ax2.scatter(x0, z0, s=4, color=C_INIT, zorder=3)
    ax2.invert_yaxis()
    ax2.set_xlabel("east (km)")
    ax2.set_ylabel("depth (km)")
    ax2.set_title("depth section (east-depth)")
    return fig


def plot_shift_histograms(reloc: pd.DataFrame):
    """Histograms of dx, dy, dz and origin-time shifts."""
    cols = [
        ("dx_km", "east shift (km)"),
        ("dy_km", "north shift (km)"),
        ("dz_km", "depth shift (km)"),
    ]
    if reloc["dt_s"].notna().any():
        cols.append(("dt_s", "origin-time shift (s)"))
    fig, axs = new_figure(ncols=len(cols), figsize=(3.2 * len(cols), 3.2), squeeze=False)
    for ax, (c, lab) in zip(axs[0], cols, strict=True):
        v = reloc[c].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=30, color=C_FINAL, alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("events")
        if v.size:
            ax.set_title(f"mean {v.mean():.2f}, sd {v.std():.2f}")
    return fig


def plot_seismicity_before_after(reloc: pd.DataFrame, stations: pd.DataFrame | None = None):
    """Seismicity before/after relocation: map, east-depth and north-depth sections."""
    (x0, y0, x1, y1), st = _local(reloc, stations)
    z0, z1 = reloc["depth0_km"].to_numpy(), reloc["depth_km"].to_numpy()
    fig, axs = new_figure(nrows=2, ncols=3, figsize=(13, 7.5), squeeze=False)
    vmin = float(min(z0.min(), z1.min())) if len(z0) else 0.0
    vmax = float(max(z0.max(), z1.max())) if len(z0) else 1.0
    kw = {"cmap": DEPTH_CMAP, "vmin": vmin, "vmax": vmax, "s": 8, "edgecolors": "none"}
    sc = None
    for r, (x, y, z, lab) in enumerate(((x0, y0, z0, "before"), (x1, y1, z1, "after"))):
        sc = axs[r, 0].scatter(x, y, c=z, **kw)
        axs[r, 1].scatter(x, z, c=z, **kw)
        axs[r, 2].scatter(y, z, c=z, **kw)
        if st is not None:
            axs[r, 0].scatter(*st, marker="^", c="white", edgecolors="k", s=25, zorder=5)
        axs[r, 0].set_aspect("equal")
        axs[r, 0].set_xlabel("east (km)")
        axs[r, 0].set_ylabel("north (km)")
        axs[r, 0].set_title(f"{lab}: map")
        axs[r, 1].set_xlabel("east (km)")
        axs[r, 1].set_title(f"{lab}: east-depth")
        axs[r, 2].set_xlabel("north (km)")
        axs[r, 2].set_title(f"{lab}: north-depth")
        for c in (1, 2):
            axs[r, c].set_ylabel("depth (km)")
    # same limits for before and after
    allx, ally = np.concatenate([x0, x1]), np.concatenate([y0, y1])
    allz = np.concatenate([z0, z1])
    if len(allx):
        for r in range(2):
            axs[r, 0].set_xlim(allx.min() - 0.5, allx.max() + 0.5)
            axs[r, 0].set_ylim(ally.min() - 0.5, ally.max() + 0.5)
            axs[r, 1].set_xlim(allx.min() - 0.5, allx.max() + 0.5)
            axs[r, 2].set_xlim(ally.min() - 0.5, ally.max() + 0.5)
            axs[r, 1].set_ylim(allz.max() + 0.3, allz.min() - 0.3)
            axs[r, 2].set_ylim(allz.max() + 0.3, allz.min() - 0.3)
    if sc is not None:
        fig.colorbar(sc, ax=axs, label="depth (km)", shrink=0.8)
    return fig


def _stations_frame(tables: dict[int, pd.DataFrame]) -> pd.DataFrame | None:
    for df in tables.values():
        if len(df):
            return (
                df.groupby("station")
                .agg(lat=("sta_lat", "first"), lon=("sta_lon", "first"))
                .reset_index()
            )
    return None


def make_residual_figures(
    run_dir: str | Path,
    mapping: pd.DataFrame,
    registry: FigureRegistry,
    *,
    nonlinear: bool = False,
    section: str | None = "residuals",
) -> dict[str, Any]:
    """Build and save all residual/relocation figures of a run.

    Figures needing per-pick times use the initial and the final iteration; the variance
    reduction uses every iteration with times plus the ``summary.json`` series.  Relocation
    figures are skipped (logged) when no event moved.

    Returns
    -------
    dict
        ``tables`` (``{iteration: residual table}``), ``variance`` (the variance-reduction table),
        ``by_station``, ``by_event``, ``by_phase`` (final residual statistics), ``relocation``
        (final table or None) and ``figures`` (names saved).
    """
    run = Path(run_dir)
    its = available_iterations(run)
    if not its:
        raise FileNotFoundError(f"{run / 'iterations'} has no iteration directories")
    tables = residual_tables(run, mapping, nonlinear=nonlinear)
    first, last = its[0], its[-1]
    if first not in tables or last not in tables:
        raise FileNotFoundError("need calculated times of the initial and the final iteration")
    ini, fin = tables[first], tables[last]
    saved: list[str] = []

    def save(fig: Any, name: str, caption: str) -> None:
        registry.save(fig, name, caption, section)
        saved.append(name)

    save(
        plot_residual_histograms(ini, fin),
        "residual_histograms",
        "Residual (observed - calculated) histograms per phase, initial vs final model.",
    )
    save(
        plot_residual_vs(ini, fin, "dist_km"),
        "residual_vs_distance",
        "Residuals against hypocentral distance with running median, initial and final.",
    )
    save(
        plot_residual_vs(ini, fin, "azimuth_deg"),
        "residual_vs_azimuth",
        "Residuals against azimuth from event to station, initial and final.",
    )
    save(
        plot_station_residual_map(ini, fin),
        "station_residual_map",
        "Mean residual per station, initial and final.",
    )
    save(
        plot_event_rms(ini, fin),
        "event_rms_before_after",
        "RMS residual per event before and after the inversion.",
    )
    vr = variance_reduction(run, mapping, tables, nonlinear=nonlinear)
    save(
        plot_variance_reduction(vr),
        "variance_reduction",
        "Variance reduction and RMS residual per iteration.",
    )

    reloc = relocation_table(run, mapping, last, nonlinear=nonlinear)
    moved = len(reloc) and float(reloc["d3_km"].max()) > 0
    if moved:
        sta = _stations_frame(tables)
        save(
            plot_hypocentre_shifts(reloc, sta),
            "hypocentre_shifts",
            "Hypocentre shifts: epicentre arrows and east-depth section.",
        )
        save(
            plot_shift_histograms(reloc),
            "hypocentre_shift_histograms",
            "Histograms of east, north, depth and origin-time shifts.",
        )
        save(
            plot_seismicity_before_after(reloc, sta),
            "seismicity_before_after",
            "Seismicity before and after relocation: map and depth sections.",
        )
    else:
        log.info("no hypocentre moved: relocation figures skipped")
    return {
        "tables": tables,
        "variance": vr,
        "by_station": residual_stats_by(fin, ["station", "phase"]),
        "by_event": residual_stats_by(fin, ["event_id", "phase"]),
        "by_phase": residual_stats_by(fin, "phase"),
        "relocation": reloc if moved else None,
        "figures": saved,
    }
