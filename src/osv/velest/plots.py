"""Stage 2 (VELEST Monte Carlo) diagnostic figures.

Every ``plot_*`` function takes the plain result objects of the Stage 2 modules
(:class:`~osv.velest.ensemble.EnsembleResult`, :class:`~osv.velest.stability.StabilityResult`,
the ``history``/``runs`` tables of :class:`~osv.velest.workflow.WorkflowResult`, the table of
:func:`~osv.velest.vpvs.vpvs_grid_search`) or DataFrames and returns a matplotlib ``Figure``.
Nothing here reads the configuration or the file system; saving is done by the caller.

Missing optional input
----------------------
A function whose *primary* input is missing or empty (no S delays at all, no stability runs of
the required test, no ``parsed`` run results, ...) returns ``None``; optional extras (station
table, input hypocentres, search bounds) are simply left out of the figure.
:func:`make_all_figures` skips ``None`` results and a figure that raises is logged and skipped,
so one bad table never loses the other figures.

Conventions
-----------
Depth is km below sea level, positive down (the axis is inverted); velocities are km/s.  Layered
models are drawn as staircases.  Station delays are VELEST ``ptcor``/``stcor`` (s); a positive
delay means a late arrival, i.e. slow ground below the station, and is drawn red.

Station tables
--------------
``stations`` may be the Stage 1 station table (``alias``/``sta``, ``lat``, ``lon``,
``elevation_m``, ``sensor_elevation_m``, ``sensor_depth_m``) or the written VELEST station table
(``sta``, ``lat``, ``lon``, ``elev_m``, ``n_usable``).  Borehole stations (sensor depth > 0)
are drawn as down-triangles.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from osv.plotting.style import (
    RC,
    add_scale_bar,
    new_figure,
    perturbation_norm,
    setup_map_axes,
)
from osv.report.checks import QCFinding

__all__ = [
    "make_all_figures",
    "plot_convergence",
    "plot_delay_vs_elevation",
    "plot_final_hypocentres",
    "plot_hypocentre_perturbation",
    "plot_hypocentre_shifts",
    "plot_layer_histograms",
    "plot_lvz_comparison",
    "plot_reference_swap",
    "plot_residuals",
    "plot_rms_vs_iteration",
    "plot_rms_vs_layer_velocity",
    "plot_spaghetti",
    "plot_stability_velocity_spread",
    "plot_stage_rms",
    "plot_station_delay_map",
    "plot_velocity_density",
    "plot_vpvs",
    "to_qc_findings",
]

logger = logging.getLogger(__name__)

KM_PER_DEG = 111.195
PHASE_COLORS = {"P": "#0072B2", "S": "#D55E00"}
GREY = "#a8a8a8"
SELECTED_COLOR = "#D55E00"
RMS_CMAP = "viridis"
#: Delay colour map: positive (late, slow) red, negative blue.
DELAY_CMAP = "RdBu_r"
_HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]
OUTCOME_COLORS = {
    "ok": "#999999",
    "converged": "#009E73",
    "four_backups": "#E69F00",
    "failed": "#D55E00",
    "diverged": "#CC79A7",
}
OUTCOMES = ("converged", "ok", "four_backups", "diverged", "failed")


# ------------------------------------------------------------------------------- helpers


def _blank(figsize: tuple[float, float]) -> Figure:
    """Empty constrained-layout figure under the osv style (axes added by the caller)."""
    fig, ax = new_figure(figsize=figsize)
    ax.remove()
    return fig


def _axes(fig: Figure, *args: Any, **kw: Any):
    """``add_subplot``/``add_gridspec`` consumer that honours the osv rcParams (grid etc.)."""
    with matplotlib.rc_context(RC):
        return fig.add_subplot(*args, **kw)


def _isnum(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _steps(depths: np.ndarray, v: np.ndarray, zbot: float) -> tuple[np.ndarray, np.ndarray]:
    """Staircase ``(velocity, depth)`` of a layered model (layer ``i`` spans ``z_i..z_{i+1}``)."""
    z = np.append(np.asarray(depths, dtype=float), zbot)
    v = np.asarray(v, dtype=float)
    xs = np.repeat(v, 2)
    zs = np.repeat(z, 2)[1:-1]
    return xs, zs


def _zbot(depths: np.ndarray, zmax: float | None = None) -> float:
    d = np.asarray(depths, dtype=float)
    if zmax is not None:
        return float(zmax)
    thick = float(d[-1] - d[-2]) if len(d) > 1 else 1.0
    return float(d[-1] + max(thick, 0.5))


def _eval_steps(depths: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(depths, z, side="right") - 1, 0, len(depths) - 1)
    out = np.asarray(v, dtype=float)[idx]
    out[z < depths[0]] = np.nan
    return out


def _pivot(fm: pd.DataFrame, phase: str, col: str) -> pd.DataFrame:
    """``id x layer`` matrix of a ``final_models`` column."""
    sub = fm[fm["phase"] == phase]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot(index="id", columns="layer", values=col).sort_index(axis=1)


def _layer_depths(ens: Any) -> np.ndarray:
    return ens.final_model["depth_top"].to_numpy(dtype=float)


def _rms_norm(rms: Sequence[float]) -> Normalize:
    r = np.asarray(rms, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return Normalize(0.0, 1.0)
    lo, hi = float(r.min()), float(np.percentile(r, 95))
    if hi <= lo:
        hi = lo + 1e-6
    return Normalize(lo, hi, clip=True)


def _label(df: pd.DataFrame) -> pd.Series:
    for c in ("alias", "sta"):
        if c in df.columns:
            return df[c].astype(str)
    return pd.Series(df.index.astype(str), index=df.index)


def _stations(stations: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise a station table to ``sta, lat, lon, elev_m, borehole, n_picks``."""
    if stations is None or len(stations) == 0:
        return None
    st = stations.copy()
    st["sta"] = _label(st)
    elev = next(
        (c for c in ("sensor_elevation_m", "elev_m", "elevation_m") if c in st.columns), None
    )
    st["elev_m"] = st[elev].astype(float) if elev else np.nan
    depth = st["sensor_depth_m"].fillna(0.0) if "sensor_depth_m" in st.columns else 0.0
    st["borehole"] = depth > 0
    for c in ("n_usable", "n_picks"):
        if c in st.columns:
            st["n_picks"] = st[c].astype(float)
            break
    else:
        st["n_picks"] = np.nan
    st = st.drop_duplicates("sta", keep="last")
    return st[["sta", "lat", "lon", "elev_m", "borehole", "n_picks"]].reset_index(drop=True)


def _depth_axis(
    ax,
    station_elev_m: np.ndarray | None = None,
    *,
    label: str | None = "Depth below sea level (km)",
    legend: bool = True,
) -> None:
    """Invert the depth axis, mark sea level and the station elevation range."""
    handles = []
    ax.axhline(0.0, color="#4a90c2", lw=1.0, ls="--", zorder=1)
    handles.append(Line2D([], [], color="#4a90c2", ls="--", lw=1.0, label="sea level"))
    if station_elev_m is not None:
        e = np.asarray(station_elev_m, dtype=float)
        e = e[np.isfinite(e)]
        if e.size:
            ax.axhspan(-e.max() / 1000, -e.min() / 1000, color="#8c6d46", alpha=0.18, zorder=0)
            handles.append(
                plt.Rectangle((0, 0), 1, 1, color="#8c6d46", alpha=0.3, label="station elevations")
            )
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if label:
        ax.set_ylabel(label)
    if legend:
        ax._osv_depth_handles = handles  # type: ignore[attr-defined]


def _stat_ymax(ax, depths: np.ndarray, zbot: float, e: np.ndarray | None) -> None:
    top = float(depths[0])
    if e is not None and np.isfinite(e).any():
        top = min(top, -float(np.nanmax(e)) / 1000)
    ax.set_ylim(zbot, top - 0.05 * (zbot - top))


def _staircase_band(ax, depths, lo, hi, zbot, **kw) -> None:
    xl, zs = _steps(depths, lo, zbot)
    xh, _ = _steps(depths, hi, zbot)
    ax.fill_betweenx(zs, xl, xh, **kw)


def _stats(ax, text: str, loc: str = "upper right") -> None:
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    y, va = (0.96, "top") if "upper" in loc else (0.04, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8, path_effects=_HALO)


def _ref_ids(ens: Any, ids: Sequence[str] | None) -> list[str]:
    return list(ids) if ids is not None else list(ens.eligible_ids)


# ------------------------------------------------------------------ velocity-depth models


def plot_spaghetti(
    ens: Any,
    *,
    second: str = "auto",
    hypocentre_depths: Sequence[float] | None = None,
    station_elevations_m: Sequence[float] | None = None,
    zmax: float | None = None,
) -> Figure | None:
    """All trial models (grey) and final models (coloured by final RMS) versus depth.

    Panels: Vp; Vp/Vs or Vs (``second``: ``"vpvs"``, ``"vs"`` or ``"auto"`` = Vs when the ratio
    is the same in all models, e.g. a fixed ratio, else the ratio; omitted without S); and a
    histogram of the hypocentre depths (the depths the data constrain).  Selected runs have a
    dark outline, the final model is the thick black staircase with its 5-95 percentile band.

    Parameters
    ----------
    ens
        :class:`~osv.velest.ensemble.EnsembleResult`.
    hypocentre_depths
        Hypocentre depths (km) for the side panel (default: those of ``ens.closest_run_id``).
    station_elevations_m
        Station elevations (m) for the elevation band.
    """
    fm = ens.final_models
    if fm is None or fm.empty or not len(ens.eligible_ids):
        return None
    ids = list(ens.eligible_ids)
    sel = set(ens.selected_ids)
    rms = ens.runs.set_index("id")["final_rms"]
    norm = _rms_norm(rms.reindex(ids))
    cmap = plt.get_cmap(RMS_CMAP)
    z0 = _layer_depths(ens)
    zbot = _zbot(z0, zmax)
    p_fin = _pivot(fm, "P", "v_final").reindex(ids)
    p_ini = _pivot(fm, "P", "v_initial").reindex(ids)
    s_fin = _pivot(fm, "S", "v_final").reindex(ids)
    s_ini = _pivot(fm, "S", "v_initial").reindex(ids)
    has_s = not s_fin.empty and np.isfinite(s_fin.to_numpy()).any()
    ratio = p_fin / s_fin if has_s else None
    use_ratio = False
    if has_s:
        spread = float(np.nanmax(ratio.to_numpy()) - np.nanmin(ratio.to_numpy()))
        use_ratio = second == "vpvs" or (second == "auto" and spread > 0.01)
    if second == "vs":
        use_ratio = False

    if hypocentre_depths is None and len(ens.hypocentres):
        h = ens.hypocentres
        hypocentre_depths = h.loc[h["id"] == ens.closest_run_id, "depth"].to_numpy()
    hd = None if hypocentre_depths is None else np.asarray(hypocentre_depths, dtype=float)
    hd = hd[np.isfinite(hd)] if hd is not None else None
    n_pan = 1 + int(has_s) + int(hd is not None and hd.size > 0)
    widths = [3.0] + ([3.0] if has_s else []) + ([1.4] if n_pan > 1 + int(has_s) else [])
    fig = _blank((3.6 * n_pan + 1.2, 6.2))
    gs = fig.add_gridspec(1, n_pan, width_ratios=widths)
    axes = [_axes(fig, gs[0, 0])]
    for k in range(1, n_pan):
        axes.append(_axes(fig, gs[0, k], sharey=axes[0]))
    elev = None if station_elevations_m is None else np.asarray(station_elevations_m, float)

    def draw(ax, ini, fin, band, final, xlabel):
        for i in ids:
            if i in ini.index and np.isfinite(ini.loc[i].to_numpy()).all():
                ax.plot(*_steps(z0, ini.loc[i].to_numpy(), zbot), color=GREY, lw=0.4, alpha=0.5)
        order = sorted(ids, key=lambda i: -rms.get(i, np.inf))  # best last, on top
        for sel_pass in (False, True):
            for i in order:
                if (i in sel) != sel_pass:
                    continue
                v = fin.loc[i].to_numpy()
                if not np.isfinite(v).all():
                    continue
                x, z = _steps(z0, v, zbot)
                if sel_pass:
                    ax.plot(x, z, color="k", lw=1.9, alpha=0.9, zorder=3)
                ax.plot(
                    x,
                    z,
                    color=cmap(norm(rms[i])),
                    lw=1.2 if sel_pass else 0.7,
                    alpha=1.0 if sel_pass else 0.8,
                    zorder=4 if sel_pass else 2,
                )
        if band is not None:
            _staircase_band(
                ax, z0, band[0], band[1], zbot, color=SELECTED_COLOR, alpha=0.28, lw=0, zorder=5
            )
        ax.plot(*_steps(z0, final, zbot), color="k", lw=3.0, zorder=6)
        ax.set_xlabel(xlabel)
        _depth_axis(ax, elev, label=None)

    fmod = ens.final_model
    draw(axes[0], p_ini, p_fin, (fmod["vp_p05"], fmod["vp_p95"]), fmod["vp"], "Vp (km/s)")
    axes[0].set_ylabel("Depth below sea level (km)")
    if has_s:
        if use_ratio:
            sel_r = ratio.loc[[i for i in ens.selected_ids if i in ratio.index]]
            r0 = (p_ini / s_ini) if not s_ini.empty else p_ini * np.nan
            band = (np.nanpercentile(sel_r, 5, axis=0), np.nanpercentile(sel_r, 95, axis=0))
            draw(axes[1], r0, ratio, band, fmod["vpvs"], "Vp/Vs")
        else:
            draw(axes[1], s_ini, s_fin, (fmod["vs_p05"], fmod["vs_p95"]), fmod["vs"], "Vs (km/s)")
        axes[1].tick_params(labelleft=False)
    if n_pan > 1 + int(has_s):
        ah = axes[-1]
        top = min(float(z0[0]), -float(np.nanmax(elev)) / 1000 if elev is not None else z0[0])
        bins = np.arange(math.floor(top * 4) / 4, zbot + 0.25, 0.25)
        ah.hist(hd, bins=bins, orientation="horizontal", color="#7f7f7f", alpha=0.85)
        ah.set_xlabel("Hypocentres")
        ah.tick_params(labelleft=False)
        ah.axhline(0.0, color="#4a90c2", lw=1.0, ls="--")
        ah.set_title("depth coverage", fontsize=9)
    _stat_ymax(axes[0], z0, zbot, elev)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, location="right", shrink=0.8, pad=0.02, extend="max")
    cb.set_label("final RMS residual (s)")
    handles = [
        Line2D([], [], color=GREY, lw=1, label="initial models"),
        Line2D([], [], color=cmap(0.4), lw=1.2, label="final models"),
        Line2D(
            [],
            [],
            color=cmap(0.4),
            lw=1.9,
            path_effects=[patheffects.withStroke(linewidth=3.4, foreground="k")],
            label=f"selected ({len(sel)})",
        ),
        Line2D([], [], color="k", lw=3, label=f"final model ({ens.select.statistic})"),
        plt.Rectangle((0, 0), 1, 1, color=SELECTED_COLOR, alpha=0.28, label="p05-p95"),
        *getattr(axes[0], "_osv_depth_handles", []),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7)
    fig.suptitle(f"Monte Carlo 1-D models: {len(ids)} eligible runs")
    return fig


def plot_velocity_density(
    ens: Any,
    best_percent: float = 10.0,
    *,
    station_elevations_m: Sequence[float] | None = None,
    zmax: float | None = None,
) -> Figure | None:
    """Density of the best ``best_percent`` % of the final Vp models (velocity vs depth).

    Each model is sampled on a fine depth grid; the colour is the fraction of those models
    with a velocity in each bin.  The final model and its p05-p95 band are overlaid.
    """
    fm = ens.final_models
    if fm is None or fm.empty or not len(ens.eligible_ids):
        return None
    n = max(1, math.ceil(best_percent / 100 * len(ens.eligible_ids)))
    ids = ens.eligible_ids[:n]
    z0 = _layer_depths(ens)
    zbot = _zbot(z0, zmax)
    vp = _pivot(fm, "P", "v_final").reindex(ids).dropna()
    if vp.empty:
        return None
    z = np.linspace(z0[0], zbot - 1e-6, 240)
    vmin, vmax = float(np.nanmin(vp)), float(np.nanmax(vp))
    pad = 0.05 * max(vmax - vmin, 0.5)
    vbins = np.linspace(vmin - pad, vmax + pad, 100)
    dens = np.zeros((len(z), len(vbins) - 1))
    for row in vp.to_numpy():
        v = _eval_steps(z0, row, z)
        ix = np.clip(np.digitize(v, vbins) - 1, 0, len(vbins) - 2)
        dens[np.arange(len(z)), ix] += 1
    dens /= len(vp)
    fig, ax = new_figure(figsize=(5.2, 6.2))
    ax.grid(False)
    dz = z[1] - z[0]
    mesh = ax.pcolormesh(
        vbins,
        np.append(z - dz / 2, z[-1] + dz / 2),
        np.ma.masked_equal(dens, 0),
        cmap="Blues",
        vmin=0,
        vmax=max(float(dens.max()), 1e-9),
        shading="flat",
    )
    f = ens.final_model
    _staircase_band(ax, z0, f["vp_p05"], f["vp_p95"], zbot, color=SELECTED_COLOR, alpha=0.2, lw=0)
    ax.plot(*_steps(z0, f["vp"], zbot), color="k", lw=2.2, label="final model")
    ax.plot(*_steps(z0, f["vp_p05"], zbot), color=SELECTED_COLOR, lw=0.8, ls=":")
    ax.plot(
        *_steps(z0, f["vp_p95"], zbot),
        color=SELECTED_COLOR,
        lw=0.8,
        ls=":",
        label="final p05 / p95",
    )
    elev = None if station_elevations_m is None else np.asarray(station_elevations_m, float)
    _depth_axis(ax, elev)
    _stat_ymax(ax, z0, zbot, elev)
    ax.set_xlabel("Vp (km/s)")
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cb.set_label("fraction of models")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + getattr(ax, "_osv_depth_handles", []), loc="lower left")
    ax.set_title(f"Best {len(vp)} of {len(ens.eligible_ids)} models ({best_percent:g}%)")
    return fig


def _grid_axes(n: int, ncols: int, cell: tuple[float, float] = (2.8, 2.3)):
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axs = new_figure(
        nrows=nrows,
        ncols=ncols,
        figsize=(cell[0] * ncols + 0.6, cell[1] * nrows + 0.8),
        squeeze=False,
    )
    flat = axs.ravel()
    for ax in flat[n:]:
        ax.set_axis_off()
    return fig, flat


def _layer_bounds(ens: Any, vp_bounds: Sequence[Sequence[float]] | None) -> np.ndarray | None:
    if vp_bounds is not None:
        return np.asarray(vp_bounds, dtype=float)
    return None


def _layer_title(ens: Any, k: int) -> str:
    d = ens.final_model["depth_top"]
    return f"layer {k + 1}  (top {d.iloc[k]:g} km)"


def plot_layer_histograms(
    ens: Any, vp_bounds: Sequence[Sequence[float]] | None = None, *, ncols: int = 4
) -> Figure | None:
    """Histogram of the final Vp of every layer (one subplot per layer).

    All eligible runs are grey, the selected runs coloured, the final model value is the black
    line and the sampled search bounds (``vp_bounds``) are red dashed lines.
    """
    fm = ens.final_models
    if fm is None or fm.empty:
        return None
    vp = _pivot(fm, "P", "v_final").reindex(ens.eligible_ids)
    if vp.empty:
        return None
    b = _layer_bounds(ens, vp_bounds)
    n = vp.shape[1]
    fig, axs = _grid_axes(n, ncols)
    sel = [i for i in ens.selected_ids if i in vp.index]
    for k, ax in enumerate(axs[:n]):
        vals = vp[k].dropna().to_numpy()
        sv = vp.loc[sel, k].dropna().to_numpy()
        lo = min(vals.min(), b[k, 0] if b is not None else np.inf)
        hi = max(vals.max(), b[k, 1] if b is not None else -np.inf)
        pad = 0.04 * max(hi - lo, 0.1)
        bins = np.linspace(lo - pad, hi + pad, 21)
        ax.hist(vals, bins=bins, color=GREY, label="all eligible")
        ax.hist(sv, bins=bins, color=SELECTED_COLOR, alpha=0.9, label="selected")
        ax.axvline(ens.final_model["vp"].iloc[k], color="k", lw=1.6, label="final model")
        if b is not None:
            for j, x in enumerate(b[k]):
                ax.axvline(
                    x, color="#cc0000", ls="--", lw=1.2, label="sampled bounds" if j == 0 else None
                )
        ax.set_title(_layer_title(ens, k), fontsize=9)
        ax.set_xlabel("Vp (km/s)", fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axs[:n:ncols]:
        ax.set_ylabel("runs")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=len(h), fontsize=8)
    fig.suptitle("Final Vp per layer")
    return fig


def plot_rms_vs_layer_velocity(
    ens: Any, vp_bounds: Sequence[Sequence[float]] | None = None, *, ncols: int = 4
) -> Figure | None:
    """Final RMS versus each layer's final Vp (scatter grid) to expose trade-offs.

    A flat cloud means the layer is poorly constrained, a clear trend (Spearman rho in the
    corner) or a sharp minimum means the data resolve it.  Selected runs are orange, the best
    run is a star, bounds are red dashed lines.
    """
    fm = ens.final_models
    if fm is None or fm.empty:
        return None
    vp = _pivot(fm, "P", "v_final").reindex(ens.eligible_ids)
    if vp.empty:
        return None
    rms = ens.runs.set_index("id")["final_rms"].reindex(vp.index)
    b = _layer_bounds(ens, vp_bounds)
    n = vp.shape[1]
    fig, axs = _grid_axes(n, ncols)
    sel = vp.index.isin(ens.selected_ids)
    r_all = rms.to_numpy(dtype=float)
    r_ok = r_all[np.isfinite(r_all)]
    ymin, p90 = float(r_ok.min()), float(np.percentile(r_ok, 90))
    ytop = p90 + 0.6 * max(p90 - ymin, 1e-4)
    ylim = (ymin - 0.12 * (ytop - ymin), ytop)
    n_clip = int((r_all > ytop).sum())
    for k, ax in enumerate(axs[:n]):
        ax.set_ylim(*ylim)
        v = vp[k].to_numpy()
        ax.scatter(v[~sel], rms[~sel], s=9, color=GREY, alpha=0.8, lw=0)
        ax.scatter(v[sel], rms[sel], s=14, color=SELECTED_COLOR, lw=0)
        if ens.best_run_id in vp.index:
            ax.scatter(
                vp.loc[ens.best_run_id, k], ens.best_rms, marker="*", s=70, color="k", zorder=5
            )
        if b is not None:
            for x in b[k]:
                ax.axvline(x, color="#cc0000", ls="--", lw=1.0)
        m = np.isfinite(v) & np.isfinite(rms.to_numpy())
        if m.sum() > 3 and np.ptp(v[m]) > 0:
            from scipy.stats import spearmanr

            rho = spearmanr(v[m], rms.to_numpy()[m])[0]
            _stats(ax, f"rho = {rho:+.2f}")
        ax.set_title(_layer_title(ens, k), fontsize=9)
        ax.set_xlabel("Vp (km/s)", fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axs[:n:ncols]:
        ax.set_ylabel("final RMS (s)", fontsize=8)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=GREY, label="eligible"),
            Line2D([], [], marker="o", ls="", color=SELECTED_COLOR, label="selected"),
            Line2D([], [], marker="*", ls="", color="k", ms=9, label="best"),
        ],
        loc="outside lower center",
        ncols=3,
        fontsize=8,
    )
    fig.suptitle(
        "Final RMS vs final layer velocity"
        + (f" ({n_clip} runs above the plotted range)" if n_clip else "")
    )
    return fig


# --------------------------------------------------------------------------- convergence


def _iteration_curve(parsed: Any) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    it = getattr(parsed, "iterations", None)
    if it is None or len(it) == 0 or "rms" not in it.columns:
        return None
    acc = it[(it["kind"] == "initial") | it["accepted"].astype(bool)]
    acc = acc.drop_duplicates("iteration", keep="last").sort_values("iteration")
    back = it[it["kind"] == "backup"]
    return acc, back


def plot_rms_vs_iteration(
    runs: pd.DataFrame, selected_ids: Sequence[str] | None = None
) -> Figure | None:
    """RMS residual versus iteration for every run (lines), with backup steps marked.

    Parameters
    ----------
    runs
        Table with ``id`` and ``parsed`` (:class:`~osv.io.velest_out.VelestRunResult`), e.g.
        ``WorkflowResult.runs`` (the last damping stage of every run) or ``run_jobs`` output.
    selected_ids
        Runs drawn on top in a heavier line.
    """
    if runs is None or "parsed" not in runs.columns:
        return None
    curves = []
    for r in runs.itertuples():
        c = _iteration_curve(getattr(r, "parsed", None))
        if c is not None and len(c[0]) > 1:
            curves.append((str(r.id), *c))
    if not curves:
        return None
    final = {i: a["rms"].iloc[-1] for i, a, _ in curves}
    norm = _rms_norm(list(final.values()))
    cmap = plt.get_cmap(RMS_CMAP)
    sel = set(selected_ids or [])
    fig, (ax, axz) = new_figure(ncols=2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    bx, by = [], []
    for i, a, b in sorted(curves, key=lambda c: -final[c[0]]):
        w = 1.6 if i in sel else 0.7
        for axx in (ax, axz):
            axx.plot(
                a["iteration"],
                a["rms"],
                color=cmap(norm(final[i])),
                lw=w,
                alpha=1.0 if i in sel else 0.7,
                zorder=3 if i in sel else 2,
            )
        bx.extend(b["iteration"])
        by.extend(b["rms"])
    for axx in (ax, axz):
        axx.scatter(bx, by, marker="x", s=10, color="#cc0000", lw=0.6, alpha=0.35, zorder=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMS residual (s)")
    ax.set_title(f"{len(curves)} runs")
    fv = np.asarray(list(final.values()), dtype=float)
    lo, p60 = float(fv.min()), float(np.percentile(fv, 60))
    axz.set_ylim(lo - 0.15 * (p60 - lo + 1e-4), p60 + 0.15 * (p60 - lo + 1e-4))
    axz.set_xlabel("iteration")
    axz.set_title("zoom on the best 60% of runs")
    ax.legend(
        handles=[
            Line2D([], [], color=cmap(0.3), label="run (colour: final RMS)"),
            Line2D([], [], marker="x", ls="", color="#cc0000", label="backup step"),
        ],
        loc="upper right",
    )
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axz, label="final RMS (s)")
    return fig


def plot_stage_rms(history: pd.DataFrame) -> Figure | None:
    """Per-stage distributions of final RMS and iteration count (workflow history).

    Parameters
    ----------
    history
        ``WorkflowResult.history`` (``round, stage_index, stage, status, final_rms,
        n_iterations``); one box per ``(round, stage)`` with the runs as points.
    """
    if history is None or len(history) == 0:
        return None
    h = history.copy()
    keys = (
        h[["round", "stage_index", "stage"]].drop_duplicates().sort_values(["round", "stage_index"])
    )
    labels = [
        f"R{r.round} S{r.stage_index}\n{str(r.stage).replace('_', ' ')}" for r in keys.itertuples()
    ]
    fig, axs = new_figure(ncols=2, figsize=(max(7.0, 1.6 * len(keys) + 3), 4.4))
    rng = np.random.default_rng(0)
    for ax, col, ylabel in (
        (axs[0], "final_rms", "final RMS (s)"),
        (axs[1], "n_iterations", "iterations"),
    ):
        data = []
        for r in keys.itertuples():
            g = h[(h["round"] == r.round) & (h["stage_index"] == r.stage_index)]
            data.append(g.loc[g["status"] == "ok", col].dropna().to_numpy(dtype=float))
        pos = np.arange(len(keys))
        nonempty = [i for i, d in enumerate(data) if d.size]
        if nonempty:
            ax.boxplot(
                [data[i] for i in nonempty],
                positions=pos[nonempty],
                widths=0.5,
                showfliers=False,
                medianprops={"color": "k"},
            )
        for i, d in enumerate(data):
            ax.scatter(
                i + rng.uniform(-0.18, 0.18, d.size),
                d,
                s=7,
                color=SELECTED_COLOR,
                alpha=0.6,
                lw=0,
                zorder=3,
            )
        ax.set_xticks(pos, labels, fontsize=7)
        ax.set_ylabel(ylabel)
    axs[0].set_title("final RMS per damping stage")
    axs[1].set_title("iterations per stage run")
    return fig


def classify_runs(runs: pd.DataFrame) -> pd.Series:
    """Outcome per run: failed > diverged > four_backups > converged > ok (ran to the limit)."""
    out = pd.Series("ok", index=runs.index, dtype=object)
    out[runs["converged"].astype(bool)] = "converged"
    if "stop_reason" in runs.columns:
        out[runs["stop_reason"] == "four_backups"] = "four_backups"
    out[runs["diverged"].astype(bool)] = "diverged"
    out[runs["failed"].astype(bool)] = "failed"
    return out


def plot_convergence(ens: Any) -> Figure | None:
    """Convergence/failure summary: outcome bars, stop reasons, and where failures start.

    Panels: runs per outcome (ok = finished without converging or failing, converged,
    four_backups, diverged, failed); runs per VELEST stop reason; and the initial Vp models
    coloured by outcome ("map" of which starting models fail).
    """
    runs = ens.runs
    if runs is None or len(runs) == 0:
        return None
    outcome = classify_runs(runs)
    counts = outcome.value_counts().reindex(OUTCOMES, fill_value=0)
    reasons = runs["stop_reason"].fillna("none").where(~runs["failed"].astype(bool), "failed")
    rc = reasons.value_counts()
    fig = _blank((12.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.3])
    a1, a2, a3 = (_axes(fig, gs[0, k]) for k in range(3))
    a1.bar(counts.index, counts.values, color=[OUTCOME_COLORS[o] for o in counts.index])
    for x, v in enumerate(counts.values):
        a1.text(x, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    a1.set_ylabel("runs")
    a1.set_title(f"outcome (convergence rate {ens.convergence_rate:.0%})")
    a1.tick_params(axis="x", rotation=30)
    a2.barh(rc.index[::-1], rc.values[::-1], color="#56b4e9")
    for y, v in enumerate(rc.values[::-1]):
        a2.text(v, y, f" {int(v)}", va="center", fontsize=8)
    a2.set_xlabel("runs")
    a2.set_title("stop reason")
    fm = ens.final_models
    p0 = _pivot(fm, "P", "v_initial") if fm is not None and len(fm) else pd.DataFrame()
    if not p0.empty:
        z0 = _layer_depths(ens)
        zbot = _zbot(z0)
        oc = outcome.set_axis(runs["id"]).reindex(p0.index)
        for o in ("ok", "converged", "four_backups", "diverged", "failed"):
            for i in p0.index[oc == o]:
                a3.plot(
                    *_steps(z0, p0.loc[i].to_numpy(), zbot),
                    color=OUTCOME_COLORS[o],
                    lw=0.5 if o in ("ok", "converged") else 1.2,
                    alpha=0.5 if o in ("ok", "converged") else 0.95,
                )
        _depth_axis(a3)
        a3.set_ylim(zbot, z0[0])
        a3.set_xlabel("initial Vp (km/s)")
        a3.set_title("initial models by outcome")
        a3.legend(
            handles=[
                Line2D([], [], color=OUTCOME_COLORS[o], label=o)
                for o in OUTCOMES
                if (oc == o).any()
            ],
            fontsize=7,
        )
    else:
        a3.set_axis_off()
    return fig


# ------------------------------------------------------------------------- station delays


def _delay_cols(phase: str) -> tuple[str, str, str | None, str | None]:
    if phase == "P":
        return "ptcor_median", "ptcor_std", "ptcor_p05", "ptcor_p95"
    return "stcor_median", "stcor_std", None, None


def _delay_frame(ens: Any, stations: pd.DataFrame | None) -> pd.DataFrame | None:
    ss = ens.station_summary
    if ss is None or len(ss) == 0:
        return None
    df = ss.copy()
    st = _stations(stations)
    if st is not None:
        df = df.merge(st, on="sta", how="left")
    else:
        for c in ("lat", "lon", "elev_m"):
            df[c] = np.nan
        df["borehole"] = False
        df["n_picks"] = np.nan
    df["borehole"] = df["borehole"].fillna(False).astype(bool)
    n = df["nobs"].to_numpy(dtype=float) if "nobs" in df else np.zeros(len(df))
    df["size_n"] = np.where(n > 0, n, df["n_picks"])
    return df


def _sizes(n: pd.Series | np.ndarray, lo: float = 25.0, hi: float = 230.0) -> np.ndarray:
    a = np.asarray(n, dtype=float)
    if not np.isfinite(a).any() or np.nanmax(a) <= 0:
        return np.full(a.shape, 70.0)
    a = np.where(np.isfinite(a), a, np.nanmin(a))
    return lo + (hi - lo) * np.sqrt(a / np.nanmax(a))


def plot_station_delay_map(
    ens: Any, stations: pd.DataFrame | None, reference_station: str | None = None
) -> Figure | None:
    """Station delays (P, and S when inverted) on a map.

    Colour is the median delay over the selected runs (diverging, red = late/slow), size is
    the number of observations (picks when the run output does not give them), the reference
    station has a thick ring, borehole stations are down-triangles.  Returns ``None`` without
    station coordinates.
    """
    df = _delay_frame(ens, stations)
    if df is None or not df[["lat", "lon"]].notna().all(axis=1).any():
        return None
    ref = reference_station or getattr(ens, "reference_station", None)
    phases = [p for p in ("P", "S") if df[_delay_cols(p)[0]].notna().any()]
    if not phases:
        return None
    fig, axs = new_figure(ncols=len(phases), figsize=(6.6 * len(phases), 6.0), squeeze=False)
    d = df.dropna(subset=["lat", "lon"])
    for ax, ph in zip(axs[0], phases, strict=True):
        col = _delay_cols(ph)[0]
        dd = d.dropna(subset=[col])
        vmax = max(float(np.abs(dd[col]).max()), 0.05)
        norm = perturbation_norm(vmax=vmax)
        sc = None
        for bh, mk in ((False, "o"), (True, "v")):
            m = dd[dd["borehole"] == bh]
            if len(m):
                sc = ax.scatter(
                    m["lon"],
                    m["lat"],
                    c=m[col],
                    cmap=DELAY_CMAP,
                    norm=norm,
                    marker=mk,
                    s=_sizes(m["size_n"]),
                    edgecolors="k",
                    linewidths=0.7,
                    zorder=4,
                )
        for r in dd.itertuples():
            ax.annotate(
                r.sta,
                (r.lon, r.lat),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=5.5,
                path_effects=_HALO,
                zorder=6,
            )
        if ref is not None:
            rr = d[d["sta"] == ref]
            if len(rr):
                ax.scatter(
                    rr["lon"],
                    rr["lat"],
                    s=300,
                    facecolors="none",
                    edgecolors="k",
                    linewidths=2.2,
                    zorder=5,
                )
        setup_map_axes(ax)
        ax.set_aspect(1.0 / math.cos(math.radians(float(d["lat"].mean()))))
        if sc is not None:
            fig.colorbar(sc, ax=ax, shrink=0.75, label=f"{ph} delay (s, positive = late)")
        ax.set_title(f"{ph} station delays (median of {len(ens.selected_ids)} selected runs)")
    handles = [
        Line2D([], [], marker="o", ls="", mfc="0.8", mec="k", label="surface station"),
        Line2D([], [], marker="v", ls="", mfc="0.8", mec="k", label="borehole station"),
    ]
    if ref is not None:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="none",
                mec="k",
                mew=2,
                ms=11,
                label=f"reference ({ref})",
            )
        )
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=8)
    return fig


def plot_delay_vs_elevation(
    ens: Any, stations: pd.DataFrame | None, reference_station: str | None = None
) -> Figure | None:
    """Station delay versus station (sensor) elevation, P and S, with spread over selected runs.

    Error bars are the 5-95 percentile range (P) or one standard deviation (S) over the
    selected runs; the Spearman correlation is annotated.  Returns ``None`` without station
    elevations.
    """
    df = _delay_frame(ens, stations)
    if df is None or not df["elev_m"].notna().any():
        return None
    ref = reference_station or getattr(ens, "reference_station", None)
    phases = [p for p in ("P", "S") if df[_delay_cols(p)[0]].notna().any()]
    if not phases:
        return None
    fig, axs = new_figure(ncols=len(phases), figsize=(5.6 * len(phases), 4.6), squeeze=False)
    for ax, ph in zip(axs[0], phases, strict=True):
        med, sd, p05, p95 = _delay_cols(ph)
        d = df.dropna(subset=["elev_m", med])
        if p05 is not None:
            err = np.vstack([(d[med] - d[p05]).clip(lower=0), (d[p95] - d[med]).clip(lower=0)])
        else:
            err = d[sd].fillna(0).to_numpy()
        ax.errorbar(
            d["elev_m"],
            d[med],
            yerr=err,
            fmt="none",
            ecolor="0.55",
            elinewidth=0.9,
            capsize=2,
            zorder=2,
        )
        for bh, mk in ((False, "o"), (True, "v")):
            m = d[d["borehole"] == bh]
            ax.scatter(
                m["elev_m"],
                m[med],
                marker=mk,
                s=34,
                color=PHASE_COLORS[ph],
                ec="k",
                lw=0.5,
                zorder=3,
                label="borehole" if bh else "surface",
            ) if len(m) else None
        if ref is not None and (d["sta"] == ref).any():
            r = d[d["sta"] == ref].iloc[0]
            ax.scatter(
                [r["elev_m"]],
                [r[med]],
                s=200,
                facecolors="none",
                edgecolors="k",
                lw=1.8,
                zorder=4,
                label=f"reference ({ref})",
            )
        ax.axhline(0, color="k", lw=0.6)
        if len(d) > 3 and d["elev_m"].nunique() > 1:
            from scipy.stats import spearmanr

            _stats(ax, f"rho = {spearmanr(d['elev_m'], d[med])[0]:+.2f}  (n = {len(d)})")
        ax.set_xlabel("station sensor elevation (m)")
        ax.set_ylabel(f"{ph} delay (s)")
        ax.set_title(f"{ph} delay vs elevation")
        ax.legend(loc="lower left", fontsize=7)
    return fig


# ------------------------------------------------------------------------- hypocentres


def _input_frame(inputs: pd.DataFrame | None) -> pd.DataFrame | None:
    """Starting hypocentres with a 0-based ``event`` column and ``lat0, lon0, depth0``."""
    if inputs is None or len(inputs) == 0:
        return None
    d = inputs.copy()
    if "event" not in d.columns:
        if "cnv_id" in d.columns:
            d["event"] = d["cnv_id"].astype(int) - 1
        else:
            d["event"] = np.arange(len(d))
    dcol = "depth_km" if "depth_km" in d.columns else "depth"
    return d.rename(columns={"lat": "lat0", "lon": "lon0", dcol: "depth0"})[
        ["event", "lat0", "lon0", "depth0"]
    ]


def _shifts(ens: Any, inputs: pd.DataFrame | None, ids: Sequence[str]) -> pd.DataFrame | None:
    """Per event: starting position, mean shift over ``ids`` (km) and spread."""
    h = ens.hypocentres
    if h is None or len(h) == 0:
        return None
    h = h[h["id"].isin(list(ids))]
    if h.empty:
        return None
    inp = _input_frame(inputs)
    if inp is None:
        if "ddepth_km" not in h.columns:
            return None
        h = h.copy()
        lat0 = h["lat"] - h["dlat_km"] / KM_PER_DEG
        h["lat0"] = lat0
        h["lon0"] = h["lon"] - h["dlon_km"] / (KM_PER_DEG * np.cos(np.radians(lat0)))
        h["depth0"] = h["depth"] - h["ddepth_km"]
    else:
        h = h.drop(
            columns=[c for c in h.columns if c.endswith("0") or c.endswith("_km")], errors="ignore"
        ).merge(inp, on="event", how="left")
        h["dlat_km"] = (h["lat"] - h["lat0"]) * KM_PER_DEG
        h["dlon_km"] = (h["lon"] - h["lon0"]) * KM_PER_DEG * np.cos(np.radians(h["lat0"]))
        h["ddepth_km"] = h["depth"] - h["depth0"]
    g = h.groupby("event")
    out = g[["lat0", "lon0", "depth0", "dlat_km", "dlon_km", "ddepth_km"]].mean()
    out["ddepth_std"] = g["ddepth_km"].std(ddof=0)
    out["dh_std"] = np.hypot(g["dlat_km"].std(ddof=0), g["dlon_km"].std(ddof=0))
    out["dh_km"] = np.hypot(out["dlat_km"], out["dlon_km"])
    return out.dropna(subset=["lat0", "lon0"]).reset_index()


def plot_hypocentre_shifts(
    ens: Any,
    input_hypocentres: pd.DataFrame | None = None,
    stations: pd.DataFrame | None = None,
    *,
    ids: Sequence[str] | None = None,
) -> Figure | None:
    """Hypocentre shifts (starting vs VELEST-relocated), mean over the selected runs.

    Panels: map with arrows (starting epicentre to mean relocated epicentre, **exaggerated** by
    the factor in the title so they are visible; colour = depth change) and histograms of the
    depth change and horizontal shift.  The grey outline histograms are the spread (standard
    deviation) of the shift between the selected runs.  Needs the starting hypocentres
    (``input_hypocentres``) unless the ensemble table already holds the shifts.
    """
    sh = _shifts(ens, input_hypocentres, _ref_ids(ens, ids) if ids else ens.selected_ids)
    if sh is None or sh.empty:
        return None
    fig = _blank((12.0, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax = _axes(fig, gs[0, 0])
    lat0 = float(sh["lat0"].mean())
    cl = math.cos(math.radians(lat0))
    width_km = max(
        float(np.ptp(sh["lon0"])) * KM_PER_DEG * cl, float(np.ptp(sh["lat0"])) * KM_PER_DEG, 1.0
    )
    med = max(float(sh["dh_km"].median()), 1e-3)
    ex = max(1.0, round(0.04 * width_km / med, -int(math.floor(math.log10(0.04 * width_km / med)))))
    dx = ex * sh["dlon_km"] / (KM_PER_DEG * cl)
    dy = ex * sh["dlat_km"] / KM_PER_DEG
    vmax = max(float(sh["ddepth_km"].abs().quantile(0.95)), 0.05)
    norm = perturbation_norm(vmax=vmax)
    st = _stations(stations)
    if st is not None:
        ax.scatter(
            st["lon"],
            st["lat"],
            marker="^",
            s=22,
            c="white",
            ec="k",
            lw=0.6,
            zorder=2,
            label="stations",
        )
    ax.scatter(sh["lon0"], sh["lat0"], s=5, color="0.4", zorder=3, label="starting epicentres")
    q = ax.quiver(
        sh["lon0"],
        sh["lat0"],
        dx,
        dy,
        sh["ddepth_km"],
        cmap="RdBu",
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        zorder=4,
    )
    ax.set_xlim(
        sh["lon0"].min() - 0.15 * np.ptp(sh["lon0"]) - 0.002,
        sh["lon0"].max() + 0.15 * np.ptp(sh["lon0"]) + 0.002,
    )
    ax.set_ylim(
        sh["lat0"].min() - 0.15 * np.ptp(sh["lat0"]) - 0.002,
        sh["lat0"].max() + 0.15 * np.ptp(sh["lat0"]) + 0.002,
    )
    setup_map_axes(ax)
    add_scale_bar(ax)
    cb = fig.colorbar(q, ax=ax, shrink=0.75)
    cb.set_label("mean depth change (km, + = deeper)")
    ax.set_title(f"epicentre shifts (arrows x{ex:g}), {len(sh)} events")
    ax.legend(loc="upper left", fontsize=7)
    bz, bh = _axes(fig, gs[0, 1]), _axes(fig, gs[0, 2])
    bins_z = np.histogram_bin_edges(np.r_[sh["ddepth_km"], sh["ddepth_std"].dropna()], 30)
    bins_h = np.histogram_bin_edges(np.r_[sh["dh_km"], sh["dh_std"].dropna()], 30)
    bz.hist(sh["ddepth_km"], bins=bins_z, color="#7f7f7f")
    bz.hist(
        sh["ddepth_std"].dropna(),
        bins=bins_z,
        histtype="step",
        color="k",
        lw=1.2,
        label="spread between runs (1 sd)",
    )
    bz.axvline(0, color="k", lw=0.6)
    bz.set_xlabel("depth change (km)")
    bz.set_ylabel("events")
    _stats(
        bz,
        f"mean {sh['ddepth_km'].mean():+.2f} km\nmedian {sh['ddepth_km'].median():+.2f} km\n"
        f"sd {sh['ddepth_km'].std():.2f} km",
    )
    bz.legend(loc="upper left", fontsize=7)
    bz.set_title("depth change")
    bh.hist(sh["dh_km"], bins=bins_h, color="#7f7f7f")
    bh.hist(sh["dh_std"].dropna(), bins=bins_h, histtype="step", color="k", lw=1.2)
    bh.set_xlabel("horizontal shift (km)")
    _stats(bh, f"median {sh['dh_km'].median():.2f} km\np90 {sh['dh_km'].quantile(0.9):.2f} km")
    bh.set_title("horizontal shift")
    return fig


def plot_final_hypocentres(
    hypocentres: pd.DataFrame,
    stations: pd.DataFrame | None = None,
    *,
    title: str = "Final hypocentres",
) -> Figure | None:
    """Final hypocentres: map plus E-W and N-S depth sections (colour = depth).

    Parameters
    ----------
    hypocentres
        One row per event with ``lat``, ``lon`` and ``depth`` (or ``depth_km``) and optional
        ``rms`` (used for the section marker size); e.g. the closest run's rows of
        ``EnsembleResult.hypocentres`` or ``FixedModelResult.hypocentres``.
    """
    if hypocentres is None or len(hypocentres) == 0:
        return None
    h = hypocentres.copy()
    dcol = "depth" if "depth" in h.columns else "depth_km"
    h = h.dropna(subset=["lat", "lon", dcol])
    if h.empty:
        return None
    z = h[dcol].to_numpy(dtype=float)
    fig = _blank((8.6, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.2], height_ratios=[3, 1.3])
    am = _axes(fig, gs[0, 0])
    ae = _axes(fig, gs[1, 0], sharex=am)
    an = _axes(fig, gs[0, 1], sharey=am)
    ac = _axes(fig, gs[1, 1])
    ac.set_axis_off()
    st = _stations(stations)
    vmin, vmax = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    kw = dict(c=z, cmap="plasma_r", vmin=vmin, vmax=max(vmax, vmin + 0.1), s=11, ec="k", lw=0.2)
    sc = am.scatter(h["lon"], h["lat"], zorder=4, **kw)
    ae.scatter(h["lon"], z, zorder=4, **kw)
    an.scatter(z, h["lat"], zorder=4, **kw)
    px = 0.15 * max(np.ptp(h["lon"]), 0.01)
    py = 0.15 * max(np.ptp(h["lat"]), 0.01)
    xl = (h["lon"].min() - px, h["lon"].max() + px)
    yl = (h["lat"].min() - py, h["lat"].max() + py)
    if st is not None:
        inside = st[st["lon"].between(*xl) & st["lat"].between(*yl)]
        am.scatter(
            inside["lon"],
            inside["lat"],
            marker="^",
            s=40,
            c="white",
            ec="k",
            lw=0.8,
            zorder=5,
            label="stations",
        )
        if inside["elev_m"].notna().any():
            ae.scatter(
                inside["lon"],
                -inside["elev_m"] / 1000,
                marker="^",
                s=40,
                c="white",
                ec="k",
                lw=0.8,
                zorder=5,
            )
            an.scatter(
                -inside["elev_m"] / 1000,
                inside["lat"],
                marker="^",
                s=40,
                c="white",
                ec="k",
                lw=0.8,
                zorder=5,
            )
        am.legend(loc="upper left", fontsize=7)
    am.set_xlim(*xl)
    am.set_ylim(*yl)
    setup_map_axes(am)
    am.set_xlabel("")
    am.tick_params(labelbottom=False)
    add_scale_bar(am)
    top = min(z.min(), 0.0)
    if st is not None and st["elev_m"].notna().any():
        top = min(top, -float(st["elev_m"].max()) / 1000)
    zlo, zhi = top - 0.15, z.max() + 0.2
    ae.set_ylim(zhi, zlo)
    an.set_xlim(zlo, zhi)
    ae.axhline(0, color="#4a90c2", ls="--", lw=1.0)
    an.axvline(0, color="#4a90c2", ls="--", lw=1.0)
    ae.set_xlabel("Longitude (deg)")
    ae.set_ylabel("Depth (km)")
    an.set_xlabel("Depth (km)")
    an.tick_params(labelleft=False)
    am.set_title(f"{title}: {len(h)} events")
    ae.set_title("E-W section", fontsize=9)
    an.set_title("N-S section", fontsize=9)
    fig.colorbar(sc, ax=ac, location="top", fraction=0.35, label="depth (km)")
    return fig


# ------------------------------------------------------------------------------ stability


def _need(stab: Any, test: str) -> bool:
    runs = getattr(stab, "runs", None)
    return runs is not None and len(runs) > 0 and bool((runs["test"] == test).any())


def plot_hypocentre_perturbation(stab: Any) -> Figure | None:
    """Hypocentre-perturbation test: recovered mislocation with the model fixed vs free.

    Histograms of the horizontal (``dh``) and vertical (``dz``) distance between each
    perturbed run's relocated hypocentres and the unperturbed fixed-model solution.  With the
    model **fixed** only the relocation algorithm is tested; with the model **free** the
    model can absorb the perturbation.  Median lines are drawn.
    """
    if not _need(stab, "hypocentre_perturbation") or len(stab.events) == 0:
        return None
    ev = stab.events[stab.events["test"] == "hypocentre_perturbation"]
    variants = [v for v in ("fixed", "free") if (ev["variant"] == v).any()]
    if not variants:
        return None
    colors = {"fixed": "#0072B2", "free": "#D55E00"}
    fig, axs = new_figure(ncols=3, figsize=(12.0, 4.2))
    for v in variants:
        e = ev[ev["variant"] == v]
        n_runs = e["id"].nunique()
        axs[0].hist(
            e["dh_km"].astype(float),
            bins=40,
            color=colors[v],
            alpha=0.5,
            density=True,
            label=f"{v} model ({n_runs} runs)",
        )
        axs[0].axvline(float(e["dh_km"].median()), color=colors[v], lw=1.6)
        dz = e["dz_km"].astype(float)
        axs[1].hist(dz, bins=40, color=colors[v], alpha=0.5, density=True, label=v)
        axs[1].axvline(float(dz.median()), color=colors[v], lw=1.6)
        axs[2].scatter(
            e["pert_dz_km"].astype(float), dz, s=3, color=colors[v], alpha=0.4, lw=0, label=v
        )
    axs[0].set_xlabel("horizontal mislocation (km)")
    axs[0].set_ylabel("density")
    axs[0].legend(fontsize=7)
    axs[0].set_title("horizontal")
    axs[1].set_xlabel("vertical error: relocated - reference depth (km)")
    axs[1].axvline(0, color="k", lw=0.6)
    axs[1].set_title("vertical")
    axs[2].set_xlabel("applied depth perturbation (km)")
    axs[2].set_ylabel("residual vertical error (km)")
    axs[2].axhline(0, color="k", lw=0.6)
    axs[2].set_title("depth: recovered vs applied")
    axs[2].legend(fontsize=7, markerscale=4)
    return fig


def plot_stability_velocity_spread(stab: Any) -> Figure | None:
    """Velocity spread per layer for the hypocentre-perturbation runs with a free model.

    Left: staircase overlay of all free-model final Vp models and the baseline.  Right: the
    per-layer standard deviation and (max - min) range of Vp.
    """
    if not _need(stab, "hypocentre_perturbation") or len(stab.models) == 0:
        return None
    m = stab.models[
        (stab.models["test"] == "hypocentre_perturbation")
        & stab.models["variant"].isin(["free", "baseline_free"])
    ]
    free = m[m["variant"] == "free"]
    if free.empty:
        return None
    lay = stab.layers[
        (stab.layers["test"] == "hypocentre_perturbation") & (stab.layers["variant"] == "free")
    ].sort_values("layer")
    depths = (free.drop_duplicates("layer").sort_values("layer")["depth_top"]).to_numpy(float)
    zbot = _zbot(depths)
    fig = _blank((9.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
    a1, a2 = _axes(fig, gs[0, 0]), _axes(fig, gs[0, 1], sharey=None)
    for _, g in free.groupby("id"):
        g = g.sort_values("layer")
        a1.plot(*_steps(depths, g["vp"].to_numpy(float), zbot), color="#D55E00", lw=0.8, alpha=0.5)
    base = m[m["variant"] == "baseline_free"].sort_values("layer")
    if len(base):
        a1.plot(
            *_steps(depths, base["vp"].to_numpy(float), zbot),
            color="k",
            lw=2.2,
            label="baseline (unperturbed, free)",
        )
    if len(lay) and lay["vp_start"].notna().any():
        a1.plot(
            *_steps(depths, lay["vp_start"].to_numpy(float), zbot),
            color="#0072B2",
            lw=1.6,
            ls="--",
            label="start (final model)",
        )
    a1.plot([], [], color="#D55E00", label=f"perturbed runs ({free['id'].nunique()})")
    a1.invert_yaxis()
    a1.set_xlabel("Vp (km/s)")
    a1.set_ylabel("Depth below sea level (km)")
    a1.legend(loc="lower left", fontsize=7)
    a1.set_title("final models of perturbed runs")
    if len(lay):
        y = np.arange(len(lay))
        a2.barh(y - 0.2, lay["vp_std"].astype(float), height=0.4, color="#D55E00", label="sd")
        a2.barh(
            y + 0.2,
            (lay["vp_max"] - lay["vp_min"]).astype(float),
            height=0.4,
            color="0.6",
            label="max - min",
        )
        a2.set_yticks(
            y,
            [f"{k + 1}: {d:g} km" for k, d in zip(lay["layer"], lay["depth_top"], strict=True)],
            fontsize=7,
        )
        a2.invert_yaxis()
        a2.set_xlabel("Vp spread (km/s)")
        a2.legend(fontsize=7)
        a2.set_title("spread per layer")
    return fig


def plot_reference_swap(stab: Any) -> Figure | None:
    """Reference-station swap: final models and station delays with other reference stations.

    Left: Vp staircases of the baseline and each swap run.  Middle: each swap's delays against
    the baseline delays (a constant offset moves the cloud off the 1:1 line; a changed pattern
    broadens it).  Right: delay residual after removing the constant offset, per station.
    """
    if not _need(stab, "reference_station") or len(stab.delays) == 0:
        return None
    mod = stab.models[stab.models["test"] == "reference_station"]
    dl = stab.delays[stab.delays["test"] == "reference_station"].copy()
    if mod.empty or dl.empty:
        return None
    ids = list(dl["id"].unique())
    cols = plt.get_cmap("tab10")
    fig = _blank((13.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.3])
    a1, a2, a3 = (_axes(fig, gs[0, k]) for k in range(3))
    depths = mod.drop_duplicates("layer").sort_values("layer")["depth_top"].to_numpy(float)
    zbot = _zbot(depths)
    base = mod[mod["variant"] == "baseline"]
    if len(base):
        g = base.sort_values("layer")
        a1.plot(*_steps(depths, g["vp"].to_numpy(float), zbot), color="k", lw=2.6, label="baseline")
    rms = stab.runs.set_index("id")["rms"] if len(stab.runs) else pd.Series(dtype=float)
    for k, rid in enumerate(ids):
        g = mod[mod["id"] == rid].sort_values("layer")
        if g.empty:
            continue
        ref = str(dl.loc[dl["id"] == rid, "reference_station"].iloc[0])
        a1.plot(
            *_steps(depths, g["vp"].to_numpy(float), zbot),
            color=cols(k),
            lw=1.2,
            label=f"ref {ref} (RMS {rms.get(rid, np.nan):.4f})",
        )
    a1.invert_yaxis()
    a1.set_xlabel("Vp (km/s)")
    a1.set_ylabel("Depth below sea level (km)")
    a1.legend(fontsize=7, loc="lower left")
    a1.set_title("final model")
    dl["base"] = dl["ptcor"] - dl["ptcor_diff"]
    lim = np.array([dl["base"].min(), dl["base"].max()])
    a2.plot(lim, lim, color="k", lw=0.8, label="1:1")
    for k, rid in enumerate(ids):
        g = dl[dl["id"] == rid]
        off = float(g["ptcor_diff"].median())
        a2.scatter(
            g["base"],
            g["ptcor"],
            s=14,
            color=cols(k),
            alpha=0.8,
            lw=0,
            label=f"ref {g['reference_station'].iloc[0]}: offset {off:+.3f} s",
        )
    a2.set_xlabel("P delay, baseline reference (s)")
    a2.set_ylabel("P delay, swapped reference (s)")
    a2.legend(fontsize=7, loc="lower right")
    a2.set_title("delays: swapped vs baseline")
    sta = sorted(dl["sta"].unique())
    pos = {s: i for i, s in enumerate(sta)}
    for k, rid in enumerate(ids):
        g = dl[dl["id"] == rid]
        a3.scatter(
            [pos[s] for s in g["sta"]], g["ptcor_resid"], s=14, color=cols(k), alpha=0.8, lw=0
        )
    a3.axhline(0, color="k", lw=0.6)
    a3.set_xticks(range(len(sta)), sta, rotation=90, fontsize=5)
    a3.set_ylabel("delay change after constant offset removed (s)")
    a3.set_title("delay pattern change")
    return fig


def plot_lvz_comparison(stab: Any) -> Figure | None:
    """Low-velocity-zone test: final models with ``lowveloclay`` on and off (RMS in legend).

    One panel per tested starting model (the final model and the ensemble's closest/best runs);
    solid = LVZ allowed, dashed = LVZ not allowed.
    """
    if not _need(stab, "lvz") or len(stab.models) == 0:
        return None
    mod = stab.models[stab.models["test"] == "lvz"]
    runs = stab.runs[stab.runs["test"] == "lvz"].set_index("id")
    keys = list(dict.fromkeys(mod["model_key"]))
    if not keys:
        return None
    fig, axs = new_figure(
        ncols=len(keys), figsize=(4.6 * len(keys), 5.4), squeeze=False, sharey=True
    )
    depths = mod.drop_duplicates("layer").sort_values("layer")["depth_top"].to_numpy(float)
    zbot = _zbot(depths)
    for ax, key in zip(axs[0], keys, strict=True):
        for variant, (c, ls, lw) in {
            "lvz0": ("#0072B2", "--", 2.2),
            "lvz1": ("#D55E00", "-", 1.3),
        }.items():
            sel = mod[(mod["model_key"] == key) & mod["variant"].str.endswith(variant)]
            for rid, g in sel.groupby("id"):
                g = g.sort_values("layer")
                r = float(runs["rms"].get(rid, np.nan))
                lab = "LVZ off" if variant == "lvz0" else "LVZ on"
                ax.plot(
                    *_steps(depths, g["vp"].to_numpy(float), zbot),
                    color=c,
                    ls=ls,
                    lw=lw,
                    label=f"{lab}: RMS {r:.4f} s",
                )
        ax.invert_yaxis()
        ax.set_xlabel("Vp (km/s)")
        ax.set_title(key.replace("_", " "), fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    axs[0][0].set_ylabel("Depth below sea level (km)")
    return fig


# ------------------------------------------------------------------- residuals and Vp/Vs


def plot_residuals(
    station_stats: pd.DataFrame | None, hypocentres: pd.DataFrame | None = None
) -> Figure | None:
    """Residuals per station and per event of one (fixed-model) run.

    Parameters
    ----------
    station_stats
        ``VelestRunResult.station_stats`` (``sta, phase, nobs, avres, avwres, std``): mean
        residual per station and phase with its standard deviation as error bar.
    hypocentres
        Optional per-event table with ``rms`` (and ``depth``/``depth_km``): histogram of the
        event RMS and RMS against depth.
    """
    ss = station_stats
    has_ss = ss is not None and len(ss) > 0
    hy = (
        hypocentres
        if hypocentres is not None and len(hypocentres) and "rms" in hypocentres
        else None
    )
    if not has_ss and hy is None:
        return None
    fig = _blank((13.5, 4.8))
    gs = fig.add_gridspec(
        1, 3 if hy is not None else 1, width_ratios=[2.6, 1, 1] if hy is not None else [1]
    )
    ax = _axes(fig, gs[0, 0])
    if has_ss:
        stas = sorted(ss["sta"].unique())
        x = {s: i for i, s in enumerate(stas)}
        for ph, off in (("P", -0.15), ("S", 0.15)):
            d = ss[ss["phase"] == ph]
            if d.empty:
                continue
            ax.errorbar(
                [x[s] + off for s in d["sta"]],
                d["avres"],
                yerr=d["std"],
                fmt="o",
                ms=4,
                color=PHASE_COLORS[ph],
                elinewidth=0.8,
                capsize=1.5,
                label=f"{ph} (n = {int(d['nobs'].sum())})",
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(stas)), stas, rotation=90, fontsize=6)
        ax.set_ylabel("mean residual +/- 1 sd (s)")
        ax.set_title("residuals per station")
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()
    if hy is not None:
        dcol = "depth" if "depth" in hy.columns else "depth_km"
        a2, a3 = _axes(fig, gs[0, 1]), _axes(fig, gs[0, 2])
        r = hy["rms"].astype(float)
        a2.hist(r.dropna(), bins=30, color="#7f7f7f")
        a2.axvline(float(r.median()), color="k", lw=1.2)
        a2.set_xlabel("event RMS (s)")
        a2.set_ylabel("events")
        a2.set_title(f"per event (median {r.median():.3f} s)")
        if dcol in hy.columns:
            a3.scatter(r, hy[dcol], s=8, color=PHASE_COLORS["P"], alpha=0.7, lw=0)
            a3.invert_yaxis()
            a3.set_xlabel("event RMS (s)")
            a3.set_ylabel("depth (km)")
            a3.set_title("RMS vs depth")
        else:
            a3.set_axis_off()
    return fig


def plot_vpvs(grid: pd.DataFrame | None, wadati: Mapping[str, Any] | None = None) -> Figure | None:
    """Vp/Vs grid search: final RMS versus fixed ratio, with the Wadati estimate and CI.

    Parameters
    ----------
    grid
        Output of :func:`osv.velest.vpvs.vpvs_grid_search` (``vpvs, final_rms, converged``;
        ``attrs`` ``best_vpvs`` / ``best_vpvs_parabola``).  ``None`` returns ``None``.
    wadati
        Stage 1 ``wadati.json`` (``vpvs``, ``vpvs_ci``) for the reference band.
    """
    if grid is None or len(grid) == 0 or not np.isfinite(grid["final_rms"]).any():
        return None
    g = grid.sort_values("vpvs")
    fig, ax = new_figure(figsize=(6.2, 4.4))
    ax.plot(g["vpvs"], g["final_rms"], "-", color="0.5", lw=1.0, zorder=2)
    conv = g["converged"].astype(bool)
    if conv.any():
        ax.scatter(
            g["vpvs"][conv],
            g["final_rms"][conv],
            s=40,
            color=PHASE_COLORS["P"],
            zorder=3,
            label="converged",
        )
    if (~conv).any():
        ax.scatter(
            g["vpvs"][~conv],
            g["final_rms"][~conv],
            s=40,
            facecolors="white",
            edgecolors=PHASE_COLORS["P"],
            zorder=3,
            label="stopped at iteration limit",
        )
    best = grid.attrs.get("best_vpvs", float(g.loc[g["final_rms"].idxmin(), "vpvs"]))
    ax.axvline(best, color="k", ls=":", lw=1.2, label=f"RMS minimum {best:.3f}")
    vert = grid.attrs.get("best_vpvs_parabola", math.nan)
    if _isnum(vert):
        ax.axvline(float(vert), color="k", ls="--", lw=1.0, label=f"parabola vertex {vert:.3f}")
    if wadati and _isnum(wadati.get("vpvs")):
        ci = wadati.get("vpvs_ci") or []
        if len(ci) == 2 and all(_isnum(c) for c in ci):
            ax.axvspan(
                ci[0],
                ci[1],
                color=PHASE_COLORS["S"],
                alpha=0.2,
                label=f"Wadati {wadati['vpvs']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})",
            )
        ax.axvline(
            float(wadati["vpvs"]),
            color=PHASE_COLORS["S"],
            lw=1.8,
            label=None if len(ci) == 2 else f"Wadati {wadati['vpvs']:.3f}",
        )
    ax.set_xlabel("fixed Vp/Vs")
    ax.set_ylabel("final RMS residual (s)")
    ax.set_title("Vp/Vs: VELEST grid search")
    ax.legend(fontsize=7, loc="best")
    return fig


# -------------------------------------------------------------------------------- driver

Saver = Callable[..., Any]


def to_qc_findings(*finding_lists: Sequence[Any]) -> list[QCFinding]:
    """Convert Stage 2 :class:`~osv.velest.ensemble.Finding` objects to ``QCFinding``.

    Severity ``error`` becomes ``fail``; the finding's ``data`` and ``layer`` go to ``details``.
    """
    out = []
    for lst in finding_lists:
        for f in lst or []:
            sev = {"error": "fail"}.get(f.severity, f.severity)
            det = dict(getattr(f, "data", {}) or {})
            if getattr(f, "layer", None) is not None:
                det["layer"] = f.layer
            out.append(QCFinding(f.code, sev, f.message, details=det))  # type: ignore[arg-type]
    return out


def make_all_figures(
    save: Saver,
    *,
    ensemble: Any,
    workflow: Any | None = None,
    stability: Any | None = None,
    vpvs_grid: pd.DataFrame | None = None,
    wadati: Mapping[str, Any] | None = None,
    stations: pd.DataFrame | None = None,
    input_hypocentres: pd.DataFrame | None = None,
    vp_bounds: Sequence[Sequence[float]] | None = None,
    best_percent: float = 10.0,
    reference_station: str | None = None,
    section: str = "Monte Carlo VELEST",
) -> list[str]:
    """Create every Stage 2 figure that its inputs allow and hand each one to ``save``.

    ``save(fig, name, caption, section)`` is called per figure
    (``FigureRegistry.save`` has this signature and closes the figure).  Figures whose input
    is missing are skipped, a figure that raises is logged and skipped.

    Parameters
    ----------
    ensemble
        :class:`~osv.velest.ensemble.EnsembleResult` (required).
    workflow
        :class:`~osv.velest.workflow.WorkflowResult` (iteration curves and stage RMS).  When
        absent the ensemble's own parsed runs are not available, so those figures are skipped.
    stability
        :class:`~osv.velest.stability.StabilityResult`.
    vpvs_grid, wadati
        Output of ``vpvs_grid_search`` and Stage 1 ``wadati.json``.
    stations
        Stage 1 station table or the written VELEST station table (see module docstring).
    input_hypocentres
        Starting hypocentres of the VELEST event file (``VelestInputs.events``).

    Returns
    -------
    list of str
        Names of the figures saved.
    """
    ens = ensemble
    st = _stations(stations)
    elev = st["elev_m"].dropna().to_numpy() if st is not None else None
    hyp = ens.hypocentres
    closest_h = hyp[hyp["id"] == ens.closest_run_id] if hyp is not None and len(hyp) else None
    fixed = getattr(stability, "fixed", None) if stability is not None else None
    final_h = fixed.hypocentres if fixed is not None and len(fixed.hypocentres) else closest_h
    fixed_stats = getattr(getattr(fixed, "parsed", None), "station_stats", None)
    if fixed_stats is None or len(fixed_stats) == 0:
        fixed_stats = None
        if workflow is not None:
            r = workflow.runs[workflow.runs["id"] == ens.closest_run_id]
            fixed_stats = getattr(r["parsed"].iloc[0], "station_stats", None) if len(r) else None
    ref = reference_station or ens.reference_station
    hd = (
        final_h["depth_km" if final_h is not None and "depth_km" in final_h else "depth"].to_numpy()
        if final_h is not None and len(final_h)
        else None
    )
    sel_ids = ens.selected_ids

    specs: list[tuple[str, str, Callable[[], Figure | None]]] = [
        (
            "velocity_spaghetti",
            "All initial (grey) and final (coloured by final RMS) 1-D models; selected runs "
            "outlined, "
            "final model in black with its p05-p95 band; hypocentre depth histogram on the right.",
            lambda: plot_spaghetti(ens, hypocentre_depths=hd, station_elevations_m=elev),
        ),
        (
            "velocity_density",
            f"Density of the best {best_percent:g}% of the final Vp models.",
            lambda: plot_velocity_density(ens, best_percent, station_elevations_m=elev),
        ),
        (
            "layer_histograms",
            "Final Vp per layer: all eligible runs (grey), selected runs (orange), final model "
            "(black) and sampled bounds (red dashed).",
            lambda: plot_layer_histograms(ens, vp_bounds),
        ),
        (
            "rms_vs_layer_velocity",
            "Final RMS against each layer's final Vp: trade-offs and poorly constrained layers.",
            lambda: plot_rms_vs_layer_velocity(ens, vp_bounds),
        ),
        (
            "rms_vs_iteration",
            "RMS residual versus iteration of the last damping stage of every run; backup steps "
            "marked.",
            lambda: plot_rms_vs_iteration(workflow.runs, sel_ids) if workflow is not None else None,
        ),
        (
            "stage_rms",
            "Final RMS and iteration count per damping stage.",
            lambda: plot_stage_rms(workflow.history) if workflow is not None else None,
        ),
        (
            "convergence_summary",
            "Run outcomes (ok, converged, four_backups, diverged, failed), stop reasons and the "
            "initial models by outcome.",
            lambda: plot_convergence(ens),
        ),
        (
            "station_delay_map",
            "Station delays on a map (red = late/slow); size ~ observations, ring = reference "
            "station, down-triangle = borehole.",
            lambda: plot_station_delay_map(ens, stations, ref),
        ),
        (
            "delay_vs_elevation",
            "Station delay against sensor elevation with the spread over the selected runs.",
            lambda: plot_delay_vs_elevation(ens, stations, ref),
        ),
        (
            "hypocentre_shifts",
            "Hypocentre shifts between the starting and relocated solution (mean over selected "
            "runs).",
            lambda: plot_hypocentre_shifts(ens, input_hypocentres, stations),
        ),
        (
            "final_hypocentres",
            "Final hypocentres: map and E-W / N-S depth sections.",
            lambda: plot_final_hypocentres(final_h, stations) if final_h is not None else None,
        ),
        (
            "stability_hypocentre_perturbation",
            "Hypocentre perturbation test: mislocation with the model fixed and free.",
            lambda: plot_hypocentre_perturbation(stability) if stability is not None else None,
        ),
        (
            "stability_velocity_spread",
            "Velocity spread per layer for the hypocentre-perturbation runs.",
            lambda: plot_stability_velocity_spread(stability) if stability is not None else None,
        ),
        (
            "stability_reference_station",
            "Reference-station swap: final models and delay changes.",
            lambda: plot_reference_swap(stability) if stability is not None else None,
        ),
        (
            "stability_lvz",
            "Low-velocity-zone test: models with the LVZ option on and off.",
            lambda: plot_lvz_comparison(stability) if stability is not None else None,
        ),
        (
            "residuals",
            "Residuals per station and per event of the final-model (fixed) run.",
            lambda: plot_residuals(fixed_stats, final_h),
        ),
        (
            "vpvs",
            "Vp/Vs grid search RMS curve with the Wadati estimate and its confidence interval.",
            lambda: plot_vpvs(vpvs_grid, wadati),
        ),
    ]
    saved: list[str] = []
    for name, caption, build in specs:
        try:
            fig = build()
        except Exception:
            logger.exception("figure %s failed", name)
            plt.close("all")
            continue
        if fig is None:
            logger.info("figure %s skipped (no data)", name)
            continue
        save(fig, name, caption, section)
        saved.append(name)
    return saved
