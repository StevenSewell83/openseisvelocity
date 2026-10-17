"""Stage 4 diagnostic figures and QC checks (trade-off curves and inversion diagnostics).

All plotting functions take plain tables (the ``tradeoff.json`` picks, the per-iteration table of
:func:`osv.fmtomo.invert.run_inversion` with the :mod:`osv.fmtomo.metrics` columns, residual
tables of :mod:`osv.analysis.residuals`) and return :class:`matplotlib.figure.Figure` objects.
:func:`make_tradeoff_figures` and :func:`make_inversion_figures` build the whole set and hand every
figure to a ``save(fig, name, caption, section)`` callable (``FigureRegistry.save``).

Trade-off figures (PLAN.md Stage 4)
    * ``lcurve_damping`` / ``lcurve_smoothing``: data variance against model variance (damping) or
      roughness of the perturbation (smoothing), every point labelled with its parameter value,
      the chosen point highlighted (and the manual one if different), curvature in a second panel.
    * ``tradeoff_heatmap``: misfit, model variance and roughness over the (damping, smoothing)
      grid (``mode: full_grid``).
    * ``misfit_vs_iteration_sweep``: RMS misfit against iteration of every sweep run.
    * ``neighbour_slices``: depth slice of the model change of the runs around the chosen point.

Inversion figures (final run)
    ``misfit_vs_iteration``, ``residual_histograms_iterations``, ``station_residual_map``
    (initial against final), ``hypocentre_shifts_iterations``, ``rays_lost_iterations`` and
    ``velocity_range_iterations`` (minimum / maximum velocity, fraction at the minimum velocity).

QC checks (:func:`tradeoff_findings`, :func:`inversion_findings`) return
:class:`osv.report.checks.QCFinding` lists; the codes are ``misfit_not_decreasing``,
``misfit_increasing``, ``lcurve_corner``, ``lcurve_at_edge``, ``min_velocity_clipped``,
``rays_lost``, ``chi2_far_from_one``, ``sweep_failed`` and ``no_convergence``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from osv.plotting.slices import colour_limits, crop_interior
from osv.plotting.style import PERTURBATION_CMAP, new_figure
from osv.report.checks import QCFinding

log = logging.getLogger(__name__)

__all__ = [
    "chi2_far_from_one",
    "inversion_findings",
    "make_inversion_figures",
    "make_tradeoff_figures",
    "plot_hypocentre_shifts_iterations",
    "plot_lcurve",
    "plot_misfit_vs_iteration",
    "plot_neighbour_slices",
    "plot_rays_lost",
    "plot_residual_histograms_iterations",
    "plot_tradeoff_heatmap",
    "plot_velocity_range",
    "read_run_iterations",
    "tradeoff_findings",
]

Saver = Callable[..., Any]

C_CHOSEN = "#d95f02"
C_MANUAL = "#1b9e77"
C_LINE = "#4d4d4d"
#: ``chi2_reduced`` outside ``1 / CHI2_TOL .. CHI2_TOL`` is commented on
CHI2_TOL = 2.0
#: a final misfit reduction below this fraction counts as "not decreasing"
MIN_REDUCTION = 0.02
#: more than this fraction of the picks lost in an iteration is flagged
LOST_FRACTION = 0.05
#: fraction of the nodes at the minimum velocity above which the clip is a warning
CLIP_WARN_FRACTION = 1e-3
#: a RMS change below this fraction between the last two iterations counts as converged
CONVERGED_TOL = 0.02


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{float(v):g}"


def _num(x: Any) -> float:
    try:
        return math.nan if x is None else float(x)
    except (TypeError, ValueError):
        return math.nan


def read_run_iterations(run_dir: str | Path) -> pd.DataFrame:
    """Per-iteration table of a run from ``iterations/NN/summary.json`` (columns of ``summary``).

    Works on heavily cleaned run directories: only the small ``summary.json`` files are read.
    Returns an empty frame if there are none.
    """
    rows = []
    idir = Path(run_dir) / "iterations"
    if not idir.is_dir():
        return pd.DataFrame()
    for d in sorted(p for p in idir.iterdir() if p.is_dir() and p.name.isdigit()):
        try:
            s = json.loads((d / "summary.json").read_text())
        except (OSError, ValueError):
            continue
        s.pop("invert3d", None)
        rows.append(s)
    return pd.DataFrame(rows)


def _finite(a: Any) -> np.ndarray:
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


# ---------------------------------------------------------------------------------------
# trade-off figures
# ---------------------------------------------------------------------------------------
def plot_lcurve(pick: Mapping[str, Any], kind: str = "damping") -> Figure:
    """L-curve of one sweep with the chosen point highlighted.

    Parameters
    ----------
    pick
        ``picks["damping"]`` or ``picks["smoothing"]`` of ``tradeoff.json``: ``curve`` (rows with
        the parameter, the x column and ``variance_s2``), ``chosen``, ``manual``, ``source``,
        ``max_curvature``, ``curvature``, ``x_axis``.
    kind
        ``"damping"`` or ``"smoothing"``.

    Returns
    -------
    matplotlib.figure.Figure
        Left: the L-curve (log-log, points labelled by parameter value; the corner in orange, the
        manual value in green if it differs).  Right: the curvature against the parameter.
    """
    cur = pd.DataFrame(pick.get("curve", []))
    xcol = str(pick.get("x_axis", "model_variance"))
    xlab = {
        "model_variance": "model variance  mean((v - v_ref)$^2$)  (km/s)$^2$",
        "pert_roughness_phys": "roughness of the perturbation  mean($\\nabla^2 \\delta v$)$^2$",
    }.get(xcol, xcol)
    fig, (ax, axc) = new_figure(
        ncols=2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    if cur.empty or kind not in cur:
        ax.text(0.5, 0.5, "no successful runs", ha="center", va="center", transform=ax.transAxes)
        return fig
    cur = cur.sort_values(kind).reset_index(drop=True)
    x, y, p = cur[xcol].to_numpy(float), cur["variance_s2"].to_numpy(float), cur[kind].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if not ok.any():
        ax.text(0.5, 0.5, "no positive finite points", ha="center", va="center",
                transform=ax.transAxes)  # fmt: skip
        return fig
    ax.plot(x[ok], y[ok], "-o", color=C_LINE, ms=5, lw=1, zorder=2)
    for xi, yi, pi in zip(x[ok], y[ok], p[ok], strict=True):
        ax.annotate(
            _fmt(pi), (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=7.5, zorder=4
        )
    chosen, manual = _num(pick.get("chosen")), _num(pick.get("manual"))

    def mark(v: float, colour: str, label: str, marker: str, size: int) -> None:
        sel = ok & np.isclose(p.astype(float), v)
        if sel.any():
            ax.scatter(x[sel], y[sel], s=size, marker=marker, color=colour, zorder=5, label=label)

    src = str(pick.get("source", ""))
    mark(chosen, C_CHOSEN, f"chosen {kind} = {_fmt(chosen)} ({src})", "o", 110)
    if math.isfinite(manual) and not math.isclose(manual, chosen):
        mark(manual, C_MANUAL, f"manual {kind} = {_fmt(manual)}", "D", 70)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlab)
    ax.set_ylabel("data variance (s$^2$)")
    ax.set_title(f"{kind.capitalize()} L-curve")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")
    # curvature (aligned with the successful runs sorted by parameter)
    kc = np.array([_num(c) for c in pick.get("curvature", [])])
    if len(kc) == len(p) and np.isfinite(kc).any():
        axc.semilogx(p.astype(float), kc, "-o", color=C_LINE, ms=4)
        mc = _num(pick.get("max_curvature"))
        if math.isfinite(mc):
            axc.axvline(mc, color=C_CHOSEN, ls="--", label=f"maximum at {_fmt(mc)}")
            axc.legend()
        axc.axhline(0, color="0.6", lw=0.6)
    else:
        axc.text(
            0.5,
            0.5,
            "curvature undefined\n" + str(pick.get("reason", "")),
            ha="center",
            va="center",
            transform=axc.transAxes,
            fontsize=8,
        )
    axc.set_xlabel(kind)
    axc.set_ylabel("Menger curvature (normalised axes)")
    axc.set_title("Curvature")
    return fig


def plot_tradeoff_heatmap(table: pd.DataFrame, chosen: tuple[float, float] | None = None) -> Figure:
    """Heatmaps of the data variance, model variance and perturbation roughness over the grid.

    Rows of ``table`` are runs (``damping``, ``smoothing``, ``variance_s2``, ``model_variance``,
    ``pert_roughness_phys``); the chosen ``(damping, smoothing)`` is outlined.
    """
    t = table[table["success"].astype(bool)] if "success" in table else table
    panels = [
        ("variance_s2", "data variance (s$^2$)"),
        ("model_variance", "model variance (km/s)$^2$"),
        ("pert_roughness_phys", "perturbation roughness"),
    ]
    fig, axes = new_figure(ncols=3, figsize=(13, 4.2))
    ds = sorted(t["damping"].unique()) if len(t) else []
    ss = sorted(t["smoothing"].unique()) if len(t) else []
    for ax, (col, lab) in zip(np.atleast_1d(axes), panels, strict=True):
        if not len(t):
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        piv = t.pivot_table(index="smoothing", columns="damping", values=col).reindex(
            index=ss, columns=ds
        )
        z = np.log10(np.where(piv.to_numpy() > 0, piv.to_numpy(), np.nan))
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ds)), [_fmt(v) for v in ds], rotation=45)
        ax.set_yticks(range(len(ss)), [_fmt(v) for v in ss])
        ax.set_xlabel("damping")
        ax.set_ylabel("smoothing")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label=f"log10 {lab}")
        for i in range(len(ss)):
            for j in range(len(ds)):
                v = piv.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=6, color="w")
        if chosen is not None and chosen[0] in ds and chosen[1] in ss:
            ax.add_patch(
                plt.Rectangle(
                    (ds.index(chosen[0]) - 0.5, ss.index(chosen[1]) - 0.5),
                    1,
                    1,
                    fill=False,
                    ec=C_CHOSEN,
                    lw=2.5,
                )
            )
        ax.set_title(lab)
    fig.suptitle("Trade-off grid (chosen cell outlined)")
    return fig


def plot_misfit_vs_iteration(
    runs: Mapping[str, pd.DataFrame],
    highlight: str | None = None,
    column: str = "rms_ms",
) -> Figure:
    """RMS misfit (ms) against iteration for several runs (one line per run)."""
    fig, ax = new_figure(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("viridis")
    names = list(runs)
    for i, name in enumerate(names):
        df = runs[name]
        if df.empty or column not in df:
            continue
        hl = name == highlight
        ax.plot(
            df["iter"],
            df[column],
            "-o",
            ms=4 if not hl else 6,
            lw=1 if not hl else 2.5,
            color=C_CHOSEN if hl else cmap(i / max(len(names) - 1, 1)),
            alpha=1 if hl else 0.8,
            label=name,
            zorder=5 if hl else 2,
        )
    ax.set_xlabel("iteration (0 = starting model)")
    ax.set_ylabel("RMS misfit (ms)" if column == "rms_ms" else column)
    ax.set_title("Misfit against iteration")
    if names:
        ax.legend(ncol=2 if len(names) > 8 else 1, fontsize=7, title="run")
    return fig


def _last_model(run: Path) -> tuple[xr.Dataset, xr.Dataset] | None:
    from osv.io.fmtomo import read_vgrids

    its = sorted(p for p in (run / "iterations").glob("*") if p.name.isdigit())
    if not its or not (its[-1] / "vgrids.in").exists() or not (run / "vgridsref.in").exists():
        return None
    return (
        read_vgrids(its[-1] / "vgrids.in").get(1, 1),
        read_vgrids(run / "vgridsref.in").get(1, 1),
    )


def plot_neighbour_slices(
    runs: list[tuple[str, Path]],
    depth_km: float,
    ncols: int = 3,
    title: str = "",
) -> Figure:
    """Depth slice of the velocity change (% against the starting model) of several runs.

    ``runs`` is a list of ``(label, run_dir)``; each panel shows the last iteration's
    ``iterations/NN/vgrids.in`` against ``vgridsref.in`` at the node nearest ``depth_km`` with a
    colour scale shared by all panels (symmetric about zero).
    """
    n = max(len(runs), 1)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = new_figure(
        nrows=nrows, ncols=ncols, figsize=(3.6 * ncols + 0.8, 3.3 * nrows), squeeze=False
    )
    fields: list[tuple[str, np.ndarray | None, Any]] = []
    for label, run in runs:
        pair = _last_model(Path(run))
        if pair is None:
            fields.append((label, None, None))
            continue
        cur, ref = (crop_interior(d) for d in pair)
        k = int(np.argmin(np.abs(cur["depth"].values - depth_km)))
        v1 = cur["velocity"].transpose("depth", "lat", "lon").values[k]
        v0 = ref["velocity"].transpose("depth", "lat", "lon").values[k]
        fields.append(
            (label, 100.0 * (v1 - v0) / v0, (cur["lon"].values, cur["lat"].values, k, cur))
        )
    vals = [f[1] for f in fields if f[1] is not None]
    lim = colour_limits(vals, "pct") if vals else (-1.0, 1.0)
    im = None
    for ax, (label, z, geo) in zip(axes.ravel(), fields, strict=False):
        if z is None:
            ax.text(
                0.5, 0.5, f"{label}\nno model", ha="center", va="center", transform=ax.transAxes
            )
            continue
        lon, lat, k, cur = geo
        im = ax.pcolormesh(
            lon, lat, z, cmap=PERTURBATION_CMAP, vmin=lim[0], vmax=lim[1], shading="nearest"
        )
        ax.set_title(f"{label}\n(z = {float(cur['depth'].values[k]):.2f} km)", fontsize=8)
        ax.set_aspect(1.0 / max(math.cos(math.radians(float(np.mean(lat)))), 1e-3))
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(False)
    for ax in axes.ravel()[len(fields) :]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="dV vs starting (%)", shrink=0.8)
    fig.suptitle(title or "Model change of neighbouring runs")
    return fig


# ---------------------------------------------------------------------------------------
# inversion figures
# ---------------------------------------------------------------------------------------
def plot_residual_histograms_iterations(
    tables: Mapping[int, pd.DataFrame], bins: int = 60, ncols: int = 4
) -> Figure:
    """Histogram of the residuals (observed - calculated) of every iteration."""
    its = sorted(tables)
    n = max(len(its), 1)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = new_figure(
        nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True, squeeze=False
    )
    allr = _finite(np.concatenate([tables[i]["residual"].to_numpy(float) for i in its] or [[0.0]]))
    lim = float(np.percentile(np.abs(allr), 99)) if allr.size else 1.0
    lim = lim if lim > 0 else 1.0
    edges = np.linspace(-lim, lim, bins + 1)
    for ax, it in zip(axes.ravel(), its, strict=False):
        r = _finite(tables[it]["residual"].to_numpy(float))
        ax.hist(r, bins=edges, color=C_LINE, alpha=0.85)
        ax.axvline(0, color="k", lw=0.6)
        rms = float(np.sqrt(np.mean(r**2))) if r.size else math.nan
        ax.set_title(
            f"iteration {it}: rms {1000 * rms:.0f} ms, mean {1000 * np.mean(r):+.0f} ms"
            if r.size
            else f"iteration {it}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(its) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("residual (s)")
    fig.suptitle("Residuals per iteration")
    return fig


def _hypo_shifts(hyp: pd.DataFrame) -> pd.DataFrame:
    """3-D shift (km) of every source at every iteration against iteration 0."""
    km = 6371.0 * math.pi / 180.0
    h = hyp.sort_values(["source", "iter"])
    first = h.groupby("source")[["lat", "lon", "depth_km"]].transform("first")
    out = h[["iter", "source"]].copy()
    out["dh_km"] = np.hypot(
        (h["lat"] - first["lat"]) * km,
        (h["lon"] - first["lon"]) * km * np.cos(np.deg2rad(first["lat"])),
    )
    out["dz_km"] = h["depth_km"] - first["depth_km"]
    return out


def plot_hypocentre_shifts_iterations(hyp: pd.DataFrame) -> Figure:
    """Hypocentre shift against the starting position per iteration (median and 90th percentile).

    ``hyp`` is the long table of :class:`~osv.fmtomo.invert.InversionResult` (``iter``,
    ``source``, ``lat``, ``lon``, ``depth_km``).
    """
    fig, (a1, a2) = new_figure(ncols=2, figsize=(9.5, 4.2))
    if hyp is None or hyp.empty:
        for a in (a1, a2):
            a.text(0.5, 0.5, "no hypocentres", ha="center", va="center", transform=a.transAxes)
        return fig
    sh = _hypo_shifts(hyp)
    g = sh.groupby("iter")
    its = np.array(sorted(sh["iter"].unique()))
    for ax, col, lab in ((a1, "dh_km", "horizontal"), (a2, "dz_km", "depth (|dz|)")):
        v = sh.assign(v=sh[col].abs()).groupby("iter")["v"]
        ax.plot(its, v.median().reindex(its), "-o", color=C_CHOSEN, label="median")
        ax.plot(its, v.quantile(0.9).reindex(its), "--s", color=C_LINE, label="90th percentile")
        ax.plot(its, v.max().reindex(its), ":", color="0.6", label="maximum")
        ax.set_xlabel("iteration")
        ax.set_ylabel(f"{lab} shift from start (km)")
        ax.set_title(f"Hypocentre {lab} shift")
        ax.legend()
    del g
    return fig


def plot_rays_lost(it: pd.DataFrame) -> Figure:
    """Valid, lost (no model ray) and invalid observed picks against iteration."""
    fig, ax = new_figure(figsize=(7, 4.2))
    x = it["iter"].to_numpy()
    for col, c, lab in (
        ("n_valid", C_LINE, "valid picks"),
        ("n_lost", C_CHOSEN, "picks whose ray was lost"),
        ("n_obs_invalid", "0.6", "invalid observations"),
    ):
        if col in it:
            ax.plot(x, it[col], "-o", color=c, ms=4, label=lab)
    ax.set_xlabel("iteration")
    ax.set_ylabel("number of picks")
    ax.set_title("Rays (picks) lost per iteration")
    ax.legend()
    return fig


def plot_velocity_range(it: pd.DataFrame, min_velocity: float | None = None) -> Figure:
    """Minimum and maximum velocity and the fraction of nodes at the minimum velocity."""
    fig, (a1, a2) = new_figure(ncols=2, figsize=(9.5, 4.2))
    x = it["iter"].to_numpy()
    if "vel_min" in it:
        a1.plot(x, it["vel_min"], "-o", color=C_CHOSEN, label="minimum")
    if "vel_max" in it:
        a1.plot(x, it["vel_max"], "-o", color=C_LINE, label="maximum")
    if min_velocity is not None:
        a1.axhline(min_velocity, color="r", ls="--", lw=0.8, label=f"min_velocity {min_velocity:g}")
    a1.set_xlabel("iteration")
    a1.set_ylabel("velocity (km/s)")
    a1.set_title("Velocity range")
    a1.legend()
    if "frac_at_min_velocity" in it:
        a2.plot(x, 100 * it["frac_at_min_velocity"], "-o", color=C_CHOSEN)
    a2.set_xlabel("iteration")
    a2.set_ylabel("nodes at minimum velocity (%)")
    a2.set_title("Fraction of nodes clipped at the minimum velocity")
    return fig


def _mean_resid_series(it: pd.DataFrame) -> bool:
    return "mean_resid_s" in it and it["mean_resid_s"].notna().any()


# ---------------------------------------------------------------------------------------
# whole sets
# ---------------------------------------------------------------------------------------
def make_tradeoff_figures(
    save: Saver,
    result: Mapping[str, Any],
    workdir: str | Path,
    *,
    depth_km: float | None = None,
    suffix: str = "",
    label: str = "",
) -> list[str]:
    """Build and save the trade-off figures of one phase.

    Parameters
    ----------
    save
        ``save(fig, name, caption, section)``.
    result
        ``tradeoff.json`` content (``picks``, ``table``, ``mode``, ``chosen_*``).
    workdir
        Directory holding the sweep run directories (``dir`` column of the table is used, with a
        fallback to ``workdir/d<damping>_s<smoothing>``).
    depth_km
        Depth of the neighbour-run slices (default: the middle of the sweep models' depth range).
    suffix, label
        Appended to figure names / captions (phase).
    """
    saved: list[str] = []
    picks = result["picks"]
    table = pd.DataFrame(result["table"])
    wd = Path(workdir)
    cd, cs = float(result["chosen_damping"]), float(result["chosen_smoothing"])

    def emit(fig: Figure, name: str, caption: str) -> None:
        save(fig, f"{name}{suffix}", f"{label}{caption}", "tradeoff")
        saved.append(f"{name}{suffix}")

    emit(
        plot_lcurve(picks["damping"], "damping"),
        "lcurve_damping",
        "Damping L-curve: data variance against model variance for each damping value "
        f"(smoothing {_fmt(picks['damping'].get('at_smoothing', math.nan))}); the L corner "
        "(maximum curvature) is highlighted.",
    )
    emit(
        plot_lcurve(picks["smoothing"], "smoothing"),
        "lcurve_smoothing",
        "Smoothing L-curve: data variance against roughness of the perturbation "
        f"(damping {_fmt(picks['smoothing'].get('at_damping', math.nan))}).",
    )
    if result.get("mode") == "full_grid" and len(table):
        emit(
            plot_tradeoff_heatmap(table, (cd, cs)),
            "tradeoff_heatmap",
            "Misfit, model variance and roughness over the full damping x smoothing grid.",
        )
    runs: dict[str, pd.DataFrame] = {}
    hl = None
    for _, r in table.iterrows():
        d, s = float(r["damping"]), float(r["smoothing"])
        run = Path(str(r.get("dir", ""))) if isinstance(r.get("dir"), str) else None
        if run is None or not run.is_dir():
            run = wd / f"d{_fmt(d)}_s{_fmt(s)}"
        df = read_run_iterations(run)
        name = f"d={_fmt(d)}, s={_fmt(s)}"
        runs[name] = df
        if math.isclose(d, cd) and math.isclose(s, cs):
            hl = name
    emit(
        plot_misfit_vs_iteration(runs, hl),
        "misfit_vs_iteration_sweep",
        "RMS misfit against iteration for every trade-off run (chosen run highlighted).",
    )

    # neighbouring runs of the chosen point
    def sel(kind: str, fixed_col: str, fixed: float) -> list[tuple[str, Path]]:
        t = table[table["success"].astype(bool) & np.isclose(table[fixed_col].astype(float), fixed)]
        vals = sorted(t[kind].astype(float).unique())
        centre = cd if kind == "damping" else cs
        if not vals:
            return []
        i = int(np.argmin([abs(math.log(v) - math.log(centre)) for v in vals]))
        out = []
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(vals):
                row = t[np.isclose(t[kind].astype(float), vals[j])].iloc[0]
                out.append(
                    (f"{kind} {_fmt(vals[j])}" + (" (chosen)" if j == i else ""), Path(row["dir"]))
                )
        return out

    nb = sel("damping", "smoothing", float(picks["damping"].get("at_smoothing", cs)))
    nb += sel("smoothing", "damping", float(picks["smoothing"].get("at_damping", cd)))
    nb = [(lab, p) for lab, p in nb if p.is_dir()]
    if nb:
        if depth_km is None:
            depth_km = 2.0
        emit(
            plot_neighbour_slices(nb, depth_km, ncols=3),
            "neighbour_slices",
            f"Depth slice (about {depth_km:g} km) of the velocity change of the runs next to the "
            "chosen damping (top row) and smoothing (bottom row), same colour scale.",
        )
    return saved


def make_inversion_figures(
    save: Saver,
    iterations: pd.DataFrame,
    *,
    tables: Mapping[int, pd.DataFrame] | None = None,
    hypocentres: pd.DataFrame | None = None,
    min_velocity: float | None = None,
    suffix: str = "",
    label: str = "",
) -> list[str]:
    """Build and save the inversion diagnostic figures of one run (see the module docstring).

    ``tables`` are residual tables per iteration (:func:`osv.analysis.residuals.residual_tables`);
    the station residual map (initial against final) is drawn from the first and last one.
    """
    saved: list[str] = []

    def emit(fig: Figure, name: str, caption: str) -> None:
        save(fig, f"{name}{suffix}", f"{label}{caption}", "inversion")
        saved.append(f"{name}{suffix}")

    emit(
        plot_misfit_vs_iteration({"final inversion": iterations}, "final inversion"),
        "misfit_vs_iteration",
        "RMS misfit against iteration of the final inversion.",
    )
    if tables:
        emit(
            plot_residual_histograms_iterations(tables),
            "residual_histograms_iterations",
            "Residual (observed - calculated) histograms of every iteration.",
        )
        its = sorted(tables)
        if len(its) >= 2:
            from osv.analysis.plots_residuals import plot_station_residual_map

            emit(
                plot_station_residual_map(tables[its[0]], tables[its[-1]]),
                "station_residual_map",
                "Mean residual per station, starting model and final model.",
            )
    if hypocentres is not None and len(hypocentres):
        emit(
            plot_hypocentre_shifts_iterations(hypocentres),
            "hypocentre_shifts_iterations",
            "Hypocentre shifts from the starting position at each iteration.",
        )
    emit(
        plot_rays_lost(iterations),
        "rays_lost_iterations",
        "Number of valid picks and of picks whose ray was lost, per iteration.",
    )
    emit(
        plot_velocity_range(iterations, min_velocity),
        "velocity_range_iterations",
        "Minimum and maximum velocity and fraction of nodes at the minimum velocity.",
    )
    return saved


# ---------------------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------------------
def chi2_far_from_one(chi2: float) -> bool:
    """True if the reduced chi-squared is outside ``[1/CHI2_TOL, CHI2_TOL]`` (NaN: False)."""
    return bool(math.isfinite(chi2) and (chi2 > CHI2_TOL or chi2 < 1.0 / CHI2_TOL))


def tradeoff_findings(result: Mapping[str, Any], phase: str = "") -> list[QCFinding]:
    """QC findings of one trade-off result (``tradeoff.json`` content).

    ``lcurve_corner`` (warn): no convex corner, so the manual value was used;
    ``lcurve_at_edge`` (warn): the corner is the second or second-last swept value;
    ``sweep_failed`` (warn): some runs failed; ``no_convergence`` (info): runs whose misfit was
    still changing by more than 2 % in the last iteration; ``chi2_far_from_one`` (info) for the
    chosen runs.
    """
    pre = f"{phase}: " if phase else ""
    out: list[QCFinding] = []
    for nm in ("damping", "smoothing"):
        pk = result["picks"][nm]
        curve = [r[nm] for r in pk.get("curve", [])]
        if pk.get("source") == "manual (fallback)":
            out.append(
                QCFinding(
                    "lcurve_corner",
                    "warn",
                    f"{pre}{nm}: no well-defined L-curve corner ({pk.get('reason', '')}); the "
                    f"configured value {_fmt(pk['manual'])} is used. Widen the sweep or choose "
                    "the value by hand (tradeoff.pick: manual).",
                    len(curve),
                )
            )
        elif curve and pk.get("source") == "max_curvature":
            ch = float(pk["chosen"])
            vals = sorted(curve)
            # end points have no curvature, so the corner can at best be the next one in
            if len(vals) >= 4 and (math.isclose(ch, vals[1]) or math.isclose(ch, vals[-2])):
                out.append(
                    QCFinding(
                        "lcurve_at_edge",
                        "warn",
                        f"{pre}{nm}: the corner ({_fmt(ch)}) is next to the edge of the swept "
                        f"range {_fmt(vals[0])}..{_fmt(vals[-1])}; the true corner may lie "
                        "outside it (widen the sweep).",
                        1,
                    )
                )
    tab = pd.DataFrame(result.get("table", []))
    if len(tab):
        nbad = int((~tab["success"].astype(bool)).sum())
        if nbad:
            out.append(QCFinding("sweep_failed", "warn", f"{pre}{nbad} sweep run(s) failed", nbad))
        ok = tab[tab["success"].astype(bool)]
        if len(ok) and "converged" in ok:
            nc = int((~ok["converged"].astype(bool)).sum())
            if nc:
                out.append(
                    QCFinding(
                        "no_convergence",
                        "info",
                        f"{pre}{nc} of {len(ok)} sweep runs still changed their misfit by more "
                        "than 2 % in the last iteration; the L-curve is from unconverged models "
                        "(raise fmtomo.tradeoff.iterations to check).",
                        nc,
                    )
                )
        cd, cs = float(result["chosen_damping"]), float(result["chosen_smoothing"])
        row = ok[
            np.isclose(ok["damping"].astype(float), cd)
            & np.isclose(ok["smoothing"].astype(float), cs)
        ]
        if len(row):
            c2 = _num(row.iloc[0].get("chi2_reduced"))
            if chi2_far_from_one(c2):
                out.append(_chi2_finding(c2, pre))
    return out


def _chi2_finding(c2: float, pre: str = "") -> QCFinding:
    if c2 > 1:
        why = (
            "the picks fit worse than their stated uncertainties allow: the uncertainties are "
            "too small, or the model is not able to explain the data (more iterations, a weaker "
            "regularisation or a better starting model)"
        )
    else:
        why = (
            "the picks fit better than their stated uncertainties: the uncertainties are too large"
        )
    return QCFinding(
        "chi2_far_from_one",
        "info",
        f"{pre}reduced chi-squared is {c2:.2f}, not near 1: {why}. Model uncertainties scaled by "
        "the pick uncertainties should be rescaled by sqrt(chi2) before they are interpreted.",
        0,
        {"chi2_reduced": c2},
    )


def inversion_findings(
    it: pd.DataFrame,
    min_velocity: float | None = None,
    phase: str = "",
) -> list[QCFinding]:
    """QC findings of an inversion's per-iteration table.

    * ``misfit_not_decreasing`` (warn/fail): the final RMS is not below the starting RMS by at
      least 2 % (fail if it is larger);
    * ``misfit_increasing`` (warn): the RMS rose after iteration ``k`` (first such ``k``);
    * ``min_velocity_clipped`` (warn): nodes sit at ``min_velocity``;
    * ``rays_lost`` (warn): more than 5 % of the valid picks lost in an iteration, or from the
      first to the last;
    * ``chi2_far_from_one`` (info): final reduced chi-squared outside 0.5..2;
    * ``no_convergence`` (info): the last iteration still changed the RMS by more than 2 %.
    """
    pre = f"{phase}: " if phase else ""
    out: list[QCFinding] = []
    if it.empty or "rms_ms" not in it:
        return [QCFinding("misfit_not_decreasing", "fail", f"{pre}no iteration results", 0)]
    it = it.sort_values("iter").reset_index(drop=True)
    rms = it["rms_ms"].to_numpy(float)
    if len(rms) >= 2:
        red = 1.0 - rms[-1] / rms[0] if rms[0] > 0 else math.nan
        if rms[-1] >= rms[0]:
            out.append(
                QCFinding(
                    "misfit_not_decreasing",
                    "fail",
                    f"{pre}misfit did not decrease: RMS {rms[0]:.1f} ms to {rms[-1]:.1f} ms. "
                    "Check the damping (too strong?), the data and the geometry.",
                    0,
                    {"rms_start_ms": rms[0], "rms_final_ms": rms[-1]},
                )
            )
        elif red < MIN_REDUCTION:
            out.append(
                QCFinding(
                    "misfit_not_decreasing",
                    "warn",
                    f"{pre}misfit decreased by only {100 * red:.1f} % ({rms[0]:.1f} to "
                    f"{rms[-1]:.1f} ms): the data are not sensitive to the model changes allowed "
                    "by the regularisation.",
                    0,
                )
            )
        up = [int(it["iter"].iloc[i]) for i in range(1, len(rms)) if rms[i] > rms[i - 1]]
        if up:
            out.append(
                QCFinding(
                    "misfit_increasing",
                    "warn",
                    f"{pre}misfit increased after iteration {up[0] - 1} "
                    f"(iterations {up}); the linearised step overshoots: raise the damping or "
                    "the smoothing, or stop at the best iteration.",
                    len(up),
                    {"iterations": up},
                )
            )
        if (
            len(rms) >= 3
            and rms[-2] > 0
            and abs(rms[-1] - rms[-2]) / rms[-2] > CONVERGED_TOL
            and not up
        ):
            out.append(
                QCFinding(
                    "no_convergence",
                    "info",
                    f"{pre}the last iteration still changed the RMS by "
                    f"{100 * abs(rms[-1] - rms[-2]) / rms[-2]:.1f} %; more iterations may help.",
                    0,
                )
            )
    if "frac_at_min_velocity" in it:
        f = float(it["frac_at_min_velocity"].max())
        if f > 0:
            sev = "warn" if f > CLIP_WARN_FRACTION else "info"
            out.append(
                QCFinding(
                    "min_velocity_clipped",
                    sev,
                    f"{pre}up to {100 * f:.2f} % of the nodes sit at the minimum velocity"
                    + (f" ({min_velocity:g} km/s)" if min_velocity else "")
                    + ": the inversion was clipped there; the model is poorly constrained or the "
                    "regularisation too weak.",
                    int(round(f * 1e6)),
                )
            )
    if "n_valid" in it and len(it) >= 2:
        v = it["n_valid"].to_numpy(float)
        drops = [
            int(it["iter"].iloc[i])
            for i in range(1, len(v))
            if v[i - 1] > 0 and (v[i - 1] - v[i]) / v[i - 1] > LOST_FRACTION
        ]
        total = (v[0] - v[-1]) / v[0] if v[0] > 0 else 0.0
        if drops or total > LOST_FRACTION:
            out.append(
                QCFinding(
                    "rays_lost",
                    "warn",
                    f"{pre}{int(max(v[0] - v[-1], 0))} of {int(v[0])} valid picks "
                    f"({100 * total:.1f} %) lost their ray between the first and last iteration"
                    + (f" (large drops at iterations {drops})" if drops else "")
                    + ": rays leave the model or sources moved outside; check the grid and the "
                    "hypocentres.",
                    int(max(v[0] - v[-1], 0)),
                )
            )
    if "chi2" in it:
        c2 = _num(it["chi2"].iloc[-1])
        if chi2_far_from_one(c2):
            out.append(_chi2_finding(c2, pre))
    return out
