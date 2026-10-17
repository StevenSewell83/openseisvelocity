"""Figures of the 1-D vs 3-D validation relocation (Stage 7).

Every function takes a :class:`~osv.validation.relocate.ValidationResult` and returns a
matplotlib figure (or ``None`` when there is nothing to show).  :func:`make_figures` saves all
of them through a :class:`~osv.plotting.registry.FigureRegistry` with captions.

Colours: 1-D is blue, 3-D is orange throughout; the same colour pair is used in every figure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from osv.plotting.registry import FigureRegistry
from osv.plotting.style import new_figure, setup_map_axes
from osv.validation.relocate import ValidationResult

__all__ = [
    "make_figures",
    "plot_depth_change_hist",
    "plot_map_sections",
    "plot_rms_change_hist",
    "plot_rms_scatter",
]

C1D = "#1f77b4"
C3D = "#ff7f0e"
CCAT = "0.55"


def _title_stats(res: ValidationResult) -> str:
    r = res.stats.get("rms", {})
    n = int(r.get("n", 0))
    p = r.get("p_two_sided", float("nan"))
    ptxt = f", Wilcoxon p = {p:.3g}" if isinstance(p, float) and math.isfinite(p) else ""
    return f"n = {n}{ptxt}"


def plot_rms_scatter(res: ValidationResult) -> Any:
    """Paired RMS (ms) of each held-out event: 1-D (x) against 3-D (y) with the 1:1 line."""
    ev = res.events
    x = 1000.0 * ev["rms_1d"].to_numpy(float)
    y = 1000.0 * ev["rms_3d"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None
    fig, ax = new_figure(figsize=(5.0, 5.0))
    hi = float(max(x[ok].max(), y[ok].max())) * 1.08
    ax.plot([0, hi], [0, hi], "k-", lw=1, label="1:1")
    ax.scatter(x[ok], y[ok], s=28, c=C3D, edgecolors="k", linewidths=0.4, zorder=3)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("RMS with 1-D model (ms)")
    ax.set_ylabel("RMS with 3-D model (ms)")
    r = res.stats["rms"]
    ax.text(
        0.04,
        0.96,
        f"median 1-D {1000 * r['median_1d']:.0f} ms\nmedian 3-D {1000 * r['median_3d']:.0f} ms\n"
        f"3-D better: {100 * r['frac_3d_better']:.0f}% of events",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.set_title(f"Held-out events: paired RMS\n({_title_stats(res)})", fontsize=10)
    ax.text(0.97, 0.03, "3-D better below the line", transform=ax.transAxes, ha="right", fontsize=7)
    return fig


def plot_rms_change_hist(res: ValidationResult) -> Any:
    """Histogram of the RMS change ``rms_1d - rms_3d`` (ms); positive = 3-D fits better."""
    d = 1000.0 * res.events["rms_change"].to_numpy(float)
    d = d[np.isfinite(d)]
    if not d.size:
        return None
    fig, ax = new_figure(figsize=(6.0, 4.0))
    lim = float(np.max(np.abs(d))) * 1.1 or 1.0
    bins = np.linspace(-lim, lim, max(5, min(25, d.size // 2 + 5)))
    ax.hist(d, bins=bins, color=C3D, edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(
        float(np.median(d)), color=C1D, lw=1.5, ls="--", label=f"median {np.median(d):.1f} ms"
    )
    ax.set_xlabel("RMS(1-D) - RMS(3-D) (ms); positive = 3-D better")
    ax.set_ylabel("Events")
    ax.set_title(f"RMS change per event ({_title_stats(res)})")
    ax.legend()
    return fig


def plot_depth_change_hist(res: ValidationResult) -> Any:
    """Histogram of the depth change ``depth_3d - depth_1d`` (km, positive = deeper in 3-D)."""
    d = res.events["dz_km"].to_numpy(float)
    d = d[np.isfinite(d)]
    if not d.size:
        return None
    fig, ax = new_figure(figsize=(6.0, 4.0))
    lim = float(np.max(np.abs(d))) * 1.1 or 0.5
    bins = np.linspace(-lim, lim, max(5, min(25, d.size // 2 + 5)))
    ax.hist(d, bins=bins, color=C3D, edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(
        float(np.median(d)), color=C1D, lw=1.5, ls="--", label=f"median {np.median(d):.2f} km"
    )
    ax.set_xlabel("Depth(3-D) - depth(1-D) (km); positive = deeper with the 3-D model")
    ax.set_ylabel("Events")
    h = res.stats["hypocentre_difference"]
    ax.set_title(f"Depth change (median horizontal shift {h['dh_median_km']:.2f} km)")
    ax.legend()
    return fig


def plot_map_sections(res: ValidationResult) -> Any:
    """Map and two sections of the 1-D (blue) and 3-D (orange) locations joined by lines.

    Panels: map (lon/lat), E-W section (longitude against depth) and N-S section (depth against
    latitude).  Catalogue positions are grey crosses.
    """
    ev = res.events
    need = ["lat_1d", "lon_1d", "depth_1d", "lat_3d", "lon_3d", "depth_3d"]
    ok = np.isfinite(ev[need].to_numpy(float)).all(axis=1)
    if not ok.any():
        return None
    e = ev[ok]
    fig = _three_panel()
    ax_map, ax_ew, ax_ns = fig.axes[:3]
    pairs = [
        (ax_map, "lon", "lat"),
        (ax_ew, "lon", "depth"),
        (ax_ns, "depth", "lat"),
    ]
    for ax, ka, kb in pairs:
        for i in range(len(e)):
            ax.plot(
                [e[f"{ka}_1d"].iloc[i], e[f"{ka}_3d"].iloc[i]],
                [e[f"{kb}_1d"].iloc[i], e[f"{kb}_3d"].iloc[i]],
                "-",
                color="0.4",
                lw=0.7,
                zorder=2,
            )
        if "lat0" in e:
            ax.scatter(
                e[f"{ka}0" if ka != "depth" else "depth0"],
                e[f"{kb}0" if kb != "depth" else "depth0"],
                marker="+",
                s=22,
                c=CCAT,
                zorder=1,
                label="catalogue",
            )
        ax.scatter(
            e[f"{ka}_1d"],
            e[f"{kb}_1d"],
            s=22,
            c=C1D,
            edgecolors="k",
            linewidths=0.3,
            zorder=3,
            label="1-D (VELEST)",
        )
        ax.scatter(
            e[f"{ka}_3d"],
            e[f"{kb}_3d"],
            s=22,
            c=C3D,
            edgecolors="k",
            linewidths=0.3,
            zorder=4,
            label="3-D (FMTOMO)",
        )
    ax_map.set_xlabel("Longitude (deg)")
    ax_map.set_ylabel("Latitude (deg)")
    ax_map.set_title("Map")
    lat0 = float(e["lat_1d"].mean())
    setup_map_axes(ax_map, lat0)
    ax_ew.set_xlabel("Longitude (deg)")
    ax_ew.set_ylabel("Depth (km)")
    ax_ew.invert_yaxis()
    ax_ew.set_title("E-W section")
    ax_ns.set_xlabel("Depth (km)")
    ax_ns.set_ylabel("Latitude (deg)")
    ax_ns.set_title("N-S section")
    ax_map.legend(loc="best")
    sh = res.stats["hypocentre_difference"]
    fig.suptitle(
        f"1-D vs 3-D locations of {len(e)} held-out events "
        f"(median shift {sh['dh_median_km']:.2f} km horizontal, "
        f"{sh['dz_abs_median_km']:.2f} km depth)"
    )
    return fig


def _three_panel() -> Any:
    import matplotlib
    import matplotlib.pyplot as plt

    from osv.plotting.style import RC

    with matplotlib.rc_context(RC):
        fig = plt.figure(figsize=(10.0, 6.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0])
        fig.add_subplot(gs[:, 0])
        fig.add_subplot(gs[0, 1])
        fig.add_subplot(gs[1, 1])
    return fig


def make_figures(
    res: ValidationResult, figures: FigureRegistry | str | Path, section: str = "Validation"
) -> list[str]:
    """Save all validation figures; returns the names written.

    ``figures`` is a registry or a directory (png and pdf are written then).
    """
    reg = figures if isinstance(figures, FigureRegistry) else FigureRegistry(Path(figures))
    items = [
        (
            "validation_rms_scatter",
            plot_rms_scatter,
            "Per-event RMS residual of the held-out events located with the 1-D and the 3-D "
            "model (same picks, origin time fitted). Points below the 1:1 line fit better with "
            "the 3-D model.",
        ),
        (
            "validation_rms_change",
            plot_rms_change_hist,
            "Change of the per-event RMS (1-D minus 3-D). Positive values mean the 3-D model "
            "fits the held-out event better.",
        ),
        (
            "validation_locations",
            plot_map_sections,
            "Held-out event locations with the 1-D model (blue) and the 3-D model (orange); a "
            "line joins the two locations of each event, grey crosses are the catalogue.",
        ),
        (
            "validation_depth_change",
            plot_depth_change_hist,
            "Depth change of the held-out events between the 3-D and the 1-D relocation.",
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
