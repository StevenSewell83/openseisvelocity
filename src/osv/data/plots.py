"""Stage 1 diagnostic figures.

Every function takes plain DataFrames / dicts (the tables written by
:class:`osv.data.stage.PrepStage`) and returns a matplotlib ``Figure``; saving is done by the
caller via ``ctx.figures.save``.  Nothing here reads the configuration or the file system.

Table conventions
-----------------
``events``
    all catalogue events with ``event_id``, ``lat``, ``lon``, ``depth_km_bsl`` (km, positive
    down), ``time``, ``rms``, ``azgap_deg``, ``n_P``, ``n_S``, ``nearest_station_km``,
    optional ``mag``/``mag_valid`` and ``selected`` (bool), ``split``
    (``calibration``/``validation`` for selected events).
``stations``
    station epochs with ``net``, ``sta``, ``alias``, ``lat``, ``lon``, ``elevation_m``,
    ``sensor_depth_m``, ``sensor_elevation_m`` (see :func:`osv.data.stations.read_stations`).
``picks``
    P/S picks with ``event_id``, ``alias``, ``phase``, ``sigma_s``, ``hypocentral_km``,
    ``travel_time_s``, ``velest_weight``, ``outlier`` and ``selected`` columns.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from osv.coords import LocalProjection
from osv.plotting.style import (
    DEPTH_CMAP,
    PERTURBATION_CMAP,
    add_scale_bar,
    new_figure,
    perturbation_norm,
    plot_events,
    plot_stations,
    setup_map_axes,
)

__all__ = [
    "plot_depth_section",
    "plot_event_timeline",
    "plot_histograms",
    "plot_picks_per_station",
    "plot_ray_map",
    "plot_ray_section",
    "plot_rejections",
    "plot_station_event_map",
    "plot_station_residual_map",
    "plot_station_residuals",
    "plot_travel_times",
    "plot_uncertainty",
    "plot_wadati",
]

#: Colours for VELEST weight classes 0..4 (colour-blind safe; 4 = not used by VELEST).
WEIGHT_COLORS = {0: "#0072B2", 1: "#009E73", 2: "#E69F00", 3: "#D55E00", 4: "#999999"}
PHASE_COLORS = {"P": "#0072B2", "S": "#D55E00"}
GREY = "#a0a0a0"
BOREHOLE_STYLE = {
    "marker": "v",
    "c": "0.85",
    "edgecolors": "black",
    "linewidths": 2.0,
    "s": 60,
    "zorder": 6,
}
_HALO = [patheffects.withStroke(linewidth=2.5, foreground="white")]
#: Gridlines are noise on the map-like panels.
_NO_GRID = {"axes.grid": False}


# ----------------------------------------------------------------------------- helpers


def _flag(df: pd.DataFrame, col: str, default: bool = True) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(bool)
    return pd.Series(default, index=df.index)


def _stations_unique(stations: pd.DataFrame) -> pd.DataFrame:
    """One row (the last epoch) per ``(net, sta)`` with a ``label`` column."""
    st = stations.copy()
    keys = [c for c in ("net", "sta") if c in st.columns]
    if keys:
        st = st.drop_duplicates(keys, keep="last")
    st["label"] = st["alias"] if "alias" in st.columns else st["sta"]
    if "sensor_depth_m" not in st.columns:
        st["sensor_depth_m"] = 0.0
    st["sensor_depth_m"] = st["sensor_depth_m"].fillna(0.0)
    if "sensor_elevation_m" not in st.columns:
        st["sensor_elevation_m"] = st["elevation_m"]
    return st.reset_index(drop=True)


def _used_stations(stations: pd.DataFrame, picks: pd.DataFrame | None) -> tuple[pd.DataFrame, set]:
    st = _stations_unique(stations)
    if picks is None or "alias" not in picks.columns:
        return st, set(st["label"])
    return st, set(picks["alias"].dropna().unique())


def _note(fig: Figure, text: str) -> Figure:
    """Placeholder content for a figure that has no data."""
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color="0.3")
    return fig


def _empty(text: str, **kw: Any) -> Figure:
    fig, ax = new_figure(**kw)
    ax.set_axis_off()
    return _note(fig, text)


def _padded(
    vals: Sequence[float], frac: float = 0.06, min_span: float = 1e-3
) -> tuple[float, float]:
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = max(hi - lo, min_span)
    return lo - frac * span, hi + frac * span


def _depth_limits(
    events: pd.DataFrame, stations: pd.DataFrame | None = None
) -> tuple[float, float]:
    d = events["depth_km_bsl"].dropna().to_numpy(dtype=float)
    lo = float(d.min()) if d.size else 0.0
    hi = float(d.max()) if d.size else 1.0
    if stations is not None and len(stations):
        lo = min(lo, float((-stations["sensor_elevation_m"] / 1000.0).min()))
    return lo - 0.05 * (hi - lo + 1e-3), hi + 0.05 * (hi - lo + 1e-3)


def _projection(origin: tuple[float, float]) -> LocalProjection:
    return LocalProjection(origin[0], origin[1])


def _station_markers(ax, st: pd.DataFrame, xcol: str, ycol: str, used: set, label_boreholes: bool):
    """Used stations as triangles, unused as small grey dots, boreholes as inverted triangles."""
    is_used = st["label"].isin(used)
    bh = st["sensor_depth_m"] > 0
    unused = st[~is_used]
    if len(unused):
        ax.scatter(
            unused[xcol], unused[ycol], marker=".", s=10, c=GREY, zorder=3, label="Unused stations"
        )
    reg = st[is_used & ~bh]
    if len(reg):
        plot_stations(ax, reg[xcol], reg[ycol], label="Stations")
    b = st[is_used & bh]
    if len(b):
        ax.scatter(b[xcol], b[ycol], label="Borehole sensors", **BOREHOLE_STYLE)
        if label_boreholes:
            for x, y, t in zip(b[xcol], b[ycol], b["label"], strict=True):
                ax.annotate(
                    t,
                    (x, y),
                    xytext=(7, 6),
                    textcoords="offset points",
                    fontsize=7,
                    path_effects=_HALO,
                    zorder=8,
                )


def _map_limits(ax, events: pd.DataFrame, st: pd.DataFrame, used: set) -> float:
    """Set lon/lat limits from events and used stations; returns centre latitude."""
    u = st[st["label"].isin(used)]
    lons = np.concatenate([events["lon"].to_numpy(float), u["lon"].to_numpy(float)])
    lats = np.concatenate([events["lat"].to_numpy(float), u["lat"].to_numpy(float)])
    lat0 = float(np.nanmean(lats))
    cosl = math.cos(math.radians(lat0))
    xlo, xhi = _padded(lons)
    ylo, yhi = _padded(lats)
    # equal ground distance: widen the smaller dimension so the box is not extreme
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    setup_map_axes(ax, lat0=lat0)
    del cosl
    return lat0


# ----------------------------------------------------------------------------- maps


def plot_station_event_map(
    events: pd.DataFrame, stations: pd.DataFrame, picks: pd.DataFrame | None = None
) -> Figure:
    """Stations and events before (left) and after (right) the selection.

    Events are coloured by depth below sea level; on the right rejected events are grey
    crosses and validation events have a thick outline (diamonds) against calibration circles.
    Borehole sensors are grey inverted triangles with a thick outline; stations without
    picks are grey dots.
    """
    if events.empty:
        return _empty("No events")
    st, used = _used_stations(stations, picks)
    sel = _flag(events, "selected")
    split = events["split"] if "split" in events.columns else pd.Series("", index=events.index)
    depth = events["depth_km_bsl"].astype(float)
    vmin, vmax = float(depth.min()), float(depth.max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    with plt.rc_context(_NO_GRID):
        fig, axes = new_figure(ncols=2, figsize=(11.0, 5.6), sharex=True, sharey=True)
    ax0, ax1 = axes
    sc = plot_events(ax0, events["lon"], events["lat"], depth, vmin=vmin, vmax=vmax, s=16)
    _station_markers(ax0, st, "lon", "lat", used, label_boreholes=False)
    ax0.set_title(f"Before selection ({len(events)} events)")

    rej = events[~sel]
    if len(rej):
        ax1.scatter(
            rej["lon"],
            rej["lat"],
            marker="x",
            s=22,
            c=GREY,
            linewidths=0.9,
            label="Rejected events",
        )
    cal = events[sel & (split != "validation")]
    val = events[sel & (split == "validation")]
    if len(cal):
        plot_events(ax1, cal["lon"], cal["lat"], cal["depth_km_bsl"], vmin=vmin, vmax=vmax, s=16)
    if len(val):
        plot_events(
            ax1,
            val["lon"],
            val["lat"],
            val["depth_km_bsl"],
            vmin=vmin,
            vmax=vmax,
            s=34,
            marker="D",
            linewidths=1.2,
        )
    _station_markers(ax1, st, "lon", "lat", used, label_boreholes=True)
    ax1.set_title(f"After selection ({int(sel.sum())} events)")
    lat0 = _map_limits(ax0, events, st, used)
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_ylim(ax0.get_ylim())
    setup_map_axes(ax1, lat0=lat0)
    ax1.set_ylabel("")
    add_scale_bar(ax0, loc="lower right", lat0=lat0)
    add_scale_bar(ax1, loc="lower right", lat0=lat0)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="0.6", mec="k", ms=5, label="Calibration event"),
        Line2D([], [], marker="D", ls="", mfc="0.6", mec="k", ms=6, label="Validation event"),
    ]
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles + h1, loc="upper right", fontsize=7)
    h0, l0 = ax0.get_legend_handles_labels()
    ax0.legend(handles=h0, loc="upper right", fontsize=7)
    fig.colorbar(sc, ax=list(axes), shrink=0.8, label="Depth below sea level (km)")
    return fig


def _section_axes(
    events: pd.DataFrame,
    xe: np.ndarray,
    st: pd.DataFrame,
    xs: np.ndarray,
    used: set,
    xlabel: str,
    title: str,
    sel: pd.Series,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    depth = events["depth_km_bsl"].to_numpy(float)
    vmin, vmax = float(np.nanmin(depth)), float(np.nanmax(depth))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    fig, ax = new_figure(figsize=(8.0, 4.8))
    r = ~sel.to_numpy()
    if r.any():
        ax.scatter(
            xe[r], depth[r], marker="x", s=20, c=GREY, linewidths=0.9, label="Rejected events"
        )
    k = sel.to_numpy()
    sc = None
    if k.any():
        sc = ax.scatter(
            xe[k],
            depth[k],
            c=depth[k],
            cmap=DEPTH_CMAP,
            vmin=vmin,
            vmax=vmax,
            s=16,
            edgecolors="k",
            linewidths=0.2,
            zorder=4,
            label="Selected events",
        )
    z = -st["sensor_elevation_m"].to_numpy(float) / 1000.0
    is_used = st["label"].isin(used).to_numpy()
    bh = (st["sensor_depth_m"] > 0).to_numpy()
    m = is_used & ~bh
    if m.any():
        ax.scatter(
            xs[m], z[m], marker="^", s=36, c="white", edgecolors="k", zorder=5, label="Stations"
        )
    m = is_used & bh
    if m.any():
        ax.scatter(
            xs[m],
            z[m],
            label="Borehole sensors",
            **{**BOREHOLE_STYLE, "s": 50},
        )
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.text(
        0.01,
        0.0,
        "sea level",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        fontsize=7,
        color="0.4",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth below sea level (km)")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    lo, hi = _depth_limits(events, st[is_used])
    ax.set_ylim(hi, lo)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", fontsize=7)
    if sc is not None:
        fig.colorbar(sc, ax=ax, shrink=0.8, label="Depth below sea level (km)")
    return fig


def plot_depth_section(
    events: pd.DataFrame,
    stations: pd.DataFrame,
    origin: tuple[float, float],
    kind: str = "ew",
    profile: Mapping[str, Any] | None = None,
    picks: pd.DataFrame | None = None,
    swath_km: float = 1.0,
) -> Figure:
    """Depth cross-section of events with stations at their elevations.

    Parameters
    ----------
    events, stations, picks
        See the module docstring (``picks`` only selects which stations are drawn as used).
    origin
        ``(lat, lon)`` of the local coordinate origin (axes are km from it).
    kind
        ``"ew"`` (x = easting, all events projected), ``"ns"`` (x = northing) or
        ``"profile"`` (needs ``profile`` with ``name``, ``start``, ``end`` as ``[lat, lon]``;
        events and stations within ``swath_km`` either side are shown).
    swath_km
        Half-width of a profile swath.
    """
    if events.empty:
        return _empty("No events")
    proj = _projection(origin)
    st, used = _used_stations(stations, picks)
    ex, ey = proj.forward(events["lat"].to_numpy(float), events["lon"].to_numpy(float))
    sx, sy = proj.forward(st["lat"].to_numpy(float), st["lon"].to_numpy(float))
    sel = _flag(events, "selected")
    if kind == "ew":
        return _section_axes(
            events,
            ex,
            st,
            sx,
            used,
            "Easting from origin (km)",
            "E-W depth section (all events projected)",
            sel,
        )
    if kind == "ns":
        return _section_axes(
            events,
            ey,
            st,
            sy,
            used,
            "Northing from origin (km)",
            "N-S depth section (all events projected)",
            sel,
        )
    if kind != "profile" or profile is None:
        raise ValueError("kind must be 'ew', 'ns' or 'profile' (with a profile)")
    x0, y0 = proj.forward(*profile["start"])
    x1, y1 = proj.forward(*profile["end"])
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length

    def along(x, y):
        return (np.asarray(x) - x0) * ux + (np.asarray(y) - y0) * uy, (
            -(np.asarray(x) - x0) * uy + (np.asarray(y) - y0) * ux
        )

    ea, eo = along(ex, ey)
    sa, so = along(sx, sy)
    keep_e = np.abs(eo) <= swath_km
    keep_s = np.abs(so) <= swath_km
    if not keep_e.any():
        return _empty(f"No events within {swath_km:g} km of profile {profile['name']}")
    ev = events[keep_e]
    return _section_axes(
        ev,
        ea[keep_e],
        st[keep_s].reset_index(drop=True),
        sa[keep_s],
        used,
        f"Distance along {profile['name']} (km)",
        f"Profile {profile['name']} (events within {swath_km:g} km; {int(keep_e.sum())} shown)",
        sel[keep_e],
        xlim=(0.0, length),
    )


# ----------------------------------------------------------------------------- histograms


def plot_histograms(events: pd.DataFrame, cutoffs: Mapping[str, Any]) -> Figure:
    """Event-parameter histograms with the selection cut-offs marked.

    Selected events are blue, rejected events orange (stacked).  ``cutoffs`` holds
    ``min_p``, ``min_s``, ``max_azgap``, ``max_rms``, ``depth_km`` (``[lo, hi]``) and
    ``max_epi_dist_km`` (the pick-level epicentral limit, drawn on the nearest-station panel).
    The magnitude panel appears only when valid magnitudes exist.
    """
    if events.empty:
        return _empty("No events")
    sel = _flag(events, "selected")
    panels: list[tuple[str, str, list[tuple[float, str]] | None, np.ndarray | None]] = [
        ("n_P", "P picks per event", [(cutoffs["min_p"], "min_p")], None),
        ("n_S", "S picks per event", [(cutoffs["min_s"], "min_s")], None),
        ("azgap_deg", "Azimuthal gap (deg)", [(cutoffs["max_azgap"], "max_azgap")], None),
        ("rms", "Catalogue RMS (s)", [(cutoffs["max_rms"], "max_rms")], None),
        (
            "depth_km_bsl",
            "Depth below sea level (km)",
            [(cutoffs["depth_km"][0], "min depth"), (cutoffs["depth_km"][1], "max depth")],
            None,
        ),
        (
            "nearest_station_km",
            "Nearest station, epicentral (km)",
            [(cutoffs["max_epi_dist_km"], "max_epi_dist_km")],
            None,
        ),
    ]
    if "mag" in events.columns and _flag(events, "mag_valid", False).any():
        panels.append(("mag", "Magnitude", None, None))
    ncols = 3
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = new_figure(nrows=nrows, ncols=ncols, figsize=(11.0, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (col, label, lines, _) in zip(axes, panels, strict=False):
        data = events[col].astype(float)
        if col == "mag":
            data = data[_flag(events, "mag_valid", False)]
        vals = data.dropna()
        if vals.empty:
            ax.set_axis_off()
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if col != "nearest_station_km":  # a far cut-off would squash the data
            for v, _name in lines or []:
                if np.isfinite(v):
                    lo, hi = min(lo, v), max(hi, v)
        bins = np.linspace(lo, hi, 21) if hi > lo else 10
        ax.hist(
            [data[sel.reindex(data.index)].dropna(), data[~sel.reindex(data.index)].dropna()],
            bins=bins,
            stacked=True,
            color=["#0072B2", "#E69F00"],
            edgecolor="white",
            linewidth=0.4,
            label=["Selected", "Rejected"],
        )
        for v, name in lines or []:
            if np.isfinite(v):
                ax.axvline(v, color="crimson", ls="--", lw=1.2, label=f"{name} = {v:g}")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel(label)
        ax.set_ylabel("Events")
    for ax in axes[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle("Event parameters and selection cut-offs", fontsize=11)
    return fig


def plot_event_timeline(events: pd.DataFrame) -> Figure:
    """Event depth against origin time with the cumulative event count."""
    if events.empty:
        return _empty("No events")
    ev = events.sort_values("time")
    sel = _flag(ev, "selected")
    t = pd.to_datetime(ev["time"], utc=True)
    fig, ax = new_figure(figsize=(9.0, 4.5))
    depth = ev["depth_km_bsl"].astype(float)
    if (~sel).any():
        ax.scatter(t[~sel], depth[~sel], marker="x", s=20, c=GREY, linewidths=0.9, label="Rejected")
    sc = ax.scatter(
        t[sel],
        depth[sel],
        c=depth[sel],
        cmap=DEPTH_CMAP,
        s=18,
        edgecolors="k",
        linewidths=0.2,
        zorder=4,
        label="Selected",
    )
    ax.invert_yaxis()
    ax.set_ylabel("Depth below sea level (km)")
    ax.set_xlabel("Origin time (UTC)")
    ax2 = ax.twinx()
    ax2.step(t, np.arange(1, len(t) + 1), where="post", color="0.25", lw=1.2)
    ax2.set_ylabel("Cumulative events", color="0.25")
    ax2.grid(False)
    ax.set_title("Event timeline")
    ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(sc, ax=[ax, ax2], shrink=0.8, label="Depth (km)", pad=0.02)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    return fig


# ----------------------------------------------------------------------------- picks


def plot_picks_per_station(picks: pd.DataFrame, min_picks: int | None = None) -> Figure:
    """Picks per station, P and S stacked, sorted by total.

    Picks retained by the selection are solid; picks that were rejected are drawn in the same
    colour with hatching.
    """
    if picks.empty:
        return _empty("No picks")
    sel = _flag(picks, "selected")
    tab = (
        picks.assign(_sel=sel)
        .groupby(["alias", "phase", "_sel"])
        .size()
        .unstack(["phase", "_sel"], fill_value=0)
    )
    for ph in ("P", "S"):
        for s in (True, False):
            if (ph, s) not in tab.columns:
                tab[(ph, s)] = 0
    tab = tab.assign(total=tab.sum(axis=1)).sort_values("total", ascending=True)
    n = len(tab)
    fig, ax = new_figure(figsize=(7.0, max(4.0, 0.2 * n + 1.5)))
    y = np.arange(n)
    left = np.zeros(n)
    for ph in ("P", "S"):
        for s in (True, False):
            v = tab[(ph, s)].to_numpy(float)
            ax.barh(
                y,
                v,
                left=left,
                color=PHASE_COLORS[ph],
                edgecolor="white" if s else "0.2",
                hatch=None if s else "///",
                linewidth=0.4 if s else 0.5,
                label=f"{ph}{'' if s else ' (rejected)'}",
            )
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels(tab.index, fontsize=7)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, float(tab["total"].max()) * 1.05)
    ax.set_xlabel("Picks")
    ax.set_ylabel("Station")
    ax.set_title(f"Picks per station ({n} stations)")
    if min_picks:
        ax.axvline(min_picks, color="crimson", ls="--", lw=1.0)
        ax.text(
            min_picks,
            0.01,
            f" {min_picks}",
            color="crimson",
            fontsize=7,
            transform=ax.get_xaxis_transform(),
        )
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    return fig


def plot_travel_times(
    picks: pd.DataFrame,
    tt_fits: Mapping[str, Mapping[str, Any]],
    outlier_sigma: float = 3.0,
) -> Figure:
    """Travel time against hypocentral distance (P left, S right).

    Points are coloured by VELEST weight class, outliers (travel-time or Wadati) get a red
    cross, and the robust polynomial fit is drawn with a ``+/- outlier_sigma * sigma`` band
    (``tt_fits[phase] = {"coef": [...], "sigma_s": float}``; coefficients highest power first).
    """
    if picks.empty:
        return _empty("No picks")
    fig, axes = new_figure(ncols=2, figsize=(11.0, 4.8), sharex=True)
    out = _flag(picks, "outlier", False)
    for ax, ph in zip(axes, ("P", "S"), strict=True):
        sub = picks[picks["phase"] == ph]
        o = out[sub.index]
        for w in sorted(sub["velest_weight"].dropna().unique()):
            s = sub[(sub["velest_weight"] == w)]
            ax.scatter(
                s["hypocentral_km"],
                s["travel_time_s"],
                s=9,
                c=WEIGHT_COLORS[int(w)],
                alpha=0.75,
                linewidths=0,
                label=f"weight {int(w)} ({len(s)})",
            )
        if o.any():
            ax.scatter(
                sub.loc[o, "hypocentral_km"],
                sub.loc[o, "travel_time_s"],
                marker="x",
                s=36,
                c="crimson",
                linewidths=1.0,
                zorder=6,
                label=f"outlier ({int(o.sum())})",
            )
        fit = tt_fits.get(ph)
        if fit is not None and len(sub):
            d = np.linspace(0.0, float(sub["hypocentral_km"].max()), 100)
            t = np.polyval(np.asarray(fit["coef"], float), d)
            ax.plot(d, t, color="k", lw=1.4, label="robust fit", zorder=5)
            band = outlier_sigma * float(fit["sigma_s"])
            ax.fill_between(
                d,
                t - band,
                t + band,
                color="0.5",
                alpha=0.18,
                lw=0,
                label=f"+/- {outlier_sigma:g} sigma",
                zorder=2,
            )
        ax.set_title(f"{ph} travel times ({len(sub)} picks)")
        ax.set_xlabel("Hypocentral distance (km)")
        ax.set_ylabel("Travel time (s)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=7)
    return fig


def plot_wadati(
    pairs: pd.DataFrame,
    wadati: Mapping[str, Any],
    event_vpvs: pd.DataFrame | None = None,
) -> Figure:
    """Wadati diagram: ``Ts - Tp`` against ``Tp`` with pooled fit, CI band and outliers.

    Parameters
    ----------
    pairs
        ``tp_s``, ``sp_s`` (``Ts - Tp``) and ``outlier`` per S pick with a P pick.
    wadati
        ``vpvs``, ``vpvs_ci`` ``[lo, hi]``, optionally ``event_vpvs_median``.
    event_vpvs
        Per-event table with ``vpvs`` (and ``inconsistent``), shown as an inset histogram.
    """
    if pairs.empty or not wadati:
        return _empty("No S-P pairs available for a Wadati diagram")
    fig, ax = new_figure(figsize=(7.5, 5.5))
    out = _flag(pairs, "outlier", False)
    ax.scatter(
        pairs.loc[~out, "tp_s"],
        pairs.loc[~out, "sp_s"],
        s=14,
        c="#0072B2",
        alpha=0.7,
        linewidths=0,
        label=f"S-P pairs ({int((~out).sum())})",
    )
    if out.any():
        ax.scatter(
            pairs.loc[out, "tp_s"],
            pairs.loc[out, "sp_s"],
            marker="x",
            s=36,
            c="crimson",
            linewidths=1.1,
            zorder=5,
            label=f"outliers ({int(out.sum())})",
        )
    tmax = float(pairs["tp_s"].max()) * 1.05
    tl = np.array([0.0, tmax])
    k = float(wadati["vpvs"]) - 1.0
    lo, hi = (float(v) - 1.0 for v in wadati["vpvs_ci"])
    ax.fill_between(tl, lo * tl, hi * tl, color="k", alpha=0.25, lw=0, label="95% CI", zorder=3)
    ax.plot(
        tl,
        k * tl,
        "k-",
        lw=1.4,
        zorder=4,
        label=f"pooled Vp/Vs = {wadati['vpvs']:.3f} "
        f"({wadati['vpvs_ci'][0]:.3f}-{wadati['vpvs_ci'][1]:.3f})",
    )
    ax.set_xlim(0, tmax)
    ax.set_ylim(bottom=min(0.0, float(pairs["sp_s"].min()) * 1.05))
    ax.set_xlabel("P travel time Tp (s)")
    ax.set_ylabel("S-P time Ts - Tp (s)")
    ax.set_title("Wadati diagram")
    ax.legend(loc="upper left", fontsize=7)
    if event_vpvs is not None and len(event_vpvs):
        ins = ax.inset_axes((0.62, 0.10, 0.35, 0.28))
        v = event_vpvs["vpvs"].astype(float).dropna()
        ins.hist(
            v, bins=min(20, max(5, len(v) // 3)), color="#009E73", edgecolor="white", linewidth=0.4
        )
        ins.axvline(float(wadati["vpvs"]), color="k", lw=1.0)
        if "event_vpvs_median" in wadati and np.isfinite(wadati["event_vpvs_median"]):
            ins.axvline(float(wadati["event_vpvs_median"]), color="#009E73", ls="--", lw=1.0)
        ins.set_xlabel("per-event Vp/Vs", fontsize=7)
        ins.set_ylabel("events", fontsize=7)
        ins.tick_params(labelsize=6)
        ins.grid(False)
        ins.set_title(f"{len(v)} events, median {v.median():.2f}", fontsize=7)
    return fig


# ----------------------------------------------------------------------------- rays


def _ray_frame(picks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if picks.empty or events.empty:
        return pd.DataFrame()
    ev = events.set_index("event_id")[["lat", "lon", "depth_km_bsl"]].rename(
        columns={"lat": "ev_lat", "lon": "ev_lon", "depth_km_bsl": "ev_depth_km"}
    )
    sub = picks[_flag(picks, "selected")]
    return sub.join(ev, on="event_id", how="inner")


def _ray_alpha(n: int) -> float:
    return float(min(0.3, max(0.03, 120.0 / max(n, 1))))


def plot_ray_map(picks: pd.DataFrame, events: pd.DataFrame, stations: pd.DataFrame) -> Figure:
    """Straight ray paths (event to station, selected picks) in map view, alpha-blended."""
    rp = _ray_frame(picks, events)
    if rp.empty:
        return _empty("No selected picks")
    st, used = _used_stations(stations, picks)
    with plt.rc_context(_NO_GRID):
        fig, ax = new_figure(figsize=(7.0, 6.0))
    segs = np.stack(
        [
            np.column_stack([rp["ev_lon"], rp["ev_lat"]]),
            np.column_stack([rp["station_lon"], rp["station_lat"]]),
        ],
        axis=1,
    )
    ax.add_collection(
        LineCollection(segs, colors="#0072B2", linewidths=0.5, alpha=_ray_alpha(len(rp)), zorder=2)
    )
    sel_ev = events[_flag(events, "selected")]
    ax.scatter(
        sel_ev["lon"], sel_ev["lat"], s=8, c="crimson", zorder=4, label="Events", linewidths=0
    )
    _station_markers(ax, st, "lon", "lat", used, label_boreholes=True)
    lat0 = _map_limits(ax, sel_ev, st, used)
    add_scale_bar(ax, loc="lower right", lat0=lat0)
    ax.set_title(f"Ray-path coverage ({len(rp)} selected picks)")
    ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_ray_section(
    picks: pd.DataFrame,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    origin: tuple[float, float],
    kind: str = "ew",
) -> Figure:
    """Ray paths projected on a depth section (``"ew"`` easting or ``"ns"`` northing)."""
    rp = _ray_frame(picks, events)
    if rp.empty:
        return _empty("No selected picks")
    proj = _projection(origin)
    ex, ey = proj.forward(rp["ev_lat"].to_numpy(float), rp["ev_lon"].to_numpy(float))
    sx, sy = proj.forward(rp["station_lat"].to_numpy(float), rp["station_lon"].to_numpy(float))
    a, b, label = (ex, sx, "Easting") if kind == "ew" else (ey, sy, "Northing")
    zs = -rp["sensor_elevation_m"].to_numpy(float) / 1000.0
    segs = np.stack([np.column_stack([a, rp["ev_depth_km"]]), np.column_stack([b, zs])], axis=1)
    fig, ax = new_figure(figsize=(8.0, 4.8))
    ax.add_collection(
        LineCollection(segs, colors="#0072B2", linewidths=0.5, alpha=_ray_alpha(len(rp)), zorder=2)
    )
    ax.scatter(a, rp["ev_depth_km"], s=4, c="crimson", zorder=4, linewidths=0, label="Events")
    ax.scatter(
        b,
        zs,
        marker="^",
        s=20,
        c="white",
        edgecolors="k",
        linewidths=0.6,
        zorder=5,
        label="Stations",
    )
    xs_all = np.concatenate([a, b])
    ax.set_xlim(*_padded(xs_all))
    lo, hi = _depth_limits(events[_flag(events, "selected")])
    ax.set_ylim(hi, min(lo, float(zs.min()) - 0.05))
    ax.set_xlabel(f"{label} from origin (km)")
    ax.set_ylabel("Depth below sea level (km)")
    ax.set_title(f"Ray paths, {'E-W' if kind == 'ew' else 'N-S'} depth section")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=7)
    return fig


# ----------------------------------------------------------------------------- uncertainty


def plot_uncertainty(picks: pd.DataFrame, weight_thresholds: Mapping[float, int]) -> Figure:
    """Pick-uncertainty (``sigma_s``) distributions per phase and weight-class counts.

    ``weight_thresholds`` is ``data_selection.pick_uncertainty_to_weight``; its keys are
    drawn as dashed lines on the histogram.
    """
    sig = picks["sigma_s"].astype(float)
    ok = np.isfinite(sig) & (sig > 0)
    if not ok.any():
        return _empty("No finite pick uncertainties")
    fig, (ax, ax2) = new_figure(ncols=2, figsize=(11.0, 4.5))
    lo, hi = float(sig[ok].min()), float(sig[ok].max())
    for k in weight_thresholds:
        lo, hi = min(lo, k), max(hi, k)
    bins = np.geomspace(lo * 0.9, hi * 1.1, 30)
    for ph in ("P", "S"):
        v = sig[ok & (picks["phase"] == ph)]
        if len(v):
            ax.hist(
                v,
                bins=bins,
                histtype="stepfilled",
                alpha=0.45,
                color=PHASE_COLORS[ph],
                label=f"{ph} ({len(v)})",
            )
            ax.hist(v, bins=bins, histtype="step", color=PHASE_COLORS[ph], lw=1.2)
    for k, w in sorted(weight_thresholds.items()):
        ax.axvline(k, color="0.3", ls="--", lw=0.9, label=f"<= {k:g} s: class {w}")
    ax.set_xscale("log")
    ax.set_xlabel("Pick uncertainty sigma (s)")
    ax.set_ylabel("Picks")
    ax.set_title("Pick uncertainty")
    ax.legend(loc="upper left", fontsize=7)
    classes = list(range(5))
    width = 0.38
    for i, ph in enumerate(("P", "S")):
        counts = [
            int(((picks["phase"] == ph) & (picks["velest_weight"] == c)).sum()) for c in classes
        ]
        bars = ax2.bar(
            np.array(classes) + (i - 0.5) * width, counts, width, color=PHASE_COLORS[ph], label=ph
        )
        for r, c in zip(bars, counts, strict=True):
            if c:
                ax2.text(
                    r.get_x() + r.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=7
                )
    ax2.set_xticks(classes)
    ax2.set_xticklabels([str(c) if c < 4 else "4 (unused)" for c in classes])
    ax2.set_xlabel("VELEST weight class")
    ax2.set_ylabel("Picks")
    ax2.set_title("Weight classes (all associated picks)")
    ax2.legend(fontsize=7)
    return fig


# ----------------------------------------------------------------------------- stations


def plot_station_residuals(resid: pd.DataFrame) -> Figure:
    """Median travel-time residual per station and phase with standard-error bars.

    ``resid`` has ``station``, ``phase``, ``n``, ``median_residual_s``, ``se_s``, ``systematic``.
    Systematic stations are drawn in red; the stations are ordered by P residual.
    """
    if resid.empty:
        return _empty("No station residuals")
    fig, axes = new_figure(nrows=2, figsize=(11.0, 6.5), sharex=False)
    for ax, ph in zip(axes, ("P", "S"), strict=True):
        sub = resid[resid["phase"] == ph].sort_values("median_residual_s").reset_index(drop=True)
        if sub.empty:
            ax.set_axis_off()
            continue
        sysm = sub["systematic"].astype(bool)
        x = np.arange(len(sub))
        cols = np.where(sysm, "crimson", "#0072B2")
        ax.errorbar(
            x,
            sub["median_residual_s"],
            yerr=sub["se_s"],
            fmt="none",
            ecolor="0.4",
            elinewidth=0.8,
            capsize=2,
            zorder=2,
        )
        ax.scatter(
            x, sub["median_residual_s"], c=cols, s=22, zorder=3, edgecolors="k", linewidths=0.3
        )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["station"], rotation=90, fontsize=7)
        ax.set_ylabel(f"Median {ph} residual (s)")
        ax.set_title(f"{ph}: {int(sysm.sum())} of {len(sub)} stations flagged as systematic (red)")
        ax.grid(axis="x", visible=False)
    axes[0].text(
        0.01,
        0.02,
        "residual = observed - 1-D fit; bars are 1 standard error",
        transform=axes[0].transAxes,
        fontsize=7,
        color="0.3",
    )
    return fig


def plot_station_residual_map(resid: pd.DataFrame, stations: pd.DataFrame) -> Figure:
    """Map of median travel-time residual per station (P left, S right).

    Systematic stations get a thick black outline.
    """
    if resid.empty:
        return _empty("No station residuals")
    st = _stations_unique(stations)
    with plt.rc_context(_NO_GRID):
        fig, axes = new_figure(ncols=2, figsize=(11.0, 5.4), sharex=True, sharey=True)
    vals = resid["median_residual_s"].astype(float)
    norm = perturbation_norm(vals, vmax=max(float(np.abs(vals).max()), 1e-3))
    sc = None
    lat0 = None
    for ax, ph in zip(axes, ("P", "S"), strict=True):
        sub = resid[resid["phase"] == ph].merge(
            st[["label", "lat", "lon"]], left_on="station", right_on="label", how="inner"
        )
        if sub.empty:
            continue
        sysm = sub["systematic"].astype(bool)
        sc = ax.scatter(
            sub["lon"],
            sub["lat"],
            c=sub["median_residual_s"],
            cmap=PERTURBATION_CMAP,
            norm=norm,
            marker="^",
            s=70,
            edgecolors=np.where(sysm, "k", "0.6"),
            linewidths=np.where(sysm, 1.8, 0.5),
            zorder=4,
        )
        ax.set_title(f"{ph} median residual ({int(sysm.sum())} systematic)")
        for x, y, t, s in zip(sub["lon"], sub["lat"], sub["station"], sysm, strict=True):
            if s:
                ax.annotate(
                    t,
                    (x, y),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=6,
                    path_effects=_HALO,
                )
        if lat0 is None:
            lat0 = float(sub["lat"].mean())
            ax.set_xlim(*_padded(sub["lon"]))
            ax.set_ylim(*_padded(sub["lat"]))
    lat0 = lat0 if lat0 is not None else 0.0
    for ax in axes:
        ax.set_xlim(axes[0].get_xlim())
        ax.set_ylim(axes[0].get_ylim())
        setup_map_axes(ax, lat0=lat0)
        add_scale_bar(ax, loc="lower right", lat0=lat0)
    axes[1].set_ylabel("")
    if sc is not None:
        fig.colorbar(sc, ax=list(axes), shrink=0.8, label="Median residual (s); red = late")
    return fig


# ----------------------------------------------------------------------------- rejections


def plot_rejections(
    rejections: pd.DataFrame, n_events: int, n_picks: int, picks_with_rejected_events: int = 0
) -> Figure:
    """Counts per rejection reason, for events (left) and picks (right).

    An item failing several filters counts once per reason.  ``picks_with_rejected_events``
    adds the picks lost because their event was rejected.
    """
    fig, axes = new_figure(ncols=2, figsize=(11.0, 4.0))
    specs = (
        ("event", n_events, "Events", 0),
        ("pick", n_picks, "Picks", picks_with_rejected_events),
    )
    for ax, (level, total, name, extra) in zip(axes, specs, strict=True):
        r = rejections[rejections["level"] == level] if len(rejections) else rejections
        counts = r.groupby("reason").size().sort_values() if len(r) else pd.Series(dtype=int)
        if extra:
            counts = pd.concat([counts, pd.Series({"event rejected": extra})]).sort_values()
        if counts.empty:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"No {name.lower()} rejected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.3",
            )
            continue
        bars = ax.barh(
            counts.index, counts.to_numpy(), height=0.5, color="#E69F00", edgecolor="white"
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for b, c in zip(bars, counts.to_numpy(), strict=True):
            ax.text(
                c,
                b.get_y() + b.get_height() / 2,
                f" {c} ({100 * c / max(total, 1):.1f}%)",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, float(counts.max()) * 1.3)
        ax.set_xlabel(f"{name} rejected (of {total})")
        ax.set_title(f"{name} rejected by reason")
        ax.grid(axis="y", visible=False)
    return fig
