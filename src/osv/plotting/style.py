"""Consistent matplotlib style and map/marker helpers (headless-safe)."""

from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; only affects a not-yet-chosen backend in practice

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: Colour map for absolute velocities (perceptually uniform; override per call).
VELOCITY_CMAP = "viridis"
#: Diverging map for perturbations: red = slow (negative), blue = fast, centred at 0.
PERTURBATION_CMAP = "RdBu"
#: Colour map for event depth.
DEPTH_CMAP = "plasma_r"

STATION_STYLE: dict[str, Any] = {
    "marker": "^",
    "s": 36,
    "c": "white",
    "edgecolors": "black",
    "linewidths": 0.8,
    "zorder": 5,
}
EVENT_STYLE: dict[str, Any] = {"marker": "o", "s": 10, "linewidths": 0.2, "edgecolors": "k"}

RC: dict[str, Any] = {
    "figure.figsize": (7.0, 5.0),
    "figure.constrained_layout.use": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "legend.framealpha": 0.8,
    "image.cmap": VELOCITY_CMAP,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    """Install the osv rcParams globally."""
    matplotlib.rcParams.update(RC)


def new_figure(**kwargs: Any):
    """Create a figure under the osv style (``plt.subplots`` arguments pass through)."""
    with matplotlib.rc_context(RC):
        return plt.subplots(**kwargs)


def velocity_cmap(name: str | None = None):
    """Colour map for absolute velocity (default `VELOCITY_CMAP`; ``"turbo"`` also works)."""
    return plt.get_cmap(name or VELOCITY_CMAP)


def perturbation_norm(data: Any = None, vmax: float | None = None, vmin: float | None = None):
    """Symmetric normalisation centred at 0 for perturbation plots.

    Give ``vmax`` explicitly or let it default to the largest absolute finite value of ``data``.
    """
    from matplotlib.colors import TwoSlopeNorm

    if vmax is None:
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        vmax = float(np.abs(arr).max()) if arr.size else 1.0
    vmax = vmax or 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-abs(vmin if vmin is not None else vmax), vmax=vmax)


def setup_map_axes(ax, lat0: float | None = None, lon_label: str = "Longitude (deg)") -> None:
    """Make an axes a lon/lat map with the correct aspect ``1/cos(lat)``.

    ``lat0`` defaults to the centre of the current y-limits.
    """
    if lat0 is None:
        lo, hi = ax.get_ylim()
        lat0 = 0.5 * (lo + hi)
    ax.set_aspect(1.0 / max(math.cos(math.radians(lat0)), 1e-3))
    ax.set_xlabel(lon_label)
    ax.set_ylabel("Latitude (deg)")


def add_scale_bar(
    ax, length_km: float | None = None, loc: str = "lower left", lat0: float | None = None
) -> float:
    """Draw a km scale bar on a lon/lat axes; returns the length used (km).

    The default length is a round number (1, 2, 5 x 10^n) near 20% of the map width.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if lat0 is None:
        lat0 = 0.5 * (y0 + y1)
    km_per_deg_lon = 111.195 * math.cos(math.radians(lat0))
    width_km = abs(x1 - x0) * km_per_deg_lon
    if length_km is None:
        target = 0.2 * width_km
        mag = 10 ** math.floor(math.log10(max(target, 1e-9)))
        length_km = min((m * mag for m in (1, 2, 5, 10)), key=lambda v: abs(v - target))
    dlon = length_km / km_per_deg_lon
    fx = 0.05 if "left" in loc else 0.95 - dlon / (x1 - x0)
    fy = 0.06 if "lower" in loc else 0.94
    xs = min(x0, x1) + fx * abs(x1 - x0)
    ys = min(y0, y1) + fy * abs(y1 - y0)
    ax.plot([xs, xs + dlon], [ys, ys], "k-", lw=2.5, solid_capstyle="butt", zorder=10)
    label = f"{length_km:g} km"
    ax.text(xs + dlon / 2, ys, label, ha="center", va="bottom", fontsize=8, zorder=10)
    return float(length_km)


def plot_stations(ax, lon, lat, labels=None, **kw):
    """Stations as white triangles; optional name labels."""
    style = {**STATION_STYLE, **kw}
    sc = ax.scatter(lon, lat, label=style.pop("label", "Stations"), **style)
    if labels is not None:
        for x, y, t in zip(lon, lat, labels, strict=False):
            ax.annotate(str(t), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
    return sc


def plot_events(ax, x, y, depth, cmap: str = DEPTH_CMAP, **kw):
    """Events as circles coloured by depth (km, increasing downwards); returns the scatter."""
    style = {**EVENT_STYLE, **kw}
    return ax.scatter(x, y, c=depth, cmap=cmap, **style)
